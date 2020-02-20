'''
save_classification_subpatches_to_npy.py
Author: Lucas Hu

Preprocess SEN12MS dataset and saves 
individual subpatches as .npy files

Resulting directory structure:
continent-season -> scene_id -> patch_id -> (subpatch_s2.npy, subpatch_label.npy)
'''

import os
import json
import argparse
import numpy as np
from classify import get_landuse_labels, \
    json_keys_to_int, patch_to_subpatches
    
from sen12ms_dataLoader import SEN12MSDataset, \
    Seasons, Sensor, S1Bands, S2Bands, LCBands

ALL_SEASONS = [season.value for season in Seasons if season != Seasons.ALL]

def get_subpatch_save_path(config, continent, season, scene, patch, subpatch, label):
    '''
    Get subpatch save path for subpatch instance
    '''
    season = season.split("_")[-1] # remove ROI
    continent = continent.replace(" ", "_")
    continent_season = "{}-{}".format(continent, season)
    save_path = os.path.join(
        config['subpatches_dataset_dir'],
        continent_season,
        "scene_{}".format(scene),
        "patch_{}".format(patch),
        "subpatch_{}_label_{}.npy".format(subpatch, label)
        )
    return save_path

def main(args):
    '''
    Main function: Preprocess SEN12MS scenes, and save subpatches to .npy files
    '''
    # load config, sen12ms
    config_json_path = args.config_path
    with open(config_json_path, 'r') as f:
        config = json.load(f, object_hook=json_keys_to_int)
    sen12ms = SEN12MSDataset(config['dataset_dir'])
    with open(config['scene_locations_json_path'], 'r') as f:
        scene_locations = json.load(f)
    # read dataset
    for season in ALL_SEASONS:
        for scene in sen12ms.get_scene_ids(season):
            season_scene = "{}_{}".format(season, scene)
            _, _, continent = scene_locations[season_scene]
            print("Processing {}...".format(season_scene))
            for patch in sen12ms.get_patch_ids(season, scene):
                # get S2 data, LC labels
                s1, s2, lc, bounds = sen12ms.get_s1s2lc_triplet(season, scene, \
                    patch, s2_bands=config['s2_input_bands'])
                # move bands to last axis
                s2, lc = np.moveaxis(s2, 0, -1), np.moveaxis(lc, 0, -1)
                s2_subpatches = patch_to_subpatches(s2, config)
                lc_subpatches = patch_to_subpatches(lc, config)
                labels = get_landuse_labels(lc_subpatches, config)
                # dump each subpatch
                for i, subpatch in enumerate(s2_subpatches):
                    # get save_path (encode label in save_path)
                    label = labels[i]
                    save_path = get_subpatch_save_path(config, \
                        continent, season, scene, \
                        patch, i, label)
                    # make directories, if necessary
                    if not os.path.exists(os.path.dirname(save_path)):
                        os.makedirs(os.path.dirname(save_path))
                    # numpy dump
                    np.save(save_path, subpatch)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Save SEN12MS subpatches as .npy files')
    parser.add_argument('-c' '--config', dest='config_path', help='config JSON path')
    args = parser.parse_args()
    main(args)
    
