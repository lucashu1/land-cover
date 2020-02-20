'''
save_segmentation_patches_to_npy.py
Author: Lucas Hu

Preprocess SEN12MS dataset and saves 
individual patches as .npy files

Resulting directory structure:
continent-season -> scene_id -> patch_id -> (s2.npy, landuse.npy)
'''

import sys
sys.path.append('../')
import os
import json
import argparse
import numpy as np
from land_cover_utils import combine_landuse_classes, json_keys_to_int
    
from sen12ms_dataLoader import SEN12MSDataset, \
    Seasons, Sensor, S1Bands, S2Bands, LCBands

ALL_SEASONS = [season.value for season in Seasons if season != Seasons.ALL]

# Remapping IGBP classes to simplified DFC classes
IGBP2DFC = np.array([0, 1, 1, 1, 1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 6, 8, 9, 10])

def get_s1_s2_landuse_dfc_save_path(config, continent, season, scene, patch):
    '''
    Input: config, continent, season, scene, patch
    Output: s2_save_path, landuse_save_path
    '''
    season = season.split("_")[-1] # remove ROI
    continent = continent.replace(" ", "_")
    continent_season = "{}-{}".format(continent, season)
    save_dir = os.path.join(
        config['segmentation_dataset_dir'],
        continent_season,
        "scene_{}".format(scene),
        "patch_{}".format(patch)
        )
    s1_save_path = os.path.join(save_dir, "s1.npy")
    s2_save_path = os.path.join(save_dir, "s2.npy")
    landuse_save_path = os.path.join(save_dir, "landuse.npy")
    dfc_save_path = os.path.join(save_dir, "dfc.npy")
    return s1_save_path, s2_save_path, landuse_save_path, dfc_save_path

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
    # store max val of each S2 band
    s1_band_max = np.zeros(len(config['s1_input_bands']))
    s1_band_min = np.zeros(len(config['s1_input_bands']))
    s2_band_max = np.zeros(len(config['s2_input_bands']))
    # read dataset
    for season in ALL_SEASONS:
        continue
        if season == 'ROIs0000_validation':
            continue
        for scene in sen12ms.get_scene_ids(season):
            season_scene = "{}_{}".format(season, scene)
            _, _, continent = scene_locations[season_scene]
            print("Processing {}...".format(season_scene))
            for patch in sen12ms.get_patch_ids(season, scene):
                # get S2 data, LC labels
                s1, s2, lc, bounds = sen12ms.get_s1s2lc_triplet(season, scene, \
                    patch, s2_bands=config['s2_input_bands'])
                # move bands to last axis
                s1, s2, lc = np.moveaxis(s1, 0, -1), np.moveaxis(s2, 0, -1), np.moveaxis(lc, 0, -1)
                landuse_labels = lc[:, :, LCBands.landuse.value-1].astype('uint8')
                landuse_labels = combine_landuse_classes(landuse_labels, config)
                igbp = lc[:, :, 0].astype('uint8') # IGBP
                dfc = IGBP2DFC[igbp].astype('uint8') # simplified IGBP
                # update S1, S2 band max vals
                s1_band_min_in_patch = np.nanmin(s1, axis=(0,1))
                s1_band_min = np.nanmin(np.vstack([s1_band_min, s1_band_min_in_patch]), axis=0)
                s1_band_max_in_patch = np.nanmax(s1, axis=(0,1))
                s1_band_max = np.amax(np.vstack([s1_band_max, s1_band_max_in_patch]), axis=0)
                s2_band_max_in_patch = np.nanmax(s2, axis=(0,1))
                s2_band_max = np.amax(np.vstack([s2_band_max, s2_band_max_in_patch]), axis=0)
                # get save paths
                s1_path, s2_path, landuse_path, dfc_path = get_s1_s2_landuse_dfc_save_path(config, \
                    continent, season, scene, patch)
                # make directories, if necessary
                if not os.path.exists(os.path.dirname(s2_path)):
                    os.makedirs(os.path.dirname(s2_path))
                # numpy dump
                # np.save(s1_path, s1)
                # np.save(s2_path, s2)
                # np.save(landuse_path, landuse_labels)
                # np.save(dfc_path, dfc)
                # garbage collection
                del s1, s2, lc, dfc, bounds
            print("S1 input bands: ", config['s1_input_bands'])
            print("S1 band max vals (so far): ", s1_band_max)
            print("S1 band min vals (so far): ", s1_band_min)
            print("S2 input bands: ", config['s2_input_bands'])
            print("S2 band max vals (so far): ", s2_band_max)

    # save validation data (s1,s2 only)
    val_season = 'ROIs0000_validation'
    for scene in sen12ms.get_scene_ids(val_season):
        print(f'Processing {val_season}, scene {scene}...')
        for patch in sen12ms.get_patch_ids(val_season, scene):
            s1, s2, lc, bounds = sen12ms.get_s1s2lc_triplet(season, scene, \
                patch, s2_bands=config['s2_input_bands'])
            # move bands to last axis
            s1, s2, lc = np.moveaxis(s1, 0, -1), np.moveaxis(s2, 0, -1), np.moveaxis(lc, 0, -1)
            igbp = lc[:,:,0].astype('uint8')
            dfc = IGBP2DFC[igbp].astype('uint8') # simplified IGBP
            val_save_dir = os.path.join(
                config['segmentation_dataset_dir'],
                val_season,
                "scene_{}".format(scene),
                "patch_{}".format(patch))
            if not os.path.exists(val_save_dir):
                os.makedirs(val_save_dir)
            # save (s1,s2) only (for prediction)
            # np.save(os.path.join(val_save_dir, "s1.npy"), s2)
            # np.save(os.path.join(val_save_dir, "s2.npy"), s2)
            np.save(os.path.join(val_save_dir, "dfc.npy"), dfc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Save SEN12MS patches as .npy files')
    parser.add_argument('-c' '--config', dest='config_path', help='config JSON path')
    args = parser.parse_args()
    main(args)
    
