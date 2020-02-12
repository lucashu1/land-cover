'''
Evaluate on predictions
'''

import os
import numpy as np
import json
import land_cover_utils
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix

# import config
config_dir = '/home/lucas/land-cover/config.json'
with open(config_dir, 'r') as f:
    config = json.load(f, object_hook=land_cover_utils.json_keys_to_int)
    
label_encoder = land_cover_utils.get_label_encoder(config)

# predictions + labels directories
segmentation_results_dir = '/data/lucas/sen12ms_segmentation_predictions'
segmentation_labels_dir = '/data/datasets/sen12ms_segmentation'

continent_seasons = os.listdir(segmentation_labels_dir)

def get_segmentation_model_prediction_dirs(mode):
    '''
    Get segmentation model directories
    mode = "by_season" or "by_continent"
    '''
    model_names = os.listdir(os.path.join(segmentation_results_dir, mode))
    return [os.path.join(segmentation_results_dir, mode, model_name) for model_name in model_names]

def scene_dir_to_season_scene(scene_dir):
    '''
    E.g. '/data/datasets/sen12ms_subpatches/Africa-fall/scene_79' --> "fall_79"
    '''
    continent_season = scene_dir.split('/')[-2]
    season = continent_season.split('-')[-1]
    scene = int(scene_dir.split('_')[-1])
    return "{}_{}".format(season, scene)

def get_train_val_season_scenes(predictions_dir):
    '''
    Given a predictions filepath, get the corresponding train-val season_scenes
    '''
    if 'by_season' in predictions_dir:
        train_val_splits_dir = '/home/lucas/land-cover/saved_models/by_season'
    elif 'by_continent' in predictions_dir:
        train_val_splits_dir = '/home/lucas/land-cover/saved_models/by_continent'
    else:
        return None
    model_name = os.path.basename(predictions_dir)
    train_val_split_file = os.path.join(train_val_splits_dir, \
                                        model_name + '_train-val-split.json')
    train_val_split = json.load(open(train_val_split_file))
    train_scene_dirs = train_val_split['train_scene_dirs']
    val_scene_dirs = train_val_split['val_scene_dirs']
    train_season_scenes = [scene_dir_to_season_scene(scene_dir) for scene_dir in train_scene_dirs]
    val_season_scenes = [scene_dir_to_season_scene(scene_dir) for scene_dir in val_scene_dirs]
    return train_season_scenes, val_season_scenes

def get_scene_prediction_dirs_for_season(model_predictions_dir, season):
    '''
    Get full prediction scene_dirs for given model and season
    '''
    scene_dirs = []
    continent_season_dirs = [continent_season for continent_season in os.listdir(model_predictions_dir) if season in continent_season]
    continent_season_dirs = [os.path.join(model_predictions_dir, c_s) for c_s in continent_season_dirs]
    for c_s_dir in continent_season_dirs:
        scenes = os.listdir(c_s_dir)
        scene_dirs.extend([os.path.join(c_s_dir, scene) for scene in scenes])
    return scene_dirs

def get_scene_prediction_dirs_for_continent(model_predictions_dir, continent):
    '''
    Get full prediction scene_dirs for given model and continent
    '''
    scene_dirs = []
    continent_season_dirs = [continent_season for continent_season in os.listdir(model_predictions_dir) if continent in continent_season]
    continent_season_dirs = [os.path.join(model_predictions_dir, c_s) for c_s in continent_season_dirs]
    for c_s_dir in continent_season_dirs:
        scenes = os.listdir(c_s_dir)
        scene_dirs.extend([os.path.join(c_s_dir, scene) for scene in scenes])
    return scene_dirs

def get_sorted_patch_ids_from_scene_dir(scene_dir):
    ''' get all patches from a given scene_dir '''
    patches = os.listdir(scene_dir)
    patches = [int(patch.split('_')[1].split('.')[0]) for patch in patches]
    patches = sorted(patches)
    return patches

def get_segmentation_label_prediction_from_scene_patch(scene_dir, patch_id):
    ''' get label array and prediction array from patch_id '''
    # get label
    continent_season_scene = '/'.join(scene_dir.split('/')[-2:])
    landuse_path = os.path.join(segmentation_labels_dir, continent_season_scene, 'patch_{}'.format(patch_id), 'landuse.npy')
    label = np.load(landuse_path)
    # get prediction
    patch_path = os.path.join(scene_dir, 'patch_{}.npz'.format(patch_id))
    prediction = np.load(patch_path)['arr_0']
    return label, prediction
            
### By-Season Segmentation Results ###

segmentation_season_model_results_dir = './results/segmentation_season_model_results.pkl'
segmentation_season_model_results = {} # train_season -> test_season -> season_scene -> 'acc' or 'confusion'

for model_predictions_dir in get_segmentation_model_prediction_dirs('by_season'):
    train_season = model_predictions_dir.split('_season_')[-1].split('_')[0] # e.g. 'winter'
    print('---------- Evaluating {} season segmentation model ----------\n'.format(train_season))
    segmentation_season_model_results[train_season] = {}
    train_season_scenes, val_season_scenes = get_train_val_season_scenes(model_predictions_dir)
    # iterate over all seasons
    for season in config['all_seasons']:
        print('--- Evaluating on season: {} ---\n'.format(season))
        segmentation_season_model_results[train_season][season] = {}
        scene_dirs = get_scene_prediction_dirs_for_season(model_predictions_dir, season)
        # iterate over all scenes
        for scene_dir in scene_dirs:
            # if model was trained on this scene, then skip
            season_scene = scene_dir_to_season_scene(scene_dir)
            print('evaluation season_scene: ', season_scene)
            if season_scene in train_season_scenes:
                print('scene was used for model training. skipping evaluation on this scene')
                continue
            labels, preds = [], []
            # get labels, predictions from each patch in this scene
            patch_ids = get_sorted_patch_ids_from_scene_dir(scene_dir)
            # get label, predictions
            for patch_id in patch_ids:
                try:
                    label, pred = get_segmentation_label_prediction_from_scene_patch(scene_dir, patch_id)
                except:
                    continue
                assert pred.shape == (256,256) and label.shape == (256,256)
                labels.append(label)
                preds.append(pred)
            # calculate results on this scene
            labels, preds = np.array(labels).flatten(), np.array(preds).flatten()
            confusion = np.array(confusion_matrix(labels, preds, labels=label_encoder.classes_)) # exclude unknown classes
            acc = np.trace(confusion) / np.sum(confusion)
            # store results in results dict
            segmentation_season_model_results[train_season][season][season_scene] = {}
            segmentation_season_model_results[train_season][season][season_scene]['acc'] = acc
            segmentation_season_model_results[train_season][season][season_scene]['confusion'] = confusion
            # pickle dump results dict
            with open(segmentation_season_model_results_dir, 'wb') as f:
                pickle.dump(segmentation_season_model_results, f)
            # print status update
            print('acc: ', acc)
            print('confusion:\n', confusion)
            print()

### By-Continent Segmentation Results ###

import pickle
from sklearn.metrics import accuracy_score, confusion_matrix

segmentation_continent_model_results_dir = './results/segmentation_continent_model_results.pkl'
segmentation_continent_model_results = {} # train_season -> test_season -> season_scene -> 'acc' or 'confusion'

for model_predictions_dir in get_segmentation_model_prediction_dirs('by_continent'):
    train_continent = model_predictions_dir.split('_continent_')[-1].split('_FC-DenseNet')[0] # e.g. 'Africa'
    print('---------- Evaluating {} continent segmentation model ----------\n'.format(train_continent))
    segmentation_continent_model_results[train_continent] = {}
    train_season_scenes, val_season_scenes = get_train_val_season_scenes(model_predictions_dir)
    # iterate over all seasons
    for continent in config['all_continents']:
        print('--- Evaluating on continent: {} ---\n'.format(continent))
        segmentation_continent_model_results[train_continent][continent] = {}
        scene_dirs = get_scene_prediction_dirs_for_continent(model_predictions_dir, continent)
        # iterate over all scenes
        for scene_dir in scene_dirs:
            # if model was trained on this scene, then skip
            season_scene = scene_dir_to_season_scene(scene_dir)
            print('evaluation season_scene: ', season_scene)
            if season_scene in train_season_scenes:
                print('scene was used for model training. skipping evaluation on this scene')
                continue
            labels, preds = [], []
            # get labels, predictions from each patch in this scene
            patch_ids = get_sorted_patch_ids_from_scene_dir(scene_dir)
            # get label, predictions
            for patch_id in patch_ids:
                try:
                    label, pred = get_segmentation_label_prediction_from_scene_patch(scene_dir, patch_id)
                except:
                    continue
                assert pred.shape == (256,256) and label.shape == (256,256)
                labels.append(label)
                preds.append(pred)
            # calculate results on this scene
            labels, preds = np.array(labels).flatten(), np.array(preds).flatten()
            confusion = np.array(confusion_matrix(labels, preds, labels=label_encoder.classes_)) # exclude unknown classes
            acc = np.trace(confusion) / np.sum(confusion)
            # store results in results dict
            segmentation_continent_model_results[train_continent][continent][season_scene] = {}
            segmentation_continent_model_results[train_continent][continent][season_scene]['acc'] = acc
            segmentation_continent_model_results[train_continent][continent][season_scene]['confusion'] = confusion
            # pickle dump results dict
            with open(segmentation_continent_model_results_dir, 'wb') as f:
                pickle.dump(segmentation_continent_model_results, f)
            # print status update
            print('acc: ', acc)
            print('confusion:\n', confusion)
            print()

