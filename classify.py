'''
Author: Lucas Hu (lucashu@usc.edu)
Timestamp: Fall 2019
Filename: classify.py
Goal: Classify land cover of various SEN12MS scenes
Model used: ResNet v1
'''

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# General imports
import glob
import argparse
import json
import pickle
from collections import defaultdict
import numpy as np
from scipy import stats
import keras
from keras.models import load_model
from keras.optimizers import Adam, Nadam
from keras.callbacks import ModelCheckpoint, \
    LearningRateScheduler, ReduceLROnPlateau, \
    EarlyStopping
from sklearn.feature_extraction.image import extract_patches_2d
from skimage.util.shape import view_as_blocks
from sklearn.preprocessing import LabelEncoder

# ResNet imports
import sys
import os
sys.path.append('./CBAM-keras')
from models import resnet_v1
from utils import lr_schedule

# SEN12MS imports
from sen12ms_dataLoader import SEN12MSDataset, \
    Seasons, Sensor, S1Bands, S2Bands, LCBands

ALL_SEASONS = [season for season in Seasons if season != Seasons.ALL]

def json_keys_to_int(x):
    '''
    Helper function to parse JSON with ints as keys
    '''
    try:
        return {int(k):v for k,v in x.items()}
    except:
        return x

def get_label_encoder(config):
    '''
    Uses config_dict's landuse_class info to get an sklearn label_encoder
    Output: sklearn label_encoder
    '''
    # get remaining classes after merging
    merged_classes = set(config['landuse_class_mappings'].keys())
    all_classes = set(config['landuse_class_descriptions'].keys())
    remaining_classes = all_classes - merged_classes
    # sort class_nums
    class_nums_sorted = sorted(list(remaining_classes))
    # get label_encoder
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(class_nums_sorted)
    return label_encoder

def patch_to_subpatches(patch, config):
    '''
    Input: single patch: B, W, H
    Output: N, B, subpatch_size, subpatch_size
    '''
    subpatch_size = config['training_params']['subpatch_size']
    subpatches = view_as_blocks(patch, \
        block_shape=(subpatch_size, subpatch_size, patch.shape[-1]))
    subpatches = np.squeeze(subpatches)
    subpatches = np.concatenate(subpatches, axis=0)
    return subpatches

def scene_to_subpatches(patches, config):
    '''
    Split square patches into smaller squre sub-patches
    Input: patches of shape D, B, W, H
    Output: patches of shape N, B, subpatch_size, subpatch_size
        N = D * (W / subpatch_size)
    '''
    subpatch_size = config['training_params']['subpatch_size']
    all_subpatches = [] # list of each patch's subpatch array
    for i, patch in enumerate(patches):
        subpatches = patch_to_subpatches(patch, config)
        all_subpatches.append(subpatches)
    # concat all subpatches
    return np.concatenate(all_subpatches, axis=0)

def combine_landuse_classes(landuse, config):
    '''
    Input: land use patches (Shape: D, W, H), config
    Output: land use patches with combined classes (see Section 5 of SEN12MS paper)
    '''
    landuse_class_mappings = config['landuse_class_mappings']
    for from_class, to_class in landuse_class_mappings.items():
        landuse = np.where(landuse==from_class, to_class, landuse)
    return landuse

def get_landuse_labels(lc, config):
    '''
    Input: lc (land cover bands, Shape: D, W, H, B=4), config
    Output: majority LCCS land-use class, Shape: D
    '''
    land_use_patches = lc[:, :, :, LCBands.landuse.value-1]
    land_use_patches = combine_landuse_classes(land_use_patches, config)
    land_use_flattened = land_use_patches.reshape(land_use_patches.shape[0], -1)
    modes, counts = stats.mode(land_use_flattened, axis=1)
    return np.ravel(modes)

def get_represented_landuse_classes_from_onehot_labels(y, label_encoder):
    '''
    Input: y (one-hot labels, 2D array), label_encoder
    Output: list of landuse class numbers that do appear in y
    '''
    labels = np.argmax(y, axis=1)
    landuse_classes = label_encoder.inverse_transform(labels)
    represented_classes = set(np.unique(landuse_classes).tolist())
    all_classes = set(label_encoder.classes_.tolist())
    missing_classes = all_classes - represented_classes
    return list(represented_classes)

def get_missing_landuse_classes_from_onehot_labels(y, label_encoder):
    '''
    Input: y (one-hot labels, 2D array), label_encoder
    Output: list of landuse class numbers that do not appear in y
    '''
    all_classes = set(label_encoder.classes_.tolist())
    represented_classes = get_represented_landuse_classes_from_onehot_labels(y)
    represented_classes = set(represented_classes)
    missing_classes = all_classes - represented_classes
    return list(missing_classes)

def preprocess_s2_lc(s2, lc, config, label_encoder):
    '''
    Input: s2, lc, config, label_encoder
    Output: X (s2 subpatches), y (one-hot labels)
    '''
    # move bands to last axis
    s2, lc = np.moveaxis(s2, 1, -1), np.moveaxis(lc, 1, -1)
    # get subpatches
    s2 = scene_to_subpatches(s2, config)
    lc = scene_to_subpatches(lc, config)
    # get majority classes
    labels = get_landuse_labels(lc, config)
    # remove instances with '0' mode label
    zero_label_inds = np.where(labels == 0)[0]
    if config['verbose'] >= 1 and len(zero_label_inds) > 0:
        print('Removing {} instances with "0" landuse label'.format(len(zero_label_inds)))
    labels = np.delete(labels, zero_label_inds, axis=0)
    s2 = np.delete(s2, zero_label_inds, axis=0)
    assert not (0 in labels)
    assert s2.shape[0] == labels.shape[0]
    # one-hot encode labels
    labels = label_encoder.transform(labels)
    y = keras.utils.to_categorical(labels, num_classes=len(label_encoder.classes_))
    X = s2
    return X, y

def get_compiled_resnet(config, label_encoder):
    '''
    Input: config dict, label_encoder
    Output: compiled ResNet model
    '''
    num_classes = len(label_encoder.classes_)
    # init resnet
    input_shape=(config['training_params']['subpatch_size'], \
        config['training_params']['subpatch_size'], \
        len(config['s2_input_bands']))
    model = resnet_v1.resnet_v1(input_shape=input_shape, \
        num_classes=len(label_encoder.classes_), \
        depth=config['resnet_params']['depth'], \
        attention_module=None)
    # compile
    model.compile(loss='categorical_crossentropy',
        optimizer=Nadam(lr=config['training_params']['learning_rate']),
        metrics=['accuracy'])
    return model

def get_callbacks(filepath, config):
    '''
    Input: model save filepath, config
    Output: model training callbacks
    '''
    checkpoint = ModelCheckpoint(filepath=filepath,
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=config['training_params']['early_stopping_patience'],
                                   restore_best_weights=True)
    lr_reducer = ReduceLROnPlateau(factor=config['lr_reducer_params']['factor'],
                                   cooldown=config['lr_reducer_params']['cooldown'],
                                   patience=config['lr_reducer_params']['patience'],
                                   min_lr=config['lr_reducer_params']['min_lr'])
    return [checkpoint, early_stopping, lr_reducer]

def get_train_val_patch_ids_from_scene(sen12ms, train_season, train_scene_id, config):
    '''
    Input: sen12ms, train_season, train_scene_id, config
    Output: train_patch_ids, val_patch_ids
    '''
    # get all patch IDs from this scene
    patch_ids = sen12ms.get_patch_ids(train_season, train_scene_id)
    # sample val_patches
    num_val_patches = int(len(patch_ids) * config['training_params']['val_size'])
    np.random.seed(config['experiment_params']['val_split_seed'])
    val_patch_ids = np.random.choice(patch_ids, size=num_val_patches).tolist()
    # get remaining train_patches
    train_patch_ids = set(patch_ids) - set(val_patch_ids)
    return train_patch_ids, val_patch_ids

def get_train_val_s2_lc_from_scene_ids(sen12ms, train_season, train_scene_ids, config):
    '''
    Input: sen12ms, train_season, train_scene_ids, config
    Output: train_s2, val_s2, train_lc, val_lc
    '''
    all_train_s2 = []
    all_val_s2 = []
    all_train_lc = []
    all_val_lc = []
    # get patches from each scene
    for scene in train_scene_ids:
        # compute train-val split
        train_patch_ids, val_patch_ids = get_train_val_patch_ids_from_scene(sen12ms, \
            train_season, scene, config)
        # train data
        train_s1, train_s2, train_lc, train_bounds = sen12ms.get_triplets(train_season, \
            scene_ids=scene, patch_ids=train_patch_ids, s2_bands=config['s2_input_bands'])
        # val data
        val_s1, val_s2, val_lc, val_bounds = sen12ms.get_triplets(train_season, \
            scene_ids=scene, patch_ids=val_patch_ids, s2_bands=config['s2_input_bands'])
        # append data
        all_train_s2.append(train_s2)
        all_val_s2.append(val_s2)
        all_train_lc.append(train_lc)
        all_val_lc.append(val_lc)
    # get final numpy arrays
    all_train_s2 = np.concatenate(all_train_s2, axis=0)
    all_val_s2 = np.concatenate(all_val_s2, axis=0)
    all_train_lc = np.concatenate(all_train_lc, axis=0)
    all_val_lc = np.concatenate(all_val_lc, axis=0)
    return all_train_s2, all_val_s2, all_train_lc, all_val_lc

def get_model_history_filepath(train_season, train_scene_ids, config):
    '''
    Inputs: train_season, train_scene_ids, config dict
    Output: model filepath, history filepath
    '''
    # if train_scene_ids is not a list, cast it to a list
    if not isinstance(train_scene_ids, (list, np.ndarray)):
        train_scene_ids = [train_scene_ids]
    model_type = 'resnet' + str(config['resnet_params']['depth'])
    model_name = 'sen12ms_%s_scene_%s_model_%s.h5' % \
        (str(train_season.value), \
        '+'.join([str(scene) for scene in train_scene_ids]), \
        model_type)
    model_filepath = os.path.join(config['model_save_dir'], model_name)
    history_filepath = model_filepath.split('.h5')[0] + '_history.json'
    return model_filepath, history_filepath

def train_and_save_resnet_model(sen12ms, train_season, train_scene_ids, config):
    '''
    Inputs: sen12ms, train_season (enum or list of enums),
        train_scene_ids (int or list of ints), config
    Output: trained keras model (and a model dump, saved to disk)
    '''
    print("--- Training ResNet model ---")
    print("Season: {}, scene(s): {}".format(train_season, train_scene_ids))
    # convert scene_ids to lists
    if not isinstance(train_scene_ids, list):
        train_scene_ids = [train_scene_ids]
    # get sen12ms data. shape: D, B, W, H (D = # patches, B = # bands)
    print("Loading sen12ms data...")
    train_s2, val_s2, train_lc, val_lc = \
        get_train_val_s2_lc_from_scene_ids(sen12ms, train_season, train_scene_ids, config)
    # print array sizes
    print("train_s2 size (bytes): {}".format(sys.getsizeof(train_s2)))
    print("val_s2 size (bytes): {}".format(sys.getsizeof(val_s2)))
    print("train_lc size (bytes): {}".format(sys.getsizeof(train_lc)))
    print("val_lc size (bytes): {}".format(sys.getsizeof(val_lc)))
    # preprocessing
    print("Preprocessing s2, lc patches...")
    label_encoder = get_label_encoder(config)
    X_train, y_train = preprocess_s2_lc(train_s2, train_lc, config, label_encoder)
    X_val, y_val = preprocess_s2_lc(val_s2, val_lc, config, label_encoder)
    print("X_train shape: ", X_train.shape)
    print("X_val shape: ", X_val.shape)
    print("y_train shape: ", y_train.shape)
    print("y_val shape: ", y_val.shape)
    # print classes represented in train, val sets
    train_represented_classes = get_represented_landuse_classes_from_onehot_labels(y_train, \
        label_encoder)
    val_represented_classes = get_represented_landuse_classes_from_onehot_labels(y_train, \
        label_encoder)
    print("y_train represented classes: ", train_represented_classes)
    print("y_val represented classes: ", val_represented_classes)
    # get compiled model
    model = get_compiled_resnet(config, label_encoder)
    model_filepath, history_filepath = get_model_history_filepath(train_season, train_scene_ids, config)
    # train keras model
    callbacks = get_callbacks(model_filepath, config)
    print("Training keras model...")
    history = model.fit(X_train, y_train,
              batch_size=config['training_params']['batch_size'],
              epochs=config['training_params']['max_epochs'],
              validation_data=(X_val, y_val),
              shuffle=True,
              callbacks=callbacks)
    print("Done training!")
    # save model history
    with open(history_filepath, 'wb') as f:
        pickle.dump(history, f)
    print("Model history saved to: ", history_filepath)
    return model, history

def evaluate_on_single_scene(sen12ms, config, label_encoder, \
    model=None, model_path=None, \
    test_season=None, test_scene_id=None):
    '''
    Inputs: sen12ms, config, model or model_path, test_season, test_scene_id
    Output: classification report for this scene
    '''
    assert model is not None or model_path is not None
    if model is None:
        model = load_model(model_path)
    # load data
    s1, s2, lc, bounds = sen12ms.get_triplets(test_season, test_scene_id, \
        s2_bands=config['s2_input_bands'])
    # preprocessing: get subpatches, majority landuse class, etc.
    X_test, y_test = preprocess_s2_lc(s2, lc, config, label_encoder)
    # run evaluation
    results = model.evaluate(X_test, y_test, verbose=1)
    return results

def evaluate_on_multiple_scenes(sen12ms, config, label_encoder, \
    model=None, model_path=None, \
    test_seasons=None, test_scene_ids=None, \
    results_path=None):
    '''
    Inputs: sen12ms, config, model or model_path,
        test_seasons (list of length S), test_scene_ids (list of length S),
        results_path (optional)
    Output: classification report for each scene (dict),
        and save results to results_path (on disk)
    '''
    # input checks
    assert model is not None or model_path is not None
    assert len(test_seasons) == len(test_scene_ids)
    # load model if necessary
    label_encoder = get_label_encoder(config)
    if model is None: 
        model = load_model(model_path)
    # evaluate on each test scene
    all_results = defaultdict(dict)
    for i in range(len(test_scene_ids)):
        season = test_seasons[i]
        scene_id = test_scene_ids[i]
        print('Evaluating on season: {}, scene: {}'.format(season, scene_id))
        scene_results = evaluate_on_single_scene(sen12ms, config, label_encoder, \
            model=model,  test_season=season, test_scene_id=scene_id)
        all_results[season.value][scene_id] = scene_results
        # save results after each scene
        if results_path is not None:
            with open(results_path, 'wb') as f:
                pickle.dump(all_results, f)
    return all_results

def train_models_for_each_season(config):
    '''
    Inputs: config (dict)
    Output: saved Keras models (on disk)
    '''
    sen12ms = SEN12MSDataset(config['dataset_dir'])
    for season in ALL_SEASONS:
        # sample N scenes from this season
        scene_ids = list(sen12ms.get_scene_ids(season))
        np.random.seed(config['experiment_params']['train_scene_sampling_seed'])
        sampled_scenes = np.random.choice(scene_ids, \
            size=int(config['experiment_params']['num_models_per_season']))
        # train and save keras models for each sampled scene
        for scene in sampled_scenes:
            # check if we have already trained a model for this scene
            model_filepath, history_filepath = get_model_history_filepath(season, scene, config)
            if os.path.exists(model_filepath) and os.path.exists(history_filepath):
                print("{} exists! Skipping model training".format(history_filepath))
                continue
            train_and_save_resnet_model(sen12ms, season, scene, config)

def evaluate_saved_models_on_each_season(config):
    '''
    Inputs: config (dict)
    Output: saved results dict for each trained model (on disk)
    '''
    sen12ms = SEN12MSDataset(config['dataset_dir'])
    label_encoder = get_label_encoder(config)
    # get all seasons/scenes
    seasons = []
    scenes = []
    for season in ALL_SEASONS:
        for scene_id in sen12ms.get_scene_ids(season):
            seasons.append(season)
            scenes.append(scene_id)
    # get all saved models
    model_filepaths = glob.glob(os.path.join(config['model_save_dir'], '*.h5'))
    # evaluate each saved model on each seasons/scene
    for model_path in model_filepaths:
        print('Evaluating model path: ', model_path)
        # get results pkl dump filepath
        model_name = os.path.basename(model_path)
        results_name = model_name.split('.h5')[0] + '_results.pkl'
        results_path = os.path.join(config['results_dir'], results_name)
        # check if results are already complete for this model
        if os.path.exists(results_path):
            with open(results_path, 'rb') as f:
                existing_results = pickle.load(f)
            print("Note: {} already exists!".format(results_path))
            season_results_complete = [len(season_results) == len(sen12ms.get_scene_ids(season)) \
                for season, season_results in existing_results.items()]
            print("Seasons complete? {}".format(season_results_complete))
            if all(season_results_complete):
                print("{} already complete! Skipping model eval".format(results_path))
                continue
        # evaluate model
        evaluate_on_multiple_scenes(sen12ms, config, label_encoder, \
            model_path=model_path, \
            test_seasons=seasons, test_scene_ids=scenes, \
            results_path=results_path)

def main(args):
    '''
    Main function: train new models, or test existing models on SEN12MS seasons/scenes
    '''
    # get config
    config_json_path = args.config_path
    with open(config_json_path, 'r') as f:
        config = json.load(f, object_hook=json_keys_to_int)
    # train new models on sampled seasons/scenes
    if args.train:
        train_models_for_each_season(config)
    # evaluate saved models on each season/scene
    if args.test:
        evaluate_saved_models_on_each_season(config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train or test land-cover ResNet model(s)')
    parser.add_argument('-c', '--config', dest='config_path', help='config JSON path')
    parser.add_argument('--train', dest='train', action='store_true', help='train new models')
    parser.add_argument('--test', dest='test', action='store_true', help='test saved models')
    args = parser.parse_args()
    main(args)

