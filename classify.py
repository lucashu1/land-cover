'''
Author: Lucas Hu (lucashu@usc.edu)
Timestamp: Fall 2019
Filename: classify.py
Goal: Classify land cover of various SEN12MS scenes
Models used: ResNet v1, FC-DenseNet
'''

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# General imports
import glob
import argparse
import json
from collections import defaultdict
import numpy as np
from scipy import stats
import imageio
import tensorflow as tf
import keras
from keras.layers import Dense, Flatten
from keras.models import Model, load_model
from keras.optimizers import Adam, Nadam
from keras.losses import categorical_crossentropy
from keras.callbacks import ModelCheckpoint, \
    LearningRateScheduler, ReduceLROnPlateau, \
    EarlyStopping

# ResNet imports
import sys
import os
sys.path.append('./CBAM-keras')
from models import resnet_v1
from utils import lr_schedule

# DenseNet imports
sys.path.append('./DenseNet')
from densenet import DenseNetFCN

# SEN12MS imports
from sen12ms_dataLoader import SEN12MSDataset, \
    Seasons, Sensor, S1Bands, S2Bands, LCBands

# util imports
import datagen
import land_cover_utils

ALL_SEASONS = [season for season in Seasons if season != Seasons.ALL]

def preprocess_s2_lc_for_classification(s2, lc, config, label_encoder, patch_ids=None):
    '''
    Input: s2, lc, config, label_encoder
    Output: X (s2 subpatches), y (one-hot labels)
    '''
    # move bands to last axis
    s2, lc = np.moveaxis(s2, 1, -1), np.moveaxis(lc, 1, -1)
    s2 = s2.astype(np.float32) / config['s2_max_val'] # normalize S2
    # get subpatches
    if patch_ids is not None:
        s2, patch_ids = land_cover_utils.scene_to_subpatches(s2, config, patch_ids)
        lc, patch_ids = land_cover_utils.scene_to_subpatches(lc, config, patch_ids)
    else:
        s2 = land_cover_utils.scene_to_subpatches(s2, config)
        lc = land_cover_utils.scene_to_subpatches(lc, config)
    # get majority classes
    labels = land_cover_utils.get_landuse_labels(lc, config)
    # remove instances with '0' mode label
    zero_label_inds = np.where(labels == 0)[0]
    if config['verbose'] >= 1 and len(zero_label_inds) > 0:
        print('Removing {} instances with "0" landuse label'.format(len(zero_label_inds)))
    labels = np.delete(labels, zero_label_inds, axis=0)
    s2 = np.delete(s2, zero_label_inds, axis=0)
    if patch_ids is not None:
        patch_ids = np.delete(patch_ids, zero_label_inds, axis=0)
    assert not (0 in labels)
    assert s2.shape[0] == labels.shape[0]
    # one-hot encode labels
    labels = label_encoder.transform(labels)
    y = keras.utils.to_categorical(labels, num_classes=len(label_encoder.classes_))
    X = s2
    if patch_ids is not None:
        return X, y, patch_ids
    else:
        return X, y

def preprocess_s2_lc_for_segmentation(s2, lc, config, label_encoder, patch_ids=None):
    '''
    Input: s2, lc, config, label_encoder
    Output: X (s2 patches), y (one-hot land-use patches)
    '''
    img_size = config['training_params']['patch_size']
    num_classes = len(label_encoder.classes_)
    # move bands to last axis
    s2, lc = np.moveaxis(s2, 1, -1), np.moveaxis(lc, 1, -1)
    s2 = s2.astype(np.float32) / config['s2_max_val'] # normalize S2
    # get landuse labels
    landuse = lc[:, :, :, LCBands.landuse.value-1]
    landuse = land_cover_utils.combine_landuse_classes(landuse, config)
    # delete patches with unknown landuse values
    unknown_landuse_inds = np.where(np.isin(landuse, config['landuse_unknown_classes'])==True)[0]
    if config['verbose'] >= 1 and len(unknown_landuse_inds) > 0:
        print('Removing {} instances with unknown landuse label'.format(len(unknown_landuse_inds)))
    landuse = np.delete(landuse, unknown_landuse_inds, axis=0)
    s2 = np.delete(s2, unknown_landuse_inds, axis=0)
    if patch_ids is not None:
        patch_ids = np.delete(patch_ids, unknown_landuse_inds, axis=0)
    # encode labels
    landuse = label_encoder.transform(landuse.flatten()).reshape((landuse.shape[0],img_size,img_size))
    y = keras.utils.to_categorical(landuse, num_classes=num_classes)
    X = s2
    assert X.shape[0] == y.shape[0]
    if patch_ids is not None:
        return X, y, patch_ids
    else:
        return X, y

def get_compiled_resnet(config, label_encoder, \
    predict_continents=False, predict_seasons=False):
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
    if not predict_continents and not predict_seasons:
        model.compile(loss='categorical_crossentropy',
            optimizer=Nadam(lr=config['resnet_params']['learning_rate']),
            metrics=['accuracy'])
        return model
    if predict_continents and predict_seasons:
        print('WARNING: cannot have predict_continents and predict_seasons both set to True!')
    # return model with 'continent' output
    if predict_continents:
        last_flatten = list(filter(lambda layer: 'flatten' in layer.name, model.layers))[-1] # end of last residual block
        continent_output = Dense(len(config['all_continents']), activation='softmax')(last_flatten.output)
        full_model = Model(model.inputs[0], [model.outputs[0], continent_output])
        full_model.compile(loss='categorical_crossentropy',
            loss_weights=[1,-config['training_params']['geospatial_loss_weight']],
            optimizer=Nadam(lr=config['resnet_params']['learning_rate']),
            metrics=['accuracy'])
        return full_model
    # return model with 'season' output
    elif predict_seasons:
        last_flatten = list(filter(lambda layer: 'flatten' in layer.name, model.layers))[-1] # end of last residual block
        season_output = Dense(len(config['all_seasons']), activation='softmax')(last_flatten.output)
        full_model = Model(model.inputs[0], [model.outputs[0], season_output])
        full_model.compile(loss='categorical_crossentropy',
            loss_weights=[1,-config['training_params']['geospatial_loss_weight']],
            optimizer=Nadam(lr=config['resnet_params']['learning_rate']),
            metrics=['accuracy'])
        return full_model

def get_compiled_fc_densenet(config, label_encoder, \
    predict_continents=False, predict_seasons=False, \
    loss='categorical_crossentropy'):
    '''
    Input: config_dict, label_encoder
    Output: compiled FC-DenseNet model
    '''
    # init FC DenseNet
    num_classes = len(label_encoder.classes_)
    img_size = config['training_params']['patch_size']
    input_shape=(img_size, img_size, len(config['s1_input_bands'])+len(config['s2_input_bands']))
    model = DenseNetFCN(input_shape, include_top=True, weights=None, \
        classes=num_classes, \
        nb_dense_block=config['fc_densenet_params']['nb_dense_block'], \
        activation='softmax')
    if not predict_continents and not predict_seasons:
        model.compile(loss=loss,
            optimizer=Nadam(lr=config['fc_densenet_params']['learning_rate']),
            metrics=['accuracy'])
        return model
    if predict_continents and predict_seasons:
        print('WARNING: cannot have predict_continents and predict_seasons both set to True!')
    if predict_continents:
        last_concat = list(filter(lambda layer: 'concatenate' in layer.name, model.layers))[-1] # end of last DenseNet block
        continent_output = Flatten()(last_concat.output)
        continent_output = Dense(len(config['all_continents']), activation='softmax')(continent_output)
        full_model = Model(model.inputs[0], [model.outputs[0], continent_output])
        full_model.compile(loss='categorical_crossentropy',
            loss_weights=[1,-config['training_params']['geospatial_loss_weight']],
            optimizer=Nadam(lr=config['fc_densenet_params']['learning_rate']),
            metrics=['accuracy'])
        return full_model
    elif predict_seasons:
        last_concat = list(filter(lambda layer: 'concatenate' in layer.name, model.layers))[-1] # end of last DenseNet block
        season_output = Flatten()(last_concat.output)
        season_output = Dense(len(config['all_seasons']), activation='softmax')(season_output)
        full_model = Model(model.inputs[0], [model.outputs[0], season_output])
        full_model.compile(loss='categorical_crossentropy',
            loss_weights=[1,-config['training_params']['geospatial_loss_weight']],
            optimizer=Nadam(lr=config['fc_densenet_params']['learning_rate']),
            metrics=['accuracy'])
        return full_model

def get_callbacks(filepath, config):
    '''
    Input: model save filepath, config
    Output: model training callbacks
    '''
    checkpoint = ModelCheckpoint(filepath=filepath,
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=True)
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=config['training_params']['early_stopping_patience'],
                                   restore_best_weights=True)
    lr_reducer = ReduceLROnPlateau(factor=config['lr_reducer_params']['factor'],
                                   cooldown=config['lr_reducer_params']['cooldown'],
                                   patience=config['lr_reducer_params']['patience'],
                                   min_lr=config['lr_reducer_params']['min_lr'])
    return [checkpoint, early_stopping, lr_reducer]

def get_train_val_scene_dirs(scene_dirs, config):
    '''
    Input: scene_dirs (list), config
    Output: train_scene_dirs (list), val_scene_dirs (list)
    '''
    num_val_scenes = int(len(scene_dirs) * config['training_params']['val_size'])
    # set seed, and sort scene_dirs to get reproducible split
    np.random.seed(config['experiment_params']['val_split_seed'])
    val_scene_dirs = np.random.choice(sorted(scene_dirs), size=num_val_scenes).tolist()
    train_scene_dirs = list(set(scene_dirs) - set(val_scene_dirs))
    return train_scene_dirs, val_scene_dirs

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

def train_resnet_on_scene_dirs(scene_dirs, weights_path, config, \
    predict_continents=False, predict_seasons=False):
    '''
    Input: scene_dirs, weights_path, config
    Output: trained ResNet model (saved to disk), training history
    '''
    # get train, val subpatch .npy dirs
    print("Performing train/val split...")
    train_scene_dirs, val_scene_dirs = get_train_val_scene_dirs(scene_dirs, config)
    print("train_scene_dirs: ", train_scene_dirs)
    print("val_scene_dirs: ", val_scene_dirs)
    # save train-val split
    history_filepath = weights_path.split('_weights.h5')[0] + '_history.json'
    train_split_filepath = weights_path.split('_weights.h5')[0] + '_train-val-split.json'
    with open(train_split_filepath, 'w') as f:
        train_split = {
            'train_scene_dirs': train_scene_dirs,
            'val_scene_dirs': val_scene_dirs
        }
        json.dump(train_split, f, indent=4)
    # get subpatch filepaths
    train_subpatch_paths = land_cover_utils.get_subpatch_paths_for_scene_dirs(train_scene_dirs)
    val_subpatch_paths = land_cover_utils.get_subpatch_paths_for_scene_dirs(val_scene_dirs)
    # get compiled keras model
    label_encoder = land_cover_utils.get_label_encoder(config)
    model = get_compiled_resnet(config, label_encoder, predict_continents, predict_seasons)
    # set up callbacks, data generators
    callbacks = get_callbacks(weights_path, config)
    train_datagen = datagen.SubpatchDataGenerator(train_subpatch_paths, config, \
        return_labels=True)
    val_datagen = datagen.SubpatchDataGenerator(val_subpatch_paths, config, \
        return_labels=True)
    # fit keras model
    print("Training keras model...")
    history = None
    history = model.fit_generator(
        train_datagen,
        epochs=config['training_params']['max_epochs'],
        validation_data=val_datagen,
        callbacks=callbacks,
        max_queue_size=config['resnet_params']['batch_size'],
        use_multiprocessing=config['training_params']['use_multiprocessing'],
        workers=config['training_params']['workers']
    )
    history = land_cover_utils.make_history_json_serializable(history.history)
    # save model history
    with open(history_filepath, 'w') as f:
        json.dump(history, f, indent=4)
    print("Model history saved to: ", history_filepath)
    return model, history

def train_resnet_on_continent(continent, config):
    '''
    Input: continent, config
    Output: trained ResNet model (saved to disk), training history
    '''
    print("--- Training ResNet model on {} ---".format(continent))
    # get filepaths
    if not predict_continents and not predict_seasons:
        filename = 'sen12_continent_{}_resnet-{}_weights.h5'.format(continent, config['resnet_params']['depth'])
    elif predict_continents and not predict_seasons:
        filename = 'sen12_continent_{}_resnet-{}_predict-continents_weights.h5'.format(continent, config['resnet_params']['depth'])
    elif predict_seasons and not predict_continents:
        filename = 'sen12_continent_{}_resnet-{}_predict-seasons_weights.h5'.format(continent, config['resnet_params']['depth'])
    else:
        filename = 'sen12_continent_{}_resnet-{}_predict-continents-seasons_weights.h5'.format(continent, config['resnet_params']['depth'])
    weights_path = os.path.join(
        config['model_save_dir'],
        'by_continent',
        filename)
    history_path = weights_path.split('_weights.h5')[0] + '_history.json'
    train_split_path = weights_path.split('_weights.h5')[0] + '_train-val-split.json'
    # check if model exists
    if os.path.exists(weights_path) and os.path.exists(history_path) and os.path.exists(train_split_path):
        print('files for model {} already exist! skipping training'.format(weights_path))
        return
    # train model
    scene_dirs = land_cover_utils.get_scene_dirs_for_continent(continent, config)
    model, history = train_resnet_on_scene_dirs(scene_dirs, weights_path, config)
    return model, history

def train_resnet_on_season(season, config):
    '''
    Input: season, config
    Output: trained ResNet model (saved to disk), training history
    '''
    print("--- Training ResNet model on {} ---".format(season))
    # get filepaths
    if not predict_continents and not predict_seasons:
        filename = 'sen12_season_{}_resnet-{}_weights.h5'.format(season, config['resnet_params']['depth'])
    elif predict_continents and not predict_seasons:
        filename = 'sen12_season_{}_resnet-{}_predict-continents_weights.h5'.format(season, config['resnet_params']['depth'])
    elif predict_seasons and not predict_continents:
        filename = 'sen12_season_{}_resnet-{}_predict-seasons_weights.h5'.format(season, config['resnet_params']['depth'])
    else:
        filename = 'sen12_season_{}_resnet-{}_predict-continents-seasons_weights.h5'.format(season, config['resnet_params']['depth'])
    weights_path = os.path.join(
        config['model_save_dir'],
        'by_season',
        filename)
    history_path = weights_path.split('_weights.h5')[0] + '_history.json'
    train_split_path = weights_path.split('_weights.h5')[0] + '_train-val-split.json'
    # check if model exists
    if os.path.exists(weights_path) and os.path.exists(history_path) and os.path.exists(train_split_path):
        print('files for model {} already exist! skipping training'.format(weights_path))
        return
    # train model
    scene_dirs = land_cover_utils.get_scene_dirs_for_season(season, config)
    model, history = train_resnet_on_scene_dirs(scene_dirs, weights_path, config)
    return model, history

def save_fc_densenet_predictions_on_scene_dir(model, scene_dir, save_dir, label_encoder, config, competition_mode=False):
    '''
    Use FC-DenseNet model to predict on a single scene_dir
    Store predictions in .npz file (1 file per scene)
    '''
    if os.path.exists(save_dir):
        print('save_dir {} already exists! skipping prediction'.format(save_dir))
        return
    print('generating predictions to {}...'.format(save_dir))
    # prep datagen
    patch_paths = land_cover_utils.get_segmentation_patch_paths_for_scene_dir(scene_dir)
    patch_ids = [int(path.split('patch_')[-1]) for path in patch_paths]
    predict_datagen = datagen.SegmentationPatchDataGenerator(patch_paths, config, labels=None)
    # predict
    predictions = model.predict_generator(predict_datagen)
    # post-process predictions
    predictions = np.argmax(predictions, axis=-1) # output shape: (N, W, H)
    predictions = label_encoder.inverse_transform(predictions.flatten()).reshape(predictions.shape)
    predictions = predictions.astype('uint8')
    # save to .npz files, indexed by patch_id (each file = predictions on 1 patch)
    os.makedirs(save_dir)
    for patch_id, pred in zip(patch_ids, predictions):
        if competition_mode:
            path = os.path.join(save_dir, 'ROIs0000_validation_dfc_0_p{}.tif'.format(patch_id))
            imageio.imwrite(path, pred)
        else:
            path = os.path.join(save_dir, 'patch_{}.npz'.format(patch_id))
            np.savez_compressed(path, pred)
    print('saved fc-densenet predictions to {}'.format(save_dir))
    return predictions

def save_resnet_predictions_on_scene_dir(model, scene_dir, save_dir, label_encoder, config):
    '''
    Use ResNet model to predict on a single scene_dir
    Store predictions in .npz files (1 file per scene)
    '''
    # prep datagen
    print('generating predictions to {}...'.format(save_dir))
    subpatch_paths = land_cover_utils.get_subpatch_paths_for_scene_dir(scene_dir)
    patch_ids = [int(path.split('/patch_')[-1].split('/')[0]) for path in subpatch_paths]
    predict_datagen = datagen.SubpatchDataGenerator(subpatch_paths, config, return_labels=False)
    # predict
    predictions = model.predict_generator(predict_datagen)
    # post-process predictions
    predictions = np.argmax(predictions, axis=-1)
    predictions = predictions.astype('uint8')
    # save to .npz, indexed by patch_id (each file = predictions on subpatches from 1 patch)
    if os.path.exists(save_dir):
        print('save_dir {} already exists! skipping prediction'.format(save_dir))
        return
    else:
        os.makedirs(save_dir)
    subpatches_per_patch = config['training_params']['patch_size'] // config['training_params']['subpatch_size']
    for i in range(0, len(patch_ids), subpatches_per_patch):
        assert len(set(patch_ids[i:i+subpatches_per_patch])) == 1 # all subpatches from 1 patch
        path = os.path.join(save_dir, 'patch_{}.npz'.format(patch_ids[i]))
        np.savez_compressed(path, predictions[i:i+subpatches_per_patch])
    print('saved resnet predictions to {}'.format(save_dir))
    return predictions

def predict_model_path_on_each_scene(model_path, label_encoder, config):
    '''
    Given a weights_path,
    Save predictions on each scene
    '''
    # load model from model_path
    if 'weights' in model_path and 'resnet' in model_path:
        model = get_compiled_resnet(config, label_encoder)
        model.load_weights(model_path)
        mode = 'subpatches'
    elif 'weights' in model_path and 'DenseNet' in model_path:
        model = get_compiled_fc_densenet(config, label_encoder)
        model.load_weights(model_path)
        mode = 'segmentation'
    else:
        print('ERROR: unable to load weights file!')
        return
    model_name = os.path.basename(model_path).split('_weights.h5')[0]
    folder = 'by_continent' if 'continent' in model_name else 'by_season'
    # predict on each scene
    for continent in config['all_continents']:
        for season in config['all_seasons']:
            # get all scenes from this continent-season
            scene_dirs = land_cover_utils.get_scene_dirs_for_continent_season(continent, season, config, mode)
            if mode == 'subpatches':
                # predict in subpatch/classification mode
                for scene_dir in scene_dirs:
                    scene_name = scene_dir.split('/')[-1]
                    save_dir = os.path.join(config['subpatches_predictions_dir'],
                        folder, model_name, '{}-{}'.format(continent, season), scene_name)
                    save_resnet_predictions_on_scene_dir(model, scene_dir, save_dir, label_encoder, config)
            else:
                # predict in segmentation mode
                for scene_dir in scene_dirs:
                    scene_name = scene_dir.split('/')[-1]
                    save_dir = os.path.join(config['segmentation_predictions_dir'],
                        folder, model_name, '{}-{}'.format(continent, season), scene_name)
                    save_fc_densenet_predictions_on_scene_dir(model, scene_dir, save_dir, label_encoder, config)
    print('finished predictions using model_path: ', model_path)
    print()

def predict_saved_models_on_each_scene(config):
    '''
    Load all saved models
    Save predictions on each scene
    '''
    # get all saved models
    model_filepaths = glob.glob(os.path.join(config['model_save_dir'], '**/*.h5'))
    label_encoder = land_cover_utils.get_label_encoder(config)
    # evaluate each saved model on each seasons/scene
    for model_path in model_filepaths:
        print('Predicting using model path: ', model_path)
        predict_model_path_on_each_scene(model_path, label_encoder, config)
    return
    
def predict_model_path_on_validation_set(model_path, label_encoder, config):
    '''
    Given a weights_path for an FC-DenseNet model,
    Save predictions on each scene in the Validation set
    '''
    model = get_compiled_fc_densenet(config, label_encoder)
    model.load_weights(model_path)
    model_name = os.path.basename(model_path).split('_weights.h5')[0]
    val_season = 'ROIs0000_validation'
    # get all scenes from validation set
    scene_dirs = os.listdir(config['validation_dataset_dir'])
    scene_dirs = [os.path.join(config['validation_dataset_dir'], scene) for scene in scene_dirs]
    # predict in segmentation mode
    for scene_dir in scene_dirs:
        scene_name = scene_dir.split('/')[-1]
        save_dir = os.path.join(config['competition_predictions_dir'],
            model_name, val_season, scene_name)
        save_fc_densenet_predictions_on_scene_dir(model, scene_dir, save_dir, label_encoder, config, competition_mode=True)
    print('finished predictions using model_path: ', model_path)
    print()

def evaluate_on_single_scene(sen12ms, config, label_encoder, \
    model=None, model_path=None, \
    test_season=None, test_scene_id=None):
    '''
    Inputs: sen12ms, config, model or model_path, test_season, test_scene_id
    Output: classification report for this scene
    '''
    assert model is not None or model_path is not None
    # load model from model_path, if necessary
    if model is None:
        if 'weights' in model_path and 'resnet' in model_path:
            model = get_compiled_resnet(config, label_encoder)
            model.load_weights(model_path)
        elif 'weights' in model_path and 'DenseNet' in model_path:
            model = get_compiled_fc_densenet(config, label_encoder)
            model.load_weights(model_path)
        else:
            model = load_model(model_path)
    # load data
    s1, s2, lc, bounds = sen12ms.get_triplets(test_season, test_scene_id, \
        s2_bands=config['s2_input_bands'])
    patch_ids = sen12ms.get_patch_ids(test_season, test_scene_id)
    # preprocessing: get subpatches, majority landuse class, etc.
    if (model is not None and 'densenet' in model.name) or \
            (model_path is not None and 'DenseNet' in model_path):
        X_test, y_test, patch_ids = preprocess_s2_lc_for_segmentation(s2, lc, config, label_encoder, patch_ids)
    else:
        X_test, y_test, patch_ids = preprocess_s2_lc_for_classification(s2, lc, config, label_encoder, patch_ids)
    # run evaluation
    results = model.evaluate(X_test, y_test, verbose=1)
    predictions = model.predict(X_test)
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
    label_encoder = land_cover_utils.get_label_encoder(config)
    if model is None:
        if 'weights' in model_path and 'resnet' in model_path:
            model = get_compiled_resnet(config, label_encoder)
            model.load_weights(model_path)
        elif 'weights' in model_path and 'DenseNet' in model_path:
            model = get_compiled_fc_densenet(config, label_encoder)
            model.load_weights(model_path)
        else:
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
            with open(results_path, 'w') as f:
                json.dump(all_results, f, indent=4)
    return all_results

def evaluate_saved_models_on_each_season(config):
    '''
    Inputs: config (dict)
    Output: saved results dict for each trained model (on disk)
    '''
    sen12ms = SEN12MSDataset(config['dataset_dir'])
    label_encoder = land_cover_utils.get_label_encoder(config)
    # get all seasons/scenes
    seasons = []
    scenes = []
    for season in ALL_SEASONS:
        for scene_id in sen12ms.get_scene_ids(season):
            seasons.append(season)
            scenes.append(scene_id)
    # get all saved models
    model_filepaths = glob.glob(os.path.join(config['model_save_dir'], '**/*.h5'))
    # evaluate each saved model on each seasons/scene
    for model_path in model_filepaths:
        # for now, skip single-scene models
        if 'by_scene' in model_path:
            continue
        print('Evaluating model path: ', model_path)
        # get results json dump filepath
        model_name = os.path.basename(model_path)
        results_name = model_name.split('_weights.h5')[0] + '_results.json'
        results_path = os.path.join(config['results_dir'], results_name)
        # check if results are already complete for this model
        if os.path.exists(results_path):
            with open(results_path, 'rb') as f:
                existing_results = json.load(f, object_hook=land_cover_utils.json_keys_to_int)
            print("Note: {} already exists!".format(results_path))
            season_results_complete = [len(season_results) == len(sen12ms.get_scene_ids(season)) \
                for season, season_results in existing_results.items()]
            print("Seasons complete? {}".format(season_results_complete))
            if all(season_results_complete):
                print("{} already complete! Skipping model eval".format(results_path))
                continue
        # don't evaluate single-scene models
        if 'scene' in model_name:
            continue
        # evaluate model
        evaluate_on_multiple_scenes(sen12ms, config, label_encoder, \
            model_path=model_path, \
            test_seasons=seasons, test_scene_ids=scenes, \
            results_path=results_path)

def train_fc_densenet_on_scene_dirs(scene_dirs, weights_path, config, \
    predict_continents=False, predict_seasons=False):
    '''
    Input: scene_dirs, weights_path, config
    Output: trained FC-DenseNet model (saved to disk), training history
    '''
    # get train, val scene dirs
    print("Performing train/val split...")
    train_scene_dirs, val_scene_dirs = get_train_val_scene_dirs(scene_dirs, config)
    print("train_scene_dirs: ", train_scene_dirs)
    print("val_scene_dirs: ", val_scene_dirs)
    # save train-val-split
    train_split_filepath = weights_path.split('_weights.h5')[0] + '_train-val-split.json'
    with open(train_split_filepath, 'w') as f:
        train_split = {
            'train_scene_dirs': train_scene_dirs,
            'val_scene_dirs': val_scene_dirs,
        }
        json.dump(train_split, f, indent=4)
    # get label_encoder, patch_paths
    labels = config['training_params']['label_scheme']
    label_encoder = land_cover_utils.get_label_encoder(config, labels=labels)
    train_patch_paths = land_cover_utils.get_segmentation_patch_paths_for_scene_dirs(train_scene_dirs)
    val_patch_paths = land_cover_utils.get_segmentation_patch_paths_for_scene_dirs(val_scene_dirs)
    # set up callbacks, data generators
    callbacks = get_callbacks(weights_path, config)
    save_label_counts =  config['training_params']['class_weight'] == 'balanced'
    train_datagen = datagen.SegmentationPatchDataGenerator(train_patch_paths, config, labels='dfc', \
            label_smoothing=config['training_params']['label_smoothing'], save_label_counts=save_label_counts)
    val_datagen = datagen.SegmentationPatchDataGenerator(val_patch_paths, config, labels='dfc')
    # train_datagen = datagen.SegmentationDataGenerator(train_scene_paths, config, labels='dfc')
    # val_datagen = datagen.SegmentationDataGenerator(val_scene_paths, config, labels='dfc')
    # set up class balancing (and class masking)
    if config['training_params']['class_weight'] == 'balanced':
        print('training with balanced loss...')
        class_weights = train_datagen.get_class_weights_balanced()
    else:
        print('training with unbalanced loss...')
        class_weights = np.ones(len(label_encoder.classes_))
    def custom_loss(onehot_labels, probs):
        """
        scale loss based on class weights
        https://github.com/keras-team/keras/issues/3653#issuecomment-344068439
        """
        # computer weights based on onehot labels
        weights = tf.reduce_sum(class_weights * onehot_labels, axis=-1)
        # compute (unweighted) cross entropy loss
        unweighted_losses = categorical_crossentropy(onehot_labels, probs)
        # apply the weights, relying on broadcasting of the multiplication
        weighted_losses = unweighted_losses * weights
        # mask out '0' index
        if len(config[f'{labels}_ignored_classes']) > 0:
            print('ignoring 0 index in loss function')
            mask_value = np.zeros(len(label_encoder.classes_), dtype='float32')
            mask_value[0] = 1.0
            mask_value = tf.Variable(mask_value)
            mask = tf.reduce_all(tf.equal(onehot_labels, mask_value), axis=-1)
            mask = 1 - tf.cast(mask, tf.float32)
            weighted_losses = weighted_losses * mask
            return tf.reduce_sum(weighted_losses) / tf.reduce_sum(mask)
        # reduce the result to get your final loss
        loss = tf.reduce_mean(weighted_losses)
        return loss
    # get compiled keras model
    model = get_compiled_fc_densenet(config, label_encoder, loss=custom_loss)
    # fit keras model
    print("Training keras model...")
    history = model.fit_generator(
        train_datagen,
        epochs=config['training_params']['max_epochs'],
        validation_data=val_datagen,
        callbacks=callbacks,
        max_queue_size=config['fc_densenet_params']['batch_size'],
        use_multiprocessing=config['training_params']['use_multiprocessing'],
        workers=config['training_params']['workers']
    )
    history = land_cover_utils.make_history_json_serializable(history.history)
    # save model history
    history_filepath = weights_path.split('_weights.h5')[0] + '_history.json'
    with open(history_filepath, 'w') as f:
        json.dump(history, f, indent=4)
    print("Model history saved to: ", history_filepath)
    return model, history

def train_fc_densenet_on_season(season, config, predict_continents=False, predict_seasons=False):
    '''
    Input: continent, config
    Output: trained DenseNet model (saved to disk), training history
    '''
    print("--- Training FC-DenseNet model on {} ---".format(season))
    # get filepaths
    if not predict_continents and not predict_seasons:
        filename = 'sen12ms_season_{}_FC-DenseNet_weights.h5'.format(season)
    elif predict_continents and not predict_seasons:
        filename = 'sen12ms_season_{}_FC-DenseNet_predict-continents_weights.h5'.format(season)
    elif predict_seasons and not predict_continents:
        filename = 'sen12ms_season_{}_FC-DenseNet_predict-seasons_weights.h5'.format(season)
    else:
        filename = 'sen12ms_season_{}_FC-DenseNet_predict-continents-seasons_weights.h5'.format(season)
    weights_path = os.path.join(
        config['model_save_dir'],
        'by_season',
        filename)
    history_path = weights_path.split('_weights.h5')[0] + '_history.json'
    train_split_path = weights_path.split('_weights.h5')[0] + '_train-val-split.json'
    # check if model has already been trained
    if os.path.exists(weights_path) and os.path.exists(history_path) and os.path.exists(train_split_path):
        print('files for model {} already exist! skipping training'.format(weights_path))
        return
    # train model
    scene_dirs = land_cover_utils.get_scene_dirs_for_season(season, config, mode='segmentation')
    model, history = train_fc_densenet_on_scene_dirs(scene_dirs, weights_path, config, \
        predict_continents, predict_seasons)
    return model, history

def train_fc_densenet_on_continent(continent, config, predict_continents=False, predict_seasons=False):
    '''
    Input: season, config
    Output: trained DenseNet model (saved to disk), training history
    '''
    print("--- Training FC-DenseNet model on {} ---".format(continent))
    # get filepaths
    if not predict_continents and not predict_seasons:
        filename = 'sen12ms_continent_{}_FC-DenseNet_weights.h5'.format(continent)
    elif predict_continents and not predict_seasons:
        filename = 'sen12ms_continent_{}_FC-DenseNet_predict-continents_weights.h5'.format(continent)
    elif predict_seasons and not predict_continents:
        filename = 'sen12ms_continent_{}_FC-DenseNet_predict-seasons_weights.h5'.format(continent)
    else:
        filename = 'sen12ms_continent_{}_FC-DenseNet_predict-continents-seasons_weights.h5'.format(continent)
    weights_path = os.path.join(
        config['model_save_dir'],
        'by_continent',
        filename)
    history_path = weights_path.split('_weights.h5')[0] + '_history.json'
    train_split_path = weights_path.split('_weights.h5')[0] + '_train-val-split.json'
    # check if model exists
    if os.path.exists(weights_path) and os.path.exists(history_path) and os.path.exists(train_split_path):
        print('files for model {} already exist! skipping training'.format(weights_path))
        return
    # train model
    scene_dirs = land_cover_utils.get_scene_dirs_for_continent(continent, config, mode='segmentation')
    model, history = train_fc_densenet_on_scene_dirs(scene_dirs, weights_path, config)
    return model, history

def train_fc_densenet_on_all_scenes(config):
    '''
    Input: config
    Output: trained DenseNet model (saved to disk), training history
    '''
    print("--- Training FC-DenseNet model on all scenes ---")
    # get filepaths
    filename = 'sen12ms_all-scenes_label-smoothing-{}_balanced_ignore-3-8_FC-DenseNet_weights.h5'.format(config['training_params']['label_smoothing'])
    weights_path = os.path.join(
        config['model_save_dir'],
        filename)
    history_path = weights_path.split('_weights.h5')[0] + '_history.json'
    train_split_path = weights_path.split('_weights.h5')[0] + '_train-val-split.json'
    # check if model exists
    if os.path.exists(weights_path) and os.path.exists(history_path) and os.path.exists(train_split_path):
        print('files for model {} already exist! skipping training'.format(weights_path))
        return
    # train model
    scene_dirs = []
    for season in config['all_seasons']:
        scene_dirs.extend(land_cover_utils.get_scene_dirs_for_season(season, config))
    model, history = train_fc_densenet_on_scene_dirs(scene_dirs, weights_path, config)
    return model, history

def main(args):
    '''
    Main function: train new models, or test existing models on SEN12MS seasons/scenes
    '''
    # get config
    config_json_path = args.config_path
    with open(config_json_path, 'r') as f:
        config = json.load(f, object_hook=land_cover_utils.json_keys_to_int)
    # configure GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    GPU_ID = config['training_params'].get('gpu_id')
    if GPU_ID is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    session = tf.Session(config=tf_config)
    # show summary of keras models
    if args.model_summary:
        label_encoder = land_cover_utils.get_label_encoder(config)
        resnet = get_compiled_resnet(config, label_encoder, predict_seasons=True)
        print('----------- RESNET MODEL SUMMARY ----------')
        #print(resnet.summary())
        print('inputs: ', resnet.inputs)
        print('outputs: ', resnet.outputs)
        print()
        fc_densenet = get_compiled_fc_densenet(config, label_encoder, predict_seasons=True)
        print('---------- FC-DENSENET MODEL SUMMARY ----------')
        #print(fc_densenet.summary())
        print('inputs: ', fc_densenet.inputs)
        print('outputs: ', fc_densenet.outputs)
        print()
    # train new models on all seasons/continents
    if args.train:
        # # train resnet models
        # for continent in config['all_continents']:
        #     train_resnet_on_continent(continent, config)
        # for season in config['all_seasons']:
        #     train_resnet_on_season(season, config)
        # # train densenet models
        # for continent in config['all_continents']:
        #     train_fc_densenet_on_continent(continent, config)
        # for season in config['all_seasons']:
        #     train_fc_densenet_on_season(season, config)
        train_fc_densenet_on_all_scenes(config)
    # save each model's predictions on each scene
    if args.predict:
        # predict_saved_models_on_each_scene(config)
        # competition_model_path = os.path.join(config['model_save_dir'], 'sen12ms_all-scenes_FC-DenseNet_weights.h5')
        competition_model_path = os.path.join(config['model_save_dir'], 'sen12ms_all-scenes_label-smoothing-0.1_FC-DenseNet_weights.h5')
        label_encoder = land_cover_utils.get_label_encoder(config)
        predict_model_path_on_validation_set(competition_model_path, label_encoder, config)
    # evaluate saved models on each season/scene
    if args.test:
        evaluate_saved_models_on_each_season(config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train or test land-cover model(s)')
    parser.add_argument('-c', '--config', dest='config_path', help='config JSON path')
    parser.add_argument('--train', dest='train', action='store_true', help='train new models')
    parser.add_argument('--test', dest='test', action='store_true', help='test saved models')
    parser.add_argument('--predict', dest='predict', action='store_true', help='predict using saved models')
    parser.add_argument('--model_summary', dest='model_summary', action='store_true', help='print model summaries')
    args = parser.parse_args()
    main(args)

