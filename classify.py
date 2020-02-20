'''
Author: Lucas Hu (lucashu@usc.edu)
Timestamp: Spring 2020
Filename: classify.py
Goal: Classify land cover of various SEN12MS scenes
Models used:FC-DenseNet, Unet
'''

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# General imports
import os
import glob
import argparse
import json
from collections import defaultdict
import numpy as np
from scipy import stats
import imageio
import tensorflow as tf
import keras
from keras.models import Model, load_model

# SEN12MS imports
from sen12ms_dataLoader import SEN12MSDataset, \
    Seasons, Sensor, S1Bands, S2Bands, LCBands

# util imports
import datagen
import models
import land_cover_utils

ALL_SEASONS = [season for season in Seasons if season != Seasons.ALL]

def get_train_val_scene_dirs(scene_dirs, config):
    '''
    Input: scene_dirs (list), config
    Output: train_scene_dirs (list), val_scene_dirs (list)
    Randomly split train/val scenes
    '''
    num_val_scenes = int(len(scene_dirs) * config['training_params']['val_size'])
    # set seed, and sort scene_dirs to get reproducible split
    np.random.seed(config['experiment_params']['val_split_seed'])
    val_scene_dirs = np.random.choice(sorted(scene_dirs), size=num_val_scenes).tolist()
    train_scene_dirs = list(set(scene_dirs) - set(val_scene_dirs))
    return train_scene_dirs, val_scene_dirs

def get_competition_train_val_scene_dirs(scene_dirs, config):
    '''
    Input: scene_dirs (list), config
    Output: train_scene_dirs (list), val_scene_dirs (list)
    Use holdout split from https://arxiv.org/pdf/2002.08254.pdf
    '''
    import csv
    # get set of holdout season/scenes (e.g. summer/scene_63)
    holdout_scenes_path = config['competition_holdout_scenes']
    holdout_scenes = set()
    with open(holdout_scenes_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
           holdout_scenes.add(f'{row["season"]}/scene_{row["scene"]}')
    # sort each scene_dir into either train or val
    train_scene_dirs = []
    val_scene_dirs = []
    for scene_dir in scene_dirs:
        for holdout_str in holdout_scenes:
            if holdout_str in scene_dir:
                val_scene_dirs.append(scene_dir)
            else:
                train_scene_dirs.append(scene_dir)
    return train_scene_dirs, val_scene_dirs

def save_segmentation_predictions_on_scene_dir(model, scene_dir, save_dir, label_encoder, config, competition_mode=False):
    '''
    Use segmentation model to predict on a single scene_dir
    Store predictions in .npz file (1 file per scene)
    '''
    if os.path.exists(save_dir):
        print('save_dir {} already exists! skipping prediction'.format(save_dir))
        return
    print('generating predictions to {}...'.format(save_dir))
    # prep datagen
    patch_paths = land_cover_utils.get_segmentation_patch_paths_for_scene_dir(scene_dir)
    patch_ids = [int(path.split('patch_')[-1]) for path in patch_paths]
    predict_datagen = datagen.SegmentationDataGenerator(patch_paths, config, labels=None)
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
    print('saved segmentation predictions to {}'.format(save_dir))
    return predictions

def predict_model_path_on_each_scene(model_path, label_encoder, config):
    '''
    Given a weights_path,
    Save predictions on each scene
    '''
    # load model from model_path
    if 'weights' in model_path and 'resnet' in model_path:
        print('WARNING - resnet models have been deprecated!')
        return
    elif 'weights' in model_path and 'DenseNet' in model_path:
        model = models.get_compiled_fc_densenet(config, label_encoder)
        model.load_weights(model_path)
    elif 'weights' in model_path and 'unet' in model_path.lower():
        model = models.get_compiled_unet(config, label_encoder, predict_logits=True)
        model.load_weights(model_path)
    else:
        print('ERROR: unable to load weights file!')
        return
    model_name = os.path.basename(model_path).split('_weights.h5')[0]
    folder = 'by_continent' if 'continent' in model_name else 'by_season'
    # predict on each scene
    for continent in config['all_continents']:
        for season in config['all_seasons']:
            # get all scenes from this continent-season
            scene_dirs = land_cover_utils.get_scene_dirs_for_continent_season(continent, season, config)
            # predict in segmentation mode
            for scene_dir in scene_dirs:
                scene_name = scene_dir.split('/')[-1]
                save_dir = os.path.join(config['segmentation_predictions_dir'],
                    folder, model_name, '{}-{}'.format(continent, season), scene_name)
                save_segmentation_predictions_on_scene_dir(model, scene_dir, save_dir, label_encoder, config)
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
    # get predictions of each saved model on each seasons/scene
    for model_path in model_filepaths:
        print('Predicting using model path: ', model_path)
        predict_model_path_on_each_scene(model_path, label_encoder, config)
    return
    
def predict_model_path_on_validation_set(model_path, label_encoder, config):
    '''
    Given a weights_path for an FC-DenseNet model,
    Save predictions on each scene in the Validation set
    '''
    if 'unet' in model_path.lower():
        model = models.get_compiled_unet(config, label_encoder, predict_logits=True)
    else:
        model = models.get_compiled_fc_densenet(config, label_encoder)
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
        save_segmentation_predictions_on_scene_dir(model, scene_dir, save_dir, label_encoder, config, competition_mode=True)
    print('finished predictions using model_path: ', model_path)
    print()

def train_segmentation_model_on_scene_dirs(scene_dirs, weights_path, config, \
    predict_continents=False, predict_seasons=False, \
    competition_mode=False, \
    predict_logits=False):
    '''
    Input: scene_dirs, weights_path, config
    save_label_counts =  config['training_params']['class_weight'] == 'balanced'
    Output: trained segmentation model (saved to disk), training history
    '''
    # get train, val scene dirs
    if competition_mode:
        print("Getting competition train/val split from holdout .csv file...")
        train_scene_dirs, val_scene_dirs = get_competition_train_val_scene_dirs(scene_dirs, config)
    else:
        print("Performing random train/val split...")
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

    # get patch paths
    train_patch_paths = land_cover_utils.get_segmentation_patch_paths_for_scene_dirs(train_scene_dirs)
    val_patch_paths = land_cover_utils.get_segmentation_patch_paths_for_scene_dirs(val_scene_dirs)

    # set up data generators with label smoothing
    if config['training_params']['label_smoothing'] == 'kmeans':
        train_datagen_labels = 'kmeans'
        print('training with kmeans label smoothing...')
    else:
        train_datagen_labels = 'naive'
        label_smoothing_factor = config['training_params']['label_smoothing_factor']
        print(f'training with naive label smoothing, factor={label_smoothing_factor}...')
    train_datagen = datagen.SegmentationDataGenerator(train_patch_paths, config, labels=train_datagen_labels)
    val_datagen = datagen.SegmentationDataGenerator(val_patch_paths, config, labels='onehot')

    # get custom loss function
    label_encoder = land_cover_utils.get_label_encoder(config)
    if config['training_params']['class_weight'] == 'balanced':
        print('training with balanced loss...')
        class_weights = train_datagen.get_class_weights_balanced()
    else:
        print('training with unbalanced loss...')
        class_weights = None
    loss = models.get_custom_loss(label_encoder, class_weights, config, from_logits=predict_logits)

    # get compiled keras model
    if 'unet' in weights_path.lower():
        print('getting compiled unet model...')
        batch_size = config['unet_params']['batch_size']
        model = models.get_compiled_unet(config, label_encoder, loss=loss, predict_logits=predict_logits)
    else:
        print('getting compiled densenet model...')
        batch_size = config['fc_densenet_params']['batch_size']
        model = models.get_compiled_fc_densenet(config, label_encoder, loss=loss)

    # fit keras model
    print("Training keras model...")
    callbacks = models.get_callbacks(weights_path, config)
    history = model.fit_generator(
        train_datagen,
        epochs=config['training_params']['max_epochs'],
        validation_data=val_datagen,
        callbacks=callbacks,
        max_queue_size=batch_size,
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

def train_competition_fc_densenet(config):
    '''
    Input: config
    Output: trained DenseNet model (saved to disk), training history
    '''
    print("--- Training FC-DenseNet model on all scenes ---")
    # get filepaths
    filename = config['competition_model']
    weights_path = os.path.join(
        config['model_save_dir'],
        'competition',
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
    model, history = train_segmentation_model_on_scene_dirs(scene_dirs, weights_path, config, \
        competition_mode=True)
    return model, history

def train_competition_unet(config):
    '''
    Input: config
    Output: trained unet model (saved to disk), training history
    '''
    print("--- Training Unet model on all scenes ---")
    # get filepaths
    filename = config['competition_model']
    weights_path = os.path.join(
        config['model_save_dir'],
        'competition',
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
    model, history = train_segmentation_model_on_scene_dirs(scene_dirs, weights_path, config, \
        predict_logits=True, competition_mode=True)
    return model, history

def main(args):
    '''
    Main function: train new models, or test existing models on SEN12MS seasons/scenes
    '''
    # get config
    config_json_path = args.config_path
    with open(config_json_path, 'r') as f:
        config = json.load(f, object_hook=land_cover_utils.json_keys_to_int)
    label_encoder = land_cover_utils.get_label_encoder(config)

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
        fc_densenet = models.get_compiled_fc_densenet(config, label_encoder, predict_seasons=True)
        print('---------- FC-DENSENET MODEL SUMMARY ----------')
        #print(fc_densenet.summary())
        print('inputs: ', fc_densenet.inputs)
        print('outputs: ', fc_densenet.outputs)
        print()

    # train new models on all seasons/continents
    if args.train:
        # # train densenet models
        # for continent in config['all_continents']:
        #     train_fc_densenet_on_continent(continent, config)
        # for season in config['all_seasons']:
        #     train_fc_densenet_on_season(season, config)
        # train_competition_fc_densenet(config)
        train_competition_unet(config)

    # save each model's predictions on each scene
    if args.predict:
        # predict_saved_models_on_each_scene(config)
        competition_model_path = os.path.join(config['model_save_dir'], 
            'competition', 
            config['competition_model'])
        print(f'predicting on competition data with model {competition_model_path}')
        print(f'label_encoder.classes_: {label_encoder.classes_}')
        predict_model_path_on_validation_set(competition_model_path, label_encoder, config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train or test land-cover model(s)')
    parser.add_argument('-c', '--config', dest='config_path', help='config JSON path')
    parser.add_argument('--train', dest='train', action='store_true', help='train new models')
    parser.add_argument('--predict', dest='predict', action='store_true', help='predict using saved models')
    parser.add_argument('--model_summary', dest='model_summary', action='store_true', help='print model summaries')
    args = parser.parse_args()
    main(args)

