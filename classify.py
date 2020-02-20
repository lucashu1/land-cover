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
    '''
    num_val_scenes = int(len(scene_dirs) * config['training_params']['val_size'])
    # set seed, and sort scene_dirs to get reproducible split
    np.random.seed(config['experiment_params']['val_split_seed'])
    val_scene_dirs = np.random.choice(sorted(scene_dirs), size=num_val_scenes).tolist()
    train_scene_dirs = list(set(scene_dirs) - set(val_scene_dirs))
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
    print('saved fc-densenet predictions to {}'.format(save_dir))
    return predictions

def predict_model_path_on_each_scene(model_path, label_encoder, config):
    '''
    Given a weights_path,
    Save predictions on each scene
    '''
    # load model from model_path
    if 'weights' in model_path and 'resnet' in model_path:
        model = models.get_compiled_resnet(config, label_encoder)
        model.load_weights(model_path)
        mode = 'subpatches'
    elif 'weights' in model_path and 'DenseNet' in model_path:
        model = models.get_compiled_fc_densenet(config, label_encoder)
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
    if 'Unet' in model_path:
        model = models.get_compiled_unet(config, label_encoder, predict_logits=False)
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
            model = models.get_compiled_resnet(config, label_encoder)
            model.load_weights(model_path)
        elif 'weights' in model_path and 'DenseNet' in model_path:
            model = models.get_compiled_fc_densenet(config, label_encoder)
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
            model = models.get_compiled_resnet(config, label_encoder)
            model.load_weights(model_path)
        elif 'weights' in model_path and 'DenseNet' in model_path:
            model = models.get_compiled_fc_densenet(config, label_encoder)
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

def train_segmentation_model_on_scene_dirs(scene_dirs, weights_path, config, \
    predict_continents=False, predict_seasons=False, \
    predict_logits=False):
    '''
    Input: scene_dirs, weights_path, config
    save_label_counts =  config['training_params']['class_weight'] == 'balanced'
    Output: trained segmentation model (saved to disk), training history
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
        class_weights = np.ones(len(label_encoder.classes_))
    loss = models.get_custom_loss(label_encoder, class_weights, config, from_logits=predict_logits)

    # get compiled keras model
    model = models.get_compiled_fc_densenet(config, label_encoder, loss=loss)

    # fit keras model
    print("Training keras model...")
    callbacks = models.get_callbacks(weights_path, config)
    history = model.fit_generator(
        train_datagen,
        epochs=config['training_params']['max_epochs'],
        validation_data=val_datagen,
        callbacks=callbacks,
        max_queue_size=config['unet_params']['batch_size'],
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
    model, history = train_segmentation_model_on_scene_dirs(scene_dirs, weights_path, config, predict_logits=True)
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
        competition_model_path = os.path.join(config['model_save_dir'], config['competition_model'])
        print(f'predicting on competition data with model {competition_model_path}')
        label_encoder = land_cover_utils.get_label_encoder(config)
        print(f'label_encoder.classes_: {label_encoder.classes_}')
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

