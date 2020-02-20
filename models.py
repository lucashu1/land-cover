'''
models.py
Lucas Hu
'''

# imports
import sys
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Flatten
from keras.losses import categorical_crossentropy
from keras.optimizers import Nadam
from keras.callbacks import ModelCheckpoint, \
    LearningRateScheduler, ReduceLROnPlateau, \
    EarlyStopping

# ResNet imports
import sys
import os

# DenseNet imports
sys.path.append('./DenseNet')
from densenet import DenseNetFCN

# Unet imports
from segmentation_models import Unet

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


def get_custom_loss(label_encoder, class_weights, config, from_logits=False):
    '''
    Return custom loss function
    Supports masking (3, 8), and balanced class_weight
    '''
    labels = config['training_params']['label_scheme'] # 'dfc' or 'landuse'
    # get mask value ([1,0,0,...])
    mask_value = np.zeros(len(label_encoder.classes_), dtype='float32')
    mask_value[0] = 1
    mask_value = tf.Variable(mask_value)
    loss = categorical_crossentropy
    def custom_loss(onehot_labels, probs):
        """
        scale loss based on class weights
        https://github.com/keras-team/keras/issues/3653#issuecomment-344068439
        """
        # computer weights based on onehot labels
        weights = tf.reduce_sum(class_weights * onehot_labels, axis=-1)
        # compute (unweighted) cross entropy loss
        unweighted_losses = categorical_crossentropy(onehot_labels, probs, from_logits=from_logits)
        # apply the weights, relying on broadcasting of the multiplication
        weighted_losses = unweighted_losses * weights
        # mask out '0' index
        if len(config[f'{labels}_ignored_classes']) > 0:
            print('ignoring 0 index in loss function')
            mask = tf.reduce_all(tf.equal(onehot_labels, mask_value), axis=-1)
            mask = 1 - tf.cast(mask, tf.float32)
            # mask = tf.Print(mask, [mask])
            weighted_losses = weighted_losses * mask
            # weighted_losses = tf.Print(weighted_losses, [weighted_losses])
            return tf.reduce_mean(weighted_losses)
        # reduce the result to get your final loss
        loss = tf.reduce_mean(weighted_losses)
        return loss
    # return custom loss function
    return custom_loss

def get_compiled_resnet(config, label_encoder, \
    predict_continents=False, predict_seasons=False):
    '''
    Input: config dict, label_encoder
    Output: compiled ResNet model
    '''
    # imports
    import importlib
    cbam = importlib.import_module("CBAM-keras")
    from cbam.models import resnet_v1
    from cbam.utils import lr_schedule
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

def get_compiled_unet(config, label_encoder, predict_logits=False):
    '''
    Input: config dict, label_encoder
    Output: compiled Unet model
    '''
    activation = 'linear' if predict_logits else 'softmax'
    model = Unet(
        backbone_name=config['unet_params']['backbone_name'],
        encoder_weights=None,
        activation=activation,
        input_shape=config['training_params']['patch_size'],
        classes=len(label_encoder.classes_),
        decoder_filters=(256,128,64,64)
    )
    model.compile(loss='categorical_crossentropy',
        optimizer=Nadam(lr=config['unet_params']['learning_rate']),
        metrics=['accuracy'])
    return model
    







