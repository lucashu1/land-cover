'''
datagen.py
Adapted from https://github.com/calebrob6/land-cover/blob/master/datagen.py
'''

# imports
import os
import math
import random
import joblib
import numpy as np
from numbers import Number
import keras.utils
from land_cover_utils import get_label_encoder, \
    get_continents_label_encoder, get_seasons_label_encoder

def color_aug(colors):
    n_ch = colors.shape[-1]
    contra_adj = 0.05
    bright_adj = 0.05

    ch_mean = np.mean(colors, axis=(0, 1), keepdims=True).astype(np.float32)

    contra_mul = np.random.uniform(1-contra_adj, 1+contra_adj, (1,1,n_ch)).astype(np.float32)
    bright_mul = np.random.uniform(1-bright_adj, 1+bright_adj, (1,1,n_ch)).astype(np.float32)

    colors = (colors - ch_mean) * contra_mul + ch_mean * bright_mul
    return colors

class SegmentationDataGenerator(keras.utils.Sequence):
    'Generates semantic segmentation batch data for Keras'
    
    def __init__(self, patch_paths, config, labels='onehot'):
        '''
        patch_paths: list of patch paths (without .npy extension)
        labels: one of 'kmeans', 'naive', 'onehot', or None
        '''

        self.config = config
        self.labels = labels
        self.label_scheme = config['training_params']['label_scheme']
        self.patch_paths = patch_paths
        self.batch_size = config['fc_densenet_params']['batch_size']
        self.steps_per_epoch = math.ceil(len(self.patch_paths) / self.batch_size)
        # assert self.steps_per_epoch * batch_size < len(patch_paths)

        self.input_size = config['training_params']['patch_size']
        self.num_channels = len(config['s2_input_bands']) + len(config['s1_input_bands'])
        self.do_color_aug = config['training_params']['do_color_aug']

        # labels config
        if self.labels is not None:
            self.label_encoder = get_label_encoder(config)
            self.num_classes = len(self.label_encoder.classes_)
            self.removed_classes = np.array(config[f'{self.label_scheme}_removed_classes'])
            self.ignored_classes = np.array(config[f'{self.label_scheme}_ignored_classes'])
            self.label_smoothing_factor = config['training_params']['label_smoothing_factor']
            if config['training_params']['class_weight'] == 'balanced':
                self.label_counts = None
                self.label_counts = self.get_label_counts()

        # kmeans label smoothing helpers
        if self.labels == 'kmeans':
            kmeans_tup = joblib.load(config['kmeans_params']['kmeans_path'])
            self.kmeans, self.cluster_to_label_mapping, self.cluster_to_label_probabilities = kmeans_tup

        self.on_epoch_end()

    def get_label_counts(self):
        ''' return label counts '''
        if self.labels is None:
            return
        if self.label_counts is not None:
            return self.label_counts
        print('datagen getting label counts...')
        label_counts = np.zeros(self.num_classes, dtype='uint64')
        for patch in self.patch_paths:
            labels = np.load(os.path.join(patch, f'{self.label_scheme}.npy'))
            classes, counts = np.unique(labels, return_counts=True)
            try:
                classes = self.label_encoder.transform(classes)
            except:
                continue
            for c, count in zip(classes, counts):
                label_counts[c] += count
        num_samples = np.sum(label_counts)
        for c, count in enumerate(label_counts):
            print('{}: {}% ({} instances)'.format(
                self.label_encoder.classes_[c], 
                100*count/num_samples, 
                count))
        return label_counts

    def get_class_weights_balanced(self):
        ''' get balanced class weights '''
        label_counts = self.get_label_counts()
        num_samples = np.sum(label_counts)
        print('label_counts total num_samples: ', num_samples)
        class_weight = np.zeros(self.num_classes)
        for label in range(len(label_counts)):
            weight = num_samples / (self.num_classes*label_counts[label])
            print('{}: class_weight = {}'.format(self.label_encoder.classes_[label], weight))
            class_weight[label] = weight
        print('total class_weight: ', np.sum(class_weight))
        return class_weight

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.steps_per_epoch

    def __getitem__(self, index):
        'Generate one batch of data'

        x_batch = []
        y_batch = []

        # out of patches --> call on_epoch_end to re-shuffle
        if len(self.patch_paths) - self.patch_index < self.batch_size:
            self.on_epoch_end()

        # continue getting samples until len(x_batch) == batch_size
        while len(x_batch) < self.batch_size and self.patch_index < len(self.patch_paths):
            patch_path = self.patch_paths[self.patch_index]
            self.patch_index += 1

            # get S1
            if len(self.config['s1_input_bands']) > 0:
                s1 = np.load(os.path.join(patch_path, "s1.npy")).astype(np.float32)
                s1 = s1.squeeze()
                if np.any(np.isnan(s1)):
                    continue
                s1 = (s1 - self.config['s1_band_means']) / self.config['s1_band_std']

            # get S2
            s2 = np.load(os.path.join(patch_path, "s2.npy")).astype(np.float32)
            s2 = s2.squeeze()
            if self.config['training_params']['normalize_mode'] == 'standardize':
                s2 = (s2 - self.config['s2_band_means']) / self.config['s2_band_std']
            else:
                s2 /= self.config['s2_max_val']

            # get labels
            if self.labels is not None:
                labels = np.load(os.path.join(patch_path, f"{self.label_scheme}.npy"))
                labels = labels.squeeze()
                assert labels.shape[0] == labels.shape[1]
                assert s2.shape[0] == labels.shape[0]

            # check dimensions
            if len(self.config['s1_input_bands']) > 0:
                assert s1.shape[0] == s1.shape[1]
                assert s1.shape[0] == self.input_size
            assert s2.shape[0] == s2.shape[1]
            assert s2.shape[0] == self.input_size

            # check for removed/ignored labels
            if self.labels is not None and len(self.removed_classes) > 0:
                num_removed_classes = np.sum([np.count_nonzero(labels==c) \
                    for c in self.removed_classes])
                if num_removed_classes > 0:
                    continue

            # mask out ignored classes (use reserved '0' index)
            if self.labels is not None and len(self.ignored_classes) > 0:
                for c in self.ignored_classes:
                    labels[labels == c] = 0

            # setup x
            x = np.concatenate((s1,s2), axis=-1) if len(self.config['s1_input_bands']) > 0 else s2
            x_batch.append(x)

            # setup y (apply label-encoder)
            if self.labels is not None:
                labels = self.label_encoder.transform(labels.flatten())
                labels = labels.reshape((self.input_size, self.input_size))
                y_batch.append(labels)

        # return X only
        if self.labels is None:
            return np.array(x_batch)

        # convert x, y to numpy arrays
        x_batch = np.array(x_batch)
        y_batch = np.array(y_batch)

        # one-hot encode labels
        if len(self.ignored_classes) > 0:
            y_batch_ignored = np.where(y_batch == 0) # get indices where y_batch = '0'
        y_batch = keras.utils.to_categorical(y_batch, num_classes=self.num_classes)

        # naive label smoothing
        # https://www.pyimagesearch.com/2019/12/30/label-smoothing-with-keras-tensorflow-and-deep-learning/
        if self.labels == 'naive' and  self.label_smoothing_factor > 0:
           y_batch *= (1.0-self.label_smoothing_factor)
           y_batch += (self.label_smoothing_factor / y_batch.shape[-1])

        # k-means label-smoothing
        elif self.labels == 'kmeans':
            x_pixels = x_batch.reshape((-1, self.num_channels))
            cluster_inds = self.kmeans.predict(x_pixels)
            class_probs = self.cluster_to_label_probabilities[cluster_inds]
            if len(self.ignored_classes) > 0:
                class_probs = np.delete(class_probs, self.ignored_classes-1, axis=-1) # remove ignored classes
                class_probs = np.concatenate((np.zeros((class_probs.shape[0],1)), class_probs), axis=-1) # add dummy channel (for masking)
            class_probs = class_probs.reshape((x_batch.shape[0], self.input_size, self.input_size, -1))
            y_batch = class_probs

        # set ignored pixels = [1, 0, 0, ...]
        if len(self.ignored_classes) > 0:
            ignored_vec = np.zeros(self.num_classes)
            ignored_vec[0] = 1
            y_batch[y_batch_ignored] = ignored_vec

        assert x_batch.shape[0] == y_batch.shape[0]
        return x_batch.copy(), y_batch.copy()

    def on_epoch_end(self):
        ''' Shuffle patches '''
        self.patch_index = 0
        if self.labels is not None:
            np.random.shuffle(self.patch_paths)


