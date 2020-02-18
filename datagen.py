'''
datagen.py
Adapted from https://github.com/calebrob6/land-cover/blob/master/datagen.py
'''

# imports
import os
import math
import random
import numpy as np
import keras.utils
from land_cover_utils import get_label_encoder, \
    get_continents_label_encoder, get_seasons_label_encoder

def color_aug(colors):
    # TODO: update this
    n_ch = colors.shape[-1]
    contra_adj = 0.05
    bright_adj = 0.05

    ch_mean = np.mean(colors, axis=(0, 1), keepdims=True).astype(np.float32)

    contra_mul = np.random.uniform(1-contra_adj, 1+contra_adj, (1,1,n_ch)).astype(np.float32)
    bright_mul = np.random.uniform(1-bright_adj, 1+bright_adj, (1,1,n_ch)).astype(np.float32)

    colors = (colors - ch_mean) * contra_mul + ch_mean * bright_mul
    return colors

class SegmentationPatchDataGenerator(keras.utils.Sequence):
    'Generates semantic segmentation batch data for Keras'
    
    def __init__(self, patch_paths, config, labels='dfc', label_smoothing=0, save_label_counts=False):
        'Initialization'

        self.labels = labels
        self.patch_paths = patch_paths
        self.batch_size = config['fc_densenet_params']['batch_size']
        self.steps_per_epoch = math.ceil(len(self.patch_paths) / self.batch_size)
        # assert self.steps_per_epoch * batch_size < len(patch_paths)

        self.input_size = config['training_params']['patch_size']
        self.num_channels = len(config['s2_input_bands'])

        if labels is not None:
            self.label_encoder = get_label_encoder(config, labels=labels)
            self.num_classes = len(self.label_encoder.classes_)
            self.removed_classes = config['{}_removed_classes'.format(labels)]
            self.label_smoothing = label_smoothing
            if save_label_counts:
                self.label_counts = None
                self.label_counts = self.get_label_counts()

        self.max_s1_val = config['s1_max_val']
        self.max_s2_val = config['s2_max_val']
        self.do_color_aug = config['training_params']['do_color_aug']

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
            labels = np.load(os.path.join(patch, '{}.npy'.format(self.labels)))
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

        while len(x_batch) < self.batch_size and self.patch_index < len(self.patch_paths):
            patch_path = self.patch_paths[self.patch_index]
            self.patch_index += 1

            # get S1
            # s1 = np.load(os.path.join(patch_path, "s1.npy")).astype(np.float32)
            # s1 = s2.squeeze()
            # s1 /= self.max_s1_val
            # get S2
            s2 = np.load(os.path.join(patch_path, "s2.npy")).astype(np.float32)
            s2 = s2.squeeze()
            s2 /= self.max_s2_val
            # get labels
            if self.labels is not None:
                labels = np.load(os.path.join(patch_path, "{}.npy".format(self.labels)))
                labels = labels.squeeze()
                assert labels.shape[0] == labels.shape[1]
                assert s2.shape[0] == labels.shape[0]

            # check dimensions
            # assert s1.shape[0] == s1.shape[1]
            # assert s1.shape[0] == self.input_size
            assert s2.shape[0] == s2.shape[1]
            assert s2.shape[0] == self.input_size

            if self.labels is not None:
                # check for removed/ignored labels
                num_removed_classes = np.sum([np.count_nonzero(labels==c) for c in self.removed_classes])
                if num_removed_classes > 0:
                    continue

            # setup x
            if self.do_color_aug:
                x = color_aug(s2)
                x_batch.append(color_aug(s2))
            else:
                # x = np.concatenate((s1,s2), axis=-1)
                x = s2
                x_batch.append(x)

            if self.labels is not None:
                # setup y (apply label-encoder)
                labels = self.label_encoder.transform(labels.flatten())
                labels = labels.reshape((self.input_size, self.input_size))
                y_batch.append(labels)

        # return X only
        if self.labels == None:
            return np.array(x_batch)

        # convert x, y to numpy arrays
        x_batch = np.array(x_batch)
        y_batch = np.array(y_batch)

        # one-hot encode labels
        y_batch = keras.utils.to_categorical(y_batch, num_classes=self.num_classes)

        # apply label smoothing: https://www.pyimagesearch.com/2019/12/30/label-smoothing-with-keras-tensorflow-and-deep-learning/
        if self.label_smoothing is not None and self.label_smoothing > 0:
           y_batch *= (1.0-self.label_smoothing)
           y_batch += (self.label_smoothing / y_batch.shape[-1])

        assert x_batch.shape[0] == y_batch.shape[0]
        return x_batch.copy(), y_batch.copy()

    def on_epoch_end(self):
        ''' Shuffle patches '''
        self.patch_index = 0
        if self.labels is not None:
            np.random.shuffle(self.patch_paths)

class SubpatchDataGenerator(keras.utils.Sequence):
    'Generates subpatch batch data for Keras'
    
    def __init__(self, subpatch_paths, config, return_labels=True, \
        return_continents=False, return_seasons=False):
        'Initialization'

        self.return_labels = return_labels
        self.subpatch_paths = subpatch_paths
        self.batch_size = config['resnet_params']['batch_size']
        self.steps_per_epoch = math.ceil(len(self.subpatch_paths) / self.batch_size)
        # assert self.steps_per_epoch * batch_size < len(subpatch_paths)

        self.input_size = config['training_params']['subpatch_size']
        self.num_channels = len(config['s2_input_bands'])

        self.label_encoder = get_label_encoder(config)
        self.num_classes = len(self.label_encoder.classes_)

        self.max_input_val = config['s2_max_val']
        self.do_color_aug = config['training_params']['do_color_aug']

        self.return_continents = return_continents
        self.return_seasons = return_seasons
        self.continents_label_encoder = get_continents_label_encoder(config)
        self.seasons_label_encoder = get_seasons_label_encoder(config)

        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.steps_per_epoch

    def __getitem__(self, index):
        'Generate one batch of data'
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        batch_paths = [self.subpatch_paths[i] for i in indices]

        x_batch = []
        labels_batch = []
        continents_batch = []
        seasons_batch = []
        
        for i, subpatch_path in enumerate(batch_paths):
            if subpatch_path.endswith(".npz"):
                data = np.load(subpatch_path)["arr_0"].squeeze().astype(np.float32)
            elif subpatch_path.endswith(".npy"):
                data = np.load(subpatch_path).squeeze().astype(np.float32)

            #do a random crop if input_size is less than the prescribed size
            assert data.shape[0] == data.shape[1]
            data_size = data.shape[0]
            if self.input_size < data_size:
                print('WARNING: data_size is less than specified subpatch size!')
                x_idx = np.random.randint(0, data_size - self.input_size)
                y_idx = np.random.randint(0, data_size - self.input_size)
                data = data[y_idx:y_idx+self.input_size, x_idx:x_idx+self.input_size, :]

            if self.return_labels:
                # get label from filepath
                label = int(subpatch_path.split("label_")[-1].split(".npy")[0])
                if label == 0:
                    continue
                labels_batch.append(label)

            # setup x
            data /= self.max_input_val
            if self.do_color_aug:
                x_batch.append(color_aug(data))
            else:
                x_batch.append(data)

            # get season, continent
            continent_season = subpatch_path.split('/scene_')[0].split('/')[-1]
            continent = continent_season.split('-')[0]
            continents_batch.append(continent)
            season = continent_season.split('-')[1]
            seasons_batch.append(season)

        # convert x_batch to numpy array
        x_batch = np.array(x_batch)

        # return X only
        if not self.return_labels:
            return np.array(x_batch)

        # get one-hot y_batch from labels
        y_batch = self.label_encoder.transform(labels_batch)
        y_batch = keras.utils.to_categorical(y_batch, num_classes=self.num_classes)

        # one-hot encode continents, seasons
        continents_batch = self.continents_label_encoder.transform(continents_batch)
        continents_batch = keras.utils.to_categorical(continents_batch, \
            num_classes=continents_batch.shape[1])
        seasons_batch = self.seasons_label_encoder.transform(seasons_batch)
        seasons_batch = keras.utils.to_categorical(seasons_batch, \
            num_classes=seasons_batch.shape[1])

        assert x_batch.shape[0] == y_batch.shape[0]
        assert continents_batch.shape[0] == seasons_batch.shape[0]
        assert continents_batch.shape[0] == x_batch.shape[0]
        #if (x_batch.shape[0] != self.batch_size):
            #print('warning: x_batch.shape[0] ({}) < batch_size ({})'\
            #    .format(x_batch.shape[0], self.batch_size))

        if not return_continents and not return_seasons:
            return x_batch.copy(), y_batch.copy()
        elif return_continents and not return_seasons:
            return x_batch.copy(), (y_batch.copy(), continents_batch.copy())
        elif return_seasonsa and not return_continents:
            return x_batch.copy(), (y_batch.copy(), seasons_batch.copy())
        else:
            return x_batch.copy(), \
                (y_batch.copy(), continents_batch.copy(), seasons_batch.copy())

    def on_epoch_end(self):
        'Shuffle indices'
        self.indices = np.arange(len(self.subpatch_paths))
        if self.return_labels:
            np.random.shuffle(self.indices)

