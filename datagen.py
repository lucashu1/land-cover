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

class SegmentationDataGenerator(keras.utils.Sequence):
    'Generates semantic segmentation batch data for Keras'
    
    def __init__(self, scene_paths, config, labels='dfc'):
        'Initialization'

        np.random.seed(config['experiment_params']['val_split_seed'])

        self.labels = labels # 'dfc', 'landuse', or None
        self.scene_paths = sorted(scene_paths)
        self.batch_size = config['fc_densenet_params']['batch_size']
        self.steps_per_epoch = math.ceil(self.count_patches() / self.batch_size)
        # assert self.steps_per_epoch * batch_size < len(patch_paths)

        self.input_size = config['training_params']['patch_size']
        self.num_channels = len(config['s2_input_bands'])

        self.label_encoder = get_label_encoder(config, labels=labels)
        self.num_classes = len(self.label_encoder.classes_)
        self.unknown_classes = config['{}_unknown_classes'.format(labels)]

        self.max_input_val = config['s2_max_val']
        self.do_color_aug = config['training_params']['do_color_aug']
        
        self.num_scenes_in_mem = config['training_params']['num_scenes_in_mem'] # number of scenes to hold in memory at once

        self.on_epoch_end()

    def count_patches(self):
        ''' count total num patches in scenes '''
        print('counting patches in dataloader...')
        num_patches = 0
        for scene in self.scene_paths:
            num_patches_from_scene = len(os.listdir(scene.replace('.npz', '')))
            num_patches += num_patches_from_scene
            # x = np.load(scene)
            # num_patches += x['s2'].shape[0]
        print('num patches: {}'.format(num_patches))
        return num_patches

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.steps_per_epoch

    def __getitem__(self, index):
        'Generate one batch of data'

        # check if we have exhausted the scenes currently in memory
        # if yes, load new scenes into memory
        if self.patch_ind >= self.curr_s2.shape[0]:
            self.scene_ind += self.num_scenes_in_mem
            self.load_scenes_into_mem()

        x_batch = []
        y_batch = []

        # load 'batch_size' patches
        while len(x_batch) < self.batch_size and self.patch_ind < self.curr_s2.shape[0]:
            s2 = self.curr_s2[self.patch_ind]
            if self.labels is not None:
                labels = self.curr_labels[self.patch_ind]
                num_unknown = np.sum([np.count_nonzero(labels==c) for c in self.unknown_classes])
                if num_unknown > 0:
                    continue
    
            # setup x
            x_batch.append(s2)

            if self.labels is not None:
                # setup y (apply label-encoder)
                labels = self.label_encoder.transform(labels.flatten())
                labels = labels.reshape((self.input_size, self.input_size))
                y_batch.append(labels)

            self.patch_ind += 1

        # convert X to np array
        x_batch = np.array(x_batch)

        # return X only
        if self.labels is None:
            return x_batch

        # one-hot encode labels
        y_batch = np.array(y_batch)
        y_batch = keras.utils.to_categorical(y_batch, num_classes=self.num_classes)

        assert x_batch.shape[0] == y_batch.shape[0]
        return x_batch.copy(), y_batch.copy()

    def load_scenes_into_mem(self):
        ''' Load next batch of scenes into memory '''
        self.curr_s2 = []
        self.curr_labels = []
        # if we are out of scenes, then call on_epoch_end()
        if self.scene_ind >= len(self.scene_paths):
            self.on_epoch_end()
        # get next batch of scenes
        scenes = self.scene_paths[self.scene_ind:self.scene_ind + self.num_scenes_in_mem]
        for scene in scenes:
            print('loading {} into memory...'.format(scene))
            # load data from this scene, store in mem
            scene_data = np.load(scene)
            self.curr_s2.append(scene_data['s2'])
            if self.labels is not None:
                self.curr_labels.append(scene_data[self.labels])
        print()
        self.curr_s2 = np.concatenate(self.curr_s2, axis=0).astype(np.float32)
        if self.labels is not None:
            self.curr_labels = np.concatenate(self.curr_labels, axis=0)
        # start at patch 0 (of the patches currently in memory)
        self.patch_ind = 0
        if self.labels is not None:
            shuffle_inds = np.random.permutation(len(self.curr_s2))
            self.curr_s2 = self.curr_s2[shuffle_inds]
            self.curr_labels = self.curr_labels[shuffle_inds]
        self.curr_s2 /= self.max_input_val # normalize data
        if self.labels is None:
            return

    def on_epoch_end(self):
        ''' Shuffle scene_paths '''
        if self.labels is not None:
            self.scene_paths = np.random.permutation(self.scene_paths)
        self.scene_ind = 0
        self.load_scenes_into_mem()
        

class SegmentationPatchDataGenerator(keras.utils.Sequence):
    'Generates semantic segmentation batch data for Keras'
    
    def __init__(self, patch_paths, config, labels='dfc', label_smoothing=0):
        'Initialization'

        self.labels = labels
        self.patch_paths = patch_paths
        self.batch_size = config['fc_densenet_params']['batch_size']
        self.steps_per_epoch = math.ceil(len(self.patch_paths) / self.batch_size)
        # assert self.steps_per_epoch * batch_size < len(patch_paths)

        self.input_size = config['training_params']['patch_size']
        self.num_channels = len(config['s2_input_bands'])

        self.label_encoder = get_label_encoder(config)
        self.num_classes = len(self.label_encoder.classes_)
        self.unknown_classes = config['dfc_unknown_classes']
        self.label_smoothing = label_smoothing

        self.max_input_val = config['s2_max_val']
        self.do_color_aug = config['training_params']['do_color_aug']

        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.steps_per_epoch

    def __getitem__(self, index):
        'Generate one batch of data'
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        batch_paths = [self.patch_paths[i] for i in indices]

        x_batch = []
        y_batch = []
        continents_batch = []
        seasons_batch = []
        
        for i, patch_path in enumerate(batch_paths):
            s2 = np.load(os.path.join(patch_path, "s2.npy")).astype(np.float32)
            s2 = s2.squeeze()
            s2 /= self.max_input_val
            labels = np.load(os.path.join(patch_path, "{}.npy".format(self.labels)))
            labels = labels.squeeze()

            # check dimensions
            assert s2.shape[0] == s2.shape[1]
            assert labels.shape[0] == labels.shape[1]
            assert s2.shape[0] == labels.shape[0]
            assert s2.shape[0] == self.input_size

            if self.labels is not None:
                # check for missing/unknown labels
                num_unknown = np.sum([np.count_nonzero(labels==c) for c in self.unknown_classes])
                if num_unknown > 0:
                    continue

            # setup x
            if self.do_color_aug:
                x_batch.append(color_aug(s2))
            else:
                x_batch.append(s2)

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
        'Shuffle indices'
        self.indices = np.arange(len(self.patch_paths))
        if self.labels is not None:
            np.random.shuffle(self.indices)

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

