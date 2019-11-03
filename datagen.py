'''
datagen.py
Adapted from https://github.com/calebrob6/land-cover/blob/master/datagen.py
'''

# imports
import math
import numpy as np
import keras.utils
from land_cover_utils import get_label_encoder

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

class SubpatchDataGenerator(keras.utils.Sequence):
    'Generates subpatch batch data for Keras'
    
    def __init__(self, patch_paths, config):
        'Initialization'

        self.patch_paths = patch_paths
        self.batch_size = config['training_params']['batch_size']
        self.steps_per_epoch = math.ceil(len(self.patch_paths) / self.batch_size)
        # assert self.steps_per_epoch * batch_size < len(patch_paths)

        self.input_size = config['training_params']['subpatch_size']
        self.num_channels = len(config['s2_input_bands'])

        self.label_encoder = get_label_encoder(config)
        self.num_classes = len(self.label_encoder.classes_)

        self.do_color_aug = config['training_params']['do_color_aug']

        self.on_epoch_end() # shuffle indices

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.steps_per_epoch

    def __getitem__(self, index):
        'Generate one batch of data'
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        batch_paths = [self.patch_paths[i] for i in indices]

        x_batch = np.zeros((self.batch_size, self.input_size, self.input_size, \
            self.num_channels), dtype=np.float32)
        labels_batch = []
        
        for i, subpatch_path in enumerate(batch_paths):
            if subpatch_path.endswith(".npz"):
                data = np.load(subpatch_path)["arr_0"].squeeze()
            elif subpatch_path.endswith(".npy"):
                data = np.load(subpatch_path).squeeze()

            #do a random crop if input_size is less than the prescribed size
            assert data.shape[0] == data.shape[1]
            data_size = data.shape[0]
            if self.input_size < data_size:
                print('WARNING: data_size is less than specified subpatch size!')
                x_idx = np.random.randint(0, data_size - self.input_size)
                y_idx = np.random.randint(0, data_size - self.input_size)
                data = data[y_idx:y_idx+self.input_size, x_idx:x_idx+self.input_size, :]

            # setup x
            if self.do_color_aug:
                x_batch[i] = color_aug(data)
            else:
                x_batch[i] = data

            # get label from filepath
            label = int(subpatch_path.split("label_")[-1].split(".npy")[0])
            labels_batch.append(label)

        # get one-hot y_batch from labels
        y_batch = self.label_encoder.transform(labels_batch)
        y_batch = keras.utils.to_categorical(y_batch, num_classes=self.num_classes)

        return x_batch.copy(), y_batch.copy()

    def on_epoch_end(self):
        'Shuffle indices'
        self.indices = np.arange(len(self.patch_paths))
        np.random.shuffle(self.indices)

