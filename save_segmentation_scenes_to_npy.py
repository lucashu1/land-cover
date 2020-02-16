'''
Take patch-level .npy files and aggregate them into scene-level .npz files
E.g.: Take all the patches in /data/datasets/sen12ms_segmentation/Africa-fall/scene_40,
    and save (s2, landuse, dfc) to /data/datasets/sen12ms_segmentation/Africa-fall/scene_40.npz
'''

base = '/data/datasets/sen12ms_segmentation'

import os
import numpy as np

continent_seasons = os.listdir(base)
for continent_season in continent_seasons:
    # get scenes from this continent_season
    scenes = os.listdir(os.path.join(base, continent_season))
    for scene in scenes:
        if scene.endswith('.npz'):
            continue
        print('Processing {}, {}...'.format(continent_season, scene))
        s2 = []
        landuse = []
        dfc = []
        # get all patches from this scene
        patches = sorted(os.listdir(os.path.join(base, continent_season, scene)))
        for patch in patches:
            patch_dir = os.path.join(base, continent_season, scene, patch)
            s2.append(np.load(os.path.join(patch_dir, 's2.npy')))
            landuse.append(np.load(os.path.join(patch_dir, 'landuse.npy')))
            dfc.append(np.load(os.path.join(patch_dir, 'dfc.npy')))
        # create scene-level numpy arrays
        s2 = np.array(s2)
        landuse = np.array(landuse).astype('uint8')
        dfc = np.array(dfc).astype('uint8')
        np.savez_compressed(
            os.path.join(base, continent_season, scene + '.npz'),
            s2=s2,
            landuse=landuse,
            dfc=dfc
            )

