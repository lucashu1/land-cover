'''
count_landuse_zeros.py
Author: Lucas Hu
Iterates over all scenes in SEN12MS, and counts '0' occurrences in LCCS land-use band
Saves results to 'zero_landuse_counts.csv'
'''
# imports
import csv
import numpy as np
import pickle
from sen12ms_dataLoader import SEN12MSDataset, Seasons, Sensor, S1Bands, S2Bands, LCBands
# set up dataset
sen12ms = SEN12MSDataset("/data/datasets/sen12ms")
# open CSV writer
with open('zero_landuse_counts.csv', 'w') as f:
    writer = csv.writer(f)
	# write header
    writer.writerow(['season', 'scene', 'patch', 'num_landuse_zeros'])
	# iterate over all scenes
    for season in ["ROIs1158_spring", "ROIs1868_summer", "ROIs1970_fall", "ROIs2017_winter"]:
        scenes = sen12ms.get_scene_ids(season)
        for scene in scenes:
            print("Season: {}, Scene: {}".format(season, scene))
			# iterate over patches in scene
            patches = sen12ms.get_patch_ids(season, scene)
            for patch in patches:
                # check for a 0
                s1, s2, lc, bounds = sen12ms.get_s1s2lc_triplet(season, scene, patch)
                landuse_values, landuse_counts = np.unique(lc[2, :, :], return_counts=True)
                if 0 in landuse_values:
                    count = landuse_counts[0]
                    print('Found {} "0"s! Season: {}, Scene: {}, Patch: {}'.format(count, season, scene, patch))
                    writer.writerow([season, scene, patch, count])

