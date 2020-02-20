'''
get_locations_for_every_scene.py
Author: Lucas Hu
Map each SEN12MS (season, scene) --> (latlng, country, continent)
Store results in a pickle/json file
'''

# imports
import os
import time
import json
import traceback
from geopy.geocoders import GoogleV3
from sen12ms_dataLoader import SEN12MSDataset, Seasons, Sensor, S1Bands, S2Bands, LCBands

# sen12ms dataset
sen12ms = SEN12MSDataset("/data/datasets/sen12ms")
all_seasons = ["ROIs1158_spring", "ROIs1868_summer", "ROIs1970_fall", "ROIs2017_winter"]

API_KEY = None #TODO: fill this in

# google Maps geolocator
geolocator = GoogleV3(api_key=API_KEY)

# country short_name --> continent dict
continents_path = './continents.json'
with open(continents_path) as f:
    country_to_continent = json.load(f)

# get country shortname from lat, long
def get_country_from_lat_long(lat, lng):
    loc = geolocator.reverse((lat, lng), exactly_one=True)
    for component in loc.raw['address_components']:
        if 'country' in component['types']:
            return component['short_name']

# get continent from (season, scene_id)
def get_latlng_country_continent_from_season_and_scene_id(season, scene):
    first_patch_id = sen12ms.get_patch_ids(season, scene)[0]
    _, _, _, bounds = sen12ms.get_s1s2lc_triplet(season, scene, first_patch_id)
    lng, lat = bounds['lnglat']
    country = get_country_from_lat_long(lat, lng)
    continent = country_to_continent[country]
    return (lat, lng), country, continent

# get location for every (season, scene)
save_path = './all_scene_locations.json'
max_num_tries = 10
if os.path.exists(save_path):
    with open(save_path, 'r') as f:
        all_scene_locations = json.load(f)
else:
    all_scene_locations = {}
for season in all_seasons:
    scene_ids = list(sen12ms.get_scene_ids(season))
    i = 0
    while i < len(scene_ids):
        scene = scene_ids[i]
        season_scene = "{}_{}".format(season, scene)
        num_tries = 0
        # check if we already have results for this season
        if season_scene in all_scene_locations:
            print("already got location for " + season_scene)
            i += 1
            continue
        # hit google geolocation API
        while num_tries < max_num_tries:
            # try geolocating
            try:
                latlng, country, continent = get_latlng_country_continent_from_season_and_scene_id(season, scene) 
                all_scene_locations[season_scene] = (latlng, country, continent)
                print("successfully got location for " + season_scene)
                time.sleep(1)
                i += 1
                break
            # try again, until we hit max_num_tries
            except Exception as e:
                print(traceback.format_exc())
                num_tries += 1
                time.sleep(5)
        if num_tries >= max_num_tries:
            print("Warning: exceeded max_num_tries for " + season_scene)
        # dump results after each scene
        with open(save_path, 'w') as f:
            json.dump(all_scene_locations, f, indent=4)



