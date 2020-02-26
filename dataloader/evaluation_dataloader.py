import tensorflow as tf
import pandas as pd
import numpy as np
import time
import typing
import os
from utils import utils
import datetime
import pdb
import h5py
import copy
import tqdm as tqdm
import json
from dataloader import dataset_utils as du

debug = False

def evaluation_dataset(
    dataframe: pd.DataFrame, 
    stations, 
    target_time_offsets, 
    user_config, 
    target_datetimes):

    image_dim = user_config and user_config.get("image_dim") or 64
    channels = user_config and user_config.get("target_channels") or ["ch1", "ch2", "ch3", "ch4", "ch6"]
    with_pass_values = user_config and user_config.get("with_pass_values") or []
    # get localtimezone
    station_timezones = utils.get_station_timezone(stations)

    def _eval_dataset():

        targets = dataframe[dataframe.index.isin(target_datetimes)]
        assert targets.shape[0] == len(target_datetimes), "Could not find all specified targets dates in dataframe. Missing targets!"
        for row_date, row in targets.iterrows():
            image_path = row.loc['hdf5_8bit_path']
            # assert image_path.contains('nan|NAN|NaN'), "Target image has no image path!"
            
            with h5py.File(image_path, "r") as h5_data, du.get_previous_day_image_data(image_path) as h5_data_previous_day:
                
                # get h5 meta info
                global_start_idx = h5_data.attrs["global_dataframe_start_idx"]
                global_end_idx = h5_data.attrs["global_dataframe_end_idx"]
                h5_size = global_end_idx - global_start_idx
                
                lats, lons = du.get_lats_lon(h5_data, h5_size)
                if lats is None or lons is None:
                    assert True, f"No lats or lons for day: {row_date}"

                # Return one station at a time
                for station_idx, coords in stations.items():
                    # get station specefic data / meta we want
                    station_pixel_coords = (np.argmin(np.abs(lats - coords[0])), np.argmin(np.abs(lons - coords[1])))

                    # get meta info
                    lat, lont, alt = stations[station_idx] #lat, lont, alt
                    # Encoding of hours/minutes take into account the local time according
                        # to the different timezone of the different stations
                    sin_month, cos_month, sin_minute, cos_minute = utils.convert_time(row.name, station_timezones, station_idx)  # encoding months and hour/minutes

                    daytime_flag, clearsky, _, __ = row.loc[row.index.str.startswith(station_idx)]
                    if np.isnan(clearsky):
                        clearsky = 200 # close to average value

                    meta_array = np.array([sin_month,cos_month,sin_minute,cos_minute,
                                            lat, lont, alt, daytime_flag, clearsky], dtype=np.float64)
                    # Get image data
                    image_data = du.get_image_transformed(
                        h5_data, h5_data_previous_day, channels, station_pixel_coords, 
                        image_dim, row_date, with_pass_values)
                    if image_data is None:
                        if debug:
                            print("No croped image")
                        continue

                    yield (meta_array, image_data, 0.0) # no target

            if debug:
                print(f"Not yielding any results! or done... {hdf5_path}")
            

    # End of _eval_dataset function
    return tf.data.Dataset.from_generator(_eval_dataset, output_types=(tf.float64, tf.float64, tf.float64,))