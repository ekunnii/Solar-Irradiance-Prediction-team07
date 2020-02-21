"""
Objectif: from data path and t0, dataloader
"""
# ## debug
# import sys
# sys.path.append('../')
# ## debug

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


def BuildDataSet(
    dataframe: pd.DataFrame,
    stations: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],
    target_time_offsets: typing.List[datetime.timedelta],
    admin_config: typing.Dict[typing.AnyStr, typing.Any],# same as use functions in evaluation code, JSON format
    user_config: typing.Dict[typing.AnyStr, typing.Any],# same as use functions in evaluation code, JSON format
    target_datetimes: typing.List[datetime.datetime] = None, # This is used for evaluation!! This is the taget dates we want. No need for training
    copy_last_if_missing: bool = True,
    train: bool = True,
):
    image_dim = user_config and user_config.get("image_dim") or 64
    channels = user_config and user_config.get("target_channels") or ["ch1", "ch2", "ch3", "ch4", "ch6"]
    debug = user_config and user_config.get("debug") or False
    with_pass_values = user_config and user_config.get("with_pass_values") or []

    def _train_dataset(hdf5_path):
        # get day time index from filename, and iterate through all the day
        date = datetime.datetime.strptime( hdf5_path.split(b"/")[-1].decode(), "%Y.%m.%d.%H%M.h5")
        dataframe_day = dataframe.loc[date: date + datetime.timedelta(hours=23, minutes=45)] # we need to get the times starting at 8 to 8 the next morning
        assert dataframe_day.shape, f"No dataframe for {date}"

        # Load image file
        if not hdf5_path == 'nan' and not hdf5_path == 'NaN' and not hdf5_path == 'NAN':
            with h5py.File(hdf5_path, "r") as h5_data:

                for row_date, row in dataframe_day.iterrows():
                    # get h5 meta info
                    global_start_idx = h5_data.attrs["global_dataframe_start_idx"]
                    global_end_idx = h5_data.attrs["global_dataframe_end_idx"]
                    h5_size = global_end_idx - global_start_idx
                    
                    lats, lons = du.get_lats_lon(h5_data, h5_size)
                    if lats is None or lons is None:
                        continue

                    # Return one station at a time
                    for station_idx, coords in stations.items():
                        if not du.valid_t0_row(row, station_idx):
                            if debug:
                                print(f"Not a valid t0: {row_date}, {station_idx}")
                            continue

                        # get station specefic data / meta we want
                        station_pixel_coords = (np.argmin(np.abs(lats - coords[0])), np.argmin(np.abs(lons - coords[1])))

                        # get meta info
                        lat, lont, alt = stations[station_idx] #lat, lont, alt
                        # Warning, encoding of hours/minutes doesn't take into account the actual time according
                        # to the different timezone of the different stations?
                        sin_month,cos_month,sin_minute,cos_minute = utils.convert_time(row.name) #encoding months and hour/minutes
                        daytime_flag, clearsky, _, __ = row.loc[row.index.str.startswith(station_idx)]

                        meta_array = np.array([sin_month,cos_month,sin_minute,cos_minute,
                                                lat, lont, alt, daytime_flag, clearsky], dtype=np.float64)

                        # Get image data
                        image_data = du.get_image_transformed(
                            hdf5_path, h5_data, channels, station_pixel_coords, 
                            image_dim, row_date, with_pass_values)
                        if image_data is None:
                            if debug:
                                print("No croped image")
                            continue

                        # get station GHI targets
                        t_0 = row.name
                        station_ghis = []
                        last_available_ghi = 0
                        for offset in target_time_offsets:
                            # remove negative value
                            if (t_0 + offset) in dataframe.index:
                                if np.isnan(dataframe.loc[t_0 + offset][station_idx + "_GHI"]):
                                    station_ghis.append(
                                        round(max(dataframe.loc[t_0 + offset][station_idx + "_CLEARSKY_GHI"], 0), 2))
                                else:
                                    station_ghis.append(
                                        round(max(dataframe.loc[t_0 + offset][station_idx + "_GHI"], 0), 2))
                                    last_available_ghi = station_ghis[-1]
                            else:
                                station_ghis.append(last_available_ghi)

                        if debug:
                            print(f"Returning data for {hdf5_path}")
                        if np.isnan(station_ghis).any() or np.isnan(meta_array).any():
                            continue
                        yield (meta_array, image_data, station_ghis)

                if debug:
                    print(f"Not yielding any results! or done... {hdf5_path}")
                return
    # End of generator

    def wrap_generator(filename):
        return tf.data.Dataset.from_generator(_train_dataset, args=[filename], output_types=(tf.float64, tf.float64, tf.float64))

    # Only get dataloaders for image files that exist. 
    image_files_to_process = dataframe[('hdf5_8bit_path')] [(dataframe['hdf5_8bit_path'].str.contains('nan|NAN|NaN') == False)].unique()

    #   sub sample, take only 20% of the dataset
    image_files_to_process = np.random.choice(image_files_to_process, int(len(image_files_to_process)*0.2))

    # Create an interleaved dataset so it's faster. Each dataset is responsible to load it's own compressed image file.
    files = tf.data.Dataset.from_tensor_slices(image_files_to_process)
    dataset = files.interleave(wrap_generator, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return dataset


class TrainingDataSet(tf.data.Dataset):
    def __new__(
        cls,
        data_frame_path: typing.AnyStr,
        stations: typing.Dict[typing.AnyStr, typing.Tuple],
        admin_config: typing.Dict[typing.AnyStr, typing.Any], # JSON; Training config file, looks like the admin config for the evaluation
        user_config: typing.Dict[typing.AnyStr, typing.Any] = None, # JSON; Model options or data loader options
        copy_last_if_missing: bool = True,
        train: bool = True,
        scratch_dir: str = None,
    ):

        fast_data_frame_path = du.set_faster_path(data_frame_path, scratch_dir)

        dataframe = pd.read_pickle(fast_data_frame_path)
        if train:
            if "start_bound" in admin_config:
                dataframe = dataframe[dataframe.index >= datetime.datetime.fromisoformat(
                    admin_config["start_bound"] + ' 08:00:00')]
            if "end_bound" in admin_config:
                dataframe = dataframe[dataframe.index <= datetime.datetime.fromisoformat(
                    admin_config["end_bound"] + ' 07:45:00')]
        else:
            # year 2015 is used as validation set
            dataframe = dataframe[dataframe.index >= datetime.datetime.fromisoformat('2015-01-01 08:00:00')]
            dataframe = dataframe[dataframe.index <= datetime.datetime.fromisoformat('2015-12-31 07:45:00')]

        target_time_offsets = [pd.Timedelta(d).to_pytimedelta() for d in admin_config["target_time_offsets"]]
        return BuildDataSet(dataframe, stations, target_time_offsets, admin_config, user_config)

