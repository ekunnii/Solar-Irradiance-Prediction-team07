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
from shutil import copyfile
import datetime
import pdb
import h5py
import copy

def set_faster_path(original_file, scratch_dir):
    if scratch_dir == None or scratch_dir == "":
        print(f"The dataframe left at it's original location: {original_file}")
        return original_file
    
    split = os.path.split(original_file)
    destination = scratch_dir + "/" + split[-1]
    if not os.path.exists(destination):
        copyfile(original_file, destination)
    print(f"The dataframe has been copied to: {destination}")
    return destination

def get_lats_lon(h5_data: h5py.File, h5_size: int):
    idx, lats, lons = 0, None, None
    while (lats is None or lons is None) and idx < h5_size:
        lats, lons = utils.fetch_hdf5_sample("lat", h5_data, idx), utils.fetch_hdf5_sample("lon", h5_data, idx)
        idx += 1
    return lats, lons

def get_image_transformed(h5_data: h5py.File, channels, image_time_offset_idx: int, station_pixel_coords, cropped_img_size = 64):
    # get the right image
    # TODO Normalize images?
    all_channels = np.empty([cropped_img_size, cropped_img_size, len(channels)])
    for ch_idx, channel in enumerate(channels):
        raw_img = utils.fetch_hdf5_sample(channel, h5_data, image_time_offset_idx)
        if raw_img is None or raw_img.shape != (650, 1500): 
            return None
        
        array_cropped = utils.crop(copy.deepcopy(raw_img), station_pixel_coords, cropped_img_size)
        # raw_data[array_idx, station_idx, channel_idx, ...] = cv.flip(array_cropped, 0) # TODO why the flip??

        #array = (((array.astype(np.float32) - norm_min) / (norm_max - norm_min)) * 255).astype(np.uint8) # TODO norm? 
        array_cropped = array_cropped.astype(np.uint8) # convert to image format
        all_channels[:,:,ch_idx] = array_cropped
    
    return all_channels


# This is used both in the evaluation and for testing
def BuildDataSet(
    dataframe: pd.DataFrame,
    stations: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],
    target_time_offsets: typing.List[datetime.timedelta],
    user_config: typing.Dict[typing.AnyStr, typing.Any],# same as use functions in evaluation code, JSON format
    target_datetimes: typing.List[datetime.datetime] = None, # This is used for evaluation!! This is the taget dates we want. No need for training
    copy_last_if_missing: bool = True,
    train: bool = True,
):
    batch_size = user_config and user_config["batch_size"] or 32
    image_dim = user_config and user_config["image_dim"] or (64, 64)
    output_seq_len = user_config and user_config["output_seq_len"] or 4
    channels = user_config and user_config["target_channels"] or ["ch1", "ch2", "ch3", "ch4", "ch6"]

    def _train_dataset(hdf5_path):

        #for date_index, row in dataframe.loc["2011-12-1 08:00:00":"2011-12-2 08:00:00"].iterrows(): # TODO debug
        #for date_index, row in dataframe.iterrows():

            # Get image information
        #    hdf5_path = row['hdf5_8bit_path']
        if not hdf5_path == 'nan' and not hdf5_path == 'NaN' and not hdf5_path == 'NAN':
            
            with h5py.File(hdf5_path, "r") as h5_data:

                # get day time index from filename
                date = hdf5_path.split(b"/")[-1].split(b".")
                dataframe_day = dataframe.iloc[(dataframe.index.year == int(date[0])) & (dataframe.index.month == int(date[1])) & (dataframe.index.day == int(date[2]))]
                for date_index, row in dataframe_day.iterrows():
                    # get h5 meta info
                    global_start_idx = h5_data.attrs["global_dataframe_start_idx"]
                    global_end_idx = h5_data.attrs["global_dataframe_end_idx"]
                    h5_size = global_end_idx - global_start_idx
                    global_start_time = datetime.datetime.strptime(h5_data.attrs["global_dataframe_start_time"], "%Y.%m.%d.%H%M")
                    image_time_offset_idx = (date_index - global_start_time) / datetime.timedelta(minutes=15)

                    lats, lons = get_lats_lon(h5_data, h5_size)
                    if lats is None or lons is None:
                        continue
                    
                    # Return one station at a time
                    for station_idx, coords in stations.items():
                        # get station specefic data / meta we want
                        station_pixel_coords = (np.argmin(np.abs(lats - coords[0])), np.argmin(np.abs(lons - coords[1])))

                        # get meta info
                        lat, lont, alt = stations[station_idx] #lat, lont, alt
                        sin_month,cos_month,sin_minute,cos_minute = utils.convert_time(row.name) #encoding months and hour/minutes
                        daytime_flag, clearsky, _, __ = row.loc[row.index.str.startswith(station_idx)]
                        meta_array = np.array([sin_month,cos_month,sin_minute,cos_minute,
                                                lat, lont, alt, daytime_flag, clearsky])

                        # Get image data
                        image_data = get_image_transformed(h5_data, channels, image_time_offset_idx, station_pixel_coords, image_dim[0])
                        if image_data is None:
                            continue

                        # get station GHI targets
                        t_0 = row.name
                        station_ghis = []
                        for offset in target_time_offsets:
                            # remove negative value
                            station_ghis.append(round(max(dataframe.loc[t_0 + offset][station_idx + "_GHI"],0),2))
                        
                        yield (meta_array, image_data, station_ghis)
    # End of generator

    # ## debugging
    # next(_train_dataset())
    # ## debugging

    def wrap_generator(filename):
        return tf.data.Dataset.from_generator(_train_dataset, args=[filename], output_types=(tf.float64, tf.int8, tf.float64))
    
    debug = True
    if debug == True:
        dataframe = dataframe.loc["2011-12-1 08:00:00":"2011-12-2 08:00:00"]

    image_shape = (image_dim[0], image_dim[1], len(channels))
    image_files_to_process = dataframe[('hdf5_8bit_path')] [(dataframe['hdf5_8bit_path'].str.contains('nan|NAN|NaN') == False)]
    
    files = tf.data.Dataset.from_tensor_slices(image_files_to_process)
    dataset = files.interleave(wrap_generator, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    # data_loader = tf.data.Dataset.from_generator(
    #     _train_dataset, 
    #     output_types=(tf.float64, tf.int8, tf.float64),
    # )

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

        fast_data_frame_path = set_faster_path(data_frame_path, scratch_dir)

        data_frame = pd.read_pickle(fast_data_frame_path)

        # ## debug
        # data_frame = data_frame[data_frame.index >= datetime.datetime.fromisoformat('2010-05-01 08:00:00')]
        # ## debug

        target_time_offsets = [pd.Timedelta(d).to_pytimedelta() for d in admin_config["target_time_offsets"]]
            
        return BuildDataSet(data_frame, stations, target_time_offsets, user_config)
