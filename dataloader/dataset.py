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
import tqdm as tqdm

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
        
        try:
            array_cropped = utils.crop(copy.deepcopy(raw_img), station_pixel_coords, cropped_img_size)
        except:
            return None
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
    debug = user_config and user_config["debug"] or False

    def _train_dataset(hdf5_path):
        # Load image file
        if not hdf5_path == 'nan' and not hdf5_path == 'NaN' and not hdf5_path == 'NAN':
            
            with h5py.File(hdf5_path, "r") as h5_data:

                # get day time index from filename, and itterate through all the day
                date = hdf5_path.split(b"/")[-1].split(b".")
                dataframe_day = dataframe.iloc[(dataframe.index.year == int(date[0])) & (dataframe.index.month == int(date[1])) & (dataframe.index.day == int(date[2]))]
                assert dataframe_day.shape, f"No dataframe for {date}"
                
                for date_index, row in dataframe_day.iterrows():
                    # get h5 meta info
                    global_start_idx = h5_data.attrs["global_dataframe_start_idx"]
                    global_end_idx = h5_data.attrs["global_dataframe_end_idx"]
                    h5_size = global_end_idx - global_start_idx
                    global_start_time = datetime.datetime.strptime(h5_data.attrs["global_dataframe_start_time"], "%Y.%m.%d.%H%M")
                    image_time_offset_idx = (date_index - global_start_time) / datetime.timedelta(minutes=15)
                    print("hdf5_path:", hdf5_path)

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
                            if debug:
                                print("No croped image")
                            continue

                        # get station GHI targets
                        t_0 = row.name
                        station_ghis = []
                        for offset in target_time_offsets:
                            # remove negative value
                            station_ghis.append(round(max(dataframe.loc[t_0 + offset][station_idx + "_GHI"],0),2))

                        if debug:
                            print(f"Returning data for {hdf5_path}")
                        yield (meta_array, image_data, station_ghis)

                #pdb.set_trace()
                if debug:
                    print(f"Not yielding any results! or done... {hdf5_path}")
                #raise StopIteration
                return
    # End of generator

    def wrap_generator(filename):
        return tf.data.Dataset.from_generator(_train_dataset, args=[filename], output_types=(tf.float64, tf.int8, tf.float64))
    
    if debug == True:
        dataframe = dataframe.loc["2010-01-1 08:00:00":"2010-04-30 07:45:00"] # single day data

    # first 3 months are empty, remove to iterate faster. 
    dataframe = dataframe.loc["2010-04-13 08:00:00":]
    
    # Only get dataloaders for image files that exist. 
    image_files_to_process = dataframe[('hdf5_8bit_path')] [(dataframe['hdf5_8bit_path'].str.contains('nan|NAN|NaN') == False)].unique()
    
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

        fast_data_frame_path = set_faster_path(data_frame_path, scratch_dir)

        data_frame = pd.read_pickle(fast_data_frame_path)

        if "start_bound" in admin_config:
            data_frame = data_frame[data_frame.index >= datetime.datetime.fromisoformat(
                admin_config["start_bound"])]
        if "end_bound" in admin_config:
            data_frame = data_frame[data_frame.index < datetime.datetime.fromisoformat(
                admin_config["end_bound"])]


        # ## debug
        # # data_frame = data_frame[data_frame.index >= datetime.datetime.fromisoformat('2010-05-01 08:00:00')]
        # data_frame = data_frame[data_frame.index >= datetime.datetime.fromisoformat('2010-05-01 13:45:00')]
        # # data_frame = data_frame[data_frame.index >= datetime.datetime.fromisoformat('2010-05-04 01:15:00')]
        # # print(data_frame["BND_GHI"].value_counts())
        # # print(data_frame["BND_GHI"].describe())
        # # print(data_frame["TBL_GHI"].isna().sum())
        # # t = ["BND","TBL","DRA","FPK","GWN","PSU", "SXF"]
        # # for e in t:
        # #     data_frame = data_frame.drop(t + "_GHI",1)
        # print(data_frame.iloc[:,5:][pd.isna(data_frame.iloc[:,5:]).any(axis=1)].head())
        # # print(np.isnan(data_frame.loc[datetime.datetime.fromisoformat('2010-05-01 14:15:00')]["PSU_GHI"]))
        # ## debug

        target_time_offsets = [pd.Timedelta(d).to_pytimedelta() for d in admin_config["target_time_offsets"]]
            
        return BuildDataSet(data_frame, stations, target_time_offsets, user_config)
