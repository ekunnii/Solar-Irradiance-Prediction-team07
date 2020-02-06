"""
Objectif: from data path and t0, dataloader
"""
import tensorflow as tf
import pandas as pd
import time 
import typing
import os
import utils
from shutil import copyfile
import datetime

def set_faster_path(original_file, scratch_dir):
    if scratch_dir == None or scratch_dir == "":
        print(f"The dataframe left at it's original location: {original_file}")
        return original_file
    
    split = os.path.split(original_file)
    destination = scratch_dir + "/data/" + split[-1]
    if not os.path.exists(destination):
        copyfile(original_file, destination)
    print(f"The dataframe has been copied to: {destination}")
    return destination

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
    channels = user_config and user_config["target_channels"] or ["ch1", "ch2", "ch3", "ch4", "ch5"]

    def _train_dataset():

        #images_path = dataframe.ix[time_list]['hdf5_8bit_path']
        #fetch_and_crop_hdf5_imagery
        # Load options for training
        

        # Get hdf5_path
        
        #with h5py.File(hdf5_path, "r") as h5_data:

         #   for chanel in channels:
          #      pass
                #utils.utils.fetch_hdf5_sample(dataset_name: str,reader: h5py.File,sample_idx: int,)

        """
        Ryan's code goes here!! or we call his function
        """
        # Opening the file
        time.sleep(0.03)
        print(channels)
        
        for sample_idx in range(3):
            # Reading data (line, record) from the file
            time.sleep(0.015)
            
            random_meta = tf.random.uniform(shape=[batch_size, 1])
            random_image = tf.random.uniform(shape=[batch_size, 64,64,5])
            random_target = tf.random.uniform(shape=[batch_size, 4])
            yield (random_meta, random_image, random_target)

    
    
    image_shape = (image_dim[0], image_dim[1], len(channels))
    data_loader = tf.data.Dataset.from_generator(
        _train_dataset, 
        output_types=(tf.float64, tf.int64, tf.float64),
    )

    return data_loader


class TrainingDataSet(tf.data.Dataset):
    def __new__(
        cls, 
        data_frame_path: typing.AnyStr, 
        stations: typing.Dict[typing.AnyStr, typing.Tuple], 
        admin_config: typing.Dict[typing.AnyStr, typing.Any], # JSON; Training config file, looks like the admin config for the evaluation
        user_config: typing.Dict[typing.AnyStr, typing.Any] = None, # JSON; Model options or data loader options
        target_channels: typing.List[typing.AnyStr] = None,
        copy_last_if_missing: bool = True,
        train: bool = True,
        scratch_dir: str = None,
    ):

        fast_data_frame_path = set_faster_path(data_frame_path, scratch_dir)
        if target_channels == None:
            target_channels = ["ch1", "ch2", "ch3", "ch4", "ch5"]

        data_frame = pd.read_pickle(fast_data_frame_path)

        target_time_offsets = [pd.Timedelta(d).to_pytimedelta() for d in admin_config["target_time_offsets"]]
            
        return BuildDataSet(data_frame, stations, target_time_offsets, user_config)