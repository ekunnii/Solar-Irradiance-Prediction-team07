"""
Objectif: from data path and t0, dataloader
"""
import tensorflow as tf
import time 
import typing
import os
from shutil import copyfile

def set_faster_path(original_file, scratch_dir):
    if scratch_dir == None or scratch_dir == "":
        print(f"The dataframe left at it's original location: {original_file}")
        return original_file

    split = os.path.split(original_file)
    destination = scratch_dir + "/data/" + split[-1]
    copyfile(original_file, destination)
    print(f"The dataframe has been copied to: {destination}")

class SolarDataset(tf.data.Dataset):
    def _dummy_generator(
        data_frame_path: str, 
        channels: typing.List[str],
        stations: typing.Dict[str, typing.Tuple],
        copy_last_if_missing: bool = True,
        data_loader_options: typing.Dict[str, typing.Any] = None,
    ):
        """
        Ryan's code goes here!! or we call his function
        """
        # Opening the file
        time.sleep(0.03)
        
        for sample_idx in range(3):
            # Reading data (line, record) from the file
            time.sleep(0.015)
            
            yield (sample_idx,)


    
    def __new__(
        cls, 
        data_frame_path: str, 
        stations: typing.Dict[str, typing.Tuple], 
        target_channels: typing.List[str] = None,
        copy_last_if_missing: bool = True,
        data_loader_options: typing.Dict[str, typing.Any] = None, # JSON
        scratch_dir: str = None,
    ):
        """
        When creating a new generator, we want to make sure every thing is at the right place before 
        we start to provide data. This means, if we are working on the cluster, we want to copy the big files
        on $SCRATCH, but not the images (too mutch data). We would like to leverage the "prefetch" option from tf.dataset. 

        The tings we want to do once, we do here. 
        """
        fast_data_frame_path = set_faster_path(data_frame_path, scratch_dir)
        if target_channels == None:
            target_channels = ["ch1", "ch2", "ch3", "ch4", "ch5"]

        
        return tf.data.Dataset.from_generator(
            cls._dummy_generator,
            output_types=tf.dtypes.int64,
            output_shapes=(1,), # TODO fix output shape...
            args=(fast_data_frame_path, target_channels, stations, copy_last_if_missing, data_loader_options)
        )