import os
from utils import utils
from shutil import copyfile
import typing
import datetime
import pandas as pd
import h5py
import pdb
import numpy as np
import copy

def valid_t0_row(row, station):
    if pd.isnull(row[f"{station}_GHI"]):
        #print("No GHI")
        return False
    if not row[f"{station}_DAYTIME"]:
        #print("Not Day")
        return False
    return True
    

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

def get_previous_day_image_data(original_path:str) -> h5py.File:
    path_array = os.path.split(original_path)

    date = path_array[1]
    if isinstance(date, (bytes, bytearray)):
        date = date.decode()

    path = path_array[0]
    if isinstance(path, (bytes, bytearray)):
        path = path.decode()

    new_date = datetime.datetime.strptime(date, '%Y.%m.%d.%H%M.h5') - datetime.timedelta(days=1)
    #print(new_date.strftime("%Y.%m.%d.0800.h5"))
    #print("Dates: ", global_date.strftime("%Y.%m.%d.0800.h5"), new_date.strftime("%Y.%m.%d.0800.h5"))
    previous_day_path = path + "/" + new_date.strftime("%Y.%m.%d.0800.h5")
    assert os.path.isfile(previous_day_path), f"Unable to open previous day image h5 file: {previous_day_path}"
    return h5py.File(previous_day_path, "r")

def get_image_time_offset_idx(current_date: datetime.datetime, global_start_time: datetime.datetime) -> int:
    return (current_date - global_start_time) / datetime.timedelta(minutes=15)

def get_global_start_date(image_data):
    return datetime.datetime.strptime(image_data.attrs["global_dataframe_start_time"], "%Y.%m.%d.%H%M")



def get_image_transformed(
    h5_data: h5py.File, 
    h5_data_previous_day: h5py.File,
    channels, 
    station_pixel_coords, 
    cropped_img_size: int, 
    current_date: datetime.datetime,
    with_pass_values = [], 
):

    def get_full_image(image_data):
        all_channels = np.empty([cropped_img_size, cropped_img_size, len(channels)])
        for ch_idx, channel in enumerate(channels):
            raw_img = utils.fetch_hdf5_sample(channel, image_data, image_time_offset_idx)
            if raw_img is None or raw_img.shape != (650, 1500):
                return None
            try:
                array_cropped = utils.crop(copy.deepcopy(raw_img), station_pixel_coords, cropped_img_size)
            except:
                return None
            # raw_data[array_idx, station_idx, channel_idx, ...] = cv.flip(array_cropped, 0) # TODO why the flip??

            #array = (((array.astype(np.float32) - norm_min) / (norm_max - norm_min)) * 255).astype(np.uint8) # TODO norm?
            array_cropped = array_cropped.astype(np.float64) # convert to image format
            all_channels[:,:,ch_idx] = array_cropped

        return all_channels

    # TODO Normalize images?
    previous_valid_image = None #np.empty([len(channels)])
    time_steps = copy.deepcopy(with_pass_values) or []
    time_steps.insert(0,"0")# Time 0 image at least

    image_data = h5_data
    global_start_time = get_global_start_date(image_data)
    image_time_offset_idx = get_image_time_offset_idx(current_date, global_start_time)

    all_images = np.empty([len(time_steps), cropped_img_size, cropped_img_size, len(channels)])
    for img_idx, image_time_offset in enumerate(time_steps):
        # make sure the image is not on the previous day image file
        img_delta = pd.Timedelta(image_time_offset).to_pytimedelta()
        #print(f"1: {str(img_delta)}, {str(current_date)}, {str(global_start_time)}, {bool(current_date - img_delta < global_start_time)}")

        if (current_date - img_delta < global_start_time):
            image_data = h5_data_previous_day 
            #print(f"Getting previous day! {image_data}, old file: {h5_data}")

        # Prep images
        all_channels = get_full_image(image_data)
        
        if all_channels is not None:
            all_images[img_idx] = all_channels
            previous_valid_image = all_channels
            #print("Found first image")
        elif previous_valid_image is not None:
            all_images[img_idx] = previous_valid_image
            #print("Unable to find other image, copy old")
        else:
            #print("No Image!!")
            return None

    #print(np.array(all_images).shape)
    if len(all_images) == 1:
        #print("returning image")
        return all_images[0]
    #print("returning more then one image")
    return all_images
