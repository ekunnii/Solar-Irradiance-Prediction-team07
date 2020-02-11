from __future__ import absolute_import, division, print_function, unicode_literals

# import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import utils
import h5py


np.set_printoptions(precision=4)

if __name__ == "__main__":
    print("Visualization program executing.")
    # hdf5_path = "/project/cq-training-1/project1/data/16bit-2014.01.01.0800.h5"
    hdf5_path = "/home/ryan/data/16bit-2014.01.01.0800.h5"

    # this would correspond to: 2010.06.01.0800 + (32)*15min = 2010.06.01.1600
    hdf5_offset = 32
    with h5py.File(hdf5_path, "r") as h5_data:
        ch1_data = utils.fetch_hdf5_sample("ch1", h5_data, hdf5_offset)
    print(ch1_data.shape)

    target_channels = ["ch1", "ch2", "ch3", "ch4", "ch6"]
    dataframe_path = "/home/ryan/data/catalog.helios.public.20100101-20160101.pkl"
    stations = {
        "BND": [40.05192, -88.37309, 230],
        "TBL": [40.12498, -105.23680, 1689],
        "DRA": [36.62373, -116.01947, 1007],
        "FPK": [48.30783, -105.10170, 634],
        "GWN": [34.25470, -89.87290, 98],
        "PSU": [40.72012, -77.93085, 376],
        "SXF": [43.73403, -96.62328, 473]
    }
    utils.viz_hdf5_imagery(hdf5_path, target_channels,
                           dataframe_path, stations)
