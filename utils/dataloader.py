from __future__ import absolute_import, division, print_function, unicode_literals

# import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import utils
import h5py
import cv2 as cv
import typing
import datetime

np.set_printoptions(precision=4)


def data_generator_hdf5(
    dataframe: pd.DataFrame,
    target_datetimes: typing.List[datetime.datetime],
    stations: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],
    target_time_offsets: typing.List[datetime.timedelta],
    config: typing.Dict[typing.AnyStr, typing.Any],
):
    """
    Generate data from dataframe and h5 file.
    """
    batch_size = 32
    image_dim = (64, 64)
    n_channels = 5
    output_seq_len = 4

    for i in range(0, len(target_datetimes), batch_size):
        batch_of_datetimes = target_datetimes[i:i+batch_size]
        # This is evaluator, so there is timestamp for test time.
        # Which training, do we need to specify the datetimes? or we can get all data?

        samples = tf.random.uniform(shape=(
            len(batch_of_datetimes), image_dim[0], image_dim[1], n_channels
        ))
        targets = tf.zeros(shape=(
            len(batch_of_datetimes), output_seq_len
        ))
        # Remember that you do not have access to the targets.
        # Your dataloader should handle this accordingly.
        yield samples, targets



def load_training(
        data_dir: str,
        channels: typing.List[str],
        dataframe_path: typing.Optional[str] = None,
        stations: typing.Optional[typing.Dict[str, typing.Tuple]] = None,
        copy_last_if_missing: bool = True,
) -> None:
    """Displays a looping visualization of the imagery channels saved in an HDF5 file.
        Generate data from dataframe and h5 file.
    """

    # the final dimession is [batch, station, image_x, image_y, image_channels]
    # the final dimentsion of [batch, station, ghi_reading]

    assert os.path.exists(data_dir), f"invalid hdf5 path: {data_dir}"
    assert channels, "list of channels must not be empty"

    with h5py.File(hdf5_path, "r") as h5_data:
        global_start_idx = h5_data.attrs["global_dataframe_start_idx"]
        global_end_idx = h5_data.attrs["global_dataframe_end_idx"]
        archive_lut_size = global_end_idx - global_start_idx
        global_start_time = datetime.datetime.strptime(
            h5_data.attrs["global_dataframe_start_time"], "%Y.%m.%d.%H%M")
        lut_timestamps = [global_start_time + idx *
                          datetime.timedelta(minutes=15) for idx in range(archive_lut_size)]
        # will only display GHI values if dataframe is available
        stations_data = {}
        if stations:
            df = pd.read_pickle(dataframe_path) if dataframe_path else None

            # assume lats/lons stay identical throughout all frames; just pick the first available arrays
            idx, lats, lons = 0, None, None
            while (lats is None or lons is None) and idx < archive_lut_size:
                lats, lons = fetch_hdf5_sample(
                    "lat", h5_data, idx), fetch_hdf5_sample("lon", h5_data, idx)

            assert lats is not None and lons is not None, "could not fetch lats/lons arrays (hdf5 might be empty)"

            for reg, coords in tqdm.tqdm(stations.items(), desc="preparing stations data"):
                station_coords = (
                    np.argmin(np.abs(lats - coords[0])), np.argmin(np.abs(lons - coords[1])))
                station_data = {"coords": station_coords}
                if dataframe_path:
                    station_data["ghi"] = [
                        df.at[pd.Timestamp(t), reg + "_GHI"] for t in lut_timestamps]
                    station_data["csky"] = [
                        df.at[pd.Timestamp(t), reg + "_CLEARSKY_GHI"] for t in lut_timestamps]
                stations_data[reg] = station_data

        raw_data = np.zeros((archive_lut_size, len(channels), 650, 1500, 3), dtype=np.uint8)

        for channel_idx, channel_name in tqdm.tqdm(enumerate(channels), desc="preparing img data", total=len(channels)):
            assert channel_name in h5_data, f"missing channel: {channels}"
            norm_min = h5_data[channel_name].attrs.get("orig_min", None)
            norm_max = h5_data[channel_name].attrs.get("orig_max", None)
            channel_data = [fetch_hdf5_sample(
                channel_name, h5_data, idx) for idx in range(archive_lut_size)]
            assert all([array is None or array.shape == (650, 1500) for array in channel_data]), \
                "one of the saved channels had an expected dimension"
            last_valid_array_idx = None
            for array_idx, array in enumerate(channel_data):
                if array is None:
                    if copy_last_if_missing and last_valid_array_idx is not None:
                        raw_data[array_idx, channel_idx, :,
                                 :] = raw_data[last_valid_array_idx, channel_idx, :, :]
                    continue
                array = (((array.astype(np.float32) - norm_min) /
                          (norm_max - norm_min)) * 255).astype(np.uint8)
                array = cv.applyColorMap(array, cv.COLORMAP_BONE)
                for station_idx, (station_name, station) in enumerate(stations_data.items()):
                    station_color = get_label_color_mapping(
                        station_idx + 1).tolist()[::-1]
                    array = cv.circle(
                        array, station["coords"][::-1], radius=9, color=station_color, thickness=-1)
                raw_data[array_idx, channel_idx, :, :] = cv.flip(array, 0)
                last_valid_array_idx = array_idx
    plot_data = None
    if stations and dataframe_path:
        plot_data = preplot_live_ghi_curves(
            stations=stations, stations_data=stations_data,
            window_start=global_start_time,
            window_end=global_start_time + datetime.timedelta(hours=24),
            sample_step=datetime.timedelta(minutes=15),
            plot_title=global_start_time.strftime("GHI @ %Y-%m-%d"),
        )

        ################################### MODIFY ABOVE ##################################

        return data_loader

def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
    # This is a small dataset, only load it once, and keep it in memory.
    # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # fit in memory.
    if cache:
    if isinstance(cache, str):
        ds = ds.cache(cache)
    else:
        ds = ds.cache()

    # ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    # Repeat forever
    ds = ds.repeat()

    ds = ds.batch(BATCH_SIZE)

    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds




if __name__ == "__main__":
    # hdf5_path = "/project/cq-training-1/project1/data/16bit-2014.01.01.0800.h5"
    hdf5_path = "/home/ryan/data/16bit-2014.01.01.0800.h5"

    # this would correspond to: 2010.06.01.0800 + (32)*15min = 2010.06.01.1600
    hdf5_offset = 32
    with h5py.File(hdf5_path, "r") as h5_data:
        ch1_data = utils.fetch_hdf5_sample("ch1", h5_data, hdf5_offset)
    print(ch1_data.shape)

    target_channels = ["ch1", "ch2", "ch3", "ch4", "ch5"]
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

    data_dir = '/home/ryan/data/'
    load_training(data_dir, target_channels, dataframe_path, stations)
"""
            T_0 = lut_timestamps
            T_1 = T_0 + datetime.timedelta(hours=1)
            T_3 = T_0 + datetime.timedelta(hours=3)
            T_6 = T_0 + datetime.timedelta(hours=6)
            lut_timestamps = [T_0, T_1, T_3, T_6]

            stattion_data["ghi"] = [
                df.at[pd.Timestamp(t), reg + "_GHI"] for t in lut_timestamps]

            target = stattion_data["ghi"]
            yield sample, target_reading
