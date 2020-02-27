import datetime
import json
import math
import os
import typing
import warnings
import copy

import cv2 as cv
import h5py
import lz4.frame
import matplotlib.dates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tqdm
from timezonefinder import TimezoneFinder
import datetime
import pytz

def get_label_color_mapping(idx):
    """Returns the PASCAL VOC color triplet for a given label index."""

    # https://gist.github.com/wllhf/a4533e0adebe57e3ed06d4b50c8419ae
    def bitget(byteval, ch):
        return (byteval & (1 << ch)) != 0

    r = g = b = 0
    for j in range(8):
        r = r | (bitget(idx, 0) << 7 - j)
        g = g | (bitget(idx, 1) << 7 - j)
        b = b | (bitget(idx, 2) << 7 - j)
        idx = idx >> 3
    return np.array([r, g, b], dtype=np.uint8)


def get_label_html_color_code(idx):
    """Returns the PASCAL VOC HTML color code for a given label index."""
    color_array = get_label_color_mapping(idx)
    return f"#{color_array[0]:02X}{color_array[1]:02X}{color_array[2]:02X}"


def fig2array(fig):
    """Transforms a pyplot figure into a numpy-compatible BGR array.
    The reason why we flip the channel order (RGB->BGR) is for OpenCV compatibility. Feel free to
    edit this function if you wish to use it with another display library.
    """
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf.shape = (h, w, 3)
    return buf[..., ::-1]


def compress_array(
        array: np.ndarray,
        compr_type: typing.Optional[str] = "auto",
) -> bytes:
    """Compresses the provided numpy array according to a predetermined strategy.
    If ``compr_type`` is 'auto', the best strategy will be automatically selected based on the input
    array type. If ``compr_type`` is an empty string (or ``None``), no compression will be applied.
    """
    assert compr_type is None or compr_type in ["lz4", "float16+lz4", "uint8+jpg",
                                                "uint8+jp2", "uint16+jp2", "auto", ""], \
        f"unrecognized compression strategy '{compr_type}'"
    if compr_type is None or not compr_type:
        return array.tobytes()
    if compr_type == "lz4":
        return lz4.frame.compress(array.tobytes())
    if compr_type == "float16+lz4":
        assert np.issubdtype(array.dtype, np.floating), "no reason to cast to float16 is not float32/64"
        return lz4.frame.compress(array.astype(np.float16).tobytes())
    if compr_type == "uint8+jpg":
        assert array.ndim == 2 or (array.ndim == 3 and (array.shape[2] == 1 or array.shape[2] == 3)), \
            "jpg compression via tensorflow requires 2D or 3D image with 1/3 channels in last dim"
        if array.ndim == 2:
            array = np.expand_dims(array, axis=2)
        assert array.dtype == np.uint8, "jpg compression requires uint8 array"
        return tf.io.encode_jpeg(array).numpy()
    if compr_type == "uint8+jp2" or compr_type == "uint16+jp2":
        assert array.ndim == 2 or (array.ndim == 3 and (array.shape[2] == 1 or array.shape[2] == 3)), \
            "jp2 compression via opencv requires 2D or 3D image with 1/3 channels in last dim"
        if array.ndim == 2:
            array = np.expand_dims(array, axis=2)
        assert array.dtype == np.uint8 or array.dtype == np.uint16, "jp2 compression requires uint8/16 array"
        if os.getenv("OPENCV_IO_ENABLE_JASPER") is None:
            # for local/trusted use only; see issue here: https://github.com/opencv/opencv/issues/14058
            os.environ["OPENCV_IO_ENABLE_JASPER"] = "1"
        retval, buffer = cv.imencode(".jp2", array)
        assert retval, "JPEG2000 encoding failed"
        return buffer.tobytes()
    # could also add uint16 png/tiff via opencv...
    if compr_type == "auto":
        # we cheat for auto-decompression by prefixing the strategy in the bytecode
        if array.ndim == 2 or (array.ndim == 3 and (array.shape[2] == 1 or array.shape[2] == 3)):
            if array.dtype == np.uint8:
                return b"uint8+jpg" + compress_array(array, compr_type="uint8+jpg")
            if array.dtype == np.uint16:
                return b"uint16+jp2" + compress_array(array, compr_type="uint16+jp2")
        return b"lz4" + compress_array(array, compr_type="lz4")


def decompress_array(
        buffer: typing.Union[bytes, np.ndarray],
        compr_type: typing.Optional[str] = "auto",
        dtype: typing.Optional[typing.Any] = None,
        shape: typing.Optional[typing.Union[typing.List, typing.Tuple]] = None,
) -> np.ndarray:
    """Decompresses the provided numpy array according to a predetermined strategy.
    If ``compr_type`` is 'auto', the correct strategy will be automatically selected based on the array's
    bytecode prefix. If ``compr_type`` is an empty string (or ``None``), no decompression will be applied.
    This function can optionally convert and reshape the decompressed array, if needed.
    """
    compr_types = ["lz4", "float16+lz4", "uint8+jpg", "uint8+jp2", "uint16+jp2"]
    assert compr_type is None or compr_type in compr_types or compr_type in ["", "auto"], \
        f"unrecognized compression strategy '{compr_type}'"
    assert isinstance(buffer, bytes) or buffer.dtype == np.uint8, "invalid raw data buffer type"
    if isinstance(buffer, np.ndarray):
        buffer = buffer.tobytes()
    if compr_type == "lz4" or compr_type == "float16+lz4":
        buffer = lz4.frame.decompress(buffer)
    if compr_type == "uint8+jpg":
        # tf.io.decode_jpeg often segfaults when initializing parallel pipelines, let's avoid it...
        # buffer = tf.io.decode_jpeg(buffer).numpy()
        buffer = cv.imdecode(np.frombuffer(buffer, dtype=np.uint8), flags=cv.IMREAD_UNCHANGED)
    if compr_type.endswith("+jp2"):
        if os.getenv("OPENCV_IO_ENABLE_JASPER") is None:
            # for local/trusted use only; see issue here: https://github.com/opencv/opencv/issues/14058
            os.environ["OPENCV_IO_ENABLE_JASPER"] = "1"
        buffer = cv.imdecode(np.frombuffer(buffer, dtype=np.uint8), flags=cv.IMREAD_UNCHANGED)
    if compr_type == "auto":
        decompr_buffer = None
        for compr_code in compr_types:
            if buffer.startswith(compr_code.encode("ascii")):
                decompr_buffer = decompress_array(buffer[len(compr_code):], compr_type=compr_code,
                                                  dtype=dtype, shape=shape)
                break
        assert decompr_buffer is not None, "missing auto-decompression code in buffer"
        buffer = decompr_buffer
    array = np.frombuffer(buffer, dtype=dtype)
    if shape is not None:
        array = array.reshape(shape)
    return array


def fetch_hdf5_sample(
        dataset_name: str,
        reader: h5py.File,
        sample_idx: int,
) -> typing.Any:
    """Decodes and returns a single sample from an HDF5 dataset.

    Args:
        dataset_name: name of the HDF5 dataset to fetch the sample from using the reader. In the context of
            the GHI prediction project, this may be for example an imagery channel name (e.g. "ch1").
        reader: an HDF5 archive reader obtained via ``h5py.File(...)`` which can be used for dataset indexing.
        sample_idx: the integer index (or offset) that corresponds to the position of the sample in the dataset.

    Returns:
        The sample. This function will automatically decompress the sample if it was compressed. It the sample is
        unavailable because the input was originally masked, the function will return ``None``. The sample itself
        may be a scalar or a numpy array.
    """
    dataset_lut_name = dataset_name + "_LUT"
    if dataset_lut_name in reader:
        sample_idx = reader[dataset_lut_name][sample_idx]
        if sample_idx == -1:
            return None  # unavailable
    dataset = reader[dataset_name]
    if "compr_type" not in dataset.attrs:
        # must have been compressed directly (or as a scalar); return raw output
        return dataset[sample_idx]
    compr_type, orig_dtype, orig_shape = dataset.attrs["compr_type"], None, None
    if "orig_dtype" in dataset.attrs:
        orig_dtype = dataset.attrs["orig_dtype"]
    if "orig_shape" in dataset.attrs:
        orig_shape = dataset.attrs["orig_shape"]
    if "force_cvt_uint8" in dataset.attrs and dataset.attrs["force_cvt_uint8"]:
        array = decompress_array(dataset[sample_idx], compr_type=compr_type, dtype=np.uint8, shape=orig_shape)
        orig_min, orig_max = dataset.attrs["orig_min"], dataset.attrs["orig_max"]
        array = ((array.astype(np.float32) / 255) * (orig_max - orig_min) + orig_min).astype(orig_dtype)
    elif "force_cvt_uint16" in dataset.attrs and dataset.attrs["force_cvt_uint16"]:
        array = decompress_array(dataset[sample_idx], compr_type=compr_type, dtype=np.uint16, shape=orig_shape)
        orig_min, orig_max = dataset.attrs["orig_min"], dataset.attrs["orig_max"]
        array = ((array.astype(np.float32) / 65535) * (orig_max - orig_min) + orig_min).astype(orig_dtype)
    else:
        array = decompress_array(dataset[sample_idx], compr_type=compr_type, dtype=orig_dtype, shape=orig_shape)
    return array


def print_attrs(name, obj):
    print(name)
    for key, val in obj.attrs.items():
        print("    %s: %s" % (key, val))


def viz_hdf5_imagery(
        hdf5_path: str,
        channels: typing.List[str],
        dataframe_path: typing.Optional[str] = None,
        stations: typing.Optional[typing.Dict[str, typing.Tuple]] = None,
        copy_last_if_missing: bool = True,
) -> None:
    """Displays a looping visualization of the imagery channels saved in an HDF5 file.
    This visualization requires OpenCV3+ ('cv2'), and will loop while refreshing a local window until the program
    is killed, or 'q' is pressed. The visualization can also be paused by pressing the space bar.
    """
    assert os.path.isfile(hdf5_path), f"invalid hdf5 path: {hdf5_path}"
    assert channels, "list of channels must not be empty"
    # Parsing h5py data
    with h5py.File(hdf5_path, "r") as h5_data:
        h5_data.visititems(print_attrs)
        # h5 data contains the start and end index to offset which image to get in the array
        global_start_idx = h5_data.attrs["global_dataframe_start_idx"]
        global_end_idx = h5_data.attrs["global_dataframe_end_idx"]
        archive_lut_size = global_end_idx - global_start_idx  # Number of images in the h5 file
        global_start_time = datetime.datetime.strptime(h5_data.attrs["global_dataframe_start_time"],
                                                       "%Y.%m.%d.%H%M")  # The global start time is from 2011; first image date time
        lut_timestamps = [global_start_time + idx * datetime.timedelta(minutes=15) for idx in range(archive_lut_size)]
        # will only display GHI values if dataframe is available
        # Read and extract station position in images
        stations_data = {}
        if stations:
            df = pd.read_pickle(dataframe_path) if dataframe_path else None
            # assume lats/lons stay identical throughout all frames; just pick the first available arrays
            idx, lats, lons = 0, None, None
            while (lats is None or lons is None) and idx < archive_lut_size:
                lats, lons = fetch_hdf5_sample("lat", h5_data, idx), fetch_hdf5_sample("lon", h5_data, idx)
            assert lats is not None and lons is not None, "could not fetch lats/lons arrays (hdf5 might be empty)"
            for reg, coords in tqdm.tqdm(stations.items(), desc="preparing stations data"):
                station_coords = (np.argmin(np.abs(lats - coords[0])), np.argmin(np.abs(lons - coords[1])))
                station_data = {"coords": station_coords}
                if dataframe_path:
                    station_data["ghi"] = [df.at[pd.Timestamp(t), reg + "_GHI"] for t in lut_timestamps]
                    station_data["csky"] = [df.at[pd.Timestamp(t), reg + "_CLEARSKY_GHI"] for t in lut_timestamps]
                stations_data[reg] = station_data
        raw_data = np.zeros((archive_lut_size, len(channels), 650, 1500, 3), dtype=np.uint8)

        # Get chanel image information and store it in raw_data after transformations
        for channel_idx, channel_name in tqdm.tqdm(enumerate(channels), desc="preparing img data", total=len(channels)):
            assert channel_name in h5_data, f"missing channel: {channels}"
            norm_min = h5_data[channel_name].attrs.get("orig_min", None)
            norm_max = h5_data[channel_name].attrs.get("orig_max", None)
            channel_data = [fetch_hdf5_sample(channel_name, h5_data, idx) for idx in range(archive_lut_size)]
            assert all([array is None or array.shape == (650, 1500) for array in channel_data]), \
                "one of the saved channels had an expected dimension"
            last_valid_array_idx = None
            for array_idx, array in enumerate(channel_data):
                # If missing, copy previous valid one.
                if array is None:
                    if copy_last_if_missing and last_valid_array_idx is not None:
                        raw_data[array_idx, channel_idx, :, :] = raw_data[last_valid_array_idx, channel_idx, :, :]
                    continue
                # Otherwise, get image data
                array = (((array.astype(np.float32) - norm_min) / (norm_max - norm_min)) * 255).astype(np.uint8)
                array = cv.applyColorMap(array, cv.COLORMAP_BONE)
                for station_idx, (station_name, station) in enumerate(stations_data.items()):
                    station_color = get_label_color_mapping(station_idx + 1).tolist()[::-1]
                    array = cv.circle(array, station["coords"][::-1], radius=9, color=station_color, thickness=-1)
                raw_data[array_idx, channel_idx, :, :] = cv.flip(array, 0)
                last_valid_array_idx = array_idx
    # This section is about the ghi value plotting, less interesting for imgage extraction
    plot_data = None
    if stations and dataframe_path:
        plot_data = preplot_live_ghi_curves(
            stations=stations, stations_data=stations_data,
            window_start=global_start_time,
            window_end=global_start_time + datetime.timedelta(hours=24),
            sample_step=datetime.timedelta(minutes=15),
            plot_title=global_start_time.strftime("GHI @ %Y-%m-%d"),
        )
        assert plot_data.shape[0] == archive_lut_size
    display_data = []
    for array_idx in tqdm.tqdm(range(archive_lut_size), desc="reshaping for final display"):
        display = cv.vconcat([raw_data[array_idx, ch_idx, ...] for ch_idx in range(len(channels))])
        while any([s > 1200 for s in display.shape]):
            display = cv.resize(display, (-1, -1), fx=0.75, fy=0.75)
        if plot_data is not None:
            plot = plot_data[array_idx]
            plot_scale = display.shape[0] / plot.shape[0]
            plot = cv.resize(plot, (-1, -1), fx=plot_scale, fy=plot_scale)
            display = cv.hconcat([display, plot])
        display_data.append(display)
    display = np.stack(display_data)
    array_idx, window_name, paused = 0, hdf5_path.split("/")[-1], False
    # Infinite loop to display images and plots in window.
    while True:
        cv.imshow(window_name, display[array_idx])
        ret = cv.waitKey(30 if not paused else 300)
        if ret == ord('q') or ret == 27:  # q or ESC
            break
        elif ret == ord(' '):
            paused = ~paused
        if not paused or ret == ord('c'):
            array_idx = (array_idx + 1) % archive_lut_size


def preplot_live_ghi_curves(
        stations: typing.Dict[str, typing.Tuple],
        stations_data: typing.Dict[str, typing.Any],
        window_start: datetime.datetime,
        window_end: datetime.datetime,
        sample_step: datetime.timedelta,
        plot_title: typing.Optional[typing.AnyStr] = None,
) -> np.ndarray:
    """Pre-plots a set of GHI curves with update bars and returns the raw pixel arrays.
    This function is used in ``viz_hdf5_imagery`` to prepare GHI plots when stations & dataframe information
    is available.
    """
    plot_count = (window_end - window_start) // sample_step
    fig_size, fig_dpi, plot_row_count = (8, 6), 160, int(math.ceil(len(stations) / 2))
    plot_data = np.zeros((plot_count, fig_size[0] * fig_dpi, fig_size[1] * fig_dpi, 3), dtype=np.uint8)
    fig = plt.figure(num="ghi", figsize=fig_size[::-1], dpi=fig_dpi, facecolor="w", edgecolor="k")
    ax = fig.subplots(nrows=plot_row_count, ncols=2, sharex="all", sharey="all")
    art_handles, art_labels = [], []
    for station_idx, station_name in enumerate(stations):
        plot_row_idx, plot_col_idx = station_idx // 2, station_idx % 2
        ax[plot_row_idx, plot_col_idx] = plot_ghi_curves(
            clearsky_ghi=np.asarray(stations_data[station_name]["csky"]),
            station_ghi=np.asarray(stations_data[station_name]["ghi"]),
            pred_ghi=None,
            window_start=window_start,
            window_end=window_end - sample_step,
            sample_step=sample_step,
            horiz_offset=datetime.timedelta(hours=0),
            ax=ax[plot_row_idx, plot_col_idx],
            station_name=station_name,
            station_color=get_label_html_color_code(station_idx + 1),
            current_time=window_start
        )
        for handle, lbl in zip(*ax[plot_row_idx, plot_col_idx].get_legend_handles_labels()):
            # skipping over the duplicate labels messes up the legend, we must live with the warning
            art_labels.append("_" + lbl if lbl in art_labels or lbl == "current" else lbl)
            art_handles.append(handle)
    fig.autofmt_xdate()
    if plot_title is not None:
        fig.suptitle(plot_title, fontsize=14)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig.legend(art_handles, labels=art_labels, loc="lower center", ncol=2)
    fig.canvas.draw()  # cache renderer with default call first
    subaxbgs = [fig.canvas.copy_from_bbox(subax.bbox) for subax in ax.flatten()]
    for idx in tqdm.tqdm(range(plot_count), desc="preparing ghi plots"):
        for subax, subaxbg in zip(ax.flatten(), subaxbgs):
            fig.canvas.restore_region(subaxbg)
            for handle, lbl in zip(*subax.get_legend_handles_labels()):
                if lbl == "current":
                    curr_time = matplotlib.dates.date2num(window_start + idx * sample_step)
                    handle.set_data([curr_time, curr_time], [0, 1])
                    subax.draw_artist(handle)
            fig.canvas.blit(subax.bbox)
        plot_data[idx, ...] = np.reshape(np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8),
                                         (*(fig.canvas.get_width_height()[::-1]), 3))[..., ::-1]
    return plot_data


def plot_ghi_curves(
        clearsky_ghi: np.ndarray,
        station_ghi: np.ndarray,
        pred_ghi: typing.Optional[np.ndarray],
        window_start: datetime.datetime,
        window_end: datetime.datetime,
        sample_step: datetime.timedelta,
        horiz_offset: datetime.timedelta,
        ax: plt.Axes,
        station_name: typing.Optional[typing.AnyStr] = None,
        station_color: typing.Optional[typing.AnyStr] = None,
        current_time: typing.Optional[datetime.datetime] = None,
) -> plt.Axes:
    """Plots a set of GHI curves and returns the associated matplotlib axes object.
    This function is used in ``draw_daily_ghi`` and ``preplot_live_ghi_curves`` to create simple
    graphs of GHI curves (clearsky, measured, predicted).
    """
    assert clearsky_ghi.ndim == 1 and station_ghi.ndim == 1 and clearsky_ghi.size == station_ghi.size
    assert pred_ghi is None or (pred_ghi.ndim == 1 and clearsky_ghi.size == pred_ghi.size)
    hour_tick_locator = matplotlib.dates.HourLocator(interval=4)
    minute_tick_locator = matplotlib.dates.HourLocator(interval=1)
    datetime_fmt = matplotlib.dates.DateFormatter("%H:%M")
    datetime_range = pd.date_range(window_start, window_end, freq=sample_step)
    xrange_real = matplotlib.dates.date2num([d.to_pydatetime() for d in datetime_range])
    if current_time is not None:
        ax.axvline(x=matplotlib.dates.date2num(current_time), color="r", label="current")
    station_name = f"measured ({station_name})" if station_name else "measured"
    ax.plot(xrange_real, clearsky_ghi, ":", label="clearsky")
    if station_color is not None:
        ax.plot(xrange_real, station_ghi, linestyle="solid", color=station_color, label=station_name)
    else:
        ax.plot(xrange_real, station_ghi, linestyle="solid", label=station_name)
    datetime_range = pd.date_range(window_start + horiz_offset, window_end + horiz_offset, freq=sample_step)
    xrange_offset = matplotlib.dates.date2num([d.to_pydatetime() for d in datetime_range])
    if pred_ghi is not None:
        ax.plot(xrange_offset, pred_ghi, ".-", label="predicted")
    ax.xaxis.set_major_locator(hour_tick_locator)
    ax.xaxis.set_major_formatter(datetime_fmt)
    ax.xaxis.set_minor_locator(minute_tick_locator)
    hour_offset = datetime.timedelta(hours=1) // sample_step
    ax.set_xlim(xrange_real[hour_offset - 1], xrange_real[-hour_offset + 1])
    ax.format_xdata = matplotlib.dates.DateFormatter("%Y-%m-%d %H:%M")
    ax.grid(True)
    return ax


def draw_daily_ghi(
        clearsky_ghi: np.ndarray,
        station_ghi: np.ndarray,
        pred_ghi: np.ndarray,
        stations: typing.Iterable[typing.AnyStr],
        horiz_deltas: typing.List[datetime.timedelta],
        window_start: datetime.datetime,
        window_end: datetime.datetime,
        sample_step: datetime.timedelta,
):
    """Draws a set of 2D GHI curve plots and returns the associated matplotlib fig/axes objects.
    This function is used in ``viz_predictions`` to prepare the full-horizon, multi-station graphs of
    GHI values over numerous days.
    """
    assert clearsky_ghi.ndim == 2 and station_ghi.ndim == 2 and clearsky_ghi.shape == station_ghi.shape
    station_count = len(list(stations))
    sample_count = station_ghi.shape[1]
    assert clearsky_ghi.shape[0] == station_count and station_ghi.shape[0] == station_count
    assert pred_ghi.ndim == 3 and pred_ghi.shape[0] == station_count and pred_ghi.shape[2] == sample_count
    assert len(list(horiz_deltas)) == pred_ghi.shape[1]
    pred_horiz = pred_ghi.shape[1]
    fig = plt.figure(num="ghi", figsize=(18, 10), dpi=80, facecolor="w", edgecolor="k")
    fig.clf()
    ax = fig.subplots(nrows=pred_horiz, ncols=station_count, sharex="all", sharey="all")
    handles, labels = None, None
    for horiz_idx in range(pred_horiz):
        for station_idx, station_name in enumerate(stations):
            ax[horiz_idx, station_idx] = plot_ghi_curves(
                clearsky_ghi=clearsky_ghi[station_idx],
                station_ghi=station_ghi[station_idx],
                pred_ghi=pred_ghi[station_idx, horiz_idx],
                window_start=window_start,
                window_end=window_end,
                sample_step=sample_step,
                horiz_offset=horiz_deltas[horiz_idx],
                ax=ax[horiz_idx, station_idx],
            )
            handles, labels = ax[horiz_idx, station_idx].get_legend_handles_labels()
    for station_idx, station_name in enumerate(stations):
        ax[0, station_idx].set_title(station_name)
    for horiz_idx, horiz_delta in zip(range(pred_horiz), horiz_deltas):
        ax[horiz_idx, 0].set_ylabel(f"GHI @ T+{horiz_delta}")
    window_center = window_start + (window_end - window_start) / 2
    fig.autofmt_xdate()
    fig.suptitle(window_center.strftime("%Y-%m-%d"), fontsize=14)
    fig.legend(handles, labels, loc="lower center")
    return fig2array(fig)


def viz_predictions(
        predictions_path: typing.AnyStr,
        dataframe_path: typing.AnyStr,
        test_config_path: typing.AnyStr,
):
    """Displays a looping visualization of the GHI predictions saved by the evaluation script.
    This visualization requires OpenCV3+ ('cv2'), and will loop while refreshing a local window until the program
    is killed, or 'q' is pressed. The arrow keys allow the user to change which day is being shown.
    """
    assert os.path.isfile(test_config_path) and test_config_path.endswith(".json"), "invalid test config"
    with open(test_config_path, "r") as fd:
        test_config = json.load(fd)
    stations = test_config["stations"]
    target_datetimes = test_config["target_datetimes"]
    start_bound = datetime.datetime.fromisoformat(test_config["start_bound"])
    end_bound = datetime.datetime.fromisoformat(test_config["end_bound"])
    horiz_deltas = [pd.Timedelta(d).to_pytimedelta() for d in test_config["target_time_offsets"]]
    assert os.path.isfile(predictions_path), f"invalid preds file path: {predictions_path}"
    with open(predictions_path, "r") as fd:
        predictions = fd.readlines()
    assert len(predictions) == len(target_datetimes) * len(stations), \
        "predicted ghi sequence count mistmatch wrt target datetimes x station count"
    assert len(predictions) % len(stations) == 0
    predictions = np.asarray([float(ghi) for p in predictions for ghi in p.split(",")])
    predictions = predictions.reshape((len(stations), len(target_datetimes), -1))
    pred_horiz = predictions.shape[-1]
    target_datetimes = pd.DatetimeIndex([datetime.datetime.fromisoformat(t) for t in target_datetimes])
    assert os.path.isfile(dataframe_path), f"invalid dataframe path: {dataframe_path}"
    dataframe = pd.read_pickle(dataframe_path)
    dataframe = dataframe[dataframe.index >= start_bound]
    dataframe = dataframe[dataframe.index < end_bound]
    assert dataframe.index.get_loc(start_bound) == 0, "invalid start bound (should land at first index)"
    assert len(dataframe.index.intersection(target_datetimes)) == len(target_datetimes), \
        "bad dataframe target datetimes overlap, index values are missing"
    # we will display 24-hour slices with some overlap (configured via hard-coded param below)
    time_window, time_overlap, time_sample = \
        datetime.timedelta(hours=24), datetime.timedelta(hours=3), datetime.timedelta(minutes=15)
    assert len(dataframe.asfreq("15min").index) == len(dataframe.index), \
        "invalid dataframe index padding (should have an entry every 15 mins)"
    sample_count = ((time_window + 2 * time_overlap) // time_sample) + 1
    day_count = int(math.ceil((end_bound - start_bound) / time_window))
    clearsky_ghi_data = np.full((day_count, len(stations), sample_count), fill_value=float("nan"), dtype=np.float32)
    station_ghi_data = np.full((day_count, len(stations), sample_count), fill_value=float("nan"), dtype=np.float32)
    pred_ghi_data = np.full((day_count, len(stations), pred_horiz, sample_count), fill_value=float("nan"),
                            dtype=np.float32)
    days_range = pd.date_range(start_bound, end_bound, freq=time_window, closed="left")
    for day_idx, day_start in enumerate(tqdm.tqdm(days_range, desc="preparing daytime GHI intervals")):
        window_start, window_end = day_start - time_overlap, day_start + time_window + time_overlap
        sample_start, sample_end = (window_start - start_bound) // time_sample, (
                    window_end - start_bound) // time_sample
        for sample_iter_idx, sample_idx in enumerate(range(sample_start, sample_end + 1)):
            if sample_idx < 0 or sample_idx >= len(dataframe.index):
                continue
            sample_row = dataframe.iloc[sample_idx]
            sample_time = window_start + sample_iter_idx * time_sample
            target_iter_idx = target_datetimes.get_loc(sample_time) if sample_time in target_datetimes else None
            for station_idx, station_name in enumerate(stations):
                clearsky_ghi_data[day_idx, station_idx, sample_iter_idx] = sample_row[station_name + "_CLEARSKY_GHI"]
                station_ghi_data[day_idx, station_idx, sample_iter_idx] = sample_row[station_name + "_GHI"]
                if target_iter_idx is not None:
                    pred_ghi_data[day_idx, station_idx, :, sample_iter_idx] = predictions[station_idx, target_iter_idx]
    displays = []
    for day_idx, day_start in enumerate(tqdm.tqdm(days_range, desc="preparing plots")):
        displays.append(draw_daily_ghi(
            clearsky_ghi=clearsky_ghi_data[day_idx],
            station_ghi=station_ghi_data[day_idx],
            pred_ghi=pred_ghi_data[day_idx],
            stations=stations,
            horiz_deltas=horiz_deltas,
            window_start=(day_start - time_overlap),
            window_end=(day_start + time_window + time_overlap),
            sample_step=time_sample,
        ))
    display = np.stack(displays)
    day_idx = 0
    while True:
        cv.imshow("ghi", display[day_idx])
        ret = cv.waitKey(100)
        if ret == ord('q') or ret == 27:  # q or ESC
            break
        elif ret == 81 or ret == 84:  # UNIX: left or down arrow
            day_idx = max(day_idx - 1, 0)
        elif ret == 82 or ret == 83:  # UNIX: right or up arrow
            day_idx = min(day_idx + 1, len(displays) - 1)


def fetch_and_crop_hdf5_imagery(
        hdf5_path: str,
        channels: typing.List[str],
        stations: typing.Optional[typing.Dict[str, typing.Tuple]],
        dataframe_path: typing.Optional[str] = None,
        cropped_img_size: typing.Optional[int] = None,
        copy_last_if_missing: bool = True,
        visualize: bool = False,
) -> np.ndarray:
    """
    :param hdf5_path:
    :param channels: list of strings of the channels of interest e.g. ["ch1", "ch2", "ch3", "ch4", "ch6"]
    :param stations: dict containing for each station of interest its coordinate
    :param dataframe_path:
    :param copy_last_if_missing:
    :param cropped_img_size: size of the returned images
    :return: numpy array
    Returned array is of shape:
    if cropped
    (96 * number of stations, nb of channels,cropped_img_size, cropped_img_size)
    where 96 is the number of timestamps in a day
    where Returned_array[i*len(stations)+station_index] contains the images at ith timestamps of the station wanted
    e.g the image at the 34th timestamps in the day of the 3rd station in the stations dictionnary (given as param) is at
    Returned_array[33*len(stations)+2]

    if not cropped
    (96, nb of channels, original image height, original image width)
    Fetch images of a day and crop them
    """
    assert os.path.isfile(hdf5_path), f"invalid hdf5 path: {hdf5_path}"
    assert channels, "list of channels must not be empty"
    with h5py.File(hdf5_path, "r") as h5_data:
        global_start_idx = h5_data.attrs["global_dataframe_start_idx"]
        global_end_idx = h5_data.attrs["global_dataframe_end_idx"]
        archive_lut_size = global_end_idx - global_start_idx
        global_start_time = datetime.datetime.strptime(h5_data.attrs["global_dataframe_start_time"], "%Y.%m.%d.%H%M")
        lut_timestamps = [global_start_time + idx * datetime.timedelta(minutes=15) for idx in range(archive_lut_size)]
        stations_data = {}
        if stations:
            df = pd.read_pickle(dataframe_path) if dataframe_path else None
            # df = df.fillna(0)
            # df = df.replace(['nan'], [0])
            # assume lats/lons stay identical throughout all frames; just pick the first available arrays
            idx, lats, lons = 0, None, None
            while (lats is None or lons is None) and idx < archive_lut_size:
                lats, lons = fetch_hdf5_sample("lat", h5_data, idx), fetch_hdf5_sample("lon", h5_data, idx)
            assert lats is not None and lons is not None, "could not fetch lats/lons arrays (hdf5 might be empty)"
            # for reg, coords in tqdm.tqdm(stations.items(), desc="preparing stations data"):
            for reg, coords in stations.items():
                station_coords = (np.argmin(np.abs(lats - coords[0])), np.argmin(np.abs(lons - coords[1])))
                station_data = {"coords": station_coords}
                if dataframe_path:
                    station_data["ghi"] = [df.at[pd.Timestamp(t), reg + "_GHI"] for t in lut_timestamps]
                    station_data["csky"] = [df.at[pd.Timestamp(t), reg + "_CLEARSKY_GHI"] for t in lut_timestamps]
                stations_data[reg] = station_data

        if cropped_img_size:
            # different image for each station because cropped image around stations coordinate
            raw_data = np.zeros((archive_lut_size, len(stations), len(channels), cropped_img_size, cropped_img_size),
                                dtype=np.uint8)
        else:
            # nb of timeframe selected, nb of channel (5), height img, length img, each image has 3 channels (RGB)
            # raw_data = np.zeros((archive_lut_size, len(channels), 650, 1500, 3), dtype=np.uint8)
            raw_data = np.zeros((archive_lut_size, len(channels), 650, 1500), dtype=np.uint8)
        for channel_idx, channel_name in tqdm.tqdm(enumerate(channels), desc="preparing img data", total=len(channels)):
            assert channel_name in h5_data, f"missing channel: {channels}"
            norm_min = h5_data[channel_name].attrs.get("orig_min", None)
            norm_max = h5_data[channel_name].attrs.get("orig_max", None)
            channel_data = [fetch_hdf5_sample(channel_name, h5_data, idx) for idx in range(archive_lut_size)]
            assert all([array is None or array.shape == (650, 1500) for array in channel_data]), \
                "one of the saved channels had an expected dimension"
            last_valid_array_idx = None
            for array_idx, array in enumerate(channel_data):
                if array is None:
                    if copy_last_if_missing and last_valid_array_idx is not None:
                        if cropped_img_size:
                            for station_idx, (station_name, station) in enumerate(stations_data.items()):
                                raw_data[array_idx, station_idx, channel_idx, :, :] = raw_data[last_valid_array_idx,
                                                                                      station_idx, channel_idx, :, :]
                        else:
                            raw_data[array_idx, channel_idx, :, :] = raw_data[last_valid_array_idx, channel_idx, :, :]
                    continue
                array = (((array.astype(np.float32) - norm_min) / (norm_max - norm_min)) * 255).astype(np.uint8)
                # transform into RGB and add 3 channels per image
                # array = cv.applyColorMap(array, cv.COLORMAP_BONE)
                for station_idx, (station_name, station) in enumerate(stations_data.items()):
                    # station_color = utils.get_label_color_mapping(station_idx + 1).tolist()[::-1]
                    # shades = [(128,128,128), (255, 0, 0), (192,192,192), (112,128,144), (47,79,79), (220,220,220), (105,105,105)]
                    # array = cv.circle(array, station["coords"][::-1], radius=3, color=shades[station_idx], thickness=-1)
                    if cropped_img_size:
                        array_cropped = crop(copy.deepcopy(array), station["coords"], cropped_img_size)
                        raw_data[array_idx, station_idx, channel_idx, ...] = cv.flip(array_cropped, 0)
                if cropped_img_size is None:
                    raw_data[array_idx, channel_idx, :, :] = cv.flip(array, 0)
                last_valid_array_idx = array_idx
    if visualize:
        display_data = []
        for array_idx in tqdm.tqdm(range(archive_lut_size), desc="reshaping for final display"):
            display_data2 = []
            for st_idx in range(len(stations)):
                display = cv.hconcat([raw_data[array_idx, st_idx, ch_idx, ...] for ch_idx in range(len(channels))])
                display_data2.append(display)
            display = np.stack(display_data2)
            display = cv.vconcat([display[st_idx, ...] for st_idx in range(len(stations))])
            display_data.append(display)
        display = np.stack(display_data)
        array_idx, window_name, paused = 0, hdf5_path.split("/")[-1], False
        while True:
            cv.imshow(window_name, display[array_idx, :, :])
            ret = cv.waitKey(30 if not paused else 300)
            if ret == ord('q') or ret == 27:  # q or ESC
                break
            elif ret == ord(' '):
                paused = ~paused
            if not paused or ret == ord('c'):
                array_idx = (array_idx + 1) % archive_lut_size
    if cropped_img_size:
        raw_data = np.reshape(raw_data,
                              (archive_lut_size * len(stations), len(channels), cropped_img_size, cropped_img_size))
    return raw_data


def crop(array, center, cropped_img_size):
    """
    :param array: 2d numpy array
    :param center: coordinate of the center of where the image should be cropped
    :param cropped_img_size: size of the resulting image (resulting image is always a square)
    :return: 2d numpy array corresponding to the cropped image
    """

    crop_size = cropped_img_size - 1  # the center is excluded to compute the corners
    # get coordinate of the corners of the resulting image if crop around the center given
    corner_coord = [center[0] - math.ceil(crop_size / 2), 1 + center[0] + math.floor(crop_size / 2),
                    center[1] - math.ceil(crop_size / 2), 1 + center[1] + math.floor(crop_size / 2)]

    # while keeping the correct crop size, move corners so they are not outside of the image.
    # 'center' will not be at the middle of the image if the crop was going outside of the image border.
    for i in reversed(range(len(corner_coord))):
        # move corners that are below 0 and shift the corresponding corner to keep the correct cropped size
        corner_coord[i] -= min(0, corner_coord[(i // 2) * 2])
    for i in range(len(corner_coord)):
        # move corners that are above the border and shift the corresponding corner to keep the correct cropped size
        corner_coord[i] -= max(0, corner_coord[((i // 2) * 2) + 1] - array.shape[i // 2])

    # assert (0 < array).any(), f"Cropping failed. Array {array.shape}, Crop size {crop_size}, Center {center}, Corners of tentatively cropped image {corner_coord}"

    return array[corner_coord[0]:corner_coord[1], corner_coord[2]:corner_coord[3]]


def get_data(dataframe_path, start_date, end_date, channels=None, stations=None, cropped_img_size=200):
    """
    :param dataframe_path:
    :param start_date: string corresponding to the date of the first day wanted e.g. "2011-12-01"
    :param end_date: string corresponding to the date of the first day unwanted i.e. end_date is not included e.g. "2011-12-03"
    :return: dataframe, imgs
     where
     dataframe: contains the informations of all timestamps between those dates. one row per combination of timestamps and station
     imgs: np.array of cropped images
     shape=(number of days * 96 * number of stations, nb of channels,cropped_img_size, cropped_img_size)

    Corresponding index between dataframe and imgs

    With n = i*len(stations)+station_index
    dataframe.iloc[n] contains the metadata of imgs[n]
    where
    station_index: is the index of the station in the dictionnary 'stations'
    i: is the ith timestamps from start_date at 08:00:00
    Useful if you want to iterate on every timestamps of one station and the data fetched contains multiple

    With n = df.index.get_loc((an_iso_datetime)),
    dataframe.iloc[n] returns a dataframe shape (nb_stations, nb_columns) containing the
    metadata for each stations at the specified datetime
    imgs[n] return the corresponding images

    With n = df.index.get_loc((an_iso_datetime,station_index))
    dataframe.iloc[n] returns a series with the metadata of the specified
    station at the specified datetime
    imgs[n] returns the corresponding images
    """
    if channels is None:
        channels = ["ch1", "ch2", "ch3", "ch4", "ch6"]
    if stations is None:
        stations = {
            "BND": [40.05192, -88.37309, 230],
            "TBL": [40.12498, -105.23680, 1689],
            "DRA": [36.62373, -116.01947, 1007],
            "FPK": [48.30783, -105.10170, 634],
            "GWN": [34.25470, -89.87290, 98],
            "PSU": [40.72012, -77.93085, 376],
            "SXF": [43.73403, -96.62328, 473]
        }
    start_date, end_date = start_date + " 08:00:00", end_date + " 08:00:00"

    # select timeframe
    dataframe = pd.read_pickle(dataframe_path)
    dataframe = dataframe[dataframe.index >= datetime.datetime.fromisoformat(start_date)]
    dataframe = dataframe[dataframe.index < datetime.datetime.fromisoformat(end_date)]

    wanted_days = []
    jump_day = pd.Timedelta("P1DT0H0M0S", ).to_pytimedelta()
    start = datetime.datetime.fromisoformat(start_date)
    end = datetime.datetime.fromisoformat(end_date)
    i = 0
    while start + i * jump_day < end:
        new_datetime = start + i * jump_day
        if new_datetime in dataframe.index:
            wanted_days.append(new_datetime)
        i += 1

    hdf5_path = dataframe['hdf5_8bit_path'].iloc[dataframe.index.get_loc(wanted_days[0])]
    imgs = fetch_and_crop_hdf5_imagery(hdf5_path, channels, stations, dataframe_path, cropped_img_size=cropped_img_size)
    for day in wanted_days[1:]:
        hdf5_path = dataframe['hdf5_8bit_path'].iloc[dataframe.index.get_loc(day)]
        day_img = fetch_and_crop_hdf5_imagery(hdf5_path, channels, stations, dataframe_path,
                                              cropped_img_size=cropped_img_size)
        imgs = np.vstack((imgs, day_img))

    ## format dataframe to correspond to imgs

    # keep those columns intact
    dataframe = dataframe.set_index(
        ['ncdf_path', 'hdf5_8bit_path', 'hdf5_8bit_offset', 'hdf5_16bit_path', 'hdf5_16bit_offset'],
        append=True)

    # remove columns with about station not in the dictionary
    stations_name = tuple(stations.keys())
    dataframe = dataframe.loc[:, dataframe.columns.str.startswith(stations_name)]

    # create column for station name and change other columns to not be station specific
    # (e.g. replace columns 'BND_GHI', TBL_GHI','DRA_GHI', etc by columns STATION_NAME and GHI)
    # each rows contain information about only one station
    dataframe.columns = dataframe.columns.str.split('_', 1, expand=True)
    dataframe = dataframe.stack(0).reset_index().rename(columns={'level_6': 'STATION_NAME'}).set_index(['iso-datetime'])

    # add column station index and add it as index (with the datetime)
    dataframe['STATION_INDEX'] = dataframe.apply(lambda row: stations_name.index(row['STATION_NAME']), axis=1)
    dataframe = dataframe.set_index(['STATION_INDEX'], append=True)

    # sort dataframe so that the index correspond between the dataframe and the images np.array

    dataframe.sort_index(inplace=True)
    return dataframe, imgs


def get_sample(processed_dataframe, cropped_imgs, iso_datetime, station_idx=None):
    """
    :param processed_dataframe: dataframe processed by get_data
    :param cropped_imgs: cropped imgs from get_data
    :param iso_datetime: e.g. datetime.datetime.fromisoformat("2011-12-01 08:00:00")
    :param station_idx: optional, index of the station according to the dictionary 'stations' given to get_data. Useful
                        if the processed data contains more than one station and want to filter out the other ones.
    :return: subsection of dataframe and imgs according to the datetime and station wanted
    If the station_idx is specified, it returns a series of shape (number of columns, ) and a np.array of shape
    (number of channels, image height, image width).
    If station_idx is not specified, it returns a dataframe of shape (number of stations, number of columns) and a np.array of
    shape (number of stations, number of channels, image height, image width).

    Easily select sample at a specific datetime (and optionally specific station) inside the preprocessed data.
    """
    if station_idx is not None:
        n = df.index.get_loc((iso_datetime, station_idx))
    else:
        n = df.index.get_loc((iso_datetime))
    return processed_dataframe.iloc[n], cropped_imgs[n]


def image_data_generator(data, batch_size=32):
    for idx, (imgs, stations_data) in enumerate(data):
        imgs = imgs.reshape(-1, 64, 64, 5)
        target_list = []
        for station_name, station_reading in stations_data.items():
            ghi_reading = station_reading['ghi'] + batch_size * [0]
            for idx in range(0, len(ghi_reading) - batch_size):
                target_list.append(ghi_reading[idx:idx + 4])
        target_list = np.vstack(target_list)

        for img_idx in range(0, imgs.shape[0], batch_size):
            yield imgs[img_idx:img_idx + batch_size].astype(np.float32), target_list[
                                                                         img_idx:img_idx + batch_size].astype(
                np.float32)


def dummy_data_generator(data, batch_size=32):
    """
    Generate dummy data for the model, only for example purposes.
    """
    batch_size = 32
    image_dim = (64, 64)
    n_channels = 5
    output_seq_len = 4

    for i in range(0, len(target_datetimes), batch_size):
        batch_of_datetimes = target_datetimes[i:i + batch_size]
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


def process_df(df, big_gap=6):
    """
    Big gap is calculated in number of hours but
    code can easily be changed for days or weeks if necessary
    Definition of classes still need to be done
    Will need some modifications when the ratio is divided by 0
    or need to deal with this in class allocation
    """
    # Add ratio and flag that GHI exist
    station = ['BND', 'TBL', 'DRA', 'FPK', 'GWN', 'PSU', 'SXF']
    for stat in station:
        df[f"Flag_T0_{stat}"] = 1 - df[f"{stat}_GHI"].isnull()
        df[f"Ratio_GHI_{stat}"] = df[f"{stat}_CLEARSKY_GHI"] / df[f"{stat}_GHI"]

    # Add flag that the image at T0 exists
    df['Flag_T0_image'] = df['ncdf_path'] != 'nan'

    # Add counts of consecutive missing image paths
    df['count_no_path'] = df.groupby((df['ncdf_path'] != df['ncdf_path'].shift(1)).cumsum()).cumcount() + 1

    # Remove large gaps in dataframe
    adjust = datetime.timedelta(minutes=-15)
    gap_num = big_gap * 4
    gap_time = datetime.timedelta(hours=-(big_gap - .25))
    pos_big_gap = df[df['count_no_path'] == gap_num].index
    for pos in pos_big_gap:
        count = df['count_no_path'][pos:]
        end = count[count == 1].index[0] + adjust
        start = pos + gap_time
        to_drop = pd.date_range(start, end, freq='15min')
        df = df[~df.index.isin(to_drop)]

    # Create flag for data availability
    target = [0, 1, 3, 6]

    for stat in station:
        df[f"{stat}_data_avail"] = 0
        for goal in target:
            goaltime = datetime.timedelta(hours=goal)
            df[f"{stat}_data_avail"] += df[f"{stat}_GHI"][df[f"{stat}_GHI"].index + goaltime].isnull()
        df[f"{stat}_data_avail"] = df[f"{stat}_data_avail"] == 0
        df[f"{stat}_data_avail_day"] = df[f"{stat}_data_avail"] * df[f"{stat}_DAYTIME"]
    return df


def get_station_timezone(stations):
    station_timezone = {}
    for station_name in stations.keys():
        tf = TimezoneFinder()
        #     latitude, longitude = 52.5061, 13.358
        latitude, longitude = stations[station_name][0], stations[station_name][1]
        timezone = tf.timezone_at(lng=longitude, lat=latitude)
        station_timezone[station_name] = pytz.timezone(timezone)
    return station_timezone

def convert_time(timestamp, station_timezone, station_name):
    """
    Take the hour/minute and month of the timestamp and convert them to a
    sin/cos vector to better represent the similarity between January (1)
    and December (12) or 23h45 and 00h00
    This function can take a string (json file) or a timestamp object (pd dataframe)
    """

    fmt = "%Y-%m-%d %H:%M:%S"

    try:
        timestamp = str(timestamp)
        utc_time = datetime.datetime.strptime(timestamp, fmt).replace(tzinfo=pytz.utc)
    except TypeError:
        utc_time = timestamp

    tz = station_timezone[station_name]
    local_datetime = utc_time.astimezone(tz)

    min15_in_day = 24 * 4
    sin_month = np.sin(2 * np.pi * local_datetime.month / 12)
    cos_month = np.cos(2 * np.pi * local_datetime.month / 12)
    sin_minute = np.sin(2 * np.pi * (local_datetime.hour * 4 +
                                     local_datetime.minute / 15) / min15_in_day)
    cos_minute = np.cos(2 * np.pi * (local_datetime.hour *
                                     4 + local_datetime.minute / 15) / min15_in_day)
    return (sin_month, cos_month, sin_minute, cos_minute)


if __name__ == "__main__":
    # Show cropped images for a day
    # img_data = fetch_and_crop_hdf5_imagery(hdf5_path, target_channels, stations, dataframe_path, cropped_img_size=cropped_img_size, visualize=True)
    # print(img_data.shape)

    df, imgs = get_data(
        "/project/cq-training-1/project1/data/catalog.helios.public.20100101-20160101.pkl", "2011-12-01", "2011-12-04")

    print('Dataframe', df.shape, 'Images', imgs.shape)

    a_datetime = datetime.datetime.fromisoformat("2011-12-03 13:45:00")
    metadata, cropped_image = get_sample(df, imgs, a_datetime, 4)
    print('Specific metadata', metadata.shape, 'Specific image', cropped_image.shape)
    print(metadata)
    print(cropped_image)