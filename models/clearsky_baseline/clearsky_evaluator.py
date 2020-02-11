import argparse
import datetime
import json
import os
import typing

import pandas as pd
import numpy as np


def parse_gt_ghi_values(
        target_stations: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],
        target_datetimes: typing.List[datetime.datetime],
        target_time_offsets: typing.List[datetime.timedelta],
        dataframe: pd.DataFrame,
        clearsky=False,
) -> np.ndarray:
    """Parses all required station GHI values from the provided dataframe for the evaluation of predictions."""
    gt = []
    for station_idx, station_name in enumerate(target_stations):
        if clearsky:
            station_ghis = dataframe[station_name + "_CLEARSKY_GHI"]
        else:
            station_ghis = dataframe[station_name + "_GHI"]
        for target_datetime in target_datetimes:
            seq_vals = []
            for time_offset in target_time_offsets:
                index = target_datetime + time_offset
                if index in station_ghis.index:
                    seq_vals.append(station_ghis.iloc[station_ghis.index.get_loc(index)])
                else:
                    seq_vals.append(float("nan"))
            gt.append(seq_vals)
    return np.concatenate(gt, axis=0)


def parse_nighttime_flags(
        target_stations: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],
        target_datetimes: typing.List[datetime.datetime],
        target_time_offsets: typing.List[datetime.timedelta],
        dataframe: pd.DataFrame,
) -> np.ndarray:
    """Parses all required station daytime flags from the provided dataframe for the masking of predictions."""
    flags = []
    for station_idx, station_name in enumerate(target_stations):
        station_flags = dataframe[station_name + "_DAYTIME"]
        for target_datetime in target_datetimes:
            seq_vals = []
            for time_offset in target_time_offsets:
                index = target_datetime + time_offset
                if index in station_flags.index:
                    seq_vals.append(station_flags.iloc[station_flags.index.get_loc(index)] > 0)
                else:
                    seq_vals.append(False)
            flags.append(seq_vals)
    return np.concatenate(flags, axis=0)


def main(
        preds_output_path: typing.AnyStr,
        admin_config_path: typing.AnyStr,
        user_config_path: typing.Optional[typing.AnyStr] = None,
        stats_output_path: typing.Optional[typing.AnyStr] = None,
        ignore_nan: typing.Optional[typing.BinaryIO] = False,
) -> None:
    """Extracts predictions from a user model/data loader combo and saves them to a CSV file."""

    user_config = {}
    if user_config_path:
        assert os.path.isfile(user_config_path), f"invalid user config file: {user_config_path}"
        with open(user_config_path, "r") as fd:
            user_config = json.load(fd)

    assert os.path.isfile(admin_config_path), f"invalid admin config file: {admin_config_path}"
    with open(admin_config_path, "r") as fd:
        admin_config = json.load(fd)

    dataframe_path = admin_config["dataframe_path"]
    assert os.path.isfile(dataframe_path), f"invalid dataframe path: {dataframe_path}"
    dataframe = pd.read_pickle(dataframe_path)

    if "start_bound" in admin_config:
        dataframe = dataframe[dataframe.index >= datetime.datetime.fromisoformat(admin_config["start_bound"])]
    if "end_bound" in admin_config:
        dataframe = dataframe[dataframe.index < datetime.datetime.fromisoformat(admin_config["end_bound"])]


    if "target_datetimes" in admin_config:
        target_datetimes = [datetime.datetime.fromisoformat(d) for d in admin_config["target_datetimes"]]
    # modified from original, if no target_datetimes given
    # targets are every datetimes in the dataframe between start_bound and end_bound
    else:
        target_datetimes = []
        jump_15_min = pd.Timedelta("P0DT0H15M0S",).to_pytimedelta()
        start = datetime.datetime.fromisoformat(admin_config["start_bound"])
        end = datetime.datetime.fromisoformat(admin_config["end_bound"])
        i = 0
        while start + i*jump_15_min < end:
            new_datetime = start + i*jump_15_min
            if new_datetime in dataframe.index:
                target_datetimes.append(new_datetime)
            i += 1
        if end in dataframe.index:
            target_datetimes.append(end)
    assert target_datetimes and all([d in dataframe.index for d in target_datetimes])
    target_stations = admin_config["stations"]
    target_time_offsets = [pd.Timedelta(d).to_pytimedelta() for d in admin_config["target_time_offsets"]]

    if "bypass_predictions_path" in admin_config and admin_config["bypass_predictions_path"]:
        # re-open cached output if possible (for 2nd pass eval)
        assert os.path.isfile(preds_output_path), f"invalid preds file path: {preds_output_path}"
        with open(preds_output_path, "r") as fd:
            predictions = fd.readlines()
        assert len(predictions) == len(target_datetimes) * len(target_stations), \
            "predicted ghi sequence count mistmatch wrt target datetimes x station count"
        assert len(predictions) % len(target_stations) == 0
        predictions = np.asarray([float(ghi) for p in predictions for ghi in p.split(",")])
    else:
        # predictions = generate_all_predictions(target_stations, target_datetimes,
        #                                       target_time_offsets, dataframe, user_config)
        predictions = parse_gt_ghi_values(target_stations, target_datetimes,
                                          target_time_offsets, dataframe, clearsky=True)
        predictions = predictions.reshape((-1, len(target_time_offsets)))
        with open(preds_output_path, "w") as fd:
            for pred in predictions:
                fd.write(",".join([f"{v:0.03f}" for v in pred.tolist()]) + "\n")

    if any([s + "_GHI" not in dataframe for s in target_stations]):
        print("station GHI measures missing from dataframe, skipping stats output")
        return

    if not ignore_nan:
        assert not np.isnan(predictions).any(), "user predictions should NOT contain NaN values"
    predictions = predictions.reshape((len(target_stations), len(target_datetimes), len(target_time_offsets)))
    gt = parse_gt_ghi_values(target_stations, target_datetimes, target_time_offsets, dataframe)
    gt = gt.reshape((len(target_stations), len(target_datetimes), len(target_time_offsets)))
    day = parse_nighttime_flags(target_stations, target_datetimes, target_time_offsets, dataframe)
    day = day.reshape((len(target_stations), len(target_datetimes), len(target_time_offsets)))

    squared_errors = np.square(predictions - gt)
    stations_rmse = np.sqrt(np.nanmean(squared_errors, axis=(1, 2)))
    for station_idx, (station_name, station_rmse) in enumerate(zip(target_stations, stations_rmse)):
        print(f"station '{station_name}' RMSE = {station_rmse:.02f}")
    horizons_rmse = np.sqrt(np.nanmean(squared_errors, axis=(0, 1)))
    for horizon_idx, (horizon_offset, horizon_rmse) in enumerate(zip(target_time_offsets, horizons_rmse)):
        print(f"horizon +{horizon_offset} RMSE = {horizon_rmse:.02f}")
    overall_rmse = np.sqrt(np.nanmean(squared_errors))
    print(f"overall RMSE = {overall_rmse:.02f}")

    if stats_output_path is not None:
        # we remove nans to avoid issues in the stats comparison script, and focus on daytime predictions
        squared_errors = squared_errors[~np.isnan(gt) & day]
        with open(stats_output_path, "w") as fd:
            for err in squared_errors.reshape(-1):
                fd.write(f"{err:0.03f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("preds_out_path", type=str,
                        help="path where the raw model predictions should be saved (for visualization purposes)")
    parser.add_argument("admin_cfg_path", type=str,
                        help="path to the JSON config file used to store test set/evaluation parameters")
    parser.add_argument("-s", "--stats_output_path", type=str, default=None,
                        help="path where the prediction stats should be saved (for benchmarking)")
    parser.add_argument("-n", "--ignore_nan", type=bool, default=False,
                        help="Accept NaN prediction if True. Otherwise, AssertionError if any NaN in the predictions")
    args = parser.parse_args()
    main(
        preds_output_path=args.preds_out_path,
        admin_config_path=args.admin_cfg_path,
        stats_output_path=args.stats_output_path,
        ignore_nan=args.ignore_nan,
    )
