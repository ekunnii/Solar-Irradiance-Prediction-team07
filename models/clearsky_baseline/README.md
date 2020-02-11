# Baseline with Clear sky

It is a slightly modified version of evaluator.py that computes the rmse obtained by predicting the ghi with the Clear Sky model.

## Example of execution

```
python3 clearsky_evaluator.py clearsky_predictions.csv admin_config.json -s clearsky_rmse -n True
```

4 possibles arguments (2 optional)

clearsky_predictions.csv is the path where the raw predictions are written.

admin_config.json is a file containing configuration for the script. The "dataframe_path" is a path to catalog.helios.public.20100101-20160101.pkl. "target_datetimes" is a list of datetimes on which the rmse is computed. If target_datetimes is not given, it will compute the rmse on every datetime in the dataframe that are between "start_bound" and "end_bound". The bounds are included if they are in the dataframe.

-s clearsky_rmse is an optional arguments. It is the path to the file where the rmse of each individual sample is written. If there is no path given, the file is not written.

-n True is an optional arguments. If True, it ignores when clearsky predicts NaN. Otherwise, it alerts you when it finds a NaN value in the predictions.

## Example of Output

```
station 'BND' RMSE = 142.03
station 'TBL' RMSE = 123.40
station 'DRA' RMSE = 120.90
station 'FPK' RMSE = 67.01
station 'GWN' RMSE = 175.97
station 'PSU' RMSE = 139.32
station 'SXF' RMSE = 95.30
horizon +0:00:00 RMSE = 133.96
horizon +1:00:00 RMSE = 126.22
horizon +3:00:00 RMSE = 138.64
horizon +6:00:00 RMSE = 110.76
overall RMSE = 127.79
```
