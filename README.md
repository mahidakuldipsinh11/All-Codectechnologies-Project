Weather Forecasting with ARIMA

This small project provides a script `weather_forecast.py` that:
- Loads historical weather CSV data with a date column and temperature column.
- Runs stationarity testing (ADF test), plots ACF/PACF.
- Fits an ARIMA model (optional grid search for best (p,d,q)).
- Produces forecasts with confidence intervals and saves plots/CSV.
- Also provides a simple linear regression baseline.

Quick start

1. Install dependencies (Windows PowerShell):

```powershell
python -m pip install -r requirements.txt
```

2. Run the script with the provided sample data:

```powershell
python weather_forecast.py --data sample_data.csv --date-col Date --temp-col Temperature --forecast-steps 30 --model arima
```

Files

- `weather_forecast.py`: main script
- `sample_data.csv`: small synthetic dataset
- `requirements.txt`: Python deps

Notes

- For ARIMA model selection the script will attempt a small grid search over p in [0..3], d in [0..2], q in [0..3] and choose the model with lowest AIC. Provide `--p --d --q` to force parameters.
- The script outputs forecast CSV and PNG plots in the current folder.
