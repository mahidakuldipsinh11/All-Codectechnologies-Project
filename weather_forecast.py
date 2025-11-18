import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf 
import warnings
import math


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def generate_synthetic_data(start_date, periods):
    """
    Generates synthetic daily temperature data with a slight upward trend 
    and seasonal (yearly) component to simulate real-world data complexity.
    """
    print("--- 1. Generating Synthetic Weather Data ---")
    dates = pd.date_range(start=start_date, periods=periods, freq='D')
    
    
    base_temp = 20
   
    trend = 0.005 * np.arange(periods)
    seasonality = 10 * np.sin(2 * np.pi * np.arange(periods) / 365)
    noise = np.random.normal(0, 1.5, periods)
    
    temperature = base_temp + trend + seasonality + noise
    
    df = pd.DataFrame({'Date': dates, 'Temperature': temperature})
    df.set_index('Date', inplace=True)
    
    print(f"Data generated for {periods} days.")
    return df

def check_stationarity(series):
    """
    Performs the Augmented Dickey-Fuller (ADF) test.
    """
    print("\n--- 2. Checking for Stationarity (ADF Test) ---")
    
    
    result = adfuller(series.dropna())
    
    print(f'ADF Statistic: {result[0]:.4f}')
    print(f'p-value: {result[1]:.4f}')
    
    if result[1] <= 0.05:
        print("Conclusion: The time series is likely stationary (d=0).")
        d = 0 
    else:
        print("Conclusion: The time series is NOT stationary (d>0). Using first differencing (d=1).")
        d = 1 

    return d

def plot_acf_pacf(series, title_suffix="Original Series"):
    """
    Plots the Autocorrelation (ACF) and Partial Autocorrelation (PACF) plots.
    The number of lags is calculated dynamically to prevent the ValueError.
    """
    print(f"\n--- 2.5. Plotting ACF and PACF for {title_suffix} ---")

    
    N = len(series.dropna())
    safe_lags = math.floor(N / 2) - 1
    
    
    max_lags = min(40, safe_lags) 
    
    if max_lags < 1:
        print(f"Warning: Sample size ({N}) too small to plot meaningful ACF/PACF.")
        return

    print(f"Using {max_lags} lags for plots (Sample Size: {N}).")

    fig, ax = plt.subplots(2, 1, figsize=(12, 6))
    
   
    plot_acf(series.dropna(), ax=ax[0], lags=max_lags, title=f'Autocorrelation Function (ACF) - {title_suffix}')
    ax[0].set_xlabel('Lags')
    
    
    plot_pacf(series.dropna(), ax=ax[1], lags=max_lags, title=f'Partial Autocorrelation Function (PACF) - {title_suffix}')
    ax[1].set_xlabel('Lags')
    
    plt.tight_layout()
    plt.show()

def fit_and_forecast_arima(data, differencing_order, forecast_steps=30):
    """
    Fits an ARIMA model to the data and generates a forecast.
    Using p=5, q=1 as initial assumptions based on typical seasonal data.
    """
    p = 5
    q = 1
    order = (p, differencing_order, q)
    
    print(f"\n--- 3. Fitting ARIMA Model (Order: {order}) ---")
    try:
        model = ARIMA(data, order=order)
        model_fit = model.fit()
        print("ARIMA Model Fitted Summary:")
        
        print(f"\n--- 4. Generating {forecast_steps} Day Forecast ---")
        
        forecast_start_index = len(data)
        forecast_end_index = forecast_start_index + forecast_steps - 1
        
        forecast = model_fit.get_prediction(
            start=forecast_start_index, 
            end=forecast_end_index,
        )
        
        forecast_results = forecast.predicted_mean
        conf_int = forecast.conf_int()
        
        
        last_date = data.index[-1]
        forecast_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_steps, freq='D')
        forecast_results.index = forecast_index
        conf_int.index = forecast_index
        
        return model_fit, forecast_results, conf_int
        
    except Exception as e:
        print(f"\nError during ARIMA fitting or forecasting: {e}. Please try adjusting p, d, or q parameters.")
        return None, None, None

def visualize_results(data, forecast_results, conf_int):
    """
    Plots the historical data and the future forecast.
    """
    print("\n--- 5. Visualizing Historical Data and Forecast ---")
    plt.figure(figsize=(14, 7))
    
    
    plt.plot(data.index, data['Temperature'], label='Historical Temperature', color='#0077b6', linewidth=1.5)
    
   
    if forecast_results is not None:
        plt.plot(forecast_results.index, forecast_results.values, 
                 label=f'ARIMA Forecast ({len(forecast_results)} days)', 
                 color='#ef476f', linestyle='--', linewidth=2)
        
       
        plt.fill_between(conf_int.index,
                         conf_int.iloc[:, 0],
                         conf_int.iloc[:, 1], 
                         color='#ef476f', alpha=0.15, label='95% Confidence Interval')

    plt.title('Time Series Forecasting of Daily Temperature (ARIMA)', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Temperature (°C)', fontsize=12)
    plt.legend(loc='lower left')
    plt.grid(True, which='major', linestyle='-', alpha=0.4)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def main():
   
    START_DATE = '2023-01-01'
    DATA_PERIODS = 730  
    FORECAST_STEPS = 30 
    
    
    df = generate_synthetic_data(START_DATE, DATA_PERIODS)
    series = df['Temperature']
    
    
    differencing_order = check_stationarity(series)
    
    
    plot_acf_pacf(series, title_suffix="Original Series")
    
    
    if differencing_order == 1:
        
        diff_series = series.diff().dropna()
        
        plot_acf_pacf(diff_series, title_suffix="First Difference Series")

    
    model_fit, forecast_results, conf_int = fit_and_forecast_arima(
        series, 
        differencing_order, 
        FORECAST_STEPS
    )

    
    if forecast_results is not None:
        visualize_results(df, forecast_results, conf_int)
    else:
        print("\nVisualization skipped due to model error.")

if __name__ == "__main__":
    main()