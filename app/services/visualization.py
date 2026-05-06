import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from app.utils.logger import get_logger

logger = get_logger(__name__)

PLOT_DIR = "outputs/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

def plot_actual_vs_predicted(state: str, dates: pd.DatetimeIndex, y_true: np.ndarray, y_pred: np.ndarray, model_name: str):
    plt.figure(figsize=(12, 6))
    plt.plot(dates, y_true, label='Actual Sales', color='blue', marker='o')
    plt.plot(dates, y_pred, label=f'Predicted Sales ({model_name})', color='orange', linestyle='--', marker='x')
    plt.title(f'{state} - Actual vs Predicted Sales ({model_name})')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save
    plot_path = os.path.join(PLOT_DIR, f"{state}_actual_vs_pred.png")
    plt.savefig(plot_path)
    plt.close()

def plot_forecast(state: str, last_actual_dates: pd.DatetimeIndex, last_actual_sales: np.ndarray, forecast_dates: pd.DatetimeIndex, forecast_sales: np.ndarray, model_name: str):
    plt.figure(figsize=(12, 6))
    plt.plot(last_actual_dates, last_actual_sales, label='Historical Sales', color='blue')
    plt.plot(forecast_dates, forecast_sales, label=f'Forecasted Sales ({model_name})', color='green', linestyle='--')
    plt.title(f'{state} - Future Sales Forecast ({model_name})')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save
    plot_path = os.path.join(PLOT_DIR, f"{state}_forecast.png")
    plt.savefig(plot_path)
    plt.close()

def plot_model_comparison(state: str, results: dict):
    models = list(results.keys())
    rmse_scores = [results[m]['RMSE'] for m in models]
    
    plt.figure(figsize=(8, 5))
    bars = plt.bar(models, rmse_scores, color=['skyblue', 'lightgreen', 'salmon', 'plum'])
    plt.title(f'{state} - Model Comparison (RMSE)')
    plt.ylabel('RMSE')
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, round(yval, 2), va='bottom', ha='center')
        
    plt.tight_layout()
    plot_path = os.path.join(PLOT_DIR, f"{state}_model_comparison.png")
    plt.savefig(plot_path)
    plt.close()
