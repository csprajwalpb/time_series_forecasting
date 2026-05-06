import pandas as pd
from typing import Dict, Any, List
from app.utils.helpers import get_best_model_info
from app.models.sarima_model import SARIMAModel
from app.models.prophet_model import ProphetModel
from app.models.xgboost_model import XGBoostModel
from app.models.lstm_model import LSTMModel
from app.utils.logger import get_logger
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt
from app.services.visualization import PLOT_DIR

logger = get_logger(__name__)

# Simple in-memory cache for forecasts
FORECAST_CACHE = {}

def load_model(model_name: str, path: str):
    if model_name == "SARIMA":
        model = SARIMAModel()
    elif model_name == "Prophet":
        model = ProphetModel()
    elif model_name == "XGBoost":
        model = XGBoostModel()
    elif model_name == "LSTM":
        model = LSTMModel()
    else:
        raise ValueError(f"Unknown model name: {model_name}")
        
    model.load(path)
    return model

def generate_forecast(state: str, steps: int = 56) -> Dict[str, Any]:
    # Check cache first
    if state in FORECAST_CACHE:
        cache_entry = FORECAST_CACHE[state]
        if datetime.now() - cache_entry["generated_at"] < timedelta(hours=1):
            logger.info(f"Returning cached forecast for {state}")
            return {
                "state": state,
                "best_model": cache_entry["best_model"],
                "forecast": cache_entry["forecast"]
            }
            
    info = get_best_model_info(state)
    if not info:
        raise ValueError(f"No trained model found for state: {state}")
        
    model_name = info['best_model']
    model_path = info['model_path']
    last_date_str = info.get('last_date')
    
    if not last_date_str:
        last_date = pd.to_datetime('today').date()
    else:
        last_date = pd.to_datetime(last_date_str).date()
        
    future_dates = [str(last_date + timedelta(days=i+1)) for i in range(steps)]
    
    logger.info(f"Loading {model_name} for {state} from {model_path}")
    model = load_model(model_name, model_path)
    
    predictions = model.predict(steps=steps)
    
    forecast_list = []
    for d, p in zip(future_dates, predictions):
        forecast_list.append({"date": d, "sales": float(p)})
        
    # Generate Forecast Plot
    plt.figure(figsize=(12, 6))
    plt.plot(pd.to_datetime(future_dates), [float(p) for p in predictions], label=f'Forecasted Sales ({model_name})', color='green', linestyle='--')
    plt.title(f'{state} - Future Sales Forecast ({model_name})')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{state}_forecast_{model_name.lower()}.png"))
    plt.close()

    FORECAST_CACHE[state] = {
        "best_model": model_name,
        "forecast": forecast_list,
        "generated_at": datetime.now()
    }
    
    return {
        "state": state,
        "best_model": model_name,
        "forecast": forecast_list
    }

def generate_all_forecasts(steps: int = 56) -> List[Dict[str, Any]]:
    import json
    from pathlib import Path
    
    REGISTRY_PATH = Path("saved_models/registry.json")
    if not REGISTRY_PATH.exists():
        return []
        
    with open(REGISTRY_PATH, "r") as f:
        registry = json.load(f)
        
    results = []
    for state in registry.keys():
        try:
            forecast = generate_forecast(state, steps)
            results.append(forecast)
        except Exception as e:
            logger.error(f"Failed to forecast for {state}: {e}")
            
    return results
