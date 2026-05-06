import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error
from app.utils.logger import get_logger

logger = get_logger(__name__)

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    # Ensure no zeroes for MAPE to avoid division by zero or infinite values
    y_true_safe = np.where(y_true == 0, 1e-10, y_true)
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true_safe, y_pred)
    
    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape)
    }

def select_best_model(results: dict) -> tuple:
    """
    results: { 'model_name': {'MAE': x, 'RMSE': y, 'MAPE': z} }
    Selects the model with the lowest RMSE.
    """
    best_model = None
    best_rmse = float('inf')
    
    for model_name, metrics in results.items():
        if metrics['RMSE'] < best_rmse:
            best_rmse = metrics['RMSE']
            best_model = model_name
            
    return best_model, results[best_model]
