import pmdarima as pm
import pandas as pd
import numpy as np
import joblib
from app.utils.logger import get_logger

logger = get_logger(__name__)

class SARIMAModel:
    def __init__(self):
        self.model = None

    def train(self, df: pd.DataFrame):
        logger.info("Training SARIMA model...")
        self.model = pm.auto_arima(
            df['sales'], 
            seasonal=True, 
            m=7,
            stepwise=True, 
            suppress_warnings=True, 
            error_action="ignore",
            max_p=3, max_q=3, max_d=2,
            max_P=2, max_Q=2, max_D=1 # Constrained to save time
        )
        logger.info(f"SARIMA training complete.")

    def predict(self, steps: int) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model is not trained yet.")
        forecast = self.model.predict(n_periods=steps)
        if isinstance(forecast, pd.Series):
            return forecast.values
        return forecast

    def save(self, path: str):
        joblib.dump(self.model, path)

    def load(self, path: str):
        self.model = joblib.load(path)
