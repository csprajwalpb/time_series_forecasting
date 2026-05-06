from prophet import Prophet
import pandas as pd
import numpy as np
import joblib
from app.utils.logger import get_logger

logger = get_logger(__name__)

class ProphetModel:
    def __init__(self):
        self.model = None

    def train(self, df: pd.DataFrame):
        logger.info("Training Prophet model...")
        prophet_df = df[['date', 'sales']].rename(columns={'date': 'ds', 'sales': 'y'})
        
        self.model = Prophet(
            weekly_seasonality=True,
            yearly_seasonality=True,
            daily_seasonality=False
        )
        self.model.add_country_holidays(country_name='IN')
        self.model.fit(prophet_df)
        logger.info("Prophet training complete.")

    def predict(self, steps: int) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model is not trained yet.")
        future = self.model.make_future_dataframe(periods=steps, freq='D')
        forecast = self.model.predict(future)
        # Prophet predicts for the whole history + future, we only want the last `steps`
        return forecast['yhat'].iloc[-steps:].values

    def save(self, path: str):
        joblib.dump(self.model, path)

    def load(self, path: str):
        self.model = joblib.load(path)
