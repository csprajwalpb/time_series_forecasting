import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from app.utils.logger import get_logger
from app.services.feature_engineering import create_time_features, create_lag_features

logger = get_logger(__name__)

class XGBoostModel:
    def __init__(self):
        self.model = xgb.XGBRegressor(
            n_estimators=100, 
            learning_rate=0.1, 
            max_depth=5,
            objective='reg:squarederror'
        )
        self.features = [
            'day_of_week', 'week_of_year', 'month', 'quarter', 'year', 'weekend_flag', 'holiday_flag',
            'lag_1', 'lag_7', 'lag_30',
            'rolling_mean_7', 'rolling_std_7', 'rolling_mean_30', 'rolling_std_30'
        ]
        self.last_history = None

    def train(self, df: pd.DataFrame):
        logger.info("Training XGBoost model...")
        X = df[self.features]
        y = df['sales']
        self.model.fit(X, y)
        
        # Save the last 30 days of data to compute rolling and lag features for future
        self.last_history = df.tail(31).copy()
        logger.info("XGBoost training complete.")

    def predict(self, steps: int) -> np.ndarray:
        if self.model is None or self.last_history is None:
            raise ValueError("Model is not trained yet.")
        
        history = self.last_history.copy()
        predictions = []
        
        last_date = history['date'].max()
        
        for i in range(steps):
            next_date = last_date + pd.Timedelta(days=i+1)
            
            # Create a row for the next date
            next_row = pd.DataFrame({'date': [next_date], 'state': [history['state'].iloc[0]], 'sales': [np.nan]})
            # Append to history
            history = pd.concat([history, next_row], ignore_index=True)
            
            # Recompute features
            hist_feat = create_time_features(history)
            hist_feat = create_lag_features(hist_feat)
            
            # The features for the next_date are in the last row
            next_X = hist_feat.iloc[-1:][self.features]
            
            # Predict
            pred = self.model.predict(next_X)[0]
            predictions.append(pred)
            
            # Update the sales in history
            history.loc[history.index[-1], 'sales'] = pred
            
            # Keep history size manageable
            if len(history) > 35:
                history = history.iloc[-35:].reset_index(drop=True)
                
        return np.array(predictions)

    def save(self, path: str):
        data = {
            'model': self.model,
            'last_history': self.last_history,
            'features': self.features
        }
        joblib.dump(data, path)

    def load(self, path: str):
        data = joblib.load(path)
        self.model = data['model']
        self.last_history = data['last_history']
        self.features = data['features']
