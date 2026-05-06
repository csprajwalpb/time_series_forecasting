import pandas as pd
import numpy as np
import holidays
from app.utils.logger import get_logger

logger = get_logger(__name__)

def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['day_of_week'] = df['date'].dt.dayofweek
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['year'] = df['date'].dt.year
    df['weekend_flag'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Add Indian holidays
    india_holidays = holidays.India()
    df['holiday_flag'] = df['date'].apply(lambda d: 1 if d in india_holidays else 0)
    
    return df

def create_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Assume dataframe is already sorted by date and is for a single state
    df['lag_1'] = df['sales'].shift(1)
    df['lag_7'] = df['sales'].shift(7)
    df['lag_30'] = df['sales'].shift(30)
    
    # Rolling features must not include the current day to avoid data leakage
    # We use shift(1) to start the rolling window from the previous day
    shifted_sales = df['sales'].shift(1)
    df['rolling_mean_7'] = shifted_sales.rolling(window=7).mean()
    df['rolling_std_7'] = shifted_sales.rolling(window=7).std()
    df['rolling_mean_30'] = shifted_sales.rolling(window=30).mean()
    df['rolling_std_30'] = shifted_sales.rolling(window=30).std()
    
    # Backfill missing values caused by shifting to keep dataset size
    df.bfill(inplace=True)
    df.fillna(0, inplace=True) # Fallback for remaining NaNs
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies feature engineering state by state to avoid overlapping lags between states.
    """
    logger.info("Starting feature engineering...")
    df = df.sort_values(by=['state', 'date'])
    
    processed_dfs = []
    states = df['state'].unique()
    
    for state in states:
        state_df = df[df['state'] == state].copy()
        state_df = create_time_features(state_df)
        state_df = create_lag_features(state_df)
        processed_dfs.append(state_df)
        
    final_df = pd.concat(processed_dfs, ignore_index=True)
    logger.info("Feature engineering completed.")
    return final_df
