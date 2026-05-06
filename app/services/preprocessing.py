import pandas as pd
from app.utils.logger import get_logger

logger = get_logger(__name__)

def preprocess_data(file_path: str) -> pd.DataFrame:
    """
    Loads Excel data, aggregates by state and date, handles missing dates and values.
    Assumes columns: 'date', 'state', 'sales' (or similar recognizable names).
    """
    logger.info(f"Loading data from {file_path}")
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        logger.error(f"Error reading excel file: {e}")
        raise ValueError(f"Failed to read file: {e}")

    # Standardize column names (lowercase, strip whitespace)
    df.columns = [col.lower().strip() for col in df.columns]
    
    # Identify required columns with more flexible matching
    date_col = next((col for col in df.columns if any(x in col for x in ['date', 'time', 'day'])), None)
    state_col = next((col for col in df.columns if any(x in col for x in ['state', 'region', 'location', 'city', 'zone'])), None)
    sales_col = next((col for col in df.columns if any(x in col for x in ['sale', 'amount', 'revenue', 'value', 'total', 'qty', 'quantity', 'buyer'])), None)

    if not all([date_col, state_col, sales_col]):
        missing = [c for c, n in zip(['date', 'state', 'sales'], [date_col, state_col, sales_col]) if not n]
        raise ValueError(f"Missing required columns. Looked for equivalents of: {missing}. Actual columns in file: {list(df.columns)}")

    # Rename for consistency
    df.rename(columns={date_col: 'date', state_col: 'state', sales_col: 'sales'}, inplace=True)
    
    # Ensure date is datetime
    df['date'] = pd.to_datetime(df['date'])

    # Aggregate sales by state and date (in case there are multiple entries per day per state)
    agg_df = df.groupby(['state', 'date'])['sales'].sum().reset_index()

    processed_dfs = []
    states = agg_df['state'].unique()

    logger.info(f"Processing data for states: {states}")

    for state in states:
        state_df = agg_df[agg_df['state'] == state].copy()
        
        # Generate continuous date range
        min_date = state_df['date'].min()
        max_date = state_df['date'].max()
        all_dates = pd.date_range(start=min_date, end=max_date, freq='D')
        
        state_df.set_index('date', inplace=True)
        # Reindex to fill missing dates with NaN
        state_df = state_df.reindex(all_dates)
        state_df.index.name = 'date'
        state_df.reset_index(inplace=True)
        state_df['state'] = state
        
        # Handle missing values
        # 1. Interpolation
        state_df['sales'] = state_df['sales'].interpolate(method='linear')
        # 2. Forward fill for any remaining (e.g. if first value is nan)
        state_df['sales'] = state_df['sales'].ffill()
        # 3. Zero fallback
        state_df['sales'] = state_df['sales'].fillna(0)
        
        processed_dfs.append(state_df)

    final_df = pd.concat(processed_dfs, ignore_index=True)
    logger.info("Preprocessing completed successfully.")
    return final_df
