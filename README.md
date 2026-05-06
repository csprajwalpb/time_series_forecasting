# Time Series Forecasting System

## Project Overview
This is a complete, production-ready, end-to-end Time Series Forecasting System built with Python and FastAPI. The system is designed to forecast the next 8 weeks of sales for multiple states using historical sales data from an uploaded Excel file.

## Setup Instructions
1. **Clone the repository**
2. **Create a virtual environment**: `python -m venv venv` and activate it.
3. **Install dependencies**: `pip install -r requirements.txt`
4. **Run the server**: `python run.py`
5. **Access the API Documentation**: Navigate to `http://localhost:8000/docs` in your browser.

## Architecture Explanation
The application is built using a clean, modular architecture:
- **`app/api/`**: Contains the REST API routes for training, tracking, and forecasting.
- **`app/models/`**: Encapsulates the logic for the 4 distinct Machine Learning algorithms.
- **`app/services/`**: Holds the business logic, including:
  - `preprocessing.py`: Cleans missing dates/values and aggregates state data.
  - `feature_engineering.py`: Computes lags, rolling stats, and time variables cleanly to avoid data leakage.
  - `trainer.py`: Handles chronological time series splits and evaluates the models.
  - `forecasting.py`: Restores models and performs future recursive/sequence inferences.
  - `visualization.py`: Generates comparative model performance plots.
- **`saved_models/`**: Output directory for trained models and the centralized model registry.
- **`outputs/plots/`**: Location where actual vs predicted and future forecast graphs are saved.

## Model Explanation
We utilize four advanced models, evaluating each against a strict chronological 56-day validation split to find the lowest RMSE per state:
1. **SARIMA**: Employs `pmdarima` for auto-tuning parameters (`p,d,q`), specifically capturing the weekly historical seasonality.
2. **Prophet**: A Facebook-developed additive regression model. We transform our columns properly to `ds` and `y` while factoring in Indian holidays natively.
3. **XGBoost**: A powerful gradient-boosting regressor. We implemented **recursive forecasting**, predicting $t+1$ and recomputing the lag and rolling features iteratively.
4. **LSTM**: A Keras Deep Learning Sequential model utilizing a 30-day sequence lookback window, safely scaled iteratively utilizing `MinMaxScaler`.

## API Usage & Sample Requests

### 1. Train Models
**Endpoint**: `POST /api/v1/train`
Starts background training across all models.
**Sample Request (cURL)**:
```bash
curl -X POST "http://localhost:8000/api/v1/train" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@your_dataset.xlsx"
```

### 2. Check Task Status
**Endpoint**: `GET /api/v1/status/{task_id}`
Returns the real-time background status of your training job.

### 3. Get Model Metrics
**Endpoint**: `GET /api/v1/metrics`
Returns the comparative performance and the automatically selected best model per state.

### 4. Get 8-Week Forecast
**Endpoint**: `GET /api/v1/forecast/{state}`
**Sample Response**:
```json
{
  "state": "Karnataka",
  "best_model": "XGBoost",
  "forecast": [
    {
      "date": "2026-01-01",
      "sales": 1234.56
    }
  ]
}
```

## Screenshots
*(Insert screenshots of the `/docs` UI and the generated plots from `outputs/plots/` here)*

## Future Improvements
- **Cloud Database Integration**: Move from a local `registry.json` to PostgreSQL/MongoDB.
- **Enhanced Caching**: Transition from in-memory dictionary caching to Redis.
- **Hyperparameter Tuning**: Expand the grid search space for XGBoost and LSTM architectures.
