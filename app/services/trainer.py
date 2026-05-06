import pandas as pd
from typing import Dict
from app.models.sarima_model import SARIMAModel
from app.models.prophet_model import ProphetModel
from app.models.xgboost_model import XGBoostModel
from app.models.lstm_model import LSTMModel
from app.services.evaluator import evaluate_predictions, select_best_model
from app.utils.helpers import task_manager, update_model_registry
from app.utils.logger import get_logger
from app.services.visualization import plot_actual_vs_predicted, plot_model_comparison
import os

logger = get_logger(__name__)

def train_models_for_state(state: str, df: pd.DataFrame) -> dict:
    logger.info(f"Starting training for state: {state}")
    
    val_days = 56
    if len(df) <= val_days:
        logger.error(f"Not enough data to split for {state}. Needs > 56 days.")
        return None
    
    train_df = df.iloc[:-val_days].copy()
    val_df = df.iloc[-val_days:].copy()
    
    models = {
        "SARIMA": SARIMAModel(),
        "Prophet": ProphetModel(),
        "XGBoost": XGBoostModel(),
        "LSTM": LSTMModel(sequence_length=30)
    }
    
    results = {}
    
    for name, model in models.items():
        logger.info(f"[{state}] Training {name}...")
        try:
            model.train(train_df)
            preds = model.predict(steps=val_days)
            metrics = evaluate_predictions(val_df['sales'].values, preds)
            results[name] = metrics
            results[name]['preds'] = preds
            logger.info(f"[{state}] {name} metrics: {metrics}")
        except Exception as e:
            logger.error(f"[{state}] Model {name} failed: {e}")
            
    if not results:
        logger.error(f"[{state}] All models failed to train.")
        return None
        
    best_name, best_metrics = select_best_model(results)
    logger.info(f"[{state}] Best model selected: {best_name} with metrics {best_metrics}")
    
    # Generate visualization
    plot_model_comparison(state, results)
    plot_actual_vs_predicted(state, val_df['date'], val_df['sales'].values, results[best_name]['preds'], best_name)
    
    # Save best model
    best_model = models[best_name]
    model_dir = f"saved_models/{state}"
    os.makedirs(model_dir, exist_ok=True)
    model_path = f"{model_dir}/{best_name.lower()}_model"
    best_model.save(model_path)
    
    last_date = str(df['date'].max().date())
    
    update_model_registry(
        state=state,
        best_model_name=best_name,
        metrics={k: v for k, v in best_metrics.items() if k != 'preds'},
        model_path=model_path,
        last_date=last_date
    )
    
    return {
        "state": state,
        "best_model": best_name,
        "metrics": {k: v for k, v in best_metrics.items() if k != 'preds'},
        "all_results": {m: {k: v for k, v in res.items() if k != 'preds'} for m, res in results.items()}
    }

def run_training_pipeline(task_id: str, processed_df: pd.DataFrame) -> dict:
    try:
        task_manager.update_task(task_id, "running", 10.0, "Starting model training")
        
        states = processed_df['state'].unique()
        total_states = len(states)
        
        summary = []
        for idx, state in enumerate(states):
            state_df = processed_df[processed_df['state'] == state].copy()
            res = train_models_for_state(state, state_df)
            if res:
                summary.append(res)
            
            progress = 10.0 + ((idx + 1) / total_states) * 80.0
            task_manager.update_task(task_id, "running", progress, f"Trained {idx+1}/{total_states} states")
            
        task_manager.update_task(task_id, "completed", 100.0, "Training pipeline finished successfully")
        logger.info("Training pipeline finished successfully")
        
        # Save summary logic could be to a file, but since this runs async, 
        # the status task itself can hold a message. But for direct use, we just return it.
        return {"status": "success", "summary": summary}
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        task_manager.update_task(task_id, "failed", 0.0, f"Error: {str(e)}")
        return {"status": "failed", "error": str(e)}
