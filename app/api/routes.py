from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException
from typing import Dict, Any, List
import shutil
import os
from pathlib import Path

from app.utils.helpers import task_manager, get_best_model_info
from app.services.preprocessing import preprocess_data
from app.services.feature_engineering import engineer_features
from app.services.trainer import run_training_pipeline
from app.services.forecasting import generate_all_forecasts, generate_forecast

router = APIRouter()

def process_and_train(task_id: str, file_path: str):
    try:
        task_manager.update_task(task_id, "running", 5.0, "Preprocessing data")
        df = preprocess_data(file_path)
        
        task_manager.update_task(task_id, "running", 10.0, "Engineering features")
        engineered_df = engineer_features(df)
        
        run_training_pipeline(task_id, engineered_df)
    except Exception as e:
        task_manager.update_task(task_id, "failed", 0.0, str(e))
    finally:
        # Cleanup uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)

@router.post("/train")
async def train_models(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.endswith(('.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="Only Excel files are supported")
        
    # Save uploaded file
    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / file.filename
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    task_id = task_manager.create_task()
    background_tasks.add_task(process_and_train, task_id, str(file_path))
    
    return {"message": "Training started in background", "task_id": task_id}

@router.get("/status/{task_id}")
async def get_status(task_id: str):
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task

@router.get("/metrics")
async def get_metrics():
    registry_path = Path("saved_models/registry.json")
    if not registry_path.exists():
        raise HTTPException(status_code=404, detail="No models trained yet")
        
    import json
    with open(registry_path, "r") as f:
        registry = json.load(f)
        
    response = []
    for state, info in registry.items():
        response.append({
            "state": state,
            "best_model": info["best_model"],
            "metrics": info["metrics"]
        })
    return {"model_comparison": response}

@router.get("/forecast/{state}")
async def get_state_forecast(state: str):
    try:
        return generate_forecast(state, steps=56)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/forecast")
async def get_forecast(state: str = None, steps: int = 56):
    if state:
        try:
            return generate_forecast(state, steps)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    else:
        return generate_all_forecasts(steps)
