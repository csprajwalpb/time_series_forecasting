import uuid
from typing import Dict, Any, Optional
from datetime import datetime
import json
from pathlib import Path
from pydantic import BaseModel
import threading
from app.utils.logger import get_logger

logger = get_logger(__name__)

class TaskStatus(BaseModel):
    task_id: str
    status: str # "pending", "running", "completed", "failed"
    progress: float
    message: str
    created_at: str
    updated_at: str

class TaskManager:
    def __init__(self):
        self.tasks: Dict[str, TaskStatus] = {}
        self.lock = threading.Lock()

    def create_task(self) -> str:
        with self.lock:
            task_id = str(uuid.uuid4())
            now = datetime.now().isoformat()
            self.tasks[task_id] = TaskStatus(
                task_id=task_id,
                status="pending",
                progress=0.0,
                message="Task created",
                created_at=now,
                updated_at=now
            )
            return task_id

    def update_task(self, task_id: str, status: str, progress: float, message: str):
        with self.lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                task.status = status
                task.progress = progress
                task.message = message
                task.updated_at = datetime.now().isoformat()

    def get_task(self, task_id: str) -> Optional[TaskStatus]:
        with self.lock:
            return self.tasks.get(task_id)

task_manager = TaskManager()

REGISTRY_PATH = Path("saved_models/registry.json")

def initialize_registry():
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not REGISTRY_PATH.exists():
        with open(REGISTRY_PATH, "w") as f:
            json.dump({}, f)

def update_model_registry(state: str, best_model_name: str, metrics: Dict[str, float], model_path: str, last_date: str):
    initialize_registry()
    try:
        with open(REGISTRY_PATH, "r") as f:
            registry = json.load(f)
        
        registry[state] = {
            "best_model": best_model_name,
            "metrics": metrics,
            "model_path": model_path,
            "last_date": last_date,
            "updated_at": datetime.now().isoformat()
        }
        
        with open(REGISTRY_PATH, "w") as f:
            json.dump(registry, f, indent=4)
        logger.info(f"Model registry updated for state: {state}")
    except Exception as e:
        logger.error(f"Failed to update model registry: {e}")

def get_best_model_info(state: str) -> Optional[Dict[str, Any]]:
    initialize_registry()
    try:
        with open(REGISTRY_PATH, "r") as f:
            registry = json.load(f)
        return registry.get(state)
    except Exception as e:
        logger.error(f"Failed to read model registry: {e}")
        return None
