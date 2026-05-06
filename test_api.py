import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import time
import os

BASE_URL = "http://localhost:8000/api/v1"

def create_test_data():
    print("Generating test data...")
    dates = [datetime(2025, 1, 1) + timedelta(days=i) for i in range(150)]
    
    data = []
    for state in ["Karnataka", "Maharashtra"]:
        for date in dates:
            # Generate somewhat realistic sales data with some noise
            base_sales = 1000 if state == "Karnataka" else 1500
            sales = base_sales + np.random.normal(0, 100)
            if date.weekday() >= 5: # weekend boost
                sales += 500
            data.append({"Date": date, "State": state, "Sales": max(0, sales)})
            
    df = pd.DataFrame(data)
    file_path = "test_data.xlsx"
    df.to_excel(file_path, index=False)
    print(f"Test data saved to {file_path}")
    return file_path

def test_train(file_path):
    print(f"\nTesting POST /train with {file_path}...")
    with open(file_path, "rb") as f:
        files = {"file": (os.path.basename(file_path), f, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
        response = requests.post(f"{BASE_URL}/train", files=files)
        
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.json().get("task_id")

def poll_status(task_id):
    print(f"\nPolling status for task {task_id}...")
    while True:
        response = requests.get(f"{BASE_URL}/status/{task_id}")
        data = response.json()
        status = data.get("status")
        progress = data.get("progress")
        message = data.get("message")
        
        print(f"Status: {status}, Progress: {progress}%, Message: {message}")
        if status in ["completed", "failed"]:
            break
        time.sleep(3)
    return status

def test_metrics():
    print("\nTesting GET /metrics...")
    response = requests.get(f"{BASE_URL}/metrics")
    print(f"Status Code: {response.status_code}")
    print(f"Response snippet: {str(response.json())[:200]}...")

def test_forecast(state):
    print(f"\nTesting GET /forecast/{state}...")
    response = requests.get(f"{BASE_URL}/forecast/{state}")
    print(f"Status Code: {response.status_code}")
    data = response.json()
    if "forecast" in data:
        print(f"Forecast contains {len(data['forecast'])} days.")
        print(f"First 2 days: {data['forecast'][:2]}")
    else:
        print(f"Error: {data}")

if __name__ == "__main__":
    file_path = create_test_data()
    task_id = test_train(file_path)
    
    if task_id:
        final_status = poll_status(task_id)
        if final_status == "completed":
            test_metrics()
            test_forecast("Karnataka")
            test_forecast("Maharashtra")
        else:
            print("Training failed, skipping metric and forecast tests.")
    
    # Cleanup
    if os.path.exists(file_path):
        os.remove(file_path)
