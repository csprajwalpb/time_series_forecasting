from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(title="Time Series Forecasting API", version="1.0")

app.include_router(router, prefix="/api/v1")

@app.get("/")
def health_check():
    return {"status": "healthy", "message": "Time Series Forecasting API is running."}
