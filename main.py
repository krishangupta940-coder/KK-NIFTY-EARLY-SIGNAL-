
from fastapi import FastAPI
from datetime import datetime

app = FastAPI(
    title="KK NIFTY EARLY SIGNAL",
    description="Nifty 50 Early Signal API",
    version="1.0.0"
)

@app.get("/")
def home():
    return {
        "app": "KK NIFTY EARLY SIGNAL",
        "status": "Running",
        "message": "Welcome to KK Nifty Early Signal"
    }

@app.get("/signal")
def get_signal():
    return {
        "symbol": "NIFTY 50",
        "signal": "WAIT",
        "message": "Live market data not connected yet",
        "time": datetime.now().isoformat()
    }

@app.get("/health")
def health():
    return {"status": "OK"}
  
