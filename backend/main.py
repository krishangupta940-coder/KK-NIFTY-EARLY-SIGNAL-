
from fastapi import FastAPI

app = FastAPI(
    title="KK NIFTY EARLY SIGNAL",
    description="Nifty 50 Early Signal Backend",
    version="1.0.0"
)

@app.get("/")
def home():
    return {
        "app": "KK NIFTY EARLY SIGNAL",
        "status": "Running"
    }

@app.get("/health")
def health():
    return {
        "status": "OK"
    }
  
