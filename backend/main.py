import os
from datetime import datetime
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse
from kiteconnect import KiteConnect

app = FastAPI(title="KK NIFTY EARLY SIGNAL")

KITE_API_KEY = os.getenv("KITE_API_KEY", "").strip()
KITE_API_SECRET = os.getenv("KITE_API_SECRET", "").strip()

# The access token is held in memory. Render Free may restart/sleep,
# so complete Zerodha login again if the server loses this session.
_access_token = None
_user_id = None


def get_kite():
    if not KITE_API_KEY:
        raise HTTPException(status_code=503, detail="KITE_API_KEY is not configured.")
    if not _access_token:
        raise HTTPException(status_code=401, detail="Zerodha is not connected. Open /zerodha/login.")
    kite = KiteConnect(api_key=KITE_API_KEY)
    kite.set_access_token(_access_token)
    return kite


@app.get("/")
def home():
    return {"app": "KK NIFTY EARLY SIGNAL", "status": "Running"}


@app.get("/health")
def health():
    return {"status": "OK"}


@app.get("/zerodha/login")
def zerodha_login():
    if not KITE_API_KEY:
        raise HTTPException(status_code=503, detail="KITE_API_KEY is not configured.")
    return RedirectResponse(KiteConnect(api_key=KITE_API_KEY).login_url())


@app.get("/zerodha/callback")
def zerodha_callback(request: Request):
    global _access_token, _user_id

    request_token = request.query_params.get("request_token")
    if not request_token:
        raise HTTPException(status_code=400, detail="Missing request_token. Start at /zerodha/login.")

    if not KITE_API_KEY or not KITE_API_SECRET:
        raise HTTPException(status_code=503, detail="KITE_API_KEY or KITE_API_SECRET is not configured.")

    try:
        kite = KiteConnect(api_key=KITE_API_KEY)
        data = kite.generate_session(request_token, api_secret=KITE_API_SECRET)
        _access_token = data["access_token"]
        _user_id = data.get("user_id")
        return {
            "status": "connected",
            "message": "Zerodha login successful. Access token received for this server session.",
            "user_id": _user_id,
            "login_time": datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(),
        }
    except Exception:
        # Never expose API secrets, request tokens, or access tokens in errors.
        raise HTTPException(
            status_code=401,
            detail="Zerodha session exchange failed. Start a fresh login and check Render configuration/logs.",
        )


@app.get("/zerodha/status")
def zerodha_status():
    return {
        "connected": bool(_access_token),
        "user_id": _user_id if _access_token else None,
        "note": "Connection is stored in server memory and may reset when the free instance sleeps or restarts.",
    }


@app.get("/market/nifty/status")
def market_nifty_status():
    """Fetch the latest available NIFTY 50 index quote from Kite Connect."""
    kite = get_kite()
    try:
        result = kite.ltp(["NSE:NIFTY 50"])
        item = result.get("NSE:NIFTY 50")
        if not item:
            raise HTTPException(status_code=502, detail="Kite returned no NIFTY 50 quote.")

        return {
            "symbol": "NIFTY 50",
            "exchange": "NSE",
            "last_price": item.get("last_price"),
            "instrument_token": item.get("instrument_token"),
            "fetched_at": datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(),
            "data_source": "Zerodha Kite Connect",
            "note": "Latest available quote; updates depend on market hours and an active Kite session.",
        }
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=502,
            detail="Could not fetch NIFTY 50 quote. Check Kite session, API subscription, and Render logs.",
        )
        
