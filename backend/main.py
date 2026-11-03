import os
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse
from kiteconnect import KiteConnect

APP_NAME = "KK NIFTY EARLY SIGNAL"

app = FastAPI(
    title=APP_NAME,
    description="Market-analysis backend and Zerodha Kite Connect login callback.",
    version="1.1.0",
)

# The access token is intentionally kept in process memory for this starter version.
# Kite access tokens are short-lived (normally expire at the end of the trading day).
# A production deployment should persist them in a secure database or secret store.
_access_token: Optional[str] = None


def get_kite_client() -> KiteConnect:
    api_key = os.getenv("KITE_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="KITE_API_KEY is not configured")
    return KiteConnect(api_key=api_key)


@app.get("/")
def home():
    return {
        "app": APP_NAME,
        "status": "Running",
        "message": "Backend is live. Open /health to check service status.",
    }


@app.get("/health")
def health():
    return {"status": "OK"}


@app.get("/zerodha/login")
def zerodha_login():
    """Redirect the user to the official Kite login page."""
    kite = get_kite_client()
    return RedirectResponse(url=kite.login_url(), status_code=302)


@app.get("/zerodha/callback")
def zerodha_callback(
    request_token: Optional[str] = Query(default=None),
    status: Optional[str] = Query(default=None),
):
    """Exchange the one-time request_token for an access_token."""
    global _access_token

    if status == "error":
        raise HTTPException(status_code=401, detail="Zerodha login was not completed")

    if not request_token:
        raise HTTPException(status_code=400, detail="Missing request_token")

    api_key = os.getenv("KITE_API_KEY")
    api_secret = os.getenv("KITE_API_SECRET")
    if not api_key or not api_secret:
        raise HTTPException(status_code=503, detail="KITE_API_KEY or KITE_API_SECRET is not configured")

    try:
        kite = KiteConnect(api_key=api_key)
        session = kite.generate_session(request_token, api_secret=api_secret)
        # Keep only the token needed for subsequent API calls; never return the secret.
        _access_token = session["access_token"]
        kite.set_access_token(_access_token)
        return {
            "status": "connected",
            "message": "Zerodha login successful. Access token received for this server session.",
            "user_id": session.get("user_id"),
            "login_time": session.get("login_time"),
        }
    except Exception as exc:
        # Avoid returning request tokens, API secrets, or full upstream error payloads.
        raise HTTPException(status_code=401, detail="Zerodha session exchange failed. Check app settings and server logs.") from exc


@app.get("/zerodha/status")
def zerodha_status():
    return {
        "connected": _access_token is not None,
        "note": "Connection is stored in server memory and may reset when the free instance sleeps or restarts.",
    }


@app.get("/market/nifty/status")
def nifty_status():
    """Placeholder until the paid Kite Connect market-data plan is enabled."""
    return {
        "available": False,
        "instrument": "NIFTY 50",
        "message": "Live market data endpoint will be enabled after Kite Connect market-data access is configured.",
    }
