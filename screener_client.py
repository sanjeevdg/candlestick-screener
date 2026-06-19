import requests
import os

BASE_URL = "https://candlestick-screener.onrender.com"
API_KEY = os.getenv("INTERNAL_API_KEY")

def get_bars(symbols, tf="5Min"):
    r = requests.get(
        f"{BASE_URL}/api/bars",
        params={"symbols": ",".join(symbols), "tf": tf},
        headers={"X-API-Key": API_KEY} if API_KEY else None,
        timeout=30
    )

    r.raise_for_status()

    return r.text   # ← CSV or plain text