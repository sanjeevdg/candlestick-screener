import os
import json
import time
import requests

CACHE_FILE = "symbols_cache.json"
CACHE_EXPIRY = 24 * 3600  # 1 day
API_TOKEN = "69159f37bc76e8.77709234"
EXCHANGE_CODE = "US"

def get_symbols_from_eodhd():
    """Fetch and cache symbols from EODHD, filtering only 'Common Stock'."""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            cache = json.load(f)
            if time.time() - cache["timestamp"] < CACHE_EXPIRY:
                print(f"✅ Loaded {len(cache['symbols'])} symbols from cache")
                return cache["symbols"]

    print("🌐 Fetching symbols from EODHD...")
    url = f"https://eodhd.com/api/exchange-symbol-list/{EXCHANGE_CODE}?api_token={API_TOKEN}&fmt=json"
    data = requests.get(url).json()

    # ✅ Filter only valid, common stock tickers
    symbols = [
        item["Code"].split(".")[0]  # convert "AAPL.US" → "AAPL"
        for item in data
        if item.get("Type") == "Common Stock" and "." in item.get("Code", "")
    ]

    # ✅ Cache locally
    with open(CACHE_FILE, "w") as f:
        json.dump({"timestamp": time.time(), "symbols": symbols}, f)

    print(f"💾 Cached {len(symbols)} symbols to {CACHE_FILE}")
    return symbols
