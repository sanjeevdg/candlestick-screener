from flask import Flask, jsonify, Response, request
import requests
from flask_cors import CORS, cross_origin
import os
import pandas as pd
import yfinance as yf
from yfinance import EquityQuery
from concurrent.futures import ThreadPoolExecutor
#from yfinance.scrapers.quote import quote as yf_quote
#from yahoo_fin import stock_info as si
#import yahoo_fin.stock_info as si
#from yahoo_fin.stock_info import _requests
import threading
from yahoo_fin import stock_info as si
from datetime import datetime 
import pytz
import math
import time
from queue import Queue, Empty
import json

app = Flask(__name__)


ALLOWED_ORIGINS = [
    "https://sanjeevdg.github.io",
    "http://localhost:3000"
]

#//CORS(app, origins=ALLOWED_ORIGINS)
CORS(app, origins=[
    "http://localhost:3000",
    "https://sanjeevdg.github.io"
], resources={r"/api/*": {"origins": ["http://localhost:3000", "https://sanjeevdg.github.io"]}})
#CORS(app, origins=["http://localhost:3000","https://sanjeevdg.github.io"])


ALPHA_VANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "EALVYO7ECX58VA4T")

def is_consolidating(df, percentage=4):
    if len(df) < 15:
        return False

    recent_candlesticks = df.iloc[-15:]
    max_close = recent_candlesticks['Close'].max()
    min_close = recent_candlesticks['Close'].min()
    threshold = 1 - (percentage / 100)
    return min_close > (max_close * threshold)


def is_breaking_out(df, percentage=2.5):
    if len(df) < 16:
        return False

    last_close = df.iloc[-1]['Close']

    if is_consolidating(df.iloc[:-1], percentage=percentage):
        recent_closes = df.iloc[-16:-1]
        if last_close > recent_closes['Close'].max():
            return True

    return False


def get_company_info(symbol):
    """Fetch company name using yfinance"""
    try:
        info = yf.Ticker(symbol).info
        return info.get("shortName", symbol)
    except Exception:
        return symbol





@app.route("/api/sma", methods=["GET"])
def calculate_sma():
    symbols = request.args.get("symbols", "")
    sma_periods = [20, 50, 200]
    results = {}

    if not symbols:
        return jsonify({"error": "Please provide comma-separated symbols"}), 400

    symbol_list = [s.strip().upper() for s in symbols.split(",")]

    for symbol in symbol_list:
        try:
            df = yf.download(symbol, period="1y", progress=False, auto_adjust=True)
            if df.empty:
                results[symbol] = {"error": "No data found"}
                continue

            # --- Extract Close column robustly ---
            if isinstance(df.columns, pd.MultiIndex):
                # Try to locate ('Close', symbol)
                if ("Close", symbol) in df.columns:
                    close_data = df[("Close", symbol)]
                elif ("Adj Close", symbol) in df.columns:
                    close_data = df[("Adj Close", symbol)]
                elif "Close" in df.columns.get_level_values(0):
                    close_data = df["Close"].iloc[:, 0]
                else:
                    results[symbol] = {"error": "No Close column found"}
                    continue
            else:
                if "Close" in df.columns:
                    close_data = df["Close"]
                elif "Adj Close" in df.columns:
                    close_data = df["Adj Close"]
                else:
                    results[symbol] = {"error": "No Close or Adj Close column found"}
                    continue

            # --- Force to Series ---
            close_series = pd.Series(close_data).astype(float).dropna()
            df = pd.DataFrame({"Close": close_series})

            # --- Compute SMAs ---
            for period in sma_periods:
                df[f"SMA_{period}"] = df["Close"].rolling(window=period, min_periods=period).mean()

            if len(df) < max(sma_periods):
                results[symbol] = {"error": "Not enough data for SMA calculation"}
                continue

            curr = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else None

            entry = {"close": round(curr["Close"], 2)}
            for period in sma_periods:
                val = curr[f"SMA_{period}"]
                entry[f"sma_{period}"] = round(val, 2) if pd.notna(val) else None

            # --- Crossover signals ---
            def get_signal(short, long):
                if prev is None:
                    return "neutral"
                if (
                    prev[f"SMA_{short}"] < prev[f"SMA_{long}"]
                    and curr[f"SMA_{short}"] >= curr[f"SMA_{long}"]
                ):
                    return "bullish_cross"
                elif (
                    prev[f"SMA_{short}"] > prev[f"SMA_{long}"]
                    and curr[f"SMA_{short}"] <= curr[f"SMA_{long}"]
                ):
                    return "bearish_cross"
                return "neutral"

            entry["signal_20_50"] = get_signal(20, 50)
            entry["signal_50_200"] = get_signal(50, 200)

            results[symbol] = entry

        except Exception as e:
            results[symbol] = {"error": str(e)}

    return jsonify(results)



@app.route("/api/most_active_symbols_100")
def most_active_symbols_100():
    try:
        # Add count=100 to the query URL to get more results
        url = (
            "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
            "?scrIds=most_actives&count=100"
        )

        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json, text/plain, */*",
        }

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        quotes = data.get("finance", {}).get("result", [])[0].get("quotes", [])
        symbols = [q.get("symbol") for q in quotes if q.get("symbol")]

        if not symbols:
            raise ValueError("No symbols found in response")

        return jsonify({"count": len(symbols), "symbols": symbols})

    except Exception as e:
        print(f"Error fetching most active symbols: {e}")
        return jsonify({"error": "Could not fetch symbols"}), 500


@app.route("/api/screen_by_criteria", methods=["GET"])
def custom_screener():
    region = request.args.get("region", "us").lower()
    min_price = float(request.args.get("min_price", 0))
    max_price = float(request.args.get("max_price", 1_000_000))
    min_change = float(request.args.get("min_change", 0))
    min_day_vol = float(request.args.get("min_eodvolume", 0))
    max_day_vol = float(request.args.get("max_eodvolume", 1_000_000_000_000))
    sort_field = request.args.get("sort_field", "percentchange")
    sort_asc = request.args.get("sort_asc", "false").lower() == "true"
    limit = int(request.args.get("limit", 5))

    try:
        # 🧩 Build query dynamically
        criteria = [
            EquityQuery("eq", ["region", region]),
            EquityQuery("gt", ["dayvolume", min_day_vol]),
            EquityQuery("lt", ["dayvolume", max_day_vol]),
            EquityQuery("gt", ["eodprice", min_price]),
            EquityQuery("lt", ["eodprice", max_price])
        ]

        query = EquityQuery("and", criteria)

        # 🚀 Run the screener
        data = yf.screen(query, sortField=sort_field, sortAsc=sort_asc)
        quotes = data.get("quotes", [])

        # 🧾 Normalize
        df = pd.DataFrame(quotes)
        if df.empty:
            return jsonify([])

        df = df.rename(columns={
            "symbol": "symbol",
            "shortName": "name",
            "regularMarketPrice": "price",
            "regularMarketChangePercent": "percentchange",
            "regularMarketVolume": "volume"
        })

        df = df[["symbol", "name", "price", "percentchange", "volume"]].head(limit)
        results = df.to_dict(orient="records")
        return jsonify(clean_nans(results))

    except Exception as e:
        # 🧠 Detect rate limit / "Too Many Requests"
        error_message = str(e).lower()
        if "too many requests" in error_message or "rate limit" in error_message or "429" in error_message:
            print("❌ Screener error: Too Many Requests. Rate limited. Try after a while.")
            return jsonify({
                "error": "❌ Screener error: Too Many Requests. Rate limited. Try after a while."
            }), 429
        
        print("❌ Screener error:", e)
        return jsonify({"error": f"❌ Screener error: {str(e)}"}), 500



def clean_nans(obj):
    """Recursively replace NaN/inf/-inf with None for JSON safety."""
    if isinstance(obj, list):
        return [clean_nans(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: clean_nans(v) for k, v in obj.items()}
    elif isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj

'''
latest_quotes = {}
symbols_to_watch = []
ws = None

def start_yfinance_stream(symbols):
    global ws
    if ws:
        try:
            ws.close()
        except Exception:
            pass

    print(f"📡 Starting WebSocket for {symbols}")
    ws = yf.WebSocket()

    def handler(message):
        symbol = message.get("id")
        if symbol:
            latest_quotes[symbol] = {
                "symbol": symbol,
                "price": message.get("price"),
                "percentChange": message.get("changePercent"),
                "volume": message.get("dayVolume"),
                "timestamp": message.get("time"),
            }
            print(f"✅ {symbol} update:", latest_quotes[symbol])

    ws.subscribe(symbols)
    ws.listen(handler)

@app.route("/api/subscribe", methods=["POST"])
def subscribe_symbols():
    """Subscribe dynamically to user-selected tickers."""
    body = request.get_json()
    new_symbols = body.get("symbols", [])
    if not new_symbols:
        return jsonify({"error": "No symbols provided"}), 400

    global symbols_to_watch
    symbols_to_watch = [s.upper() for s in new_symbols]

    threading.Thread(
        target=start_yfinance_stream,
        args=(symbols_to_watch,),
        daemon=True
    ).start()

    return jsonify({"status": "subscribed", "symbols": symbols_to_watch})

@app.route("/api/quotes", methods=["GET"])
def get_quotes():
    """Return latest streamed data for the requested symbols."""
    symbols = request.args.get("symbols")
    if not symbols:
        return jsonify([])

    symbols = [s.strip().upper() for s in symbols.split(",")]

    data = [latest_quotes.get(s, {"symbol": s, "error": "no data yet"}) for s in symbols]
    return jsonify(data)
'''

# Global live quote cache and WebSocket reference

'''WORKING SNIPPET BWLOW
latest_quotes = {}
symbols_tracked = set()

def message_handler(msg):
    latest_quotes[msg["id"]] = msg

def start_ws(symbols):
    ws = yf.WebSocket()
    ws.subscribe(symbols)
    ws.listen(message_handler)

@app.route("/api/quotes")
def get_quotes():
    symbols = [s.strip().upper() for s in request.args.get("symbols", "").split(",") if s]

    # Start WebSocket thread if new symbols are added
    new_symbols = set(symbols) - symbols_tracked
    if new_symbols:
        symbols_tracked.update(new_symbols)
        threading.Thread(target=start_ws, args=(list(symbols_tracked),), daemon=True).start()

    # Return latest data (if available)
    results = []
    for sym in symbols:
        if sym in latest_quotes:
            results.append(latest_quotes[sym])
        else:
            results.append({"error": "no data yet", "symbol": sym})
    return jsonify(results)
'''




clients = []           # all SSE connections
tracked_symbols = set()
latest_data = {}        # { symbol: { ...last quote... } }

# === WebSocket handler ===
def message_handler(message):
    try:
        data = {
            "symbol": message.get("id"),
            "price": message.get("price"),
            "percentchange": message.get("change_percent"),
            "volume": message.get("day_volume"),
        }
        payload = json.dumps(data)  # ✅ Works now
        for conn in clients:
            conn.put(payload)
    except Exception as e:
        print("Error in message handler:", e)

def start_ws(symbols):
    """Start yfinance WebSocket for new symbols"""
    ws = yf.WebSocket()
    ws.subscribe(symbols)
    ws.listen(message_handler)

# === API Endpoints ===
@app.route("/api/add_symbol", methods=["POST"])
def add_symbol():
    """Frontend calls this to start tracking a new symbol"""
    body = request.get_json()
    symbol = body.get("symbol", "").upper()
    if not symbol:
        return jsonify({"error": "Missing symbol"}), 400

    if symbol not in tracked_symbols:
        tracked_symbols.add(symbol)
        threading.Thread(target=start_ws, args=([symbol],), daemon=True).start()

    return jsonify({"status": "subscribed", "symbol": symbol})

@app.route("/api/stream")
def stream():
    """Continuous stream of live updates (SSE)"""
    def event_stream():
        q = Queue()
        clients.append(q)
        try:
            while True:
                try:
                    data = q.get(timeout=15)  # wait max 15s for new data
                    yield f"data: {data}\n\n"
                except Empty:
                    # Send a heartbeat every 15s to keep connection alive
                    yield f"data: {{\"heartbeat\": {int(time.time())}}}\n\n"
        except GeneratorExit:
            clients.remove(q)

    response = Response(event_stream(), mimetype="text/event-stream")
    response.headers["Access-Control-Allow-Origin"] = "https://sanjeevdg.github.io"
    response.headers["Cache-Control"] = "no-cache"
    response.headers["Connection"] = "keep-alive"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    return response



@app.route("/api/latest")
def get_latest():
    """Optional endpoint to get current snapshot"""
    return jsonify(list(latest_data.values()))





'''
def get_quote(symbol):
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.fast_info or {}
        price = info.get("last_price")
        change = info.get("regular_market_change_percent")
        volume = info.get("regular_market_volume")

        # ✅ fallback to .info if any field is missing
        if not price or not volume:
            full = ticker.info or {}
            price = full.get("regularMarketPrice") or price
            change = full.get("regularMarketChangePercent") or change
            volume = full.get("regularMarketVolume") or volume

        name = (
            info.get("short_name")
            or full.get("shortName") if "full" in locals() else None
        )

        return {
            "symbol": symbol,
            "name": name,
            "price": round(price, 2) if price else None,
            "percentChange": round(change, 2) if change else None,
            "volume": int(volume) if volume else None
        }
    except Exception as e:
        print(f"❌ Error fetching {symbol}: {e}")
        return {"symbol": symbol, "error": str(e)}

@app.route("/api/quotes", methods=["GET"])
def quotes():
    symbols = request.args.get("symbols")
    if not symbols:
        return jsonify({"error": "symbols parameter required"}), 400

    symbols_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]

    with ThreadPoolExecutor(max_workers=8) as executor:
        data = list(executor.map(get_quote, symbols_list))

    # filter valid records
    results = [r for r in data if r.get("price") is not None]
    return jsonify(results)
'''


'''

def fetch_quote(symbol):
    url = (
        f"https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}"
    )
    try:
        resp = requests.get(url, timeout=5)
        data = resp.json().get("Global Quote", {})
        if not data:
            return {"symbol": symbol, "error": "no data"}
        price = float(data.get("05. price", 0))
        change_pct = float(data.get("10. change percent", "0%").strip("%"))
        volume = int(data.get("06. volume", 0))
        return {
            "symbol": symbol,
            "price": round(price, 2),
            "percentChange": round(change_pct, 2),
            "volume": volume,
        }
    except Exception as e:
        return {"symbol": symbol, "error": str(e)}

@app.route("/api/quotes", methods=["GET"])
def quotes():
    symbols = request.args.get("symbols")
    if not symbols:
        return jsonify({"error": "symbols parameter required"}), 400

    symbols_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]

    with ThreadPoolExecutor(max_workers=5) as executor:
        data = list(executor.map(fetch_quote, symbols_list))

    return jsonify(data)

'''





@app.route("/api/most_active_symbols")
def most_active_symbols():
    try:
        url = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved?scrIds=most_actives"
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json, text/plain, */*",
        }

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        quotes = data.get("finance", {}).get("result", [])[0].get("quotes", [])
        symbols = [q.get("symbol") for q in quotes if q.get("symbol")]

        if not symbols:
            raise ValueError("No symbols found in response")

        return jsonify({"symbols": symbols})

    except Exception as e:
        print(f"Error fetching most active symbols: {e}")
        return jsonify({"error": "Could not fetch symbols"}), 500



@app.route("/api/day_gainers")
def day_gainers():
    try:
        url = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved?scrIds=day_gainers"
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json, text/plain, */*",
        }

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        quotes = data.get("finance", {}).get("result", [])[0].get("quotes", [])
        symbols = [q.get("symbol") for q in quotes if q.get("symbol")]

        if not symbols:
            raise ValueError("No symbols found in response")

        return jsonify({"symbols": symbols})

    except Exception as e:
        print(f"Error fetching day gainers symbols: {e}")
        return jsonify({"error": "Could not fetch symbols"}), 500




@app.route("/api/day_losers")
def day_losers():
    try:
        url = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved?scrIds=day_losers"
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json, text/plain, */*",
        }

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        quotes = data.get("finance", {}).get("result", [])[0].get("quotes", [])
        symbols = [q.get("symbol") for q in quotes if q.get("symbol")]

        if not symbols:
            raise ValueError("No symbols found in response")

        return jsonify({"symbols": symbols})

    except Exception as e:
        print(f"Error fetching day losers symbols: {e}")
        return jsonify({"error": "Could not fetch symbols"}), 500



def clean_json(obj):
    """Recursively replace NaN and Infinity with None for valid JSON."""
    if isinstance(obj, dict):
        return {k: clean_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_json(i) for i in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    else:
        return obj



@app.route("/api/patterns", methods=["GET"])
def get_patterns():
    symbols_str = request.args.get("symbols")
    if not symbols_str:
        return jsonify({"error": "symbols param required"}), 400

    symbols = [s.strip().upper() for s in symbols_str.split(",") if s.strip()]

    try:
        data = yf.download(
            tickers=symbols,
            period="3mo",
            interval="1d",
            group_by="ticker",
            threads=True,
            auto_adjust=True,
            progress=False,
        )

        results = []

        for symbol in symbols:
            try:
                df = data[symbol] if isinstance(data.columns, pd.MultiIndex) else data
                if df.empty:
                    results.append({"symbol": symbol, "error": "no data"})
                    continue

                consolidating = bool(is_consolidating(df))
                breaking_out = bool(is_breaking_out(df))

                latest_close = float(df["Close"].iloc[-1])
                prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else latest_close
                percent_change = round(((latest_close - prev_close) / prev_close) * 100, 2)

                results.append({
                    "symbol": symbol,
                    "latest_close": round(latest_close, 2),
                    "percent_change": percent_change,
                    "consolidating": consolidating,
                    "breaking_out": breaking_out
                })
            except Exception as e:
                results.append({
                    "symbol": symbol,
                    "error": str(e)
                })

        safe_results = clean_json(results)         

        return jsonify(safe_results)

    except Exception as e:
        print(f"Error fetching patterns: {e}")
        return jsonify({"error": str(e)}), 500


YAHOO_URL = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"

@app.route("/api/most_actives")
def most_actives():
    try:
        # Try yfinance helper if available
        try:
            tickers = yf.get_day_most_active()
            if tickers is not None and len(tickers) > 0:
                results = [
                    {
                        "symbol": t.get("symbol"),
                        "name": t.get("shortName") or t.get("longName") or "N/A",
                        "price": t.get("regularMarketPrice"),
                        "change": t.get("regularMarketChangePercent"),
                        "volume": t.get("regularMarketVolume")
                    }
                    for t in tickers
                ]
                return jsonify({"data": results})
        except Exception:
            # fallback to manual fetch
            pass

        # Manual Yahoo Finance Screener API
        params = {
            "count": 100,
            "scrIds": "most_actives",
        }

        response = requests.get(YAHOO_URL, params=params, timeout=10)

        if response.status_code != 200:
            return jsonify({
                "error": f"Yahoo API HTTP {response.status_code}",
                "body": response.text[:200]  # show first part of response
            }), 502

        # Ensure JSON is valid
        try:
            data = response.json()
        except ValueError:
            return jsonify({
                "error": "Invalid JSON from Yahoo API",
                "body": response.text[:200]
            }), 502

        # Extract quotes safely
        quotes = data.get("finance", {}).get("result", [{}])[0].get("quotes", [])
        if not quotes:
            return jsonify({"error": "No quotes found in Yahoo response"}), 502

        results = [
            {
                "symbol": q.get("symbol"),
                "name": q.get("shortName") or q.get("longName") or "N/A",
                "price": q.get("regularMarketPrice"),
                "change": q.get("regularMarketChangePercent"),
                "volume": q.get("regularMarketVolume"),
            }
            for q in quotes
            if q.get("symbol")
        ]

        return jsonify({"data": results})

    except Exception as e:
        print("Error in Python fallback:", e)
        return jsonify({"error": str(e)}), 500



@app.route("/api/breakouts")
def get_breakouts():
    print("🚀 Hit /api/breakouts")
    results = []

    base_dir = "datasets/daily"
    if not os.path.exists(base_dir):
        print("❌ datasets/daily folder not found")
        return jsonify({"error": "datasets/daily folder not found"}), 404

    for filename in os.listdir(base_dir):
        print("🗂 Processing:", filename)
        if not filename.endswith(".csv"):
            continue

        path = os.path.join(base_dir, filename)
        df = pd.read_csv(path)
        print("✅ Read file", path, "rows:", len(df))

        if df.empty or len(df) < 2:
            continue

        symbol = filename.replace(".csv", "")
        company_name = get_company_info(symbol)

        # Latest candle
        latest = df.iloc[-1]
        prev_close = df.iloc[-2]["Close"] if len(df) >= 2 else latest["Close"]

        # Compute derived metrics
        sell = round(latest["Open"], 2)
        buy = round(latest["Close"], 2)
        high = round(latest["High"], 2)
        low = round(latest["Low"], 2)
        close = round(latest["Close"], 2)
        change = round(buy - prev_close, 2)
        pct_change = round((change / prev_close) * 100, 2) if prev_close else 0

        chart_url = f"https://finviz.com/chart.ashx?t={symbol}&ty=c&ta=1&p=d&s=l"

        if is_consolidating(df):
            status = "consolidating"
        elif is_breaking_out(df):
            status = "breaking_out"
        else:
            status = "neutral"

        results.append({
            "company": company_name,
            "symbol": symbol,
            "sell": sell,
            "buy": buy,
            "high": high,
            "low": low,
            "close": close,
            "change": change,
            "%change": pct_change,
            "status": status,
            "chart": chart_url
        })

    return jsonify(results)


@app.route("/ping")
def ping():
    return jsonify({"msg": "pong"})




'''
@app.route("/api/stock_extras")
def stock_extras():
    symbol = request.args.get("symbol")
    if not symbol:
        return jsonify({"error": "symbol param required"}), 400

    try:
        ticker = yf.Ticker(symbol)

        # 1️⃣ Get fast_info (lightweight)
        fast_info = getattr(ticker, "fast_info", {})
        last_price = getattr(fast_info, "last_price", None)
        pre_price = getattr(fast_info, "pre_market_price", None)
        volume = getattr(fast_info, "last_volume", None) or getattr(fast_info, "volume", None)

        # 2️⃣ Get 5-day history for averages
        hist = ticker.history(period="5d", interval="1d")
        if hist.empty:
            return jsonify({"error": "no historical data"}), 404

        avg_vol = hist["Volume"].mean()
        prev_close = hist["Close"].iloc[-1]

        # 3️⃣ Compute metrics
        rel_vol = round(volume / avg_vol, 2) if (volume and avg_vol) else None
        pre_gap = round((pre_price - prev_close) / prev_close * 100, 2) if (pre_price and prev_close) else None

        return jsonify({
            "symbol": symbol,
            "relative_volume": rel_vol,
            "premarket_gap": pre_gap,
            "price": last_price,
            "prev_close": prev_close,
        })

    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return jsonify({"error": str(e)}), 500
'''

def get_market_status():
    """Determine U.S. market status based on time (Eastern)."""
    tz = pytz.timezone("US/Eastern")
    now = datetime.now(tz)
    open_t = time(9, 30)
    close_t = time(16, 0)

    if now.weekday() >= 5:
        return "closed"
    elif now.time() < open_t:
        return "premarket"
    elif now.time() > close_t:
        return "postmarket"
    else:
        return "regular"


@app.route("/api/stock_extras")
def stock_extras():
    symbol = request.args.get("symbol")
    if not symbol:
        return jsonify({"error": "symbol param required"}), 400

    try:
        ticker = yf.Ticker(symbol)
        market_status = get_market_status()

        # Fetch 1d minute-level data to get near-live price
        hist_1m = ticker.history(period="1d", interval="1m")

        if hist_1m.empty:
            return jsonify({"error": "no intraday data"}), 404

        last_price = hist_1m["Close"].iloc[-1]
        volume_today = hist_1m["Volume"].sum()

        # Compute average volume from last 5 trading days
        hist_5d = ticker.history(period="5d", interval="1d")
        avg_vol = hist_5d["Volume"].mean() if not hist_5d.empty else None
        prev_close = hist_5d["Close"].iloc[-2] if len(hist_5d) > 1 else None

        relative_volume = (
            round(volume_today / avg_vol, 2)
            if avg_vol and volume_today
            else None
        )

        # Try to detect premarket / postmarket gap
        fast_info = getattr(ticker, "fast_info", {})
        pre_price = getattr(fast_info, "pre_market_price", None)
        post_price = getattr(fast_info, "post_market_price", None)

        gap_type, market_gap = None, None

        if market_status == "premarket" and pre_price and prev_close:
            market_gap = round((pre_price - prev_close) / prev_close * 100, 2)
            gap_type = "premarket"
        elif market_status == "postmarket" and post_price and prev_close:
            market_gap = round((post_price - prev_close) / prev_close * 100, 2)
            gap_type = "postmarket"

        tz = pytz.timezone("US/Eastern")
        timestamp = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z")

        return jsonify({
            "symbol": symbol,
            "price": float(last_price),
            "prev_close": float(prev_close) if prev_close else None,
            "relative_volume": relative_volume,
            "market_gap": market_gap,
            "gap_type": gap_type,
            "market_status": market_status,
            "timestamp": timestamp
        })

    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return jsonify({"error": str(e)}), 500





def fetch_screener(scr_id="most_actives"):
    """Fetch screener data from Yahoo Finance predefined lists"""
    url = f"https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved?scrIds={scr_id}"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://finance.yahoo.com/",
    }

    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    data = response.json()
    quotes = data["finance"]["result"][0]["quotes"]

    def safe_value(v):
        if isinstance(v, dict):
            return v.get("raw", v.get("fmt", None))
        return v

    rows = []
    for q in quotes:
        rows.append({
            "symbol": q.get("symbol"),
            "name": q.get("shortName"),
            "price": safe_value(q.get("regularMarketPrice")),
            "change": safe_value(q.get("regularMarketChange")),
            "percent_change": safe_value(q.get("regularMarketChangePercent")),
            "volume": safe_value(q.get("regularMarketVolume")),
        })

    return rows

@app.route("/api/screener")
def get_screener():
    scr_type = request.args.get("type", "most_actives")
    data = fetch_screener(scr_type)
    return jsonify({"type": scr_type, "data": data})


@app.route("/api/top_stocks")
def get_top_stocks():
    try:
        # Step 1: Read symbol list (no header, e.g. "AAPL,Apple Inc.")
        with open("datasets/symbols_cleaned.csv", "r") as f:
            symbols = [line.strip().split(",")[0] for line in f.readlines() if line.strip()]

        # Step 2: Limit or batch into chunks of 100
        batch_size = 100
        top_stocks = []

        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            print(f"📊 Fetching batch {i // batch_size + 1}: {len(batch)} symbols")

            # Step 3: Download batch data
            data = yf.download(
                tickers=" ".join(batch),
                period="1d",
                group_by="ticker",
                threads=True,
                progress=False
            )

            # Step 4: Handle multi-ticker vs single-ticker response
            if isinstance(data.columns, pd.MultiIndex):
                for symbol in batch:
                    try:
                        stock_data = data[symbol].iloc[-1]
                        top_stocks.append({
                            "symbol": symbol,
                            "open": round(stock_data["Open"], 2),
                            "high": round(stock_data["High"], 2),
                            "low": round(stock_data["Low"], 2),
                            "close": round(stock_data["Close"], 2),
                            "volume": int(stock_data["Volume"]),
                        })
                    except Exception as e:
                        print(f"⚠️ Skipping {symbol}: {e}")
            else:
                # Single ticker case (if batch has only one)
                try:
                    stock_data = data.iloc[-1]
                    top_stocks.append({
                        "symbol": batch[0],
                        "open": round(stock_data["Open"], 2),
                        "high": round(stock_data["High"], 2),
                        "low": round(stock_data["Low"], 2),
                        "close": round(stock_data["Close"], 2),
                        "volume": int(stock_data["Volume"]),
                    })
                except Exception as e:
                    print(f"⚠️ Single-ticker error: {e}")

        print(f"✅ Total stocks fetched: {len(top_stocks)}")
        return jsonify(top_stocks)

    except Exception as e:
        print(f"❌ Error in /api/top_stocks: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

