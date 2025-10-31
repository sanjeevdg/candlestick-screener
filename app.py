from flask import Flask, jsonify, request
import requests
from flask_cors import CORS
import os
import pandas as pd
import yfinance as yf
#from yfinance.scrapers.quote import quote as yf_quote
#from yahoo_fin import stock_info as si
#import yahoo_fin.stock_info as si
#from yahoo_fin.stock_info import _requests

from yahoo_fin import stock_info as si
from datetime import datetime
import pytz


app = Flask(__name__)
CORS(app)


def is_consolidating(df, percentage=2):
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



@app.route("/api/stock_extras")
def stock_extras():
    symbol = request.args.get("symbol")
    if not symbol:
        return jsonify({"error": "symbol param required"}), 400

    try:
        ticker = yf.Ticker(symbol)
        fast_info = getattr(ticker, "fast_info", {})

        # Get data fields (use getattr to handle missing fields safely)
        last_price = getattr(fast_info, "last_price", None)
        pre_price = getattr(fast_info, "pre_market_price", None)
        post_price = getattr(fast_info, "post_market_price", None)
        volume = getattr(fast_info, "last_volume", None) or getattr(fast_info, "volume", None)

        hist = ticker.history(period="5d", interval="1d")
        if hist.empty:
            return jsonify({"error": "no historical data"}), 404

        avg_vol = hist["Volume"].mean()
        prev_close = hist["Close"].iloc[-1]
        rel_vol = round(volume / avg_vol, 2) if (volume and avg_vol) else None

        # Determine which market data is available
        market_gap = None
        gap_type = None
        market_status = "regular"

        if pre_price and prev_close:
            market_gap = round((pre_price - prev_close) / prev_close * 100, 2)
            gap_type = "premarket"
            market_status = "premarket"
        elif post_price and prev_close:
            market_gap = round((post_price - prev_close) / prev_close * 100, 2)
            gap_type = "postmarket"
            market_status = "postmarket"

        # Add timestamp with timezone (US Eastern)
        tz = pytz.timezone("US/Eastern")
        timestamp = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z")

        return jsonify({
            "symbol": symbol,
            "relative_volume": rel_vol,
            "market_gap": market_gap,
            "gap_type": gap_type,
            "market_status": market_status,
            "price": last_price,
            "prev_close": prev_close,
            "timestamp": timestamp
        })

    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return jsonify({"error": str(e)}), 500



@app.route("/api/stock_extras2")
def get_stock_extras2():
    """Fetch relative volume and premarket gap using Yahoo's public quote API"""
    symbol = request.args.get("symbol")
    if not symbol:
        return jsonify({"error": "symbol parameter is required"}), 400

    try:
        url = f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={symbol}"

        # --- Proper browser-like headers ---
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://finance.yahoo.com/",
            "Origin": "https://finance.yahoo.com",
            "Connection": "keep-alive",
        }

        resp = requests.get(url, headers=headers, timeout=8)
        resp.raise_for_status()

        payload = resp.json()
        result = payload.get("quoteResponse", {}).get("result", [])
        if not result:
            return jsonify({"error": f"No data for symbol {symbol}"}), 404

        q = result[0]

        prev_close = q.get("regularMarketPreviousClose")
        curr_vol = q.get("regularMarketVolume")
        avg_vol = q.get("averageDailyVolume3Month") or q.get("averageDailyVolume10Day")
        pre_market = q.get("preMarketPrice")

        # --- Compute metrics ---
        relative_volume = round(curr_vol / avg_vol, 2) if curr_vol and avg_vol else None
        premarket_gap = (
            round(((pre_market - prev_close) / prev_close) * 100, 2)
            if pre_market and prev_close
            else None
        )

        return jsonify({
            "symbol": symbol,
            "relative_volume": relative_volume,
            "premarket_gap": premarket_gap,
            "pre_market_price": pre_market,
            "previous_close": prev_close,
            "current_volume": curr_vol,
            "average_volume": avg_vol,
        })

    except requests.exceptions.RequestException as e:
        print(f"Network error fetching {symbol}: {e}")
        return jsonify({"error": "Yahoo request failed"}), 502
    except Exception as e:
        print(f"Unexpected error for {symbol}: {e}")
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

