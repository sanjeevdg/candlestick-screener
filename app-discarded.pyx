from flask import Flask, jsonify, Response, request
import requests
import warnings
from flask_cors import CORS, cross_origin
import os
from flask_socketio import SocketIO, emit
import pandas as pd
import talib
#from tradingview_scraper.symbols.stream import RealTimeData
#from tradingview_scraper.analysis import Analysis
from tradingview_scraper.symbols.news import NewsScraper
from tradingview_scraper.symbols.technicals import Indicators
from tradingview_scraper.symbols.stream import RealTimeData
from alpaca.data.live import StockDataStream
from alpaca.data.enums import DataFeed

from tradingview_scraper.symbols.screener import Screener
from tradingview_scraper.symbols.market_movers import MarketMovers
from tvDatafeed import TvDatafeed
from tvDatafeed import Interval as TVInterval
from tradingview_screener import Query, Column 
from tradingview_ta import TA_Handler, Interval, Exchange
from ta.momentum import RSIIndicator
from ta.trend import MACD


import yfinance as yf
from yfinance import EquityQuery
from concurrent.futures import ThreadPoolExecutor, as_completed
#from yfinance.scrapers.quote import quote as yf_quote
#from yahoo_fin import stock_info as si
#import yahoo_fin.stock_info as si
#from yahoo_fin.stock_info import _requests
import threading
import asyncio
import aiohttp
from yahoo_fin import stock_info as si
from datetime import datetime, timedelta, time as dt_time, timezone
#from datetime import datetime, timedelta
import pytz
import math
import time
from queue import Queue, Empty
import json
import numpy as np
from io import StringIO
import ssl
from alpaca.data.historical import StockHistoricalDataClient

from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass
from alpaca.data.historical.screener import ScreenerClient
from alpaca.data.requests import MostActivesRequest



app = Flask(__name__)



socketio = SocketIO(app, cors_allowed_origins="*")

warnings.filterwarnings("ignore", category=FutureWarning)

ALLOWED_ORIGINS = [
    "https://sanjeevdg.github.io",
    "http://localhost:3000"
]

FINNHUB_API_KEY = "d3nr05hr01qtm4jdum8gd3nr05hr01qtm4jdum90"  # <-- replace with your own
CACHE_SP500 = "gainers_sp500_cache.json"
CACHE_NASDAQ100 = "gainers_nasdaq100_cache.json"
CACHE_TTL = 10 * 3600  # 1 hour
BATCH_SIZE = 50
MAX_THREADS = 8

col = Column  # alias for readability
#print(f"✅ Loaded cache from {Scanner.names()}")


alpaca_stream = None
alpaca_loop = None
alpaca_thread = None

active_symbols = set()
symbol_clients = {}


# --------------------------------
# Initialize scrapers (reuse them)
# --------------------------------
news_scraper = NewsScraper(
    export_result=False,   # we return JSON ourselves
    export_type="json"
)

tv_screener = Screener()

tv = TvDatafeed()
#'sanjeev_dasgupta','1QazxsW234!@#$' 
#ta_scraper = TechnicalAnalysisScraper()
# Create MarketMovers instance (JSON export)
market_movers = MarketMovers(
    export_result=True,
    export_type="json"
)
#//CORS(app, origins=ALLOWED_ORIGINS)
CORS(app, origins=[
    "http://localhost:3000",
    "https://sanjeevdg.github.io"
], resources={r"/api/*": {"origins": ["http://localhost:3000", "https://sanjeevdg.github.io"]}})
#CORS(app, origins=["http://localhost:3000","https://sanjeevdg.github.io"])




data_client = StockHistoricalDataClient(
    "PKC7D4XB4OTV2VDEFUF5BRL33P",
    "DZxMp2TBaqQWZH3seGm7ZSWbHjw25xLBD7ZGrG4F4GaF"
)


trading_client = TradingClient(
    api_key="PKC7D4XB4OTV2VDEFUF5BRL33P",
    secret_key="DZxMp2TBaqQWZH3seGm7ZSWbHjw25xLBD7ZGrG4F4GaF",
    paper=True
)

screener_client = ScreenerClient(
    api_key="PKC7D4XB4OTV2VDEFUF5BRL33P",
    secret_key="DZxMp2TBaqQWZH3seGm7ZSWbHjw25xLBD7ZGrG4F4GaF"    
)

EXCHANGE_CURRENCY_MAP = {
    "NSE": "INR",
    "BSE": "INR",
    "NASDAQ": "USD",
    "NYSE": "USD",
    "AMEX": "USD",
    "FOREX": "USD",
    "BINANCE": "USDT"
}

# Render.com Fix - Prevent urllib3 TLS recursion on Python 3.13
try:
    ssl.SSLContext.minimum_version = ssl.TLSVersion.TLSv1
except Exception:
    pass

def log(msg):
    print(f"[{datetime.utcnow().strftime('%H:%M:%S')}] {msg}", flush=True)

FINNHUB_API_KEY = "d3nr05hr01qtm4jdum8gd3nr05hr01qtm4jdum90"  # 🔑 Replace with your Finnhub key
FINNHUB_BASE = "https://finnhub.io/api/v1"


#finnhub_client = finnhub.Client(api_key='d3nr05hr01qtm4jdum8gd3nr05hr01qtm4jdum90')

ALPACA_KEY = "PKC7D4XB4OTV2VDEFUF5BRL33P"
ALPACA_SECRET = "DZxMp2TBaqQWZH3seGm7ZSWbHjw25xLBD7ZGrG4F4GaF"

latest_prices = {}
active_symbols = set()

_alpaca_stream = None
_alpaca_thread = None
_alpaca_started = False
_lock = threading.Lock()


WATCHLIST_FILE = "watchlist.txt"



def normalize_symbol(symbol):
    # "NASDAQ:AVAV" → "AVAV"
    return symbol.split(":")[-1].upper()

async def handle_quote(q):
    latest_prices[q.symbol] = {
        "price": q.ask_price or q.bid_price,
        "timestamp": q.timestamp
    }
    print("📈 QUOTE:", q.symbol, latest_prices[q.symbol]["price"])    

async def handle_bar(bar):
    socketio.emit("realtime_bar", {
        "symbol": bar.symbol,
        "time": int(bar.timestamp.timestamp()),
        "open": bar.open,
        "high": bar.high,
        "low": bar.low,
        "close": bar.close,
        "volume": bar.volume
    })


@app.route("/api/watchlist", methods=["POST"])
def add_to_watchlist():
    data = request.get_json(silent=True) or {}
    symbol = data.get("symbol")

    if not symbol:
        return jsonify({"error": "symbol required"}), 400

    symbols = read_watchlist()
    symbols.add(symbol.upper())
    write_watchlist(symbols)

    subscribe_watchlist_quotes()

    return jsonify({"success": True})

@app.route("/api/watchlist/<symbol>", methods=["DELETE"])
def remove_from_watchlist(symbol):
    symbols = read_watchlist()
    symbols.discard(symbol.upper())
    write_watchlist(symbols)

    subscribe_watchlist_quotes()

    return jsonify({"success": True})

@socketio.on("subscribe")
def on_subscribe(data):
    symbol = data["symbol"]
    sid = request.sid

    symbol_clients.setdefault(symbol, set()).add(sid)
    subscribe_bar(symbol)

@socketio.on("unsubscribe")
def on_unsubscribe(data):
    symbol = data["symbol"]
    sid = request.sid

    symbol_clients[symbol].discard(sid)

    if not symbol_clients[symbol]:
        unsubscribe_bar(symbol)
        del symbol_clients[symbol]

@socketio.on("disconnect")
def on_disconnect():
    sid = request.sid

    for symbol, clients in list(symbol_clients.items()):
        if sid in clients:
            clients.remove(sid)

            if not clients:
                unsubscribe_bar(symbol)
                del symbol_clients[symbol]



def read_watchlist():
    if not os.path.exists(WATCHLIST_FILE):
        return set()

    with open(WATCHLIST_FILE, "r") as f:
        return set(
            line.strip().upper()
            for line in f
            if line.strip()
        )

def write_watchlist(symbols):
    with open(WATCHLIST_FILE, "w") as f:
        for s in sorted(symbols):
            f.write(f"{s}\n")


@app.route("/api/watchlist", methods=["GET"])
def get_watchlist():
    symbols = sorted(read_watchlist())

    return jsonify({
        "count": len(symbols),
        "data": [{"symbol": s} for s in symbols]
    })







@app.route("/api/watchlist/prices")
def watchlist_prices():
    symbols = request.args.get("symbols", "").split(",")

    data = []
    for raw in symbols:
        symbol = normalize_symbol(raw)
        p = latest_prices.get(symbol)

        if symbol not in latest_prices:
            continue

        if p:
            data.append({
                "symbol": raw,      # keep original for frontend match
                "price": p["price"]
            })

    return jsonify(data)






async def handle_bar(bar):
    socketio.emit("realtime_bar", {
        "symbol": bar.symbol,
        "time": int(bar.timestamp.timestamp()),
        "open": bar.open,
        "high": bar.high,
        "low": bar.low,
        "close": bar.close,
        "volume": bar.volume
    })

def start_alpaca_stream_once():
    global alpaca_stream, alpaca_loop

    alpaca_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(alpaca_loop)

    alpaca_stream = StockDataStream(
        "PKC7D4XB4OTV2VDEFUF5BRL33P",
        "DZxMp2TBaqQWZH3seGm7ZSWbHjw25xLBD7ZGrG4F4GaF",
        feed=DataFeed.IEX
    )

    alpaca_loop.run_until_complete(alpaca_stream.run())

def ensure_stream_running():
    global alpaca_thread

    if alpaca_thread and alpaca_thread.is_alive():
        return

    alpaca_thread = threading.Thread(
        target=start_alpaca_stream_once,
        daemon=True
    )
    alpaca_thread.start()

def subscribe_symbol(symbol):
    ensure_stream_running()

    if symbol in active_symbols:
        return

    active_symbols.add(symbol)

    alpaca_loop.call_soon_threadsafe(
        alpaca_stream.subscribe_bars,
        handle_bar,
        symbol
    )

def unsubscribe_symbol(symbol):
    if symbol not in active_symbols:
        return

    active_symbols.remove(symbol)

    alpaca_loop.call_soon_threadsafe(
        alpaca_stream.unsubscribe_bars,
        symbol
    )

def stop_alpaca_stream(symbol):
    if symbol not in stream_tasks:
        return

    loop = stream_loops[symbol]
    task = stream_tasks[symbol]

    def cancel():
        task.cancel()

    loop.call_soon_threadsafe(cancel)

    del active_streams[symbol]
    del stream_tasks[symbol]
    del stream_loops[symbol]


@socketio.on("subscribe")
def on_subscribe(data):
    symbol = data["symbol"]
    sid = request.sid

    symbol_clients.setdefault(symbol, set()).add(sid)
    subscribe_symbol(symbol)


@socketio.on("unsubscribe")
def on_unsubscribe(data):
    symbol = data["symbol"]
    sid = request.sid

    symbol_clients[symbol].discard(sid)

    if not symbol_clients[symbol]:
        unsubscribe_symbol(symbol)
        del symbol_clients[symbol]


@socketio.on("disconnect")
def on_disconnect():
    sid = request.sid

    for symbol, clients in list(symbol_clients.items()):
        if sid in clients:
            clients.remove(sid)

            if not clients:
                unsubscribe_symbol(symbol)
                del symbol_clients[symbol]


def get_market_status():
    clock = trading_client.get_clock()
    return {
        "is_open": clock.is_open,
        "next_open": clock.next_open.isoformat(),
        "next_close": clock.next_close.isoformat(),
        "timestamp": clock.timestamp.isoformat()
    }


def get_assets_tradeable(symbols):
    if not symbols:
        return {}

    req = GetAssetsRequest(
        asset_class=AssetClass.US_EQUITY,
        symbols=symbols
    )

    assets = trading_client.get_all_assets(req)

    asset_map = {}
    for a in assets:
        asset_map[a.symbol] = {
            "tradable": a.tradable,
            "shortable": a.shortable,
            "fractionable": a.fractionable,
            "status": a.status,
            "exchange": a.exchange
        }

    return asset_map


#ALPHA_VANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "EALVYO7ECX58VA4T")

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





def load_cache(cache_file):
    if not os.path.exists(cache_file):
        return {"timestamp": 0, "results": []}
    try:
        with open(cache_file, "r") as f:
            data = json.load(f)
        if time.time() - data.get("timestamp", 0) < CACHE_TTL:
            print(f"✅ Loaded cache from {cache_file}")
            return data
        else:
            print(f"♻️ Cache expired for {cache_file}")
            return {"timestamp": 0, "results": []}
    except Exception:
        return {"timestamp": 0, "results": []}

def load_cache2(cache_file):
    if not os.path.exists(cache_file):
        return {"timestamp": 0, "results": []}
    try:
        with open(cache_file, "r") as f:
            data = json.load(f)
            print(f"✅ Loaded cache2 from {cache_file} -- with data -- {data} ")
            return data        
    except Exception:
        return {"timestamp": 0, "results": []}

def save_cache(cache_file, data):
    try:
        with open(cache_file, "w") as f:
            json.dump(data, f)
        print(f"💾 Saved cache to {cache_file}")
    except Exception as e:
        print(f"⚠️ Failed to save cache {cache_file}: {e}")


def get_date_range(months=8):
    end = datetime.utcnow()
    start = end - timedelta(days=months * 30)
    return start, end



INTERVAL_MAP = {
    "1m": Interval.INTERVAL_1_MINUTE,
    "1h": Interval.INTERVAL_1_HOUR,
    "4h": Interval.INTERVAL_4_HOURS,
    "1d": Interval.INTERVAL_1_DAY,
    "1w": Interval.INTERVAL_1_WEEK,
    "1M": Interval.INTERVAL_1_MONTH,
}

@app.route("/api/fchart2", methods=["GET"])
def fchart2():
    try:
        raw_symbol = request.args.get("symbol")
        if not raw_symbol:
            return jsonify({"error": "Symbol required"}), 400

        # ---------------- SYMBOL PARSING ----------------
        if ":" in raw_symbol:
            exchange, symbol = raw_symbol.split(":")
        else:
            exchange, symbol = "NASDAQ", raw_symbol

        exchange = exchange.upper()
        symbol = symbol.upper()

        # ---------------- DATE RANGE ----------------
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=420)

        request_params = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
            limit=300,
            feed=DataFeed.IEX,
        )

        bars = data_client.get_stock_bars(request_params).df

        if bars.empty:
            return jsonify({
                "meta": {
                    "symbol": symbol,
                    "exchange": exchange,
                    "currency": EXCHANGE_CURRENCY_MAP.get(exchange, "USD"),
                },
                "quotes": [],
                "indicators": {"rsi": [], "macd": []},
            })

        # Alpaca → normalize dataframe
        bars = bars.xs(symbol).reset_index()

        # ---------------- QUOTES ----------------
        quotes = []
        closes = []

        for _, row in bars.iterrows():
            date_str = row["timestamp"].strftime("%Y-%m-%d")
            quotes.append({
                "date": date_str,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": int(row["volume"]),
            })
            closes.append(float(row["close"]))

        closes_np = np.array(closes, dtype=float)

        # ---------------- INDICATORS (TA-Lib) ----------------

        # RSI (14)
        rsi_raw = talib.RSI(closes_np, timeperiod=14)
        rsi = [
            {
                "time": quotes[i]["date"],
                "value": round(float(rsi_raw[i]), 2),
            }
            for i in range(len(rsi_raw))
            if not np.isnan(rsi_raw[i])
        ]

        sma_raw = talib.SMA(closes_np, timeperiod=20)

        sma = [
            {
                "time": quotes[i]["date"],
                "value": round(float(sma_raw[i]), 4),
            }
            for i in range(len(sma_raw))
            if not np.isnan(sma_raw[i])
        ]




        # MACD (12,26,9)
        macd_raw, signal_raw, hist_raw = talib.MACD(
            closes_np,
            fastperiod=12,
            slowperiod=26,
            signalperiod=9,
        )

        macd = [
            {
                "time": quotes[i]["date"],
                "macd": round(float(macd_raw[i]), 4),
                "signal": round(float(signal_raw[i]), 4),
                "hist": round(float(hist_raw[i]), 4),
            }
            for i in range(len(macd_raw))
            if not (
                np.isnan(macd_raw[i]) or
                np.isnan(signal_raw[i]) or
                np.isnan(hist_raw[i])
            )
        ]

        # ---------------- META ----------------
        currency = EXCHANGE_CURRENCY_MAP.get(exchange, "USD")

        meta = {
            "symbol": symbol,
            "exchange": exchange,
            "currency": currency,
            "regularMarketPrice": quotes[-1]["close"] if quotes else None,
            "longName": symbol,
        }

        # ---------------- RESPONSE ----------------
        return jsonify({
            "meta": meta,
            "quotes": quotes,
            "indicators": {
                "rsi": rsi,
                "macd": macd,
                "sma": {
                    "period": 20,
                    "data": sma
                }
            },
        })

    except Exception as e:
        print("⛔ API ERROR:", e)
        return jsonify({"error": str(e)}), 500


'''
@app.route("/api/fchart2", methods=["GET"])
def fchart2():
    try:
        raw_symbol = request.args.get("symbol")
        if not raw_symbol:
            return jsonify({"error": "Symbol required"}), 400

        # ---------------- SYMBOL ----------------
        if ":" in raw_symbol:
            exchange, symbol = raw_symbol.split(":")
        else:
            exchange, symbol = "NASDAQ", raw_symbol

        exchange = exchange.upper()
        symbol = symbol.upper()

        # ---------------- DATE RANGE ----------------
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=450)

        # ---------------- FETCH BARS ----------------
        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
            limit=400,
            feed=DataFeed.IEX,
        )

        bars = data_client.get_stock_bars(req).df

        if bars.empty:
            return jsonify({
                "meta": {
                    "symbol": symbol,
                    "exchange": exchange,
                    "currency": "USD",
                },
                "quotes": [],
                "indicators": {"rsi": [], "macd": []},
            })

        bars = bars.xs(symbol).reset_index()

        # ---------------- DATAFRAME ----------------
        df = bars.copy()
        df["time"] = df["timestamp"].dt.strftime("%Y-%m-%d")
        close = df["close"]

        # ---------------- QUOTES ----------------
        quotes = [
            {
                "date": row["time"],
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": int(row["volume"]),
            }
            for _, row in df.iterrows()
        ]

        # ================= RSI (14) =================
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()

        rs = avg_gain / avg_loss
        df["rsi"] = 100 - (100 / (1 + rs))

        rsi = [
            {"time": row["time"], "value": round(row["rsi"], 2)}
            for _, row in df.iterrows()
            if not pd.isna(row["rsi"])
        ]

        # ================= MACD (12, 26, 9) =================
        ema12 = df["close"].ewm(span=12, adjust=False).mean()
        ema26 = df["close"].ewm(span=26, adjust=False).mean()

        df["macd"] = ema12 - ema26
        df["signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["hist"] = df["macd"] - df["signal"]

        macd = [
            {
                "time": row["time"],
                "macd": round(row["macd"], 4),
                "signal": round(row["signal"], 4),
                "hist": round(row["hist"], 4),
            }
            for _, row in df.iterrows()
            if not pd.isna(row["macd"])
        ]

        # ---------------- META ----------------
        meta = {
            "symbol": symbol,
            "exchange": exchange,
            "currency": "USD",
            "regularMarketPrice": float(df.iloc[-1]["close"]),
            "longName": symbol,
        }

        return jsonify({
            "meta": meta,
            "quotes": quotes,
            "indicators": {
                "rsi": rsi,
                "macd": macd,
            },
        })

    except Exception as e:
        print("⛔ API ERROR:", e)
        return jsonify({"error": str(e)}), 500
'''


def normalize_response(result):
    """
    Ensure safe JSON response for React / MUI DataGrid
    """
    if not result or "data" not in result:
        return []

    data = result["data"]

    # Replace NaN / None with safe values
    for row in data:
        for k, v in row.items():
            if v is None:
                row[k] = 0

    return data

VALID_CATEGORIES = {
    "gainers": "regular",
    "most-active": "regular",
    "after-hours-gainers": "after-hours",
    "pre-market-gainers": "pre-market",
}

ALLOWED_EXCHANGES = ("NASDAQ", "NYSE", "AMEX")
OTC_EXCHANGES = ("OTC", "PINK", "OTCQB", "OTCQX")

def is_otc_stock(row):
    symbol = str(row.get("symbol", "")).upper()
    return symbol.startswith("OTC:")
  
@app.route("/api/market-movers", methods=["GET"])
def market_movers_unified():
    category = request.args.get("category", "gainers")
    limit = int(request.args.get("limit", 50))

    if category not in VALID_CATEGORIES:
        return jsonify({
            "success": False,
            "error": f"Invalid category: {category}"
        }), 400

    result = market_movers.scrape(
        market="stocks-usa",
        category=category,
        limit=limit * 20
    )

    data = normalize_response(result)

    # ONLY exclude OTC:* symbols
    filtered = [
        row for row in data
        if not is_otc_stock(row)
    ][:limit]

    return jsonify({
        "success": True,
        "category": category,
        "session": VALID_CATEGORIES[category],
        "count": len(filtered),
        "data": filtered
    })




@app.route("/api/technical-analysis", methods=["GET"])
def get_technical_analysis():
    try:
        symbol = request.args.get("symbol")
        exchange = request.args.get("exchange", "NASDAQ")
        interval = request.args.get("interval", "1d")

        if not symbol:
            return jsonify({"success": False, "error": "symbol is required"}), 400

        handler = TA_Handler(
            symbol=symbol.upper(),
            exchange=exchange.upper(),
            screener="america",
            interval=INTERVAL_MAP.get(interval, Interval.INTERVAL_1_DAY)
        )

        analysis = handler.get_analysis()

        return jsonify({
            "success": True,
            "symbol": symbol.upper(),
            "exchange": exchange.upper(),
            "interval": interval,
            "summary": analysis.summary,
            "oscillators": analysis.oscillators,
            "moving_averages": analysis.moving_averages,
            "indicators": analysis.indicators
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/tv/news/<symbol>", methods=["GET"])
def get_tv_news(symbol):
    try:
        symbol = symbol.upper()
        exchange = request.args.get("exchange", "NASDAQ")

        headlines = news_scraper.scrape_headlines(
            symbol=symbol,
            exchange=exchange,
            sort="latest"
        )

        # headlines IS A LIST
        return jsonify({
            "success": True,
            "symbol": symbol,
            "exchange": exchange,
            "count": len(headlines),
            "data": headlines
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route("/api/tv/news/content", methods=["GET"])
def get_tv_news_content():
    try:
        story_path = request.args.get("storyPath")
        if not story_path:
            return jsonify({"success": False, "error": "storyPath required"}), 400

        content = news_scraper.scrape_news_content(story_path)
        return jsonify({"success": True, "data": content})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500



@app.route("/api/tv/indicators/<symbol>", methods=["GET"])
def get_all_indicators(symbol):
    try:
        symbol = symbol.upper()
        timeframe = request.args.get("timeframe", "1d")
        exchange = request.args.get("exchange", "NASDAQ")

        indicators_scraper = Indicators(
            export_result=False,
            export_type="json"
        )

        indicators = indicators_scraper.scrape(
            symbol=symbol,
            timeframe=timeframe,
            exchange=exchange,      # IMPORTANT
            allIndicators=True
        )

        return jsonify({
            "success": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "exchange": exchange,
            "data": indicators
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/stocks/price-range", methods=["GET"])
def stocks_in_price_range():
    try:
        # ---- Query params ----
        min_price = float(request.args.get("min", 50))
        max_price = float(request.args.get("max", 200))
        limit = int(request.args.get("limit", 50))

        # ---- TradingView filters ----
        filters = [
            {
                "left": "close",
                "operation": "in_range",
                "right": [min_price, max_price]
            }
        ]

        # ---- Run screener ----
        results = tv_screener.screen(
            market="america",
            filters=filters,
            limit=limit
        )

        data = results.get("data", [])

        return jsonify({
            "success": True,
            "count": len(data),
            "min_price": min_price,
            "max_price": max_price,
            "data": data
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500




@app.route("/api/scan_top_gainers", methods=["GET"])
def scan_top_gainers():
    try:
        # -----------------------------
        # Query params
        # -----------------------------
        strategy = request.args.get("strategy")
        top_gainers = request.args.get("top_gainers", "0") == "1"

        min_price = float(request.args.get("min_price", 5))
        max_price = float(request.args.get("max_price", 500))
        min_volume = int(request.args.get("min_volume", 1_000_000))
        limit = int(request.args.get("limit", 100))

        # -----------------------------
        # Base TradingView query
        # -----------------------------
        query = (
            Query()
            .set_markets("america")
            .select(
                "name",
                "close",
                "volume",
                "RSI",
                "change",
                "change_abs",
                'AnalystRating.tr',
                'AnalystRating',
            )     
            .where(
                Column("volume") >= min_volume
            )
        )

        # -----------------------------
        # Strategy filters
        # -----------------------------
        if strategy == "rsi_oversold":
            query = query.where(Column("RSI") < 30)

        elif strategy == "rsi_overbought":
            query = query.where(Column("RSI") > 70)

        elif strategy == "momentum":
            query = query.where(
                Column("RSI") > 50,
                Column("change") > 2
            )

        # -----------------------------
        # Sorting
        # -----------------------------
        if top_gainers:
            query = query.order_by("change", ascending=False)
        else:
            query = query.order_by("RSI", ascending=True)

        query = query.limit(limit * 100)  # fetch extra, filter later

        # -----------------------------
        # Execute
        # -----------------------------
        result = query.get_scanner_data()

        # -----------------------------
        # Normalize result
        # -----------------------------
        if isinstance(result, pd.DataFrame):
            rows = result.to_dict(orient="records")
        elif isinstance(result, dict):
            rows = result.get("data", [])
        elif isinstance(result, tuple):
            a, b = result
            rows = (
                a.to_dict(orient="records")
                if isinstance(a, pd.DataFrame)
                else b.to_dict(orient="records")
                if isinstance(b, pd.DataFrame)
                else a if isinstance(a, list)
                else b
            )
        else:
            raise TypeError(f"Unsupported return type: {type(result)}")

        # -----------------------------
        # 🔥 HARD SERVER-SIDE FILTERS
        # -----------------------------
        cleaned = []

        for r in rows:
            try:
                ticker = r.get("ticker", "")
                close = float(r.get("close", 0))
                volume = int(r.get("volume", 0))
                change_abs = float(r.get("change_abs", 0))

                # 🚫 Remove OTC
                if ticker.startswith("OTC:"):
                    continue

                # 🚫 Enforce price AGAIN
                if not (min_price <= close <= max_price):
                    continue

                # 🚫 Dead stocks (no movement)
                if abs(change_abs) < 1e-9:
                    continue

                # 🚫 Bad volume
                if volume < min_volume:
                    continue

                cleaned.append(r)

            except Exception:
                continue

            if len(cleaned) >= limit:
                break

        return jsonify({
            "success": True,
            "count": len(cleaned),
            "strategy": strategy,
            "top_gainers": top_gainers,
            "data": cleaned
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500



#http://192.168.150.105:5000/api/scan?min_price=10&max_price=200&min_volume=2000000&min_rsi=50&limit=25
@app.route("/api/scan", methods=["GET"])
def scan_stocks():
    try:
        min_price = float(request.args.get("min_price", 5))
        max_price = float(request.args.get("max_price", 500))
        min_volume = int(request.args.get("min_volume", 1_000_000))
        min_rsi = float(request.args.get("min_rsi", 40))
        limit = int(request.args.get("limit", 20))

        query = (
            Query()
            .set_markets("america")
            .select(
                "name",
                "close",
                "volume",
                "RSI",
                "change",
                "change_abs"
            )
            .where(
                Column("close") >= min_price,
                Column("close") <= max_price,
                Column("volume") >= min_volume,
                Column("RSI") >= min_rsi
            )
            .order_by("change", ascending=False)
            .limit(limit)
        )

        result = query.get_scanner_data()

        # 🔒 NORMALIZE ALL POSSIBLE RETURN TYPES
        if isinstance(result, pd.DataFrame):
            df = result
            rows = df.to_dict(orient="records")
            total_count = len(rows)

        elif isinstance(result, dict):
            rows = result.get("data", [])
            total_count = result.get("totalCount", len(rows))

        elif isinstance(result, tuple):
            a, b = result
            if isinstance(a, pd.DataFrame):
                rows = a.to_dict(orient="records")
                total_count = b
            elif isinstance(b, pd.DataFrame):
                rows = b.to_dict(orient="records")
                total_count = a
            else:
                raise TypeError("Unexpected tuple structure")

        else:
            raise TypeError(f"Unsupported return type: {type(result)}")

        return jsonify({
            "success": True,
            "count": total_count,
            "data": rows
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route("/api/tv/volume-leaders", methods=["GET"])
def volume_leaders():
    """
    Example:
    /api/tv/volume-leaders?page=1&limit=100
    """

    page = int(request.args.get("page", 1))      # 1-based
    limit = int(request.args.get("limit", 100))

    # safety
    page = max(page, 1)
    limit = min(limit, 500)

    offset = (page - 1) * limit
    print(f"page==: {page} & limit=== {limit} & offset== {offset}")
    try:
        query = (
            Query()
            .select(
                'name',
                'description',
                'logoid',
                'update_mode',
                'type',
                'typespecs',
                'Value.Traded',
                'currency',
                'close',
                'pricescale',
                'minmov',
                'fractional',
                'minmove2',
                'change',
                'volume',
                'relative_volume_10d_calc',
                'market_cap_basic',
                'fundamental_currency_code',
                'price_earnings_ttm',
                'earnings_per_share_diluted_ttm',
                'earnings_per_share_diluted_yoy_growth_ttm',
                'dividends_yield_current',
                'sector.tr',
                'market',
                'sector',
                'AnalystRating.tr',
                'AnalystRating',
            )
            .where(
                col('exchange').isin(['AMEX', 'CBOE', 'NASDAQ', 'NYSE']),
                col('is_primary') == True,
                col('typespecs').has('common'),
                col('typespecs').has_none_of('preferred'),
                col('type') == 'stock',
                col('close').between(2, 10000),
                col('active_symbol') == True,
            )
            .order_by('Value.Traded', ascending=False, nulls_first=False)
            .offset(offset)              # 🔑 NEW
            .limit(limit)            
            .set_markets('america')
            .set_property(
                'symbols',
                {'query': {'types': ['stock', 'fund', 'dr', 'structured']}}
            )
            #.set_property('preset', 'volume_leaders')
        )

        total_count, data = query.get_scanner_data()

        if data is None:
            return jsonify({
                "success": True,
                "page": page,
                "limit": limit,
                "total": total_count,
                "data": []
            })

        if hasattr(data, "to_dict"):
            data = (
                data
                .replace([np.nan, np.inf, -np.inf], None)
                .reset_index(drop=True)
                .to_dict(orient="records")
            )

        return jsonify({
            "success": True,
            "page": page,
            "limit": limit,
            "total": total_count,
            "data": data
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/tv/best-performing", methods=["GET"])
def best_performing():
    """
    Example:
    /api/tv/volume-leaders?limit=100
    """
    page = int(request.args.get("page", 1))      # 1-based
    limit = int(request.args.get("limit", 100))

    # safety
    page = max(page, 1)
    limit = min(limit, 500)

    offset = (page - 1) * limit

    #limit = int(request.args.get("limit", 100))

    try:
        query = (
           (Query()
             .select(
                 'name',
                 'description',
                 'logoid',
                 'update_mode',
                 'type',
                 'typespecs',
                 'Perf.Y',
                 'close',
                 'pricescale',
                 'minmov',
                 'fractional',
                 'minmove2',
                 'currency',
                 'change',
                 'volume',
                 'market_cap_basic',
                 'fundamental_currency_code',
                 'price_earnings_ttm',
                 'earnings_per_share_diluted_ttm',
                 'earnings_per_share_diluted_yoy_growth_ttm',
                 'dividends_yield_current',
                 'sector.tr',
                 'sector',
                 'market',
                 'AnalystRating.tr',
                 'AnalystRating',
                 'relative_volume_10d_calc',
             )
             .where(
                 col('exchange').isin(['AMEX', 'CBOE', 'NASDAQ', 'NYSE']),
                 col('is_primary') == True,
                 col('typespecs').has('common'),
                 col('typespecs').has_none_of('preferred'),
                 col('type') == 'stock',
                 col('active_symbol') == True,
                 col('market_cap_basic') > 0,
             )
             .order_by('Perf.Y', ascending=False, nulls_first=False)
             .offset(offset)
             .limit(100)
             .set_markets('america')
             .set_property('symbols', {'query': {'types': ['stock', 'fund', 'dr', 'structured']}})
             .set_property('preset', 'best_performing'))
        )

        total_count, data = query.get_scanner_data()

        if data is None:
            return jsonify({
                "success": True,
                "page": page,
                "limit": limit,
                "total": total_count,
                "data": []
            })


        if hasattr(data, "to_dict"):
            data = (
                data
                .replace([np.nan, np.inf, -np.inf], None)
                .reset_index(drop=True)
                .to_dict(orient="records")
            )

        return jsonify({
            "success": True,
            "total": total_count,
            "data": data
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/tv/top-gainers", methods=["GET"])
def top_gainers():
    
    page = int(request.args.get("page", 1))      # 1-based
    limit = int(request.args.get("limit", 100))

    # safety
    page = max(page, 1)
    limit = min(limit, 500)

    offset = (page - 1) * limit

    #limit = int(request.args.get("limit", 100))

    try:
        query = (
            Query()
            .select(
               'name',
                'description',
                'logoid',
                'update_mode',
                'type',
                'typespecs',
                'market_cap_basic',
                'fundamental_currency_code',
                'close',
                'pricescale',
                'minmov',
                'fractional',
                'minmove2',
                'currency',
                'change',
                'volume',
                'price_earnings_ttm',
                'earnings_per_share_diluted_ttm',
                'earnings_per_share_diluted_yoy_growth_ttm',
                'dividends_yield_current',
                'sector.tr',
                'sector',
                'market',
                'AnalystRating.tr',
                'AnalystRating',
                'relative_volume_10d_calc',
            )
            .where(
                col('exchange').isin(['AMEX', 'CBOE', 'NASDAQ', 'NYSE']),
                col('is_primary') == True,
                col('typespecs').has('common'),
                col('typespecs').has_none_of('preferred'),
                col('type') == 'stock',
                col('close').between(2, 10000),
                col('change') > 0,
                col('active_symbol') == True,
            )
            .order_by('change', ascending=False, nulls_first=False)
            .offset(offset)
            .limit(limit)
            .set_markets('america')
            .set_property(
                'symbols',
                {'query': {'types': ['stock', 'fund', 'dr', 'structured']}}
            )
            .set_property('preset', 'gainers')
        )

        total_count, data = query.get_scanner_data()

        if data is None:
            return jsonify({
                "success": True,
                "page": page,
                "limit": limit,
                "total": total_count,
                "data": []
            })

        if hasattr(data, "to_dict"):
            data = (
                data
                .replace([np.nan, np.inf, -np.inf], None)
                .reset_index(drop=True)
                .to_dict(orient="records")
            )

        return jsonify({
            "success": True,
            "total": total_count,
            "data": data
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/tv/small-cap", methods=["GET"])
def small_cap():
    
    page = int(request.args.get("page", 1))      # 1-based
    limit = int(request.args.get("limit", 100))

    # safety
    page = max(page, 1)
    limit = min(limit, 500)

    offset = (page - 1) * limit

    #limit = int(request.args.get("limit", 100))

    try:
        query = (
            Query()
            .select(
                 'name',
                 'description',
                 'logoid',
                 'update_mode',
                 'type',
                 'typespecs',
                 'market_cap_basic',
                 'fundamental_currency_code',
                 'close',
                 'pricescale',
                 'minmov',
                 'fractional',
                 'minmove2',
                 'currency',
                 'change',
                 'volume',
                 'price_earnings_ttm',
                 'earnings_per_share_diluted_ttm',
                 'earnings_per_share_diluted_yoy_growth_ttm',
                 'dividends_yield_current',
                 'sector.tr',
                 'sector',
                 'market',
                 'AnalystRating.tr',
                 'AnalystRating',
                 'relative_volume_10d_calc',
             )
             .where(
                 col('exchange').isin(['AMEX', 'CBOE', 'NASDAQ', 'NYSE']),
                 col('is_primary') == True,
                 col('typespecs').has('common'),
                 col('typespecs').has_none_of('preferred'),
                 col('type') == 'stock',
             )
             .order_by('market_cap_basic', ascending=True, nulls_first=False)
             .limit(100)
             .set_markets('america')
             .set_property('symbols', {'query': {'types': ['stock', 'fund', 'dr', 'structured']}})
             .set_property('preset', 'small_cap')
        )

        total_count, data = query.get_scanner_data()

        if data is None:
            return jsonify({
                "success": True,
                "page": page,
                "limit": limit,
                "total": total_count,
                "data": []
            })

        if hasattr(data, "to_dict"):
            data = (
                data
                .replace([np.nan, np.inf, -np.inf], None)
                .reset_index(drop=True)
                .to_dict(orient="records")
            )

        return jsonify({
            "success": True,
            "total": total_count,
            "data": data
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

#http://192.168.150.105:5000/api/tvscreener?min_rsi=50&max_rsi=65&sort_by=rsi&sort_order=desc
@app.route("/api/tvscreener", methods=["GET"])
def screener():
    """
    Query params:
    market=america|india
    min_price=20
    max_price=500
    min_volume=1000000
    min_rsi=30
    max_rsi=70
    sort_by=volume|close|RSI
    sort_order=asc|desc
    """

    # -------- Defaults --------
    market = request.args.get("market", "america")

    min_price = float(request.args.get("min_price", 0))
    max_price = float(request.args.get("max_price", 1_000_000))

    min_volume = int(request.args.get("min_volume", 0))

    min_rsi = float(request.args.get("min_rsi", 0))
    max_rsi = float(request.args.get("max_rsi", 100))

    sort_by = request.args.get("sort_by", "volume")
    sort_order = request.args.get("sort_order", "desc")

    # -------- Build Query --------
    query = (
        Query()
        .set_markets(market)
        .select(
            "name",
            "open",
            "close",
            "change",            # absolute price change (P/L)
            "volume",
            "RSI"            
        )
        .where(
            Column("close") >= min_price,
            Column("close") <= max_price,
            Column("volume") >= min_volume,
            Column("RSI") >= min_rsi,
            Column("RSI") <= max_rsi,
        )
    )

    # -------- Sorting --------
    allowed_sort_fields = {
        "price": "close",
        "close": "close",
        "volume": "volume",
        "rsi": "RSI"
    }

    if sort_by.lower() in allowed_sort_fields:
        query = query.order_by(
            Column(allowed_sort_fields[sort_by.lower()]),
            ascending=(sort_order.lower() == "asc")
        )

    # -------- Execute --------
    try:
        _, df = query.get_scanner_data()
        df["change_percent"] = (df["change"] / (df["close"] - df["change"])) * 100
        return jsonify(df.to_dict(orient="records"))
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


'''
# ---------- Load cache (if exists and fresh) ----------
def load_cache():
    if not os.path.exists(CACHE_FILE):
        return {"timestamp": 0, "results": []}
    try:
        with open(CACHE_FILE, "r") as f:
            data = json.load(f)
        if time.time() - data.get("timestamp", 0) < CACHE_TTL:
            print("✅ Loaded cache from file")
            return data
        else:
            print("♻️ Cache file expired — refreshing")
            return {"timestamp": 0, "results": []}
    except Exception:
        return {"timestamp": 0, "results": []}


def save_cache(data):
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(data, f)
        print("💾 Cache saved to file")
    except Exception as e:
        print(f"⚠️ Failed to save cache: {e}")


cache_data = load_cache()
'''
#https://en.wikipedia.org/wiki/Nasdaq-100
'''
def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/130.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }

    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()

    # Sanity check to ensure we actually got a Wikipedia page
    if "<table" not in response.text or "Symbol" not in response.text:
        raise RuntimeError("Wikipedia page returned unexpected content")

    tables = pd.read_html(StringIO(html))
    for table in tables:
        for col in table.columns:
            if "symbol" in str(col).lower() or "ticker" in str(col).lower():
                tickers = table[col].astype(str).str.replace(".", "-", regex=False).tolist()
                return tickers
    raise ValueError("No Symbol/Ticker column found in Wikipedia table.")
'''
def get_zacks_rank(symbol):
    try:
        res = requests.get(f"http://localhost:7000/api/zacks?symbol={symbol}")
        return res.json() if res.ok else None
    except:
        return None


def get_alpaca_tickers(
    api_key: str,
    secret_key: str,
    base_url: str = "https://paper-api.alpaca.markets",
):
    url = f"{base_url}/v2/assets"

    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret_key,
    }

    params = {
        "status": "active",
        "asset_class": "us_equity",
    }

    response = requests.get(url, headers=headers, params=params, timeout=10)
    response.raise_for_status()

    assets = response.json()

    tickers = [
        a["symbol"]
        for a in assets
        if a["tradable"]
        and a["exchange"] in {"NYSE", "NASDAQ", "AMEX"}
    ]

    return sorted(tickers)

# ---------- STEP 2: Compute changes ----------
'''
def compute_changes(batch):
    results = []
    try:
        data = yf.download(batch, period="6mo", interval="1d", group_by="ticker", progress=False)
        for ticker in batch:
            try:
                close = data[ticker]["Close"]
                if close.empty:
                    continue

                start_6m, end_6m = close.iloc[0], close.iloc[-1]
                change_6m = (end_6m - start_6m) / start_6m * 100 if start_6m else None

                mid_index = int(len(close) * 0.5)
                start_3m, end_3m = close.iloc[mid_index], close.iloc[-1]
                change_3m = (end_3m - start_3m) / start_3m * 100 if start_3m else None

                change_1w = ((latest - data["Close"].iloc[-5]) / data["Close"].iloc[-5]) * 100



                results.append({
                    "symbol": ticker,
                    "change_3m_pct": round(change_3m, 2) if change_3m is not None else None,
                    "change_6m_pct": round(change_6m, 2) if change_6m is not None else None,
                    "change_1w_pct": round(change_1w, 2) if change_1w is not None else None,
                })
            except Exception:
                continue
    except Exception:
        pass
    return results
'''

def compute_changes(batch):
    results = []

    def safe_pct(start, end):
        if start is None or end is None:
            return None
        if pd.isna(start) or pd.isna(end):
            return None
        start = float(start)
        end = float(end)
        if start == 0:
            return None
        val = (end - start) / start * 100
        if math.isnan(val) or math.isinf(val):
            return None
        return round(val, 2)

    data = yf.download(
        batch,
        period="3mo",
        interval="1d",
        group_by="ticker",
        progress=False,
        threads=False,
    )

    for ticker in batch:
        try:
            if ticker not in data:
                continue

            close = data[ticker]["Close"].dropna()

            if close.empty or len(close) < 2:
                continue

            change_3m = safe_pct(close.iloc[0], close.iloc[-1])
            change_1w = safe_pct(close.iloc[-5], close.iloc[-1]) if len(close) >= 5 else None

            results.append({
                "symbol": str(ticker),
                "change_3m_pct": change_3m,
                "change_1w_pct": change_1w,
            })

        except Exception as e:
            print(f"⚠️ {ticker}: {e}")

    return results

'''
def get_top_gainers_data(tickers):
    data = yf.download(
        tickers,
        period="3mo",
        interval="1d",
        group_by="ticker",
        progress=False,
        threads=False,
    )

    results = []

    def safe_pct(start, end):
        if pd.isna(start) or pd.isna(end) or start == 0:
            return None
        val = (float(end) - float(start)) / float(start) * 100
        return round(val, 2) if math.isfinite(val) else None

    for ticker in tickers:
        try:
            if ticker not in data:
                continue

            close = data[ticker]["Close"].dropna()
            if len(close) < 2:
                continue

            results.append({
                "symbol": ticker,
                "change_3m_pct": safe_pct(close.iloc[0], close.iloc[-1]),
                "change_1w_pct": safe_pct(close.iloc[-5], close.iloc[-1]) if len(close) >= 5 else None,
            })

        except Exception as e:
            print(f"⚠️ {ticker}: {e}")

    df = pd.DataFrame(results)
    df = df[df["change_1w_pct"].notna()]
    df = df.sort_values("change_1w_pct", ascending=False).head(20)

    # 🔥 FINAL JSON GUARANTEE
    clean = []
    for r in df.to_dict("records"):
        clean.append({
            k: (None if v is None or (isinstance(v, float) and not math.isfinite(v)) else float(v) if isinstance(v, (float, np.floating)) else v)
            for k, v in r.items()
        })

    return clean
'''

def get_top_gainers_data(tickers):
    log(f"🚀 Starting top gainers for {len(tickers)} tickers")

    start = datetime.utcnow() - timedelta(days=120)
    end = datetime.utcnow()

    results = []
    BATCH_SIZE = 25
    batches = [tickers[i:i + BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]

    for i, batch in enumerate(batches, 1):
        log(f"📡 Fetching batch {i}/{len(batches)}")

        try:
            request = StockBarsRequest(
                symbol_or_symbols=batch,
                timeframe=TimeFrame.Day,
                start=start,
                end=end,
                limit=120,
                feed=DataFeed.IEX   # IMPORTANT
            )

            bars = data_client.get_stock_bars(request)
            if bars.df.empty:
                continue

            df = bars.df.reset_index()

        except Exception as e:
            log(f"⚠️ Batch {i} skipped: {e}")
            continue

        for symbol, df_sym in df.groupby("symbol"):
            close = df_sym["close"].dropna().reset_index(drop=True)
            if len(close) < 20:
                continue

            current_price = close.iloc[-1]
            base_price_1w = close.iloc[-5] if len(close) >= 5 else None
            base_price_3m = close.iloc[0]

            change_1w = (
                (current_price - base_price_1w) / base_price_1w * 100
                if base_price_1w and base_price_1w > 0 else None
            )
            change_3m = (
                (current_price - base_price_3m) / base_price_3m * 100
                if base_price_3m > 0 else None
            )

            results.append({
                "symbol": symbol,
                "current_price": round(current_price, 2),
                "base_price_1w": round(base_price_1w, 2) if base_price_1w else None,
                "base_price_3m": round(base_price_3m, 2),
                "change_1w_pct": round(change_1w, 2) if change_1w is not None else None,
                "change_3m_pct": round(change_3m, 2) if change_3m is not None else None,
            })

    df = pd.DataFrame(results)
    if df.empty:
        return []

    df = df[df["change_1w_pct"].notna()]
    df = df.sort_values("change_1w_pct", ascending=False).head(20)

    log(f"✅ Returning {len(df)} gainers")
    #return df.to_dict("records")
    df = pd.DataFrame(results)
    if df.empty:
        return []

    df = df[df["change_1w_pct"].notna()]
    df = df.sort_values("change_1w_pct", ascending=False).head(20)

    # 👇 NEW PART STARTS HERE
    symbols = df["symbol"].tolist()

    market = get_market_status()
    assets = get_assets_tradeable(symbols)

    final_results = []
    for row in df.to_dict("records"):
        row["market"] = market
        row["asset"] = assets.get(row["symbol"], {})
        final_results.append(row)

    log(f"✅ Returning {len(final_results)} gainers")
    return final_results
        


@app.route("/api/top_gainers_sp500", methods=["GET"])
def top_gainers_sp500():
    force = request.args.get("force") == "1"  # 👈 NEW
    cache_file = CACHE_SP500
    cache_data = load_cache(cache_file)
    now = time.time()

    # Serve cached version unless force refresh is requested
    # and (now - cache_data["timestamp"] < CACHE_TTL)
    if not force and cache_data["results"]:
        print("✅ Serving S&P 500 from cache")
        return jsonify(cache_data["results"])

    print("♻️ Refreshing S&P 500 gainers...")
    
    tickers = get_alpaca_tickers(api_key="PKC7D4XB4OTV2VDEFUF5BRL33P",
    secret_key="DZxMp2TBaqQWZH3seGm7ZSWbHjw25xLBD7ZGrG4F4GaF",)
    data = get_top_gainers_data(tickers)
    response = {
    "market": get_market_status(),
    "data": data
    }
    print(f"✅ data === ", jsonify(data))
    cache_data = {"timestamp": now, "results": response}
    save_cache(cache_file, cache_data)

    return jsonify(data)



def get_nasdaq100_tickers():
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/130.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }

    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()

    html = response.text

    if "<table" not in html:
        raise RuntimeError("Wikipedia returned unexpected content for NASDAQ-100")

    tables = pd.read_html(StringIO(html))

    # NASDAQ-100 table has column "Ticker" but we check flexibly
    for table in tables:
        for col in table.columns:
            if "ticker" in str(col).lower() or "symbol" in str(col).lower():

                tickers = (
                    table[col]
                    .astype(str)
                    .str.replace(".", "-", regex=False)  # Yahoo-style tickers
                    .tolist()
                )

                # NASDAQ-100 table contains header rows sometimes – filter weird entries
                tickers = [t for t in tickers if t.isalnum() or "-" in t]

                return tickers

    raise ValueError("No ticker column found in NASDAQ-100 table.")


@app.route("/api/top_gainers_nasdaq100", methods=["GET"])
def top_gainers_nasdaq100():
    force = request.args.get("force") == "1"  # 👈 NEW
    cache_file = CACHE_NASDAQ100
    cache_data = load_cache(cache_file)
    now = time.time()

    # Serve cached version unless force refresh is requested
    if not force and cache_data["results"] and (now - cache_data["timestamp"] < CACHE_TTL):
        print("✅ Serving NASDAQ-100 from cache")
        return jsonify(cache_data["results"])

    print("♻️ Refreshing NASDAQ-100 gainers...")
    tickers = get_nasdaq100_tickers()
    data = get_top_gainers_data(tickers)

    cache_data = {"timestamp": now, "results": data}
    save_cache(cache_file, cache_data)

    return jsonify(data)


def symbols_to_quoted_string(results):
    symbols = [item["symbol"] for item in results if "symbol" in item]
    return ",".join([f'"{s}"' for s in symbols])

@app.route("/api/symbol_list_sp500", methods=["GET"])
def symbol_list_sp500():
    cache_file = CACHE_SP500
    cache_data = load_cache2(cache_file)

    results = cache_data.get("results", {}).get("data", [])
    if not results:
        return jsonify({"error": "Cache empty. Hit /api/top_gainers_sp500 first."}), 400

    # 🔥 Extract ONLY symbol strings
    symbols = [
        item["symbol"]
        for item in results
        if isinstance(item, dict) and "symbol" in item
    ]

    return jsonify(symbols)

@app.route("/api/symbol_list_nasdaq100", methods=["GET"])
def symbol_list_nasdaq100():
    cache_file = CACHE_NASDAQ100
    cache_data = load_cache(cache_file)

    results = cache_data.get("results", [])
    if not results:
        return jsonify({"error": "Cache empty. Hit /api/top_gainers_nasdaq100 first."}), 400

    output = symbols_to_quoted_string(results)
    return jsonify({"symbols": output})






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



'''
async def fetch_quote(session, symbol):
    """Fetch a single symbol quote concurrently."""
    url = f"{FINNHUB_BASE}/quote"
    try:
        async with session.get(url, params={"symbol": symbol, "token": FINNHUB_API_KEY}) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
            price = data.get("c")
            open_price = data.get("o")
            dp = data.get("dp")
            if not price or not open_price or price == 0 or open_price == 0:
                return None
            if dp is None:
                dp = ((price - open_price) / open_price) * 100
            return {
                "symbol": symbol,
                "price": round(price, 2),
                "change": round(price - open_price, 2),
                "percentchange": round(dp, 2),
            }
    except Exception:
        return None


@app.route("/api/screen_by_criteria_finnhub", methods=["GET"])
def custom_screener2():
    region = request.args.get("region", "us").lower()
    min_price = float(request.args.get("min_price", 0))
    max_price = float(request.args.get("max_price", 1_000_000))
    min_change = float(request.args.get("min_change", 0))
    sort_field = request.args.get("sort_field", "percentchange")
    sort_asc = request.args.get("sort_asc", "false").lower() == "true"
    limit = int(request.args.get("limit", 5))

    async def run_screen():
        print("\n=== [Finnhub Screener - async] START ===")
        exchange = "US" if region == "us" else region.upper()

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            # Fetch symbol list
            async with session.get(f"{FINNHUB_BASE}/stock/symbol",
                                   params={"exchange": exchange, "token": FINNHUB_API_KEY}) as resp:
                all_symbols = await resp.json()

            print(f"→ Retrieved {len(all_symbols)} symbols")

            # Sample subset for rate safety
            sample = [s.get("symbol") for s in all_symbols[:80] if s.get("symbol")]
            print(f"Fetching quotes for {len(sample)} symbols concurrently...")

            quotes = await asyncio.gather(*[fetch_quote(session, sym) for sym in sample])
            results = [q for q in quotes if q]

            # Apply filters
            filtered = [
                r for r in results
                if min_price <= r["price"] <= max_price and r["change"] >= min_change
            ]

            print(f"✅ Retrieved {len(filtered)} valid quotes")

            if not filtered:
                return []

            df = pd.DataFrame(filtered)
            df = df.sort_values(by=sort_field, ascending=sort_asc).head(limit)
            return clean_nans(df.to_dict(orient="records"))

    try:
        results = asyncio.run(run_screen())
        return jsonify(results)
    except Exception as e:
        print("❌ Error:", e)
        return jsonify({"error": str(e)}), 500

'''

def clean_nans(obj):
    """Recursively replace NaN/inf/-inf with None for JSON safety."""
    if isinstance(obj, list):
        return [clean_nans(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: clean_nans(v) for k, v in obj.items()}
    elif isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


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


'''

# Global live quote cache and WebSocket reference

'''
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

clients = []           # all SSE connections
tracked_symbols = set()
latest_data = {}        # { symbol: { ...last quote... } }

# === WebSocket handler ===
def message_handler(message):
    try:
        print("Incoming message keys:", message.keys())
        print("Full message:", message)
        data = {
            "symbol": message.get("id"),
            "price": message.get("price"),
            "change": message.get("change"),
            "percentchange": message.get("change_percent"),
            
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
    print("trying to add symbol:", symbol)
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
    #http://localhost:3000
    #https://sanjeevdg.github.io             
    response = Response(event_stream(), mimetype="text/event-stream")
    response.headers["Access-Control-Allow-Origin"] = "https://sanjeevdg.github.io"
    response.headers["Cache-Control"] = "no-cache"
    response.headers["Connection"] = "keep-alive"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    return response


def get_market_status():
    """Determine U.S. market status based on time (Eastern)."""
    tz = pytz.timezone("US/Eastern")
    now = datetime.now(tz)
    open_t = dt_time(9, 30)
    close_t = dt_time(16, 0)

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


@app.route("/api/patterns", methods=["GET"])
def get_patterns():
    symbols_str = request.args.get("symbols")
    if not symbols_str:
        return jsonify({"error": "symbols param required"}), 400

    symbols = [s.strip().upper() for s in symbols_str.split(",") if s.strip()]

    try:
        data = yf.download(
            tickers=symbols,
            period="6mo",
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

        return jsonify(results)

    except Exception as e:
        print(f"Error fetching patterns: {e}")
        return jsonify({"error": str(e)}), 500




if __name__ == "__main__":
    socketio.run(app, host="192.168.150.105", port=5000, debug=False,use_reloader=False )

