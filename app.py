import eventlet
eventlet.monkey_patch()

#from flask import Flask, make_response,jsonify, send_file,Response, request, abort

from fastapi import FastAPI,Request#from flask_socketio import SocketIO, emit
#import eventlet

#eventlet.monkey_patch()
#from flask_sock import Sock
import requests
import warnings
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi import HTTPException
from fastapi import Query as APIQuery
from fastapi.responses import StreamingResponse, Response
from pathlib import Path
from fastapi.encoders import jsonable_encoder
from fastapi.concurrency import run_in_threadpool
import traceback
from scanner.momentum import get_top_momentum
from momscan_config import data_client

from fastapi.middleware.cors import CORSMiddleware


import os
import pandas as pd
import talib
import sys


from alpaca.data.live import StockDataStream
from alpaca.data.enums import DataFeed


from tvDatafeed import TvDatafeed
from tvDatafeed import Interval as TVInterval
from tradingview_screener import Query, Column 
from tradingview_ta import TA_Handler, Interval, Exchange
from ta.momentum import RSIIndicator
from ta.trend import MACD

from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
#import asyncio
import aiohttp
from yahoo_fin import stock_info as si
from datetime import datetime, timedelta, time as dt_time, timezone
#from datetime import datetime, timedelta
import pytz
from pytz import UTC
import math
import time
from queue import Queue, Empty
import json
import numpy as np
from io import StringIO
import io
import ssl
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockLatestBarRequest
from alpaca.data.requests import StockLatestQuoteRequest
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest, MarketOrderRequest, TakeProfitRequest, StopLossRequest
from alpaca.trading.enums import AssetClass, OrderSide, TimeInForce
from alpaca.data.historical.screener import ScreenerClient
from alpaca.data.requests import MostActivesRequest
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import OrderStatus

from pydantic import BaseModel


from alpaca.data.requests import StockSnapshotRequest


import matplotlib
matplotlib.use("Agg")  # IMPORTANT for servers
import matplotlib.pyplot as plt

from config import ALPACA_KEY, ALPACA_SECRET



#from iexfinance.stocks import Stock
#from alpaca_stream import subscribe_symbol, unsubscribe_symbol 
#from alpaca_stream import start_stream, subscribe_symbol

#from trading_bot2 import MomentumScalpBot
#import alpaca_trade_api as tradeapi


app = FastAPI()

#sock = Sock(app)

#async_mode="eventlet", 
#socketio = SocketIO(app, cors_allowed_origins="*")

warnings.filterwarnings("ignore", category=FutureWarning)

print("ALPACA_KEY:", ALPACA_KEY)
print("ALPACA_SECRET:", ALPACA_SECRET)

clients = set()   

ALLOWED_ORIGINS = [
    "https://sanjeevdg.github.io",
    "http://localhost:5173"
]

FINNHUB_API_KEY = "d3nr05hr01qtm4jdum8gd3nr05hr01qtm4jdum90"  # <-- replace with your own

col = Column  # alias for readability

latest_prices = {}


WATCHLIST_FILE = "watchlist.txt"

active_symbols = set()
symbol_clients = {}


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

BASE_URL = "https://paper-api.alpaca.markets"

     

HEADERS = {
    "APCA-API-KEY-ID": ALPACA_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET
}


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://sanjeevdg.github.io"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

data_client = StockHistoricalDataClient(
        ALPACA_KEY,
        ALPACA_SECRET
)

trading_client = TradingClient(
    api_key=ALPACA_KEY,
    secret_key=ALPACA_SECRET,
    paper=True
)

screener_client = ScreenerClient(
    api_key=ALPACA_KEY,
    secret_key=ALPACA_SECRET    
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


CSV_PATH = os.path.join("data", "bars_5Min.csv")

@app.get("/api/csv")
async def get_csv():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(BASE_DIR, CSV_PATH)

    df = pd.read_csv(path)

    # convert NaN → None
    df = df.replace({np.nan: None})

    data = df.to_dict(orient="records")

    return jsonable_encoder(data)


async def scan_top_gainers(strategy: str):
    return {"strategy": strategy}

@app.get("/api/scanner")
async def scanner(
    file: str = "data/bars_5Min.csv",
    rule: str = "macd_cross",
    adx: float | None = None,
    roc: float | None = None,
):

    file_path = file
    rule = rule

    adx_filter = adx
    roc_filter = roc

    df = pd.read_csv(file_path)
    df.columns = [c.lower() for c in df.columns]

    for col in ["open","high","low","close","volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    results = []

    for symbol, g in df.groupby("symbol"):

        g = g.sort_values("time").copy()

        close = g["close"].values
        high = g["high"].values
        low = g["low"].values

        # ------------------------
        # Indicators
        # ------------------------

        macd, signal, hist = talib.MACD(close)
        adx = talib.ADX(high, low, close, timeperiod=14)
        roc = talib.ROC(close, timeperiod=10)

        g["macd"] = macd
        g["signal"] = signal
        g["adx"] = adx
        g["roc"] = roc

        # ------------------------
        # Base rule
        # ------------------------

        if rule == "macd_cross":

            g["entry"] = (
                (g["macd"] > g["signal"]) &
                (g["macd"].shift(1) <= g["signal"].shift(1))
            )

            g["exit"] = (
                (g["macd"] < g["signal"]) &
                (g["macd"].shift(1) >= g["signal"].shift(1))
            )

        else:
            g["entry"] = False
            g["exit"] = False

        # ------------------------
        # Apply filters
        # ------------------------

        if adx_filter is not None:
            g["entry"] = g["entry"] & (g["adx"] > adx_filter)

        if roc_filter is not None:
            g["entry"] = g["entry"] & (g["roc"] > roc_filter)

        last = g.iloc[-1]

        signal_state = "NONE"

        if last["entry"]:
            signal_state = "LONG"

        elif last["exit"]:
            signal_state = "EXIT"

        results.append({
            "symbol": symbol,
            "signal": signal_state,
            "price": float(last["close"]),
            "adx": float(last["adx"]) if pd.notna(last["adx"]) else None,
            "roc": float(last["roc"]) if pd.notna(last["roc"]) else None
        })

    print("Symbols scanned:", len(results))
    print("Rule:", rule)

    return results

UNIVERSE = [
    "AAPL","MSFT","NVDA","TSLA","AMD","META","AMZN","GOOG",
    "NFLX","COIN","PLTR","SMCI","AVGO","INTC","PYPL",
    "CRM","ADBE","QCOM","MU","SHOP","UBER","SNOW",
    "ZS","PANW","ROKU","SQ","DKNG","RIVN","LCID",
    "ACIW,NGVC,DMRA,DSGX,CWT,ALMS,TREE,SION,VKTX,PHVS,KNSA,STNG",
    "CELC","SCI","KOD","NAVN","FATN","ICU","LOVE","GDEV","OKUR","CBIO","MDGL","CRBP","SGMT",
    "ELVN","KALV","SMTI","CNTB","DRIO","WTI","ORGN","DNTH","WTTR","PR","DEC","GLNG","CRGY",
    "PBR.A","PRGS","MMYT","BZ","PAR","EPRT","MAZE","OLLI","CELH","GOF","RLMD","BW","SLGL",
    "TNGX","QTI","ONDS","ERAS","PRAX","ANRO","LASR","CTMX","NKTR","OVID","DFDV","COGT",
    "CLYM","STTK","CRML","LPTH","AMPX","PRE","ARMP","WULF","CIFR","PRLD","IMMX",
    "STOK","TE","VFF","STRR","PVLA","UUUU","NAUT","TTMI","TYGO","AREC","SEPN",
    "ZURA","PZG","UAMY","TRX","VELO","SLS","IRD","ORKA","HUT","MU","TYRA","MBX"
]

@app.get("/scanner/momentum")
async def momentum_scanner(limit: int = 15):
    try:
        results = get_top_momentum(data_client, UNIVERSE, limit)

        return {
            "count": len(results),
            "top_stocks": results
        }

    except Exception as e:
        return {"error": str(e)}




@app.get("/api/top-movers")
async def top_movers(symbols):


    #symbols = request.args.get("symbols")

    if not symbols:
        return {"error": "symbols parameter required"}, 400

    SYMBOLS = symbols.split(",")

    print("Scanning:", SYMBOLS)


    print("Scanning top movers...")

    snapshot_request = StockSnapshotRequest(
        symbol_or_symbols=SYMBOLS
    )

    snapshots = data_client.get_stock_snapshot(snapshot_request)

    movers = []

    for symbol, snap in snapshots.items():

        if not snap.minute_bar or not snap.daily_bar:
            continue

        price = snap.minute_bar.close
        volume = snap.minute_bar.volume
        prev_close = snap.daily_bar.close

        if not prev_close:
            continue

        change = (price - prev_close) / prev_close
        change_pct = change * 100
        avg_volume = snap.daily_bar.volume
        score = change_pct * (volume / avg_volume)

        movers.append({
            "symbol": symbol,
            "score": score,
            "price": price,
            "volume": volume,
            "change_pct": round(change * 100, 2)
        })

    movers = sorted(movers, key=lambda x: x["score"], reverse=True)

    top = movers[:10]

    print(f"Top movers found: {len(top)}")

    return top





def normalize_symbol(symbol: str) -> str:
    # NASDAQ:AAPL → AAPL
    return symbol.split(":")[-1].upper()



def get_market_status():
    clock = trading_client.get_clock()
    return {
        "is_open": clock.is_open,
        "next_open": clock.next_open.isoformat(),
        "next_close": clock.next_close.isoformat(),
        "timestamp": clock.timestamp.isoformat()
    }


TF_MAP = {
    "1Min": TimeFrame(1, TimeFrameUnit.Minute),
    "5Min": TimeFrame(5, TimeFrameUnit.Minute),
    "15Min": TimeFrame(15, TimeFrameUnit.Minute),
    "1Day": TimeFrame.Day
}

DATA_DIR = "data"

BASE_DIR = Path(DATA_DIR).resolve()

@app.get("/api/files/{filename:path}")
async def get_file(filename: str):
    file_path = (BASE_DIR / filename).resolve()

    # 🔒 सुरक्षा: prevent path traversal
    if not str(file_path).startswith(str(BASE_DIR)):
        raise HTTPException(status_code=403, detail="Forbidden")

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    def generate():
        with open(file_path, "r") as f:
            for line in f:
                yield line

    return StreamingResponse(
        generate(),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"inline; filename={file_path.name}"
        }
    )

@app.get("/api/orders")
async def get_orders():
    try:
        request = GetOrdersRequest(
            limit=100
        )

        orders = trading_client.get_orders(request)

        return [
            {
                "id": o.id,
                "symbol": o.symbol,
                "side": o.side,
                "qty": o.qty,
                "filled_qty": o.filled_qty,
                "type": o.type,
                "status": o.status,
                "limit_price": o.limit_price,
                "stop_price": o.stop_price,
                "created_at": o.created_at.isoformat() if o.created_at else None
            }
            for o in orders
        ]

    except Exception as e:
        return {"error": str(e)}, 500

def obj_to_dict(obj):
    """Convert alpaca-py objects to JSON-safe dicts"""
    return obj.model_dump()


@app.get("/api/assets")
async def get_assets():
    req = GetAssetsRequest(asset_class=AssetClass.US_EQUITY)
    assets = trading_client.get_all_assets(req)

    payload = [a.model_dump(mode="json") for a in assets]

    return Response(
        json.dumps(payload, default=str),
        mimetype="application/json"
    )


@app.post("/api/order")
async def place_order():
    data = request.get_json()

    symbol = data.get("symbol")
    qty = int(data.get("qty", 1))
    side = data.get("side")

    if side not in ["buy", "sell"]:
        return {"error": "Invalid side"}, 400

    order = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
        time_in_force=TimeInForce.DAY
    )

    submitted = trading_client.submit_order(order)

    return submitted.model_dump()

@app.get("/api/account")
async def account():
    acct = trading_client.get_account()
    return {
        "equity": float(acct.equity),
        "cash": float(acct.cash),
        "buying_power": float(acct.buying_power),
        "status": acct.status
    }

@app.get("/api/bars")
async def get_stock_bars(symbols: str = "TQQQ,SPY",tf: str = "1D"):
    try:
        print("SYMBOLS receiveIVEDS:", symbols)
        #symbols = request.args.get("symbols", "TQQQ").split(",")
        
        #tf = request.args.get("tf", "1D")
        symbols = symbols.split(",")
        timeframe_map = {
            "1D": TimeFrame.Day,
            "1H": TimeFrame.Hour,
            "15Min": TimeFrame(15, TimeFrameUnit.Minute),
            "5Min": TimeFrame(5, TimeFrameUnit.Minute),
        }

        timeframe = timeframe_map.get(tf, TimeFrame.Day)
        now = datetime.now(UTC)

        if tf == "5Min":
            start = now - timedelta(days=20)
        elif tf == "15Min":
            start = now - timedelta(days=90)
        else:
            start = datetime(2023, 1, 1, tzinfo=UTC)

        req = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=timeframe,
            start=start,
        )

        bars = data_client.get_stock_bars(req).df.reset_index()

        if bars.empty:
            return {"message": "No data returned"}, 200

        bars = bars.sort_values(["symbol", "timestamp"])
        bars["time"] = bars["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

        dfs = []

        for symbol in symbols:
            sdf = bars[bars["symbol"] == symbol].copy()
            if sdf.empty:
                continue

            open_ = sdf["open"].values
            high = sdf["high"].values
            low = sdf["low"].values
            close = sdf["close"].values
            volume = sdf["volume"].values

            # -------------------------------
            # Indicators
            # -------------------------------
            sdf["ROC"] = talib.ROC(close, timeperiod=10)

            macd, macd_signal, macd_hist = talib.MACD(
                close, fastperiod=12, slowperiod=26, signalperiod=9
            )
            sdf["MACD"] = macd
            sdf["Signal"] = macd_signal
            sdf["Histogram"] = macd_hist

            sdf["EMA_50"] = talib.EMA(close, timeperiod=50)
            sdf["ADX"] = talib.ADX(high, low, close, timeperiod=14)
            sdf["ATR_14"] = talib.ATR(high, low, close, timeperiod=14)
            sdf["C-O"] = sdf["close"] - sdf["open"]

            # ensure symbol column exists
            sdf["symbol"] = symbol

            sdf = sdf[
                [
                    "symbol",
                    "time",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "ROC",
                    "Histogram",
                    "MACD",
                    "Signal",
                    "ADX",
                    "C-O",
                    "EMA_50",
                    "ATR_14",
                ]
            ]

            dfs.append(sdf)

        final_df = pd.concat(dfs, ignore_index=True)

        # sort for clean structure
        final_df = final_df.sort_values(["symbol", "time"])

        # optimize memory + speed
        final_df["symbol"] = final_df["symbol"].astype("category")

        os.makedirs("data", exist_ok=True)
        filename = f"data/bars_{tf}.csv"

        final_df.to_csv(filename, index=False)
        print(bars.columns)
        print(bars.head())
        print(bars["symbol"].unique() if "symbol" in bars else "NO SYMBOL COLUMN")
        return {
            "status": "success",
            "symbols": symbols,
            "timeframe": tf,
            "rows_written": len(final_df),
            "file": filename,
        }

    except Exception as e:
        return {"error": str(e)}, 500



@app.get("/api/backtest")
async def backtest_from_csv(
    filename: str = APIQuery("bars_5Min.csv"),
    sl: float = APIQuery(0.0075),
    tp: float = APIQuery(0.015),
):
    try:
        # -------------------------------
        # VALIDATION
        # -------------------------------
        if not filename:
            raise HTTPException(status_code=400, detail="file parameter required")

        path = os.path.join("data", filename)

        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="file not found")

        # -------------------------------
        # LOAD DATA (⚠️ heavy operation)
        # -------------------------------
        df = pd.read_csv(path).dropna().reset_index(drop=True)

        if len(df) < 50:
            raise HTTPException(status_code=400, detail="Not enough data")

        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values(["symbol", "time"]).reset_index(drop=True)

        results = []
        total_trades = 0

        # -------------------------------
        # RUN BACKTEST PER SYMBOL
        # -------------------------------
        for symbol, sdf in df.groupby("symbol"):

            sdf = sdf.reset_index(drop=True)

            # ENTRY SIGNAL
            sdf["entry"] = (
                (sdf["Histogram"] > sdf["Histogram"].shift(1)) &
                (sdf["ADX"] > 22) &
                (sdf["ROC"] > 0) &
                (sdf["ATR_14"] > sdf["ATR_14"].rolling(50).mean()) &
                (sdf["close"] > sdf["EMA_50"])
            )

            equity = 1.0
            equity_curve = []
            trades = []

            in_trade = False
            entry_price = 0.0

            for i in range(1, len(sdf) - 1):

                ts = sdf["time"].iat[i]
                close_price = float(sdf["close"].iat[i])
                next_open = float(sdf["open"].iat[i + 1])

                # MANAGE TRADE
                if in_trade:
                    ret = (close_price - entry_price) / entry_price

                    if ret >= tp or ret <= -sl:
                        equity *= (1 + ret)
                        trades.append(ret)
                        in_trade = False

                    equity_curve.append(equity)
                    continue

                # TIME FILTER
                hour = ts.hour
                minute = ts.minute

                liquid = (
                    (hour == 9 and minute >= 30) or
                    (10 <= hour < 11) or
                    (14 <= hour < 16)
                )

                if not liquid:
                    equity_curve.append(equity)
                    continue

                # ENTRY
                if sdf["entry"].iat[i]:
                    in_trade = True
                    entry_price = next_open

                equity_curve.append(equity)

            # CLOSE LAST TRADE
            if in_trade:
                last_close = float(sdf["close"].iat[-1])
                ret = (last_close - entry_price) / entry_price
                equity *= (1 + ret)
                trades.append(ret)
                equity_curve.append(equity)

            equity_series = pd.Series(equity_curve)

            max_dd = (equity_series / equity_series.cummax() - 1).min()

            wins = [t for t in trades if t > 0]
            losses = [t for t in trades if t <= 0]

            win_rate = (len(wins) / len(trades)) if trades else 0

            avg_win = sum(wins) / len(wins) if wins else 0
            avg_loss = sum(losses) / len(losses) if losses else 0

            gross_profit = sum(wins)
            gross_loss = abs(sum(losses))

            profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0

            total_trades += len(trades)

            results.append({
                "symbol": symbol,
                "bars_tested": len(sdf),
                "trades": len(trades),
                "win_rate_pct": round(win_rate * 100, 2),
                "average_win_pct": round(avg_win * 100, 3),
                "average_loss_pct": round(avg_loss * 100, 3),
                "profit_factor": round(profit_factor, 3),
                "max_drawdown_pct": round(max_dd * 100, 2),
                "final_equity": round(equity, 4)
            })

        return jsonable_encoder({
            "file": filename,
            "symbols_tested": len(results),
            "total_trades": total_trades,
            "stop_loss_pct": sl * 100,
            "take_profit_pct": tp * 100,
            "results": results
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/")
def health():
    return {"status": "ok"}, 200

@app.get("/screen")
def screen():
    screener = StockScreener()
    screener.load_dataframes()
    return screener.screener_df.head(10).to_dict(orient="records")

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

def fetch_ohlc(symbol, limit=200, timeframe=TimeFrame.Day):
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=timeframe,
        limit=limit,
        adjustment="raw"
    )

    bars = data_client.get_stock_bars(request).df

    if bars.empty:
        return None

    df = bars.reset_index()
    df = df[df["symbol"] == symbol]

    df.rename(columns={
        "timestamp": "time",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume"
    }, inplace=True)

    df["time"] = df["time"].astype("int64") // 10**9
    return df

@app.delete("/api/positions/{symbol}")
async def close_position(symbol):
    try:
        url = f"{BASE_URL}/v2/positions/{symbol}"

        r = requests.delete(url, headers=HEADERS)

        if r.status_code not in [200, 204]:
            return jsonify({
                "status": "error",
                "message": r.text
            }), r.status_code

        return {
            "status": "success",
            "message": f"Position for {symbol} closed"
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }, 500

@app.get("/api/positions")
async def get_positions():
    try:
        positions = trading_client.get_all_positions()

        return [
            {
                "symbol": p.symbol,
                "qty": float(p.qty),
                "side": p.side,
                "avg_entry_price": float(p.avg_entry_price),
                "current_price": float(p.current_price),
                "market_value": float(p.market_value),
                "unrealized_pl": float(p.unrealized_pl)
            }
            for p in positions
        ]

    except Exception as e:
        return {"error": str(e)}, 500


@app.get("/api/newchart")
async def newchart( symbol: str = "AAPL",):
    #symbol = request.args.get("symbol", "AAPL")

    # fake data (replace with real market data)
    x = np.arange(0, 50)
    y = np.cumsum(np.random.randn(50))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y, label=symbol)
    ax.set_title(f"{symbol} Price Chart")
    ax.legend()
    ax.grid(True)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    return send_file(buf, mimetype="image/png")



def detect_bull_flag(df):
    patterns = []

    df["returns"] = df["close"].pct_change()

    for i in range(20, len(df)):
        impulse = df["returns"].iloc[i-10:i-5].sum()
        consolidation = df["returns"].iloc[i-5:i].abs().mean()

        if impulse > 0.10 and consolidation < 0.01:
            patterns.append({
                "type": "bull_flag",
                "points": [
                    {
                        "time": int(df["time"].iloc[i-10]),
                        "price": float(df["low"].iloc[i-10])
                    },
                    {
                        "time": int(df["time"].iloc[i]),
                        "price": float(df["high"].iloc[i])
                    }
                ]
            })

    return patterns


@app.get("/api/patterns")
async def get_patterns( symbol: str = "TQQQ", tf: str = "1D"):
    #symbol = sy
    timeframe = tf

    tf = TimeFrame.Day if timeframe == "1D" else TimeFrame.Minute

    df = fetch_ohlc(symbol, timeframe=tf)

    if df is None:
        return jsonify({"error": "No data"}), 400

    patterns = detect_bull_flag(df)

    return {
        "symbol": symbol,
        "patterns": patterns
    }



def fetch_latest_rest(symbols):
    req = StockLatestBarRequest(symbol_or_symbols=symbols)
    bars = data_client.get_stock_latest_bar(req)

    for symbol, bar in bars.items():
        if not bar:
            continue

        latest_prices[symbol] = {
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
            "vwap": bar.vwap,
            "timestamp": bar.timestamp,
            "source": "rest"
        }


@app.get("/api/watchlist/prices")
def watchlist_prices(symbols: str = "TQQQ,SPY"):
    #symbols = request.args.get("symbols", "")
    symbols = [s.split(":")[-1] for s in symbols.split(",")]

    fetch_latest_rest(symbols)

    print("LATEST_PRICES:", latest_prices)
    return latest_prices


@app.post("/api/watchlist")
async def add_to_watchlist(request: Request):
    data = await request.json(silent=True) or {}
    symbol = data.get("symbol")

    if not symbol:
        return jsonify({"error": "symbol required"}), 400

    symbols = read_watchlist()
    symbols.add(symbol.upper())
    write_watchlist(symbols)

    #subscribe_watchlist_quotes()

    return jsonify({"success": True})




@app.delete("/api/watchlist/{symbol}")
async def remove_from_watchlist(symbol):
    symbols = read_watchlist()
    symbols.discard(symbol.upper())
    write_watchlist(symbols)

    #subscribe_watchlist_quotes()

    return jsonify({"success": True})

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


@app.get("/api/watchlist")
async def get_watchlist():
    symbols = sorted(read_watchlist())

    return {
        "count": len(symbols),
        "data": [{"symbol": s} for s in symbols]
    }


def parse_timeframe(tf: str):
    if tf == "1m":
        return TimeFrame.Minute
    if tf == "5m":
        return TimeFrame(5, TimeFrameUnit.Minute)
    if tf == "15m":
        return TimeFrame(15, TimeFrameUnit.Minute)
    raise ValueError("Invalid timeframe")

@app.get("/api/chart")
async def chart(symbol:str,indicators: str):
    #symbol = request.args.get("symbol", "AAPL")
    #indicators = request.args.get("indicators", "")
    indicator_list = [i for i in indicators.split(",") if i]
    #indicator = request.args.get("indicator", "RSI")
    timeframe = request.args.get("tf", "1m")

    tf = parse_timeframe(timeframe)

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=180)  
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=tf,
        start=start,
        end=end,        
        feed=DataFeed.IEX,
    )

    bars = data_client.get_stock_bars(req).df

    if bars.empty:
        return {"error": "No data"}, 400

    # Single-symbol → flat index
    df = bars.reset_index()

    df.columns = [c.lower() for c in df.columns]


    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)

    if missing:
        return {
            "error": "Missing required columns from Alpaca",
            "missing": list(missing),
            "columns": list(df.columns)
        }, 500

  

    # Standardize columns
    df.rename(columns={
        "timestamp": "time",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
    }, inplace=True)

    # Convert to UNIX seconds (required by Lightweight Charts)
    df["time"] = df["time"].astype("int64") // 1_000_000_000

    candles = df[["time", "open", "high", "low", "close"]].to_dict("records")
    volume = df[["time", "volume"]].rename(columns={"volume": "value"}).to_dict("records")

    indicator_payload = {}    
    for name in indicator_list:
        if name == "RSI":
            values = talib.RSI(df["close"], timeperiod=14)
        elif name == "SMA":
            values = talib.SMA(df["close"], timeperiod=20)
        elif name == "EMA":
            values = talib.EMA(df["close"], timeperiod=20)
        
        if values is None:
            continue

    indicator_payload[name] = [
        {"time": int(t), "value": float(v)}
        for t, v in zip(df["time"], values)
        if not pd.isna(v)
    ]    

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "candles": candles,
        "volume": volume,
        "indicators": indicator_payload        
    }



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






INTERVAL_MAP = {
    "1m": Interval.INTERVAL_1_MINUTE,
    "1h": Interval.INTERVAL_1_HOUR,
    "4h": Interval.INTERVAL_4_HOURS,
    "1d": Interval.INTERVAL_1_DAY,
    "1w": Interval.INTERVAL_1_WEEK,
    "1M": Interval.INTERVAL_1_MONTH,
}


@app.get("/api/fchart2")
async def fchart2(symbol:str):
    try:
        raw_symbol = symbol
        if not raw_symbol:
            return {"error": "Symbol required"}, 400

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
            return {
                "meta": {
                    "symbol": symbol,
                    "exchange": exchange,
                    "currency": EXCHANGE_CURRENCY_MAP.get(exchange, "USD"),
                },
                "quotes": [],
                "indicators": {"rsi": [], "macd": []},
            }

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
        sma_raw = talib.SMA(closes_np, timeperiod=20)

        sma = [
            {
                "time": quotes[i]["date"],
                "value": round(float(sma_raw[i]), 4),
            }
            for i in range(len(sma_raw))
            if not np.isnan(sma_raw[i])
        ]




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
        return {
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
        }

    except Exception as e:
        print("⛔ API ERROR:", e)
        return {"error": str(e)}, 500


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
  
@app.get("/api/market-movers")
async def market_movers_unified(category:str = "gainers",limit:int = 50):
    #category = request.args.get("category", "gainers")
    #limit = int(request.args.get("limit", 50))

    if category not in VALID_CATEGORIES:
        return {
            "success": False,
            "error": f"Invalid category: {category}"
        }, 400

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

    return {
        "success": True,
        "category": category,
        "session": VALID_CATEGORIES[category],
        "count": len(filtered),
        "data": filtered
    }




@app.get("/api/technical-analysis")
async def get_technical_analysis(symbol:str,exchange:str = "NASDAQ",interval:str = "1D"):
    try:
        symbol = symbol
        exchange = exchange
        interval = interval

        if not symbol:
            return {"success": False, "error": "symbol is required"}, 400

        handler = TA_Handler(
            symbol=symbol.upper(),
            exchange=exchange.upper(),
            screener="america",
            interval=INTERVAL_MAP.get(interval, Interval.INTERVAL_1_DAY)
        )

        analysis = handler.get_analysis()

        return {
            "success": True,
            "symbol": symbol.upper(),
            "exchange": exchange.upper(),
            "interval": interval,
            "summary": analysis.summary,
            "oscillators": analysis.oscillators,
            "moving_averages": analysis.moving_averages,
            "indicators": analysis.indicators
        }

    except Exception as e:
        return {"success": False, "error": str(e)}, 500


#https://candlestick-screener.onrender.com/api/scan_top_gainers?strategy=momentum&min_price=8&max_price=200&min_volume=900000&top_gainers=1&limit=100
def safe_float(x, default=0.0):
    try:
        x = float(x)
        if math.isnan(x) or math.isinf(x):
            return default
        return x
    except:
        return default

@app.get("/api/scan_top_gainers")
async def scan_top_gainers(strategy:str,top_gainers:int,min_price:float,max_price:float,min_volume:int,limit:int):
    try:
        # -----------------------------
        # Query params
        # -----------------------------
        #strategy = request.args.get("strategy")
        #top_gainers = top_gainers, "0") == "1"

        #min_price = float(request.args.get("min_price", 5))
        #max_price = float(request.args.get("max_price", 500))
        #min_volume = int(request.args.get("min_volume", 1_000_000))
        #limit = int(request.args.get("limit", 100))

        # -----------------------------
        # Base TradingView query
        # -----------------------------
        print('MYDEBUGGGG>>>>>',max_price)
        
        query = (
            Query()
            .set_markets("america")
            .select(
                "name",
                "close",
                "volume",
                "EMA5",
                "EMA20",
                "type",
                "RSI",
                "change",
                "change_abs",
                "market_cap_basic",
                "relative_volume_10d_calc",
                "Value.Traded",
                "AnalystRating",
                "Recommend.All"
            )     
            .where(
                #Column("volume") >= min_volume,
                Column("market_cap_basic") > 1_000_000_000, 
                Column('close').between(min_price, max_price),
                Column('close').between(Column('EMA5'), Column('EMA20')),
                Column('type').isin(['stock', 'fund']),
                Column("change").between(2, 20),         # 🔥 BIG FIX
                #Column('MACD.macd') >= Column('MACD.signal'),
                #Column("relative_volume_10d_calc") > 1.0,            # 🔥 momentum confirmation
                Column("Value.Traded") > 20_000_000,    
                Column("Recommend.All") <= 0
            )
        )
        # NASDAQ,NYSE,AMEX
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
                close = safe_float(r.get("close"))
                volume = int(safe_float(r.get("volume")))
                change_abs = safe_float(r.get("change_abs"))

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

                    # 🔥 sanitize entire row before appending
                cleaned_row = {}
                for k, v in r.items():
                    if isinstance(v, float):
                        if math.isnan(v) or math.isinf(v):
                            cleaned_row[k] = None   # or 0
                        else:
                            cleaned_row[k] = v
                    else:
                        cleaned_row[k] = v

                cleaned.append(cleaned_row)

            except Exception:
                continue

            if len(cleaned) >= limit:
                break

        return {
            "success": True,
            "count": len(cleaned),
            "strategy": strategy,
            "top_gainers": top_gainers,
            "data": cleaned
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }, 500



#http://192.168.150.105:5000/api/scan?min_price=10&max_price=200&min_volume=2000000&min_rsi=50&limit=25
@app.get("/api/scan")
async def scan_stocks(min_price:float,max_price:float,min_volume:int,min_rsi:float,limit:int ):
    try:
        #min_price = float(request.args.get("min_price", 5))
        #max_price = float(request.args.get("max_price", 500))
        #min_volume = int(request.args.get("min_volume", 1_000_000))
        #min_rsi = float(request.args.get("min_rsi", 40))
        #limit = int(request.args.get("limit", 20))

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

        return {
            "success": True,
            "count": total_count,
            "data": rows
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }, 500

@app.get("/api/tv/volume-leaders")
async def volume_leaders(page:int = 1,limit:int = 100):
    """
    Example:
    /api/tv/volume-leaders?page=1&limit=100
    """

    #page = int(request.args.get("page", 1))      # 1-based
    #limit = int(request.args.get("limit", 100))

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

        return {
            "success": True,
            "page": page,
            "limit": limit,
            "total": total_count,
            "data": data
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }, 500


@app.get("/api/tv/best-performing")
async def best_performing(page:int = 1,limit:int = 100):
    """
    Example:
    /api/tv/volume-leaders?limit=100
    """
    #page = int(request.args.get("page", 1))      # 1-based
    #limit = int(request.args.get("limit", 100))

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
            return {
                "success": True,
                "page": page,
                "limit": limit,
                "total": total_count,
                "data": []
            }


        if hasattr(data, "to_dict"):
            data = (
                data
                .replace([np.nan, np.inf, -np.inf], None)
                .reset_index(drop=True)
                .to_dict(orient="records")
            )

        return {
            "success": True,
            "total": total_count,
            "data": data
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }, 500


@app.get("/api/strategy-chart")
async def strategy_chart(symbol:str,fast:int = 20, slow:int = 50, limit:int = 200  ):
    #symbol = request.args.get("symbol", "AAPL")
    #fast = int(request.args.get("fast", 20))
    #slow = int(request.args.get("slow", 50))
    #limit = int(request.args.get("limit", 200))

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=420)
    # 1️⃣ Fetch historical OHLCV from Alpaca
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
        limit=300,
        feed=DataFeed.IEX,
        adjustment="raw"
    )
    bars = data_client.get_stock_bars(req).df

    if bars.empty:
        return "No data", 404

    df = bars.loc[symbol].copy()

    # 2️⃣ Prepare data
    close = df["close"].values

    df["sma_fast"] = talib.SMA(close, timeperiod=fast)
    df["sma_slow"] = talib.SMA(close, timeperiod=slow)

    # 3️⃣ Strategy logic
    df["signal"] = 0
    df.loc[df["sma_fast"] > df["sma_slow"], "signal"] = 1
    df["position"] = df["signal"].diff()

    # 4️⃣ Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(df.index, df["close"], label="Close", linewidth=1)
    ax.plot(df.index, df["sma_fast"], "--", label=f"SMA {fast}")
    ax.plot(df.index, df["sma_slow"], "--", label=f"SMA {slow}")

    buys = df[df["position"] == 1]
    sells = df[df["position"] == -1]

    ax.scatter(buys.index, buys["close"], marker="^", s=90, label="Buy")
    ax.scatter(sells.index, sells["close"], marker="v", s=90, label="Sell")

    ax.set_title(f"{symbol} SMA Crossover (Alpaca)")
    ax.legend()
    ax.grid(True)

    # 5️⃣ Return PNG
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    buf.seek(0)

    print('Buffer-length===',len(buf.getvalue()))
    headers = {
        "Cache-Control": "no-store",
        "X-Content-Type-Options": "nosniff"
    }

    return Response(content=buf.getvalue(), media_type="image/png", headers=headers)


'''
@app.get("/scanner/momentum")
async def momentum_scanner():
    movers = data_client.get_market_movers()

    filtered = []

    for s in movers.gainers:
        if (
            s.price > 10 and
            s.volume > 1_000_000 and
            s.price * s.volume > 20_000_000
        ):
            filtered.append({
                "symbol": s.symbol,
                "price": s.price,
                "change": s.change_percent,
                "volume": s.volume,
                "dollar_vol": s.price * s.volume
            })

    return sorted(filtered, key=lambda x: x["change"], reverse=True)[:20]
'''

#-------------------------------------------------
# Helper functions
# -------------------------------------------------

def find_pivots(series, window=4):
    highs, lows = [], []
    for i in range(window, len(series) - window):
        chunk = series[i - window:i + window + 1]
        if series[i] == chunk.max():
            highs.append(i)
        if series[i] == chunk.min():
            lows.append(i)
    return highs, lows


def trendline(x, y):
    m, b = np.polyfit(x, y, 1)
    return m, b


# -------------------------------------------------
# Flask endpoint
# -------------------------------------------------

@app.get("/api/tqqq/patterns")
async def tqqq_patterns():
    symbol = "TQQQ"
    lookback_days = 200

    # 1️⃣ Fetch OHLCV from Alpaca
    end = datetime.utcnow()
    start = end - timedelta(days=lookback_days)

    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
        feed=DataFeed.IEX,
        adjustment="raw"
    )

    bars = data_client.get_stock_bars(req).df
       

    if bars.empty:
        raise HTTPException(status_code=404, detail="No data")

    df = bars.loc[symbol].copy().reset_index(drop=True)

    close = df["close"].values

    # 2️⃣ Detect pivots
    pivot_highs, pivot_lows = find_pivots(close, window=4)

    # Use recent pivots for pattern structure
    recent_highs = pivot_highs[-6:]
    recent_lows = pivot_lows[-6:]

    upper = lower = None

    if len(recent_highs) >= 3:
        x = np.array(recent_highs)
        y = close[x]
        upper = trendline(x, y)

    if len(recent_lows) >= 3:
        x = np.array(recent_lows)
        y = close[x]
        lower = trendline(x, y)

    # 3️⃣ Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(close, label="TQQQ Close", linewidth=1)

    # Pivot points
    ax.scatter(
        pivot_highs, close[pivot_highs],
        s=40, marker="^", label="Pivot Highs"
    )
    ax.scatter(
        pivot_lows, close[pivot_lows],
        s=40, marker="v", label="Pivot Lows"
    )

    # Trendlines
    x_vals = np.arange(len(close))
    if upper:
        ax.plot(
            x_vals,
            upper[0] * x_vals + upper[1],
            linestyle="--",
            label="Resistance"
        )
    if lower:
        ax.plot(
            x_vals,
            lower[0] * x_vals + lower[1],
            linestyle="--",
            label="Support"
        )

    ax.set_title("TQQQ Price Pattern Mapping (Alpaca)")
    ax.legend()
    ax.grid(True)

    # 4️⃣ Return image
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    print('my tqqqbuff-length===',len(buf.getvalue()))
    headers = {
        "Cache-Control": "no-store",
        "X-Content-Type-Options": "nosniff"
    }

    return Response(content=buf.getvalue(), media_type="image/png", headers=headers)


@app.get("/api/strategy-rsi")
async def strategy_rsi(symbol:str,fast:int = 20, slow:int=50,rsi:int =14,days:int = 180 ):
    #symbol = request.args.get("symbol", "AAPL").upper()
    #fast = int(request.args.get("fast", 20))
    #slow = int(request.args.get("slow", 50))
    rsi_len = rsi
    lookback_days = days

    # -------- 1️⃣ Fetch Alpaca bars --------
    end = datetime.utcnow()
    start = end - timedelta(days=lookback_days)

    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
        adjustment="raw",
        feed=DataFeed.IEX
    )

    bars = data_client.get_stock_bars(req).df

    if bars.empty or symbol not in bars.index.get_level_values(0):
        return f"No data for {symbol}", 404

    df = bars.loc[symbol].copy()

    # -------- 2️⃣ Indicators --------
    close = df["close"].values

    df["sma_fast"] = talib.SMA(close, timeperiod=fast)
    df["sma_slow"] = talib.SMA(close, timeperiod=slow)
    df["rsi"] = talib.RSI(close, timeperiod=rsi_len)

    # -------- 3️⃣ Strategy (SMA crossover) --------
    df["signal"] = 0
    df.loc[df["sma_fast"] > df["sma_slow"], "signal"] = 1
    df["position"] = df["signal"].diff()

    buys = df[df["position"] == 1]
    sells = df[df["position"] == -1]

    # -------- 4️⃣ Plot --------
    fig, (ax_price, ax_rsi) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]}
    )

    # ---- Price pane
    ax_price.plot(df.index, df["close"], label="Close", linewidth=1)
    ax_price.plot(df.index, df["sma_fast"], "--", label=f"SMA {fast}")
    ax_price.plot(df.index, df["sma_slow"], "--", label=f"SMA {slow}")

    ax_price.scatter(
        buys.index, buys["close"],
        marker="^", s=80, label="Buy"
    )
    ax_price.scatter(
        sells.index, sells["close"],
        marker="v", s=80, label="Sell"
    )

    ax_price.set_title(f"{symbol} — SMA Strategy + RSI (Alpaca)")
    ax_price.legend()
    ax_price.grid(True)

    # ---- RSI pane
    ax_rsi.plot(df.index, df["rsi"], label="RSI", linewidth=1)
    ax_rsi.axhline(70, linestyle="--", linewidth=1)
    ax_rsi.axhline(30, linestyle="--", linewidth=1)
    ax_rsi.set_ylim(0, 100)
    ax_rsi.set_ylabel("RSI")
    ax_rsi.grid(True)

    # -------- 5️⃣ Return image --------
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    print('Buffer-length===',len(buf.getvalue()))
    headers = {
        "Cache-Control": "no-store",
        "X-Content-Type-Options": "nosniff"
    }

    return Response(content=buf.getvalue(), media_type="image/png", headers=headers)



@app.get("/api/tv/top-gainers")
async def top_gainers(page:int = 1, limit:int = 100):
    
    #page = int(request.args.get("page", 1))      # 1-based
    #limit = int(request.args.get("limit", 100))

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
                'relative_volume_10d_calc',
                'Value.Traded',
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
                
            )
            .where(
                col('is_primary') == True,
                col('typespecs').has('common'),
                col('typespecs').has_none_of('preferred'),
                col('type') == 'stock',
                col('close').between(2, 10000),
                col('change') > 0,
                col('active_symbol') == True,
                Column("market_cap_basic") > 1_000_000_000, 
                #Column("RSI").between(55, 70),
                Column("change").between(2, 20),         # 🔥 BIG FIX
                Column("relative_volume_10d_calc") > 1.5,            # 🔥 momentum confirmation
                Column("Value.Traded") > 20_000_000,    
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

        return {
            "success": True,
            "total": total_count,
            "data": data
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }, 500


@app.get("/api/tv/small-cap")
async def small_cap(page:int = 1, limit:int = 100):
    
    #page = int(request.args.get("page", 1))      # 1-based
    #limit = int(request.args.get("limit", 100))

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

        return {
            "success": True,
            "total": total_count,
            "data": data
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }, 500

#http://192.168.150.105:5000/api/tvscreener?min_rsi=50&max_rsi=65&sort_by=rsi&sort_order=desc
'''        
@app.get("/api/tvscreener")
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

    try:
        _, df = query.get_scanner_data()
        df["change_percent"] = (df["change"] / (df["close"] - df["change"])) * 100
        return jsonify(df.to_dict(orient="records"))
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500
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
        




def symbols_to_quoted_string(results):
    symbols = [item["symbol"] for item in results if "symbol" in item]
    return ",".join([f'"{s}"' for s in symbols])


@app.get("/api/sma")
async def calculate_sma(symbols:str):
    #symbols = request.args.get("symbols", "")
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

    return results




if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port,  use_reloader=False, debug=True)
