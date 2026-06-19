import logging 
import asyncio
import pandas as pd
import numpy as np
import logging
from datetime import datetime

from alpaca.data.live import StockDataStream
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from config import ALPACA_KEY, ALPACA_SECRET

from alpaca.data.historical import StockHistoricalDataClient

from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta
from alpaca.data.enums import DataFeed
import argparse
from collections import defaultdict


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

log = logging.getLogger("BOT")

trade_buffer = defaultdict(list)
current_minute = {}

#SYMBOLS = ["AAPL", "TSLA", "NVDA"]  # change as needed
TIMEFRAME = "15Min"
CAPITAL_PER_TRADE = 1000

logging.basicConfig(level=logging.INFO)

trades = []
data = {}
last_entry_price = {}


last_loss_price = {}   # {symbol: price}
last_loss_time = {} 
retry_count = {}
last_trade_bar = {}

last_breakout_price = {}



SIM_MODE = True   # 🔥 toggle this
REPLAY_SPEED = 0.01
LOOKBACK_DAYS = 4
TIMEFRAME = TimeFrame.Minute


positions = {}
trade_history = []



parser = argparse.ArgumentParser()

parser.add_argument("--symbols")

args = parser.parse_args()

SYMBOLS = args.symbols.split(",")


historical_data = data.copy()   # original
live_data = {s: pd.DataFrame() for s in SYMBOLS}



# ================= CLIENTS =================
trading_client = TradingClient(ALPACA_KEY, ALPACA_SECRET, paper=True)
data_stream = StockDataStream(ALPACA_KEY, ALPACA_SECRET)

# ================= STORAGE =================
data = {symbol: pd.DataFrame() for symbol in SYMBOLS}





historical_client = StockHistoricalDataClient(ALPACA_KEY, ALPACA_SECRET)

def preload_data():
    global historical_data

    print("Loading historical data...")

    end = datetime.utcnow()
    start = end - timedelta(days=LOOKBACK_DAYS)

    request = StockBarsRequest(
        symbol_or_symbols=SYMBOLS,
        timeframe=TimeFrame.Minute,
        start=start,
        end=end,
        feed=DataFeed.IEX
    )

    bars = historical_client.get_stock_bars(request)

    for symbol in SYMBOLS:
        try:
            df = bars.df.loc[symbol].reset_index()

            df = df[["timestamp", "open", "high", "low", "close", "volume"]]
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp").set_index("timestamp")

            if len(df) < 100:
                continue

            historical_data[symbol] = df
            live_data[symbol] = df.iloc[:50].copy()  # warmup

            print(f"{symbol}: {len(df)} rows")

        except Exception as e:
            print(f"Error loading {symbol}: {e}")


async def run_simulation():
    print("Starting SIMULATION mode...")
    for symbol in historical_data:
        print(f"{symbol}: {len(historical_data[symbol])} rows")
    

    # Get valid symbols only
    symbols = [symbol for symbol in SYMBOLS if symbol in historical_data and len(historical_data[symbol]) > 50]

    for symbol, df in historical_data.items():
        print(symbol, type(df), getattr(df, "shape", "NO SHAPE"))

    print("SYMBOLS:", symbols)
    #print("DATA KEYS:", list(indices.keys()))
    print("DATA KEYS:", list(historical_data.keys()))

    if not symbols:
        print("❌ No valid symbols to simulate")
        return

    # Track current index per symbol
    indices = {symbol: 50 for symbol in symbols}  # start after indicators ready

    active = True
    #print("DATA TYPE:", type(data))
    #print("DATA VALUE:", data)


    

    while active:
        active = False

        for symbol in indices:
            #print('executing loopnow....',)
            df = historical_data[symbol]
            i = indices[symbol]

            if i >= len(df):
                continue  # this symbol finished

            row = df.iloc[i]

            bar = SimBar(symbol, row)

            await handle_bar(bar)

            indices[symbol] += 1
            active = True  # at least one symbol still running

        await asyncio.sleep(REPLAY_SPEED)

    print("Simulation complete")


class SimBar:
    def __init__(self, symbol, row):
        self.symbol = symbol
        self.open = row["open"]
        self.high = row["high"]
        self.low = row["low"]
        self.close = row["close"]
        self.volume = row["volume"]
        #self.timestamp = pd.to_datetime(row.name)

        ts = pd.to_datetime(row.name)

        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")   # or "America/New_York"
        else:
            ts = ts.tz_convert("UTC")

        self.timestamp = ts


# ================= POSITION CLASS =================
class Position:
    def __init__(self, symbol, entry_price, qty, stop_loss, target,atr):
        self.symbol = symbol
        self.entry_price = entry_price
        self.qty = qty
        self.stop_loss = stop_loss
        self.target = target
        self.active = True
        self.atr = atr        
        self.exit = None
        self.exit_reason = None
        self.pnl = 0
     
    def close(self, exit_price, reason):
        self.exit = exit_price
        self.exit_reason = reason
        self.pnl = (exit_price - self.entry_price) * self.qty
        self.active = False
            
    def update(self, current_price):
        if current_price <= self.stop_loss:
            return "STOP_LOSS"

        if current_price >= self.target:
            return "TARGET"

        # trailing stop
        new_sl = current_price - self.atr * 1.0
        self.stop_loss = max(self.stop_loss, new_sl)
            #self.stop_loss = max(self.stop_loss, current_price * 0.998)

        return "HOLD"

# ================= INDICATORS =================
def add_indicators(df):
    df = df.copy()
    df.index = pd.to_datetime(df.index)

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")


    df["ema9"] = df["close"].ewm(span=9).mean()
    df["ema20"] = df["close"].ewm(span=20).mean()

    # ATR
    df["tr"] = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            abs(df["high"] - df["close"].shift()),
            abs(df["low"] - df["close"].shift())
        )
    )
    df["atr"] = df["tr"].rolling(14).mean()

    # VWAP (intraday reset)
    df["tp"] = (df["high"] + df["low"] + df["close"]) / 3
    df["date"] = df.index.to_series().dt.date 

    df["cum_vol"] = df.groupby("date")["volume"].cumsum()
    df["cum_tpv"] = (df["tp"] * df["volume"]).groupby(df["date"]).cumsum()

    df["vwap"] = df["cum_tpv"] / df["cum_vol"]

    df["vol_avg"] = df["volume"].rolling(20).mean()

    return df

def candle_wick_analysis(candle):
    body = abs(candle["close"] - candle["open"])
    upper_wick = candle["high"] - max(candle["close"], candle["open"])
    lower_wick = min(candle["close"], candle["open"]) - candle["low"]
    total_range = candle["high"] - candle["low"]

    # Avoid divide by zero
    if total_range == 0:
        return 0, 0

    upper_wick_ratio = upper_wick / total_range
    body_ratio = body / total_range

    return upper_wick_ratio, body_ratio

def generate_report(trade_history):
    print("\n" + "="*80)
    print("📊 TRADING SUMMARY REPORT")
    print("="*80)

    header = f"{'SYMBOL':<10} {'ENTRY':<10} {'EXIT':<10} {'QTY':<6} {'P&L':<12} {'RESULT':<10}"
    print(header)
    print("-"*80)

    total_pnl = 0
    wins = 0
    losses = 0

    for pos in trade_history:
        pnl = pos.pnl
        total_pnl += pnl

        result = "WIN" if pnl > 0 else "LOSS"
        if pnl > 0:
            wins += 1
        else:
            losses += 1

        print(f"{pos.symbol:<10} {pos.entry_price:<10.2f} {pos.exit:<10.2f} {pos.qty:<6} {pnl:<12.2f} {result:<10}")

    print("-"*80)

    total_trades = len(trade_history)
    win_rate = (wins / total_trades * 100) if total_trades else 0

    print(f"Total Trades : {total_trades}")
    print(f"Wins         : {wins}")
    print(f"Losses       : {losses}")
    print(f"Win Rate     : {win_rate:.2f}%")
    print(f"Total P&L    : {total_pnl:.2f}")
    print("="*80)





def is_fake_breakout(df):
    if len(df) < 25:
        return False

    last = df.iloc[-1]
    prev = df.iloc[-2]

    breakout_level = df["high"].rolling(20).max().iloc[-2]

    # Must be a breakout attempt first
    if last["high"] <= breakout_level:
        return False

    upper_wick_ratio, body_ratio = candle_wick_analysis(last)

    # 🔴 CONDITION 1: Strong rejection (long upper wick)
    wick_rejection = upper_wick_ratio > 0.4 and body_ratio < 0.5

    # 🔴 CONDITION 2: Close back inside range (FAILED breakout)
    failed_close = last["close"] < breakout_level

    # 🔴 CONDITION 3: No follow-through (next candle weak)
    weak_follow = (
        last["close"] < prev["close"] or
        last["close"] < last["open"]
    )

    # 🔴 CONDITION 4: Exhaustion move (too extended)
    move = (last["close"] - df["close"].iloc[-5]) / df["close"].iloc[-5]
    exhaustion = move > 0.004   # 0.4% spike in short time

    if wick_rejection or failed_close or weak_follow or exhaustion:
        return True

    return False

# ================= SIGNAL LOGIC =================
def is_strong_candle(row):
    body = abs(row["close"] - row["open"])
    rng = row["high"] - row["low"]
    return rng > 0 and body / rng > 0.6 and row["close"] > row["open"]

def detect_breakout(df):
    recent_high = df["high"].rolling(20).max().iloc[-2]
    return df["close"].iloc[-1] > recent_high

def volume_spike(df):
    if len(df) < 20:
        return False
    avg_volume = df["volume"].rolling(20).mean().iloc[-1]
    return df["volume"].iloc[-1] > avg_volume * 1.5

def get_vwap(df):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    return (tp * df["volume"]).cumsum().iloc[-1] / df["volume"].cumsum().iloc[-1]

def near_vwap(df, threshold=0.0015):
    if len(df) < 20:
        return False

    vwap = get_vwap(df)
    price = df["close"].iloc[-1]

    return abs(price - vwap) / vwap < threshold

def indicators_ready(df):
    return (
        len(df) > 50 and
        not df["ema"].isna().iloc[-1] and
        not df["atr"].isna().iloc[-1]
    )

def check_entry(df,symbol):
    if len(df) < 30:
        return False, None

    row = df.iloc[-1]
    prev = df.iloc[-2]


    price = row["close"]
    vwap = row["vwap"]
    atr = row["atr"]

    #print('atr===',atr)
    #print('vwap===',vwap)
    #print('price===',price)




    if pd.isna(vwap) or pd.isna(atr):
        return False, None
    
    last_price = last_entry_price.get(symbol) 
    

    trend_up = price > vwap
    pullback = price <= vwap + 0.5 * atr

    bullish = row["close"] > row["open"]
    breakout = row["close"] > prev["high"]

    if trend_up and pullback and bullish:
    
        #if (last_price is None) or (price <= last_price - atr):
        return True, {
            "entry_price": price,
            "stop_loss": price * 0.998,
            "target": price + 1.2 * atr,
            "atr": atr
        }

    return False, None



# ================= ORDER FUNCTIONS =================
sim_cash = 10000
sim_positions = {}

def place_order(symbol, side, qty, price):
    global sim_cash

    if side == OrderSide.BUY:
        cost = qty * price
        if sim_cash >= cost:
            sim_cash -= cost
            sim_positions[symbol] = {
                "qty": qty,
                "entry": price
            }
            print(f"[SIM BUY] {symbol} {qty} @ {price}")

    else:
        if symbol in sim_positions:
            entry = sim_positions[symbol]["entry"]
            pnl = (price - entry) * qty
            sim_cash += qty * price

            print(f"[SIM SELL] {symbol} @ {price} | PnL: {pnl}")
            del sim_positions[symbol]

def trend_ok(df):
    return (
        df["close"].iloc[-1] > df["ema20"].iloc[-1] and
        df["ema9"].iloc[-1] > df["ema20"].iloc[-1]
    )


async def on_trade(trade):
    symbol = trade.symbol
    price = trade.price
    volume = trade.size
    ts = trade.timestamp.replace(second=0, microsecond=0)

    # Initialize minute tracking
    if symbol not in current_minute:
        current_minute[symbol] = ts

    # 🔥 New minute → build candle
    if ts != current_minute[symbol]:
        prices = trade_buffer[symbol]

        if prices:  # avoid empty
            candle = {
                "open": prices[0][0],
                "high": max(p[0] for p in prices),
                "low": min(p[0] for p in prices),
                "close": prices[-1][0],
                "volume": sum(p[1] for p in prices)
            }

            await handle_bar(symbol, candle)

        # Reset for new minute
        trade_buffer[symbol] = []
        current_minute[symbol] = ts

    # Append trade
    trade_buffer[symbol].append((price, volume))







async def run_live():
    print("Starting LIVE trading...")

    stream = StockDataStream(ALPACA_KEY, ALPACA_SECRET)

   
    #stream.subscribe_bars(on_bar, *SYMBOLS)
    stream.subscribe_trades(on_trade, *SYMBOLS)

    try:
        await stream._run_forever()
    except Exception as e:
        print(f"Stream crashed: {e}")


REENTRY_BUFFER = 0.002   # 0.2% (tune this)

def is_near_last_loss(symbol, price):
    if symbol not in last_loss_price:
        return False
    
    last_price = last_loss_price[symbol]
    return abs(price - last_price) / last_price < REENTRY_BUFFER

COOLDOWN_SECONDS = 900  # 15 mins

def in_cooldown(symbol):
    if symbol not in last_loss_time:
        return False
    
    return (datetime.now() - last_loss_time[symbol]).total_seconds() < COOLDOWN_SECONDS

MIN_MOVE_AWAY = 0.005  # 0.5%

def moved_away(symbol, price):
    if symbol not in last_loss_price:
        return True
    
    last_price = last_loss_price[symbol]
    return abs(price - last_price) / last_price > MIN_MOVE_AWAY

async def handle_bar(bar):
    symbol = bar.symbol
    close = bar.close
    high = bar.high
    low = bar.low
    volume = bar.volume

    df = historical_data[symbol]

    if len(df) < 60:
        log.info(f"{symbol} warming up... ({len(df)})")
        return

    # Append new row (safe concat)
    new_row = pd.DataFrame([{
        "open": bar.open,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume
    }], index=[bar.timestamp])

    df = pd.concat([df, new_row])
    df = df[~df.index.duplicated(keep='last')].tail(200)

    df = add_indicators(df)
    historical_data[symbol] = df

    current_price = close

    # =========================
    # ===== EXIT LOGIC ========
    # =========================



    if symbol in positions:
        pos = positions[symbol]

        
        if low <= pos.stop_loss:
            if bar.open >= pos.stop_loss:
                exit_price = bar.open
            else:
                exit_price = pos.stop_loss
            reason = "STOP"

        elif high >= pos.target:
            if bar.open >= pos.target:
                exit_price = bar.open
            else:
                exit_price = pos.target
            reason = "TARGET"

        else:
            return

        await asyncio.to_thread(place_order, symbol, OrderSide.SELL, pos.qty, exit_price)

        pos.close(exit_price, reason)
        trade_history.append(pos)

        # ---- LOSS TRACKING ----
        pnl = (exit_price - pos.entry_price) * pos.qty

        if pnl < 0:
            last_loss_price[symbol] = pos.entry_price
            last_loss_time[symbol] = datetime.now()
            retry_count[symbol] = retry_count.get(symbol, 0) + 1
        else:
            retry_count[symbol] = 0

        del positions[symbol]

        print(f"EXIT {symbol} {reason} @ {exit_price}")
        return

    # =========================
    # ===== ENTRY LOGIC =======
    # =========================
    entry, extra = check_entry(df, symbol)

    # ---- ENTRY FILTERS ----
    
    if (
        is_near_last_loss(symbol, current_price)
        or in_cooldown(symbol)
        or not moved_away(symbol, current_price)
        or retry_count.get(symbol, 0) >= 2
    ):
        return
    
    # ---- REQUIRE PULLBACK BEFORE ENTRY ----
    recent_high = df["high"].rolling(10).max().iloc[-2]
    recent_low = df["low"].rolling(10).min().iloc[-2]

    # define pullback depth (tune this)
    PULLBACK_THRESHOLD = 0.003  # 0.3%

    pulled_back = (recent_high - current_price) / recent_high > PULLBACK_THRESHOLD

    if not pulled_back:
        return        

    # block same candle re-entry
    if last_trade_bar.get(symbol) == bar.timestamp:
        return

    if entry:
        qty = int(CAPITAL_PER_TRADE / current_price)
        if qty == 0:
            return

        await asyncio.to_thread(place_order, symbol, OrderSide.BUY, qty, current_price)

        positions[symbol] = Position(
            symbol,
            current_price,
            qty,
            extra["stop_loss"],
            extra["target"],
            extra["atr"]
        )

        last_entry_price[symbol] = current_price
        last_trade_bar[symbol] = bar.timestamp
        last_breakout_price[symbol] = current_price
        print(f"ENTRY {symbol} @ {current_price}")

        print("ENTRY symbol==", symbol)
        print("ENTRY price==",extra["entry_price"]) 
        print("ENTRY stop_loss==",extra["stop_loss"]) 
        print("ENTRY target==",extra["target"]) 



# ================= STREAM SETUP =================

async def start_stream():
    preload_data()

    if SIM_MODE:
        await run_simulation()
    else:
        await run_live()


def run():
    try:
        loop = asyncio.get_running_loop()
        # ✅ If already running → create task
        print("Detected running event loop")
        loop.create_task(start_stream())
    except RuntimeError:
        # ✅ No loop → safe to run normally
        print("Starting fresh event loop")
        asyncio.run(start_stream())
    generate_report(trade_history)    

if __name__ == "__main__":
    run()