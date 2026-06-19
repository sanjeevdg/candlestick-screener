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
# ================= CONFIG =================


#SYMBOLS = ["AAPL", "TSLA", "NVDA"]  # change as needed
TIMEFRAME = "15Min"
CAPITAL_PER_TRADE = 1000

logging.basicConfig(level=logging.INFO)

trades = []
data = {}
SIM_MODE = True   # 🔥 toggle this
REPLAY_SPEED = 0.01
LOOKBACK_DAYS = 15
TIMEFRAME = TimeFrame.Minute


parser = argparse.ArgumentParser()

parser.add_argument("--symbols")

args = parser.parse_args()

SYMBOLS = args.symbols.split(",")



# ================= CLIENTS =================
trading_client = TradingClient(ALPACA_KEY, ALPACA_SECRET, paper=True)
data_stream = StockDataStream(ALPACA_KEY, ALPACA_SECRET)

# ================= STORAGE =================
data = {symbol: pd.DataFrame() for symbol in SYMBOLS}
positions = {}
trade_history = []




historical_client = StockHistoricalDataClient(ALPACA_KEY, ALPACA_SECRET)

def preload_data():
    global data

    print("Loading historical data...")

    end = datetime.utcnow()
    start = end - timedelta(days=LOOKBACK_DAYS)

    request = StockBarsRequest(
        symbol_or_symbols=SYMBOLS,
        timeframe=TIMEFRAME,
        start=start,
        end=end,
        feed=DataFeed.IEX
    )

    bars = historical_client.get_stock_bars(request)

    valid_symbols = []

    for symbol in SYMBOLS:
        try:
            df = bars.df.loc[symbol].reset_index()

            if df.empty or len(df) < 50:
                print(f"⚠️ Skipping {symbol} (not enough data)")
                continue

            df = df[["timestamp", "open", "high", "low", "close", "volume"]]

            # Ensure sorted
            df = df.sort_values("timestamp").reset_index(drop=True)

            data[symbol] = df
            valid_symbols.append(symbol)

            print(f"{symbol}: {len(df)} rows loaded")

        except Exception as e:
            print(f"❌ Error loading {symbol}: {e}")

    print("Valid symbols:", valid_symbols)
    return valid_symbols


async def run_simulation():
    print("Starting SIMULATION mode...")
    for s in data:
        print(f"{s}: {len(data[s])} rows")
    

    # Get valid symbols only
    symbols = [s for s in SYMBOLS if s in data and len(data[s]) > 50]

    if not symbols:
        print("❌ No valid symbols to simulate")
        return

    # Track current index per symbol
    indices = {s: 50 for s in symbols}  # start after indicators ready

    active = True
    print("DATA TYPE:", type(data))
    print("DATA VALUE:", data)
    while active:
        active = False

        for symbol in symbols:
            df = data[symbol]
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
        #self.timestamp = row["datetime"]


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
    df["ema9"] = df["close"].ewm(span=9).mean()
    df["ema20"] = df["close"].ewm(span=20).mean()

    df["cum_vol"] = df["volume"].cumsum()
    df["cum_vol_price"] = (df["close"] * df["volume"]).cumsum()
    df["vwap"] = df["cum_vol_price"] / df["cum_vol"]

    df["avg_volume"] = df["volume"].rolling(20).mean()
    df["atr"] = (df["high"] - df["low"]).rolling(14).mean()

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
    #df["avg_volume"] = df["volume"].rolling(20).mean()
    return df["volume"].iloc[-1] > df["avg_volume"].iloc[-1] * 1.5

def near_vwap(df):
    price = df["close"].iloc[-1]
    vwap = df["vwap"].iloc[-1]
    return abs(price - vwap) / vwap < 0.002
'''
def check_entry(df):
    if len(df) < 25:
        return False, None

    last = df.iloc[-1]

    breakout = detect_breakout(df)
    strong = is_strong_candle(last)
    vol = volume_spike(df)
    vwap_pullback = near_vwap(df)


    df["tr"] = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            abs(df["high"] - df["close"].shift()),
            abs(df["low"] - df["close"].shift())
        )
    )

    df["atr"] = df["tr"].rolling(14).mean()
    

    atr = df["atr"].iloc[-1]


    entry_price = df["close"].iloc[-1]

    stop_loss = entry_price - atr * 1.2
    target = entry_price + 2 * (entry_price - stop_loss)

    if breakout and strong and vol and trend_ok(df):
        return True, {
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "target": target,
            "atr": atr   # ✅ ADD THIS
        }

    if vwap_pullback and strong and vol:
        return True, {
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "target": target,
            "atr": atr   # ✅ ADD THIS
        }

    return False, None 
'''

def check_entry(df):
    if len(df) < 30:
        return False, None
    
    last = df.iloc[-1]
        
    breakout = detect_breakout(df)
    strong = is_strong_candle(last)
    vol = volume_spike(df)# ===== TRUE ATR =====

    # Pullback must HOLD support
    vwap_pullback = near_vwap(df)

    prev_low = df["low"].iloc[-2]
    curr_low = df["low"].iloc[-1]

    higher_low = curr_low > prev_low

    bullish_reclaim = last["close"] > last["open"]

    if not (vwap_pullback and higher_low and bullish_reclaim):
        return False, None
    

    current_price = df["close"].iloc[-1]  
    df["tr"] = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            abs(df["high"] - df["close"].shift()),
            abs(df["low"] - df["close"].shift())
        )
    )
    df["atr"] = df["tr"].rolling(14).mean()

    atr = df["atr"].iloc[-1]

    # ❌ Avoid bad ATR
    if np.isnan(atr) or atr == 0:
        return False, None

    df["ema9"] = df["close"].ewm(span=9).mean()
    df["ema20"] = df["close"].ewm(span=20).mean()

    trend_strength = df["ema9"].iloc[-1] - df["ema20"].iloc[-1]

    if trend_strength <= 0:
        return False, None  # ❌ no uptrend    

    entry_price = df["close"].iloc[-1]

    # ===== NO-CHASE FILTER =====
    recent_move = (df["close"].iloc[-1] - df["close"].iloc[-10]) / df["close"].iloc[-10]

    if recent_move > 0.002:   # 0.2% over 10 candles
        return False, None

    # ===== BREAKOUT CONFIRMATION =====
    
    body = abs(last["close"] - last["open"])
    range_candle = last["high"] - last["low"]
    breakout_level = df["high"].rolling(20).max().iloc[-2]
    strong_body = body > 0.6 * range_candle

    valid_breakout = (
        last["close"] > breakout_level and
        strong_body and
        last["volume"] > df["volume"].rolling(20).mean().iloc[-1] * 2
    )    


    df["atr"] = df["tr"].rolling(14).mean()

    atr_now = df["atr"].iloc[-1]
    atr_prev = df["atr"].iloc[-5]

    # ❌ Reject if volatility is not expanding





    range_20 = df["high"].rolling(20).max().iloc[-1] - df["low"].rolling(20).min().iloc[-1]
    
    stop_loss = entry_price - atr * 1.0
    target    = entry_price + atr * 2.5
    
    price_range = df["high"].rolling(20).max().iloc[-1] - df["low"].rolling(20).min().iloc[-1]

    risk = entry_price - stop_loss
    reward = target - entry_price


    price = df["close"].iloc[-1]


    recent_high = df["high"].rolling(50).max().iloc[-2]


    prev = df.iloc[-2]
    last = df.iloc[-1]

    # Breakout happened on previous candle
    breakout_happened = prev["close"] > df["high"].rolling(20).max().iloc[-3]

    # Now confirm continuation
    follow_through = last["close"] > prev["high"]
    '''
    if not (breakout_happened and follow_through):
        return False, None
    '''    
    # Ensure breakout has room
    if (recent_high - entry_price) < atr_now * 1.2:
        return False, None
        
    '''
    # Reject cheap / slow movers
    if atr_now / price < 0.0015:   # 0.15%
        return False, None
    '''

        
    if atr_now <= atr_prev:
        return False, None
    

    if valid_breakout and is_fake_breakout(df):
        return False, None


    if reward / risk < 2:
        return False, None

    if price_range < atr * 4:
        return False, None    

    if is_fake_breakout(df):
        return False, None
    '''    
    if df["close"].iloc[-1] < df["open"].iloc[-1]:
        return False, None

    if range_20 < atr * 3:
        return False, None
    
    if current_price <= (entry_price - atr * 1.0):
        return False, None   # ===== ENTRY TYPES =====
    '''
    if current_price > entry_price + atr:
        stop_loss = entry_price   # breakeven# 🔹 BREAKOUT ENTRY (strong move)
    #and vol 
    if valid_breakout and strong and trend_ok(df):

        stop_loss = entry_price - atr * 1.0   # wider SL
        target = entry_price + 2.5 * atr
        
        return True, {
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "target": target,
            "atr": atr
        }
     
    # 🔹 PULLBACK ENTRY (safer)    and vol
    if vwap_pullback and strong and trend_strength > 0:

        stop_loss = entry_price - atr * 0.8   # tighter SL
        target = entry_price + 2.0 * atr
        
    
        return True, {
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "target": target,
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


    # ================= MAIN LOGIC =================
async def handle_bar(bar):
    global data

    symbol = bar.symbol

    new_row = {
        "open": bar.open,
        "high": bar.high,
        "low": bar.low,
        "close": bar.close,
        "volume": bar.volume
    }

    df = data.get(symbol)

    if df is None or df.empty:
        df = pd.DataFrame([new_row])
        data[symbol] = df
        return

    df = pd.concat([df, pd.DataFrame([new_row])]).tail(200)
    df = add_indicators(df)

    data[symbol] = df

    current_price = df["close"].iloc[-1]

    # ===== EXISTING POSITION =====
    if symbol in positions and positions[symbol].active:
        pos = positions[symbol]

        bar_low = bar.low
        bar_high = bar.high

        exit_price = None
        status = "HOLD"

        if bar.low <= pos.entry_price * 0.98:
            exit_price = pos.entry_price * 0.98
            status = "FORCE_STOP"
        # 🔴 STOP LOSS (check LOW)
        if bar_low <= pos.stop_loss:
            exit_price = pos.stop_loss
            status = "STOP_LOSS"

        # 🟢 TARGET (check HIGH)
        elif bar_high >= pos.target:
            exit_price = pos.target
            status = "TARGET"

        else:
            # 🔵 TRAILING STOP (only if still in trade)
            new_sl = bar.close - pos.atr * 1.0
            pos.stop_loss = max(pos.stop_loss, new_sl)

        # 🚀 EXECUTE EXIT
        if status in ["STOP_LOSS", "TARGET"]:
            await asyncio.to_thread(place_order, symbol, OrderSide.SELL, pos.qty, exit_price)
            pos.active = False
            pos.close(exit_price, status)
            trade_history.append(pos)
            del positions[symbol]
            print(f"EXIT {symbol} {status} at {exit_price}")

        return

    # ===== NEW ENTRY =====
    entry, extra = check_entry(df)

    

    if entry:
        qty = int(CAPITAL_PER_TRADE / extra["entry_price"])
        if qty == 0:
            return

        #await asyncio.to_thread(place_order, symbol, OrderSide.BUY, qty)
        await asyncio.to_thread(place_order, symbol, OrderSide.BUY, qty, current_price)
        positions[symbol] = Position(
            symbol,
            extra["entry_price"],
            qty,
            extra["stop_loss"],
            extra["target"],
            extra["atr"]
        )

        print("ENTRY symbol==", symbol)
        print("ENTRY price==",extra["entry_price"]) 

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