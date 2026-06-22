import logging 
import asyncio
import pandas as pd
import numpy as np
import logging
from datetime import datetime

from alpaca.data.live import StockDataStream
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass

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
TIMEFRAME = "1Min"
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

latest_price = {}

SIM_MODE = False   # 🔥 toggle this
REPLAY_SPEED = 0.01
LOOKBACK_DAYS = 4
TIMEFRAME = TimeFrame.Minute


parser = argparse.ArgumentParser()

parser.add_argument("--symbols")

args = parser.parse_args()

SYMBOLS = args.symbols.split(",")


  # original
#live_data = {symbol: pd.DataFrame() for symbol in SYMBOLS}


# ================= CLIENTS =================
trading_client = TradingClient(ALPACA_KEY, ALPACA_SECRET, paper=True)
data_stream = StockDataStream(ALPACA_KEY, ALPACA_SECRET)

# ================= STORAGE =================
data = {symbol: pd.DataFrame() for symbol in SYMBOLS}

historical_data = data.copy() 

print("SYMBOLS:", SYMBOLS)
print("historical_data keys:", list(historical_data.keys()))
live_data = {
    symbol: historical_data[symbol].copy()
    for symbol in SYMBOLS
}
print("live_data keys:", list(live_data.keys()))

positions = {}
trade_history = []




historical_client = StockHistoricalDataClient(ALPACA_KEY, ALPACA_SECRET)

def preload_data():
    global data
    global last_entry_price

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

            df = df[["timestamp", "open", "high", "low", "close", "volume"]]
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            # Ensure sorted
            df = df.sort_values("timestamp").reset_index(drop=True)

            if df.empty or len(df) < 100:
                print(f"⚠️ Skipping {symbol} (not enough data)")
                continue

            data[symbol] = df
            #del last_entry_price[symbol]
            #last_entry_price[symbol] = None
            valid_symbols.append(symbol)

            print(f"{symbol}: {len(df)} rows loaded")

        except Exception as e:
            print(f"❌ Error loading {symbol}: {e}")

    print("Valid symbols:", valid_symbols)
    return valid_symbols


async def run_simulation():
    print("Starting SIMULATION mode...")
    for s in live_data:
        print(f"{s}: {len(live_data[s])} rows")
    

    # Get valid symbols only
    symbols = [symbol for symbol in SYMBOLS if symbol in live_data and len(live_data[symbol]) > 50]

    if not symbols:
        print("❌ No valid symbols to simulate")
        return

    # Track current index per symbol
    indices = {symbol: 50 for symbol in symbols}  # start after indicators ready

    active = True
    print("DATA TYPE:", type(live_data))
    print("DATA VALUE:", live_data)
    while active:
        active = False

        for symbol in symbols:
            df = live_data[symbol]
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

def round_price(price):
    if price >= 1:
        return round(price, 2)
    else:
        return round(price, 4)

def round_price_two_decs(price):
    return round(price + 1e-9, 2)


def check_entry(df,symbol):
    if len(df) < 30:
        return False, None

    row = df.iloc[-1]
    prev = df.iloc[-2]


    price = row["close"]
    vwap = row["vwap"]
    atr = row["atr"]

    



    if pd.isna(vwap) or pd.isna(atr):
        return False, None
    
    last_price = last_entry_price.get(symbol) 
    

    trend_up = price > vwap
    pullback = price <= vwap + 0.5 * atr

    bullish = row["close"] > row["open"]
    breakout = row["close"] > prev["high"]


    print('atr===',atr)
    print('vwap===',vwap)
    print('price===',price)
    print('trend_up=',trend_up)
    print('pullback=',pullback)
    print('bullish=',bullish)
    print('breakout=',breakout)
    #trend_up and 
    #pullback
    #bullish
    if trend_up and bullish and breakout:
    
        #if (last_price is None) or (price <= last_price - atr):
        return True, {
            "entry_price": price,
            "stop_loss": price * 0.998,
            "target": price + 1.2 * atr,
            "atr": atr
        }

    return False, None


async def run_live():
    print("Starting LIVE trading...")

    stream = StockDataStream(ALPACA_KEY, ALPACA_SECRET)
    '''
    async def on_bar(bar):
        symbol = bar.symbol

        # ===== UPDATE DATAFRAME =====
        df = pd.concat([historical_data[symbol], live_data[symbol]])

        new_row = {
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume
        }

        new_df = pd.DataFrame([new_row])

        df = pd.concat([df, new_df], ignore_index=True)

        # keep last N rows
        df = df.iloc[-200:]

        #data[symbol] = df
        df = add_indicators(df)   # keep if needed
        live_data[symbol] = df

        current_price = df["close"].iloc[-1]

        # ===== EXISTING POSITION =====
        if symbol in positions and positions[symbol].active:
            pos = positions[symbol]

            bar_low = candle["low"]
            bar_high = candle["high"]

            exit_price = None
            status = "HOLD"

            # 🔴 FORCE STOP
            if bar_low <= pos.entry_price * 0.98:
                exit_price = pos.entry_price * 0.98
                status = "FORCE_STOP"

            # 🔴 STOP LOSS
            elif bar_low <= pos.stop_loss:
                exit_price = pos.stop_loss
                status = "STOP_LOSS"

            # 🟢 TARGET
            elif bar_high >= pos.target:
                exit_price = pos.target
                status = "TARGET"

            else:
                # 🔵 TRAILING STOP
                new_sl = candle["close"] - pos.atr * 1.0
                pos.stop_loss = max(pos.stop_loss, new_sl)

            # 🚀 EXECUTE EXIT
            if status in ["STOP_LOSS", "TARGET", "FORCE_STOP"]:
                await asyncio.to_thread(
                    place_order, symbol, OrderSide.SELL, pos.qty, exit_price
                )

                pos.active = False
                pos.close(exit_price, status)
                trade_history.append(pos)

                del positions[symbol]

                print(f"EXIT {symbol} {status} at {exit_price}")

            return

        # ===== NEW ENTRY =====  , symbol, last_entry_price
        entry, extra = check_entry(df,symbol)

        if entry:
            qty = int(CAPITAL_PER_TRADE / extra["entry_price"])
            if qty == 0:
                return

            await asyncio.to_thread(
                place_order, symbol, OrderSide.BUY, qty, current_price
            )

            positions[symbol] = Position(
                symbol,
                extra["entry_price"],
                qty,
                extra["stop_loss"],
                extra["target"],
                extra["atr"]
            )

            last_entry_price[symbol] = extra["entry_price"]
    
            print(f"ENTRY {symbol} @ {extra['entry_price']} ({extra.get('type','')})")
    # ===== SUBSCRIBE =====
    #stream.subscribe_bars(on_bar, *SYMBOLS)
    '''
    stream.subscribe_trades(on_trade, *SYMBOLS)

    try:
        await stream._run_forever()
    except Exception as e:
        print(f"Stream crashed: {e}")

# ================= ORDER FUNCTIONS =================
sim_cash = 10000
sim_positions = {}

'''
MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                    #extended_hours=True
                )
MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY,
                    #extended_hours=True
                )                
'''



def place_order(symbol, side, qty, price,stop_loss,target):
    global sim_cash


    print('my-stop-loss==',stop_loss)
    print('my-target==',target)
    print('my-price==',price)
    new_sl = price * 0.995
    new_tp = price * 1.02
    print('new_sl==',new_sl)
    print('new_tp==',new_tp)


    if side == OrderSide.BUY:
        cost = qty * price
        if sim_cash >= cost:
            sim_cash -= cost
            sim_positions[symbol] = {
                "qty": qty,
                "entry": price
            }
            order_data = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
                order_class=OrderClass.BRACKET,
                take_profit={"limit_price": round_price_two_decs(new_tp)},
                stop_loss={"stop_price": round_price_two_decs(new_sl)},
            )
            order = trading_client.submit_order(order_data)

            log.info(f"Order sent {order.id}")
            print(f"[SIM BUY] {symbol} {qty} @ {price}")

    else:
        if symbol in sim_positions:
            entry = sim_positions[symbol]["entry"]
            pnl = (price - entry) * qty
            sim_cash += qty * price
            order_data = MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY,
                    #extended_hours=True
                )              

            order = trading_client.submit_order(order_data)
            log.info(f"Order sent {order.id}")
            print(f"[SIM SELL] {symbol} @ {price} | PnL: {pnl}")
            
            if pnl < 0:
                last_loss_price[symbol] = entry
                        
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
    ts = pd.Timestamp(trade.timestamp).floor("1min")
    latest_price[symbol] = price
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
                "volume": sum(p[1] for p in prices),
                "timestamp": current_minute[symbol]
            }

            await handle_bar(symbol, candle)

        # Reset for new minute
        trade_buffer[symbol] = []
        current_minute[symbol] = ts
    
    if symbol not in trade_buffer:
        trade_buffer[symbol] = [] 
        current_minute[symbol] = ts   
    # Append trade
    trade_buffer[symbol].append((price, volume,trade.timestamp))


'''
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
    entry, extra = check_entry(df,symbol)

    

    if entry:
        qty = int(CAPITAL_PER_TRADE / current_price)
        if qty == 0:
            return

        #await asyncio.to_thread(place_order, symbol, OrderSide.BUY, qty)
        await asyncio.to_thread(place_order, symbol, OrderSide.BUY, qty, current_price)
        positions[symbol] = Position(
            symbol,
            current_price,
            qty,
            extra["stop_loss"],
            extra["target"],
            extra["atr"]
        )

        print("ENTRY symbol==", symbol)
        print("ENTRY price==",extra["entry_price"]) 
'''




    # ================= MAIN LOGIC =================


def enforce_min_distance(entry, target, stop):
    # ensure TP is at least +0.01
    if target <= entry:
        target = entry + 0.01
    elif target - entry < 0.01:
        target = entry + 0.01

    # ensure SL is at least -0.01
    if stop >= entry:
        stop = entry - 0.01
    elif entry - stop < 0.01:
        stop = entry - 0.01

    return target, stop

def clean_price(price):
    return float(f"{price:.2f}") if price >= 1 else float(f"{price:.4f}")


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

async def handle_bar(symbol, candle):
    global live_data, positions, last_entry_price

    log.info(f"BAR: {symbol} {candle['close']}")

    new_row = pd.DataFrame([{
        "open": candle["open"],
        "high": candle["high"],
        "low": candle["low"],
        "close": candle["close"],
        "volume": candle["volume"]
     }], index=[pd.Timestamp(candle["timestamp"])])
    
    new_row.index.name = "timestamp"


    df = data.get(symbol)

    # ===== SAFETY CHECK =====
    if df is None:
        log.warning(f"{symbol} missing from live_data")
        return

    #log.error(f"{symbol} | columns={df.columns.tolist()} | len={len(df)}")    

    # ===== APPEND =====
    #df.loc[len(df)] = new_row

    # Keep rolling window
    

    # ===== INDICATORS =====
    df = pd.concat([df, new_row], ignore_index=True)
    df = df[~df.index.duplicated(keep='last')].tail(200)
    



    if len(df) > 200:
        df = df.iloc[-200:]

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.set_index("timestamp")

    df = df.sort_index()    

    df = df[~df.index.isna()]

    df = add_indicators(df)

    # ===== SAVE BACK =====
    live_data[symbol] = df

    #log.info(f"df-tail===\n{df.tail(3)}")


    current_price = df["close"].iloc[-1]

    # ===== EXISTING POSITION =====
    if symbol in positions and positions[symbol].active:
        pos = positions[symbol]

        bar_low = candle["low"]
        bar_high = candle["high"]

        exit_price = None
        status = "HOLD"


        # 🔴 FORCE STOP
        if bar_low <= pos.entry_price * 0.98:
            exit_price = pos.entry_price * 0.98
            status = "FORCE_STOP"

        # 🔴 STOP LOSS
        elif bar_low <= pos.stop_loss:
            if candle['open']  >= pos.stop_loss:
                exit_price = candle['open']
            else:
                exit_price = pos.stop_loss
            status = "STOP_LOSS"

        # 🟢 TARGET
        elif bar_high >= pos.target:
            if candle['open'] >= pos.target:
                exit_price = candle['open']
            else:
                exit_price = pos.target
            status = "TARGET"

        else:
            # 🔵 TRAILING STOP
            new_sl = candle["close"] - pos.atr * 1.0
            pos.stop_loss = max(pos.stop_loss, new_sl)

        # 🚀 EXECUTE EXIT

        '''    
        if status in ["STOP_LOSS", "TARGET", "FORCE_STOP"]:
            await asyncio.to_thread(
                place_order, symbol, OrderSide.SELL, pos.qty, exit_price,0.0,0.0
            )

            pos.active = False
            pos.close(exit_price, status)
            trade_history.append(pos)

            del positions[symbol]

            print(f"EXIT {symbol} {status} at {exit_price}")
        '''
        return
        

    # ===== NEW ENTRY =====  , symbol, last_entry_price
    entry, extra = check_entry(df,symbol)

    if not extra:
        return
    

    # ---- ENTRY FILTERS ----
    '''    
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
    '''
    #print('CURRENTPRICE-ACT==',extra["entry_price"])
    print('STOPLOSS==',extra["stop_loss"])
    print('TARGET==',extra["target"])
    print('CURRENTPRICE==',current_price) 
    print('CURRENTPRICE222==',extra["entry_price"]) 

    if entry:
        qty = int(CAPITAL_PER_TRADE / extra["entry_price"])
        if qty == 0:
            return

        await asyncio.to_thread(
            place_order, symbol, OrderSide.BUY, qty, current_price,round_price_two_decs(extra["stop_loss"]),round_price_two_decs(extra["target"])
        )

        positions[symbol] = Position(
            symbol,
            extra["entry_price"],
            qty,
            round_price_two_decs(extra["stop_loss"]),
            round_price_two_decs(extra["target"]),
            extra["atr"]
        )

        last_entry_price[symbol] = current_price
        last_trade_bar[symbol] = candle["timestamp"]
        last_breakout_price[symbol] = current_price

        print(f"ENTRY {symbol} @ {extra['entry_price']} ({extra.get('type','')}) - STOPLOSS== {extra['stop_loss']} - target={extra['target']}")

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