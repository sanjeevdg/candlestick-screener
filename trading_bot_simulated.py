import logging
import pandas as pd
import numpy as np
from alpaca.data.live import StockDataStream
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockSnapshotRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from config import ALPACA_KEY, ALPACA_SECRET
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta
from alpaca.data.enums import DataFeed
import json
import time
import talib as ta
import argparse
import asyncio
import yfinance as yf
from scanner.momentum import get_top_momentum
# -------------------------------------------------------
# Logging
# -------------------------------------------------------

WAITING = 0
BREAKOUT = 1
PULLBACK = 2
READY = 3
signal = None#signal = None | "BUY"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


class FakeBar:
    def __init__(self, symbol, row):
        self.symbol = symbol
        
        close = row["close"]
        volume = row["volume"]
        high = row["high"]
        low = row["low"]
        open = row["open"]
        

        # 🔥 FORCE SCALAR HERE
        if isinstance(close, pd.Series):
            close = close.iloc[0]

        if isinstance(volume, pd.Series):
            volume = volume.iloc[0]

        self.close = float(close)
        self.volume = float(volume)
        self.high = float(high)
        self.low = float(low)
        self.open = float(open)




def optimize():

    data = preload_data()

    momentum_vals = [0.1, 0.2]
    volume_vals = [1.1, 1.5]
    tp_vals = [0.01, 0.02]
    sl_vals = [0.003, 0.005]

    results = []

    for mom in momentum_vals:
        for vol in volume_vals:
            for tp in tp_vals:
                for sl in sl_vals:

                    print(f"\nTesting MOM={mom}, VOL={vol}, TP={tp}, SL={sl}")

                    bot = MomentumScannerBot(ALPACA_KEY, ALPACA_SECRET)

                    bot.MOMENTUM = mom
                    bot.VOL_MULT = vol
                    bot.TP = tp
                    bot.SL = sl

                    stats = bot.run(data)

                    results.append({
                        "momentum": mom,
                        "volume": vol,
                        "tp": tp,
                        "sl": sl,
                        "pnl": stats["pnl"]
                    })

    df = pd.DataFrame(results)

    print("\n===== BEST =====")
    print(df.sort_values("pnl", ascending=False).head())
    

data_client = StockHistoricalDataClient(ALPACA_KEY, ALPACA_SECRET)

log = logging.getLogger("BOT")

parser = argparse.ArgumentParser()

parser.add_argument("--symbols")

args = parser.parse_args()

SYMBOLS = args.symbols.split(",")




async def replay_data(bot, data, delay=0):

    # assume all symbols have same length
        max_len = max(len(df) for df in data.values())

        for i in range(max_len):

            bot.active_symbols = bot.scan_historical(data, i)

            for symbol, df in data.items():

                if i >= len(df):
                    continue
                
                if isinstance(df, list):
                    df = pd.DataFrame(df)    

                row = df.iloc[i]

                bar = FakeBar(symbol, row)

                await bot.on_bar(bar)

            await asyncio.sleep(delay)   








# =======================================================
# MOMENTUM BOT
# =======================================================

def preload_data():

    print("Loading historical bars...")
    print('SYMBOLS==',SYMBOLS)
    end = datetime.utcnow()
    #start = end - timedelta(minutes=3000)
    start = end - timedelta(days=8)

    request = StockBarsRequest(
        symbol_or_symbols=SYMBOLS,
        timeframe=TimeFrame.Minute,
        start=start,
        end=end,
        feed=DataFeed.IEX
    )

    bars = data_client.get_stock_bars(request)

    data = {}

    df_all = bars.df.reset_index()  # 🔥 CRITICAL FIX

    #print(df_all.head())
    print(df_all.columns)

    for symbol in SYMBOLS:

        df = df_all[df_all["symbol"] == symbol].copy()

        if df.empty:
            print(f"⚠️ No data for {symbol}")
            continue

        df = df.sort_values("timestamp").reset_index(drop=True)

        #data[symbol] = df


        for symbol in SYMBOLS:
            df = bars.df.loc[symbol]

            data[symbol] = df[["open","close","high","low","volume"]].to_dict("records")

        return data




        print(f"{symbol} loaded {len(df)} bars")

    return data   


def check_breakout_state(symbol, df, state, state_data):
        


        if len(df) < 50:
            return state, state_data, None
        

        price = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        close = price.iloc[-1]
        prev_close = price.iloc[-2]

        # === LEVELS ===
        recent_high = high.rolling(20).max().iloc[-2]

        # === ATR ===
        atr_series = (high - low).rolling(14).mean()
        atr = atr_series.iloc[-1]

        # === VOLUME ===
        vol_ma = volume.rolling(20).mean().iloc[-1]
        rvol = volume.iloc[-1] / vol_ma if vol_ma > 0 else 1

        # === TREND FILTER ===
        ema50 = price.ewm(span=50).mean().iloc[-1]
        trend_ok = close > ema50

        # === COMPRESSION ===
        recent_range = (high - low).iloc[-5:-1].mean()
        compression = recent_range < atr * 0.8

        range_now = high.iloc[-1] - low.iloc[-1]
        range_prev = (high - low).iloc[-5:-1].mean()

        expansion = range_now > range_prev * 1.05
        #print('expansion==',expansion)
        atr_pct = atr / close

        
        # =========================
        # STATE MACHINE
        # =========================

        # 1. WAITING → BREAKOUT
        if state == WAITING:
            breakout = close > recent_high
            log.info(f"{symbol} state={state} close={close} rvol={rvol:.2f}")
            if breakout and compression and 1.2 < rvol < 3.5 and trend_ok and expansion:
                return BREAKOUT, {
                    "breakout_level": recent_high,
                    "bars_since": 0
                }, None

        # 2. BREAKOUT → PULLBACK
        elif state == BREAKOUT:

            if "ready_bars" not in state_data:
                state_data["ready_bars"] = 0
            
            state_data["bars_since"] += 1
            
            if state_data is None:
                return WAITING, None, None
            
            level = state_data["breakout_level"]
            # price pulls back but holds level
            distance_from_level = (close - level) / level
            pullback = low.iloc[-1] <= level * 1.0 and close >= level * 0.995


            if state_data is None:
                return WAITING, None, None

            if distance_from_level > 0.01:
                return WAITING, None, None    

            if pullback:
                return PULLBACK, state_data, None

            # breakout failed
            if close < level:
                return WAITING, None, None

            # timeout (no pullback = too extended)
            if state_data["bars_since"] > 5:
                return WAITING, None, None

            if atr_pct < 0.003:   # less than 0.3%
                return WAITING, None, None
    

        elif state == READY:

            level = state_data["breakout_level"]
            state_data["ready_bars"] += 1

            hold = close > level
            continuation = close > prev_close
            candle_range = high.iloc[-1] - low.iloc[-1]

            strong_close = (close - low.iloc[-1]) / (candle_range + 1e-6) > 0.5
            # wait at least 1 candle
            if state_data["ready_bars"] >= 2 and hold and continuation and strong_close:
                return WAITING, None, "BUY"   # trigger entry

            # fail
            if close < level:
                return WAITING, None, None

            # timeout
            if state_data["ready_bars"] > 4:
                return WAITING, None, None        
        # 3. PULLBACK → READY (RECLAIM)
        elif state == PULLBACK:
            level = state_data["breakout_level"]

            candle_range = high.iloc[-1] - low.iloc[-1]

            strong_reclaim = (
                close > level and
                close > prev_close and
                (close - low.iloc[-1]) / (candle_range + 1e-6) > 0.5   # closes strong
            )

            distance_from_level = (close - level) / level

            if state_data is None:
                return WAITING, None, None            

            if strong_reclaim:
                state_data["confirm_bars"] = 0
                return READY, state_data, None

            if distance_from_level > 0.01:
                return WAITING, None, None

            # failed structure
            if close < level:
                return WAITING, None, None

        return state, state_data, signal





class MomentumScannerBot:

    def __init__(self, key, secret):

        self.key = key
        self.secret = secret

        self.trading_client = TradingClient(key, secret, paper=True)
        self.data_client = StockHistoricalDataClient(key, secret)

        self.stream = StockDataStream(key, secret)

        self.symbols = SYMBOLS
        self.bars = {}
        self.trades = []
        self.positions = {}

        self.TP = 0.02   # 1.5%
        self.SL = 0.003   # 0.5%
        self.VOL_MULT = 1.1
        self.last_trade_time = {}
        self.COOLDOWN = 300
        self.active_symbols = set()
        self.last_scan_time = None
        self.scan_interval = 300  # 5 minutes
        self.max_symbols = 10
        self.universe = [
        "KOD","NAVN","CTMX","WTI","PR","CRM","BZ","DKNG","CELH","NFLX","CRGY","AAPL","PYPL","QCOM","SMCI"
        ]
# track open trades
        self.open_positions = set()
        self.MODE = "SIM"   # or "SIM" OR "LIVE"    
        self.symbol_state = {}
        self.symbol_state_data = {}

        self.last_trade_time = {}
        

        self.COOLDOWN = 300   # seconds (tune this)
        log.info("Bot initialized")



    
    # ---------------------------------------------------
    # Scan market for top movers
    # ---------------------------------------------------

    def scan_top_movers(self):

        log.info("Scanning top movers...")

        universe = [
        "KOD","NAVN","CTMX","WTI","PR","CRM","BZ","DKNG","CELH","NFLX","CRGY","AAPL","PYPL","QCOM","SMCI"
        ]

        request = StockSnapshotRequest(symbol_or_symbols=universe)

        snapshots = self.data_client.get_stock_snapshot(request)

        movers = []

        for symbol, snap in snapshots.items():

            if snap.daily_bar and snap.previous_daily_bar:

                price = snap.daily_bar.close
                prev = snap.previous_daily_bar.close

                change = (price - prev) / prev * 100

                movers.append((symbol, change))

        movers.sort(key=lambda x: x[1], reverse=True)

        top = [m[0] for m in movers[:5]]

        log.info(f"Top movers: {top}")

        return top



    def scan_historical(self, data, i):

        results = []

        for symbol, df in data.items():

            if i < 20 or i >= len(df):
                continue

            

            if isinstance(df, list):
                df = pd.DataFrame(df)    
            
            window = df.iloc[:i]
                
            price_now = window["close"].iloc[-1]
            price_prev = window["close"].iloc[-5]

            momentum = (price_now - price_prev) / price_prev

            avg_vol = window["volume"].tail(20).mean()
            curr_vol = window["volume"].iloc[-1]

            rvol = curr_vol / avg_vol if avg_vol > 0 else 0

            results.append({
                "symbol": symbol,
                "momentum": momentum,
                "rvol": rvol
            })

        # rank by momentum * volume
        results.sort(key=lambda x: x["momentum"] * x["rvol"], reverse=True)

        top = [x["symbol"] for x in results[:self.max_symbols]]

        return set(top)



    def refresh_symbols(self):
        now = time.time()

        if self.MODE == "SIM":
            return

        if self.last_scan_time and now - self.last_scan_time < self.scan_interval:
            return

        print("🔄 Running momentum scan...")

        #results = get_top_momentum(self.data_client, self.universe, limit=20)

        #new_symbols = set([x["symbol"] for x in results[:self.max_symbols]])

        #print("New symbols:", new_symbols)

        # --- KEEP symbols if already in trade ---
        #self.active_symbols = new_symbols | self.open_positions

        #self.last_scan_time = now



    # ---------------------------------------------------
    # Place order
    # ---------------------------------------------------

    def place_order(self, symbol, side, qty=10):

        #log.info(f"{side.upper()} {symbol} x{qty}")


        if self.MODE == "SIM":
            return


        order = self.trading_client.submit_order(
            MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
                #extended_hours=True
            )
        )

        log.info(f"Order sent {order.id}")

    # ---------------------------------------------------
    # Exit logic
    # ---------------------------------------------------

    def check_exit(self, symbol, price):

        pos = self.positions.get(symbol)

        if not pos:
            return

        entry = pos["entry"]

        change = (price - entry) / entry

        if change >= self.TP:
            #log.info(f"TAKE PROFIT {symbol}")
            self.place_order(symbol, "sell")
            self.trades.append({
                "symbol": symbol,
                "entry": entry,
                "exit": price,
                "pnl": change,
                "type": "TP"
            })
            del self.positions[symbol]

        elif change <= -self.SL:
            #log.info(f"STOP LOSS {symbol}")
            self.place_order(symbol, "sell")
            self.trades.append({
                "symbol": symbol,
                "entry": entry,
                "exit": price,
                "pnl": change,
                "type": "SL"
            })
            self.open_positions.discard(symbol)
            del self.positions[symbol]


    
    async def on_bar(self, bar):
        log.info(f"BAR: {bar.symbol} {bar.close}")
        s = bar.symbol

        if s not in self.symbol_state:
            self.symbol_state[s] = WAITING
            self.symbol_state_data[s] = None

        if s not in self.bars:
            self.bars[s] = []    
        
        
        if len(self.bars[s]) < 50:
            return    

        #price = bar.close
        volume = bar.volume
        high = bar.high
        low = bar.low
        price = bar.close
        open = bar.open
        close = bar.close

        # store minimal data (no OHLC available)
        
        self.bars[s].append({
            "price": price,
            "volume": volume,
            "high": high,
            "low": low,
            "close":price,
            "open":open
        })
        
        #df = pd.DataFrame(self.bars[s]).tail(60)
        df = pd.DataFrame(self.bars[s])
      
        state = self.symbol_state[s]
        state_data = self.symbol_state_data[s]

        new_state, new_data, signal = check_breakout_state(s, df, state, state_data)

        #print('new-state===',new_state)
        #print('new-state-data====',new_data)# store updated state
        self.symbol_state[s] = new_state
        self.symbol_state_data[s] = new_data

       
        # need enough data for indicators
        if len(df) < 50:
            return

        # -----------------------------------
        # Exit logic
        # -----------------------------------
        if s in self.positions:
            pos = self.positions[s]
            
            level = pos["breakout_level"]

            # 🚨 BREAKOUT FAILURE EXIT
            self.check_exit(s, price)
            if close < level:
                return
            
            return

        # cooldown
        if s in self.last_trade_time:
            if (datetime.utcnow() - self.last_trade_time[s]).seconds < self.COOLDOWN:
                return

                #if entry:
        if signal == "BUY" and s not in self.positions:
        #if breakout and volume_trend and momentum_acceleration:
            log.info(f"BREAKOUT {s}")
            #if s in self.open_positions:
            #    return
            self.place_order(s, "buy")   
            print("ORDER-PLACED!!! BUY",s)
            self.positions[s] = {
                "entry": price,
                "entry_time": datetime.utcnow(),
                "breakout_level": price * 0.998
            }

            self.last_trade_time[s] = datetime.utcnow()
            self.open_positions.add(s)
            self.symbol_state[s] = WAITING
            self.symbol_state_data[s] = None
            self.signal = None





    # ---------------------------------------------------
    # Start bot
    # ---------------------------------------------------


    def run(self):

        data = preload_data()
        '''
        self.active_symbols = {
            s for s in self.active_symbols
            if s in new_symbols or s in self.open_positions
        }


        for s in self.symbols:

            self.bars[s] = []

            self.stream.subscribe_bars(self.on_bar, s)

        log.info(f"Subscribed to {self.symbols}")

        self.stream.run()

        '''
        #self.active_symbols = set(data.keys())
        #self.symbols = list(data.keys())

        for s in self.symbols:
            self.bars[s] = data.get(s, [])
            #self.stream.subscribe_bars(self.on_bar, s)

        #log.info("🚀 Running in SIMULATION mode")
        log.info(f"Subscribed to {self.symbols}")

        #self.stream.run()
        
        asyncio.run(replay_data(self, data, delay=0.0))

        log.info("Simulation complete")

        if not self.trades:
            return {
                "trades": 0,
                "win_rate": 0,
                "pnl": 0
            }
        if self.trades:

            df = pd.DataFrame(self.trades)

            total = len(df)
            wins = (df["pnl"] > 0).sum()
            pnl = df["pnl"].sum()

            print("\n===== RESULTS =====")
            print(f"Trades: {total}")
            print(f"Win rate: {wins/total:.2%}")
            print(f"Total PnL: {pnl:.4f}")

            print("\nSample trades:")
            print(df.head())

        

            return {
                "trades": total,
                "win_rate": wins / total,
                "pnl": pnl
            }    

# =======================================================
# MAIN
# =======================================================

def main():

    #data = preload_data()
    bot = MomentumScannerBot(
        key="PKC7D4XB4OTV2VDEFUF5BRL33P",
        secret="DZxMp2TBaqQWZH3seGm7ZSWbHjw25xLBD7ZGrG4F4GaF",
    )
    #optimize() 
    bot.MOMENTUM = 1.2
    bot.VOL_MULT = 1.1
    bot.TP = 0.02
    bot.SL = 0.003
    bot.run()


if __name__ == "__main__":
    main()