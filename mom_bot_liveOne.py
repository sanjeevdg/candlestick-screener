import logging
import pandas as pd
from datetime import datetime
from alpaca.data.live import StockDataStream
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from alpaca.data.enums import DataFeed

from datetime import datetime, timedelta

from config import ALPACA_KEY, ALPACA_SECRET
import argparse
# =========================
# CONFIG
# =========================
#API_KEY = "YOUR_KEY"
#API_SECRET = "YOUR_SECRET"

#SYMBOLS = ["AAPL", "TSLA", "NVDA", "AMD", "META"]

TP = 0.02
SL = 0.005
COOLDOWN = 300  # seconds

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger()


parser = argparse.ArgumentParser()

parser.add_argument("--symbols")

args = parser.parse_args()

SYMBOLS = args.symbols.split(",")


def preload_recent_data(symbols):

    

    client = StockHistoricalDataClient(ALPACA_KEY, ALPACA_SECRET)

    end = datetime.utcnow()
    start = end - timedelta(minutes=300)

    request = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Minute,
        start=start,
        end=end,
        feed=DataFeed.IEX

    )

    bars = client.get_stock_bars(request).df.reset_index()

    data = {}

    for s in symbols:
        df = bars[bars["symbol"] == s].sort_values("timestamp")
        data[s] = df.to_dict("records")

    return data

# =========================
# BOT
# =========================
class LiveMomentumBot:

    def __init__(self):
        self.stream = StockDataStream(ALPACA_KEY, ALPACA_SECRET)
        self.trading = TradingClient(ALPACA_KEY, ALPACA_SECRET, paper=True)

        self.bars = preload_recent_data(SYMBOLS)
        self.positions = {}
        self.last_trade_time = {}

    # -------------------------
    # ENTRY LOGIC (SIMPLIFIED)
    # -------------------------
    def check_entry(self, df):
        if len(df) < 20:
            return False, None

        high = df["high"]
        close = df["close"]
        volume = df["volume"]

        recent_high = high.rolling(20).max().iloc[-2]

        vol_ma = volume.rolling(20).mean().iloc[-1]
        rvol = volume.iloc[-1] / vol_ma if vol_ma > 0 else 1

        breakout = close.iloc[-1] > recent_high

        if breakout and rvol > 1.2:
            return True, recent_high

        return False, None

    # -------------------------
    # EXIT LOGIC
    # -------------------------
    def check_exit(self, symbol, price):
        pos = self.positions[symbol]
        entry = pos["entry"]

        change = (price - entry) / entry

        if change >= TP or change <= -SL:
            log.info(f"EXIT {symbol} @ {price:.2f} PnL={change:.3f}")

            self.trading.submit_order(
                MarketOrderRequest(
                    symbol=symbol,
                    qty=1,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY,
                )
            )

            del self.positions[symbol]

    # -------------------------
    # ON BAR
    # -------------------------
    async def on_bar(self, bar):

        s = bar.symbol
        price = bar.close

        log.info(f"{s} BAR {price}")

        self.bars[s].append({
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume
        })

        df = pd.DataFrame(self.bars[s]).tail(50)

        # ---- EXIT FIRST ----
        if s in self.positions:
            self.check_exit(s, price)
            return

        # ---- COOLDOWN ----
        if s in self.last_trade_time:
            if (datetime.utcnow() - self.last_trade_time[s]).seconds < COOLDOWN:
                return

        # ---- ENTRY ----
        entry, level = self.check_entry(df)

        if entry:
            log.info(f"🚀 BUY {s} @ {price}")

            self.trading.submit_order(
                MarketOrderRequest(
                    symbol=s,
                    qty=1,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                )
            )

            self.positions[s] = {"entry": price}
            self.last_trade_time[s] = datetime.utcnow()

    # -------------------------
    # START
    # -------------------------
    def run(self):

        for s in SYMBOLS:
            self.stream.subscribe_bars(self.on_bar, s)

        log.info(f"Subscribed to {SYMBOLS}")

        self.stream.run()


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    bot = LiveMomentumBot()
    bot.run()