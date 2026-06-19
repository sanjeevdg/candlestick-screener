import logging
import pandas as pd

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
import talib
import argparse
# -------------------------------------------------------
# Logging
# -------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

data_client = StockHistoricalDataClient(ALPACA_KEY, ALPACA_SECRET)

log = logging.getLogger("BOT")

parser = argparse.ArgumentParser()

parser.add_argument("--symbols")

args = parser.parse_args()

SYMBOLS = args.symbols.split(",")

# =======================================================
# MOMENTUM BOT
# =======================================================
def preload_data():

        print("Loading historical bars...")

        end = datetime.utcnow()
        start = end - timedelta(minutes=3000)

        request = StockBarsRequest(
            symbol_or_symbols=SYMBOLS,
            timeframe=TimeFrame.Minute,
            start=start,
            end=end,
            feed=DataFeed.IEX
        )

        bars = data_client.get_stock_bars(request)

        for symbol in SYMBOLS:

            df = bars.df.loc[symbol]

            df = df.reset_index()

            df.rename(columns={
                "timestamp": "time"
            }, inplace=True)

            #data[symbol] = df

            print(f"{symbol} loaded {len(df)} bars")   

class MomentumScannerBot:

    def __init__(self, key, secret):

        self.key = key
        self.secret = secret

        self.trading_client = TradingClient(key, secret, paper=True)
        self.data_client = StockHistoricalDataClient(key, secret)

        self.stream = StockDataStream(key, secret)

        self.symbols = []
        self.bars = {}

        self.positions = {}

        log.info("Bot initialized")

    # ---------------------------------------------------
    # Scan market for top movers
    # ---------------------------------------------------

    def scan_top_movers(self):

        log.info("Scanning top movers...")

        universe = [
        "REED","PTN","CCCC","AIRG","NSRX","SER","DLTH","LIDR","CHNR","LNZA"
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

    # ---------------------------------------------------
    # Place order
    # ---------------------------------------------------

    def place_order(self, symbol, side, qty=10,price=None):

        log.info(f"place_order called with price={price}, qty={qty}")
        #if price is None:
        #    log.warning(f"Price is None for {symbol}, skipping order")
        #    return

        #if price > 100:
        #    qty = 2

        log.info(f"{side.upper()} {symbol} x{qty}")

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

        if change >= 0.02:
            log.info(f"TAKE PROFIT {symbol}")
            self.place_order(symbol, "sell")
            del self.positions[symbol]

        elif change <= -0.003:
            log.info(f"STOP LOSS {symbol}")
            self.place_order(symbol, "sell")
            del self.positions[symbol]

    # ---------------------------------------------------
    # Bar handler
    # ---------------------------------------------------

    async def on_bar(self, bar):

        s = bar.symbol
        price = bar.close
        volume = bar.volume

        log.info(f"{s} close {price}")

        self.bars[s].append({
            "price": price,
            "volume": volume
        })

        df = pd.DataFrame(self.bars[s]).tail(30)

        if len(df) < 10:
            return

        # -----------------------------------
        # Exit if already in trade
        # -----------------------------------

        if s in self.positions:
            self.check_exit(s, price)
            return

        # -----------------------------------
        # Indicators
        # -----------------------------------

        df["vwap"] = (df["price"] * df["volume"]).cumsum() / df["volume"].cumsum()

        vwap = df["vwap"].iloc[-1]

        avg_volume = df["volume"].mean()

        # -----------------------------------
        # Entry conditions
        # -----------------------------------
        log.info(f"{s} price={price:.2f} vwap={vwap:.2f} vol={volume} avg={int(avg_volume)}")    
    
        ema20 = df["price"].ewm(span=20).mean()
        ema50 = df["price"].ewm(span=50).mean()
        trend = price > ema20.iloc[-1] > ema50.iloc[-1]





        momentum = df["price"].iloc[-1] - df["price"].iloc[-5]

        breakout = price > vwap
        volume_spike = volume > avg_volume * 1.5

        if breakout and volume_spike and momentum > 0.2 and trend:

            log.info(f"BREAKOUT {s}")

            self.place_order(s, "buy",price=price)

            self.positions[s] = {
                "entry": price
            }
        
    # ---------------------------------------------------
    # Start bot
    # ---------------------------------------------------

    def run(self):

        preload_data()
        
        self.symbols = SYMBOLS
        #self.scan_top_movers()

        for s in self.symbols:

            self.bars[s] = []

            self.stream.subscribe_bars(self.on_bar, s)

        log.info(f"Subscribed to {self.symbols}")

        self.stream.run()


# =======================================================
# MAIN
# =======================================================

def main():

    bot = MomentumScannerBot(
        key="PKC7D4XB4OTV2VDEFUF5BRL33P",
        secret="DZxMp2TBaqQWZH3seGm7ZSWbHjw25xLBD7ZGrG4F4GaF",
    )

    bot.run()


if __name__ == "__main__":
    main()