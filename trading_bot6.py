import os
import time
import logging
import pandas as pd

from alpaca.data.live import StockDataStream
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockSnapshotRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

log = logging.getLogger("BOT")


class MomentumBot:

    def __init__(self):

        key="PKC7D4XB4OTV2VDEFUF5BRL33P"
        secret="DZxMp2TBaqQWZH3seGm7ZSWbHjw25xLBD7ZGrG4F4GaF"

        self.trading = TradingClient(key, secret, paper=True)
        self.data = StockHistoricalDataClient(key, secret)
        self.stream = StockDataStream(key, secret)

        self.symbols = []
        self.bars = {}
        self.positions = {}

        log.info("Bot initialized")

    # -------------------------------------------------
    # MARKET SCANNER
    # -------------------------------------------------

    def scan_market(self):

        log.info("Scanning market...")

        universe = [
            "TLYS","LWLG","VRA","WOOF","POLA","ACDC","BATL","CVGI"
        ]

        request = StockSnapshotRequest(symbol_or_symbols=universe)

        snapshots = self.data.get_stock_snapshot(request)

        movers = []

        for symbol, snap in snapshots.items():

            if snap.daily_bar and snap.previous_daily_bar:

                price = snap.daily_bar.close
                prev = snap.previous_daily_bar.close
                volume = snap.daily_bar.volume

                change = (price - prev) / prev * 100

                movers.append((symbol, change, volume))

        movers.sort(key=lambda x: x[1], reverse=True)

        top = [m[0] for m in movers[:5]]

        log.info(f"Top movers: {top}")

        return top

    # -------------------------------------------------
    # PLACE ORDER
    # -------------------------------------------------

    def place_order(self, symbol, side, qty=10):

        log.info(f"{side.upper()} {symbol}")

        order = self.trading.submit_order(
            MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
                extended_hours=True
            )
        )

        log.info(f"Order placed {order.id}")

    # -------------------------------------------------
    # EXIT LOGIC
    # -------------------------------------------------

    def check_exit(self, symbol, price):

        entry = self.positions[symbol]

        change = (price - entry) / entry

        if change >= 0.01:

            log.info(f"TAKE PROFIT {symbol}")

            self.place_order(symbol, "sell")

            del self.positions[symbol]

        elif change <= -0.005:

            log.info(f"STOP LOSS {symbol}")

            self.place_order(symbol, "sell")

            del self.positions[symbol]

    # -------------------------------------------------
    # BAR EVENT
    # -------------------------------------------------

    async def on_bar(self, bar):

        s = bar.symbol
        price = bar.close
        volume = bar.volume

        self.bars[s].append({
            "price": price,
            "volume": volume
        })

        df = pd.DataFrame(self.bars[s]).tail(30)

        if len(df) < 10:
            return

        if s in self.positions:

            self.check_exit(s, price)
            return

        df["vwap"] = (df["price"] * df["volume"]).cumsum() / df["volume"].cumsum()

        vwap = df["vwap"].iloc[-1]
        avg_vol = df["volume"].mean()

        momentum = df["price"].iloc[-1] - df["price"].iloc[-5]

        if price > vwap and volume > avg_vol * 1.3 and momentum > 0.15:

            log.info(f"BREAKOUT {s}")

            self.place_order(s, "buy")

            self.positions[s] = price

    # -------------------------------------------------
    # START STREAM
    # -------------------------------------------------

    def start_stream(self):

        for s in self.symbols:

            self.bars[s] = []

            self.stream.subscribe_bars(self.on_bar, s)

        log.info(f"Subscribed to {self.symbols}")

        self.stream.run()

    # -------------------------------------------------
    # MAIN LOOP
    # -------------------------------------------------

    def run(self):

        while True:

            try:

                self.symbols = self.scan_market()

                self.start_stream()

                time.sleep(600)  # rescan every 10 minutes

            except Exception as e:

                log.error(e)

                time.sleep(30)


# -------------------------------------------------

def main():

    bot = MomentumBot()

    bot.run()


if __name__ == "__main__":
    main()
