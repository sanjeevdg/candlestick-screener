import asyncio
import logging
import pandas as pd
import time

from alpaca.data.live import StockDataStream
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockSnapshotRequest

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce


# -------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

log = logging.getLogger("BOT")


# -------------------------------------------------
class MomentumBot:

    def __init__(self, key, secret):

        self.key = key
        self.secret = secret

        self.trading = TradingClient(key, secret, paper=True)

        self.data_client = StockHistoricalDataClient(key, secret)

        self.stream = StockDataStream(key, secret)

        self.symbols = []
        self.bars = {}

        self.positions = {}
        self.last_trade = {}

        self.qty = 10
        self.take_profit = 0.01
        self.stop_loss = -0.005
        self.cooldown = 120

        log.info("Bot initialized")

    # -------------------------------------------------
    # TOP MOVERS SCANNER
    # -------------------------------------------------

    def scan_top_movers(self):

        log.info("Scanning top movers...")

        # universe of liquid stocks
        universe = [
            "PRSO","EDSA","CRE","ANTX","DAWN","SEV"
        ]

        req = StockSnapshotRequest(symbol_or_symbols=universe)

        snapshots = self.data_client.get_stock_snapshot(req)

        movers = []

        for symbol, snap in snapshots.items():

            if snap.daily_bar is None:
                continue

            prev = snap.previous_daily_bar.close
            today = snap.daily_bar.close

            pct = (today - prev) / prev * 100

            movers.append((symbol, pct))

        movers.sort(key=lambda x: x[1], reverse=True)

        top = [m[0] for m in movers[:10]]

        log.info(f"Top movers: {top}")

        return top

    # -------------------------------------------------
    def place_order(self, symbol, side):

        log.info(f"{side.upper()} {symbol}")

        self.trading.submit_order(

            MarketOrderRequest(
                symbol=symbol,
                qty=self.qty,
                side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
                extended_hours=True
            )
        )
        log.info(f"ORDER SENT {order}")
    # -------------------------------------------------

    def can_trade(self, symbol):

        if symbol not in self.last_trade:
            return True

        return time.time() - self.last_trade[symbol] > self.cooldown

    # -------------------------------------------------

    def check_exit(self, symbol, price):

        entry = self.positions[symbol]

        change = (price - entry) / entry

        if change >= self.take_profit:

            log.info(f"{symbol} TAKE PROFIT")

            self.place_order(symbol, "sell")

            del self.positions[symbol]

        elif change <= self.stop_loss:

            log.info(f"{symbol} STOP LOSS")

            self.place_order(symbol, "sell")

            del self.positions[symbol]

    # -------------------------------------------------

    async def on_bar(self, bar):

        symbol = bar.symbol
        price = bar.close

        log.info(f"{symbol} close {price}")

        self.bars[symbol].append(price)

        df = pd.Series(self.bars[symbol]).tail(20)

        if len(df) < 10:
            return

        # exit logic

        if symbol in self.positions:
            self.check_exit(symbol, price)
            return

        # cooldown

        if not self.can_trade(symbol):
            return

        # simple momentum signal

        momentum = df.iloc[-1] - df.iloc[-5]
        log.info(f"{s} momentum {momentum:.3f}")
        if momentum > 0.01:

            log.info(f"{symbol} MOMENTUM ENTRY")

            self.place_order(symbol, "buy")

            self.positions[symbol] = price

            self.last_trade[symbol] = time.time()

    # -------------------------------------------------

    def start_stream(self):

        for s in self.symbols:

            self.bars[s] = []

            self.stream.subscribe_bars(self.on_bar, s)

        log.info(f"Subscribed to {self.symbols}")

        self.stream.run()

    # -------------------------------------------------

    def run(self):

        # scan market

        self.symbols = self.scan_top_movers()

        # start websocket

        self.start_stream()


# -------------------------------------------------

def main():

    bot = MomentumBot(

        key="PKC7D4XB4OTV2VDEFUF5BRL33P",
        secret="DZxMp2TBaqQWZH3seGm7ZSWbHjw25xLBD7ZGrG4F4GaF",
    )

    bot.run()


if __name__ == "__main__":
    main()
