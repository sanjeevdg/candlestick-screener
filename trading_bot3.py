import asyncio
import logging
import pandas as pd

from alpaca.data.live import StockDataStream
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# ------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("BOT")


class MomentumBot:

    def __init__(self, symbols, key, secret):
        self.symbols = symbols
        self.bars = {s: [] for s in symbols}

        self.position = None
        self.entry_price = None

        self.trading_client = TradingClient(key, secret, paper=True)
        self.stream = StockDataStream(key, secret)

        for s in self.symbols:
            self.stream.subscribe_bars(self.on_bar, s)

        log.info("MomentumScalpBot initialized")

    # ------------------------------------------------

    def place_order(self, symbol, side, qty=10):

        log.info(f"{side.upper()} {symbol} x{qty}")

        self.trading_client.submit_order(
            MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY
            )
        )

    # ------------------------------------------------

    def check_exit(self, price):

        if self.position is None:
            return

        change = (price - self.entry_price) / self.entry_price

        if change >= 0.01:
            log.info("TAKE PROFIT")
            self.place_order(self.position, "sell")
            self.position = None
            self.entry_price = None

        elif change <= -0.005:
            log.info("STOP LOSS")
            self.place_order(self.position, "sell")
            self.position = None
            self.entry_price = None

    # ------------------------------------------------

    async def on_bar(self, bar):

        s = bar.symbol
        price = bar.close

        log.info(f"{s} close={price}")

        self.bars[s].append(price)
        self.bars[s] = self.bars[s][-50:]

        df = pd.Series(self.bars[s]).tail(20)

        if len(df) < 5:
            return

        # Manage open position
        if self.position == s:
            self.check_exit(price)
            return

        if self.position is not None:
            return

        momentum = df.iloc[-1] - df.iloc[-5]

        if momentum > 0.3:
            log.info(f"MOMENTUM LONG {s}")

            self.place_order(s, "buy")

            self.position = s
            self.entry_price = price

    # ------------------------------------------------

    def run(self):
        log.info("Starting stream")
        self.stream.run()

# ------------------------------------------
def main():
    bot = MomentumBot(
        symbols=["TQQQ", "SQQQ"],
        key="PKC7D4XB4OTV2VDEFUF5BRL33P",
        secret="DZxMp2TBaqQWZH3seGm7ZSWbHjw25xLBD7ZGrG4F4GaF",
    )

    bot.run()


if __name__ == "__main__":
    main()

