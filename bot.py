# bot.py
import asyncio
import logging
import pandas as pd

from alpaca.data.live import StockDataStream
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("BOT")


class MomentumScalpBot:
    def __init__(self, symbols, key, secret):
        self.symbols = symbols
        self.running = False
        self.bars = {s: [] for s in symbols}
        self.in_position = set()

        self.trading_client = TradingClient(
            "PKC7D4XB4OTV2VDEFUF5BRL33P",
            "DZxMp2TBaqQWZH3seGm7ZSWbHjw25xLBD7ZGrG4F4GaF",
            paper=True
        )

        self.stream = StockDataStream(key, secret)

        # subscribe handlers
        for s in symbols:
            self.stream.subscribe_bars(self.on_bar, s)

        log.info("MomentumScalpBot initialized")

    async def run(self):
        log.info("MomentumScalpBot entering run loop")
        self.running = True

        while self.running:
            try:
                await self.stream._run_forever()
            except asyncio.CancelledError:
                break
            except Exception:
                log.exception("Stream error, reconnecting in 5s")
                await asyncio.sleep(5)

        log.warning("MomentumScalpBot stopped")

    def stop(self):
        self.running = False
        self.stream.stop()

    def market_open(self):
        return self.trading_client.get_clock().is_open

    def place_market(self, symbol, qty=10):
        log.info(f"BUY {symbol} x{qty}")
        self.trading_client.submit_order(
            MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY
            )
        )
        self.in_position.add(symbol)

    async def on_bar(self, bar):
        if not self.running or not self.market_open():
            return

        s = bar.symbol

        self.bars[s].append({
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume
        })

        df = pd.DataFrame(self.bars[s]).tail(50)

        log.info(f"{s} close={bar.close}")

        if s not in self.in_position and len(df) >= 3:
            self.place_market(s)


