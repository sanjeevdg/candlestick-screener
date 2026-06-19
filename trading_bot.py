import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

log = logging.getLogger("BOT")

import threading
import pandas as pd
from datetime import datetime, timezone
from alpaca.data.live import StockDataStream
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, TakeProfitRequest, StopLossRequest
from alpaca.trading.enums import OrderSide, TimeInForce

class MomentumScalpBot:
    def __init__(self, symbols, key, secret):

        log.info("Initializing MomentumScalpBot")

        self.symbols = symbols
        self.running = False
        self.bars = {s: [] for s in symbols}
        self.in_position = set()

        self.trading_client = TradingClient(key, secret, paper=True)
        log.info("TradingClient initialized")
        self.stream = StockDataStream(key, secret)
        log.info("StockDataStream initialized")

    def is_momentum_long(self, df):
        if len(df) < 60:
            log.info(f"Not enough bars: {len(df)}")
            return False

        df["ema9"] = df["close"].ewm(span=9).mean()
        df["ema21"] = df["close"].ewm(span=21).mean()
        df["ema50"] = df["close"].ewm(span=50).mean()

        last = df.iloc[-1]
        prev = df.iloc[-2]

        avg_vol = df["volume"].rolling(20).mean().iloc[-1]
        body = abs(last["close"] - last["open"])
        range_ = last["high"] - last["low"]

        log.info(
            f"CHECK {last.name} | "
            f"ema9={last.ema9:.2f} "
            f"ema21={last.ema21:.2f} "
            f"ema50={last.ema50:.2f} "
            f"vol={last.volume:.0f}/{avg_vol:.0f} "
            f"body_ratio={body / range_:.2f}"
        )

        return (
            last["ema9"] > last["ema21"] > last["ema50"] and
            last["close"] > prev["high"] and
            last["volume"] > 1.5 * avg_vol and
            body / range_ > 0.6
        )
    
    def _run_stream(self):
        log.info("Starting Alpaca data websocket")

        try:
            self.stream.run()  # blocking call
        except Exception as e:
            log.exception("Alpaca WS crashed — stopping bot")
        finally:
            self.running = False
            log.warning("Bot stopped — manual restart required")

    # ---------------- ORDER ----------------
    def place_bracket(self, symbol, qty):
        last_price = self.bars[symbol][-1]["close"]

        tp = round(last_price * 1.008, 2)  # +0.8%
        sl = round(last_price * 0.995, 2)  # -0.5%

        log.info(
            f"PLACING ORDER {symbol} | "
            f"price={last_price} TP={tp} SL={sl}"
        )
        order = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
            take_profit=TakeProfitRequest(limit_price=tp),
            stop_loss=StopLossRequest(stop_price=sl)
        )

        self.trading_client.submit_order(order)
        self.in_position.add(symbol)

        print(f"[BOT] LONG {symbol} @ {last_price} | TP {tp} SL {sl}")

    # ---------------- DATA HANDLER ----------------
    async def on_bar(self, bar):
        log.info(f"BAR RECEIVED {bar.symbol} {bar.close}")

        if not self.running:
            log.warning("Bot not running — bar ignored")
            return

        clock = self.trading_client.get_clock()
        if not clock.is_open:
            log.info("Market closed — bar ignored")
            return

        symbol = bar.symbol
        self.bars[symbol].append({
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume
        })

        log.info(f"{symbol} bars stored: {len(self.bars[symbol])}")

        df = pd.DataFrame(self.bars[symbol])

        if symbol not in self.in_position:
            if self.is_momentum_long(df):
                log.info(f"ENTRY SIGNAL for {symbol}")
                self.place_bracket(symbol, qty=10)
            else:
                log.info(f"No signal for {symbol}")


    # ---------------- CONTROL ----------------
    def start(self):
        if self.running:
            log.warning("Bot already running")
            return

        self.running = True
        log.info("BOT START requested")

        for s in self.symbols:
            log.info(f"Subscribing to bars for {s}")
            self.stream.subscribe_bars(self.on_bar, s)
        '''    
        threading.Thread(
            target=self.stream.run,
            daemon=True
        ).start()
        '''
        threading.Thread(target=self._run_stream, daemon=True).start()
        log.info("Alpaca stream thread started")


    def stop(self):
        log.info("BOT STOP requested")
        self.running = False
        self.stream.stop()
