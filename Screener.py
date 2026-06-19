import os
import pandas as pd
import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta


class Screener:
    def __init__(self):
        self.api = tradeapi.REST(
            os.getenv("PKC7D4XB4OTV2VDEFUF5BRL33P"),
            os.getenv("DZxMp2TBaqQWZH3seGm7ZSWbHjw25xLBD7ZGrG4F4GaF"),
            os.getenv("https://paper-api.alpaca.markets"),
            api_version="v2",
        )

    def fetch_bars(self, symbols, timeframe="1Day", lookback_days=200):
        start = (datetime.utcnow() - timedelta(days=lookback_days)).isoformat()

        bars = self.api.get_bars(
            symbols,
            timeframe,
            start=start,
            adjustment="raw",
        ).df

        if isinstance(bars.index, pd.MultiIndex):
            bars = bars.reset_index(level=0)

        return bars

    def rsi(self, series, period=14):
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def screen(self, symbols):
        bars = self.fetch_bars(symbols)
        results = []

        for symbol in symbols:
            df = bars[bars["symbol"] == symbol].copy()
            if len(df) < 50:
                continue

            df["ma20"] = df["close"].rolling(20).mean()
            df["ma50"] = df["close"].rolling(50).mean()
            df["rsi"] = self.rsi(df["close"])

            last = df.iloc[-1]

            results.append(
                {
                    "symbol": symbol,
                    "close": round(float(last["close"]), 2),
                    "ma20": round(float(last["ma20"]), 2),
                    "ma50": round(float(last["ma50"]), 2),
                    "rsi": round(float(last["rsi"]), 2),
                }
            )

        return results
