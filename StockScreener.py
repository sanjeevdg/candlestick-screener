import os
import pandas as pd
import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta


class StockScreener:
    def __init__(self):
        self.api = tradeapi.REST(
            os.getenv("PKC7D4XB4OTV2VDEFUF5BRL33P"),
            os.getenv("DZxMp2TBaqQWZH3seGm7ZSWbHjw25xLBD7ZGrG4F4GaF"),
            os.getenv("https://paper-api.alpaca.markets"),
            api_version="v2",
        )
        self.df = None

    def fetch_bars(
        self,
        symbols,
        timeframe="1Day",
        lookback_days=200,
    ):
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

    def compute_indicators(self, df):
        df = df.copy()

        df["ma_5"] = df["close"].rolling(5).mean()
        df["ma_20"] = df["close"].rolling(20).mean()
        df["ma_50"] = df["close"].rolling(50).mean()

        df["rsi"] = self._rsi(df["close"])

        return df

    def _rsi(self, series, period=14):
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def screen(
        self,
        symbols,
        min_price=None,
        above_ma_20=False,
        rsi_below=None,
    ):
        bars = self.fetch_bars(symbols)
        results = []

        for symbol in symbols:
            sdf = bars[bars["symbol"] == symbol].copy()
            if len(sdf) < 50:
                continue

            sdf = self.compute_indicators(sdf)
            last = sdf.iloc[-1]

            if min_price and last["close"] < min_price:
                continue

            if above_ma_20 and last["close"] < last["ma_20"]:
                continue

            if rsi_below and last["rsi"] > rsi_below:
                continue

            results.append(
                {
                    "symbol": symbol,
                    "close": round(last["close"], 2),
                    "ma_20": round(last["ma_20"], 2),
                    "ma_50": round(last["ma_50"], 2),
                    "rsi": round(last["rsi"], 2),
                }
            )

        return results
