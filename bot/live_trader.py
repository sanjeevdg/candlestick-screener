from alpaca.data.live import StockDataStream
import pandas as pd

from bot.strategy import MomentumStrategy
from bot.execution import place_order
from bot.indicators import compute_indicators
from config import ALPACA_KEY, ALPACA_SECRET
from alpaca.data.enums import DataFeed

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta


SYMBOLS = ["PRSO","EDSA","ANTX","DAWN","SEV","COHN","MVLL","QURE"]

stream = StockDataStream(ALPACA_KEY, ALPACA_SECRET)
data_client = StockHistoricalDataClient(ALPACA_KEY, ALPACA_SECRET)
strategy = MomentumStrategy()

# store per symbol dataframe
data = {}


async def on_bar(bar):

    symbol = bar.symbol

    print(f"BAR: {symbol} {bar.timestamp} close={bar.close}")

    row = {
        "time": bar.timestamp,
        "open": bar.open,
        "high": bar.high,
        "low": bar.low,
        "close": bar.close,
        "volume": bar.volume,
    }

    # initialize dataframe
    if symbol not in data:
        data[symbol] = pd.DataFrame()

    # append bar
    data[symbol] = pd.concat(
        [data[symbol], pd.DataFrame([row])],
        ignore_index=True
    )

    df = data[symbol]

    # need enough bars
    if len(df) < 60:
        print(f"{symbol}: waiting for data ({len(df)}/60)")
        return

    # compute indicators
    df = compute_indicators(df)
    data[symbol] = df

    row = df.iloc[-1]
    prev = df.iloc[-2]

    atr_mean = df["ATR_14"].rolling(50).mean().iloc[-1]

    signal = strategy.entry_signal(row, prev, atr_mean)

    print(
        f"{symbol} close={row['close']} "
        f"ATR={row['ATR_14']:.4f} "
        f"signal={signal}"
    )

    if signal:
        print(f"BUY SIGNAL: {symbol} @ {row['close']}")
        place_order(symbol)

def preload_data():

    print("Loading historical bars...")

    end = datetime.utcnow()
    start = end - timedelta(minutes=120)

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

        data[symbol] = df

        print(f"{symbol} loaded {len(df)} bars")


def start_bot():

    preload_data()

    for symbol in SYMBOLS:
        print(f"Subscribing to {symbol}")
        stream.subscribe_bars(on_bar, symbol)

    print("🚀 Alpaca paper trading bot started")

    stream.run()


if __name__ == "__main__":
    start_bot()