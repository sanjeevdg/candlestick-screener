import os
import pandas as pd

def is_consolidating(df, percentage=2):
    # Skip if not enough candles
    if len(df) < 15:
        return False

    
    print(f"📉 {df} this is df............")


    recent_candlesticks = df.iloc[-15:]
    print(f"📉 {recent_candlesticks} this is recent_candlesticks............")

    max_close = recent_candlesticks['Close'].max()
    min_close = recent_candlesticks['Close'].min()

    threshold = 1 - (percentage / 100)
    return min_close > (max_close * threshold)


def is_breaking_out(df, percentage=2.5):
    # Skip if not enough candles
    if len(df) < 16:
        return False

    last_close = df.iloc[-1]['Close']

    if is_consolidating(df.iloc[:-1], percentage=percentage):
        recent_closes = df.iloc[-16:-1]
        if last_close > recent_closes['Close'].max():
            return True

    return False


def main():
    print("🚀 Scanning charts in datasets/daily ...\n")

    for filename in os.listdir('datasets/daily'):
        if not filename.endswith('.csv'):
            continue

        path = f'datasets/daily/{filename}'
        df = pd.read_csv(path)

        if df.empty:
            continue

        symbol = filename.replace('.csv', '')

        if is_consolidating(df, percentage=2.5):
            print(f"📉 {symbol} is consolidating")

        if is_breaking_out(df):
            print(f"🚀 {symbol} is breaking out")


if __name__ == "__main__":
    main()
