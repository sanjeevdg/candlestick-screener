import talib

def compute_indicators(df):

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values

    df["ROC"] = talib.ROC(close, timeperiod=10)

    macd, macd_signal, macd_hist = talib.MACD(
        close, 12, 26, 9
    )

    df["MACD"] = macd
    df["Signal"] = macd_signal
    df["Histogram"] = macd_hist

    df["EMA_50"] = talib.EMA(close, 50)
    df["ADX"] = talib.ADX(high, low, close, 14)
    df["ATR_14"] = talib.ATR(high, low, close, 14)

    return df
