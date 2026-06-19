import pandas as pd
from bot.backtest_engine import run_backtest

FILE = "data/bars_5Min.csv"

df = pd.read_csv(FILE)

equity, trades = run_backtest(
    df=df,
    sl=0.0075,
    tp=0.015
)

print("Final equity:", equity)
print("Trades:", len(trades))
