import pandas as pd
import json

from bot.indicators import compute_indicators
from bot.backtest_engine import run_backtest

FILE = "data/bars_5Min.csv"
SL = 0.005
TP = 0.018

df = pd.read_csv(FILE)

symbols = df["symbol"].unique()

results = []
total_trades = 0

for sym in symbols:

    sdf = df[df["symbol"] == sym].copy()
    sdf = compute_indicators(sdf)

    res = run_backtest(sdf, sym, SL, TP)

    total_trades += res["trades"]
    results.append(res)

report = {
    "file": FILE,
    "stop_loss_pct": SL*100,
    "take_profit_pct": TP*100,
    "symbols_tested": len(results),
    "total_trades": total_trades,
    "results": results
}

print(json.dumps(report))