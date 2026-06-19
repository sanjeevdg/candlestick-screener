from bot.strategy import MomentumStrategy
import pandas as pd

strategy = MomentumStrategy()

def run_backtest(df, symbol, sl, tp):

    equity = 1.0
    equity_curve = [equity]
    trades = []

    in_trade = False
    entry_price = 0

    for i in range(1, len(df)-1):

        row = df.iloc[i]
        prev = df.iloc[i-1]

        atr_mean = df["ATR_14"].rolling(50).mean().iloc[i]

        if in_trade:

            ret = (row.close - entry_price) / entry_price

            if ret >= tp or ret <= -sl:
                equity *= (1 + ret)
                trades.append(ret)
                equity_curve.append(equity)
                in_trade = False

            continue

        if strategy.entry_signal(row, prev, atr_mean):
            entry_price = df.iloc[i+1].open
            in_trade = True

    bars = len(df)
    total_trades = len(trades)

    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]

    win_rate = (len(wins) / total_trades * 100) if total_trades else 0

    avg_win = (sum(wins)/len(wins))*100 if wins else 0
    avg_loss = (sum(losses)/len(losses))*100 if losses else 0

    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))

    profit_factor = gross_profit/gross_loss if gross_loss else 0

    eq = pd.Series(equity_curve)
    max_dd = (eq/eq.cummax()-1).min()*100

    return {
        "symbol": symbol,
        "bars_tested": bars,
        "trades": total_trades,
        "win_rate_pct": round(win_rate,2),
        "average_win_pct": round(avg_win,3),
        "average_loss_pct": round(avg_loss,3),
        "profit_factor": round(profit_factor,3),
        "max_drawdown_pct": round(max_dd,2),
        "final_equity": round(equity,4)
    }