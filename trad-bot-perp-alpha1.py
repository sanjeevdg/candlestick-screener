# backtest_alpaca_simple_strategy_multi.py

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from config import ALPACA_KEY, ALPACA_SECRET
from alpaca.data.enums import DataFeed


# === Configuration (edit these) ===
#ALPACA_API_KEY = "YOUR_API_KEY"  # replace
#ALPACA_API_SECRET = "YOUR_SECRET_KEY"  # replace
SYMBOLS = ["MITK","WSHP","CUK","MNRO","CCL","FLS"]  # add as many symbols as needed
TIMEFRAME = TimeFrame.Day
START_DATE = (datetime.now() - timedelta(days=365)).isoformat()
END_DATE = datetime.now().isoformat()
INITIAL_CAPITAL_PER_SYMBOL = 10000.0  # each symbol gets its own capital

# === Simple strategy: SMA crossover (5 vs 20) ===
def add_indicators(df):
    df["sma_5"] = df["close"].rolling(5).mean()
    df["sma_20"] = df["close"].rolling(20).mean()
    df["signal"] = 0
    df["signal"] = np.where(
        (df["sma_5"] > df["sma_20"])
        & (df["sma_5"].shift(1) <= df["sma_20"].shift(1)),
        1,
        df["signal"],
    )
    df["signal"] = np.where(
        (df["sma_5"] < df["sma_20"])
        & (df["sma_5"].shift(1) >= df["sma_20"].shift(1)),
        -1,
        df["signal"],
    )
    return df

# === Backtest single symbol ===
def backtest_symbol(symbol, client, initial_capital, start, end, timeframe):
    print(f"Fetching data for {symbol}...")

    # 1. Fetch historical data
    request_params = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=timeframe,
        start=pd.Timestamp(start, tz="America/New_York"),
        end=pd.Timestamp(end, tz="America/New_York"),
        feed=DataFeed.IEX
    )
    try:
        bars = client.get_stock_bars(request_params).df
        bars = bars.reset_index()
    except Exception as e:
        print(f"⚠️  Failed to fetch data for {symbol}: {e}")
        return None

    # 2. Add indicators and signals
    df = bars.copy()
    df = add_indicators(df)

    # 3. Initialize backtest variables
    cash = initial_capital
    position = 0
    equity = initial_capital
    trades = []

    # 4. Simulate each bar (day)
    for i in range(1, len(df)):
        row = df.iloc[i]
        close = row["close"]
        signal = row["signal"]

        # Buy on 5 crossing above 20 (only if not already long)
        if signal == 1 and position == 0:
            shares = cash // close
            if shares > 0:
                position = shares
                cost = shares * close
                cash -= cost
                trades.append({
                    "symbol": symbol,
                    "date": row["timestamp"],
                    "type": "buy",
                    "price": close,
                    "shares": shares,
                    "equity_before": cash + cost,
                    "equity_after": cash + position * close,
                })

        # Sell on 5 crossing below 20 (only if holding)
        elif signal == -1 and position > 0:
            revenue = position * close
            cash += revenue
            trades.append({
                "symbol": symbol,
                "date": row["timestamp"],
                "type": "sell",
                "price": close,
                "shares": position,
                "equity_before": cash - revenue,
                "equity_after": cash,
            })
            position = 0

        equity = cash + (position * close)

    # 5. Final stats for this symbol
    total_return = equity / initial_capital - 1.0
    stats = {
        "symbol": symbol,
        "initial_capital": initial_capital,
        "final_equity": equity,
        "total_return": total_return,
        "num_trades": len(trades),
    }

    print(f"→ {symbol}: ${equity:,.2f} ({total_return:.2%}) | Trades: {len(trades)}\n")

    return pd.DataFrame(trades), stats

# === Main execution (multi‑symbol) ===
if __name__ == "__main__":
    # Authenticate with Alpaca (paper / historical data only)
    client = StockHistoricalDataClient(
        api_key=ALPACA_KEY,
        secret_key=ALPACA_SECRET,
    )

    all_trades = []
    all_stats = []

    print(f"=== Backtesting {len(SYMBOLS)} symbols ===")
    print(f"From {START_DATE} to {END_DATE} | Timeframe: {TIMEFRAME}\n")

    for symbol in SYMBOLS:
        result = backtest_symbol(
            symbol=symbol,
            client=client,
            initial_capital=INITIAL_CAPITAL_PER_SYMBOL,
            start=START_DATE,
            end=END_DATE,
            timeframe=TIMEFRAME,
        )
        if result is not None:
            trades_df, stats = result
            if not trades_df.empty:
                all_trades.append(trades_df)
            all_stats.append(stats)

    # --- Overall summary ---
    if all_stats:
        df_stats = pd.DataFrame(all_stats)
        total_initial = df_stats["initial_capital"].sum()
        total_final = df_stats["final_equity"].sum()
        total_return = total_final / total_initial - 1.0

        print("=== Summary ===")
        print(f"Total initial capital: ${total_initial:,.2f}")
        print(f"Total final equity:    ${total_final:,.2f}")
        print(f"Aggregated return:     {total_return:.2%}")
        print(df_stats[["symbol", "initial_capital", "final_equity", "total_return"]])

    # --- Save all trades (optional) ---
    if all_trades:
        combined_trades = pd.concat(all_trades, ignore_index=True)
        combined_trades.to_csv("backtest_trades.csv", index=False)
        print("\nAll trades saved to 'backtest_trades.csv'.")


