def backtest(df, sl=0.0075, tp=0.015):
    df = df.copy()

    in_trade = False
    entry_price = 0.0

    equity = [1.0]
    trades = []

    for i in range(1, len(df)):
        price = df.loc[i, "close"]

        # ---------------- ENTRY ----------------
        if not in_trade and df.loc[i-1, "entry"]:
            in_trade = True
            entry_price = price
            entry_time = df.loc[i, "time"]
            equity.append(equity[-1])
            continue

        # ---------------- IN TRADE ----------------
        if in_trade:
            ret = (price - entry_price) / entry_price

            # TAKE PROFIT
            if ret >= tp:
                equity.append(equity[-1] * (1 + ret))
                trades.append(ret)
                in_trade = False
                continue

            # STOP LOSS
            if ret <= -sl:
                equity.append(equity[-1] * (1 + ret))
                trades.append(ret)
                in_trade = False
                continue

            equity.append(equity[-1])

        else:
            equity.append(equity[-1])

    df["equity"] = equity
    return df, trades