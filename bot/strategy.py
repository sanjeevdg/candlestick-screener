class MomentumStrategy:

    def entry_signal(self, row, prev_row, atr_mean):

        cond1 = row["Histogram"] > prev_row["Histogram"]
        cond2 = row["ADX"] > 22
        cond3 = row["ROC"] > 0
        cond4 = row["ATR_14"] > atr_mean
        cond5 = row["close"] > row["EMA_50"]

        return cond1 and cond2 and cond3 and cond4 and cond5
