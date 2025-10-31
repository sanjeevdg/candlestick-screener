import yfinance as yf
t = yf.Ticker("NVDA")
print(t.fast_info.pre_market_price)
print(t.fast_info.post_market_price)
print(t.fast_info.last_price)