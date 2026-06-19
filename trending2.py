from yahoo_fin import stock_info as si

# Most active tickers (based on trading volume)
most_active = si.get_day_most_active()
print(most_active.head())

# Trending tickers (most popular on Yahoo Finance)
trending = si.get_trending_tickers()
print(trending)