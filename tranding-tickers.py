import yfinance as yf

# Fetch the top 20 trending tickers from the U.S. market
trending_tickers = yf.trending.get_trending_tickers()

# Alternatively, fetch the list of most active tickers, which indicates popularity
most_active_tickers = yf.most_active.get_most_active_tickers()

# Print the list of tickers
print("Trending Tickers:")
print(trending_tickers)
print("\nMost Active Tickers:")
print(most_active_tickers)
