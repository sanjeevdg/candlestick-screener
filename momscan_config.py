from config import ALPACA_KEY, ALPACA_SECRET# config.py

from alpaca.data import StockHistoricalDataClient

data_client = StockHistoricalDataClient(ALPACA_KEY, ALPACA_SECRET)