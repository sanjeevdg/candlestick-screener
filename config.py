from dotenv import load_dotenv
import os

# load .env file
load_dotenv()

# Alpaca API
ALPACA_KEY = os.getenv("ALPACA_KEY")
ALPACA_SECRET = os.getenv("ALPACA_SECRET")
ALPACA_PAPER = os.getenv("ALPACA_PAPER", "true").lower() == "true"

# Trading settings
DEFAULT_SYMBOLS = ["SPY"]
ORDER_SIZE = 1

# Safety check
if not ALPACA_KEY or not ALPACA_SECRET:
    raise ValueError("Missing Alpaca API credentials in .env")