from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from config import ALPACA_KEY, ALPACA_SECRET





client = TradingClient(ALPACA_KEY, ALPACA_SECRET, paper=True)

def place_order(symbol):

    order = MarketOrderRequest(
        symbol=symbol,
        qty=10,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY
    )

    client.submit_order(order)
