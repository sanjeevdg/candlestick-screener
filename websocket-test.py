import yfinance as yf



def message_handler(msg):
    print("🟢 Got message:", msg)

ws = yf.WebSocket()
ws.subscribe(["BHF", "AIP", "LITE","FTRE","OKLL"])
ws.listen(message_handler)

