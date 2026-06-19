# alpaca_stream.py
import asyncio
import threading
from typing import Set, Callable

from alpaca.data.live.stock import StockDataStream
from alpaca.data.enums import DataFeed


API_KEY = "PKC7D4XB4OTV2VDEFUF5BRL33P"
API_SECRET = "DZxMp2TBaqQWZH3seGm7ZSWbHjw25xLBD7ZGrG4F4GaF"


# -------------------------------
# INTERNAL STATE
# -------------------------------

_stream: StockDataStream | None = None
_loop: asyncio.AbstractEventLoop | None = None
_thread: threading.Thread | None = None
_started = False
_lock = threading.Lock()

_subscribed: Set[str] = set()
_socketio = None


# -------------------------------
# BAR HANDLER
# -------------------------------

async def handle_bar(bar):
    if not _socketio:
        return

    _socketio.emit(
        "realtime_bar",
        {
            "symbol": bar.symbol,
            "time": int(bar.timestamp.timestamp()),
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
        },
    )


# -------------------------------
# STREAM THREAD
# -------------------------------

def _stream_runner():
    global _loop, _stream

    _loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_loop)

    _stream = StockDataStream(
        API_KEY,
        API_SECRET,
        feed=DataFeed.IEX,
    )

    # start async loop manually (NO asyncio.run)
    try:
        _loop.run_until_complete(_stream._run_forever())
    except ValueError as e:
        print("❌ Alpaca stream fatal error:", e)


# -------------------------------
# PUBLIC API
# -------------------------------

def start_stream(socketio):
    global _started, _thread, _socketio

    with _lock:
        if _started:
            return
        _started = True
        _socketio = socketio

    if _thread and _thread.is_alive():
        return

    print("🚀 Starting Alpaca websocket (ONCE)")

    _thread = threading.Thread(
        target=_stream_runner,
        daemon=True,
        name="alpaca-stream",
    )
    _thread.start()


def subscribe_symbol(raw_symbol: str):
    if not _stream:
        return

    if isinstance(raw_symbol, dict):
        raw_symbol = raw_symbol.get("symbol")

    if not isinstance(raw_symbol, str):
        raise ValueError(f"Invalid symbol: {raw_symbol}")

    clean = (
        raw_symbol
        .replace("NASDAQ:", "")
        .replace("NYSE:", "")
        .replace("AMEX:", "")
    )

    #clean = raw_symbol.replace("NASDAQ:", "").replace("AMEX:", "")

    if clean in _subscribed:
        return

    _subscribed.add(clean)
    print(f"📡 Subscribed to {clean}")

    _stream.subscribe_bars(handle_bar, clean)


def unsubscribe_symbol(raw_symbol: str):
    if not _stream:
        return

    clean = raw_symbol.split(":")[-1].upper()

    if clean not in _subscribed:
        return

    print(f"🛑 Unsubscribed from {clean}")
    _subscribed.remove(clean)

    _stream.unsubscribe_bars(clean)
