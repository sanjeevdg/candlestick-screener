import pandas as pd
import talib
from datetime import datetime, timedelta
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed
from alpaca.data.live import StockDataStream
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from dataclasses import dataclass
from typing import List, Dict, Optional
from config import ALPACA_KEY, ALPACA_SECRET
import argparse
import time
import logging
import threading
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
TIMEFRAME = TimeFrame.Day
DATA_FEED = DataFeed.IEX  # SIP is required for TQQQ/SQQQ
PAPER_TRADE = True
TRADE_SIZE = 100

parser = argparse.ArgumentParser()
parser.add_argument("--symbols")
parser.add_argument("--interval", type=int, default=60, help="Check interval in seconds")
args = parser.parse_args()

SYMBOLS = args.symbols.split(",")
CHECK_INTERVAL = args.interval

# Shared state for live data
latest_bars = {}
latest_bars_lock = threading.Lock()

@dataclass
class Trade:
    symbol: str
    entry_price: float
    entry_time: datetime
    side: str
    size: float
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl: float = 0.0
    order_id: Optional[str] = None

    def close(self, price: float, time: datetime):
        self.exit_price = price
        self.exit_time = time
        if self.side == 'long':
            self.pnl = (price - self.entry_price) * self.size
        else:
            self.pnl = (self.entry_price - price) * self.size

class SymbolSession:
    def __init__(self, symbol: str, timeframe: TimeFrame, trading_client: TradingClient):
        self.symbol = symbol
        self.timeframe = timeframe
        self.data_client = StockHistoricalDataClient(ALPACA_KEY, ALPACA_SECRET)
        self.trading_client = trading_client
        self.bars: List[pd.Series] = []
        self.trades: List[Trade] = []
        self.in_position = False
        self.position: Optional[Trade] = None
        self.last_bar_time = None

    def preload_historical_bars(self, days: int = 100) -> pd.DataFrame:
        end = datetime.now()
        start = end - timedelta(days=days)
        request_params = StockBarsRequest(
            symbol_or_symbols=[self.symbol],
            timeframe=self.timeframe,
            start=start,
            end=end,
            feed=DATA_FEED
        )
        try:
            bars_dict = self.data_client.get_stock_bars(request_params)
            if self.symbol not in bars_dict:
                raise ValueError(f"Symbol {self.symbol} not found in historical data feed {DATA_FEED}.")
            bars_list = bars_dict[self.symbol]
            if not bars_list:
                raise ValueError(f"No historical data returned for {self.symbol}.")
            
            data = []
            for bar in bars_list:
                data.append({
                    'timestamp': bar.timestamp,
                    'open': bar.open,
                    'high': bar.high,
                    'low': bar.low,
                    'close': bar.close,
                    'volume': bar.volume,
                    'trade_count': bar.trade_count,
                    'vwap': bar.vwap
                })
            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            return df.sort_index()
        except Exception as e:
            logger.error(f"Error preloading historical data for {self.symbol}: {e}")
            raise

    def get_latest_bar(self) -> Optional[pd.Series]:
        """
        Fetches the latest bar from the shared live data stream.
        Falls back to historical client if stream hasn't updated yet.
        """
        # 1. Try to get from live stream
        with latest_bars_lock:
            if self.symbol in latest_bars:
                bar_data = latest_bars[self.symbol]
                return pd.Series(bar_data)

        # 2. Fallback: Fetch last 1 bar from historical client
        end = datetime.now()
        start = end - timedelta(hours=1)
        request_params = StockBarsRequest(
            symbol_or_symbols=[self.symbol],
            timeframe=self.timeframe,
            start=start,
            end=end,
            feed=DATA_FEED,
            limit=1
        )
        try:
            bars_dict = self.data_client.get_stock_bars(request_params)
            if self.symbol in bars_dict and bars_dict[self.symbol]:
                latest_bar = bars_dict[self.symbol][-1]
                return pd.Series({
                    'timestamp': latest_bar.timestamp,
                    'open': latest_bar.open,
                    'high': latest_bar.high,
                    'low': latest_bar.low,
                    'close': latest_bar.close,
                    'volume': latest_bar.volume,
                    'trade_count': latest_bar.trade_count,
                    'vwap': latest_bar.vwap
                })
        except Exception as e:
            logger.warning(f"Fallback historical fetch failed for {self.symbol}: {e}")
        return None

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        close = df['close'].values
        df['ema9'] = talib.EMA(close, timeperiod=9)
        df['ema20'] = talib.EMA(close, timeperiod=20)
        df['rsi'] = talib.RSI(close, timeperiod=14)
        return df

    def determine_entry_signal(self, df: pd.DataFrame, last_idx: int) -> Optional[str]:
        if last_idx < 25:
            return None
        curr = df.iloc[last_idx]
        if pd.isna(curr['ema9']) or pd.isna(curr['ema20']) or pd.isna(curr['rsi']):
            return None
        
        ema_trend_long = curr['ema9'] > curr['ema20']
        ema_trend_short = curr['ema9'] < curr['ema20']
        rsi_oversold = curr['rsi'] < 30
        rsi_overbought = curr['rsi'] > 70

        recent_highs = df['high'].iloc[max(0, last_idx-5):last_idx].max()
        breakout_long = curr['close'] > recent_highs
        recent_ema20 = df['ema20'].iloc[max(0, last_idx-3):last_idx]
        pullback_long = (curr['low'] <= recent_ema20.max() * 1.01) and (curr['close'] > curr['ema20']) and ema_trend_long

        if ema_trend_long and (rsi_oversold or pullback_long or breakout_long):
            return 'long'

        recent_lows = df['low'].iloc[max(0, last_idx-5):last_idx].min()
        breakout_short = curr['close'] < recent_lows
        pullback_short = (curr['high'] >= recent_ema20.min() * 0.99) and (curr['close'] < curr['ema20']) and ema_trend_short

        if ema_trend_short and (rsi_overbought or pullback_short or breakout_short):
            return 'short'
        return None

    def place_order(self, side: str, price: float):
        try:
            order_side = OrderSide.BUY if side == 'long' else OrderSide.SELL
            order_data = MarketOrderRequest(
                symbol=self.symbol,
                qty=TRADE_SIZE,
                side=order_side,
                time_in_force=TimeInForce.DAY
            )
            order = self.trading_client.submit_order(order_data=order_data)
            logger.info(f"[{self.symbol}] ORDER PLACED: {side.upper()} {TRADE_SIZE} @ market")
            return order
        except Exception as e:
            logger.error(f"[{self.symbol}] ORDER FAILED: {e}")
            return None

    def close_position(self, price: float):
        if not self.position:
            return
        try:
            order_side = OrderSide.SELL if self.position.side == 'long' else OrderSide.BUY
            order_data = MarketOrderRequest(
                symbol=self.symbol,
                qty=self.position.size,
                side=order_side,
                time_in_force=TimeInForce.DAY
            )
            order = self.trading_client.submit_order(order_data=order_data)
            self.position.close(price, datetime.now())
            self.position.order_id = order.id if order else None
            logger.info(f"[{self.symbol}] POSITION CLOSED: {self.position.side.upper()} @ {price:.2f} | PnL: ${self.position.pnl:.2f}")
            self.in_position = False
            self.position = None
        except Exception as e:
            logger.error(f"[{self.symbol}] CLOSE POSITION FAILED: {e}")

    def on_bar(self, bar: pd.Series):
        self.bars.append(bar)
        if len(self.bars) > 100:
            self.bars = self.bars[-100:]
        
        df = pd.DataFrame(self.bars)
        df = self.calculate_indicators(df)
        current_idx = len(df) - 1
        
        signal = self.determine_entry_signal(df, current_idx)
        
        if not self.in_position:
            if signal:
                order = self.place_order(signal, bar['close'])
                if order:
                    self.in_position = True
                    trade = Trade(
                        symbol=self.symbol,
                        entry_price=bar['close'],
                        entry_time=datetime.now(),
                        side=signal,
                        size=TRADE_SIZE,
                        order_id=order.id
                    )
                    self.position = trade
                    self.trades.append(trade)
                    logger.info(f"[{self.symbol}] OPENED {signal.upper()} @ {bar['close']:.2f}")
        else:
            if signal and signal != self.position.side:
                self.close_position(bar['close'])

    def check_existing_position(self):
        try:
            positions = self.trading_client.get_all_positions()
            for pos in positions:
                if pos.symbol == self.symbol:
                    qty = float(pos.qty)
                    if qty > 0:
                        self.in_position = True
                        self.position = Trade(
                            symbol=self.symbol,
                            entry_price=float(pos.avg_entry_price),
                            entry_time=datetime.now(),
                            side='long',
                            size=qty,
                            order_id=pos.id
                        )
                        logger.info(f"[{self.symbol}] RESTORED EXISTING LONG POSITION: {qty} @ {self.position.entry_price:.2f}")
                    elif qty < 0:
                        self.in_position = True
                        self.position = Trade(
                            symbol=self.symbol,
                            entry_price=float(pos.avg_entry_price),
                            entry_time=datetime.now(),
                            side='short',
                            size=abs(qty),
                            order_id=pos.id
                        )
                        logger.info(f"[{self.symbol}] RESTORED EXISTING SHORT POSITION: {abs(qty)} @ {self.position.entry_price:.2f}")
                    break
        except Exception as e:
            logger.error(f"Error checking existing positions: {e}")

# --- Live Data Stream Handler ---
async def run_stream(symbols, on_bar_callback):
    """Runs the StockDataStream in a separate thread."""
    async with StockDataStream(ALPACA_KEY,ALPACA_SECRET) as stream:
        async def on_bar(bar):
            with latest_bars_lock:
                latest_bars[bar.symbol] = {
                    'timestamp': bar.timestamp,
                    'open': bar.open,
                    'high': bar.high,
                    'low': bar.low,
                    'close': bar.close,
                    'volume': bar.volume,
                    'trade_count': bar.trade_count,
                    'vwap': bar.vwap
                }
            if on_bar_callback:
                on_bar_callback(bar)
        
        await stream.subscribe_bars(on_bar, *symbols)
        logger.info(f"Stream subscribed to: {symbols}")
        
        # Keep the stream running
        while True:
            await asyncio.sleep(1)

class LiveTradingBot:
    def __init__(self, symbols: List[str], timeframe: TimeFrame, paper: bool = True):
        self.symbols = symbols
        self.timeframe = timeframe
        self.paper = paper
        self.trading_client = TradingClient(ALPACA_KEY, ALPACA_SECRET, paper=paper)
        self.sessions: Dict[str, SymbolSession] = {
            s: SymbolSession(s, timeframe, self.trading_client) for s in symbols
        }
        self._preload_all_data()
        self._start_stream()

    def _preload_all_data(self):
        logger.info(f"Preloading historical data using Feed: {DATA_FEED}...")
        for symbol, session in self.sessions.items():
            try:
                bars = session.preload_historical_bars(days=100)
                logger.info(f"Loaded {len(bars)} bars for {symbol}")
                session.check_existing_position()
            except Exception as e:
                logger.error(f"Error loading data for {symbol}: {e}")

    def _start_stream(self):
        """Starts the data stream in a background thread."""
        def run_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(run_stream(self.symbols, None))
        
        self.stream_thread = threading.Thread(target=run_loop, daemon=True)
        self.stream_thread.start()
        logger.info("Live data stream started in background thread.")

    def run(self):
        logger.info(f"Starting live paper trading for {len(self.symbols)} symbols")
        logger.info(f"Check interval: {CHECK_INTERVAL} seconds")
        logger.info(f"Paper trading: {self.paper}")
        logger.info(f"Data Feed: {DATA_FEED}")
        
        try:
            while True:
                for symbol, session in self.sessions.items():
                    try:
                        latest_bar = session.get_latest_bar()
                        if latest_bar is None:
                            continue
                        if session.last_bar_time == latest_bar['timestamp']:
                            continue
                        session.last_bar_time = latest_bar['timestamp']
                        session.on_bar(latest_bar)
                    except Exception as e:
                        logger.error(f"Error processing {symbol}: {e}")
                time.sleep(CHECK_INTERVAL)
        except KeyboardInterrupt:
            logger.info("Trading bot stopped by user")
        except Exception as e:
            logger.error(f"Fatal error: {e}")

    def get_report(self) -> Dict:
        all_trades = []
        total_pnl = 0.0
        for session in self.sessions.values():
            all_trades.extend(session.trades)
            total_pnl += sum(t.pnl for t in session.trades if t.exit_price is not None)
        
        completed = [t for t in all_trades if t.exit_price is not None]
        if not completed:
            return {'total_trades': 0, 'win_rate': 0.0, 'total_pnl': total_pnl, 'trades': []}
        
        wins = [t for t in completed if t.pnl > 0]
        win_rate = (len(wins) / len(completed)) * 100
        return {
            'total_trades': len(completed),
            'winning_trades': len(wins),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'trades': completed
        }

if __name__ == "__main__":
    bot = LiveTradingBot(SYMBOLS, TIMEFRAME, paper=PAPER_TRADE)
    report = bot.get_report()
    print(f"\nInitial Report: {report['total_trades']} trades, PnL: ${report['total_pnl']:.2f}")
    bot.run()
