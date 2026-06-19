import pandas as pd
import talib
from datetime import datetime, timedelta
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from config import ALPACA_KEY, ALPACA_SECRET
import argparse


# Configuration
#API_KEY = 'YOUR_API_KEY'
#API_SECRET = 'YOUR_API_SECRET'

TIMEFRAME = TimeFrame.Day
DATA_FEED = DataFeed.IEX


#SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'TSLA'] # Add as many symbols as you want




parser = argparse.ArgumentParser()

parser.add_argument("--symbols")

args = parser.parse_args()

SYMBOLS = args.symbols.split(",")



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

    def close(self, price: float, time: datetime):
        self.exit_price = price
        self.exit_time = time
        if self.side == 'long':
            self.pnl = (price - self.entry_price) * self.size
        else:
            self.pnl = (self.entry_price - price) * self.size

class SymbolSession:
    """Holds state for a single symbol simulation."""
    def __init__(self, symbol: str, timeframe: TimeFrame):
        self.symbol = symbol
        self.timeframe = timeframe
        self.client = StockHistoricalDataClient(ALPACA_KEY, ALPACA_SECRET)
        self.bars: List[pd.Series] = []
        self.trades: List[Trade] = []
        self.in_position = False
        self.position: Optional[Trade] = None

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
        
        bars_dict = self.client.get_stock_bars(request_params)
        bars_list = bars_dict[self.symbol]
        
        if not bars_list:
            raise ValueError(f"No data returned for {self.symbol}.")

        data = []
        for bar in bars_list:
            data.append({
                'timestamp': bar.timestamp,
                'open': bar.open, 'high': bar.high, 'low': bar.low, 
                'close': bar.close, 'volume': bar.volume, 
                'trade_count': bar.trade_count, 'vwap': bar.vwap
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df.sort_index()

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        close = df['close'].values
        df['ema9'] = talib.EMA(close, timeperiod=9)
        df['ema20'] = talib.EMA(close, timeperiod=20)
        df['rsi'] = talib.RSI(close, timeperiod=14)
        return df

    def determine_entry_signal(self, df: pd.DataFrame, last_idx: int) -> Optional[str]:
        if last_idx < 25: return None
        
        curr = df.iloc[last_idx]
        if pd.isna(curr['ema9']) or pd.isna(curr['ema20']) or pd.isna(curr['rsi']):
            return None

        ema_trend_long = curr['ema9'] > curr['ema20']
        ema_trend_short = curr['ema9'] < curr['ema20']
        rsi_oversold = curr['rsi'] < 30
        rsi_overbought = curr['rsi'] > 70
        
        # Breakout
        recent_highs = df['high'].iloc[max(0, last_idx-5):last_idx].max()
        breakout_long = curr['close'] > recent_highs
        
        # Pullback
        recent_ema20 = df['ema20'].iloc[max(0, last_idx-3):last_idx]
        pullback_long = (curr['low'] <= recent_ema20.max() * 1.01) and (curr['close'] > curr['ema20']) and ema_trend_long

        if ema_trend_long and (rsi_oversold or pullback_long or breakout_long):
            return 'long'

        # Short logic
        recent_lows = df['low'].iloc[max(0, last_idx-5):last_idx].min()
        breakout_short = curr['close'] < recent_lows
        pullback_short = (curr['high'] >= recent_ema20.min() * 0.99) and (curr['close'] < curr['ema20']) and ema_trend_short

        if ema_trend_short and (rsi_overbought or pullback_short or breakout_short):
            return 'short'

        return None

    def on_bar(self, bar: pd.Series):
        self.bars.append(bar)
        df = pd.DataFrame(self.bars)
        df = self.calculate_indicators(df)
        current_idx = len(df) - 1

        if not self.in_position:
            signal = self.determine_entry_signal(df, current_idx)
            if signal:
                self._open_trade(signal, bar['close'])
        else:
            signal = self.determine_entry_signal(df, current_idx)
            if signal and signal != self.position.side:
                self._close_trade(bar['open'])

    def _open_trade(self, side: str, price: float):
        self.in_position = True
        size = 100 # Fixed size per symbol
        trade = Trade(symbol=self.symbol, entry_price=price, entry_time=datetime.now(), side=side, size=size)
        self.position = trade
        self.trades.append(trade)
        print(f"[{self.symbol}] OPENED {side.upper()} @ {price:.2f}")

    def _close_trade(self, price: float):
        if self.position:
            self.position.close(price, datetime.now())
            print(f"[{self.symbol}] CLOSED {self.position.side.upper()} @ {price:.2f} | PnL: ${self.position.pnl:.2f}")
            self.in_position = False
            self.position = None

    def run_simulation(self, bars: pd.DataFrame):
        # Reset state
        self.bars = []
        self.trades = []
        self.in_position = False
        self.position = None
        
        for index, row in bars.iterrows():
            self.on_bar(row)
        
        # Force close at end
        if self.in_position and self.position:
            last_price = bars.iloc[-1]['close']
            self._close_trade(last_price)

    def get_report(self) -> Dict:
        completed = [t for t in self.trades if t.exit_price is not None]
        if not completed:
            return {'symbol': self.symbol, 'total_trades': 0, 'win_rate': 0.0, 'total_pnl': 0.0, 'trades': []}
        
        total_pnl = sum(t.pnl for t in completed)
        wins = [t for t in completed if t.pnl > 0]
        win_rate = (len(wins) / len(completed)) * 100
        
        return {
            'symbol': self.symbol,
            'total_trades': len(completed),
            'winning_trades': len(wins),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'trades': completed
        }

class MultiSymbolBot:
    def __init__(self, symbols: List[str], timeframe: TimeFrame):
        self.symbols = symbols
        self.timeframe = timeframe
        self.sessions: Dict[str, SymbolSession] = {s: SymbolSession(s, timeframe) for s in symbols}

    def run(self, days: int = 60):
        print(f"--- Starting Multi-Symbol Simulation for {len(self.symbols)} symbols ---")
        total_pnl = 0.0
        all_trades = []

        for symbol in self.symbols:
            session = self.sessions[symbol]
            try:
                print(f"\nFetching data for {symbol}...")
                bars = session.preload_historical_bars(days)
                print(f"Running simulation for {symbol}...")
                session.run_simulation(bars)
                
                report = session.get_report()
                total_pnl += report['total_pnl']
                all_trades.extend(report['trades'])
                
                print(f"Completed {symbol}: PnL ${report['total_pnl']:.2f} | Trades: {report['total_trades']}")
            except Exception as e:
                print(f"Error processing {symbol}: {e}")

        self.generate_master_report(all_trades, total_pnl)

    def generate_master_report(self, all_trades: List[Trade], total_pnl: float):
        print("\n" + "="*60)
        print("MASTER SIMULATION REPORT (All Symbols)")
        print("="*60)
        
        if not all_trades:
            print("No completed trades across all symbols.")
            return

        winning_trades = [t for t in all_trades if t.pnl > 0]
        total_trades = len(all_trades)
        win_rate = (len(winning_trades) / total_trades) * 100

        print(f"Total Symbols: {len(self.symbols)}")
        print(f"Total Trades: {total_trades}")
        print(f"Winning Trades: {len(winning_trades)}")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Total PnL: ${total_pnl:.2f}")

        print("\nDetailed Trade Log:")
        print(f"{'Symbol':<8} {'Time':<20} {'Side':<6} {'Entry':<10} {'Exit':<10} {'PnL':<10}")
        print("-" * 65)
        
        # Sort by time for cleaner log
        sorted_trades = sorted(all_trades, key=lambda x: x.entry_time)
        for t in sorted_trades:
            print(f"{t.symbol:<8} {t.entry_time.strftime('%Y-%m-%d %H:%M')} {t.side:<6} {t.entry_price:<10.2f} {t.exit_price:<10.2f} ${t.pnl:<10.2f}")
        
        print("="*60)

if __name__ == "__main__":
    bot = MultiSymbolBot(SYMBOLS, TIMEFRAME)
    bot.run(days=60)


