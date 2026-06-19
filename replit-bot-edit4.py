"""
Trading Bot using Alpaca API + TA-Lib
Modes: live | simulation
"""

import os
import time
import logging
import argparse
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import talib

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from config import ALPACA_KEY, ALPACA_SECRET
from alpaca.data.enums import DataFeed


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class BotConfig:
    # Alpaca credentials
    api_key: str = ALPACA_KEY
    api_secret: str = ALPACA_SECRET
    paper: bool = True           # True = paper trading endpoint

    # Universe
    symbols: list[str] = field(default_factory=lambda: ["AAPL", "NVDA", "SPY"])

    # Timeframe for bars
    timeframe: TimeFrame = TimeFrame(1, TimeFrameUnit.Hour)
    lookback_bars: int = 200      # how many bars to fetch each cycle

    # Indicator periods
    ema_fast: int = 9
    ema_slow: int = 21
    ema_trend: int = 50           # long-term trend filter EMA
    rsi_period: int = 14
    atr_period: int = 14
    momentum_period: int = 10
    breakout_lookback: int = 20   # bars used for breakout high/low

    # Entry thresholds
    rsi_oversold: float = 30.0    # tighter than before (was 35)
    rsi_overbought: float = 70.0  # tighter than before (was 65)
    momentum_threshold: float = 0.5  # MOM must exceed this magnitude to count

    # Confluence: how many conditions must fire simultaneously to enter
    min_signals: int = 2

    # Risk management
    stop_loss_atr_mult: float = 2.0   # stop = entry - (ATR * mult)
    take_profit_atr_mult: float = 3.0 # TP   = entry + (ATR * mult)
    risk_per_trade_pct: float = 1.0   # % of equity risked per trade
    max_open_positions: int = 5

    # Simulation
    sim_initial_cash: float = 100_000.0

    # Poll interval (seconds) — ignored in simulation
    poll_interval: int = 60


# ---------------------------------------------------------------------------
# Position tracking (used in simulation mode)
# ---------------------------------------------------------------------------

@dataclass
class SimPosition:
    symbol: str
    qty: float
    entry_price: float
    stop_loss: float
    take_profit: float
    side: str = "long"   # "long" | "short"


@dataclass
class SimState:
    cash: float
    positions: dict[str, SimPosition] = field(default_factory=dict)
    trade_log: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Market data helpers
# ---------------------------------------------------------------------------

def fetch_historical_bars(
    data_client: StockHistoricalDataClient,
    symbol: str,
    timeframe: TimeFrame,
    limit: int,
) -> Optional[dict[str, np.ndarray]]:
    """
    Fetch historical OHLCV bars for *symbol* and return a dict of
    numpy arrays keyed by 'open', 'high', 'low', 'close', 'volume'.

    Returns None if data is unavailable or too short.
    """
    end = datetime.now(tz=timezone.utc)
    # Fetch extra buffer to ensure we have enough after market-hour gaps
    start = end - timedelta(days=limit * 2)

    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=timeframe,
        start=start,
        end=end,
        feed=DataFeed.IEX,
        limit=limit,
        adjustment="all",
    )

    try:
        bars_response = data_client.get_stock_bars(request)
        bars = bars_response[symbol]
    except Exception as exc:
        logger.warning("fetch_historical_bars(%s): %s", symbol, exc)
        return None

    if not bars or len(bars) < 50:
        logger.warning("fetch_historical_bars(%s): insufficient data (%d bars)", symbol, len(bars) if bars else 0)
        return None

    data = {
        "open":   np.array([b.open   for b in bars], dtype=np.float64),
        "high":   np.array([b.high   for b in bars], dtype=np.float64),
        "low":    np.array([b.low    for b in bars], dtype=np.float64),
        "close":  np.array([b.close  for b in bars], dtype=np.float64),
        "volume": np.array([b.volume for b in bars], dtype=np.float64),
    }
    logger.debug("fetch_historical_bars(%s): fetched %d bars", symbol, len(data["close"]))
    return data


# ---------------------------------------------------------------------------
# Indicator computation
# ---------------------------------------------------------------------------

@dataclass
class Indicators:
    ema_fast: float
    ema_slow: float
    trend_ema: float          # long-term trend filter (e.g. 50-period)
    rsi: float
    momentum: float
    atr: float
    open: float               # current bar open — used to anchor stop loss
    close: float
    high_breakout: float   # highest high over breakout_lookback bars
    low_breakout: float    # lowest low  over breakout_lookback bars
    prev_ema_fast: float
    prev_ema_slow: float


def compute_indicators(data: dict[str, np.ndarray], cfg: BotConfig) -> Optional[Indicators]:
    """Compute all TA-Lib indicators needed for entry/exit decisions."""
    close  = data["close"]
    high   = data["high"]
    low    = data["low"]

    min_bars = max(cfg.ema_trend, cfg.ema_slow, cfg.rsi_period, cfg.atr_period, cfg.momentum_period, cfg.breakout_lookback) + 5
    if len(close) < min_bars:
        return None

    ema_fast_arr  = talib.EMA(close, timeperiod=cfg.ema_fast)
    ema_slow_arr  = talib.EMA(close, timeperiod=cfg.ema_slow)
    ema_trend_arr = talib.EMA(close, timeperiod=cfg.ema_trend)
    rsi_arr       = talib.RSI(close, timeperiod=cfg.rsi_period)
    mom_arr       = talib.MOM(close, timeperiod=cfg.momentum_period)
    atr_arr       = talib.ATR(high, low, close, timeperiod=cfg.atr_period)

    # Use -1 for current bar, -2 for previous bar
    if any(np.isnan(v) for v in [
        ema_fast_arr[-1], ema_fast_arr[-2],
        ema_slow_arr[-1], ema_slow_arr[-2],
        ema_trend_arr[-1],
        rsi_arr[-1], mom_arr[-1], atr_arr[-1],
    ]):
        return None

    breakout_window = cfg.breakout_lookback + 1  # +1 to exclude current bar
    high_breakout = float(np.max(high[-breakout_window:-1]))
    low_breakout  = float(np.min(low[-breakout_window:-1]))

    return Indicators(
        ema_fast=float(ema_fast_arr[-1]),
        ema_slow=float(ema_slow_arr[-1]),
        trend_ema=float(ema_trend_arr[-1]),
        rsi=float(rsi_arr[-1]),
        momentum=float(mom_arr[-1]),
        atr=float(atr_arr[-1]),
        open=float(data["open"][-1]),
        close=float(close[-1]),
        high_breakout=high_breakout,
        low_breakout=low_breakout,
        prev_ema_fast=float(ema_fast_arr[-2]),
        prev_ema_slow=float(ema_slow_arr[-2]),
    )


# ---------------------------------------------------------------------------
# Entry logic
# ---------------------------------------------------------------------------

@dataclass
class EntrySignal:
    direction: str          # "long" | "short"
    reason: str
    stop_loss: float
    take_profit: float


def check_entry(ind: Indicators, cfg: BotConfig) -> Optional[EntrySignal]:
    """
    Confluence-based entry: score each condition independently, then only
    enter when at least cfg.min_signals conditions fire simultaneously AND
    the 50-period trend EMA agrees with the direction.

    Scored conditions (each worth 1 point):
        1. EMA crossover   — fast EMA freshly crosses slow EMA this bar
        2. RSI extreme     — RSI below oversold (long) or above overbought (short)
        3. Breakout        — close beyond the N-bar high/low with momentum
        4. Momentum        — MOM magnitude exceeds threshold with price on right side of slow EMA

    Hard filters (must pass regardless of score):
        - Trend EMA: close > trend_ema for longs, close < trend_ema for shorts
        - EMA alignment: fast EMA above slow EMA for longs, below for shorts
          (crossover condition is exempt from the EMA alignment filter since it
           represents the moment of the cross)
    """
    price = ind.close
    atr   = ind.atr

    # ---- LONG scoring -------------------------------------------------------
    in_long_trend = price > ind.trend_ema                        # hard trend filter
    ema_aligned_long = ind.ema_fast > ind.ema_slow

    if in_long_trend:
        long_conditions: list[str] = []

        # 1. EMA crossover long (the cross itself is the trend-alignment event)
        if (ind.prev_ema_fast <= ind.prev_ema_slow) and (ind.ema_fast > ind.ema_slow):
            long_conditions.append("ema_cross")

        # 2. RSI extreme long
        if ind.rsi < cfg.rsi_oversold and ema_aligned_long:
            long_conditions.append("rsi_oversold")

        # 3. Breakout long
        if (price > ind.high_breakout
                and ind.momentum > cfg.momentum_threshold
                and ema_aligned_long):
            long_conditions.append("breakout")

        # 4. Momentum long
        if (ind.momentum > cfg.momentum_threshold
                and price > ind.ema_fast
                and ind.rsi < cfg.rsi_overbought
                and ema_aligned_long):
            long_conditions.append("momentum")

        if len(long_conditions) >= cfg.min_signals:
            reason = "long[" + "+".join(long_conditions) + "]"
            # Anchor stop to the top of the bar body (max of open/close) so the
            # stop sits as high as possible — minimising the loss if hit.
            sl_anchor = max(ind.open, ind.close)
            sl = sl_anchor - atr * cfg.stop_loss_atr_mult
            tp = price + atr * cfg.take_profit_atr_mult
            return EntrySignal("long", reason, sl, tp)

    # ---- SHORT scoring ------------------------------------------------------
    in_short_trend = price < ind.trend_ema                       # hard trend filter
    ema_aligned_short = ind.ema_fast < ind.ema_slow

    if in_short_trend:
        short_conditions: list[str] = []

        # 1. EMA crossover short
        if (ind.prev_ema_fast >= ind.prev_ema_slow) and (ind.ema_fast < ind.ema_slow):
            short_conditions.append("ema_cross")

        # 2. RSI extreme short
        if ind.rsi > cfg.rsi_overbought and ema_aligned_short:
            short_conditions.append("rsi_overbought")

        # 3. Breakout short
        if (price < ind.low_breakout
                and ind.momentum < -cfg.momentum_threshold
                and ema_aligned_short):
            short_conditions.append("breakout")

        # 4. Momentum short
        if (ind.momentum < -cfg.momentum_threshold
                and price < ind.ema_fast
                and ind.rsi > cfg.rsi_oversold
                and ema_aligned_short):
            short_conditions.append("momentum")

        if len(short_conditions) >= cfg.min_signals:
            reason = "short[" + "+".join(short_conditions) + "]"
            # Anchor stop to the bottom of the bar body (min of open/close) so the
            # stop sits as low as possible — minimising the loss if hit.
            sl_anchor = min(ind.open, ind.close)
            sl = sl_anchor + atr * cfg.stop_loss_atr_mult
            tp = price - atr * cfg.take_profit_atr_mult
            return EntrySignal("short", reason, sl, tp)

    return None


# ---------------------------------------------------------------------------
# Exit logic
# ---------------------------------------------------------------------------

@dataclass
class ExitSignal:
    reason: str    # "stop_loss" | "take_profit" | "signal_reversal"


def check_exit(
    current_price: float,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    direction: str,
    ind: Indicators,
    cfg: BotConfig,
) -> Optional[ExitSignal]:
    """
    Evaluate exit conditions for an open position.

    Checks:
        - Stop loss  : price moved against by stop_loss_atr_mult * ATR
        - Take profit: price moved in favour by take_profit_atr_mult * ATR
        - Signal reversal: EMA cross in opposite direction
    """
    if direction == "long":
        if current_price <= stop_loss:
            return ExitSignal("stop_loss")
        if current_price >= take_profit:
            return ExitSignal("take_profit")
        # Signal reversal — fast EMA crossed below slow EMA
        if ind.prev_ema_fast >= ind.prev_ema_slow and ind.ema_fast < ind.ema_slow:
            return ExitSignal("signal_reversal")

    elif direction == "short":
        if current_price >= stop_loss:
            return ExitSignal("stop_loss")
        if current_price <= take_profit:
            return ExitSignal("take_profit")
        # Signal reversal — fast EMA crossed above slow EMA
        if ind.prev_ema_fast <= ind.prev_ema_slow and ind.ema_fast > ind.ema_slow:
            return ExitSignal("signal_reversal")

    return None


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------

def compute_qty(equity: float, entry_price: float, stop_loss: float, cfg: BotConfig) -> float:
    """Risk-based position sizing. Returns number of whole shares."""
    risk_amount = equity * (cfg.risk_per_trade_pct / 100.0)
    risk_per_share = abs(entry_price - stop_loss)
    if risk_per_share < 1e-8:
        return 0.0
    qty = risk_amount / risk_per_share
    return max(1.0, round(qty))


# ---------------------------------------------------------------------------
# Live trading executor
# ---------------------------------------------------------------------------

class LiveTrader:
    def __init__(self, cfg: BotConfig):
        self.cfg = cfg
        self.trading_client = TradingClient(cfg.api_key, cfg.api_secret, paper=cfg.paper)
        self.data_client = StockHistoricalDataClient(cfg.api_key, cfg.api_secret)

    def get_equity(self) -> float:
        account = self.trading_client.get_account()
        return float(account.equity)

    def get_open_positions(self) -> dict[str, object]:
        positions = self.trading_client.get_all_positions()
        return {p.symbol: p for p in positions}

    def place_market_order(self, symbol: str, qty: float, side: OrderSide) -> None:
        req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY,
        )
        order = self.trading_client.submit_order(req)
        logger.info("ORDER PLACED | %s | %s | qty=%s | id=%s", symbol, side, qty, order.id)

    def close_position(self, symbol: str) -> None:
        try:
            self.trading_client.close_position(symbol)
            logger.info("POSITION CLOSED | %s", symbol)
        except Exception as exc:
            logger.warning("close_position(%s): %s", symbol, exc)

    def run(self) -> None:
        logger.info("=== LIVE TRADER STARTED (paper=%s) ===", self.cfg.paper)
        cfg = self.cfg

        # In-memory stop/TP tracking (broker brackets are not used here for simplicity)
        position_meta: dict[str, dict] = {}

        while True:
            try:
                equity = self.get_equity()
                open_positions = self.get_open_positions()

                for symbol in cfg.symbols:
                    data = fetch_historical_bars(self.data_client, symbol, cfg.timeframe, cfg.lookback_bars)
                    if data is None:
                        continue

                    ind = compute_indicators(data, cfg)
                    if ind is None:
                        continue

                    current_price = ind.close

                    # --- Check exit for existing positions ---
                    if symbol in open_positions and symbol in position_meta:
                        meta = position_meta[symbol]
                        exit_sig = check_exit(
                            current_price=current_price,
                            entry_price=meta["entry_price"],
                            stop_loss=meta["stop_loss"],
                            take_profit=meta["take_profit"],
                            direction=meta["direction"],
                            ind=ind,
                            cfg=cfg,
                        )
                        if exit_sig:
                            logger.info("EXIT SIGNAL | %s | reason=%s | price=%.4f", symbol, exit_sig.reason, current_price)
                            self.close_position(symbol)
                            del position_meta[symbol]
                        continue   # don't look for entry while in position

                    # --- Check entry ---
                    if symbol not in open_positions and len(open_positions) < cfg.max_open_positions:
                        entry_sig = check_entry(ind, cfg)
                        if entry_sig:
                            qty = compute_qty(equity, current_price, entry_sig.stop_loss, cfg)
                            if qty <= 0:
                                continue
                            side = OrderSide.BUY if entry_sig.direction == "long" else OrderSide.SELL
                            logger.info(
                                "ENTRY SIGNAL | %s | %s | reason=%s | price=%.4f | sl=%.4f | tp=%.4f | qty=%d",
                                symbol, entry_sig.direction, entry_sig.reason,
                                current_price, entry_sig.stop_loss, entry_sig.take_profit, int(qty),
                            )
                            self.place_market_order(symbol, qty, side)
                            position_meta[symbol] = {
                                "entry_price": current_price,
                                "stop_loss": entry_sig.stop_loss,
                                "take_profit": entry_sig.take_profit,
                                "direction": entry_sig.direction,
                            }
                            # Refresh open positions count
                            open_positions = self.get_open_positions()

            except KeyboardInterrupt:
                logger.info("Interrupted by user — shutting down.")
                break
            except Exception as exc:
                logger.exception("Unhandled error in main loop: %s", exc)

            logger.debug("Sleeping %ds ...", cfg.poll_interval)
            time.sleep(cfg.poll_interval)


# ---------------------------------------------------------------------------
# Simulation executor
# ---------------------------------------------------------------------------

class SimulationTrader:
    """
    Runs bar-by-bar simulation over fetched historical data.
    No real orders are placed.
    """

    def __init__(self, cfg: BotConfig):
        self.cfg = cfg
        self.data_client = StockHistoricalDataClient(cfg.api_key, cfg.api_secret)
        self.state = SimState(cash=cfg.sim_initial_cash)

    def _record_trade(
        self,
        action: str,
        symbol: str,
        qty: float,
        price: float,
        reason: str,
        pnl: float = 0.0,
        entry_price: float = 0.0,
    ) -> None:
        self.state.trade_log.append({
            "action": action,
            "symbol": symbol,
            "qty": qty,
            "price": price,          # exit price for exits, entry price for entries
            "entry_price": entry_price,
            "reason": reason,
            "pnl": pnl,
        })

    def _equity(self) -> float:
        """Cash + mark-to-market value of all open positions (using last known price)."""
        pos_value = sum(
            p.qty * p.entry_price for p in self.state.positions.values()
        )
        return self.state.cash + pos_value

    def run(self) -> None:
        cfg = self.cfg
        logger.info("=== SIMULATION STARTED | initial_cash=%.2f ===", cfg.sim_initial_cash)

        # Fetch full history for each symbol
        all_data: dict[str, dict[str, np.ndarray]] = {}
        for symbol in cfg.symbols:
            data = fetch_historical_bars(self.data_client, symbol, cfg.timeframe, cfg.lookback_bars)
            if data is None:
                logger.warning("SIM: no data for %s — skipping", symbol)
            else:
                all_data[symbol] = data
                logger.info("SIM: loaded %d bars for %s", len(data["close"]), symbol)

        if not all_data:
            logger.error("SIM: no data loaded — cannot run simulation.")
            return

        # Determine minimum bars needed before we can start computing indicators
        min_bars = max(cfg.ema_trend, cfg.ema_slow, cfg.rsi_period, cfg.atr_period, cfg.momentum_period, cfg.breakout_lookback) + 10

        # Find the max number of bars across symbols to iterate over
        n_bars = max(len(d["close"]) for d in all_data.values())
        logger.info("SIM: iterating over up to %d bars ...", n_bars)

        for i in range(min_bars, n_bars):
            for symbol, full_data in all_data.items():
                if i >= len(full_data["close"]):
                    continue  # this symbol ran out of data

                # Slice data up to bar i (simulating "now")
                data_slice: dict[str, np.ndarray] = {k: v[:i+1] for k, v in full_data.items()}
                ind = compute_indicators(data_slice, cfg)
                if ind is None:
                    continue

                current_price = ind.close

                # --- Check exit ---
                if symbol in self.state.positions:
                    pos = self.state.positions[symbol]
                    exit_sig = check_exit(
                        current_price=current_price,
                        entry_price=pos.entry_price,
                        stop_loss=pos.stop_loss,
                        take_profit=pos.take_profit,
                        direction=pos.side,
                        ind=ind,
                        cfg=cfg,
                    )
                    if exit_sig:
                        if pos.side == "long":
                            pnl = (current_price - pos.entry_price) * pos.qty
                        else:
                            pnl = (pos.entry_price - current_price) * pos.qty
                        self.state.cash += pos.entry_price * pos.qty + pnl
                        logger.info(
                            "SIM EXIT  | bar=%d | %s | %s | price=%.4f | pnl=%.2f | reason=%s",
                            i, symbol, pos.side, current_price, pnl, exit_sig.reason,
                        )
                        self._record_trade("exit", symbol, pos.qty, current_price, exit_sig.reason, pnl, entry_price=pos.entry_price)
                        del self.state.positions[symbol]
                    continue  # don't check entry while in position

                # --- Check entry ---
                if len(self.state.positions) >= cfg.max_open_positions:
                    continue

                entry_sig = check_entry(ind, cfg)
                if entry_sig:
                    equity = self._equity()
                    qty = compute_qty(equity, current_price, entry_sig.stop_loss, cfg)
                    cost = current_price * qty
                    if qty <= 0 or cost > self.state.cash:
                        continue
                    self.state.cash -= cost
                    self.state.positions[symbol] = SimPosition(
                        symbol=symbol,
                        qty=qty,
                        entry_price=current_price,
                        stop_loss=entry_sig.stop_loss,
                        take_profit=entry_sig.take_profit,
                        side=entry_sig.direction,
                    )
                    logger.info(
                        "SIM ENTRY | bar=%d | %s | %s | reason=%s | price=%.4f | sl=%.4f | tp=%.4f | qty=%d",
                        i, symbol, entry_sig.direction, entry_sig.reason,
                        current_price, entry_sig.stop_loss, entry_sig.take_profit, int(qty),
                    )
                    self._record_trade("entry", symbol, qty, current_price, entry_sig.reason)

        # Close any remaining positions at last price
        for symbol, pos in list(self.state.positions.items()):
            last_price = all_data[symbol]["close"][-1]
            pnl = (last_price - pos.entry_price) * pos.qty if pos.side == "long" else (pos.entry_price - last_price) * pos.qty
            self.state.cash += pos.entry_price * pos.qty + pnl
            logger.info("SIM FORCE-CLOSE | %s | price=%.4f | pnl=%.2f", symbol, last_price, pnl)
            self._record_trade("force_close", symbol, pos.qty, last_price, "end_of_simulation", pnl, entry_price=pos.entry_price)
        self.state.positions.clear()

        self._print_summary()

    def _print_summary(self) -> None:
        closed = [t for t in self.state.trade_log if t["action"] in ("exit", "force_close")]
        wins   = [t for t in closed if t["pnl"] > 0]
        losses = [t for t in closed if t["pnl"] <= 0]
        total_pnl   = sum(t["pnl"] for t in closed)
        final_equity = self.state.cash

        # ---- Per-trade table ------------------------------------------------
        COL_W = {"symbol": 10, "entry": 10, "exit": 10, "qty": 7, "pnl": 13, "result": 10}
        sep = "-" * 80

        def row(symbol: str, entry: str, exit_: str, qty: str, pnl: str, result: str) -> str:
            return (
                f"{symbol:<{COL_W['symbol']}}"
                f"{entry:<{COL_W['entry']}}"
                f"{exit_:<{COL_W['exit']}}"
                f"{qty:<{COL_W['qty']}}"
                f"{pnl:<{COL_W['pnl']}}"
                f"{result:<{COL_W['result']}}"
            )

        header = row("SYMBOL", "ENTRY", "EXIT", "QTY", "P&L", "RESULT")

        print("")  # blank line before table
        print(header)
        print(sep)

        for t in closed:
            entry_px = t.get("entry_price", 0.0)
            exit_px  = t["price"]
            pnl_val  = t["pnl"]
            result   = "WIN" if pnl_val > 0 else "LOSS"
            print(row(
                t["symbol"],
                f"{entry_px:.2f}",
                f"{exit_px:.2f}",
                str(int(t["qty"])),
                f"{pnl_val:+.2f}",
                result,
            ))

        print(sep)
        print("")

        # ---- Aggregate summary ----------------------------------------------
        logger.info("=" * 60)
        logger.info("SIMULATION SUMMARY")
        logger.info("  Initial cash  : %.2f", self.cfg.sim_initial_cash)
        logger.info("  Final equity  : %.2f", final_equity)
        logger.info("  Total P&L     : %.2f (%.2f%%)", total_pnl, 100 * total_pnl / self.cfg.sim_initial_cash)
        logger.info("  Closed trades : %d", len(closed))
        logger.info("  Wins          : %d", len(wins))
        logger.info("  Losses        : %d", len(losses))
        if closed:
            win_rate = 100.0 * len(wins) / len(closed)
            avg_win  = sum(t["pnl"] for t in wins)  / max(1, len(wins))
            avg_loss = sum(t["pnl"] for t in losses) / max(1, len(losses))
            logger.info("  Win rate      : %.1f%%", win_rate)
            logger.info("  Avg win       : %.2f", avg_win)
            logger.info("  Avg loss      : %.2f", avg_loss)
        logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_config_from_env() -> BotConfig:
    """Pull credentials from environment variables."""
    return BotConfig(
        api_key=ALPACA_KEY,
        api_secret=ALPACA_SECRET,
        paper=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Alpaca Trading Bot")
    parser.add_argument(
        "--mode",
        choices=["live", "simulation"],
        default="simulation",
        help="Operation mode (default: simulation)",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["AAPL", "NVDA", "SPY"],
        help="Ticker symbols to trade",
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        default=True,
        help="Use Alpaca paper trading endpoint (default: True)",
    )
    parser.add_argument(
        "--live-paper",
        dest="live_paper",
        action="store_false",
        help="Use real money endpoint in live mode (overrides --paper)",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=200,
        help="Number of historical bars to fetch per cycle",
    )
    parser.add_argument(
        "--poll",
        type=int,
        default=60,
        help="Poll interval in seconds (live mode only)",
    )
    parser.add_argument(
        "--initial-cash",
        type=float,
        default=100_000.0,
        help="Starting cash for simulation mode",
    )
    parser.add_argument(
        "--risk",
        type=float,
        default=1.0,
        help="Risk per trade as %% of equity (default: 1.0)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG logging",
    )
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    cfg = build_config_from_env()
    # Support both space-separated and comma-separated symbol lists:
    #   --symbols AAPL NVDA SPY   (argparse nargs="+")
    #   --symbols AAPL,NVDA,SPY   (single comma-joined string)
    raw_symbols: list[str] = []
    for token in args.symbols:
        raw_symbols.extend(token.split(","))
    cfg.symbols = [s.strip().upper() for s in raw_symbols if s.strip()]
    cfg.paper = args.paper if args.mode == "live" else True
    cfg.lookback_bars = args.lookback
    cfg.poll_interval = args.poll
    cfg.sim_initial_cash = args.initial_cash
    cfg.risk_per_trade_pct = args.risk

    if not cfg.api_key or not cfg.api_secret:
        logger.error(
            "Missing Alpaca credentials. "
            "Set ALPACA_API_KEY and ALPACA_API_SECRET environment variables."
        )
        raise SystemExit(1)

    if args.mode == "live":
        if not cfg.paper:
            logger.warning("!!! LIVE MODE WITH REAL MONEY — proceeding with live endpoint !!!")
        trader = LiveTrader(cfg)
        trader.run()
    else:
        trader = SimulationTrader(cfg)
        trader.run()


if __name__ == "__main__":
    main()
