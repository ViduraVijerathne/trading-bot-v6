"""
Binance Futures Trading Bot with Streamlit Dashboard
=====================================================
A production-ready, single-file crypto trading bot supporting:
- 4 technical analysis strategies (Trend Following, Mean Reversion, Momentum, Breakout)
- 3 operational modes (backtrack, test, real)
- Real-time WebSocket data streaming
- Interactive Streamlit dashboard with live controls
- Thread-safe state management
- Risk management with SL/TP and position sizing
"""

# =============================================================================
# SECTION 1 - IMPORTS
# =============================================================================
import asyncio
import threading
import queue
import time
import json
import logging
import math
import uuid
from datetime import datetime, timedelta
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Callable

import streamlit as st
import pandas as pd
import numpy as np
import websockets
from streamlit_autorefresh import st_autorefresh

# =============================================================================
# SECTION 2 - CONSTANTS & CONFIG
# =============================================================================
DEFAULT_CAPITAL = 20.0
DEFAULT_LEVERAGE = 3
MAX_CONCURRENT_TRADES = 2
RISK_REWARD_RATIO = 2.0
STOP_LOSS_PCT = 0.01
TAKE_PROFIT_PCT = 0.02

SCAN_SYMBOLS = [
    'BTC/USDT:USDT',
    'ETH/USDT:USDT',
    'BNB/USDT:USDT',
    'SOL/USDT:USDT',
    'XRP/USDT:USDT',
    'DOGE/USDT:USDT',
    'ADA/USDT:USDT',
    'AVAX/USDT:USDT',
    'LINK/USDT:USDT',
    'MATIC/USDT:USDT',
    'DOT/USDT:USDT',
    'UNI/USDT:USDT',
]

AVAILABLE_SYMBOLS = [
    'BTC/USDT:USDT',
    'ETH/USDT:USDT',
    'BNB/USDT:USDT',
    'SOL/USDT:USDT',
    'XRP/USDT:USDT',
    'DOGE/USDT:USDT',
    'ADA/USDT:USDT',
    'AVAX/USDT:USDT',
    'LINK/USDT:USDT',
    'MATIC/USDT:USDT',
    'DOT/USDT:USDT',
    'UNI/USDT:USDT',
    'NEAR/USDT:USDT',
    'APT/USDT:USDT',
    'ARB/USDT:USDT',
    'OP/USDT:USDT',
    'FIL/USDT:USDT',
    'LTC/USDT:USDT',
    'ATOM/USDT:USDT',
    'FTM/USDT:USDT',
]

DEFAULT_SELECTED_SYMBOLS = [
    'BTC/USDT:USDT',
    'ETH/USDT:USDT',
]

TIMEFRAME = '5m'
CANDLE_LIMIT = 200
WS_BASE_URL = 'wss://fstream.binance.com/stream'

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# SECTION 3 - DATA MODELS
# =============================================================================
@dataclass
class TradeRecord:
    id: str
    symbol: str
    strategy: str
    side: str
    entry_price: float
    exit_price: float
    sl_price: float
    tp_price: float
    quantity: float
    pnl: float
    status: str
    timestamp: datetime
    closed_at: Optional[datetime] = None


@dataclass
class PositionState:
    id: str
    symbol: str
    strategy: str
    side: str
    entry_price: float
    sl_price: float
    tp_price: float
    quantity: float
    timestamp: datetime


# =============================================================================
# SECTION 4 - INDICATOR UTILITIES
# =============================================================================
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical indicators on an OHLCV DataFrame.
    Expects columns: open, high, low, close, volume.
    Returns DataFrame with added indicator columns.
    Uses only pandas and numpy (no pandas_ta dependency).
    """
    df = df.copy()

    # EMA 20 and EMA 50
    df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()

    # RSI 14
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss
    df['RSI_14'] = 100.0 - (100.0 / (1.0 + rs))

    # Bollinger Bands (20, 2)
    bb_mid = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std(ddof=0)
    df['BBL_20_2.0'] = bb_mid - 2.0 * bb_std
    df['BBM_20_2.0'] = bb_mid
    df['BBU_20_2.0'] = bb_mid + 2.0 * bb_std

    # MACD (12, 26, 9)
    ema_fast = df['close'].ewm(span=12, adjust=False).mean()
    ema_slow = df['close'].ewm(span=26, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - macd_signal
    df['MACD_12_26_9'] = macd_line
    df['MACDs_12_26_9'] = macd_signal
    df['MACDh_12_26_9'] = macd_hist

    # ATR 14
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift(1)).abs()
    low_close = (df['low'] - df['close'].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATRr_14'] = true_range.ewm(alpha=1.0 / 14, min_periods=14, adjust=False).mean()

    # Volume SMA 20
    df['volume_sma20'] = df['volume'].rolling(20).mean()

    return df


# =============================================================================
# SECTION 5 - STRATEGY ENGINE
# =============================================================================
def strategy_trend_following(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Strategy A: Trend Following using EMA 20/50 Crossover + RSI confirmation.
    Returns signal dict or None.
    """
    if len(df) < 55:
        return None

    last = df.iloc[-1]
    prev_rows = df.iloc[-4:-1]

    ema20_col = 'EMA_20'
    ema50_col = 'EMA_50'
    rsi_col = 'RSI_14'

    if ema20_col not in df.columns or ema50_col not in df.columns or rsi_col not in df.columns:
        return None

    ema20_now = last.get(ema20_col)
    ema50_now = last.get(ema50_col)
    rsi_now = last.get(rsi_col)

    if pd.isna(ema20_now) or pd.isna(ema50_now) or pd.isna(rsi_now):
        return None

    # Check for EMA crossover in last 3 candles
    ema_crossed_up = False
    ema_crossed_down = False
    for i in range(len(prev_rows)):
        row = prev_rows.iloc[i]
        prev_ema20 = row.get(ema20_col)
        prev_ema50 = row.get(ema50_col)
        if pd.isna(prev_ema20) or pd.isna(prev_ema50):
            continue
        if i > 0:
            prev_prev = prev_rows.iloc[i - 1]
            pp_ema20 = prev_prev.get(ema20_col)
            pp_ema50 = prev_prev.get(ema50_col)
            if not pd.isna(pp_ema20) and not pd.isna(pp_ema50):
                if pp_ema20 <= pp_ema50 and prev_ema20 > prev_ema50:
                    ema_crossed_up = True
                if pp_ema20 >= pp_ema50 and prev_ema20 < prev_ema50:
                    ema_crossed_down = True

    # Also check last candle vs previous
    if len(prev_rows) > 0:
        last_prev = prev_rows.iloc[-1]
        lp_ema20 = last_prev.get(ema20_col)
        lp_ema50 = last_prev.get(ema50_col)
        if not pd.isna(lp_ema20) and not pd.isna(lp_ema50):
            if lp_ema20 <= lp_ema50 and ema20_now > ema50_now:
                ema_crossed_up = True
            if lp_ema20 >= lp_ema50 and ema20_now < ema50_now:
                ema_crossed_down = True

    # LONG: EMA20 > EMA50 AND RSI between 45-60
    if ema20_now > ema50_now and 45 <= rsi_now <= 60 and ema_crossed_up:
        return {
            'side': 'LONG',
            'strategy': 'Trend Following',
        }

    # SHORT: EMA20 < EMA50 AND RSI between 40-55
    if ema20_now < ema50_now and 40 <= rsi_now <= 55 and ema_crossed_down:
        return {
            'side': 'SHORT',
            'strategy': 'Trend Following',
        }

    return None


def strategy_mean_reversion(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Strategy B: Mean Reversion using Bollinger Bands + RSI.
    Returns signal dict or None.
    """
    if len(df) < 25:
        return None

    last = df.iloc[-1]

    bbl_col = 'BBL_20_2.0'
    bbu_col = 'BBU_20_2.0'
    rsi_col = 'RSI_14'

    if bbl_col not in df.columns or bbu_col not in df.columns or rsi_col not in df.columns:
        return None

    close = last.get('close')
    bbl = last.get(bbl_col)
    bbu = last.get(bbu_col)
    rsi = last.get(rsi_col)

    if pd.isna(close) or pd.isna(bbl) or pd.isna(bbu) or pd.isna(rsi):
        return None

    # LONG: close <= lower Bollinger Band AND RSI < 35
    if close <= bbl and rsi < 35:
        return {
            'side': 'LONG',
            'strategy': 'Mean Reversion',
        }

    # SHORT: close >= upper Bollinger Band AND RSI > 65
    if close >= bbu and rsi > 65:
        return {
            'side': 'SHORT',
            'strategy': 'Mean Reversion',
        }

    return None


def strategy_momentum(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Strategy C: Momentum using MACD Crossover + Volume Spike.
    Returns signal dict or None.
    """
    if len(df) < 30:
        return None

    macd_col = 'MACD_12_26_9'
    signal_col = 'MACDs_12_26_9'
    vol_sma_col = 'volume_sma20'

    if macd_col not in df.columns or signal_col not in df.columns or vol_sma_col not in df.columns:
        return None

    last = df.iloc[-1]
    macd_now = last.get(macd_col)
    signal_now = last.get(signal_col)
    volume_now = last.get('volume')
    vol_sma = last.get(vol_sma_col)

    if pd.isna(macd_now) or pd.isna(signal_now) or pd.isna(volume_now) or pd.isna(vol_sma):
        return None

    # Check MACD crossover in last 2 candles
    macd_crossed_up = False
    macd_crossed_down = False

    if len(df) >= 3:
        for i in range(-2, 0):
            curr = df.iloc[i]
            prev = df.iloc[i - 1]
            curr_macd = curr.get(macd_col)
            curr_signal = curr.get(signal_col)
            prev_macd = prev.get(macd_col)
            prev_signal = prev.get(signal_col)
            if pd.isna(curr_macd) or pd.isna(curr_signal) or pd.isna(prev_macd) or pd.isna(prev_signal):
                continue
            if prev_macd <= prev_signal and curr_macd > curr_signal:
                macd_crossed_up = True
            if prev_macd >= prev_signal and curr_macd < curr_signal:
                macd_crossed_down = True

    # Also check the latest candle
    if len(df) >= 2:
        prev = df.iloc[-2]
        prev_macd = prev.get(macd_col)
        prev_signal = prev.get(signal_col)
        if not pd.isna(prev_macd) and not pd.isna(prev_signal):
            if prev_macd <= prev_signal and macd_now > signal_now:
                macd_crossed_up = True
            if prev_macd >= prev_signal and macd_now < signal_now:
                macd_crossed_down = True

    volume_spike = volume_now > 1.5 * vol_sma if vol_sma > 0 else False

    # LONG: MACD crosses above signal AND volume spike
    if macd_crossed_up and volume_spike:
        return {
            'side': 'LONG',
            'strategy': 'Momentum',
        }

    # SHORT: MACD crosses below signal AND volume spike
    if macd_crossed_down and volume_spike:
        return {
            'side': 'SHORT',
            'strategy': 'Momentum',
        }

    return None


def strategy_breakout(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Strategy D: Breakout based on recent 24H High/Low (96 candles of 5m = 8H lookback).
    Uses last 96 candles as lookback window.
    Returns signal dict or None.
    """
    if len(df) < 97:
        return None

    last = df.iloc[-1]
    lookback = df.iloc[-97:-1]
    vol_sma_col = 'volume_sma20'

    close = last.get('close')
    volume_now = last.get('volume')
    vol_sma = last.get(vol_sma_col)

    if pd.isna(close) or pd.isna(volume_now) or pd.isna(vol_sma):
        return None

    high_max = lookback['high'].max()
    low_min = lookback['low'].min()

    if pd.isna(high_max) or pd.isna(low_min):
        return None

    volume_confirm = volume_now > 1.3 * vol_sma if vol_sma > 0 else False

    # LONG: close breaks above recent high with volume confirmation
    if close > high_max and volume_confirm:
        return {
            'side': 'LONG',
            'strategy': 'Breakout',
        }

    # SHORT: close breaks below recent low with volume confirmation
    if close < low_min and volume_confirm:
        return {
            'side': 'SHORT',
            'strategy': 'Breakout',
        }

    return None


# =============================================================================
# SECTION 6 - RISK MANAGER
# =============================================================================
class RiskManager:
    """Handles position sizing, SL/TP computation, and trade validation."""

    def compute_position_size(
        self,
        capital: float,
        leverage: int,
        entry_price: float,
        sl_price: float,
        risk_pct: float = 0.02
    ) -> float:
        """
        Compute position size based on risk management rules.
        Ensures minimum notional value of $5.
        """
        risk_amount = capital * risk_pct
        sl_distance = abs(entry_price - sl_price)
        if sl_distance == 0:
            sl_distance = entry_price * 0.005

        qty = risk_amount / sl_distance
        notional = qty * entry_price

        # Ensure minimum notional value
        if notional < 5.0:
            qty = 5.5 / entry_price

        return round(qty, 6)

    def compute_sl_tp(
        self,
        entry_price: float,
        side: str,
        atr: float,
        rr_ratio: float = 2.0
    ) -> tuple:
        """
        Compute Stop Loss and Take Profit using ATR-based distance.
        SL distance = ATR * 1.5
        TP distance = SL distance * rr_ratio
        """
        sl_distance = atr * 1.5

        if sl_distance <= 0:
            sl_distance = entry_price * STOP_LOSS_PCT

        if side == 'LONG':
            sl = entry_price - sl_distance
            tp = entry_price + (sl_distance * rr_ratio)
        else:
            sl = entry_price + sl_distance
            tp = entry_price - (sl_distance * rr_ratio)

        return (sl, tp)

    def validate_trade(
        self,
        symbol: str,
        side: str,
        open_positions: Dict,
        max_concurrent: int = MAX_CONCURRENT_TRADES
    ) -> bool:
        """
        Validate if a new trade is allowed.
        Returns False if max concurrent positions reached or symbol already has open position.
        """
        if len(open_positions) >= max_concurrent:
            return False
        if symbol in open_positions:
            return False
        return True


# =============================================================================
# SECTION 7 - BOT STATE (Thread-Safe)
# =============================================================================
class BotState:
    """Thread-safe state container for all bot data."""

    def __init__(self):
        self._lock = threading.RLock()
        self.balance: float = DEFAULT_CAPITAL
        self.initial_balance: float = DEFAULT_CAPITAL
        self.total_pnl: float = 0.0
        self.trade_history: List[TradeRecord] = []
        self.open_positions: Dict[str, PositionState] = {}
        self.win_count: int = 0
        self.loss_count: int = 0
        self.is_running: bool = False
        self.mode: str = 'test'
        self.active_strategies: Dict[str, bool] = {
            'trend': True,
            'mean_reversion': True,
            'momentum': True,
            'breakout': True,
        }
        self.capital: float = DEFAULT_CAPITAL
        self.leverage: int = DEFAULT_LEVERAGE
        self.api_key: str = ''
        self.api_secret: str = ''
        self.backtest_results: Optional[Dict] = None
        self.last_update: Optional[datetime] = None
        self.ws_connected: bool = False
        self.selected_symbols: List[str] = list(DEFAULT_SELECTED_SYMBOLS)
        self.log_entries: List[str] = []

    def add_log(self, message: str) -> None:
        """Add a timestamped entry to the activity log."""
        with self._lock:
            timestamp = datetime.now().strftime('%H:%M:%S')
            self.log_entries.append(f"[{timestamp}] {message}")
            if len(self.log_entries) > 100:
                self.log_entries = self.log_entries[-100:]

    def add_open_position(self, pos: PositionState) -> None:
        """Add a new open position (thread-safe)."""
        with self._lock:
            self.open_positions[pos.symbol] = pos
            self.log_entries.append(f"[{datetime.now().strftime('%H:%M:%S')}] TRADE OPENED: {pos.side} {pos.symbol} @ {pos.entry_price:.4f}")
            logger.info(f"Opened {pos.side} position on {pos.symbol} at {pos.entry_price}")

    def close_position(self, symbol: str, exit_price: float) -> Optional[TradeRecord]:
        """
        Close a position, compute PnL, create TradeRecord, update stats.
        Returns the TradeRecord or None if position not found.
        """
        with self._lock:
            if symbol not in self.open_positions:
                return None

            pos = self.open_positions[symbol]

            # Compute PnL with leverage
            if pos.side == 'LONG':
                pnl = (exit_price - pos.entry_price) * pos.quantity
            else:
                pnl = (pos.entry_price - exit_price) * pos.quantity

            # Apply leverage to PnL
            pnl = pnl * self.leverage

            # Update win/loss counts
            if pnl > 0:
                self.win_count += 1
                status = 'WIN'
            else:
                self.loss_count += 1
                status = 'LOSS'

            # Update balance
            self.balance += pnl
            self.total_pnl += pnl

            # Create TradeRecord
            trade = TradeRecord(
                id=uuid.uuid4().hex[:8],
                symbol=pos.symbol,
                strategy=pos.strategy,
                side=pos.side,
                entry_price=pos.entry_price,
                exit_price=exit_price,
                sl_price=pos.sl_price,
                tp_price=pos.tp_price,
                quantity=pos.quantity,
                pnl=pnl,
                status=status,
                timestamp=pos.timestamp,
                closed_at=datetime.now(),
            )

            self.trade_history.append(trade)
            del self.open_positions[symbol]

            self.log_entries.append(f"[{datetime.now().strftime('%H:%M:%S')}] TRADE CLOSED: {pos.side} {symbol} | PnL: ${pnl:.4f} | {status}")
            logger.info(f"Closed {pos.side} on {symbol} | PnL: ${pnl:.4f} | Status: {status}")
            return trade

    def update_from_ui(
        self,
        capital: float,
        leverage: int,
        mode: str,
        active_strategies: Dict[str, bool],
        api_key: str,
        api_secret: str
    ) -> None:
        """Update bot configuration from UI inputs."""
        with self._lock:
            self.capital = capital
            self.leverage = leverage
            self.mode = mode
            self.active_strategies = active_strategies
            self.api_key = api_key
            self.api_secret = api_secret

    def get_stats(self) -> Dict[str, Any]:
        """Get current stats for dashboard display."""
        with self._lock:
            total_trades = len(self.trade_history)
            win_rate = 0.0
            if total_trades > 0:
                win_rate = (self.win_count / total_trades) * 100.0
            return {
                'balance': self.balance,
                'initial_balance': self.initial_balance,
                'total_pnl': self.total_pnl,
                'total_trades': total_trades,
                'win_count': self.win_count,
                'loss_count': self.loss_count,
                'win_rate': win_rate,
                'open_positions': len(self.open_positions),
            }

    def to_trade_history_df(self) -> Optional[pd.DataFrame]:
        """Convert trade history to a pandas DataFrame for display."""
        with self._lock:
            if not self.trade_history:
                return None
            rows = []
            for t in self.trade_history:
                rows.append({
                    'Time': t.timestamp.strftime('%Y-%m-%d %H:%M:%S') if t.timestamp else '',
                    'Symbol': t.symbol,
                    'Strategy': t.strategy,
                    'Side': t.side,
                    'Entry': round(t.entry_price, 4),
                    'Exit': round(t.exit_price, 4),
                    'SL': round(t.sl_price, 4),
                    'TP': round(t.tp_price, 4),
                    'Qty': round(t.quantity, 6),
                    'PnL ($)': round(t.pnl, 4),
                    'Status': t.status,
                })
            return pd.DataFrame(rows)

    def to_open_positions_df(self) -> Optional[pd.DataFrame]:
        """Convert open positions to a pandas DataFrame for display."""
        with self._lock:
            if not self.open_positions:
                return None
            rows = []
            for symbol, pos in self.open_positions.items():
                rows.append({
                    'Symbol': pos.symbol,
                    'Strategy': pos.strategy,
                    'Side': pos.side,
                    'Entry Price': round(pos.entry_price, 4),
                    'SL': round(pos.sl_price, 4),
                    'TP': round(pos.tp_price, 4),
                    'Qty': round(pos.quantity, 6),
                    'Opened': pos.timestamp.strftime('%Y-%m-%d %H:%M:%S') if pos.timestamp else '',
                })
            return pd.DataFrame(rows)


# =============================================================================
# SECTION 8 - MARKET DATA LAYER
# =============================================================================
def generate_mock_ohlcv(symbol: str, num_candles: int = 500) -> pd.DataFrame:
    """
    Generate realistic mock OHLCV data using a random walk.
    Base price varies by symbol for realism.
    """
    base_prices = {
        'BTC/USDT:USDT': 40000.0,
        'ETH/USDT:USDT': 2500.0,
        'BNB/USDT:USDT': 300.0,
        'SOL/USDT:USDT': 100.0,
        'XRP/USDT:USDT': 0.55,
        'DOGE/USDT:USDT': 0.08,
        'ADA/USDT:USDT': 0.45,
        'AVAX/USDT:USDT': 35.0,
        'LINK/USDT:USDT': 14.0,
        'MATIC/USDT:USDT': 0.85,
        'DOT/USDT:USDT': 7.0,
        'UNI/USDT:USDT': 6.5,
    }

    base_price = base_prices.get(symbol, 100.0)
    volatility = base_price * 0.002

    np.random.seed(hash(symbol) % (2**31))

    # Generate returns using normal distribution
    returns = np.random.normal(0, 0.003, num_candles)
    prices = np.zeros(num_candles)
    prices[0] = base_price

    for i in range(1, num_candles):
        prices[i] = prices[i - 1] * (1 + returns[i])

    # Build OHLCV data
    timestamps = pd.date_range(
        end=datetime.now(),
        periods=num_candles,
        freq='5min'
    )

    opens = prices.copy()
    closes = np.roll(prices, -1)
    closes[-1] = prices[-1] * (1 + np.random.normal(0, 0.001))

    highs = np.maximum(opens, closes) * (1 + np.abs(np.random.normal(0, 0.001, num_candles)))
    lows = np.minimum(opens, closes) * (1 - np.abs(np.random.normal(0, 0.001, num_candles)))

    # Generate volume
    base_volume = base_price * 100
    volumes = np.abs(np.random.normal(base_volume, base_volume * 0.3, num_candles))

    df = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes,
    }, index=timestamps)

    return df


# =============================================================================
# SECTION 9 - WEBSOCKET MANAGER
# =============================================================================
class WebSocketManager:
    """
    Manages WebSocket connections to Binance Futures in a background thread.
    Maintains a cache of latest OHLCV data for each symbol.
    Uses bot_state.selected_symbols for dynamic symbol selection.
    """

    def __init__(self, bot_state: 'BotState'):
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self.kline_cache: Dict[str, deque] = {}
        self.connected: bool = False
        self._stop_event = threading.Event()
        self.bot_state = bot_state

        # Initialize cache for all available symbols
        for symbol in AVAILABLE_SYMBOLS:
            self.kline_cache[symbol] = deque(maxlen=500)

    def start(self) -> None:
        """Start the WebSocket background thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("WebSocket manager started")

    def stop(self) -> None:
        """Stop the WebSocket background thread."""
        self._stop_event.set()
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        self.connected = False
        logger.info("WebSocket manager stopped")

    def _run_loop(self) -> None:
        """Run the asyncio event loop in the background thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._main())
        except Exception as e:
            logger.error(f"WebSocket loop error: {e}")
        finally:
            self._loop.close()

    async def _main(self) -> None:
        """Main async loop with reconnection logic and exponential backoff."""
        attempt = 0
        max_attempts = 10

        while not self._stop_event.is_set() and attempt < max_attempts:
            try:
                # Build stream names from selected symbols
                streams = []
                symbols_to_stream = self.bot_state.selected_symbols if self.bot_state.selected_symbols else SCAN_SYMBOLS
                for symbol in symbols_to_stream:
                    stream_name = self._symbol_to_stream(symbol)
                    streams.append(stream_name)

                stream_url = f"{WS_BASE_URL}?streams={'/'.join(streams)}"

                async with websockets.connect(stream_url, ping_interval=20, ping_timeout=10) as ws:
                    self.connected = True
                    attempt = 0
                    logger.info("WebSocket connected to Binance Futures")

                    while not self._stop_event.is_set():
                        try:
                            message = await asyncio.wait_for(ws.recv(), timeout=30)
                            self._process_message(message)
                        except asyncio.TimeoutError:
                            # Send ping to keep connection alive
                            continue
                        except websockets.exceptions.ConnectionClosed:
                            logger.warning("WebSocket connection closed")
                            break

            except (websockets.exceptions.ConnectionClosed, OSError, Exception) as e:
                self.connected = False
                attempt += 1
                backoff = min(2 ** attempt, 60)
                logger.warning(f"WebSocket reconnecting (attempt {attempt}/{max_attempts}), backoff {backoff}s: {e}")
                await asyncio.sleep(backoff)

        self.connected = False
        logger.info("WebSocket manager exiting")

    def _symbol_to_stream(self, symbol: str) -> str:
        """Convert symbol like 'BTC/USDT:USDT' to stream name 'btcusdt@kline_5m'."""
        stream = symbol.lower().replace('/', '').replace(':usdt', '')
        return f"{stream}@kline_5m"

    def _process_message(self, message: str) -> None:
        """Parse WebSocket message and update kline cache."""
        try:
            data = json.loads(message)
            if 'data' not in data:
                return

            k = data['data'].get('k')
            if k is None:
                return

            # Only process closed candles
            if not k.get('x', False):
                return

            # Determine symbol from stream name
            stream = data.get('stream', '')
            symbol = self._stream_to_symbol(stream)
            if symbol is None:
                return

            candle = [
                k['t'],           # timestamp
                float(k['o']),    # open
                float(k['h']),    # high
                float(k['l']),    # low
                float(k['c']),    # close
                float(k['v']),    # volume
            ]

            with self._lock:
                self.kline_cache[symbol].append(candle)

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.debug(f"Error processing WS message: {e}")

    def _stream_to_symbol(self, stream: str) -> Optional[str]:
        """Convert stream name back to symbol."""
        # stream format: btcusdt@kline_5m
        pair = stream.split('@')[0] if '@' in stream else ''
        for symbol in AVAILABLE_SYMBOLS:
            expected = symbol.lower().replace('/', '').replace(':usdt', '')
            if pair == expected:
                return symbol
        return None

    def get_latest_ohlcv(self, symbol: str, limit: int = 200) -> Optional[pd.DataFrame]:
        """Get latest OHLCV data from cache as a DataFrame."""
        with self._lock:
            cache = self.kline_cache.get(symbol)
            if cache is None or len(cache) == 0:
                return None

            data = list(cache)[-limit:]

        if len(data) < 10:
            return None

        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df


# =============================================================================
# SECTION 10 - BOT SCANNER
# =============================================================================
class BotScanner:
    """
    Background scanner that monitors symbols and generates trade signals.
    Runs in a daemon thread with configurable scan interval.
    """

    def __init__(self, bot_state: BotState, ws_manager: WebSocketManager, risk_manager: RiskManager):
        self.bot_state = bot_state
        self.ws_manager = ws_manager
        self.risk_manager = risk_manager
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the scanner background thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._scan_loop, daemon=True)
        self._thread.start()
        logger.info("Bot scanner started")

    def stop(self) -> None:
        """Stop the scanner background thread."""
        self._stop_event.set()
        logger.info("Bot scanner stopped")

    def _scan_loop(self) -> None:
        """Main scan loop running every 30 seconds."""
        while not self._stop_event.is_set():
            try:
                self._do_scan()
            except Exception as e:
                logger.error(f"Scan error: {e}")
            # Sleep in small intervals to allow quick stop
            for _ in range(30):
                if self._stop_event.is_set():
                    break
                time.sleep(1)

    def _do_scan(self) -> None:
        """Perform a full scan of selected symbols."""
        if not self.bot_state.is_running:
            return

        if self.bot_state.mode == 'backtrack':
            return

        symbols_to_scan = self.bot_state.selected_symbols if self.bot_state.selected_symbols else SCAN_SYMBOLS
        self.bot_state.add_log(f"Scanning {len(symbols_to_scan)} symbols...")
        for symbol in symbols_to_scan:
            if self._stop_event.is_set():
                break

            try:
                # Get OHLCV data
                df = self.ws_manager.get_latest_ohlcv(symbol)

                # REST fallback if WebSocket cache is empty
                if df is None or len(df) < 60:
                    df = self._fetch_rest_fallback(symbol)

                # Mock data fallback for test mode if real data unavailable
                if (df is None or len(df) < 60) and self.bot_state.mode == 'test':
                    df = generate_mock_ohlcv(symbol, 200)
                    self.bot_state.add_log(f"Using simulated data for {symbol}")

                if df is None or len(df) < 60:
                    self.bot_state.add_log(f"No data for {symbol}, skipping")
                    continue

                # Compute indicators
                df = compute_indicators(df)

                # Check open positions for SL/TP hit
                current_price = df['close'].iloc[-1]
                self._check_open_positions(symbol, current_price)

                # Run enabled strategies
                if self.bot_state.active_strategies.get('trend', False):
                    signal = strategy_trend_following(df)
                    if signal:
                        self.bot_state.add_log(f"Signal: {signal['strategy']} {signal['side']} on {symbol}")
                        self._try_enter_trade(symbol, signal, df)

                if self.bot_state.active_strategies.get('mean_reversion', False):
                    signal = strategy_mean_reversion(df)
                    if signal:
                        self.bot_state.add_log(f"Signal: {signal['strategy']} {signal['side']} on {symbol}")
                        self._try_enter_trade(symbol, signal, df)

                if self.bot_state.active_strategies.get('momentum', False):
                    signal = strategy_momentum(df)
                    if signal:
                        self.bot_state.add_log(f"Signal: {signal['strategy']} {signal['side']} on {symbol}")
                        self._try_enter_trade(symbol, signal, df)

                if self.bot_state.active_strategies.get('breakout', False):
                    signal = strategy_breakout(df)
                    if signal:
                        self.bot_state.add_log(f"Signal: {signal['strategy']} {signal['side']} on {symbol}")
                        self._try_enter_trade(symbol, signal, df)

            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")

        self.bot_state.add_log("Scan cycle complete. Next scan in 30s.")
        self.bot_state.last_update = datetime.now()

    def _fetch_rest_fallback(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data via REST API as fallback."""
        try:
            import ccxt
            exchange = ccxt.binanceusdm({
                'enableRateLimit': True,
                'timeout': 10000,
                'options': {'defaultType': 'future'},
            })
            ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=CANDLE_LIMIT)
            if not ohlcv:
                return None
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            logger.debug(f"REST fallback failed for {symbol}: {e}")
            return None

    def _try_enter_trade(self, symbol: str, signal: Dict[str, Any], df: pd.DataFrame) -> None:
        """Attempt to enter a trade based on a signal."""
        # Validate trade
        if not self.risk_manager.validate_trade(
            symbol, signal['side'], self.bot_state.open_positions
        ):
            return

        entry_price = df['close'].iloc[-1]
        atr_col = 'ATRr_14'
        atr = df[atr_col].iloc[-1] if atr_col in df.columns and not pd.isna(df[atr_col].iloc[-1]) else entry_price * 0.01

        # Compute SL and TP
        sl, tp = self.risk_manager.compute_sl_tp(entry_price, signal['side'], atr)

        # Compute position size
        qty = self.risk_manager.compute_position_size(
            self.bot_state.capital, self.bot_state.leverage, entry_price, sl
        )

        if self.bot_state.mode == 'test':
            # Paper trade - create position in state
            pos = PositionState(
                id=uuid.uuid4().hex[:8],
                symbol=symbol,
                strategy=signal['strategy'],
                side=signal['side'],
                entry_price=entry_price,
                sl_price=sl,
                tp_price=tp,
                quantity=qty,
                timestamp=datetime.now(),
            )
            self.bot_state.add_open_position(pos)

        elif self.bot_state.mode == 'real':
            self._execute_real_order(symbol, signal, entry_price, qty, sl, tp)

    def _execute_real_order(
        self,
        symbol: str,
        signal: Dict[str, Any],
        entry_price: float,
        qty: float,
        sl: float,
        tp: float
    ) -> None:
        """Execute a real order on Binance Futures using CCXT."""
        try:
            import ccxt
            exchange = ccxt.binanceusdm({
                'apiKey': self.bot_state.api_key,
                'secret': self.bot_state.api_secret,
                'enableRateLimit': True,
                'options': {'defaultType': 'future'},
            })

            # Set leverage
            exchange.set_leverage(self.bot_state.leverage, symbol)

            # Set margin mode to isolated
            try:
                exchange.set_margin_mode('isolated', symbol)
            except ccxt.BaseError:
                pass  # May already be set

            side = signal['side'].lower()
            opposite_side = 'sell' if side == 'buy' else 'buy'
            if signal['side'] == 'LONG':
                side = 'buy'
                opposite_side = 'sell'
            else:
                side = 'sell'
                opposite_side = 'buy'

            # Place market order
            order = exchange.create_order(symbol, 'market', side, qty)
            logger.info(f"Market order placed: {order.get('id', 'N/A')}")

            # Place Stop Loss order
            sl_order = exchange.create_order(
                symbol, 'stop_market', opposite_side, qty,
                params={'stopPrice': sl, 'closePosition': True}
            )
            logger.info(f"SL order placed: {sl_order.get('id', 'N/A')}")

            # Place Take Profit order
            tp_order = exchange.create_order(
                symbol, 'take_profit_market', opposite_side, qty,
                params={'stopPrice': tp, 'closePosition': True}
            )
            logger.info(f"TP order placed: {tp_order.get('id', 'N/A')}")

            # Create position in state
            pos = PositionState(
                id=uuid.uuid4().hex[:8],
                symbol=symbol,
                strategy=signal['strategy'],
                side=signal['side'],
                entry_price=entry_price,
                sl_price=sl,
                tp_price=tp,
                quantity=qty,
                timestamp=datetime.now(),
            )
            self.bot_state.add_open_position(pos)

        except ccxt.BaseError as e:
            logger.error(f"Order execution failed for {symbol}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error executing order for {symbol}: {e}")

    def _check_open_positions(self, symbol: str, current_price: float) -> None:
        """Check if any open position hit SL or TP."""
        if symbol not in self.bot_state.open_positions:
            return

        pos = self.bot_state.open_positions[symbol]

        hit_sl = False
        hit_tp = False

        if pos.side == 'LONG':
            hit_sl = current_price <= pos.sl_price
            hit_tp = current_price >= pos.tp_price
        else:
            hit_sl = current_price >= pos.sl_price
            hit_tp = current_price <= pos.tp_price

        if hit_sl or hit_tp:
            exit_price = pos.sl_price if hit_sl else pos.tp_price
            self.bot_state.close_position(symbol, exit_price)


# =============================================================================
# SECTION 11 - BACKTEST ENGINE
# =============================================================================
class BacktestEngine:
    """
    Backtesting engine that runs strategies over historical (mock) data.
    Produces performance metrics including PnL, win rate, Sharpe, and max drawdown.
    """

    def run(
        self,
        bot_state: BotState,
        symbols: Optional[List[str]] = None,
        num_candles: int = 500
    ) -> Dict[str, Any]:
        """Run backtest over mock data for enabled strategies."""
        if symbols is None:
            symbols = bot_state.selected_symbols if bot_state.selected_symbols else SCAN_SYMBOLS[:3]

        strategy_fns: Dict[str, Callable] = {}
        if bot_state.active_strategies.get('trend', False):
            strategy_fns['Trend Following'] = strategy_trend_following
        if bot_state.active_strategies.get('mean_reversion', False):
            strategy_fns['Mean Reversion'] = strategy_mean_reversion
        if bot_state.active_strategies.get('momentum', False):
            strategy_fns['Momentum'] = strategy_momentum
        if bot_state.active_strategies.get('breakout', False):
            strategy_fns['Breakout'] = strategy_breakout

        if not strategy_fns:
            return {
                'trades': [],
                'total_pnl': 0.0,
                'win_rate': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'by_strategy': {},
            }

        all_trades: List[Dict] = []
        pnl_series: List[float] = []
        by_strategy: Dict[str, Dict] = {name: {'pnl': 0.0, 'trades': 0, 'wins': 0} for name in strategy_fns}

        rm = RiskManager()

        for symbol in symbols:
            df = generate_mock_ohlcv(symbol, num_candles)
            df = compute_indicators(df)

            for strat_name, strat_fn in strategy_fns.items():
                open_pos: Optional[Dict] = None

                for i in range(60, len(df)):
                    sub_df = df.iloc[:i + 1].copy()
                    current_close = sub_df['close'].iloc[-1]
                    current_high = sub_df['high'].iloc[-1]
                    current_low = sub_df['low'].iloc[-1]

                    # Check if open position hits SL/TP
                    if open_pos is not None:
                        hit_sl = False
                        hit_tp = False

                        if open_pos['side'] == 'LONG':
                            if current_low <= open_pos['sl']:
                                hit_sl = True
                            elif current_high >= open_pos['tp']:
                                hit_tp = True
                        else:
                            if current_high >= open_pos['sl']:
                                hit_sl = True
                            elif current_low <= open_pos['tp']:
                                hit_tp = True

                        if hit_sl or hit_tp:
                            exit_price = open_pos['sl'] if hit_sl else open_pos['tp']
                            if open_pos['side'] == 'LONG':
                                pnl = (exit_price - open_pos['entry']) * open_pos['qty'] * bot_state.leverage
                            else:
                                pnl = (open_pos['entry'] - exit_price) * open_pos['qty'] * bot_state.leverage

                            trade_rec = {
                                'symbol': open_pos['symbol'],
                                'strategy': strat_name,
                                'side': open_pos['side'],
                                'entry': open_pos['entry'],
                                'exit': exit_price,
                                'sl': open_pos['sl'],
                                'tp': open_pos['tp'],
                                'qty': open_pos['qty'],
                                'pnl': pnl,
                            }
                            all_trades.append(trade_rec)
                            pnl_series.append(pnl)
                            by_strategy[strat_name]['pnl'] += pnl
                            by_strategy[strat_name]['trades'] += 1
                            if pnl > 0:
                                by_strategy[strat_name]['wins'] += 1
                            open_pos = None

                    # Try to open new position if none open
                    if open_pos is None:
                        signal = strat_fn(sub_df)
                        if signal is not None:
                            entry = current_close
                            atr_col = 'ATRr_14'
                            atr_val = sub_df[atr_col].iloc[-1] if atr_col in sub_df.columns and not pd.isna(sub_df[atr_col].iloc[-1]) else entry * 0.01
                            sl, tp = rm.compute_sl_tp(entry, signal['side'], atr_val)
                            qty = rm.compute_position_size(bot_state.capital, bot_state.leverage, entry, sl)
                            open_pos = {
                                'symbol': symbol,
                                'side': signal['side'],
                                'entry': entry,
                                'sl': sl,
                                'tp': tp,
                                'qty': qty,
                            }

        # Compute aggregate metrics
        total_pnl = sum(t['pnl'] for t in all_trades) if all_trades else 0.0
        total_trades_count = len(all_trades)
        wins = sum(1 for t in all_trades if t['pnl'] > 0)
        win_rate = (wins / total_trades_count * 100.0) if total_trades_count > 0 else 0.0

        # Compute equity curve for max drawdown
        equity_curve = [bot_state.capital]
        for pnl_val in pnl_series:
            equity_curve.append(equity_curve[-1] + pnl_val)

        max_drawdown = self._compute_max_drawdown(equity_curve)
        sharpe = self._compute_sharpe(pnl_series)

        # Compute per-strategy win rates
        for name in by_strategy:
            s = by_strategy[name]
            s['win_rate'] = (s['wins'] / s['trades'] * 100.0) if s['trades'] > 0 else 0.0

        return {
            'trades': all_trades,
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe,
            'by_strategy': by_strategy,
        }

    def _compute_sharpe(self, pnl_series: List[float]) -> float:
        """Compute Sharpe ratio from PnL series."""
        if not pnl_series:
            return 0.0
        daily = np.array(pnl_series)
        mean = daily.mean()
        std = daily.std()
        if std == 0:
            return 0.0
        return float((mean / std) * np.sqrt(252))

    def _compute_max_drawdown(self, equity_curve: List[float]) -> float:
        """Compute maximum drawdown from equity curve."""
        if not equity_curve:
            return 0.0
        peak = equity_curve[0]
        max_dd = 0.0
        for v in equity_curve:
            peak = max(peak, v)
            if peak > 0:
                dd = (peak - v) / peak
                max_dd = max(max_dd, dd)
        return max_dd


# =============================================================================
# SECTION 12 - STREAMLIT UI
# =============================================================================
def main():
    """Main Streamlit application function."""
    try:
        st.set_page_config(
            page_title='Crypto Futures Bot',
            layout='wide',
            page_icon='\U0001f4c8'
        )

        # ---------------------------------------------------------------------
        # Use session_state for bot state (resets on page refresh)
        # ---------------------------------------------------------------------
        if 'bot_state' not in st.session_state:
            st.session_state.bot_state = BotState()
        if 'ws_manager' not in st.session_state:
            st.session_state.ws_manager = WebSocketManager(st.session_state.bot_state)
        if 'scanner' not in st.session_state:
            st.session_state.scanner = BotScanner(st.session_state.bot_state, st.session_state.ws_manager, RiskManager())
        if 'backtest_engine' not in st.session_state:
            st.session_state.backtest_engine = BacktestEngine()

        bot = st.session_state.bot_state
        ws_manager = st.session_state.ws_manager
        scanner = st.session_state.scanner

        # ---------------------------------------------------------------------
        # SIDEBAR - Bot Control
        # ---------------------------------------------------------------------
        with st.sidebar:
            st.title('\u2699\ufe0f Bot Control')
            mode = st.radio('Mode', ['backtrack', 'test', 'real'], index=1)
            st.divider()

            if mode == 'real':
                api_key = st.text_input('Binance API Key', type='password')
                api_secret = st.text_input('Binance Secret Key', type='password')
            else:
                api_key = ''
                api_secret = ''

            capital = st.slider('Capital ($)', 0.0, 20.0, 20.0, 0.5)
            leverage = st.slider('Leverage', 1, 10, 3)
            max_concurrent = st.selectbox('Max Concurrent Trades', [1, 2], index=1)
            st.divider()

            # Coin Pair Selector
            st.subheader('\U0001f4b1 Coins to Scan')
            if 'selected_coins' not in st.session_state:
                st.session_state.selected_coins = list(DEFAULT_SELECTED_SYMBOLS)

            selected_symbols = st.multiselect(
                'Select coin pairs',
                options=AVAILABLE_SYMBOLS,
                key='selected_coins',
                help='Choose which coin pairs the bot will scan for trading signals.'
            )
            # Update bot state with selected symbols
            if selected_symbols:
                bot.selected_symbols = selected_symbols
            else:
                bot.selected_symbols = list(DEFAULT_SELECTED_SYMBOLS)
            st.divider()

            st.subheader('\U0001f4ca Strategy Toggles')
            strat_trend = st.checkbox('A: Trend Following (EMA Cross + RSI)', value=True)
            strat_mean = st.checkbox('B: Mean Reversion (BB + RSI)', value=True)
            strat_mom = st.checkbox('C: Momentum (MACD + Volume)', value=True)
            strat_break = st.checkbox('D: Breakout (24H High/Low)', value=True)
            active_strategies = {
                'trend': strat_trend,
                'mean_reversion': strat_mean,
                'momentum': strat_mom,
                'breakout': strat_break,
            }
            st.divider()

            col_start, col_stop = st.columns(2)
            with col_start:
                if st.button('\u25b6 Start', use_container_width=True, type='primary'):
                    bot.update_from_ui(capital, leverage, mode, active_strategies, api_key, api_secret)
                    if not bot.is_running:
                        bot.is_running = True
                        bot.add_log(f"Bot started in {mode.upper()} mode")
                        if mode in ('test', 'real'):
                            ws_manager.start()
                            scanner.start()
                        st.rerun()

            with col_stop:
                if st.button('\u23f9 Stop', use_container_width=True):
                    bot.is_running = False
                    bot.add_log("Bot stopped by user")
                    scanner.stop()
                    ws_manager.stop()
                    st.rerun()

            # Always sync strategy toggles
            bot.active_strategies = active_strategies
            bot.capital = capital
            bot.leverage = leverage

            # Status indicator
            if bot.is_running:
                st.success('\U0001f7e2 Bot Running')
            else:
                st.error('\U0001f534 Bot Stopped')

            ws_status = '\U0001f7e2 Connected' if ws_manager.connected else '\U0001f534 Disconnected'
            st.caption(f'WebSocket: {ws_status}')

        # ---------------------------------------------------------------------
        # MAIN AREA
        # ---------------------------------------------------------------------
        st.title('\U0001f4c8 Binance Futures Trading Bot')
        last_update_str = bot.last_update.strftime("%H:%M:%S") if bot.last_update else "Never"
        st.caption(f'Mode: {mode.upper()} | Last Update: {last_update_str}')

        # Metrics row
        stats = bot.get_stats()
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric('\U0001f4b0 Balance', f'${stats["balance"]:.2f}', delta=f'{stats["total_pnl"]:+.2f}')
        c2.metric('\U0001f4ca Total Trades', stats['total_trades'])
        c3.metric('\u2705 Win Rate', f'{stats["win_rate"]:.1f}%')
        c4.metric('\U0001f3c6 Wins / Losses', f'{stats["win_count"]} / {stats["loss_count"]}')
        c5.metric('\U0001f4c2 Open Positions', stats['open_positions'])

        # ---------------------------------------------------------------------
        # Backtest section
        # ---------------------------------------------------------------------
        if mode == 'backtrack':
            st.subheader('\U0001f52c Backtest Results')
            if st.button('Run Backtest Now', type='primary'):
                with st.spinner('Running backtest...'):
                    bot.add_log(f"Starting backtest on {len(bot.selected_symbols)} symbols...")
                    results = st.session_state.backtest_engine.run(bot)
                    bot.backtest_results = results
                    bot.add_log(f"Backtest done: {len(results['trades'])} trades, PnL: ${results['total_pnl']:.2f}")

            if bot.backtest_results:
                r = bot.backtest_results
                bc1, bc2, bc3, bc4 = st.columns(4)
                bc1.metric('Total PnL', f'${r["total_pnl"]:.2f}')
                bc2.metric('Win Rate', f'{r["win_rate"]:.1f}%')
                bc3.metric('Max Drawdown', f'{r["max_drawdown"] * 100:.1f}%')
                bc4.metric('Sharpe Ratio', f'{r["sharpe_ratio"]:.2f}')

                # Per-strategy breakdown
                if r.get('by_strategy'):
                    st.subheader('Strategy Breakdown')
                    strat_rows = []
                    for sname, sdata in r['by_strategy'].items():
                        strat_rows.append({
                            'Strategy': sname,
                            'Trades': sdata['trades'],
                            'PnL ($)': round(sdata['pnl'], 2),
                            'Win Rate (%)': round(sdata['win_rate'], 1),
                        })
                    st.dataframe(pd.DataFrame(strat_rows), use_container_width=True)

                # Backtest trades table
                if r.get('trades'):
                    st.subheader('Simulated Trades')
                    trades_df = pd.DataFrame([
                        {
                            'Symbol': t['symbol'],
                            'Strategy': t['strategy'],
                            'Side': t['side'],
                            'Entry': round(t['entry'], 4),
                            'Exit': round(t['exit'], 4),
                            'PnL': round(t['pnl'], 4),
                        }
                        for t in r['trades']
                    ])
                    st.dataframe(trades_df, use_container_width=True)

        # ---------------------------------------------------------------------
        # Open Positions table
        # ---------------------------------------------------------------------
        st.subheader('\U0001f4c2 Open Positions')
        open_df = bot.to_open_positions_df()
        if open_df is not None and not open_df.empty:
            st.dataframe(open_df, use_container_width=True)
        else:
            st.info('No open positions.')

        # ---------------------------------------------------------------------
        # Trade History table
        # ---------------------------------------------------------------------
        st.subheader('\U0001f4dc Trade History')
        history_df = bot.to_trade_history_df()
        if history_df is not None and not history_df.empty:
            st.dataframe(history_df, use_container_width=True, height=400)
        else:
            st.info('No completed trades yet.')

        # ---------------------------------------------------------------------
        # Activity Log
        # ---------------------------------------------------------------------
        st.subheader('\U0001f4cb Activity Log')
        if bot.log_entries:
            log_text = '\n'.join(reversed(bot.log_entries[-50:]))
            st.code(log_text, language='text')
        else:
            st.info('No activity yet. Start the bot to see logs.')

        # ---------------------------------------------------------------------
        # Auto-refresh when bot is running (non-blocking browser-side refresh)
        # ---------------------------------------------------------------------
        if bot.is_running:
            st_autorefresh(interval=5000, limit=None, key="bot_autorefresh")

    except Exception as e:
        st.error(f"Application error: {e}")
        st.exception(e)


# =============================================================================
# SECTION 13 - ENTRY POINT
# =============================================================================
if __name__ == '__main__':
    main()
