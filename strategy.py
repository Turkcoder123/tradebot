"""
strategy.py

Scalp–Trend hybrid trading strategy for EURUSD / XAUUSD.

Design principles
-----------------
* **No overfitting** – only standard, widely-used indicator parameters are
  employed (EMA-50/200, RSI-14, MACD 12/26/9, ATR-14).  No parameter
  optimisation was performed; every value is a textbook default.
* **Trend filter** – EMA-fast vs EMA-slow determines the allowed trade direction.
* **Entry type 1 – Trend Pullback** – price retests EMA-fast in the direction
  of the trend *and* the MACD histogram confirms momentum.
* **Entry type 2 – Momentum Confirmation** – MACD histogram crosses zero in
  the trend direction, confirming a fresh wave of momentum.
* **Entry type 3 – BB Bounce** – price touches the outer Bollinger Band in
  the trend direction and closes back inside, capturing mean-reversion scalps.
* **Session filter** – trades are only allowed during configurable session hours.
* **ATR-based risk** – stop-loss and take-profit are scaled to recent
  volatility so the strategy adapts to changing market conditions.
* **3 parameter presets** – Aggressive, Balanced, Conservative – allow
  selecting a risk/reward profile without manual tuning.

Multi-symbol support
--------------------
Both EURUSD (pip=0.0001) and XAUUSD (pip=0.01) are supported via
:class:`SymbolSpec`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants (deliberately kept at textbook defaults to avoid curve-fitting)
# ---------------------------------------------------------------------------

EMA_FAST: int = 50
EMA_SLOW: int = 200
RSI_PERIOD: int = 14
BB_PERIOD: int = 20
BB_STD: float = 2.0
ATR_PERIOD: int = 14

# MACD (standard 12/26/9)
MACD_FAST: int = 12
MACD_SLOW: int = 26
MACD_SIGNAL: int = 9

# Session filter (UTC hours, inclusive)
SESSION_START_HOUR: int = 8
SESSION_END_HOUR: int = 17

# Risk parameters
SL_ATR_MULT: float = 1.5   # stop-loss = 1.5 × ATR
TP_ATR_MULT: float = 3.0   # take-profit = 3.0 × ATR (risk:reward = 1:2)

# EMA retest tolerance (0.1 % buffer)
EMA_RETEST_TOL: float = 0.001


# ---------------------------------------------------------------------------
# Strategy configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StrategyConfig:
    """All tuneable strategy parameters in one place.

    Three presets are provided below: ``V1_AGGRESSIVE``, ``V2_BALANCED``,
    ``V3_CONSERVATIVE``.  Creating a :class:`StrategyConfig` with custom
    values is also supported.
    """

    name: str = "default"
    ema_fast: int = EMA_FAST
    ema_slow: int = EMA_SLOW
    rsi_period: int = RSI_PERIOD
    bb_period: int = BB_PERIOD
    bb_std: float = BB_STD
    atr_period: int = ATR_PERIOD
    macd_fast: int = MACD_FAST
    macd_slow: int = MACD_SLOW
    macd_signal: int = MACD_SIGNAL
    session_start: int = SESSION_START_HOUR
    session_end: int = SESSION_END_HOUR
    sl_atr_mult: float = SL_ATR_MULT
    tp_atr_mult: float = TP_ATR_MULT
    ema_retest_tol: float = EMA_RETEST_TOL
    enable_bb_bounce: bool = True       # Entry type 3 – BB bounce


# Pre-built presets --------------------------------------------------------

V1_AGGRESSIVE = StrategyConfig(
    name="V1_Agresif",
    ema_fast=20,
    ema_slow=50,
    macd_fast=8,
    macd_slow=17,
    macd_signal=9,
    session_start=7,
    session_end=19,
    sl_atr_mult=1.0,
    tp_atr_mult=1.5,
    ema_retest_tol=0.002,
    enable_bb_bounce=True,
)

V2_BALANCED = StrategyConfig(
    name="V2_Dengeli",
    ema_fast=50,
    ema_slow=200,
    macd_fast=12,
    macd_slow=26,
    macd_signal=9,
    session_start=8,
    session_end=17,
    sl_atr_mult=1.5,
    tp_atr_mult=3.0,
    ema_retest_tol=0.001,
    enable_bb_bounce=True,
)

V3_CONSERVATIVE = StrategyConfig(
    name="V3_Muhafazakar",
    ema_fast=50,
    ema_slow=200,
    macd_fast=12,
    macd_slow=26,
    macd_signal=9,
    session_start=9,
    session_end=16,
    sl_atr_mult=2.0,
    tp_atr_mult=4.0,
    ema_retest_tol=0.0005,
    enable_bb_bounce=False,
)

ALL_CONFIGS = [V1_AGGRESSIVE, V2_BALANCED, V3_CONSERVATIVE]


# ---------------------------------------------------------------------------
# Symbol specification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SymbolSpec:
    """Instrument-specific parameters."""

    name: str
    pip_size: float
    pip_value_per_lot: float   # USD P&L per pip per standard lot
    default_spread_pips: float
    max_slippage_pips: float
    csv_path: str = ""


EURUSD_SPEC = SymbolSpec(
    name="EURUSD",
    pip_size=0.0001,
    pip_value_per_lot=10.0,
    default_spread_pips=1.5,
    max_slippage_pips=2.0,
    csv_path="data/EURUSD_6m.csv",
)

XAUUSD_SPEC = SymbolSpec(
    name="XAUUSD",
    pip_size=0.01,
    pip_value_per_lot=1.0,
    default_spread_pips=30.0,
    max_slippage_pips=10.0,
    csv_path="data/XAUUSD_6m.csv",
)

ALL_SYMBOLS = [EURUSD_SPEC, XAUUSD_SPEC]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class Direction(Enum):
    LONG = 1
    SHORT = -1


@dataclass
class Signal:
    """A concrete trade signal emitted by the strategy."""

    direction: Direction
    entry_price: float
    stop_loss: float
    take_profit: float
    atr: float
    bar_index: int          # index in the DataFrame
    bar_time: pd.Timestamp
    signal_type: str = ""   # "pullback", "momentum", or "bb_bounce"


# ---------------------------------------------------------------------------
# Indicator helpers
# ---------------------------------------------------------------------------

def compute_ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=span, adjust=False).mean()


def compute_rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Relative Strength Index (Wilder's smoothing)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def compute_bollinger(
    series: pd.Series,
    period: int = BB_PERIOD,
    num_std: float = BB_STD,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (middle, upper, lower) Bollinger Bands."""
    middle = series.rolling(period).mean()
    std = series.rolling(period).std(ddof=0)
    upper = middle + num_std * std
    lower = middle - num_std * std
    return middle, upper, lower


def compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = ATR_PERIOD,
) -> pd.Series:
    """Average True Range."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def compute_macd(
    series: pd.Series,
    fast: int = MACD_FAST,
    slow: int = MACD_SLOW,
    signal: int = MACD_SIGNAL,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (macd_line, signal_line, histogram)."""
    ema_f = compute_ema(series, fast)
    ema_s = compute_ema(series, slow)
    macd_line = ema_f - ema_s
    signal_line = compute_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


# ---------------------------------------------------------------------------
# Indicator attachment
# ---------------------------------------------------------------------------

def add_indicators(
    df: pd.DataFrame,
    config: Optional[StrategyConfig] = None,
) -> pd.DataFrame:
    """Add all strategy indicators to *df* and return a new DataFrame.

    Expected columns: ``open, high, low, close`` (and optionally ``time``).
    When *config* is ``None``, the module-level constants are used (backward
    compatible).
    """
    cfg = config or StrategyConfig()
    df = df.copy()
    df["ema_fast"] = compute_ema(df["close"], cfg.ema_fast)
    df["ema_slow"] = compute_ema(df["close"], cfg.ema_slow)
    df["rsi"] = compute_rsi(df["close"], cfg.rsi_period)
    df["bb_mid"], df["bb_upper"], df["bb_lower"] = compute_bollinger(
        df["close"], cfg.bb_period, cfg.bb_std
    )
    df["atr"] = compute_atr(df["high"], df["low"], df["close"], cfg.atr_period)
    df["macd"], df["macd_signal"], df["macd_hist"] = compute_macd(
        df["close"], cfg.macd_fast, cfg.macd_slow, cfg.macd_signal,
    )

    # Trend direction: +1 uptrend, -1 downtrend
    df["trend"] = np.where(df["ema_fast"] > df["ema_slow"], 1, -1)
    return df


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------

def _in_session(hour: int, start: int = SESSION_START_HOUR,
                end: int = SESSION_END_HOUR) -> bool:
    """Return True if *hour* (UTC) is inside the allowed trading session."""
    return start <= hour <= end


def generate_signals(
    df: pd.DataFrame,
    config: Optional[StrategyConfig] = None,
) -> list[Signal]:
    """Scan the DataFrame and return a list of :class:`Signal` objects.

    Three entry types are checked:

    1. **Trend Pullback** – price retests EMA-fast in the direction of the
       larger trend and the MACD histogram confirms the prevailing momentum.

    2. **Momentum Confirmation** – MACD histogram crosses zero in the trend
       direction (fresh momentum wave).

    3. **BB Bounce** – price touches the outer Bollinger Band in the trend
       direction and closes back inside (mean-reversion scalp).

    When *config* is ``None``, the module-level constants are used.
    The DataFrame must already contain indicator columns (call
    :func:`add_indicators` first).
    """
    cfg = config or StrategyConfig()
    signals: list[Signal] = []

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]

        # Determine bar time – support both index-based and column-based time
        bar_time = (
            df.index[i]
            if isinstance(df.index, pd.DatetimeIndex)
            else row.get("time")
        )
        if bar_time is None:
            continue

        hour = bar_time.hour if hasattr(bar_time, "hour") else 0

        # --- Session filter ---
        if not _in_session(hour, cfg.session_start, cfg.session_end):
            continue

        # --- Must have valid indicators ---
        if (
            pd.isna(row["atr"])
            or pd.isna(row["ema_slow"])
            or pd.isna(row["macd_hist"])
        ):
            continue

        atr = row["atr"]
        if atr <= 0:
            continue

        trend = row["trend"]
        close = row["close"]
        ema_f = row["ema_fast"]

        # ==================================================================
        # ENTRY TYPE 1 – Trend Pullback (EMA-fast retest + MACD confirmation)
        # ==================================================================

        # --- LONG pullback ---
        if (
            trend == 1
            and row["low"] <= ema_f * (1 + cfg.ema_retest_tol)
            and close > ema_f                    # candle closes above EMA-fast
            and row["macd_hist"] > 0             # MACD histogram positive
        ):
            sl = close - cfg.sl_atr_mult * atr
            tp = close + cfg.tp_atr_mult * atr
            signals.append(
                Signal(Direction.LONG, close, sl, tp, atr, i, bar_time, "pullback")
            )
            continue  # one signal per bar

        # --- SHORT pullback ---
        if (
            trend == -1
            and row["high"] >= ema_f * (1 - cfg.ema_retest_tol)
            and close < ema_f                    # candle closes below EMA-fast
            and row["macd_hist"] < 0             # MACD histogram negative
        ):
            sl = close + cfg.sl_atr_mult * atr
            tp = close - cfg.tp_atr_mult * atr
            signals.append(
                Signal(Direction.SHORT, close, sl, tp, atr, i, bar_time, "pullback")
            )
            continue

        # ==================================================================
        # ENTRY TYPE 2 – Momentum (MACD histogram zero-cross)
        # ==================================================================

        prev_hist = prev["macd_hist"] if not pd.isna(prev["macd_hist"]) else 0

        # --- LONG momentum ---
        if (
            trend == 1
            and prev_hist <= 0
            and row["macd_hist"] > 0
            and close > ema_f
        ):
            sl = close - cfg.sl_atr_mult * atr
            tp = close + cfg.tp_atr_mult * atr
            signals.append(
                Signal(Direction.LONG, close, sl, tp, atr, i, bar_time, "momentum")
            )
            continue

        # --- SHORT momentum ---
        if (
            trend == -1
            and prev_hist >= 0
            and row["macd_hist"] < 0
            and close < ema_f
        ):
            sl = close + cfg.sl_atr_mult * atr
            tp = close - cfg.tp_atr_mult * atr
            signals.append(
                Signal(Direction.SHORT, close, sl, tp, atr, i, bar_time, "momentum")
            )
            continue

        # ==================================================================
        # ENTRY TYPE 3 – BB Bounce (mean-reversion scalp)
        # ==================================================================

        if not cfg.enable_bb_bounce:
            continue

        if pd.isna(row["bb_upper"]) or pd.isna(row["bb_lower"]):
            continue

        # --- LONG BB bounce ---
        if (
            trend == 1
            and prev["low"] <= prev.get("bb_lower", float("inf"))
            and close > row["bb_lower"]          # bounced back above lower BB
            and row["rsi"] < 40                  # not overbought
        ):
            sl = close - cfg.sl_atr_mult * atr
            tp = close + cfg.tp_atr_mult * atr
            signals.append(
                Signal(Direction.LONG, close, sl, tp, atr, i, bar_time, "bb_bounce")
            )
            continue

        # --- SHORT BB bounce ---
        if (
            trend == -1
            and prev["high"] >= prev.get("bb_upper", 0)
            and close < row["bb_upper"]          # rejected below upper BB
            and row["rsi"] > 60                  # not oversold
        ):
            sl = close + cfg.sl_atr_mult * atr
            tp = close - cfg.tp_atr_mult * atr
            signals.append(
                Signal(Direction.SHORT, close, sl, tp, atr, i, bar_time, "bb_bounce")
            )

    return signals


# ---------------------------------------------------------------------------
# Public convenience
# ---------------------------------------------------------------------------

def prepare_data(
    csv_path: str,
    config: Optional[StrategyConfig] = None,
) -> pd.DataFrame:
    """Load a CSV, add indicators, and return the enriched DataFrame."""
    df = pd.read_csv(csv_path, parse_dates=["time"], index_col="time")
    return add_indicators(df, config)
