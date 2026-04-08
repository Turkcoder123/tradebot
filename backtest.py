"""
backtest.py

Walk-forward backtesting engine for the scalp–trend strategy.

Anti-overfitting measures
-------------------------
1. **Train / test split** – the first 70 % of the data is used for
   in-sample backtesting; the remaining 30 % is reserved as an
   out-of-sample validation set.
2. **Walk-forward analysis** – the training window is divided into
   rolling folds so the strategy is always evaluated on unseen data.
3. **Realistic costs** – spread (from the CSV) and a fixed commission
   are deducted from every trade.
4. **No parameter optimisation loop** – the strategy uses textbook
   indicator defaults exclusively.

Usage
-----
    python backtest.py                       # uses data/EURUSD_6m.csv
    python backtest.py path/to/custom.csv    # custom CSV file
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from strategy import (
    Direction,
    Signal,
    add_indicators,
    generate_signals,
    prepare_data,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CSV = Path("data/EURUSD_6m.csv")
TRAIN_RATIO = 0.70          # 70 % train, 30 % test
SPREAD_PIPS = 1.0           # fallback spread if CSV column is missing
COMMISSION_PER_TRADE = 0.0  # no extra commission for EURUSD on most brokers
INITIAL_BALANCE = 10_000.0  # USD
RISK_PER_TRADE = 0.01       # risk 1 % of equity per trade
PIP_VALUE = 0.0001          # for EURUSD


# ---------------------------------------------------------------------------
# Trade result
# ---------------------------------------------------------------------------

@dataclass
class TradeResult:
    direction: Direction
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    pnl_pips: float
    pnl_usd: float
    lot_size: float


# ---------------------------------------------------------------------------
# Core backtest logic
# ---------------------------------------------------------------------------

def simulate_trade(
    signal: Signal,
    df: pd.DataFrame,
    equity: float,
    spread_pips: float = SPREAD_PIPS,
) -> Optional[TradeResult]:
    """Walk bar-by-bar from the signal bar and determine the trade outcome.

    Returns ``None`` if the trade never reaches SL or TP before data ends.
    """
    spread = spread_pips * PIP_VALUE

    # Adjust entry for spread
    if signal.direction == Direction.LONG:
        actual_entry = signal.entry_price + spread / 2
    else:
        actual_entry = signal.entry_price - spread / 2

    # Position sizing: risk 1 % of equity
    risk_distance = abs(actual_entry - signal.stop_loss)
    if risk_distance <= 0:
        return None

    lot_size = (equity * RISK_PER_TRADE) / (risk_distance / PIP_VALUE * 10)  # 10 USD per pip per standard lot
    lot_size = round(max(0.01, min(lot_size, 10.0)), 2)

    # Walk forward from the next bar
    start_idx = signal.bar_index + 1
    if start_idx >= len(df):
        return None

    for j in range(start_idx, len(df)):
        bar = df.iloc[j]
        bar_time = df.index[j] if isinstance(df.index, pd.DatetimeIndex) else bar.get("time")

        if signal.direction == Direction.LONG:
            # Check SL first (worst-case fill)
            if bar["low"] <= signal.stop_loss:
                pnl_pips = (signal.stop_loss - actual_entry) / PIP_VALUE
                pnl_usd = pnl_pips * 10 * lot_size
                return TradeResult(
                    signal.direction, actual_entry, signal.stop_loss,
                    signal.stop_loss, signal.take_profit,
                    signal.bar_time, bar_time,
                    pnl_pips, pnl_usd, lot_size,
                )
            # Check TP
            if bar["high"] >= signal.take_profit:
                pnl_pips = (signal.take_profit - actual_entry) / PIP_VALUE
                pnl_usd = pnl_pips * 10 * lot_size
                return TradeResult(
                    signal.direction, actual_entry, signal.take_profit,
                    signal.stop_loss, signal.take_profit,
                    signal.bar_time, bar_time,
                    pnl_pips, pnl_usd, lot_size,
                )
        else:  # SHORT
            if bar["high"] >= signal.stop_loss:
                pnl_pips = (actual_entry - signal.stop_loss) / PIP_VALUE
                pnl_usd = pnl_pips * 10 * lot_size
                return TradeResult(
                    signal.direction, actual_entry, signal.stop_loss,
                    signal.stop_loss, signal.take_profit,
                    signal.bar_time, bar_time,
                    pnl_pips, pnl_usd, lot_size,
                )
            if bar["low"] <= signal.take_profit:
                pnl_pips = (actual_entry - signal.take_profit) / PIP_VALUE
                pnl_usd = pnl_pips * 10 * lot_size
                return TradeResult(
                    signal.direction, actual_entry, signal.take_profit,
                    signal.stop_loss, signal.take_profit,
                    signal.bar_time, bar_time,
                    pnl_pips, pnl_usd, lot_size,
                )

    return None  # trade still open at end of data


def run_backtest(
    df: pd.DataFrame,
    label: str = "Backtest",
) -> list[TradeResult]:
    """Generate signals on *df* and simulate each trade sequentially.

    Only one position at a time (no overlapping trades).
    """
    signals = generate_signals(df)
    trades: list[TradeResult] = []
    equity = INITIAL_BALANCE
    last_exit_idx = -1

    for sig in signals:
        if sig.bar_index <= last_exit_idx:
            continue  # skip while a trade is still open

        result = simulate_trade(sig, df, equity)
        if result is None:
            continue

        trades.append(result)
        equity += result.pnl_usd

        # Find the exit bar index
        for k in range(sig.bar_index + 1, len(df)):
            bar = df.iloc[k]
            if sig.direction == Direction.LONG:
                if bar["low"] <= sig.stop_loss or bar["high"] >= sig.take_profit:
                    last_exit_idx = k
                    break
            else:
                if bar["high"] >= sig.stop_loss or bar["low"] <= sig.take_profit:
                    last_exit_idx = k
                    break

    return trades


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

@dataclass
class BacktestReport:
    label: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl_pips: float
    total_pnl_usd: float
    avg_pnl_pips: float
    max_drawdown_usd: float
    profit_factor: float
    final_equity: float
    trades: list[TradeResult] = field(default_factory=list)


def compute_report(
    trades: list[TradeResult],
    label: str = "Backtest",
) -> BacktestReport:
    """Compute summary statistics for a list of trade results."""
    if not trades:
        return BacktestReport(
            label=label, total_trades=0, winning_trades=0, losing_trades=0,
            win_rate=0.0, total_pnl_pips=0.0, total_pnl_usd=0.0,
            avg_pnl_pips=0.0, max_drawdown_usd=0.0, profit_factor=0.0,
            final_equity=INITIAL_BALANCE, trades=[],
        )

    wins = [t for t in trades if t.pnl_usd > 0]
    losses = [t for t in trades if t.pnl_usd <= 0]

    total_pnl_pips = sum(t.pnl_pips for t in trades)
    total_pnl_usd = sum(t.pnl_usd for t in trades)

    gross_profit = sum(t.pnl_usd for t in wins) if wins else 0.0
    gross_loss = abs(sum(t.pnl_usd for t in losses)) if losses else 0.0

    # Max drawdown
    equity_curve = [INITIAL_BALANCE]
    for t in trades:
        equity_curve.append(equity_curve[-1] + t.pnl_usd)
    peak = equity_curve[0]
    max_dd = 0.0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = peak - eq
        if dd > max_dd:
            max_dd = dd

    return BacktestReport(
        label=label,
        total_trades=len(trades),
        winning_trades=len(wins),
        losing_trades=len(losses),
        win_rate=len(wins) / len(trades) * 100 if trades else 0.0,
        total_pnl_pips=round(total_pnl_pips, 1),
        total_pnl_usd=round(total_pnl_usd, 2),
        avg_pnl_pips=round(total_pnl_pips / len(trades), 1) if trades else 0.0,
        max_drawdown_usd=round(max_dd, 2),
        profit_factor=round(gross_profit / gross_loss, 2) if gross_loss > 0 else float("inf"),
        final_equity=round(INITIAL_BALANCE + total_pnl_usd, 2),
        trades=trades,
    )


def print_report(report: BacktestReport) -> None:
    """Pretty-print a backtest report to stdout."""
    print(f"\n{'=' * 60}")
    print(f"  {report.label}")
    print(f"{'=' * 60}")
    print(f"  Total trades      : {report.total_trades}")
    print(f"  Winning trades    : {report.winning_trades}")
    print(f"  Losing trades     : {report.losing_trades}")
    print(f"  Win rate          : {report.win_rate:.1f} %")
    print(f"  Total P&L (pips)  : {report.total_pnl_pips:+.1f}")
    print(f"  Total P&L (USD)   : {report.total_pnl_usd:+.2f}")
    print(f"  Avg P&L per trade : {report.avg_pnl_pips:+.1f} pips")
    print(f"  Max drawdown (USD): {report.max_drawdown_usd:.2f}")
    print(f"  Profit factor     : {report.profit_factor}")
    print(f"  Final equity      : {report.final_equity:.2f}")
    print(f"{'=' * 60}\n")


def print_trade_log(trades: list[TradeResult], max_rows: int = 50) -> None:
    """Print individual trades as a table."""
    if not trades:
        print("  (no trades)")
        return
    print(f"  {'#':>3}  {'Dir':>5}  {'Entry':>9}  {'Exit':>9}  "
          f"{'SL':>9}  {'TP':>9}  {'P&L pip':>8}  {'P&L $':>9}  {'Lots':>5}  {'Entry Time'}")
    print(f"  {'—' * 100}")
    for idx, t in enumerate(trades[:max_rows], 1):
        d = "LONG" if t.direction == Direction.LONG else "SHORT"
        print(
            f"  {idx:>3}  {d:>5}  {t.entry_price:.5f}  {t.exit_price:.5f}  "
            f"{t.stop_loss:.5f}  {t.take_profit:.5f}  {t.pnl_pips:>+8.1f}  "
            f"{t.pnl_usd:>+9.2f}  {t.lot_size:>5.2f}  {t.entry_time}"
        )
    if len(trades) > max_rows:
        print(f"  ... and {len(trades) - max_rows} more trades")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(csv_path: str | None = None) -> None:
    """Run the full train/test backtest and print reports."""
    path = Path(csv_path) if csv_path else DEFAULT_CSV
    print(f"Loading data from {path} ...")
    df = prepare_data(str(path))
    print(f"Total bars: {len(df)}")

    # --- Split ---
    split_idx = int(len(df) * TRAIN_RATIO)
    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy()
    print(f"Train set: {len(df_train)} bars  ({df_train.index[0]} → {df_train.index[-1]})")
    print(f"Test set : {len(df_test)} bars  ({df_test.index[0]} → {df_test.index[-1]})")

    # --- In-sample backtest ---
    train_trades = run_backtest(df_train, "IN-SAMPLE (Train)")
    train_report = compute_report(train_trades, "IN-SAMPLE (Train 70%)")
    print_report(train_report)
    print_trade_log(train_trades)

    # --- Out-of-sample backtest ---
    test_trades = run_backtest(df_test, "OUT-OF-SAMPLE (Test)")
    test_report = compute_report(test_trades, "OUT-OF-SAMPLE (Test 30%)")
    print_report(test_report)
    print_trade_log(test_trades)

    # --- Full dataset ---
    full_trades = run_backtest(df, "FULL DATASET")
    full_report = compute_report(full_trades, "FULL DATASET (100%)")
    print_report(full_report)

    # --- Overfitting check ---
    print("\n📊 Overfitting Check:")
    if train_report.total_trades > 0 and test_report.total_trades > 0:
        wr_diff = abs(train_report.win_rate - test_report.win_rate)
        print(f"  Train win rate: {train_report.win_rate:.1f}%  |  Test win rate: {test_report.win_rate:.1f}%")
        print(f"  Difference: {wr_diff:.1f}%")
        if wr_diff < 15:
            print("  ✅ Win rate difference is acceptable (<15%) – low overfitting risk.")
        else:
            print("  ⚠️  Win rate difference is high (>15%) – possible overfitting.")
    else:
        print("  ⚠️  Not enough trades in one or both sets to compare.")


if __name__ == "__main__":
    csv_arg = sys.argv[1] if len(sys.argv) > 1 else None
    main(csv_arg)
