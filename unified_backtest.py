"""
unified_backtest.py

Unified 1-month backtest engine that runs all 3 strategy parameter versions
on both EURUSD and XAUUSD with detailed logging including slippage and
spread simulation.

Features
--------
* 3 strategy presets (Aggressive / Balanced / Conservative).
* 2 symbols (EURUSD + XAUUSD).
* Fixed 0.05 lot, $50 starting balance.
* Simulated slippage (random within broker-realistic bounds).
* Spread deducted from every entry.
* Per-trade CSV log: entry/exit time, prices, slippage, spread, P&L, duration.
* JSON summary comparing all versions.
* 1-month backtest window (last ~30 calendar days of available data).

Usage
-----
    python unified_backtest.py                     # all presets, all symbols
    python unified_backtest.py --version V1        # single preset
    python unified_backtest.py --days 30           # custom window (days)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from strategy import (
    ALL_CONFIGS,
    ALL_SYMBOLS,
    EURUSD_SPEC,
    XAUUSD_SPEC,
    Direction,
    Signal,
    StrategyConfig,
    SymbolSpec,
    V1_AGGRESSIVE,
    V2_BALANCED,
    V3_CONSERVATIVE,
    add_indicators,
    generate_signals,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LOT_SIZE = 0.05
INITIAL_BALANCE = 50.0
RESULTS_DIR = Path("results")
RANDOM_SEED = 42       # reproducible slippage simulation


# ---------------------------------------------------------------------------
# Detailed trade result
# ---------------------------------------------------------------------------

@dataclass
class DetailedTrade:
    """One backtest trade with full slippage / spread / timing detail."""

    trade_no: int
    symbol: str
    version: str
    direction: str              # "LONG" / "SHORT"
    signal_type: str            # "pullback" / "momentum" / "bb_bounce"

    # Prices
    signal_price: float         # raw signal entry price
    entry_price: float          # after spread + slippage
    exit_price: float
    stop_loss: float
    take_profit: float
    atr: float

    # Costs
    spread_pips: float
    spread_usd: float
    slippage_entry_pips: float
    slippage_entry_usd: float
    slippage_exit_pips: float
    slippage_exit_usd: float
    total_cost_usd: float

    # Timing
    entry_time: str
    exit_time: str
    duration_bars: int

    # P&L
    gross_pnl_pips: float
    gross_pnl_usd: float
    net_pnl_pips: float
    net_pnl_usd: float

    # Running state
    balance_after: float
    lot_size: float


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def _simulate_slippage(max_pips: float, rng: random.Random) -> float:
    """Return a random slippage in pips (always adverse, i.e. positive)."""
    return round(rng.uniform(0, max_pips), 2)


def simulate_trade(
    signal: Signal,
    df: pd.DataFrame,
    equity: float,
    sym: SymbolSpec,
    rng: random.Random,
    trade_no: int,
    version_name: str,
) -> DetailedTrade | None:
    """Walk bar-by-bar and determine the outcome, logging slippage & spread."""

    pip = sym.pip_size
    ppv = sym.pip_value_per_lot * LOT_SIZE  # USD per pip for our lot

    # --- Spread ---
    spread_col = df.iloc[signal.bar_index].get("spread")
    if spread_col is not None and not pd.isna(spread_col) and float(spread_col) > 0:
        spread_pips = float(spread_col)
    else:
        spread_pips = sym.default_spread_pips
    spread_price = spread_pips * pip
    spread_usd = spread_pips * ppv

    # --- Entry slippage ---
    slip_entry_pips = _simulate_slippage(sym.max_slippage_pips, rng)
    slip_entry_price = slip_entry_pips * pip
    slip_entry_usd = slip_entry_pips * ppv

    # Actual entry (worse for us)
    if signal.direction == Direction.LONG:
        actual_entry = signal.entry_price + spread_price / 2 + slip_entry_price
    else:
        actual_entry = signal.entry_price - spread_price / 2 - slip_entry_price

    # --- Walk forward ---
    start_idx = signal.bar_index + 1
    if start_idx >= len(df):
        return None

    exit_price = None
    exit_bar_idx = None
    hit_sl = False

    for j in range(start_idx, len(df)):
        bar = df.iloc[j]

        if signal.direction == Direction.LONG:
            if bar["low"] <= signal.stop_loss:
                exit_price = signal.stop_loss
                exit_bar_idx = j
                hit_sl = True
                break
            if bar["high"] >= signal.take_profit:
                exit_price = signal.take_profit
                exit_bar_idx = j
                break
        else:
            if bar["high"] >= signal.stop_loss:
                exit_price = signal.stop_loss
                exit_bar_idx = j
                hit_sl = True
                break
            if bar["low"] <= signal.take_profit:
                exit_price = signal.take_profit
                exit_bar_idx = j
                break

    if exit_price is None:
        return None

    # --- Exit slippage (only on SL, TP is limit order → no slippage) ---
    slip_exit_pips = _simulate_slippage(sym.max_slippage_pips, rng) if hit_sl else 0.0
    slip_exit_price = slip_exit_pips * pip
    slip_exit_usd = slip_exit_pips * ppv

    if signal.direction == Direction.LONG:
        actual_exit = exit_price - slip_exit_price  # worse for longs
        # Gross = raw signal move (no costs)
        gross_pnl_pips = (exit_price - signal.entry_price) / pip
        # Net = actual prices (includes spread + all slippage)
        net_pnl_pips = (actual_exit - actual_entry) / pip
    else:
        actual_exit = exit_price + slip_exit_price  # worse for shorts
        gross_pnl_pips = (signal.entry_price - exit_price) / pip
        net_pnl_pips = (actual_entry - actual_exit) / pip

    gross_pnl_usd = gross_pnl_pips * ppv
    net_pnl_usd = net_pnl_pips * ppv

    total_cost = gross_pnl_usd - net_pnl_usd

    # Timing
    entry_time = signal.bar_time
    exit_time_val = (
        df.index[exit_bar_idx]
        if isinstance(df.index, pd.DatetimeIndex)
        else df.iloc[exit_bar_idx].get("time", "")
    )
    duration_bars = exit_bar_idx - signal.bar_index

    balance_after = equity + net_pnl_usd

    return DetailedTrade(
        trade_no=trade_no,
        symbol=sym.name,
        version=version_name,
        direction="LONG" if signal.direction == Direction.LONG else "SHORT",
        signal_type=signal.signal_type,
        signal_price=round(signal.entry_price, 5),
        entry_price=round(actual_entry, 5),
        exit_price=round(actual_exit, 5),
        stop_loss=round(signal.stop_loss, 5),
        take_profit=round(signal.take_profit, 5),
        atr=round(signal.atr, 5),
        spread_pips=round(spread_pips, 2),
        spread_usd=round(spread_usd, 4),
        slippage_entry_pips=round(slip_entry_pips, 2),
        slippage_entry_usd=round(slip_entry_usd, 4),
        slippage_exit_pips=round(slip_exit_pips, 2),
        slippage_exit_usd=round(slip_exit_usd, 4),
        total_cost_usd=round(total_cost, 4),
        entry_time=str(entry_time),
        exit_time=str(exit_time_val),
        duration_bars=duration_bars,
        gross_pnl_pips=round(gross_pnl_pips, 2),
        gross_pnl_usd=round(gross_pnl_usd, 4),
        net_pnl_pips=round(net_pnl_pips, 2),
        net_pnl_usd=round(net_pnl_usd, 4),
        balance_after=round(balance_after, 2),
        lot_size=LOT_SIZE,
    )


# ---------------------------------------------------------------------------
# Run backtest for one (config, symbol) pair
# ---------------------------------------------------------------------------

def run_single_backtest(
    df_raw: pd.DataFrame,
    cfg: StrategyConfig,
    sym: SymbolSpec,
    seed: int = RANDOM_SEED,
) -> list[DetailedTrade]:
    """Run a backtest on *df_raw* with the given config and symbol spec."""

    rng = random.Random(seed)
    df = add_indicators(df_raw, cfg)
    signals = generate_signals(df, cfg)

    trades: list[DetailedTrade] = []
    equity = INITIAL_BALANCE
    last_exit_idx = -1
    trade_no = 0

    for sig in signals:
        if sig.bar_index <= last_exit_idx:
            continue

        # Margin check – skip if balance too low
        if equity < 5.0:
            break

        trade_no += 1
        result = simulate_trade(sig, df, equity, sym, rng, trade_no, cfg.name)
        if result is None:
            trade_no -= 1
            continue

        trades.append(result)
        equity = result.balance_after

        # Find exit bar for sequencing
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
# Summary computation
# ---------------------------------------------------------------------------

@dataclass
class BacktestSummary:
    version: str
    symbol: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    gross_pnl_pips: float
    gross_pnl_usd: float
    net_pnl_pips: float
    net_pnl_usd: float
    total_spread_usd: float
    total_slippage_usd: float
    total_cost_usd: float
    max_drawdown_usd: float
    profit_factor: float
    avg_trade_duration_bars: float
    initial_balance: float
    final_balance: float
    trades_by_type: dict = field(default_factory=dict)


def compute_summary(
    trades: list[DetailedTrade],
    version: str,
    symbol: str,
) -> BacktestSummary:
    """Compute statistics from a list of detailed trades."""

    if not trades:
        return BacktestSummary(
            version=version, symbol=symbol,
            total_trades=0, winning_trades=0, losing_trades=0, win_rate=0.0,
            gross_pnl_pips=0.0, gross_pnl_usd=0.0,
            net_pnl_pips=0.0, net_pnl_usd=0.0,
            total_spread_usd=0.0, total_slippage_usd=0.0, total_cost_usd=0.0,
            max_drawdown_usd=0.0, profit_factor=0.0,
            avg_trade_duration_bars=0.0,
            initial_balance=INITIAL_BALANCE,
            final_balance=INITIAL_BALANCE,
        )

    wins = [t for t in trades if t.net_pnl_usd > 0]
    losses = [t for t in trades if t.net_pnl_usd <= 0]

    gross_profit = sum(t.net_pnl_usd for t in wins) if wins else 0.0
    gross_loss = abs(sum(t.net_pnl_usd for t in losses)) if losses else 0.0

    # Max drawdown
    eq = INITIAL_BALANCE
    peak = eq
    max_dd = 0.0
    for t in trades:
        eq += t.net_pnl_usd
        if eq > peak:
            peak = eq
        dd = peak - eq
        if dd > max_dd:
            max_dd = dd

    # Trades by type
    types: dict[str, int] = {}
    for t in trades:
        types[t.signal_type] = types.get(t.signal_type, 0) + 1

    return BacktestSummary(
        version=version,
        symbol=symbol,
        total_trades=len(trades),
        winning_trades=len(wins),
        losing_trades=len(losses),
        win_rate=round(len(wins) / len(trades) * 100, 1),
        gross_pnl_pips=round(sum(t.gross_pnl_pips for t in trades), 2),
        gross_pnl_usd=round(sum(t.gross_pnl_usd for t in trades), 2),
        net_pnl_pips=round(sum(t.net_pnl_pips for t in trades), 2),
        net_pnl_usd=round(sum(t.net_pnl_usd for t in trades), 2),
        total_spread_usd=round(sum(t.spread_usd for t in trades), 2),
        total_slippage_usd=round(
            sum(t.slippage_entry_usd + t.slippage_exit_usd for t in trades), 2
        ),
        total_cost_usd=round(sum(t.total_cost_usd for t in trades), 2),
        max_drawdown_usd=round(max_dd, 2),
        profit_factor=round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0.0,
        avg_trade_duration_bars=round(
            sum(t.duration_bars for t in trades) / len(trades), 1
        ),
        initial_balance=INITIAL_BALANCE,
        final_balance=round(trades[-1].balance_after, 2),
        trades_by_type=types,
    )


# ---------------------------------------------------------------------------
# Reporting & saving
# ---------------------------------------------------------------------------

def print_summary(s: BacktestSummary) -> None:
    """Pretty-print one summary block."""
    print(f"\n{'=' * 70}")
    print(f"  {s.version}  |  {s.symbol}  |  ${s.initial_balance:.0f} → ${s.final_balance:.2f}")
    print(f"{'=' * 70}")
    print(f"  Toplam işlem       : {s.total_trades}")
    print(f"  Kazanan / Kaybeden : {s.winning_trades} / {s.losing_trades}")
    print(f"  Kazanma oranı      : {s.win_rate:.1f} %")
    print(f"  Brüt K/Z (pip)     : {s.gross_pnl_pips:+.1f}")
    print(f"  Brüt K/Z (USD)     : {s.gross_pnl_usd:+.2f}")
    print(f"  Net K/Z (pip)      : {s.net_pnl_pips:+.1f}")
    print(f"  Net K/Z (USD)      : {s.net_pnl_usd:+.2f}")
    print(f"  Toplam spread      : ${s.total_spread_usd:.2f}")
    print(f"  Toplam slippage    : ${s.total_slippage_usd:.2f}")
    print(f"  Toplam maliyet     : ${s.total_cost_usd:.2f}")
    print(f"  Maks. drawdown     : ${s.max_drawdown_usd:.2f}")
    print(f"  Profit factor      : {s.profit_factor}")
    print(f"  Ort. süre (bar)    : {s.avg_trade_duration_bars:.1f}")
    print(f"  Sinyal dağılımı    : {s.trades_by_type}")
    print(f"{'=' * 70}")


def print_trade_log(trades: list[DetailedTrade], max_rows: int = 200) -> None:
    """Print individual trades with slippage/spread detail."""
    if not trades:
        print("  (işlem yok)")
        return

    header = (
        f"  {'#':>3}  {'Yön':>5}  {'Tip':>10}  {'Giriş':>10}  {'Çıkış':>10}  "
        f"{'SL':>10}  {'TP':>10}  {'Spread':>6}  {'Slip':>5}  "
        f"{'Net pip':>8}  {'Net $':>8}  {'Bakiye':>8}  {'Süre':>4}  Giriş Zamanı"
    )
    print(header)
    print(f"  {'—' * (len(header) - 2)}")

    price_fmt = ".5f" if trades[0].symbol == "EURUSD" else ".2f"

    for t in trades[:max_rows]:
        print(
            f"  {t.trade_no:>3}  {t.direction:>5}  {t.signal_type:>10}  "
            f"{t.entry_price:{price_fmt}}  {t.exit_price:{price_fmt}}  "
            f"{t.stop_loss:{price_fmt}}  {t.take_profit:{price_fmt}}  "
            f"{t.spread_pips:>6.1f}  {t.slippage_entry_pips:>5.1f}  "
            f"{t.net_pnl_pips:>+8.1f}  {t.net_pnl_usd:>+8.2f}  "
            f"{t.balance_after:>8.2f}  {t.duration_bars:>4}  {t.entry_time}"
        )
    if len(trades) > max_rows:
        print(f"  ... ve {len(trades) - max_rows} işlem daha")


def save_trades_csv(trades: list[DetailedTrade], path: Path) -> None:
    """Save detailed trade log to CSV."""
    if not trades:
        return
    rows = [
        {
            "trade_no": t.trade_no,
            "symbol": t.symbol,
            "version": t.version,
            "direction": t.direction,
            "signal_type": t.signal_type,
            "signal_price": t.signal_price,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "stop_loss": t.stop_loss,
            "take_profit": t.take_profit,
            "atr": t.atr,
            "spread_pips": t.spread_pips,
            "spread_usd": t.spread_usd,
            "slippage_entry_pips": t.slippage_entry_pips,
            "slippage_entry_usd": t.slippage_entry_usd,
            "slippage_exit_pips": t.slippage_exit_pips,
            "slippage_exit_usd": t.slippage_exit_usd,
            "total_cost_usd": t.total_cost_usd,
            "entry_time": t.entry_time,
            "exit_time": t.exit_time,
            "duration_bars": t.duration_bars,
            "gross_pnl_pips": t.gross_pnl_pips,
            "gross_pnl_usd": t.gross_pnl_usd,
            "net_pnl_pips": t.net_pnl_pips,
            "net_pnl_usd": t.net_pnl_usd,
            "balance_after": t.balance_after,
            "lot_size": t.lot_size,
        }
        for t in trades
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def save_summary_json(summaries: list[BacktestSummary], path: Path) -> None:
    """Save all summaries as a single JSON file."""
    data = []
    for s in summaries:
        d = {
            "version": s.version,
            "symbol": s.symbol,
            "total_trades": s.total_trades,
            "winning_trades": s.winning_trades,
            "losing_trades": s.losing_trades,
            "win_rate": s.win_rate,
            "gross_pnl_pips": s.gross_pnl_pips,
            "gross_pnl_usd": s.gross_pnl_usd,
            "net_pnl_pips": s.net_pnl_pips,
            "net_pnl_usd": s.net_pnl_usd,
            "total_spread_usd": s.total_spread_usd,
            "total_slippage_usd": s.total_slippage_usd,
            "total_cost_usd": s.total_cost_usd,
            "max_drawdown_usd": s.max_drawdown_usd,
            "profit_factor": s.profit_factor,
            "avg_trade_duration_bars": s.avg_trade_duration_bars,
            "initial_balance": s.initial_balance,
            "final_balance": s.final_balance,
            "trades_by_type": s.trades_by_type,
        }
        data.append(d)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Data loading with 1-month window
# ---------------------------------------------------------------------------

def load_symbol_data(sym: SymbolSpec, days: int = 30) -> pd.DataFrame | None:
    """Load CSV data and slice to the last *days* calendar days.

    Returns ``None`` if the CSV file does not exist.
    """
    csv = Path(sym.csv_path)
    if not csv.exists():
        print(f"  ⚠️  {csv} bulunamadı – {sym.name} atlanıyor.")
        print(f"      Veriyi çekmek için:  python fetch_prices.py {sym.name}")
        return None
    df = pd.read_csv(csv, parse_dates=["time"], index_col="time")
    if len(df) == 0:
        return None

    # Slice last N days
    end = df.index[-1]
    start = end - timedelta(days=days)
    df_sliced = df.loc[start:]
    if len(df_sliced) < 50:
        print(f"  ⚠️  {sym.name}: son {days} gün için yeterli veri yok ({len(df_sliced)} bar)")
        return None
    return df_sliced


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

VERSION_MAP: dict[str, StrategyConfig] = {
    "V1": V1_AGGRESSIVE,
    "V2": V2_BALANCED,
    "V3": V3_CONSERVATIVE,
}


def main(
    version: str | None = None,
    days: int = 30,
) -> None:
    """Run the unified backtest and save results."""

    configs = [VERSION_MAP[version]] if version else ALL_CONFIGS
    symbols = ALL_SYMBOLS

    RESULTS_DIR.mkdir(exist_ok=True)

    all_summaries: list[BacktestSummary] = []
    all_trades: list[DetailedTrade] = []
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    print("=" * 70)
    print(f"  UNIFIED BACKTEST  |  Bakiye: ${INITIAL_BALANCE}  |  Lot: {LOT_SIZE}")
    print(f"  Pencere: son {days} gün  |  Zaman: {timestamp}")
    print("=" * 70)

    for sym in symbols:
        df_raw = load_symbol_data(sym, days)
        if df_raw is None:
            continue

        print(f"\n📊 {sym.name}: {len(df_raw)} bar  "
              f"({df_raw.index[0]} → {df_raw.index[-1]})")

        for cfg in configs:
            print(f"\n  🔧 {cfg.name} çalıştırılıyor ...")
            trades = run_single_backtest(df_raw, cfg, sym)
            summary = compute_summary(trades, cfg.name, sym.name)
            all_summaries.append(summary)
            all_trades.extend(trades)

            print_summary(summary)
            print_trade_log(trades)

            # Save per-version per-symbol CSV
            csv_name = f"backtest_{cfg.name}_{sym.name}_{timestamp}.csv"
            save_trades_csv(trades, RESULTS_DIR / csv_name)
            print(f"\n  💾 İşlem logu: {RESULTS_DIR / csv_name}")

    # Save combined summary
    if all_summaries:
        summary_path = RESULTS_DIR / f"backtest_summary_{timestamp}.json"
        save_summary_json(all_summaries, summary_path)
        print(f"\n💾 Özet: {summary_path}")

    # Save all trades combined
    if all_trades:
        all_csv = RESULTS_DIR / f"backtest_all_trades_{timestamp}.csv"
        save_trades_csv(all_trades, all_csv)
        print(f"💾 Tüm işlemler: {all_csv}")

    # --- Comparison table ---
    if len(all_summaries) > 1:
        print(f"\n\n{'=' * 90}")
        print("  KARŞILAŞTIRMA TABLOSU")
        print(f"{'=' * 90}")
        print(
            f"  {'Versiyon':<18} {'Sembol':<8} {'İşlem':>6} {'Kazanma':>8} "
            f"{'Net pip':>9} {'Net $':>9} {'Maliyet':>9} {'DD':>9} {'PF':>6} {'Bakiye':>9}"
        )
        print(f"  {'—' * 86}")
        for s in all_summaries:
            print(
                f"  {s.version:<18} {s.symbol:<8} {s.total_trades:>6} "
                f"{s.win_rate:>7.1f}% {s.net_pnl_pips:>+9.1f} "
                f"{s.net_pnl_usd:>+9.2f} {s.total_cost_usd:>9.2f} "
                f"{s.max_drawdown_usd:>9.2f} {s.profit_factor:>6.2f} "
                f"{s.final_balance:>9.2f}"
            )
        print(f"{'=' * 90}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified multi-version backtest")
    p.add_argument(
        "--version", choices=["V1", "V2", "V3"], default=None,
        help="Run only this version (default: all)",
    )
    p.add_argument(
        "--days", type=int, default=30,
        help="Backtest window in calendar days (default: 30)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(version=args.version, days=args.days)
