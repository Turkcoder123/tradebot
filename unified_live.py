"""
unified_live.py

1-day forward test bot that trades EURUSD and XAUUSD on a MetaTrader 5
demo account with detailed slippage / spread logging.

Features
--------
* Fixed 0.05 lot, $50 starting balance tracking.
* Trades both EURUSD and XAUUSD simultaneously.
* Selectable strategy version (V1 / V2 / V3) via ``--version``.
* Actual slippage recorded (requested price vs filled price).
* Actual spread recorded from MT5 tick data.
* Full trade log saved to ``results/live_trades_<timestamp>.csv``.
* JSON summary saved to ``results/live_summary_<timestamp>.json``.
* Auto-shutdown after ``--hours`` (default: 24).

Usage
-----
    python unified_live.py                       # V2 (balanced), 24 h
    python unified_live.py --version V1          # aggressive preset
    python unified_live.py --hours 8             # 8-hour session
    python unified_live.py --version V3 --hours 1
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

from strategy import (
    ALL_SYMBOLS,
    EURUSD_SPEC,
    XAUUSD_SPEC,
    Direction,
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
LOOKBACK_BARS = 300
CHECK_INTERVAL = 60       # seconds between checks
TIMEFRAME_MT5 = 16385     # mt5.TIMEFRAME_H1
DEVIATION = 20            # max price deviation (points)

RESULTS_DIR = Path("results")
LOG_DIR = Path("logs")

VERSION_MAP: dict[str, StrategyConfig] = {
    "V1": V1_AGGRESSIVE,
    "V2": V2_BALANCED,
    "V3": V3_CONSERVATIVE,
}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(version: str) -> logging.Logger:
    """Create a logger that writes to both file and console."""
    LOG_DIR.mkdir(exist_ok=True)

    logger = logging.getLogger("unified_live")
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(LOG_DIR / f"live_{version}_{ts}.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ---------------------------------------------------------------------------
# Detailed trade record
# ---------------------------------------------------------------------------

@dataclass
class LiveTrade:
    """Record of a completed live trade."""
    trade_no: int
    symbol: str
    version: str
    direction: str
    signal_type: str
    signal_price: float
    requested_price: float
    filled_price: float
    slippage_pips: float
    slippage_usd: float
    spread_pips: float
    spread_usd: float
    stop_loss: float
    take_profit: float
    atr: float
    entry_time: str
    exit_price: float
    exit_time: str
    exit_slippage_pips: float
    exit_slippage_usd: float
    gross_pnl_usd: float
    net_pnl_usd: float
    total_cost_usd: float
    balance_after: float
    lot_size: float
    ticket: int
    duration_seconds: int


# ---------------------------------------------------------------------------
# MT5 helpers
# ---------------------------------------------------------------------------

log: logging.Logger = logging.getLogger("unified_live")


def connect_mt5() -> None:
    if mt5 is None:
        raise RuntimeError("MetaTrader5 paketi yüklü değil.")
    if not mt5.initialize():
        raise RuntimeError(f"MT5 bağlantı hatası: {mt5.last_error()}")
    info = mt5.terminal_info()
    log.info("MT5 bağlantı: %s  build %s", info.name, info.build)
    acc = mt5.account_info()
    log.info(
        "Hesap #%s  bakiye=%.2f  equity=%.2f  leverage=%d",
        acc.login, acc.balance, acc.equity, acc.leverage,
    )


def disconnect_mt5() -> None:
    if mt5 is not None:
        mt5.shutdown()
    log.info("MT5 bağlantı kesildi.")


def fetch_bars(symbol: str, count: int = LOOKBACK_BARS) -> pd.DataFrame:
    rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME_MT5, 0, count)
    if rates is None or len(rates) == 0:
        raise ValueError(f"{symbol} bar verisi yok: {mt5.last_error()}")
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("time", inplace=True)
    return df


def get_magic(sym_name: str, version: str) -> int:
    """Generate a unique magic number per symbol+version."""
    base = 300000
    sym_offset = {"EURUSD": 1000, "XAUUSD": 2000}.get(sym_name, 3000)
    ver_offset = {"V1": 100, "V2": 200, "V3": 300}.get(version, 0)
    return base + sym_offset + ver_offset


def has_open_position(symbol: str, magic: int) -> bool:
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        return False
    return any(p.magic == magic for p in positions)


def get_open_position(symbol: str, magic: int):
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        return None
    for p in positions:
        if p.magic == magic:
            return p
    return None


def get_tick_spread(symbol: str, sym: SymbolSpec) -> tuple[float, float]:
    """Return (spread_pips, spread_usd) from the current tick."""
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return sym.default_spread_pips, sym.default_spread_pips * sym.pip_value_per_lot * LOT_SIZE
    raw_spread = tick.ask - tick.bid
    spread_pips = raw_spread / sym.pip_size
    ppv = sym.pip_value_per_lot * LOT_SIZE
    return round(spread_pips, 2), round(spread_pips * ppv, 4)


def place_order(
    direction: Direction,
    symbol: str,
    sl: float,
    tp: float,
    magic: int,
    sym: SymbolSpec,
    cfg_name: str,
) -> dict | None:
    """Place a market order and return fill details or None on failure."""
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        log.error("Tick verisi alınamadı: %s", symbol)
        return None

    if direction == Direction.LONG:
        order_type = mt5.ORDER_TYPE_BUY
        requested_price = tick.ask
    else:
        order_type = mt5.ORDER_TYPE_SELL
        requested_price = tick.bid

    # Round SL/TP
    sym_info = mt5.symbol_info(symbol)
    digits = sym_info.digits if sym_info else 5

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": LOT_SIZE,
        "type": order_type,
        "price": requested_price,
        "sl": round(sl, digits),
        "tp": round(tp, digits),
        "deviation": DEVIATION,
        "magic": magic,
        "comment": f"unified_{cfg_name}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result is None:
        log.error("order_send None: %s", mt5.last_error())
        return None
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        log.error("Emir başarısız – retcode=%d  yorum=%s", result.retcode, result.comment)
        return None

    filled_price = result.price
    slippage_price = abs(filled_price - requested_price)
    slippage_pips = slippage_price / sym.pip_size
    ppv = sym.pip_value_per_lot * LOT_SIZE
    slippage_usd = slippage_pips * ppv

    # Spread at fill
    spread_pips, spread_usd = get_tick_spread(symbol, sym)

    log.info(
        "✅ %s %s %.2f lot @ %.5f (talep=%.5f, kayma=%.1f pip/$%.4f, "
        "spread=%.1f pip/$%.4f)  SL=%.5f  TP=%.5f  ticket=%d",
        "BUY" if direction == Direction.LONG else "SELL",
        symbol, LOT_SIZE, filled_price, requested_price,
        slippage_pips, slippage_usd, spread_pips, spread_usd,
        round(sl, digits), round(tp, digits), result.order,
    )

    return {
        "ticket": result.order,
        "requested_price": requested_price,
        "filled_price": filled_price,
        "slippage_pips": round(slippage_pips, 2),
        "slippage_usd": round(slippage_usd, 4),
        "spread_pips": round(spread_pips, 2),
        "spread_usd": round(spread_usd, 4),
        "entry_time": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Trade monitoring
# ---------------------------------------------------------------------------

@dataclass
class OpenTrade:
    """Tracks an open position for result logging."""
    symbol: str
    sym: SymbolSpec
    cfg_name: str
    direction: Direction
    signal_type: str
    signal_price: float
    fill_info: dict
    sl: float
    tp: float
    atr: float
    magic: int
    trade_no: int


def check_closed_trades(
    open_trades: list[OpenTrade],
    equity: float,
) -> tuple[list[LiveTrade], list[OpenTrade], float]:
    """Check if any open trades have been closed by SL/TP.

    Returns (closed_trades, remaining_open, new_equity).
    """
    closed: list[LiveTrade] = []
    still_open: list[OpenTrade] = []

    for ot in open_trades:
        pos = get_open_position(ot.symbol, ot.magic)
        if pos is not None:
            # Still open — log current P&L
            log.debug(
                "Açık: %s %s ticket=%d  anlık K/Z=%.2f",
                ot.symbol, ot.cfg_name, ot.fill_info["ticket"], pos.profit,
            )
            still_open.append(ot)
            continue

        # Position gone — it was closed (by SL/TP or manually)
        ppv = ot.sym.pip_value_per_lot * LOT_SIZE

        # Fetch deal history for this ticket
        now = datetime.now(timezone.utc)
        deals = mt5.history_deals_get(
            now - timedelta(days=2), now,
            group=f"*{ot.symbol}*",
        )

        exit_price = 0.0
        exit_time = now.isoformat()
        gross_pnl = 0.0
        exit_slip_pips = 0.0
        exit_slip_usd = 0.0

        if deals:
            # Find the closing deal matching our ticket/magic
            for d in reversed(deals):
                if d.magic == ot.magic and d.position_id == ot.fill_info["ticket"]:
                    exit_price = d.price
                    exit_time = datetime.fromtimestamp(d.time, tz=timezone.utc).isoformat()
                    gross_pnl = d.profit
                    # Estimate exit slippage by comparing fill to nearest
                    # SL/TP level.  Assumes exits only via SL/TP; manual or
                    # broker-initiated closes will show larger apparent slippage.
                    if ot.direction == Direction.LONG:
                        expected_exit = ot.sl if exit_price <= ot.sl else ot.tp
                    else:
                        expected_exit = ot.sl if exit_price >= ot.sl else ot.tp
                    exit_slip_pips = abs(exit_price - expected_exit) / ot.sym.pip_size
                    exit_slip_usd = exit_slip_pips * ppv
                    break

        total_cost = (
            ot.fill_info["spread_usd"]
            + ot.fill_info["slippage_usd"]
            + exit_slip_usd
        )
        net_pnl = gross_pnl - total_cost
        equity += net_pnl

        entry_dt = datetime.fromisoformat(ot.fill_info["entry_time"])
        exit_dt = datetime.fromisoformat(exit_time)
        duration = int((exit_dt - entry_dt).total_seconds())

        lt = LiveTrade(
            trade_no=ot.trade_no,
            symbol=ot.symbol,
            version=ot.cfg_name,
            direction="LONG" if ot.direction == Direction.LONG else "SHORT",
            signal_type=ot.signal_type,
            signal_price=round(ot.signal_price, 5),
            requested_price=ot.fill_info["requested_price"],
            filled_price=ot.fill_info["filled_price"],
            slippage_pips=ot.fill_info["slippage_pips"],
            slippage_usd=ot.fill_info["slippage_usd"],
            spread_pips=ot.fill_info["spread_pips"],
            spread_usd=ot.fill_info["spread_usd"],
            stop_loss=round(ot.sl, 5),
            take_profit=round(ot.tp, 5),
            atr=round(ot.atr, 5),
            entry_time=ot.fill_info["entry_time"],
            exit_price=round(exit_price, 5),
            exit_time=exit_time,
            exit_slippage_pips=round(exit_slip_pips, 2),
            exit_slippage_usd=round(exit_slip_usd, 4),
            gross_pnl_usd=round(gross_pnl, 4),
            net_pnl_usd=round(net_pnl, 4),
            total_cost_usd=round(total_cost, 4),
            balance_after=round(equity, 2),
            lot_size=LOT_SIZE,
            ticket=ot.fill_info["ticket"],
            duration_seconds=duration,
        )
        closed.append(lt)

        log.info(
            "🔴 Kapandı: %s %s %s  brüt=%.2f  maliyet=%.4f  net=%.4f  bakiye=%.2f",
            ot.symbol, lt.direction, ot.cfg_name,
            gross_pnl, total_cost, net_pnl, equity,
        )

    return closed, still_open, equity


# ---------------------------------------------------------------------------
# Signal check for one symbol
# ---------------------------------------------------------------------------

def check_and_trade(
    sym: SymbolSpec,
    cfg: StrategyConfig,
    magic: int,
    lot: float,
    open_trades: list[OpenTrade],
    trade_counter: int,
) -> tuple[OpenTrade | None, int]:
    """Check for signals and place an order if appropriate.

    Returns (new_open_trade_or_None, updated_counter).
    """
    if has_open_position(sym.name, magic):
        return None, trade_counter

    try:
        df = fetch_bars(sym.name)
    except ValueError as exc:
        log.warning("fetch_bars: %s", exc)
        return None, trade_counter

    df = add_indicators(df, cfg)
    if len(df) < 5:
        return None, trade_counter

    tail = df.iloc[-5:].copy()
    signals = generate_signals(tail, cfg)

    if not signals:
        log.debug("Sinyal yok: %s %s", sym.name, cfg.name)
        return None, trade_counter

    sig = signals[-1]

    # Use live tick for SL/TP
    tick = mt5.symbol_info_tick(sym.name)
    if tick is None:
        return None, trade_counter

    current = tick.ask if sig.direction == Direction.LONG else tick.bid
    atr = sig.atr

    if sig.direction == Direction.LONG:
        sl = current - cfg.sl_atr_mult * atr
        tp = current + cfg.tp_atr_mult * atr
    else:
        sl = current + cfg.sl_atr_mult * atr
        tp = current - cfg.tp_atr_mult * atr

    spread_pips, _ = get_tick_spread(sym.name, sym)
    log.info(
        "📈 Sinyal: %s %s %s  fiyat=%.5f  SL=%.5f  TP=%.5f  "
        "ATR=%.5f  spread=%.1f pip  tip=%s",
        sym.name, sig.direction.name, cfg.name,
        current, sl, tp, atr, spread_pips, sig.signal_type,
    )

    fill = place_order(sig.direction, sym.name, sl, tp, magic, sym, cfg.name)
    if fill is None:
        return None, trade_counter

    trade_counter += 1
    ot = OpenTrade(
        symbol=sym.name,
        sym=sym,
        cfg_name=cfg.name,
        direction=sig.direction,
        signal_type=sig.signal_type,
        signal_price=sig.entry_price,
        fill_info=fill,
        sl=sl,
        tp=tp,
        atr=atr,
        magic=magic,
        trade_no=trade_counter,
    )
    return ot, trade_counter


# ---------------------------------------------------------------------------
# Result saving
# ---------------------------------------------------------------------------

def save_live_results(
    trades: list[LiveTrade],
    cfg: StrategyConfig,
    start_time: datetime,
    final_equity: float,
) -> None:
    """Persist trade log and summary to disk."""
    RESULTS_DIR.mkdir(exist_ok=True)
    ts = start_time.strftime("%Y%m%d_%H%M%S")

    # CSV
    if trades:
        rows = []
        for t in trades:
            rows.append({
                "trade_no": t.trade_no, "symbol": t.symbol, "version": t.version,
                "direction": t.direction, "signal_type": t.signal_type,
                "signal_price": t.signal_price,
                "requested_price": t.requested_price,
                "filled_price": t.filled_price,
                "slippage_pips": t.slippage_pips, "slippage_usd": t.slippage_usd,
                "spread_pips": t.spread_pips, "spread_usd": t.spread_usd,
                "stop_loss": t.stop_loss, "take_profit": t.take_profit,
                "atr": t.atr,
                "entry_time": t.entry_time, "exit_time": t.exit_time,
                "exit_price": t.exit_price,
                "exit_slippage_pips": t.exit_slippage_pips,
                "exit_slippage_usd": t.exit_slippage_usd,
                "gross_pnl_usd": t.gross_pnl_usd,
                "net_pnl_usd": t.net_pnl_usd,
                "total_cost_usd": t.total_cost_usd,
                "balance_after": t.balance_after,
                "lot_size": t.lot_size,
                "ticket": t.ticket,
                "duration_seconds": t.duration_seconds,
            })
        csv_path = RESULTS_DIR / f"live_trades_{cfg.name}_{ts}.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        log.info("💾 İşlem logu: %s", csv_path)

    # JSON summary
    wins = [t for t in trades if t.net_pnl_usd > 0]
    losses = [t for t in trades if t.net_pnl_usd <= 0]
    summary = {
        "version": cfg.name,
        "start_time": start_time.isoformat(),
        "end_time": datetime.now(timezone.utc).isoformat(),
        "initial_balance": INITIAL_BALANCE,
        "final_balance": round(final_equity, 2),
        "total_trades": len(trades),
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "win_rate": round(len(wins) / len(trades) * 100, 1) if trades else 0.0,
        "net_pnl_usd": round(sum(t.net_pnl_usd for t in trades), 2),
        "total_spread_usd": round(sum(t.spread_usd for t in trades), 2),
        "total_slippage_usd": round(
            sum(t.slippage_usd + t.exit_slippage_usd for t in trades), 2
        ),
        "total_cost_usd": round(sum(t.total_cost_usd for t in trades), 2),
        "lot_size": LOT_SIZE,
        "symbols": list({t.symbol for t in trades}),
        "trades": [
            {
                "trade_no": t.trade_no, "symbol": t.symbol, "direction": t.direction,
                "signal_type": t.signal_type, "filled_price": t.filled_price,
                "exit_price": t.exit_price, "slippage_pips": t.slippage_pips,
                "spread_pips": t.spread_pips, "net_pnl_usd": t.net_pnl_usd,
                "entry_time": t.entry_time, "exit_time": t.exit_time,
            }
            for t in trades
        ],
    }
    json_path = RESULTS_DIR / f"live_summary_{cfg.name}_{ts}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log.info("💾 Özet: %s", json_path)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_live(version: str = "V2", hours: float = 24.0) -> None:
    """Main entry point – connect and trade for *hours*."""
    cfg = VERSION_MAP[version]
    global log
    log = setup_logging(version)

    connect_mt5()

    symbols = ALL_SYMBOLS
    magics = {sym.name: get_magic(sym.name, version) for sym in symbols}

    log.info(
        "Bot başladı – versiyon=%s  lot=%.2f  süre=%d saat  semboller=%s",
        cfg.name, LOT_SIZE, int(hours),
        ", ".join(s.name for s in symbols),
    )
    log.info(
        "Parametreler: EMA %d/%d  MACD %d/%d/%d  SL=%.1fx ATR  TP=%.1fx ATR  "
        "Seans %02d:00-%02d:00 UTC  BB_bounce=%s",
        cfg.ema_fast, cfg.ema_slow,
        cfg.macd_fast, cfg.macd_slow, cfg.macd_signal,
        cfg.sl_atr_mult, cfg.tp_atr_mult,
        cfg.session_start, cfg.session_end,
        cfg.enable_bb_bounce,
    )

    start_time = datetime.now(timezone.utc)
    end_time = start_time + timedelta(hours=hours)
    equity = INITIAL_BALANCE
    trade_counter = 0

    open_trades: list[OpenTrade] = []
    closed_trades: list[LiveTrade] = []
    last_bar_time = None

    try:
        while datetime.now(timezone.utc) < end_time:
            now = datetime.now(timezone.utc)
            current_bar = now.replace(minute=0, second=0, microsecond=0)

            # Check closed positions
            newly_closed, open_trades, equity = check_closed_trades(open_trades, equity)
            closed_trades.extend(newly_closed)

            # Only check for new signals on new bar
            if current_bar != last_bar_time:
                last_bar_time = current_bar
                log.info("⏰ Yeni bar: %s  bakiye=$%.2f  açık=%d",
                         current_bar, equity, len(open_trades))

                if equity < 5.0:
                    log.warning("Bakiye çok düşük ($%.2f) – yeni işlem açılmayacak.", equity)
                else:
                    for sym in symbols:
                        magic = magics[sym.name]
                        new_trade, trade_counter = check_and_trade(
                            sym, cfg, magic, LOT_SIZE, open_trades, trade_counter,
                        )
                        if new_trade is not None:
                            open_trades.append(new_trade)

            time.sleep(CHECK_INTERVAL)

    except KeyboardInterrupt:
        log.info("Bot kullanıcı tarafından durduruldu.")
    finally:
        # Final check for closed trades
        newly_closed, open_trades, equity = check_closed_trades(open_trades, equity)
        closed_trades.extend(newly_closed)

        log.info(
            "Bot durdu – toplam işlem=%d  bakiye=$%.2f",
            len(closed_trades), equity,
        )
        save_live_results(closed_trades, cfg, start_time, equity)
        disconnect_mt5()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified Live Forward Test Bot")
    p.add_argument(
        "--version", choices=["V1", "V2", "V3"], default="V2",
        help="Strategy version (default: V2)",
    )
    p.add_argument(
        "--hours", type=float, default=24.0,
        help="Test duration in hours (default: 24)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_live(version=args.version, hours=args.hours)
