"""
mt5_bot.py

Forward-testing (live demo) bot that connects to MetaTrader 5 and trades
the scalp–trend strategy on EURUSD in real time.

The bot:
1. Connects to the running MT5 terminal.
2. Every H1 bar close it re-computes indicators on the last N bars.
3. If a signal fires, it places a market order with ATR-based SL / TP.
4. Manages one position at a time (no pyramiding).
5. Logs every action to ``logs/mt5_bot.log`` and to the console.

Usage
-----
    python mt5_bot.py              # default: EURUSD, 0.01 lot
    python mt5_bot.py --lot 0.05   # custom lot size

Requirements
------------
    MetaTrader 5 terminal must be running and logged in to a demo account.
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None  # allow import for testing without MT5

from strategy import (
    Direction,
    add_indicators,
    generate_signals,
    EMA_SLOW,
    ATR_PERIOD,
    SESSION_START_HOUR,
    SESSION_END_HOUR,
    SL_ATR_MULT,
    TP_ATR_MULT,
    MACD_FAST,
    MACD_SLOW,
    MACD_SIGNAL,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SYMBOL = "EURUSD"
TIMEFRAME_MT5 = 16385  # mt5.TIMEFRAME_H1 == 16385
LOOKBACK_BARS = 300     # enough history for EMA-200 warm-up
CHECK_INTERVAL = 60     # seconds between checks
DEFAULT_LOT = 0.01
MAGIC_NUMBER = 202604   # unique identifier for this bot's orders
DEVIATION = 20          # max price deviation (points)

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging() -> logging.Logger:
    logger = logging.getLogger("mt5_bot")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s")

    fh = logging.FileHandler(LOG_DIR / "mt5_bot.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


log = setup_logging()


# ---------------------------------------------------------------------------
# MT5 helpers
# ---------------------------------------------------------------------------

def connect_mt5() -> None:
    """Initialise the MT5 connection."""
    if mt5 is None:
        raise RuntimeError("MetaTrader5 package is not installed.")
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    info = mt5.terminal_info()
    log.info("Connected to MT5: %s  build %s", info.name, info.build)
    acc = mt5.account_info()
    log.info(
        "Account #%s  balance=%.2f  equity=%.2f  leverage=%d",
        acc.login, acc.balance, acc.equity, acc.leverage,
    )


def disconnect_mt5() -> None:
    if mt5 is not None:
        mt5.shutdown()
    log.info("Disconnected from MT5.")


def fetch_bars(symbol: str = SYMBOL, count: int = LOOKBACK_BARS) -> pd.DataFrame:
    """Fetch the last *count* H1 bars from MT5."""
    rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME_MT5, 0, count)
    if rates is None or len(rates) == 0:
        raise ValueError(f"No bars returned for {symbol}: {mt5.last_error()}")
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("time", inplace=True)
    return df


def has_open_position(symbol: str = SYMBOL) -> bool:
    """Return True if there is an open position for *symbol* by this bot."""
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        return False
    return any(p.magic == MAGIC_NUMBER for p in positions)


def get_open_position(symbol: str = SYMBOL):
    """Return the open position object for this bot, or None."""
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        return None
    for p in positions:
        if p.magic == MAGIC_NUMBER:
            return p
    return None


def place_order(
    direction: Direction,
    symbol: str,
    lot: float,
    sl: float,
    tp: float,
) -> bool:
    """Place a market order and return True on success."""
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        log.error("Cannot get tick for %s", symbol)
        return False

    if direction == Direction.LONG:
        order_type = mt5.ORDER_TYPE_BUY
        price = tick.ask
    else:
        order_type = mt5.ORDER_TYPE_SELL
        price = tick.bid

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": order_type,
        "price": price,
        "sl": round(sl, 5),
        "tp": round(tp, 5),
        "deviation": DEVIATION,
        "magic": MAGIC_NUMBER,
        "comment": "scalp_trend_bot",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result is None:
        log.error("order_send returned None: %s", mt5.last_error())
        return False

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        log.error(
            "Order failed – retcode=%d  comment=%s",
            result.retcode, result.comment,
        )
        return False

    log.info(
        "✅ %s %s %.2f lots @ %.5f  SL=%.5f  TP=%.5f  ticket=%d",
        "BUY" if direction == Direction.LONG else "SELL",
        symbol, lot, price, sl, tp, result.order,
    )
    return True


def close_position(position) -> bool:
    """Close an open position."""
    tick = mt5.symbol_info_tick(position.symbol)
    if tick is None:
        return False

    if position.type == mt5.POSITION_TYPE_BUY:
        price = tick.bid
        order_type = mt5.ORDER_TYPE_SELL
    else:
        price = tick.ask
        order_type = mt5.ORDER_TYPE_BUY

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": position.symbol,
        "volume": position.volume,
        "type": order_type,
        "position": position.ticket,
        "price": price,
        "deviation": DEVIATION,
        "magic": MAGIC_NUMBER,
        "comment": "scalp_trend_close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        log.error("Close failed: %s", mt5.last_error())
        return False

    log.info("🔴 Closed position ticket=%d  P&L=%.2f", position.ticket, position.profit)
    return True


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def check_signal_and_trade(lot: float) -> None:
    """Fetch latest bars, compute indicators, and trade if a signal fires."""
    if has_open_position():
        pos = get_open_position()
        if pos:
            log.debug(
                "Open position: ticket=%d  type=%s  profit=%.2f",
                pos.ticket,
                "BUY" if pos.type == 0 else "SELL",
                pos.profit,
            )
        return  # one trade at a time

    try:
        df = fetch_bars()
    except ValueError as exc:
        log.warning("fetch_bars failed: %s", exc)
        return

    df = add_indicators(df)

    # Only check the latest completed bar (second-to-last, since last may be incomplete)
    if len(df) < 2:
        return

    # Generate signals on the last few bars only
    tail = df.iloc[-5:].copy()
    signals = generate_signals(tail)

    if not signals:
        log.debug("No signal on latest bar.")
        return

    # Take the most recent signal
    sig = signals[-1]
    log.info(
        "📈 Signal: %s  entry=%.5f  SL=%.5f  TP=%.5f  ATR=%.5f",
        sig.direction.name, sig.entry_price, sig.stop_loss, sig.take_profit, sig.atr,
    )

    # Use current tick price for SL/TP calculation
    tick = mt5.symbol_info_tick(SYMBOL)
    if tick is None:
        return

    current_price = tick.ask if sig.direction == Direction.LONG else tick.bid
    atr = sig.atr

    if sig.direction == Direction.LONG:
        sl = current_price - SL_ATR_MULT * atr
        tp = current_price + TP_ATR_MULT * atr
    else:
        sl = current_price + SL_ATR_MULT * atr
        tp = current_price - TP_ATR_MULT * atr

    place_order(sig.direction, SYMBOL, lot, sl, tp)


def run_bot(lot: float = DEFAULT_LOT) -> None:
    """Main entry point – connect and loop forever."""
    connect_mt5()
    log.info("Bot started – symbol=%s  lot=%.2f  magic=%d", SYMBOL, lot, MAGIC_NUMBER)
    log.info(
        "Strategy: EMA 50/200 trend | Pullback + MACD(%d/%d/%d) confirmation | "
        "SL=%.1fx ATR  TP=%.1fx ATR | Session %02d:00-%02d:00 UTC",
        MACD_FAST, MACD_SLOW, MACD_SIGNAL,
        SL_ATR_MULT, TP_ATR_MULT,
        SESSION_START_HOUR, SESSION_END_HOUR,
    )

    last_bar_time = None

    try:
        while True:
            now = datetime.now(timezone.utc)

            # Only act on new H1 bar close
            current_bar = now.replace(minute=0, second=0, microsecond=0)
            if current_bar != last_bar_time:
                last_bar_time = current_bar
                log.info("⏰ New bar: %s", current_bar)
                check_signal_and_trade(lot)

            time.sleep(CHECK_INTERVAL)

    except KeyboardInterrupt:
        log.info("Bot stopped by user.")
    finally:
        disconnect_mt5()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scalp-Trend MT5 Forward-Test Bot")
    parser.add_argument("--lot", type=float, default=DEFAULT_LOT, help="Lot size (default: 0.01)")
    parser.add_argument("--symbol", type=str, default=SYMBOL, help="Trading symbol (default: EURUSD)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    SYMBOL = args.symbol
    run_bot(lot=args.lot)
