"""Tests for backtest.py simulation logic."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategy import Direction, Signal, add_indicators
from backtest import (
    TradeResult,
    simulate_trade,
    run_backtest,
    compute_report,
    INITIAL_BALANCE,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def trending_up_df() -> pd.DataFrame:
    """DataFrame with a clear uptrend for testing."""
    np.random.seed(123)
    n = 400
    trend = np.linspace(1.1000, 1.1200, n)
    noise = np.random.randn(n) * 0.0002
    closes = trend + noise
    df = pd.DataFrame({
        "open": closes - 0.0001,
        "high": closes + 0.0005,
        "low": closes - 0.0005,
        "close": closes,
    })
    df.index = pd.date_range("2025-10-01 08:00", periods=n, freq="h", tz="UTC")
    df.index.name = "time"
    return add_indicators(df)


@pytest.fixture
def simple_signal(trending_up_df: pd.DataFrame) -> Signal:
    """A synthetic LONG signal for testing."""
    idx = 250
    row = trending_up_df.iloc[idx]
    return Signal(
        direction=Direction.LONG,
        entry_price=row["close"],
        stop_loss=row["close"] - 0.0020,
        take_profit=row["close"] + 0.0030,
        atr=0.0013,
        bar_index=idx,
        bar_time=trending_up_df.index[idx],
    )


# ---------------------------------------------------------------------------
# simulate_trade tests
# ---------------------------------------------------------------------------

class TestSimulateTrade:
    def test_returns_trade_result_or_none(self, trending_up_df, simple_signal):
        result = simulate_trade(simple_signal, trending_up_df, INITIAL_BALANCE)
        assert result is None or isinstance(result, TradeResult)

    def test_trade_direction_matches(self, trending_up_df, simple_signal):
        result = simulate_trade(simple_signal, trending_up_df, INITIAL_BALANCE)
        if result is not None:
            assert result.direction == Direction.LONG

    def test_lot_size_positive(self, trending_up_df, simple_signal):
        result = simulate_trade(simple_signal, trending_up_df, INITIAL_BALANCE)
        if result is not None:
            assert result.lot_size >= 0.01


# ---------------------------------------------------------------------------
# compute_report tests
# ---------------------------------------------------------------------------

class TestReport:
    def test_empty_trades(self):
        report = compute_report([], "empty")
        assert report.total_trades == 0
        assert report.final_equity == INITIAL_BALANCE

    def test_single_winning_trade(self):
        trade = TradeResult(
            direction=Direction.LONG,
            entry_price=1.10000,
            exit_price=1.10200,
            stop_loss=1.09800,
            take_profit=1.10200,
            entry_time=pd.Timestamp("2025-10-01 10:00", tz="UTC"),
            exit_time=pd.Timestamp("2025-10-01 14:00", tz="UTC"),
            pnl_pips=20.0,
            pnl_usd=20.0,
            lot_size=0.01,
        )
        report = compute_report([trade], "single_win")
        assert report.total_trades == 1
        assert report.winning_trades == 1
        assert report.win_rate == 100.0
        assert report.total_pnl_usd == 20.0

    def test_drawdown_calculation(self):
        trades = [
            TradeResult(Direction.LONG, 1.1, 1.102, 1.098, 1.102,
                        pd.Timestamp("2025-10-01", tz="UTC"),
                        pd.Timestamp("2025-10-02", tz="UTC"),
                        20.0, 200.0, 0.10),
            TradeResult(Direction.LONG, 1.102, 1.098, 1.098, 1.106,
                        pd.Timestamp("2025-10-02", tz="UTC"),
                        pd.Timestamp("2025-10-03", tz="UTC"),
                        -40.0, -400.0, 0.10),
        ]
        report = compute_report(trades, "dd_test")
        assert report.max_drawdown_usd == 400.0  # peak was 10200, dropped to 9800


# ---------------------------------------------------------------------------
# run_backtest integration
# ---------------------------------------------------------------------------

class TestRunBacktest:
    def test_on_real_data(self):
        """Integration test with actual CSV."""
        from strategy import prepare_data
        try:
            df = prepare_data("data/EURUSD_6m.csv")
        except FileNotFoundError:
            pytest.skip("CSV not available")

        # Skip if CSV was overwritten by mock data from other tests
        if len(df) < 200:
            pytest.skip("CSV contains mock data from other tests")

        trades = run_backtest(df)
        assert isinstance(trades, list)
        # All trades should have valid PnL
        for t in trades:
            assert isinstance(t.pnl_pips, float)
            assert isinstance(t.pnl_usd, float)
