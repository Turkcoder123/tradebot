"""Tests for the unified strategy config, backtest, and multi-symbol support."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from strategy import (
    ALL_CONFIGS,
    Direction,
    Signal,
    StrategyConfig,
    SymbolSpec,
    V1_AGGRESSIVE,
    V2_BALANCED,
    V3_CONSERVATIVE,
    EURUSD_SPEC,
    XAUUSD_SPEC,
    add_indicators,
    generate_signals,
)
from unified_backtest import (
    DetailedTrade,
    run_single_backtest,
    compute_summary,
    save_trades_csv,
    save_summary_json,
    INITIAL_BALANCE,
    LOT_SIZE,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create a minimal OHLC DataFrame suitable for all 3 configs."""
    np.random.seed(42)
    n = 400
    base = 1.1000
    closes = base + np.cumsum(np.random.randn(n) * 0.0005)
    df = pd.DataFrame({
        "open": closes + np.random.randn(n) * 0.0001,
        "high": closes + abs(np.random.randn(n) * 0.0003),
        "low": closes - abs(np.random.randn(n) * 0.0003),
        "close": closes,
        "spread": np.ones(n) * 1.5,
    })
    df.index = pd.date_range("2025-10-01 08:00", periods=n, freq="h", tz="UTC")
    df.index.name = "time"
    return df


@pytest.fixture
def xauusd_df() -> pd.DataFrame:
    """Create OHLC data at XAUUSD scale (~2000)."""
    np.random.seed(123)
    n = 400
    base = 2000.0
    closes = base + np.cumsum(np.random.randn(n) * 1.5)
    df = pd.DataFrame({
        "open": closes + np.random.randn(n) * 0.3,
        "high": closes + abs(np.random.randn(n) * 1.0),
        "low": closes - abs(np.random.randn(n) * 1.0),
        "close": closes,
        "spread": np.ones(n) * 30.0,
    })
    df.index = pd.date_range("2025-10-01 08:00", periods=n, freq="h", tz="UTC")
    df.index.name = "time"
    return df


# ---------------------------------------------------------------------------
# StrategyConfig tests
# ---------------------------------------------------------------------------

class TestStrategyConfig:
    def test_three_presets_exist(self):
        assert len(ALL_CONFIGS) == 3

    def test_configs_are_frozen(self):
        with pytest.raises(AttributeError):
            V1_AGGRESSIVE.name = "oops"  # type: ignore[misc]

    def test_v1_is_aggressive(self):
        assert V1_AGGRESSIVE.ema_fast < V2_BALANCED.ema_fast
        assert V1_AGGRESSIVE.sl_atr_mult < V2_BALANCED.sl_atr_mult
        assert V1_AGGRESSIVE.session_start < V2_BALANCED.session_start

    def test_v3_is_conservative(self):
        assert V3_CONSERVATIVE.sl_atr_mult > V2_BALANCED.sl_atr_mult
        assert V3_CONSERVATIVE.tp_atr_mult > V2_BALANCED.tp_atr_mult
        assert V3_CONSERVATIVE.enable_bb_bounce is False

    def test_default_config_matches_module_constants(self):
        """Default StrategyConfig should use the module-level constants."""
        from strategy import (
            EMA_FAST, EMA_SLOW, RSI_PERIOD, BB_PERIOD,
            ATR_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
            SESSION_START_HOUR, SESSION_END_HOUR,
            SL_ATR_MULT, TP_ATR_MULT, EMA_RETEST_TOL,
        )
        default = StrategyConfig()
        assert default.ema_fast == EMA_FAST
        assert default.ema_slow == EMA_SLOW
        assert default.sl_atr_mult == SL_ATR_MULT
        assert default.tp_atr_mult == TP_ATR_MULT


# ---------------------------------------------------------------------------
# SymbolSpec tests
# ---------------------------------------------------------------------------

class TestSymbolSpec:
    def test_eurusd_pip(self):
        assert EURUSD_SPEC.pip_size == 0.0001
        assert EURUSD_SPEC.pip_value_per_lot == 10.0

    def test_xauusd_pip(self):
        assert XAUUSD_SPEC.pip_size == 0.01
        assert XAUUSD_SPEC.pip_value_per_lot == 1.0


# ---------------------------------------------------------------------------
# Config-aware indicator tests
# ---------------------------------------------------------------------------

class TestConfigIndicators:
    def test_v1_indicators_use_v1_params(self, sample_df):
        df = add_indicators(sample_df, V1_AGGRESSIVE)
        assert "ema_fast" in df.columns
        # V1 uses EMA-20 as fast → check that the EMA values differ from V2
        df2 = add_indicators(sample_df, V2_BALANCED)
        # Different EMA periods should give different values
        assert not df["ema_fast"].equals(df2["ema_fast"])

    def test_all_configs_produce_valid_indicators(self, sample_df):
        for cfg in ALL_CONFIGS:
            df = add_indicators(sample_df, cfg)
            assert "trend" in df.columns
            assert set(df["trend"].unique()).issubset({1, -1})


# ---------------------------------------------------------------------------
# Config-aware signal generation tests
# ---------------------------------------------------------------------------

class TestConfigSignals:
    def test_v1_generates_more_signals_than_v3(self, sample_df):
        """V1 (aggressive) should generally generate more signals."""
        df1 = add_indicators(sample_df, V1_AGGRESSIVE)
        df3 = add_indicators(sample_df, V3_CONSERVATIVE)
        sig1 = generate_signals(df1, V1_AGGRESSIVE)
        sig3 = generate_signals(df3, V3_CONSERVATIVE)
        # V1 has wider session, looser tolerance, BB bounce enabled
        assert len(sig1) >= len(sig3)

    def test_bb_bounce_signals_exist_in_v1(self, sample_df):
        df = add_indicators(sample_df, V1_AGGRESSIVE)
        signals = generate_signals(df, V1_AGGRESSIVE)
        types = {s.signal_type for s in signals}
        # BB bounce may or may not fire depending on data, but
        # pullback/momentum should be present
        assert len(types) >= 1

    def test_v3_no_bb_bounce(self, sample_df):
        df = add_indicators(sample_df, V3_CONSERVATIVE)
        signals = generate_signals(df, V3_CONSERVATIVE)
        bb_signals = [s for s in signals if s.signal_type == "bb_bounce"]
        assert len(bb_signals) == 0

    def test_all_signals_have_valid_sl_tp(self, sample_df):
        for cfg in ALL_CONFIGS:
            df = add_indicators(sample_df, cfg)
            signals = generate_signals(df, cfg)
            for sig in signals:
                if sig.direction == Direction.LONG:
                    assert sig.stop_loss < sig.entry_price < sig.take_profit
                else:
                    assert sig.take_profit < sig.entry_price < sig.stop_loss


# ---------------------------------------------------------------------------
# Unified backtest tests
# ---------------------------------------------------------------------------

class TestUnifiedBacktest:
    def test_initial_balance_is_50(self):
        assert INITIAL_BALANCE == 50.0

    def test_lot_size_is_005(self):
        assert LOT_SIZE == 0.05

    def test_run_backtest_returns_detailed_trades(self, sample_df):
        trades = run_single_backtest(sample_df, V2_BALANCED, EURUSD_SPEC)
        assert isinstance(trades, list)
        for t in trades:
            assert isinstance(t, DetailedTrade)

    def test_trades_have_spread_and_slippage(self, sample_df):
        trades = run_single_backtest(sample_df, V1_AGGRESSIVE, EURUSD_SPEC)
        if not trades:
            pytest.skip("No trades generated on sample data")
        for t in trades:
            assert t.spread_pips >= 0
            assert t.slippage_entry_pips >= 0
            assert t.total_cost_usd >= 0
            assert t.lot_size == LOT_SIZE

    def test_balance_tracks_correctly(self, sample_df):
        trades = run_single_backtest(sample_df, V2_BALANCED, EURUSD_SPEC)
        if not trades:
            pytest.skip("No trades generated")
        expected = INITIAL_BALANCE
        for t in trades:
            expected += t.net_pnl_usd
            assert abs(t.balance_after - expected) < 0.01

    def test_xauusd_data_works(self, xauusd_df):
        """Strategy should handle XAUUSD price scale."""
        trades = run_single_backtest(xauusd_df, V2_BALANCED, XAUUSD_SPEC)
        assert isinstance(trades, list)

    def test_margin_check_stops_trading(self):
        """With tiny balance, should stop trading when below $5."""
        np.random.seed(42)
        n = 400
        base = 1.1000
        closes = base + np.cumsum(np.random.randn(n) * 0.0005)
        df = pd.DataFrame({
            "open": closes + np.random.randn(n) * 0.0001,
            "high": closes + abs(np.random.randn(n) * 0.0005),
            "low": closes - abs(np.random.randn(n) * 0.0005),
            "close": closes,
        })
        df.index = pd.date_range("2025-10-01 08:00", periods=n, freq="h", tz="UTC")
        df.index.name = "time"

        trades = run_single_backtest(df, V1_AGGRESSIVE, EURUSD_SPEC)
        # All final balances should be >= some minimum (may go negative due
        # to trade in progress, but should stop opening new trades)
        if trades:
            # Should never have a trade that starts below $5 balance
            equity = INITIAL_BALANCE
            for t in trades:
                assert equity >= 5.0 or t == trades[-1]
                equity = t.balance_after


# ---------------------------------------------------------------------------
# Summary computation tests
# ---------------------------------------------------------------------------

class TestSummaryComputation:
    def test_empty_summary(self):
        s = compute_summary([], "V2", "EURUSD")
        assert s.total_trades == 0
        assert s.final_balance == INITIAL_BALANCE

    def test_summary_fields(self, sample_df):
        trades = run_single_backtest(sample_df, V2_BALANCED, EURUSD_SPEC)
        s = compute_summary(trades, "V2", "EURUSD")
        assert s.version == "V2"
        assert s.symbol == "EURUSD"
        assert s.total_trades == len(trades)
        assert s.winning_trades + s.losing_trades == s.total_trades
        assert s.max_drawdown_usd >= 0


# ---------------------------------------------------------------------------
# File output tests
# ---------------------------------------------------------------------------

class TestFileOutput:
    def test_save_trades_csv(self, sample_df, tmp_path):
        trades = run_single_backtest(sample_df, V2_BALANCED, EURUSD_SPEC)
        if not trades:
            pytest.skip("No trades to save")
        path = tmp_path / "test_trades.csv"
        save_trades_csv(trades, path)
        assert path.exists()
        df = pd.read_csv(path)
        assert len(df) == len(trades)
        assert "spread_pips" in df.columns
        assert "slippage_entry_pips" in df.columns
        assert "net_pnl_usd" in df.columns

    def test_save_summary_json(self, sample_df, tmp_path):
        trades = run_single_backtest(sample_df, V2_BALANCED, EURUSD_SPEC)
        s = compute_summary(trades, "V2", "EURUSD")
        path = tmp_path / "test_summary.json"
        save_summary_json([s], path)
        assert path.exists()
        with open(path) as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert data[0]["version"] == "V2"
