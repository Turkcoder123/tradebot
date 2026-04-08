"""Tests for strategy.py indicator calculations and signal generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategy import (
    Direction,
    Signal,
    add_indicators,
    compute_atr,
    compute_bollinger,
    compute_ema,
    compute_macd,
    compute_rsi,
    generate_signals,
    prepare_data,
    EMA_FAST,
    EMA_SLOW,
    RSI_PERIOD,
    BB_PERIOD,
    ATR_PERIOD,
    MACD_FAST,
    MACD_SLOW,
    MACD_SIGNAL,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create a minimal OHLC DataFrame for testing."""
    np.random.seed(42)
    n = 300
    base = 1.1000
    closes = base + np.cumsum(np.random.randn(n) * 0.0005)
    df = pd.DataFrame({
        "open": closes + np.random.randn(n) * 0.0001,
        "high": closes + abs(np.random.randn(n) * 0.0003),
        "low": closes - abs(np.random.randn(n) * 0.0003),
        "close": closes,
    })
    df.index = pd.date_range("2025-10-01 08:00", periods=n, freq="h", tz="UTC")
    df.index.name = "time"
    return df


# ---------------------------------------------------------------------------
# Indicator tests
# ---------------------------------------------------------------------------

class TestEMA:
    def test_length(self, sample_df: pd.DataFrame):
        ema = compute_ema(sample_df["close"], 50)
        assert len(ema) == len(sample_df)

    def test_no_nans_after_warmup(self, sample_df: pd.DataFrame):
        ema = compute_ema(sample_df["close"], 50)
        # EWM with adjust=False produces values from the start
        assert not ema.isna().any()

    def test_fast_above_slow_in_uptrend(self):
        """If prices are strictly rising, fast EMA > slow EMA eventually."""
        prices = pd.Series(np.linspace(1.0, 1.05, 300))
        fast = compute_ema(prices, EMA_FAST)
        slow = compute_ema(prices, EMA_SLOW)
        assert fast.iloc[-1] > slow.iloc[-1]


class TestRSI:
    def test_range(self, sample_df: pd.DataFrame):
        rsi = compute_rsi(sample_df["close"])
        valid = rsi.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_overbought_on_constant_rise(self):
        prices = pd.Series(np.linspace(1.0, 1.10, 200))
        rsi = compute_rsi(prices)
        assert rsi.iloc[-1] > 90  # should be very overbought


class TestBollinger:
    def test_upper_above_lower(self, sample_df: pd.DataFrame):
        mid, upper, lower = compute_bollinger(sample_df["close"])
        valid_idx = ~(upper.isna() | lower.isna())
        assert (upper[valid_idx] >= lower[valid_idx]).all()

    def test_mid_is_sma(self, sample_df: pd.DataFrame):
        mid, _, _ = compute_bollinger(sample_df["close"], period=20)
        sma = sample_df["close"].rolling(20).mean()
        pd.testing.assert_series_equal(mid, sma, check_names=False)


class TestATR:
    def test_positive(self, sample_df: pd.DataFrame):
        atr = compute_atr(
            sample_df["high"], sample_df["low"], sample_df["close"]
        )
        valid = atr.dropna()
        assert (valid > 0).all()


# ---------------------------------------------------------------------------
# add_indicators tests
# ---------------------------------------------------------------------------

class TestMACD:
    def test_histogram_shape(self, sample_df: pd.DataFrame):
        macd, signal, hist = compute_macd(sample_df["close"])
        assert len(hist) == len(sample_df)

    def test_histogram_is_diff(self, sample_df: pd.DataFrame):
        macd, signal, hist = compute_macd(sample_df["close"])
        pd.testing.assert_series_equal(hist, macd - signal, check_names=False)


class TestAddIndicators:
    def test_columns_present(self, sample_df: pd.DataFrame):
        df = add_indicators(sample_df)
        expected = {"ema_fast", "ema_slow", "rsi", "bb_mid", "bb_upper",
                    "bb_lower", "atr", "trend", "macd", "macd_signal",
                    "macd_hist"}
        assert expected.issubset(set(df.columns))

    def test_trend_values(self, sample_df: pd.DataFrame):
        df = add_indicators(sample_df)
        assert set(df["trend"].unique()).issubset({1, -1})

    def test_does_not_modify_original(self, sample_df: pd.DataFrame):
        original_cols = set(sample_df.columns)
        add_indicators(sample_df)
        assert set(sample_df.columns) == original_cols


# ---------------------------------------------------------------------------
# Signal generation tests
# ---------------------------------------------------------------------------

class TestSignals:
    def test_signals_are_list_of_signal(self, sample_df: pd.DataFrame):
        df = add_indicators(sample_df)
        signals = generate_signals(df)
        assert isinstance(signals, list)
        for sig in signals:
            assert isinstance(sig, Signal)

    def test_signal_directions(self, sample_df: pd.DataFrame):
        df = add_indicators(sample_df)
        signals = generate_signals(df)
        for sig in signals:
            assert sig.direction in (Direction.LONG, Direction.SHORT)

    def test_sl_tp_consistency(self, sample_df: pd.DataFrame):
        df = add_indicators(sample_df)
        signals = generate_signals(df)
        for sig in signals:
            if sig.direction == Direction.LONG:
                assert sig.stop_loss < sig.entry_price < sig.take_profit
            else:
                assert sig.take_profit < sig.entry_price < sig.stop_loss


# ---------------------------------------------------------------------------
# prepare_data integration test (requires CSV file)
# ---------------------------------------------------------------------------

class TestPrepareData:
    def test_with_real_csv(self):
        """Integration test using the actual EURUSD CSV."""
        csv_path = "data/EURUSD_6m.csv"
        try:
            df = prepare_data(csv_path)
        except FileNotFoundError:
            pytest.skip("CSV file not available")

        # Only run full assertion when the real (large) CSV is present;
        # the test_fetch_prices suite may overwrite data/ with mock data.
        if len(df) < 200:
            pytest.skip("CSV contains mock data from other tests")

        assert len(df) > 2000
        assert "ema_fast" in df.columns
        assert "trend" in df.columns
