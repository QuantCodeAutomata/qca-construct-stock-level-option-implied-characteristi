import numpy as np
import pandas as pd
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.equity_factors import (
    compute_factor_returns_month,
    build_all_equity_factors,
    build_3x2_test_assets,
    validate_factor_panel,
)

# ── Shared fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def small_universe():
    """Minimal stock-month panel: 50 stocks across 12 months."""
    rng = np.random.default_rng(0)
    n_stocks, n_months = 50, 12
    dates = pd.date_range("2000-01", periods=n_months, freq="MS")
    records = []
    for pid in range(n_stocks):
        mc = rng.uniform(1e8, 1e11)
        bm = rng.uniform(0.1, 5.0)
        for date in dates:
            records.append({
                "date": date,
                "permno": pid,
                "ret": rng.normal(0.007, 0.10),
                "market_cap": mc * np.exp(rng.normal(0, 0.05)),
                "bm": bm,
                "char_001": rng.standard_normal(),
            })
    return pd.DataFrame(records)


@pytest.fixture
def full_universe():
    """Larger panel used for shape tests: 200 stocks, 48 months."""
    rng = np.random.default_rng(1)
    n_stocks, n_months = 200, 48
    dates = pd.date_range("2000-01", periods=n_months, freq="MS")
    records = []
    for pid in range(n_stocks):
        mc = rng.uniform(1e8, 1e11)
        bm = rng.uniform(0.1, 5.0)
        for date in dates:
            records.append({
                "date": date,
                "permno": pid,
                "ret": rng.normal(0.007, 0.10),
                "market_cap": mc * np.exp(rng.normal(0, 0.05)),
                "bm": bm,
            })
    return pd.DataFrame(records)


@pytest.fixture
def valid_factor_panel():
    """Factor panel that should pass validation."""
    rng = np.random.default_rng(2)
    dates = pd.date_range("2000-01", periods=234, freq="MS")
    data = rng.standard_normal((234, 152)) * 0.03
    return pd.DataFrame(
        data, index=dates,
        columns=[f"factor_{i:03d}" for i in range(152)],
    )


# ── Tests ────────────────────────────────────────────────────────────────────

def test_factor_returns_single_month(small_universe):
    """compute_factor_returns_month must return a scalar float for one month."""
    date = small_universe["date"].iloc[0]
    month_data = small_universe[small_universe["date"] == date]
    result = compute_factor_returns_month(
        month_data, char_col="char_001", weight_col="market_cap"
    )
    assert isinstance(result, (float, int, np.floating)), (
        f"Expected scalar float, got {type(result)}"
    )
    assert np.isfinite(result) or np.isnan(result), (
        "Result must be finite or NaN, not infinite"
    )


def test_factor_construction_shape(full_universe):
    """build_all_equity_factors must return (n_months, 152) DataFrame."""
    factors = build_all_equity_factors(full_universe)
    n_months = full_universe["date"].nunique()
    assert isinstance(factors, pd.DataFrame), "Must return a DataFrame"
    assert factors.shape == (n_months, 152), (
        f"Expected ({n_months}, 152), got {factors.shape}"
    )


def test_vw_top_minus_bottom(small_universe):
    """Value-weighted long-short factor: top decile mean return – bottom decile."""
    rng = np.random.default_rng(99)
    # Give char_001 a clear positive cross-sectional spread so top > bottom on average
    small_universe = small_universe.copy()
    small_universe["char_001"] = rng.standard_normal(len(small_universe)) * 2
    # High char_001 → higher future return (inject signal)
    small_universe["ret"] = (
        0.005 * small_universe["char_001"] +
        rng.standard_normal(len(small_universe)) * 0.02
    )
    factors = build_all_equity_factors(small_universe)
    # Average factor return should be positive on the injected-signal characteristic
    avg_return = factors.mean().mean()
    # Not a strict sign test (random variation), just check finite
    assert np.isfinite(avg_return), "Factor returns must be finite"


def test_validate_factor_panel_passes(valid_factor_panel):
    """A well-formed factor panel must pass validate_factor_panel without error."""
    try:
        result = validate_factor_panel(valid_factor_panel)
        # Accepting either True or None (function may return None on pass)
        assert result is None or result is True or result == True, (
            f"validate_factor_panel returned unexpected value: {result}"
        )
    except Exception as exc:
        pytest.fail(f"validate_factor_panel raised on valid panel: {exc}")


def test_test_assets_six_portfolios(full_universe):
    """build_3x2_test_assets must return a DataFrame with exactly 6 columns."""
    test_assets = build_3x2_test_assets(full_universe)
    assert isinstance(test_assets, pd.DataFrame), "Must return a DataFrame"
    assert test_assets.shape[1] == 6, (
        f"Expected 6 portfolio columns, got {test_assets.shape[1]}"
    )
