import numpy as np
import pandas as pd
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.joint_tests import (
    grs_test,
    compute_max_sharpe,
    compare_sharpe_ratios,
    canonical_correlation_analysis,
    stewart_love_redundancy,
)
from sklearn.covariance import LedoitWolf

# ── Shared fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def factor_data():
    """Synthetic returns: T=200, N=5 test assets, K=3 benchmark factors."""
    rng = np.random.default_rng(0)
    T, N, K = 200, 5, 3
    dates = pd.date_range("2000-01", periods=T, freq="MS")
    factors = pd.DataFrame(
        rng.standard_normal((T, K)) * 0.04,
        index=dates, columns=[f"f{i}" for i in range(K)],
    )
    # Test assets: weak loadings on factors + idiosyncratic noise
    B = rng.standard_normal((K, N)) * 0.5
    assets = pd.DataFrame(
        factors.values @ B + rng.standard_normal((T, N)) * 0.05,
        index=dates, columns=[f"asset_{i}" for i in range(N)],
    )
    return assets, factors


@pytest.fixture
def option_equity_pair():
    """Matched option and equity factor blocks for CCA/redundancy tests."""
    rng = np.random.default_rng(1)
    T, p, q = 200, 8, 10
    dates = pd.date_range("2000-01", periods=T, freq="MS")
    latent = rng.standard_normal((T, 3))
    X = pd.DataFrame(
        latent @ rng.standard_normal((3, p)) * 0.5 +
        rng.standard_normal((T, p)) * 0.03,
        index=dates, columns=[f"op_{i}" for i in range(p)],
    )
    Y = pd.DataFrame(
        latent @ rng.standard_normal((3, q)) * 0.5 +
        rng.standard_normal((T, q)) * 0.03,
        index=dates, columns=[f"eq_{i}" for i in range(q)],
    )
    return X, Y


# ── Tests ────────────────────────────────────────────────────────────────────

def test_grs_statistic_nonneg(factor_data):
    """GRS F-statistic must be non-negative (quadratic form over PSD matrix)."""
    assets, factors = factor_data
    result = grs_test(assets, factors)
    assert result["grs_stat"] >= 0, (
        f"GRS statistic must be ≥ 0, got {result['grs_stat']}"
    )


def test_grs_pvalue_in_unit_interval(factor_data):
    """GRS p-value must lie in the closed interval [0, 1]."""
    assets, factors = factor_data
    result = grs_test(assets, factors)
    p = result["p_value"]
    assert 0.0 <= p <= 1.0, f"p-value={p} is outside [0, 1]"


def test_max_sharpe_positive():
    """compute_max_sharpe must return a strictly positive value for non-trivial returns."""
    rng = np.random.default_rng(2)
    T, p = 150, 10
    dates = pd.date_range("2000-01", periods=T, freq="MS")
    # Inject positive means so the tangency portfolio has a definite Sharpe
    data = pd.DataFrame(
        rng.standard_normal((T, p)) * 0.04 + 0.005,
        index=dates, columns=[f"f{i}" for i in range(p)],
    )
    sr = compute_max_sharpe(data, annualize=True)
    assert sr > 0, f"Maximum Sharpe ratio must be positive, got {sr}"


def test_ledoit_wolf_psd():
    """LedoitWolf shrinkage covariance must be positive semi-definite."""
    rng = np.random.default_rng(3)
    T, p = 150, 30
    X = rng.standard_normal((T, p)) * 0.04
    lw = LedoitWolf().fit(X)
    Sigma = lw.covariance_
    eigenvalues = np.linalg.eigvalsh(Sigma)
    assert np.all(eigenvalues >= -1e-10), (
        f"LedoitWolf covariance has negative eigenvalue(s): min={eigenvalues.min():.2e}"
    )


def test_canonical_correlations_in_01(option_equity_pair):
    """All canonical correlations must lie in [0, 1]."""
    X, Y = option_equity_pair
    result = canonical_correlation_analysis(X, Y)
    rho = result["canonical_correlations"]
    assert len(rho) > 0, "Must return at least one canonical correlation"
    assert np.all(rho >= 0.0), f"Negative canonical correlation: {rho.min()}"
    assert np.all(rho <= 1.0), f"Canonical correlation > 1: {rho.max()}"


def test_redundancy_index_in_01(option_equity_pair):
    """Both Stewart-Love redundancy indices must lie in [0, 1]."""
    X, Y = option_equity_pair
    result = stewart_love_redundancy(X, Y)
    ri_op = result["RI_options_given_equity"]
    ri_eq = result["RI_equity_given_options"]
    assert 0.0 <= ri_op <= 1.0, f"RI_options_given_equity = {ri_op} outside [0,1]"
    assert 0.0 <= ri_eq <= 1.0, f"RI_equity_given_options = {ri_eq} outside [0,1]"


def test_unique_share_complement(option_equity_pair):
    """unique_option_share must exactly equal 1 − RI_options_given_equity."""
    X, Y = option_equity_pair
    result = stewart_love_redundancy(X, Y)
    ri_op = result["RI_options_given_equity"]
    ri_eq = result["RI_equity_given_options"]
    unique_op = result["unique_option_share"]
    unique_eq = result["unique_equity_share"]
    assert abs(unique_op - (1.0 - ri_op)) < 1e-10, (
        f"unique_option_share ({unique_op}) ≠ 1 − RI ({1.0 - ri_op})"
    )
    assert abs(unique_eq - (1.0 - ri_eq)) < 1e-10, (
        f"unique_equity_share ({unique_eq}) ≠ 1 − RI ({1.0 - ri_eq})"
    )
