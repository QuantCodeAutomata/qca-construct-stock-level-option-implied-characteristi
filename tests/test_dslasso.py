import numpy as np
import pandas as pd
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.dslasso import (
    compute_covariance,
    select_I1,
    lambda_estimate,
    lambda_s_in_bp,
    apply_multiple_testing,
    residualize,
)

# ── Shared fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def ds_data():
    """Synthetic data: T=120 obs, K=50 benchmark factors, N=6 test assets."""
    rng = np.random.default_rng(0)
    T, K, N = 120, 50, 6
    H = rng.standard_normal((T, K)) * 0.03      # benchmark (H) factors
    f = H[:, :3] @ rng.standard_normal(3) * 0.5 + rng.standard_normal(T) * 0.02
    test_assets = rng.standard_normal((T, N)) * 0.04
    return {"H": H, "f": f, "test_assets": test_assets, "T": T, "K": K, "N": N}


@pytest.fixture
def results_table():
    """Synthetic DS-LASSO results table with p-values for multiple-testing."""
    rng = np.random.default_rng(1)
    n = 20
    factors = [f"char_{i:03d}" for i in range(n)]
    p_vals = np.concatenate([
        rng.uniform(0.0, 0.01, 5),   # 5 highly significant
        rng.uniform(0.05, 1.0, 15),  # 15 not significant
    ])
    rng.shuffle(p_vals)
    return pd.DataFrame({
        "factor": factors,
        "lambda_s_bp": rng.standard_normal(n) * 10,
        "t_stat": rng.standard_normal(n) * 2,
        "p_value": p_vals,
    }).set_index("factor")


# ── Tests ────────────────────────────────────────────────────────────────────

def test_compute_covariance_shape(ds_data):
    """compute_covariance must return an (N, N) matrix for N test assets."""
    N = ds_data["N"]
    ta = ds_data["test_assets"]
    cov = compute_covariance(ta)
    assert cov.shape == (N, N), (
        f"Expected covariance shape ({N}, {N}), got {cov.shape}"
    )
    # Diagonal elements (variances) must be non-negative
    assert np.all(np.diag(cov) >= 0), "Diagonal of covariance must be non-negative"


def test_select_I1_returns_indices(ds_data):
    """select_I1 must return a list of integer indices into the benchmark columns."""
    K = ds_data["K"]
    indices = select_I1(ds_data["f"], ds_data["H"])
    assert isinstance(indices, (list, np.ndarray)), (
        "select_I1 must return a list or array of indices"
    )
    assert all(isinstance(int(i), int) for i in indices), (
        "All returned indices must be integer-compatible"
    )
    if len(indices) > 0:
        assert max(indices) < K, f"Index {max(indices)} out of bounds for K={K}"
        assert min(indices) >= 0, "Indices must be non-negative"


def test_lambda_estimate_shape(ds_data):
    """lambda_estimate must return a scalar for a single factor."""
    lam = lambda_estimate(ds_data["f"], ds_data["H"], ds_data["test_assets"])
    assert np.isscalar(lam) or (hasattr(lam, "shape") and lam.shape == ()), (
        f"lambda_estimate must be scalar, got shape={getattr(lam, 'shape', 'N/A')}"
    )
    assert np.isfinite(float(lam)) or np.isnan(float(lam)), (
        "lambda_estimate must be finite or NaN"
    )


def test_lambda_s_in_bp(ds_data):
    """lambda_s_in_bp must return annualised factor pricing error in basis points."""
    lam_raw = lambda_estimate(ds_data["f"], ds_data["H"], ds_data["test_assets"])
    lam_bp = lambda_s_in_bp(float(lam_raw))
    # 1 bp = 0.0001; monthly lambda in raw return units, so bp = raw * 10_000
    # Check that the conversion is 10_000x (within floating-point tolerance)
    if np.isfinite(float(lam_raw)) and np.isfinite(float(lam_bp)):
        ratio = abs(float(lam_bp)) / (abs(float(lam_raw)) + 1e-12)
        assert abs(ratio - 10_000) < 1.0, (
            f"Expected bp = raw × 10,000; got ratio={ratio:.2f}"
        )


def test_multiple_testing_adds_columns(results_table):
    """apply_multiple_testing must add 'bh_significant' and 'bonferroni_significant'."""
    corrected = apply_multiple_testing(results_table)
    assert "bh_significant" in corrected.columns, (
        "apply_multiple_testing must add 'bh_significant' column"
    )
    assert "bonferroni_significant" in corrected.columns, (
        "apply_multiple_testing must add 'bonferroni_significant' column"
    )
    assert corrected["bh_significant"].dtype == bool, (
        "'bh_significant' must be boolean"
    )
    assert corrected["bonferroni_significant"].dtype == bool, (
        "'bonferroni_significant' must be boolean"
    )


def test_residualize_reduces_correlation(ds_data):
    """residualize(f, H) must produce residuals less correlated with H than f is."""
    f = ds_data["f"]
    H = ds_data["H"]

    f_resid = residualize(f, H)
    assert len(f_resid) == len(f), "Residual length must match input length"

    # Measure max absolute correlation of f vs H columns, and residual vs H
    max_corr_original = max(abs(np.corrcoef(f, H[:, j])[0, 1]) for j in range(H.shape[1]))
    max_corr_residual = max(abs(np.corrcoef(f_resid, H[:, j])[0, 1]) for j in range(H.shape[1]))

    assert max_corr_residual < max_corr_original, (
        f"Residual max|corr| with H ({max_corr_residual:.4f}) should be less than "
        f"original ({max_corr_original:.4f})"
    )
