import numpy as np
import pandas as pd
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.rnd_estimation import (
    black_scholes_call, black_scholes_put, fit_iv_curve_moneyness,
    extrapolate_iv_linear, recover_rnd, compute_rnd_moments, validate_density
)

# ── Shared fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def bs_params():
    """Standard Black-Scholes parameters."""
    return dict(S=100.0, K=100.0, r=0.02, sigma=0.25, T=0.5)


@pytest.fixture
def iv_surface():
    """Minimal IV surface: 7 strikes around ATM."""
    S = 100.0
    strikes = np.array([80., 85., 90., 95., 100., 105., 110.])
    ivs = np.array([0.32, 0.29, 0.27, 0.25, 0.24, 0.24, 0.25])
    return dict(strikes=strikes, ivs=ivs, S=S, r=0.02, T=0.25)


@pytest.fixture
def fitted_iv_grid(iv_surface):
    """Pre-fitted IV grid on a 200-point moneyness grid."""
    return fit_iv_curve_moneyness(**iv_surface)


# ── Tests ────────────────────────────────────────────────────────────────────

def test_black_scholes_put_call_parity(bs_params):
    """Put-call parity: C - P = S - K·e^{-rT}."""
    p = bs_params
    C = black_scholes_call(p["S"], p["K"], p["r"], p["sigma"], p["T"])
    P = black_scholes_put(p["S"], p["K"], p["r"], p["sigma"], p["T"])
    parity_lhs = C - P
    parity_rhs = p["S"] - p["K"] * np.exp(-p["r"] * p["T"])
    assert abs(parity_lhs - parity_rhs) < 1e-8, (
        f"Put-call parity violated: C-P={parity_lhs:.6f}, S-Ke^(-rT)={parity_rhs:.6f}"
    )


def test_iv_curve_fitting_returns_valid_grid(fitted_iv_grid):
    """fit_iv_curve_moneyness must return a 200-point (moneyness, iv) pair."""
    moneyness_grid, iv_grid = fitted_iv_grid
    assert len(moneyness_grid) == 200, f"Expected 200 moneyness points, got {len(moneyness_grid)}"
    assert len(iv_grid) == 200, f"Expected 200 IV points, got {len(iv_grid)}"
    assert np.all(iv_grid > 0), "All fitted IVs must be positive"
    assert np.all(moneyness_grid > 0), "Moneyness grid must be positive"


def test_iv_extrapolation_preserves_shape(fitted_iv_grid):
    """Extrapolation must not change values in the interior of the observed range."""
    moneyness_grid, iv_grid = fitted_iv_grid
    iv_extrap = extrapolate_iv_linear(moneyness_grid, iv_grid, extrapolate=True)
    # Interior region (10%-90% of grid) should be unchanged
    lo, hi = int(0.10 * 200), int(0.90 * 200)
    np.testing.assert_allclose(
        iv_extrap[lo:hi], iv_grid[lo:hi], rtol=1e-6,
        err_msg="Extrapolation altered interior IV values",
    )


def test_rnd_integrates_to_one(fitted_iv_grid, iv_surface):
    """Recovered risk-neutral density must integrate to approximately 1.0."""
    moneyness_grid, iv_grid = fitted_iv_grid
    S, r, T = iv_surface["S"], iv_surface["r"], iv_surface["T"]
    S_grid = moneyness_grid * S
    rnd = recover_rnd(moneyness_grid, iv_grid, S, r, T)
    integral = np.trapezoid(rnd, S_grid)
    assert abs(integral - 1.0) < 0.05, (
        f"RND integral = {integral:.4f}, expected ≈ 1.0 (tolerance 0.05)"
    )


def test_rnd_nonnegative(fitted_iv_grid, iv_surface):
    """Every point of the recovered risk-neutral density must be ≥ 0."""
    moneyness_grid, iv_grid = fitted_iv_grid
    rnd = recover_rnd(moneyness_grid, iv_grid, iv_surface["S"],
                      iv_surface["r"], iv_surface["T"])
    assert np.all(rnd >= 0), f"RND has {(rnd < 0).sum()} negative values"


def test_rnd_moments_variance_positive(fitted_iv_grid, iv_surface):
    """Variance derived from RND moments must be strictly positive."""
    moneyness_grid, iv_grid = fitted_iv_grid
    S, r, T = iv_surface["S"], iv_surface["r"], iv_surface["T"]
    rnd = recover_rnd(moneyness_grid, iv_grid, S, r, T)
    S_grid = moneyness_grid * S
    moments = compute_rnd_moments(rnd, S_grid)
    assert "variance" in moments, "compute_rnd_moments must return 'variance' key"
    assert moments["variance"] > 0, (
        f"Variance must be positive, got {moments['variance']}"
    )


def test_moments_edge_case_single_strike():
    """compute_rnd_moments must handle degenerate (spike) density gracefully."""
    S_grid = np.linspace(50.0, 150.0, 200)
    rnd = np.zeros(200)
    rnd[100] = 1.0 / (S_grid[101] - S_grid[99])   # approximate delta
    # Should return a dict without raising
    try:
        moments = compute_rnd_moments(rnd, S_grid)
        assert isinstance(moments, dict), "Return type must be dict"
    except Exception as exc:
        pytest.fail(f"compute_rnd_moments raised on edge case: {exc}")


def test_validate_density_rejects_negative():
    """validate_density must return False when any density value is negative."""
    S_grid = np.linspace(50.0, 150.0, 200)
    rnd_bad = np.ones(200) / (150.0 - 50.0)
    rnd_bad[50] = -0.01    # inject a negative value
    result = validate_density(rnd_bad, S_grid)
    assert result is False or result == False, (
        "validate_density must return False for density with negative values"
    )
