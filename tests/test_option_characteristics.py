import numpy as np
import pandas as pd
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.option_characteristics import (
    model_free_moments,
    jump_measures,
    disaster_measures,
    bakshi_variance,
    svix_measures,
    compute_all_characteristics,
)

# ── Shared fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def rnd_fixture():
    """Synthetic RND dict matching the output of rnd_estimation.recover_rnd."""
    rng = np.random.default_rng(0)
    S = 100.0
    S_grid = np.linspace(20.0, 250.0, 200)
    iv_atm = 0.25
    T = 0.25
    r = 0.02
    d = (np.log(S_grid / S) - (r - 0.5 * iv_atm**2) * T) / (iv_atm * np.sqrt(T))
    pdf = np.exp(-0.5 * d**2) / (iv_atm * np.sqrt(T) * S_grid * np.sqrt(2 * np.pi))
    pdf = np.clip(pdf, 0.0, None)
    pdf /= np.trapezoid(pdf, S_grid)
    return {"S_grid": S_grid, "rnd": pdf, "S": S, "r": r, "T": T}


# ── Tests ────────────────────────────────────────────────────────────────────

def test_model_free_moments_output_keys(rnd_fixture):
    """model_free_moments must return all 8 required moment keys."""
    required_keys = {
        "RVN_var", "RVN_skew", "RVN_kurt",
        "W", "X", "Q", "V",
        "forward_variance",
    }
    result = model_free_moments(
        rnd_fixture["rnd"], rnd_fixture["S_grid"],
        rnd_fixture["S"], rnd_fixture["r"], rnd_fixture["T"]
    )
    missing = required_keys - set(result.keys())
    assert not missing, f"Missing keys from model_free_moments: {missing}"


def test_variance_positive(rnd_fixture):
    """RVN_var (risk-neutral variance) must be strictly positive."""
    result = model_free_moments(
        rnd_fixture["rnd"], rnd_fixture["S_grid"],
        rnd_fixture["S"], rnd_fixture["r"], rnd_fixture["T"]
    )
    assert result["RVN_var"] > 0, (
        f"RVN_var must be positive, got {result['RVN_var']}"
    )


def test_jump_measures_output_keys(rnd_fixture):
    """jump_measures must return the BTX family of keys."""
    required_keys = {"BTX_jump_prob", "BTX_jump_mean", "BTX_jump_var"}
    result = jump_measures(
        rnd_fixture["rnd"], rnd_fixture["S_grid"],
        rnd_fixture["S"], rnd_fixture["r"], rnd_fixture["T"]
    )
    missing = required_keys - set(result.keys())
    assert not missing, f"Missing BTX keys: {missing}"


def test_disaster_measures_output_keys(rnd_fixture):
    """disaster_measures must return expected disaster-related keys."""
    required_keys = {"disaster_prob", "disaster_intensity", "left_tail_var"}
    result = disaster_measures(
        rnd_fixture["rnd"], rnd_fixture["S_grid"],
        rnd_fixture["S"], rnd_fixture["r"], rnd_fixture["T"]
    )
    missing = required_keys - set(result.keys())
    assert not missing, f"Missing disaster keys: {missing}"


def test_bakshi_variance_nonneg(rnd_fixture):
    """bakshi_variance estimate must be non-negative."""
    result = bakshi_variance(
        rnd_fixture["rnd"], rnd_fixture["S_grid"],
        rnd_fixture["S"], rnd_fixture["r"], rnd_fixture["T"]
    )
    assert isinstance(result, (int, float, np.floating)), (
        "bakshi_variance must return a scalar"
    )
    assert result >= 0, f"bakshi_variance must be >= 0, got {result}"


def test_svix_measures_nonneg(rnd_fixture):
    """All SVIX values must be non-negative."""
    result = svix_measures(
        rnd_fixture["rnd"], rnd_fixture["S_grid"],
        rnd_fixture["S"], rnd_fixture["r"], rnd_fixture["T"]
    )
    for key, val in result.items():
        if val is not None and not np.isnan(float(val)):
            assert float(val) >= 0, f"SVIX measure '{key}' = {val} is negative"


def test_all_characteristics_count(rnd_fixture):
    """compute_all_characteristics must return a dict with ≥ 100 entries."""
    result = compute_all_characteristics(
        rnd_fixture, rnd_fixture["S"], rnd_fixture["r"], rnd_fixture["T"]
    )
    assert isinstance(result, dict), "compute_all_characteristics must return a dict"
    assert len(result) >= 100, (
        f"Expected ≥ 100 characteristics, got {len(result)}"
    )


def test_characteristics_handle_nan(rnd_fixture):
    """compute_all_characteristics must return NaN values rather than raising
    when the density input is degenerate (all zeros)."""
    degenerate = {
        **rnd_fixture,
        "rnd": np.zeros_like(rnd_fixture["rnd"]),
    }
    try:
        result = compute_all_characteristics(
            degenerate, degenerate["S"], degenerate["r"], degenerate["T"]
        )
        # At least some values should be NaN (not a crash)
        n_nan = sum(1 for v in result.values() if v is None or
                    (isinstance(v, float) and np.isnan(v)))
        assert n_nan > 0, "Degenerate input should produce some NaN characteristics"
    except Exception as exc:
        pytest.fail(
            f"compute_all_characteristics raised instead of returning NaN: {exc}"
        )
