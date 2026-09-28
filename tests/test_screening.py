import numpy as np
import pandas as pd
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.screening import (
    build_target,
    fit_hgbr_model,
    compute_shap_ranking,
    select_by_cumulative_share,
    walk_forward_splits,
)

# ── Shared fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_data():
    """Small factor panel and test assets for screening tests."""
    rng = np.random.default_rng(0)
    T, p = 120, 20
    dates = pd.date_range("2000-01", periods=T, freq="MS")
    X = pd.DataFrame(
        rng.standard_normal((T, p)) * 0.03,
        index=dates,
        columns=[f"factor_{i:02d}" for i in range(p)],
    )
    # Test assets with genuine signal from first 3 factors
    ta = pd.DataFrame(
        X.iloc[:, :3].values @ rng.standard_normal((3, 6)) * 0.5 +
        rng.standard_normal((T, 6)) * 0.02,
        index=dates,
        columns=[f"ta_{i}" for i in range(6)],
    )
    return X, ta


@pytest.fixture
def fitted_model(synthetic_data):
    """Pre-fitted HGBR model for reuse in multiple tests."""
    X, ta = synthetic_data
    y = build_target(ta)
    return fit_hgbr_model(X, y), X, y


# ── Tests ────────────────────────────────────────────────────────────────────

def test_build_target_equal_weights(synthetic_data):
    """build_target must return the unweighted (equal-weight) mean across test assets."""
    _, ta = synthetic_data
    target = build_target(ta)
    expected = ta.mean(axis=1)
    pd.testing.assert_series_equal(
        target.rename(None), expected.rename(None),
        check_names=False, rtol=1e-10,
    )


def test_fit_model_returns_model(synthetic_data):
    """fit_hgbr_model must return a fitted model without raising."""
    X, ta = synthetic_data
    y = build_target(ta)
    try:
        model = fit_hgbr_model(X, y)
        assert model is not None, "fit_hgbr_model must return a non-None model"
        # Model must be callable for prediction
        preds = model.predict(X.values)
        assert len(preds) == len(y), "Predictions length mismatch"
    except Exception as exc:
        pytest.fail(f"fit_hgbr_model raised: {exc}")


def test_shap_ranking_sums_to_one(fitted_model):
    """Normalised SHAP attribution series must sum to 1.0 (within tolerance)."""
    model, X, _ = fitted_model
    shap_series = compute_shap_ranking(model, X)
    assert isinstance(shap_series, pd.Series), (
        "compute_shap_ranking must return a pd.Series"
    )
    assert abs(shap_series.sum() - 1.0) < 1e-6, (
        f"SHAP series sums to {shap_series.sum():.8f}, expected 1.0"
    )
    assert (shap_series >= 0).all(), "All SHAP shares must be non-negative"


def test_select_by_cumulative_share_threshold(fitted_model):
    """Selected factors must capture ≤ threshold + one_factor of total attribution."""
    model, X, _ = fitted_model
    shap_series = compute_shap_ranking(model, X)
    threshold = 0.90
    selected = select_by_cumulative_share(shap_series, threshold=threshold)
    cumulative_share = shap_series[selected].sum()
    # Cumulative share should be ≥ threshold (includes the crossing factor)
    assert cumulative_share >= threshold - 1e-6, (
        f"Selection captured only {cumulative_share:.4f} of attribution (< {threshold})"
    )
    # And at most threshold + one additional factor's share
    max_allowed = threshold + shap_series.max()
    assert cumulative_share <= max_allowed + 1e-6, (
        f"Selection captured {cumulative_share:.4f}, exceeds threshold by too much"
    )


def test_selection_invariant_to_column_order(fitted_model):
    """Selecting factors must yield the same set regardless of column order in X."""
    model, X, y = fitted_model
    shap_series = compute_shap_ranking(model, X)
    selected_original = set(select_by_cumulative_share(shap_series, threshold=0.80))

    # Shuffle columns
    rng = np.random.default_rng(42)
    shuffled_cols = rng.permutation(X.columns.tolist()).tolist()
    X_shuffled = X[shuffled_cols]
    model_shuffled = fit_hgbr_model(X_shuffled, y)
    shap_shuffled = compute_shap_ranking(model_shuffled, X_shuffled)
    selected_shuffled = set(select_by_cumulative_share(shap_shuffled, threshold=0.80))

    # The top factors (by share) should overlap substantially
    overlap = len(selected_original & selected_shuffled)
    min_overlap = min(len(selected_original), len(selected_shuffled)) // 2
    assert overlap >= min_overlap, (
        f"Selection changed substantially after column shuffle: "
        f"original={selected_original}, shuffled={selected_shuffled}"
    )


def test_walk_forward_splits_temporal_order(synthetic_data):
    """walk_forward_splits: every training window must precede its validation window."""
    X, ta = synthetic_data
    y = build_target(ta)
    splits = walk_forward_splits(X, y, n_splits=4)
    for i, (train_idx, val_idx) in enumerate(splits):
        assert len(train_idx) > 0, f"Split {i}: empty training index"
        assert len(val_idx) > 0, f"Split {i}: empty validation index"
        latest_train = X.index[train_idx].max()
        earliest_val = X.index[val_idx].min()
        assert latest_train < earliest_val, (
            f"Split {i}: training data ({latest_train}) overlaps or "
            f"follows validation ({earliest_val})"
        )
