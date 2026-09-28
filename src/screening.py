"""
Factor screening using histogram-based gradient boosted regression trees and SHAP.

Follows the paper's screening procedure:
- Target: equal-weighted average return of all test assets
- Features: monthly option factor returns  
- Model: HistGradientBoostingRegressor with learning_rate=0.01, max_iter=100, max_depth=3
- Selection: top factors with cumulative SHAP attribution <= 90%

Uses scikit-learn HistGradientBoostingRegressor (Context7: confirmed available)
Uses shap.Explainer for SHAP values (Context7: confirmed available)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
import shap
from typing import List, Dict, Tuple, Optional
import json
import warnings

# Default HGBR hyper-parameters from the paper.
_DEFAULT_HGBR_PARAMS: Dict = {
    "learning_rate": 0.01,
    "max_iter": 100,
    "max_depth": 3,
}


def build_target(test_asset_returns: pd.DataFrame) -> pd.Series:
    """
    Compute the equal-weighted average test-asset return for each month.

    The screening target y_t is the simple cross-portfolio mean, computed
    month-by-month ignoring NaN columns so that partially missing months
    are handled gracefully.

    Parameters
    ----------
    test_asset_returns : pd.DataFrame
        Wide panel of portfolio returns (rows = months, cols = portfolios).

    Returns
    -------
    pd.Series
        Monthly equal-weighted target, indexed identically to
        ``test_asset_returns``. Named ``'ew_target'``.
    """
    return test_asset_returns.mean(axis=1).rename("ew_target")


def fit_screen_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: Optional[Dict] = None,
) -> HistGradientBoostingRegressor:
    """
    Fit a HistGradientBoostingRegressor with the paper's hyperparameters.

    Parameters
    ----------
    X_train : pd.DataFrame
        Feature matrix (months × factors). NaN values are handled natively
        by the estimator.
    y_train : pd.Series
        Target return series aligned with X_train rows.
    params : dict, optional
        Override any subset of the default hyperparameters
        ``{'learning_rate': 0.01, 'max_iter': 100, 'max_depth': 3}``.
        Keys not present in params fall back to the paper defaults.

    Returns
    -------
    HistGradientBoostingRegressor
        Fitted model.
    """
    hgbr_params = {**_DEFAULT_HGBR_PARAMS, **(params or {})}
    model = HistGradientBoostingRegressor(**hgbr_params)
    model.fit(X_train, y_train)
    return model


def compute_shap_ranking(
    model: HistGradientBoostingRegressor,
    X: pd.DataFrame,
    feature_names: Optional[List[str]] = None,
) -> pd.Series:
    """
    Compute mean absolute SHAP values per feature and return a ranked series.

    Attempts ``shap.Explainer`` first (the unified interface); falls back to
    ``shap.TreeExplainer`` if the unified interface raises an exception (e.g.
    due to SHAP version constraints).

    Parameters
    ----------
    model : HistGradientBoostingRegressor
        Fitted HGBR model.
    X : pd.DataFrame
        Data on which to evaluate SHAP values. Typically the full training
        set or the held-out fold.
    feature_names : list of str, optional
        Column names for the returned Series. Defaults to ``X.columns`` when
        X is a DataFrame, or integer indices otherwise.

    Returns
    -------
    pd.Series
        Mean |SHAP| per feature, sorted descending. Index is feature names.
        Named ``'mean_abs_shap'``.
    """
    names = feature_names if feature_names is not None else (
        list(X.columns) if hasattr(X, "columns") else list(range(X.shape[1]))
    )

    X_arr = X.values if hasattr(X, "values") else np.asarray(X)

    try:
        explainer = shap.Explainer(model, X_arr)
        shap_values = explainer(X_arr).values
    except Exception:
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_arr)
        except Exception as exc:
            raise RuntimeError(
                f"Both shap.Explainer and shap.TreeExplainer failed: {exc}"
            ) from exc

    mean_abs = np.abs(shap_values).mean(axis=0)
    total = mean_abs.sum()
    if total > 0:
        mean_abs = mean_abs / total
    ranking = pd.Series(mean_abs, index=names, name="mean_abs_shap").sort_values(
        ascending=False
    )
    return ranking


def select_by_cumulative_share(
    ranking: pd.Series,
    threshold: float = 0.90,
) -> List[str]:
    """
    Select the smallest set of top-ranked factors whose cumulative SHAP
    share first meets or exceeds ``threshold``.

    Factors are added greedily in descending importance order.  The factor
    that causes the running total to cross ``threshold`` is *included* so
    that the returned selection always covers at least that share.

    Parameters
    ----------
    ranking : pd.Series
        Mean absolute SHAP values sorted descending (as from
        :func:`compute_shap_ranking`).  Values should already be normalised
        to sum to 1 when coming from :func:`compute_shap_ranking`.
    threshold : float
        Cumulative share cut-off (default 0.90 = 90 %).

    Returns
    -------
    list of str
        Names of the selected factors in importance order.
    """
    total = ranking.sum()
    if total == 0:
        return []

    cumulative = 0.0
    selected: List[str] = []
    for name, value in ranking.items():
        share = value / total
        cumulative += share
        selected.append(name)
        if cumulative >= threshold:
            break

    return selected


def run_walk_forward_splits(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    mode: str = "expanding",
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate time-series cross-validation index pairs.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (months × factors). Only the index/length is used.
    y : pd.Series
        Target series aligned with X.
    n_splits : int
        Number of folds (default 5).
    mode : str
        ``'expanding'``: each training fold grows by one block (sklearn
        TimeSeriesSplit default). ``'rolling'``: training window is fixed at
        the same size as the first expanding fold; older observations are
        dropped.

    Returns
    -------
    list of (train_idx, val_idx)
        Each element is a tuple of integer index arrays suitable for
        ``X.iloc[train_idx]`` / ``X.iloc[val_idx]``.
    """
    if mode not in ("expanding", "rolling"):
        raise ValueError(f"mode must be 'expanding' or 'rolling', got '{mode}'.")

    tscv = TimeSeriesSplit(n_splits=n_splits)
    splits = [(train, val) for train, val in tscv.split(X)]

    if mode == "rolling":
        # Fix window size to match the length of the first training fold.
        first_train_size = len(splits[0][0])
        rolling_splits = []
        for train_idx, val_idx in splits:
            # Keep only the most recent `first_train_size` observations.
            if len(train_idx) > first_train_size:
                train_idx = train_idx[-first_train_size:]
            rolling_splits.append((train_idx, val_idx))
        splits = rolling_splits

    return splits


def compute_selection_stability(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    threshold: float = 0.90,
    mode: str = "expanding",
    params: Optional[Dict] = None,
) -> pd.DataFrame:
    """
    Measure how consistently each factor is selected across CV folds.

    For each fold the model is fit on the training set, SHAP importances
    are computed on the validation set, and factors are selected by the
    cumulative-share rule. The returned DataFrame records how often each
    factor was selected and its average SHAP rank across folds.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (months × factors).
    y : pd.Series
        Target series.
    n_splits : int
        Number of time-series CV folds.
    threshold : float
        Cumulative SHAP share cut-off used in each fold (default 0.90).
    mode : str
        ``'expanding'`` or ``'rolling'`` window.
    params : dict, optional
        HGBR hyperparameter overrides.

    Returns
    -------
    pd.DataFrame
        Index: factor names.  Columns:
        - ``selection_count`` (int): number of folds in which the factor
          was selected.
        - ``selection_freq`` (float): fraction of folds (0–1).
        - ``avg_shap_rank`` (float): average rank (1 = most important) across
          folds in which the factor appeared in the SHAP ranking.
    """
    splits = run_walk_forward_splits(X, y, n_splits=n_splits, mode=mode)
    feature_names = list(X.columns)
    n_features = len(feature_names)

    selection_counts = {f: 0 for f in feature_names}
    shap_ranks: Dict[str, List[float]] = {f: [] for f in feature_names}

    for train_idx, val_idx in splits:
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_val = X.iloc[val_idx]

        model = fit_screen_model(X_tr, y_tr, params=params)

        try:
            ranking = compute_shap_ranking(model, X_val, feature_names=feature_names)
        except RuntimeError as exc:
            warnings.warn(f"SHAP computation failed on a CV fold: {exc}")
            continue

        selected = select_by_cumulative_share(ranking, threshold=threshold)
        for f in selected:
            selection_counts[f] += 1

        # Record rank of each feature (1 = most important).
        for rank, f in enumerate(ranking.index, start=1):
            shap_ranks[f].append(float(rank))

    n_folds = len(splits)
    records = []
    for f in feature_names:
        avg_rank = float(np.mean(shap_ranks[f])) if shap_ranks[f] else float(n_features)
        records.append(
            {
                "factor": f,
                "selection_count": selection_counts[f],
                "selection_freq": selection_counts[f] / n_folds if n_folds > 0 else 0.0,
                "avg_shap_rank": avg_rank,
            }
        )

    return (
        pd.DataFrame(records)
        .set_index("factor")
        .sort_values("selection_freq", ascending=False)
    )


def run_screening_pipeline(
    option_factor_returns: pd.DataFrame,
    test_asset_returns: pd.DataFrame,
    threshold: float = 0.90,
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict:
    """
    Execute the full factor screening pipeline on the aligned panels.

    Steps
    -----
    1. Build the equal-weighted target from test-asset returns.
    2. Align X (option factor returns) and y (target) on the common date index.
    3. Fit the HGBR model on the full aligned sample.
    4. Compute SHAP importances and rank factors.
    5. Select factors by cumulative SHAP share ≤ ``threshold``.
    6. Measure selection stability via walk-forward CV.
    7. Collect per-fold validation diagnostics (R² on train and val sets).

    Parameters
    ----------
    option_factor_returns : pd.DataFrame
        Monthly factor return panel (rows = months, cols = factors).
        NaN values are tolerated by the HGBR estimator.
    test_asset_returns : pd.DataFrame
        Monthly test-asset return panel (rows = months, cols = portfolios).
    threshold : float
        Cumulative SHAP share cut-off for factor selection (default 0.90).
    n_splits : int
        Number of walk-forward CV folds used for stability assessment.
    random_state : int
        Random seed passed to the HGBR ``random_state`` parameter.

    Returns
    -------
    dict
        Keys:
        - ``selected_factors`` (list of str): factors selected on full sample.
        - ``cumulative_share`` (float): actual cumulative SHAP share of
          selected factors.
        - ``shap_ranking`` (pd.Series): mean |SHAP| per factor, sorted
          descending.
        - ``full_shap_table`` (pd.DataFrame): shap_ranking as a DataFrame
          with columns ``mean_abs_shap`` and ``cumulative_share``.
        - ``validation_diagnostics`` (list of dict): per-fold train/val R².
        - ``stability_by_factor`` (pd.DataFrame): selection frequency and
          average SHAP rank per factor across CV folds.
    """
    # Build target
    y_full = build_target(test_asset_returns)

    # Align on common dates
    common_idx = option_factor_returns.index.intersection(y_full.index)
    if len(common_idx) == 0:
        raise ValueError(
            "option_factor_returns and test_asset_returns share no common dates."
        )
    X = option_factor_returns.loc[common_idx]
    y = y_full.loc[common_idx]

    hgbr_params = {**_DEFAULT_HGBR_PARAMS, "random_state": random_state}

    # Full-sample model and SHAP ranking
    full_model = fit_screen_model(X, y, params=hgbr_params)
    shap_ranking = compute_shap_ranking(full_model, X, feature_names=list(X.columns))
    selected_factors = select_by_cumulative_share(
        shap_ranking, threshold=threshold
    )

    # Full SHAP table with cumulative share column
    total_shap = shap_ranking.sum()
    cum_series = (shap_ranking / total_shap).cumsum() if total_shap > 0 else shap_ranking * 0
    cumulative_share = float(shap_ranking[selected_factors].sum()) if selected_factors else 0.0
    full_shap_table = pd.DataFrame(
        {"mean_abs_shap": shap_ranking, "cumulative_share": cum_series}
    )

    # Per-fold validation diagnostics
    splits = run_walk_forward_splits(X, y, n_splits=n_splits, mode="expanding")
    validation_diagnostics: List[Dict] = []
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        fold_model = fit_screen_model(X_tr, y_tr, params=hgbr_params)
        train_r2 = float(fold_model.score(X_tr, y_tr))
        val_r2 = float(fold_model.score(X_val, y_val))
        validation_diagnostics.append(
            {
                "fold": fold_idx,
                "train_size": len(train_idx),
                "val_size": len(val_idx),
                "train_r2": train_r2,
                "val_r2": val_r2,
            }
        )

    # Selection stability across folds
    stability_by_factor = compute_selection_stability(
        X, y, n_splits=n_splits, threshold=threshold, mode="expanding",
        params=hgbr_params,
    )

    return {
        "selected_factors": selected_factors,
        "cumulative_share": cumulative_share,
        "shap_ranking": shap_ranking,
        "full_shap_table": full_shap_table,
        "validation_diagnostics": validation_diagnostics,
        "stability_by_factor": stability_by_factor,
    }


# ---------------------------------------------------------------------------
# Public aliases matching test-expected names
# ---------------------------------------------------------------------------

def fit_hgbr_model(
    X: pd.DataFrame,
    y: pd.Series,
    params: Optional[Dict] = None,
) -> HistGradientBoostingRegressor:
    """Fit a HistGradientBoostingRegressor. Alias for fit_screen_model."""
    return fit_screen_model(X, y, params=params)


def walk_forward_splits(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    mode: str = "expanding",
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Time-series CV splits. Alias for run_walk_forward_splits."""
    return run_walk_forward_splits(X, y, n_splits=n_splits, mode=mode)
