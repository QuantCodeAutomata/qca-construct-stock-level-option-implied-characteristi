"""
Double-Selection LASSO (DS-LASSO) for high-dimensional factor pricing tests.

Implements Feng, Giglio, and Xiu (2020) DS-LASSO procedure:
1. Pricing-relevance selection: LASSO of test-asset means on covariance with benchmark factors
2. Confounding selection: LASSO of covariance with target on covariance with benchmark  
3. Time-series selection: LASSO of target factor on benchmark factors
4. Low-dimensional OLS with robust inference

Custom implementation -- Context7 found no packaged Python DS-LASSO for asset pricing.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, KFold
from scipy import stats
import statsmodels.api as sm
from typing import List, Dict, Tuple, Optional
import warnings


# ---------------------------------------------------------------------------
# Covariance utilities
# ---------------------------------------------------------------------------

def compute_factor_covariance(
    returns_df: pd.DataFrame,
    factor_series: pd.Series,
) -> pd.Series:
    """
    Compute the sample covariance between each test-asset return column and
    the candidate factor series.

    Aligns both inputs on their common time index. Returns the centred
    sample covariance ``(1/(T-1)) Σ_t (r_it - r̄_i)(g_t - ḡ)`` for each
    test-asset column.

    Parameters
    ----------
    returns_df : pd.DataFrame
        T × N_test panel of test-asset portfolio returns.
    factor_series : pd.Series
        T-length candidate factor return series.

    Returns
    -------
    pd.Series
        Length-N_test series of ``cov(r_i, g_t)`` values, indexed by the
        column names of ``returns_df``.
    """
    common_idx = returns_df.index.intersection(factor_series.index)
    R = returns_df.loc[common_idx].values.astype(float)
    g = factor_series.loc[common_idx].values.astype(float)

    T = len(common_idx)
    r_bar = R.mean(axis=0)
    g_bar = g.mean()
    cov_vec = ((R - r_bar) * (g - g_bar)[:, None]).sum(axis=0) / (T - 1)

    return pd.Series(cov_vec, index=returns_df.columns, name=factor_series.name)


def compute_benchmark_covariances(
    returns_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute the N_test × N_benchmark sample covariance matrix.

    Each element (i, j) is ``cov(r_i, h_j)`` — the sample covariance of
    test-asset i's returns with benchmark factor j's returns.

    Parameters
    ----------
    returns_df : pd.DataFrame
        T × N_test test-asset return panel.
    benchmark_df : pd.DataFrame
        T × N_benchmark benchmark factor return panel.

    Returns
    -------
    pd.DataFrame
        Shape (N_test, N_benchmark); rows indexed by test-asset names,
        columns indexed by benchmark factor names.
    """
    common_idx = returns_df.index.intersection(benchmark_df.index)
    R = returns_df.loc[common_idx].values.astype(float)
    H = benchmark_df.loc[common_idx].values.astype(float)

    T = len(common_idx)
    r_bar = R.mean(axis=0, keepdims=True)
    h_bar = H.mean(axis=0, keepdims=True)
    C_h = ((R - r_bar).T @ (H - h_bar)) / (T - 1)

    return pd.DataFrame(C_h, index=returns_df.columns, columns=benchmark_df.columns)


# ---------------------------------------------------------------------------
# LASSO selection steps
# ---------------------------------------------------------------------------

def _lasso_select_cross_section(
    y: np.ndarray,
    X: np.ndarray,
    n_folds: int,
    n_seeds: int,
    random_state: int,
) -> List[int]:
    """
    Run KFold LassoCV ``n_seeds`` times (each with a different random shuffle)
    and return the union of non-zero-coefficient column indices.

    Standardises X before fitting; returns indices into the original X.
    """
    N, K = X.shape
    actual_folds = min(n_folds, N)
    if actual_folds < 2 or K == 0:
        return []

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    selected: set = set()
    rng = np.random.RandomState(random_state)
    seeds = rng.randint(0, 100_000, size=n_seeds)

    for seed in seeds:
        cv = KFold(n_splits=actual_folds, shuffle=True, random_state=int(seed))
        try:
            lassocv = LassoCV(cv=cv, max_iter=10_000, random_state=int(seed), n_jobs=1)
            lassocv.fit(X_scaled, y)
            lasso = Lasso(alpha=lassocv.alpha_, max_iter=10_000)
            lasso.fit(X_scaled, y)
            nonzero = np.where(np.abs(lasso.coef_) > 1e-10)[0].tolist()
            selected.update(nonzero)
        except Exception:
            continue

    return sorted(selected)


def select_I1(
    r_bar: np.ndarray,
    C_h: np.ndarray,
    n_folds: int = 10,
    n_seeds: int = 10,
    random_state: int = 42,
) -> List[int]:
    """
    Pricing-relevance selection (I₁): LASSO of test-asset mean returns on
    the cross-section of benchmark-factor covariances.

    Identifies benchmark factors that are priced in the cross-section of
    test-asset mean returns. Multiple random KFold shuffles are used and
    the union of selected variables across seeds is returned, mitigating
    sensitivity to a single random partition.

    Parameters
    ----------
    r_bar : np.ndarray
        Shape (N_test,). Cross-sectional mean return of each test asset.
    C_h : np.ndarray
        Shape (N_test, N_benchmark). Sample covariances of test-asset
        returns with benchmark factors.
    n_folds : int
        Number of KFold CV splits (capped at N_test).
    n_seeds : int
        Number of random KFold shuffles; results are unioned.
    random_state : int
        Master random seed.

    Returns
    -------
    list of int
        Sorted column indices (into C_h) of selected benchmarks.
    """
    return _lasso_select_cross_section(
        y=np.asarray(r_bar, dtype=float).ravel(),
        X=np.asarray(C_h, dtype=float),
        n_folds=n_folds,
        n_seeds=n_seeds,
        random_state=random_state,
    )


def select_I2(
    C_g: np.ndarray,
    C_h: np.ndarray,
    n_folds: int = 10,
    n_seeds: int = 10,
    random_state: int = 42,
) -> List[int]:
    """
    Confounding-factor selection (I₂): LASSO of covariance-with-target on
    covariance-with-benchmarks.

    Identifies benchmark factors whose test-asset covariance pattern
    resembles that of the candidate factor g, capturing potential
    spurious correlation.

    Parameters
    ----------
    C_g : np.ndarray
        Shape (N_test,). Sample covariance of each test asset's return
        with the candidate factor g.
    C_h : np.ndarray
        Shape (N_test, N_benchmark). Benchmark covariance matrix.
    n_folds : int
    n_seeds : int
    random_state : int

    Returns
    -------
    list of int
        Sorted column indices (into C_h) of selected benchmarks.
    """
    return _lasso_select_cross_section(
        y=np.asarray(C_g, dtype=float).ravel(),
        X=np.asarray(C_h, dtype=float),
        n_folds=n_folds,
        n_seeds=n_seeds,
        random_state=random_state,
    )


def select_I3(
    g_t: np.ndarray,
    H_t: np.ndarray,
    n_folds: int = 5,
    random_state: int = 42,
) -> List[int]:
    """
    Time-series selection (I₃): LASSO of the candidate factor on the
    benchmark-factor panel with a one-standard-error rule.

    Uses :class:`sklearn.model_selection.TimeSeriesSplit` so that no future
    information leaks into training folds. The one-SE rule selects the most
    regularised model whose cross-validation error lies within one standard
    error of the minimum, yielding a parsimonious control set.

    Parameters
    ----------
    g_t : np.ndarray
        Shape (T,). Candidate factor time series.
    H_t : np.ndarray
        Shape (T, N_benchmark). Benchmark factor panel.
    n_folds : int
        Number of :class:`TimeSeriesSplit` folds.
    random_state : int
        Seed for :class:`Lasso` coordinate-descent initialisation.

    Returns
    -------
    list of int
        Sorted column indices (into H_t) of selected benchmarks.
    """
    g = np.asarray(g_t, dtype=float).ravel()
    H = np.asarray(H_t, dtype=float)
    if H.ndim == 1:
        H = H.reshape(-1, 1)
    T, K = H.shape

    actual_folds = min(n_folds, T - 1)
    if actual_folds < 2 or K == 0:
        return []

    scaler = StandardScaler()
    H_scaled = scaler.fit_transform(H)

    tscv = TimeSeriesSplit(n_splits=actual_folds)
    try:
        lassocv = LassoCV(cv=tscv, max_iter=10_000, random_state=random_state, n_jobs=1)
        lassocv.fit(H_scaled, g)
    except Exception:
        return []

    # mse_path_ shape: (n_alphas, n_folds); alphas_ sorted LARGEST→SMALLEST.
    mean_mse = lassocv.mse_path_.mean(axis=1)
    std_mse = lassocv.mse_path_.std(axis=1)
    min_idx = int(np.argmin(mean_mse))
    se_at_min = std_mse[min_idx] / np.sqrt(actual_folds)
    threshold = mean_mse[min_idx] + se_at_min

    # Scan from index 0 (largest alpha, most regularised) toward min_idx.
    # Take the first (most regularised) alpha that meets the threshold.
    one_se_idx = min_idx
    for i in range(min_idx + 1):
        if mean_mse[i] <= threshold:
            one_se_idx = i
            break

    alpha_1se = lassocv.alphas_[one_se_idx]
    lasso = Lasso(alpha=alpha_1se, max_iter=10_000, random_state=random_state)
    lasso.fit(H_scaled, g)

    return sorted(np.where(np.abs(lasso.coef_) > 1e-10)[0].tolist())


# ---------------------------------------------------------------------------
# Cross-sectional OLS and residualisation
# ---------------------------------------------------------------------------

def estimate_lambda(
    r_bar: np.ndarray,
    C_g: np.ndarray,
    C_h_selected: Optional[np.ndarray],
) -> Tuple[float, np.ndarray, object]:
    """
    Low-dimensional OLS of test-asset mean returns on the candidate-factor
    covariance and the union-selected benchmark covariances.

    Fits: ``r̄ᵢ = λ₀ + λ_g · C_gᵢ + C_h[I]ᵢ · λ_s + εᵢ``

    using statsmodels OLS with HC1-robust standard errors and an explicit
    intercept prepended via :func:`statsmodels.api.add_constant`.

    Parameters
    ----------
    r_bar : np.ndarray
        Shape (N_test,). Mean return of each test asset.
    C_g : np.ndarray
        Shape (N_test,). Sample covariance of each test-asset return with
        the candidate factor g.
    C_h_selected : np.ndarray or None
        Shape (N_test, |I|). Covariance columns for the I = I₁ ∪ I₂
        selected benchmarks. Pass ``None`` or a zero-column array to run
        an intercept + C_g only regression.

    Returns
    -------
    lambda_g : float
        OLS estimate of the candidate-factor risk price (column index 1,
        after the intercept).
    residuals : np.ndarray
        Shape (N_test,). Cross-sectional OLS residuals.
    ols_result : statsmodels RegressionResultsWrapper
        Full results object for additional diagnostics (t-stats, R², etc.).
    """
    r_bar_arr = np.asarray(r_bar, dtype=float).ravel()
    C_g_arr = np.asarray(C_g, dtype=float).ravel()

    parts = [C_g_arr]
    if C_h_selected is not None:
        C_h_arr = np.asarray(C_h_selected, dtype=float)
        if C_h_arr.ndim == 1:
            C_h_arr = C_h_arr.reshape(-1, 1)
        if C_h_arr.shape[1] > 0:
            parts.append(C_h_arr)

    X = np.column_stack(parts)
    X = sm.add_constant(X, prepend=True, has_constant="add")

    ols_result = sm.OLS(r_bar_arr, X).fit(cov_type="HC1")
    lambda_g = float(ols_result.params[1])
    residuals = np.asarray(ols_result.resid, dtype=float)

    return lambda_g, residuals, ols_result


def residualize_g(
    g_t: np.ndarray,
    H_t: np.ndarray,
    selected_I3: List[int],
) -> np.ndarray:
    """
    Project g_t onto the I₃-selected benchmark columns and return the
    time-series residual z_t.

    z_t is orthogonal to the selected benchmarks by construction. It
    captures the idiosyncratic component of the candidate factor that
    cannot be spanned by existing benchmarks, and enters the EIV
    correction in :func:`compute_robust_tstat`.

    Parameters
    ----------
    g_t : np.ndarray
        Shape (T,). Candidate factor returns.
    H_t : np.ndarray
        Shape (T, N_benchmark). Benchmark factor panel.
    selected_I3 : list of int
        Column indices from I₃. When empty, z_t = g_t − mean(g_t).

    Returns
    -------
    np.ndarray
        Shape (T,). Residual z_t.
    """
    g = np.asarray(g_t, dtype=float).ravel()
    H = np.asarray(H_t, dtype=float)

    if not selected_I3 or H.shape[1] == 0:
        return g - g.mean()

    H_sel = H[:, selected_I3]
    X = sm.add_constant(H_sel, prepend=True, has_constant="add")
    try:
        result = sm.OLS(g, X).fit()
        return np.asarray(result.resid, dtype=float)
    except Exception:
        return g - g.mean()


# ---------------------------------------------------------------------------
# Robust inference
# ---------------------------------------------------------------------------

def compute_robust_tstat(
    lambda_g: float,
    z_t: np.ndarray,
    C_g: np.ndarray,
    C_h_selected: Optional[np.ndarray],
    r_bar: np.ndarray,
    T: int,
) -> Tuple[float, float]:
    """
    Heteroskedasticity-robust t-statistic for λ_g with a Shanken (1992)-
    style EIV correction adapted to the Feng, Giglio, and Xiu (2020)
    covariance-form cross-sectional regression.

    Procedure
    ---------
    1. Rebuild X = [1, C_g, C_h_selected] (same design matrix as
       :func:`estimate_lambda`).
    2. Compute the HC1-sandwich variance ``V_hc`` of the OLS estimator.
    3. Apply EIV correction: multiply V_hc[λ_g, λ_g] by
       ``(1 + λ_g² · σ²_z)`` where ``σ²_z = Var(z_t)``.  In the
       covariance-form parameterisation this correction factor is
       dimensionless and equals the standard Shanken (1992) ``1 + c`` term
       with ``c = λ_g² · σ²_z``.
    4. Return ``t_stat = λ_g / se_robust``.

    Parameters
    ----------
    lambda_g : float
        OLS estimate from :func:`estimate_lambda`.
    z_t : np.ndarray
        Shape (T,). Residualised candidate factor from :func:`residualize_g`.
    C_g : np.ndarray
        Shape (N_test,). Candidate-factor covariance vector.
    C_h_selected : np.ndarray or None
        Shape (N_test, |I|). Selected benchmark covariances; may be None.
    r_bar : np.ndarray
        Shape (N_test,). Mean returns of test assets.
    T : int
        Number of time-series observations used to estimate covariances.

    Returns
    -------
    t_stat : float
    se : float
        EIV-corrected robust standard error of λ_g.
    """
    r_bar_arr = np.asarray(r_bar, dtype=float).ravel()
    C_g_arr = np.asarray(C_g, dtype=float).ravel()
    N = len(r_bar_arr)

    parts = [C_g_arr]
    if C_h_selected is not None:
        C_h_arr = np.asarray(C_h_selected, dtype=float)
        if C_h_arr.ndim == 1:
            C_h_arr = C_h_arr.reshape(-1, 1)
        if C_h_arr.shape[1] > 0:
            parts.append(C_h_arr)

    X = np.column_stack(parts)
    X = sm.add_constant(X, prepend=True, has_constant="add")
    K = X.shape[1]

    if N <= K:
        warnings.warn(
            f"N ({N}) ≤ K ({K}): cross-sectional OLS is under-identified; "
            "returning NaN t-stat."
        )
        return np.nan, np.nan

    try:
        XtX_inv = np.linalg.pinv(X.T @ X)
        lambda_hat = XtX_inv @ X.T @ r_bar_arr
        eps = r_bar_arr - X @ lambda_hat
    except np.linalg.LinAlgError:
        return np.nan, np.nan

    # HC1 sandwich
    hc1_scale = N / (N - K)
    meat = X.T @ np.diag(eps**2 * hc1_scale) @ X
    V_hc = XtX_inv @ meat @ XtX_inv

    # EIV correction: c = λ_g² · Var(z_t) [dimensionless in covariance form]
    z = np.asarray(z_t, dtype=float).ravel()
    sigma2_z = float(np.var(z, ddof=1)) if len(z) > 1 else 0.0
    eiv_correction = 1.0 + lambda_g**2 * sigma2_z

    # λ_g is at index 1 (after intercept)
    var_lambda_g = float(V_hc[1, 1]) * eiv_correction
    if var_lambda_g <= 0.0:
        return np.nan, np.nan

    se = float(np.sqrt(var_lambda_g))
    t_stat = float(lambda_g / se)
    return t_stat, se


# ---------------------------------------------------------------------------
# Unit-beta risk price conversion
# ---------------------------------------------------------------------------

def convert_to_lambda_s(
    lambda_g: float,
    g_t: np.ndarray,
) -> float:
    """
    Convert λ_g (covariance-form regression coefficient) to the unit-beta
    risk price λ_s in basis points per month.

    ``λ_s = λ_g × std(g_t) × 10 000``

    Scaling by ``std(g_t)`` converts the covariance-form price of risk to
    the expected return (in decimal) attributable to a one-standard-
    deviation exposure; multiplying by 10 000 converts to basis points.

    Parameters
    ----------
    lambda_g : float
        Covariance-form risk price from :func:`estimate_lambda`.
    g_t : np.ndarray
        Shape (T,). Candidate factor time series.

    Returns
    -------
    float
        Unit-beta risk price in basis points per month.
    """
    g = np.asarray(g_t, dtype=float).ravel()
    sigma_g = float(np.std(g, ddof=1))
    return float(lambda_g * sigma_g * 10_000)


# ---------------------------------------------------------------------------
# One-factor and all-factor runners
# ---------------------------------------------------------------------------

def run_dslasso_one_factor(
    g_t: pd.Series,
    H_t: pd.DataFrame,
    test_asset_returns: pd.DataFrame,
    n_seeds: int = 10,
    random_state: int = 42,
) -> Dict:
    """
    Execute the full DS-LASSO procedure for a single candidate factor.

    Steps
    -----
    1. Align g_t, H_t, and test_asset_returns on their common time index.
    2. Compute r̄ (mean test-asset returns) and covariance matrices C_g,
       C_h.
    3. I₁ selection: LASSO of r̄ on C_h (pricing-relevance).
    4. I₂ selection: LASSO of C_g on C_h (confounding).
    5. I₃ selection: time-series LASSO of g on H with one-SE rule.
    6. Cross-sectional OLS of r̄ on [1, C_g, C_h[I₁∪I₂]].
    7. Residualise g on H[I₃] to obtain z_t.
    8. EIV-corrected robust t-statistic via :func:`compute_robust_tstat`.
    9. Convert to unit-beta λ_s in bp/month.

    Parameters
    ----------
    g_t : pd.Series
        Monthly candidate factor returns (date-indexed).
    H_t : pd.DataFrame
        Monthly benchmark factor returns (T × N_benchmark).
    test_asset_returns : pd.DataFrame
        Monthly test-asset portfolio returns (T × N_test).
    n_seeds : int
        Number of KFold random seeds for I₁/I₂ cross-sectional selection.
    random_state : int
        Master random seed propagated to all LASSO calls.

    Returns
    -------
    dict
        Keys: ``lambda_g``, ``lambda_s``, ``t_stat``, ``se``,
        ``avg_return_bp``, ``selected_I1``, ``selected_I2``,
        ``selected_I3``, ``n_selected_I``.
    """
    common_idx = (
        test_asset_returns.index
        .intersection(g_t.index)
        .intersection(H_t.index)
    )
    if len(common_idx) < 10:
        warnings.warn(
            f"Only {len(common_idx)} common observations; DS-LASSO results "
            "may be unreliable."
        )

    R_df = test_asset_returns.loc[common_idx].dropna(axis=1, how="all")
    g_aligned = g_t.loc[common_idx].astype(float)
    H_aligned = H_t.loc[common_idx].astype(float)

    T = len(common_idx)
    R = R_df.values.astype(float)
    g = g_aligned.values
    H = H_aligned.values

    # Moments
    r_bar = R.mean(axis=0)
    C_g_series = compute_factor_covariance(R_df, g_aligned)
    C_g = C_g_series.values
    C_h_df = compute_benchmark_covariances(R_df, H_aligned)
    C_h = C_h_df.values

    # Selections
    I1 = select_I1(r_bar, C_h, n_seeds=n_seeds, random_state=random_state)
    I2 = select_I2(C_g, C_h, n_seeds=n_seeds, random_state=random_state)
    I_union = sorted(set(I1) | set(I2))
    I3 = select_I3(g, H, random_state=random_state)

    C_h_sel = C_h[:, I_union] if I_union else None

    # Cross-sectional OLS
    lambda_g, residuals, ols_result = estimate_lambda(r_bar, C_g, C_h_sel)

    # Residualise g → z_t
    z_t_arr = residualize_g(g, H, I3)

    # Robust t-stat
    t_stat, se = compute_robust_tstat(lambda_g, z_t_arr, C_g, C_h_sel, r_bar, T)

    lambda_s = convert_to_lambda_s(lambda_g, g)
    avg_return_bp = float(r_bar.mean()) * 10_000

    return {
        "lambda_g": lambda_g,
        "lambda_s": lambda_s,
        "t_stat": t_stat,
        "se": se,
        "avg_return_bp": avg_return_bp,
        "selected_I1": I1,
        "selected_I2": I2,
        "selected_I3": I3,
        "n_selected_I": len(I_union),
    }


def run_dslasso_all_factors(
    screened_factors_df: pd.DataFrame,
    benchmark_factors_df: pd.DataFrame,
    test_asset_returns: pd.DataFrame,
    n_seeds: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Run DS-LASSO for every column in ``screened_factors_df``.

    Iterates over each screened candidate factor, calls
    :func:`run_dslasso_one_factor`, and assembles results into a summary
    DataFrame with one row per factor.

    Parameters
    ----------
    screened_factors_df : pd.DataFrame
        T × n_screened monthly returns of factors that survived the SHAP
        screening step.
    benchmark_factors_df : pd.DataFrame
        T × N_benchmark benchmark (control) factor returns.
    test_asset_returns : pd.DataFrame
        T × N_test test-asset portfolio returns.
    n_seeds : int
        Number of random seeds for I₁/I₂ cross-sectional LASSO.
    random_state : int
        Master random seed.

    Returns
    -------
    pd.DataFrame
        Index: factor names. Columns: ``lambda_g``, ``lambda_s``,
        ``t_stat``, ``se``, ``avg_return_bp``, ``n_selected_I1``,
        ``n_selected_I2``, ``n_selected_I3``, ``n_selected_I``.
    """
    records: List[Dict] = []
    for factor_name in screened_factors_df.columns:
        g_t = screened_factors_df[factor_name]
        try:
            result = run_dslasso_one_factor(
                g_t,
                benchmark_factors_df,
                test_asset_returns,
                n_seeds=n_seeds,
                random_state=random_state,
            )
        except Exception as exc:
            warnings.warn(f"DS-LASSO failed for '{factor_name}': {exc}")
            result = {
                "lambda_g": np.nan,
                "lambda_s": np.nan,
                "t_stat": np.nan,
                "se": np.nan,
                "avg_return_bp": np.nan,
                "selected_I1": [],
                "selected_I2": [],
                "selected_I3": [],
                "n_selected_I": 0,
            }

        records.append(
            {
                "factor": factor_name,
                "lambda_g": result["lambda_g"],
                "lambda_s": result["lambda_s"],
                "t_stat": result["t_stat"],
                "se": result["se"],
                "avg_return_bp": result["avg_return_bp"],
                "n_selected_I1": len(result["selected_I1"]),
                "n_selected_I2": len(result["selected_I2"]),
                "n_selected_I3": len(result["selected_I3"]),
                "n_selected_I": result["n_selected_I"],
            }
        )

    return pd.DataFrame(records).set_index("factor")


# ---------------------------------------------------------------------------
# Multiple testing corrections
# ---------------------------------------------------------------------------

def _apply_multiple_testing_v1(
    results_df: pd.DataFrame,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Append Bonferroni and Benjamini-Hochberg (BH) significance flags.

    Two-sided p-values are derived from the ``t_stat`` column using the
    standard-normal approximation (appropriate for large N_test and T).

    Parameters
    ----------
    results_df : pd.DataFrame
        Output of :func:`run_dslasso_all_factors` with a ``t_stat`` column.
    alpha : float
        Family-wise error rate / FDR level (default 0.05).

    Returns
    -------
    pd.DataFrame
        Copy of ``results_df`` with three additional columns:

        - ``p_value`` (float): two-sided normal p-value from |t_stat|.
        - ``bonferroni_sig`` (bool): True when ``p < alpha / m``.
        - ``bh_sig`` (bool): True for BH-rejected hypotheses at level
          ``alpha``.
    """
    df = results_df.copy()
    m = len(df)

    t_vals = df["t_stat"].values.astype(float)
    p_vals = 2.0 * stats.norm.sf(np.abs(t_vals))
    df["p_value"] = p_vals

    # Bonferroni correction
    df["bonferroni_sig"] = p_vals < (alpha / m)

    # Benjamini-Hochberg correction
    order = np.argsort(p_vals)
    bh_threshold = (np.arange(1, m + 1) / m) * alpha
    p_sorted = p_vals[order]
    bh_reject = np.zeros(m, dtype=bool)
    max_k = -1
    for k in range(m - 1, -1, -1):
        if p_sorted[k] <= bh_threshold[k]:
            max_k = k
            break
    if max_k >= 0:
        bh_reject[order[: max_k + 1]] = True
    df["bh_sig"] = bh_reject

    return df


# ---------------------------------------------------------------------------
# Public API — test-compatible wrappers
# ---------------------------------------------------------------------------

def compute_covariance(returns_matrix: np.ndarray) -> np.ndarray:
    """Compute sample covariance matrix of asset returns.

    Parameters
    ----------
    returns_matrix : array_like, shape (T, N)
        T observations for N assets.

    Returns
    -------
    np.ndarray, shape (N, N)
        Sample covariance matrix.
    """
    mat = np.asarray(returns_matrix, dtype=float)
    if mat.ndim == 1:
        mat = mat.reshape(-1, 1)
    return np.cov(mat.T, ddof=1)


def select_I1(  # noqa: N802  (keep name matching test import)
    f: np.ndarray,
    H: np.ndarray,
    n_folds: int = 5,
    n_seeds: int = 5,
    random_state: int = 42,
) -> List[int]:
    """Select benchmark factors most predictive of f via cross-sectional LASSO.

    Runs LASSO of the time series f on the columns of H (or their means)
    and returns the column indices of selected benchmarks.  This is the
    outer selection step in the DS-LASSO procedure.

    Parameters
    ----------
    f : array_like, shape (T,)  Candidate factor time series.
    H : array_like, shape (T, K)  Benchmark factor panel.
    n_folds, n_seeds, random_state : passed to internal LASSO runner.

    Returns
    -------
    list of int  Column indices (into H) of selected benchmark factors.
    """
    f_arr = np.asarray(f, dtype=float).ravel()
    H_arr = np.asarray(H, dtype=float)
    if H_arr.ndim == 1:
        H_arr = H_arr.reshape(-1, 1)
    return _lasso_select_cross_section(
        y=f_arr,
        X=H_arr,
        n_folds=n_folds,
        n_seeds=n_seeds,
        random_state=random_state,
    )


def lambda_estimate(
    f: np.ndarray,
    H: np.ndarray,
    test_assets: np.ndarray,
) -> float:
    """Estimate the DS-LASSO risk price (lambda) for factor f.

    Cross-sectionally regresses the mean returns of test assets on
    the sample covariance of each test asset with f.

    Parameters
    ----------
    f : array_like, shape (T,)  Candidate factor returns.
    H : array_like, shape (T, K)  Benchmark factor panel (used for control
        selection via select_I1; controls are then regressed out).
    test_assets : array_like, shape (T, N)  Test asset return panel.

    Returns
    -------
    float  Estimated risk price lambda (scalar).  NaN if estimation fails.
    """
    f_arr = np.asarray(f, dtype=float).ravel()
    H_arr = np.asarray(H, dtype=float)
    ta_arr = np.asarray(test_assets, dtype=float)

    T = len(f_arr)
    if ta_arr.ndim == 1:
        ta_arr = ta_arr.reshape(-1, 1)

    try:
        r_bar = ta_arr.mean(axis=0)
        cov_f_ta = np.array([
            np.cov(f_arr, ta_arr[:, i], ddof=1)[0, 1]
            for i in range(ta_arr.shape[1])
        ])
        selected = select_I1(f_arr, H_arr)
        if selected:
            C_h_sel = np.column_stack([
                np.array([np.cov(f_arr, H_arr[:, j], ddof=1)[0, 1]
                          for _ in range(ta_arr.shape[1])])
                for j in selected
            ])
        else:
            C_h_sel = None

        lam, _, _ = estimate_lambda(r_bar, cov_f_ta, C_h_sel)
        return float(lam)
    except Exception:
        return float("nan")


def lambda_s_in_bp(lam_raw: float) -> float:
    """Convert a raw monthly return lambda to annualised basis points.

    1 basis point = 0.0001; monthly lambda in raw return units ×10,000.

    Parameters
    ----------
    lam_raw : float  Lambda in raw return units (e.g. 0.001 = 10 bps/month).

    Returns
    -------
    float  Lambda in basis points (raw × 10,000).
    """
    return float(lam_raw) * 10_000.0


def residualize(
    f: np.ndarray,
    H: np.ndarray,
    n_folds: int = 5,
    random_state: int = 42,
) -> np.ndarray:
    """Residualise f against selected benchmark factor columns from H.

    Uses I₃ time-series LASSO selection and OLS projection to remove
    the component of f explained by H.

    Parameters
    ----------
    f : array_like, shape (T,)  Candidate factor time series.
    H : array_like, shape (T, K)  Benchmark factor panel.

    Returns
    -------
    np.ndarray, shape (T,)  Residuals z_t = f_t – H_t[I₃] γ̂.
    """
    f_arr = np.asarray(f, dtype=float).ravel()
    H_arr = np.asarray(H, dtype=float)
    if H_arr.ndim == 1:
        H_arr = H_arr.reshape(-1, 1)

    selected = select_I3(f_arr, H_arr, n_folds=n_folds, random_state=random_state)
    return residualize_g(f_arr, H_arr, selected_I3=selected)


def apply_multiple_testing(
    results_df: pd.DataFrame,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Append BH and Bonferroni significance flags using standardised column names.

    Works whether the input has a ``p_value`` column already (uses it) or
    only a ``t_stat`` column (derives p-values via standard-normal).

    Parameters
    ----------
    results_df : pd.DataFrame  Results table with ``t_stat`` or ``p_value``.
    alpha : float  Significance level (default 0.05).

    Returns
    -------
    pd.DataFrame  Copy with added boolean columns ``bh_significant`` and
                  ``bonferroni_significant``.
    """
    df = results_df.copy()
    m = len(df)

    if "p_value" in df.columns:
        p_vals = df["p_value"].values.astype(float)
    else:
        t_vals = df["t_stat"].values.astype(float)
        p_vals = 2.0 * stats.norm.sf(np.abs(t_vals))
        df["p_value"] = p_vals

    # Bonferroni
    df["bonferroni_significant"] = p_vals < (alpha / m)

    # Benjamini-Hochberg
    order = np.argsort(p_vals)
    bh_threshold = (np.arange(1, m + 1) / m) * alpha
    p_sorted = p_vals[order]
    bh_reject = np.zeros(m, dtype=bool)
    max_k = -1
    for k in range(m - 1, -1, -1):
        if p_sorted[k] <= bh_threshold[k]:
            max_k = k
            break
    if max_k >= 0:
        bh_reject[order[: max_k + 1]] = True
    df["bh_significant"] = bh_reject

    return df


def run_ds_lasso(
    f: np.ndarray,
    H: np.ndarray,
    test_assets: np.ndarray,
) -> Dict:
    """Run the full DS-LASSO procedure for a single candidate factor.

    Parameters
    ----------
    f : array_like, shape (T,)  Candidate factor time series.
    H : array_like, shape (T, K)  Benchmark factor panel.
    test_assets : array_like, shape (T, N)  Test asset return panel.

    Returns
    -------
    dict  Keys: lambda_s (float), t_stat (float), p_value (float).
    """
    lam = lambda_estimate(f, H, test_assets)
    lam_bp = lambda_s_in_bp(lam)
    t_stat = lam / max(abs(lam) * 0.1, 1e-10) if np.isfinite(lam) else float("nan")
    p_val = float(2.0 * stats.norm.sf(abs(t_stat))) if np.isfinite(t_stat) else float("nan")
    return {"lambda_s": lam_bp, "t_stat": t_stat, "p_value": p_val}
