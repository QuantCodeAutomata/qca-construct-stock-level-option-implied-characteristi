"""
Joint alpha tests (GRS), maximum Sharpe ratio comparisons, and
canonical correlation redundancy analysis.

Implements:
- GRS test (Gibbons, Ross, Shanken 1989) for joint zero alphas
- Maximum Sharpe ratio with Ledoit-Wolf shrinkage covariance
- Bidirectional canonical correlation and Stewart-Love redundancy index

GRS: custom implementation (Context7: no standard Python package)
Ledoit-Wolf: sklearn.covariance.LedoitWolf (Context7: confirmed)
CCA: scipy.linalg for stable decomposition (custom redundancy)
"""

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from scipy import linalg
from scipy.stats import f as f_dist
import statsmodels.api as sm
from typing import Tuple, Dict, List
import warnings


def grs_test(
    test_asset_returns: pd.DataFrame,
    benchmark_factor_returns: pd.DataFrame,
) -> Dict:
    """GRS (1989) joint alpha test for zero pricing errors.

    Regresses each test asset on benchmark factors via OLS, collects the
    N intercepts (alphas) and the N×N residual covariance matrix, then
    forms the GRS F-statistic:

        F = [(T - N - K) / N] * [α′ Σ⁻¹ α] / [1 + μ_f′ Ω_f⁻¹ μ_f]

    which is distributed F(N, T-N-K) under H₀: α = 0.

    Parameters
    ----------
    test_asset_returns : DataFrame, shape (T, N)
    benchmark_factor_returns : DataFrame, shape (T, K)

    Returns
    -------
    dict with keys:
        grs_stat   – F-statistic
        p_value    – right-tail p-value
        alphas     – array of N intercepts
        T, N, K    – sample dimensions
    """
    R = np.asarray(test_asset_returns, dtype=float)
    F = np.asarray(benchmark_factor_returns, dtype=float)
    T, N = R.shape
    K = F.shape[1]

    if T <= N + K:
        raise ValueError(
            f"Insufficient observations: T={T} must exceed N+K={N + K}."
        )

    # ── OLS for each test asset ──────────────────────────────────────────
    F_const = sm.add_constant(F)          # (T, K+1)
    alphas = np.empty(N)
    residuals = np.empty((T, N))

    for i in range(N):
        res = sm.OLS(R[:, i], F_const).fit()
        alphas[i] = res.params[0]
        residuals[:, i] = res.resid

    # ── MLE residual covariance (divide by T) ────────────────────────────
    Sigma = residuals.T @ residuals / T

    # ── Factor moments ───────────────────────────────────────────────────
    mu_f = F.mean(axis=0)                 # (K,)
    Omega_f = np.cov(F, rowvar=False, ddof=1)   # (K, K)

    Sigma_inv = linalg.inv(Sigma)
    Omega_inv = linalg.inv(Omega_f)

    quad_alpha = float(alphas @ Sigma_inv @ alphas)
    quad_mu = float(mu_f @ Omega_inv @ mu_f)

    grs_stat = ((T - N - K) / N) * quad_alpha / (1.0 + quad_mu)
    p_value = f_dist.sf(grs_stat, dfn=N, dfd=T - N - K)

    return {
        "grs_stat": grs_stat,
        "p_value": float(p_value),
        "alphas": alphas,
        "T": T,
        "N": N,
        "K": K,
    }


def compute_max_sharpe(
    returns_df: pd.DataFrame,
    annualize: bool = True,
    monthly_factor: int = 12,
) -> float:
    """Maximum Sharpe ratio of the mean-variance efficient portfolio.

    Uses Ledoit-Wolf shrinkage for the covariance matrix to avoid
    ill-conditioning when the factor universe is large.

    SR_max = sqrt(μ′ Σ⁻¹ μ)

    Parameters
    ----------
    returns_df    : DataFrame of excess returns, shape (T, p)
    annualize     : multiply by sqrt(monthly_factor) when True
    monthly_factor: periods per year (12 for monthly data)

    Returns
    -------
    float  – maximum Sharpe ratio (annualised by default)
    """
    X = np.asarray(returns_df, dtype=float)
    mu = X.mean(axis=0)

    lw = LedoitWolf().fit(X)
    Sigma = lw.covariance_

    try:
        Sigma_inv = linalg.inv(Sigma)
    except linalg.LinAlgError:
        Sigma_inv = linalg.pinv(Sigma)

    sr = float(np.sqrt(np.clip(mu @ Sigma_inv @ mu, 0.0, None)))
    if annualize:
        sr *= np.sqrt(monthly_factor)
    return sr


def compare_sharpe_ratios(
    equity_factors: pd.DataFrame,
    option_factors: pd.DataFrame,
    rf_series: pd.Series = None,
) -> Dict:
    """Compare maximum Sharpe ratios across three universes.

    Aligns equity and option factor DataFrames on their common index,
    optionally subtracts a risk-free rate, then computes the maximum
    Sharpe ratio for:
      - equity-only universe
      - option-only universe
      - combined universe

    Parameters
    ----------
    equity_factors : DataFrame, shape (T, p_e) of equity factor excess returns
    option_factors : DataFrame, shape (T, p_o) of option factor excess returns
    rf_series      : optional Series of risk-free rates to subtract

    Returns
    -------
    dict with keys:
        sr_equity, sr_option, sr_combined – maximum Sharpe ratios
        sr_improvement                    – (sr_combined - sr_equity) / sr_equity
    """
    idx = equity_factors.index.intersection(option_factors.index)
    eq = equity_factors.loc[idx].copy()
    op = option_factors.loc[idx].copy()

    if rf_series is not None:
        rf = rf_series.reindex(idx).fillna(0.0)
        eq = eq.subtract(rf, axis=0)
        op = op.subtract(rf, axis=0)

    combined = pd.concat([eq, op], axis=1)

    sr_equity = compute_max_sharpe(eq)
    sr_option = compute_max_sharpe(op)
    sr_combined = compute_max_sharpe(combined)

    improvement = (sr_combined - sr_equity) / sr_equity if sr_equity != 0.0 else np.nan

    return {
        "sr_equity": sr_equity,
        "sr_option": sr_option,
        "sr_combined": sr_combined,
        "sr_improvement": improvement,
    }


def canonical_correlation_analysis(
    X: pd.DataFrame,
    Y: pd.DataFrame,
) -> Dict:
    """Canonical correlation analysis between X and Y via SVD.

    Uses QR+SVD decomposition for numerical stability.  Returns all
    canonical correlations together with loading matrices and structure
    coefficients (correlations of original variables with canonical variates).

    Parameters
    ----------
    X : DataFrame, shape (T, p)  – option factors (or any left-side block)
    Y : DataFrame, shape (T, q)  – equity factors (or any right-side block)

    Returns
    -------
    dict with keys:
        canonical_correlations  – array of s = min(p, q) correlations ∈ [0, 1]
        X_loadings              – (p, s) matrix  A  such that U = X_std @ A
        Y_loadings              – (q, s) matrix  B  such that V = Y_std @ B
        X_structure_coeffs      – (p, s) corr(X_j, U_i)
        Y_structure_coeffs      – (q, s) corr(Y_k, V_i)
    """
    Xm = np.asarray(X, dtype=float)
    Ym = np.asarray(Y, dtype=float)
    T = Xm.shape[0]

    # centre
    Xc = Xm - Xm.mean(axis=0)
    Yc = Ym - Ym.mean(axis=0)

    # scale to unit variance (avoid division-by-zero for constant columns)
    sx = Xc.std(axis=0, ddof=1)
    sy = Yc.std(axis=0, ddof=1)
    sx[sx == 0] = 1.0
    sy[sy == 0] = 1.0
    Xs = Xc / sx
    Ys = Yc / sy

    # QR decomposition for stable column-space representation
    Qx, Rx = linalg.qr(Xs, mode="economic")
    Qy, Ry = linalg.qr(Ys, mode="economic")

    # SVD of Qx'Qy  →  canonical correlations as singular values
    M = Qx.T @ Qy
    Ux, rho, VyT = linalg.svd(M, full_matrices=False)
    Vy = VyT.T

    s = min(Xs.shape[1], Ys.shape[1])
    rho = np.clip(rho[:s], 0.0, 1.0)

    # Loading matrices in the original (standardised) coordinate system
    A = linalg.solve(Rx, Ux[:, :s])   # (p, s)
    B = linalg.solve(Ry, Vy[:, :s])   # (q, s)

    # Canonical variates
    U = Xs @ A    # (T, s)
    V = Ys @ B    # (T, s)

    # Structure coefficients: pearson corr of each original var with each variate
    def _structure(Z, W):
        """Pearson r between every column of Z and every column of W."""
        Zc = Z - Z.mean(axis=0)
        Wc = W - W.mean(axis=0)
        cov = Zc.T @ Wc / (T - 1)
        std_Z = Zc.std(axis=0, ddof=1)
        std_W = Wc.std(axis=0, ddof=1)
        std_Z[std_Z == 0] = 1.0
        std_W[std_W == 0] = 1.0
        return cov / np.outer(std_Z, std_W)

    X_sc = _structure(Xs, U)   # (p, s)
    Y_sc = _structure(Ys, V)   # (q, s)

    return {
        "canonical_correlations": rho,
        "X_loadings": A,
        "Y_loadings": B,
        "X_structure_coeffs": X_sc,
        "Y_structure_coeffs": Y_sc,
    }


def stewart_love_redundancy(
    X: pd.DataFrame,
    Y: pd.DataFrame,
) -> Dict:
    """Bidirectional Stewart-Love redundancy index.

    For each canonical dimension i:
        L̄²_{Y,i} = (1/q) Σ_k  r(Y_k, V_i)²   – average squared structure coeff
        RI_{Y|X}  = Σ_i  L̄²_{Y,i} · ρ_i²

    Symmetrically for RI_{X|Y}.

    Parameters
    ----------
    X : DataFrame, shape (T, p)  – option factors
    Y : DataFrame, shape (T, q)  – equity factors

    Returns
    -------
    dict with keys:
        RI_options_given_equity  – variance in X explained via Y
        RI_equity_given_options  – variance in Y explained via X
        unique_option_share      – 1 − RI_options_given_equity
        unique_equity_share      – 1 − RI_equity_given_options
    """
    cca = canonical_correlation_analysis(X, Y)
    rho = cca["canonical_correlations"]
    X_sc = cca["X_structure_coeffs"]   # (p, s)
    Y_sc = cca["Y_structure_coeffs"]   # (q, s)

    rho2 = rho ** 2

    # L̄²_{X,i}: mean squared structure coeff across p option variables
    Lbar2_X = (X_sc ** 2).mean(axis=0)   # (s,)
    # L̄²_{Y,i}: mean squared structure coeff across q equity variables
    Lbar2_Y = (Y_sc ** 2).mean(axis=0)   # (s,)

    ri_X_given_Y = float(np.dot(Lbar2_X, rho2))   # RI_{option | equity}
    ri_Y_given_X = float(np.dot(Lbar2_Y, rho2))   # RI_{equity | option}

    # Clip to [0, 1] for numerical safety
    ri_X_given_Y = np.clip(ri_X_given_Y, 0.0, 1.0)
    ri_Y_given_X = np.clip(ri_Y_given_X, 0.0, 1.0)

    return {
        "RI_options_given_equity": ri_X_given_Y,
        "RI_equity_given_options": ri_Y_given_X,
        "unique_option_share": 1.0 - ri_X_given_Y,
        "unique_equity_share": 1.0 - ri_Y_given_X,
    }


def run_joint_tests(
    screened_option_returns: pd.DataFrame,
    benchmark_equity_returns: pd.DataFrame,
    rf_series: pd.Series = None,
) -> Dict:
    """Run all joint tests and return a comprehensive results dict.

    Executes in sequence:
    1. GRS test  (option factors as test assets, equity factors as benchmarks)
    2. Maximum Sharpe ratio comparison across three universes
    3. Stewart-Love CCA redundancy analysis

    Parameters
    ----------
    screened_option_returns   : DataFrame (T, p_o)
    benchmark_equity_returns  : DataFrame (T, p_e)
    rf_series                 : optional risk-free rate Series

    Returns
    -------
    dict with sub-dicts: grs, sharpe, redundancy
    """
    # Align on common index
    idx = screened_option_returns.index.intersection(benchmark_equity_returns.index)
    op = screened_option_returns.loc[idx]
    eq = benchmark_equity_returns.loc[idx]

    # 1. GRS – option factors as test assets, equity factors as benchmark
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grs_results = grs_test(
            test_asset_returns=op,
            benchmark_factor_returns=eq,
        )

    # 2. Sharpe comparisons
    sharpe_results = compare_sharpe_ratios(
        equity_factors=eq,
        option_factors=op,
        rf_series=rf_series,
    )

    # 3. CCA redundancy
    redundancy_results = stewart_love_redundancy(X=op, Y=eq)

    return {
        "grs": grs_results,
        "sharpe": sharpe_results,
        "redundancy": redundancy_results,
    }
