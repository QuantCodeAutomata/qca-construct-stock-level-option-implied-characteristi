"""
Canonical Correlation Analysis and Stewart-Love redundancy index.

Separate module for CCA-based redundancy analysis following the paper's
Methodology section on shared variation between option and equity factors.

Custom -- Context7 confirmed no library computes Stewart-Love index directly.
"""

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA
from scipy import linalg
from typing import Dict, Tuple
import warnings


def standardize_returns(returns_df: pd.DataFrame) -> pd.DataFrame:
    """Standardize each column to zero mean and unit variance.

    Constant columns (zero std) are left as zeros rather than producing NaN.

    Parameters
    ----------
    returns_df : DataFrame, shape (T, p)

    Returns
    -------
    DataFrame of the same shape with mean=0, std=1 per column.
    """
    mu = returns_df.mean()
    sigma = returns_df.std(ddof=1)
    sigma[sigma == 0.0] = 1.0
    return (returns_df - mu) / sigma


def compute_canonical_correlations(
    X: np.ndarray,
    Y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Numerically stable canonical correlation analysis via QR + SVD.

    Centres and scales X and Y, then uses QR decomposition to obtain
    orthonormal bases followed by SVD of Qx′Qy to extract canonical
    correlations (singular values) and transformation matrices.

    Parameters
    ----------
    X : array, shape (T, p)  – left variable block (e.g. option factors)
    Y : array, shape (T, q)  – right variable block (e.g. equity factors)

    Returns
    -------
    rho : ndarray, shape (s,)  – canonical correlations in [0, 1],
          s = min(p, q)
    A   : ndarray, shape (p, s) – left loading matrix (X_std @ A = U)
    B   : ndarray, shape (q, s) – right loading matrix (Y_std @ B = V)
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    # Centre
    Xc = X - X.mean(axis=0)
    Yc = Y - Y.mean(axis=0)

    # Scale to unit variance
    sx = Xc.std(axis=0, ddof=1)
    sy = Yc.std(axis=0, ddof=1)
    sx[sx == 0.0] = 1.0
    sy[sy == 0.0] = 1.0
    Xs = Xc / sx
    Ys = Yc / sy

    # Economy QR for numerical stability
    Qx, Rx = linalg.qr(Xs, mode="economic")
    Qy, Ry = linalg.qr(Ys, mode="economic")

    # SVD of cross-product of orthonormal factors
    Ux, rho_raw, VyT = linalg.svd(Qx.T @ Qy, full_matrices=False)
    Vy = VyT.T

    s = min(X.shape[1], Y.shape[1])
    rho = np.clip(rho_raw[:s], 0.0, 1.0)

    # Map back to original standardised variable space
    A = linalg.solve(Rx, Ux[:, :s])   # (p, s)
    B = linalg.solve(Ry, Vy[:, :s])   # (q, s)

    return rho, A, B


def compute_structure_coefficients(
    X: np.ndarray,
    A: np.ndarray,
) -> np.ndarray:
    """Pearson correlations of each original variable with each canonical variate.

    The canonical variates are U = X_standardised @ A, where A comes from
    `compute_canonical_correlations`.  Structure coefficients measure how
    well each original variable loads onto each variate.

    Parameters
    ----------
    X : array, shape (T, p) – original (standardised) variable block
    A : array, shape (p, s) – loading matrix from CCA

    Returns
    -------
    S : ndarray, shape (p, s) – S[j, i] = corr(X_j, U_i)
    """
    X = np.asarray(X, dtype=float)
    A = np.asarray(A, dtype=float)
    T, p = X.shape
    s = A.shape[1]

    # Standardise X for consistent correlation computation
    Xc = X - X.mean(axis=0)
    sx = Xc.std(axis=0, ddof=1)
    sx[sx == 0.0] = 1.0
    Xs = Xc / sx

    U = Xs @ A                          # (T, s) canonical variates

    # Column-wise Pearson r
    Uc = U - U.mean(axis=0)
    su = Uc.std(axis=0, ddof=1)
    su[su == 0.0] = 1.0

    cov_XU = Xs.T @ Uc / (T - 1)       # (p, s)
    S = cov_XU / np.outer(sx / sx, su)  # outer cancels xs std (already 1)
    # simplify: Xs already has unit std, so sx/sx = 1
    S = Xs.T @ Uc / ((T - 1) * su)     # (p, s)

    return S


def compute_redundancy_index(
    structure_coeffs: np.ndarray,
    canonical_correlations: np.ndarray,
) -> float:
    """Stewart-Love redundancy index for one direction.

    RI = Σ_i  L̄²_i · ρ_i²

    where L̄²_i = (1/p) Σ_j  r(X_j, U_i)²  is the average squared
    structure coefficient for dimension i and ρ_i is the i-th canonical
    correlation.

    Parameters
    ----------
    structure_coeffs      : ndarray, shape (p, s) – from compute_structure_coefficients
    canonical_correlations: ndarray, shape (s,)   – from compute_canonical_correlations

    Returns
    -------
    float in [0, 1]
    """
    S = np.asarray(structure_coeffs, dtype=float)
    rho = np.asarray(canonical_correlations, dtype=float)

    Lbar2 = (S ** 2).mean(axis=0)   # (s,) average squared structure coeff per dim
    ri = float(np.dot(Lbar2, rho ** 2))
    return float(np.clip(ri, 0.0, 1.0))


def full_redundancy_analysis(
    option_returns: pd.DataFrame,
    equity_returns: pd.DataFrame,
) -> Dict:
    """Complete bidirectional CCA redundancy analysis.

    Standardises both factor blocks, computes canonical correlations and
    structure coefficients, then derives the Stewart-Love redundancy index
    in both directions.

    Parameters
    ----------
    option_returns : DataFrame, shape (T, p_o) – option factor returns
    equity_returns : DataFrame, shape (T, p_e) – equity factor returns

    Returns
    -------
    dict with keys:
        canonical_correlations    – array of ρ_i values in [0, 1]
        n_canonical_dimensions    – s = min(p_o, p_e)
        X_structure_coefficients  – (p_o, s) structure coefficient matrix
        Y_structure_coefficients  – (p_e, s) structure coefficient matrix
        RI_option_given_equity    – RI_{X|Y}: option variance explained via equity
        RI_equity_given_option    – RI_{Y|X}: equity variance explained via option
        unique_option_share       – 1 − RI_option_given_equity
        unique_equity_share       – 1 − RI_equity_given_option
        option_columns            – list of option factor names
        equity_columns            – list of equity factor names
    """
    # Align indices
    idx = option_returns.index.intersection(equity_returns.index)
    op = standardize_returns(option_returns.loc[idx])
    eq = standardize_returns(equity_returns.loc[idx])

    Xm = np.asarray(op, dtype=float)
    Ym = np.asarray(eq, dtype=float)

    rho, A, B = compute_canonical_correlations(Xm, Ym)
    s = len(rho)

    X_sc = compute_structure_coefficients(Xm, A)   # (p_o, s)
    Y_sc = compute_structure_coefficients(Ym, B)   # (p_e, s)

    ri_op_given_eq = compute_redundancy_index(X_sc, rho)
    ri_eq_given_op = compute_redundancy_index(Y_sc, rho)

    return {
        "canonical_correlations": rho,
        "n_canonical_dimensions": s,
        "X_structure_coefficients": X_sc,
        "Y_structure_coefficients": Y_sc,
        "RI_option_given_equity": ri_op_given_eq,
        "RI_equity_given_option": ri_eq_given_op,
        "unique_option_share": 1.0 - ri_op_given_eq,
        "unique_equity_share": 1.0 - ri_eq_given_op,
        "option_columns": list(option_returns.columns),
        "equity_columns": list(equity_returns.columns),
    }
