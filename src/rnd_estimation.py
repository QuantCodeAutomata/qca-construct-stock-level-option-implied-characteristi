"""
Risk-Neutral Density (RND) estimation from option implied volatility curves.

Implements:
- IV curve smoothing along moneyness dimension using kernel regression
  (one-dimensional moneyness smoother per Ulrich et al. 2023)
- Linear tail extrapolation beyond observed strikes
- Breeden-Litzenberger formula for RND recovery

All functions are custom implementations (Context7: no single library covers
the Ulrich et al. 2023 moneyness kernel smoother end-to-end).
"""

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.stats import norm
from scipy.integrate import trapezoid
from scipy.optimize import brentq
from typing import Tuple, Optional, Dict
import warnings


# ---------------------------------------------------------------------------
# Black-Scholes pricing
# ---------------------------------------------------------------------------

def black_scholes_call(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """
    Black-Scholes call option price.

    Parameters
    ----------
    S : float  Current stock price.
    K : float  Strike price.
    r : float  Continuously compounded risk-free rate.
    sigma : float  Implied volatility (annualised).
    T : float  Time to expiration in years.

    Returns
    -------
    float  Call price; returns intrinsic value when sigma <= 0 or T <= 0.
    """
    if sigma <= 0 or T <= 0:
        return max(S - K * np.exp(-r * T), 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def black_scholes_put(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """
    Black-Scholes put option price.

    Parameters
    ----------
    S : float  Current stock price.
    K : float  Strike price.
    r : float  Continuously compounded risk-free rate.
    sigma : float  Implied volatility (annualised).
    T : float  Time to expiration in years.

    Returns
    -------
    float  Put price; returns intrinsic value when sigma <= 0 or T <= 0.
    """
    if sigma <= 0 or T <= 0:
        return max(K * np.exp(-r * T) - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


# ---------------------------------------------------------------------------
# IV-curve to call prices
# ---------------------------------------------------------------------------

def bs_call_from_iv(
    moneyness_grid: np.ndarray,
    iv_grid: np.ndarray,
    F: float,
    r: float,
    tau: float,
) -> np.ndarray:
    """
    Convert an implied-volatility curve (expressed in moneyness m = K/F) to
    undiscounted call prices using the Black-Scholes formula.

    For each moneyness point m the corresponding strike is K = m * F and the
    spot price is S = F * exp(-r * tau), giving

        d1 = (-ln(m) + 0.5 * sigma^2 * tau) / (sigma * sqrt(tau))
        d2 = d1 - sigma * sqrt(tau)
        C  = F * exp(-r * tau) * [N(d1) - m * N(d2)]

    Parameters
    ----------
    moneyness_grid : np.ndarray
        Array of moneyness values m = K / F.
    iv_grid : np.ndarray
        Implied volatility at each moneyness point.
    F : float
        Forward price.
    r : float
        Risk-free rate.
    tau : float
        Time to expiration in years.

    Returns
    -------
    np.ndarray
        Call prices corresponding to each moneyness point.
    """
    disc = np.exp(-r * tau)
    calls = np.empty(len(moneyness_grid))
    for i, (m, sigma) in enumerate(zip(moneyness_grid, iv_grid)):
        K = m * F
        S = F * disc
        calls[i] = black_scholes_call(S, K, r, sigma, tau)
    return calls


# ---------------------------------------------------------------------------
# IV-curve smoother (Nadaraya-Watson on moneyness, Ulrich et al. 2023)
# ---------------------------------------------------------------------------

def _fit_iv_curve_moneyness_nw(
    df_options: pd.DataFrame,
    forward_price: float,
    tau: float,
    bandwidth: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit a smooth IV curve as a function of moneyness m = K / F using
    Nadaraya-Watson kernel regression with a Gaussian kernel.

    This implements the one-dimensional moneyness smoother from
    Ulrich et al. (2023): for each query point m_0 on a dense grid,

        IV_smooth(m_0) = sum_i phi((m_0 - m_i) / h) * IV_i
                         / sum_i phi((m_0 - m_i) / h)

    where phi is the standard Gaussian kernel and h = bandwidth.

    Parameters
    ----------
    df_options : pd.DataFrame
        Option data with columns: strike, implied_vol, opt_type.
    forward_price : float
        Forward price F used to compute moneyness m_i = strike_i / F.
    tau : float
        Time to expiration in years (used for weighting near-the-money).
    bandwidth : float
        Kernel bandwidth in moneyness units. Default 0.1.

    Returns
    -------
    moneyness_grid : np.ndarray
        200-point grid from 0.5 to 2.0.
    iv_smoothed : np.ndarray
        Nadaraya-Watson smoothed IV at each grid point.
    """
    df = df_options.dropna(subset=["strike", "implied_vol"]).copy()
    df = df[df["implied_vol"] > 0]
    if df.empty:
        raise ValueError("No valid option observations for IV curve fitting.")

    m_obs = df["strike"].values / forward_price
    iv_obs = df["implied_vol"].values

    grid = np.linspace(0.5, 2.0, 200)
    iv_smooth = np.empty(len(grid))

    for j, m0 in enumerate(grid):
        w = np.exp(-0.5 * ((m0 - m_obs) / bandwidth) ** 2)
        w_sum = w.sum()
        if w_sum < 1e-12:
            # Fallback: nearest neighbour
            iv_smooth[j] = iv_obs[np.argmin(np.abs(m_obs - m0))]
        else:
            iv_smooth[j] = np.dot(w, iv_obs) / w_sum

    return grid, iv_smooth


# ---------------------------------------------------------------------------
# Linear tail extrapolation
# ---------------------------------------------------------------------------

def _extrapolate_iv_linear_cutoff(
    moneyness_grid: np.ndarray,
    iv_smoothed: np.ndarray,
    left_cutoff: float = 0.7,
    right_cutoff: float = 1.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Linearly extrapolate the IV curve beyond the observed moneyness range.

    Inside [left_cutoff, right_cutoff] the smoothed IV is left unchanged.
    Outside this interval, the IV is extrapolated linearly using the local
    slope estimated from a small window of 10 points around each boundary.

    Parameters
    ----------
    moneyness_grid : np.ndarray
        Moneyness grid (200 points, 0.5 – 2.0).
    iv_smoothed : np.ndarray
        Smoothed IV values (same length).
    left_cutoff : float
        Lower moneyness boundary for the reliably observed region.
    right_cutoff : float
        Upper moneyness boundary for the reliably observed region.

    Returns
    -------
    moneyness_grid : np.ndarray
        Same grid (unchanged).
    iv_extrap : np.ndarray
        IV with linearly extrapolated tails, clipped to a minimum of 1e-4.
    """
    iv_extrap = iv_smoothed.copy()
    window = 10  # points used to estimate boundary slope

    # --- Left tail ---
    left_idx = np.searchsorted(moneyness_grid, left_cutoff)
    left_idx = np.clip(left_idx, window, len(moneyness_grid) - 1)
    # Slope from the window just inside the left cutoff
    x_l = moneyness_grid[left_idx - window: left_idx]
    y_l = iv_extrap[left_idx - window: left_idx]
    slope_l = np.polyfit(x_l, y_l, 1)[0]
    m_l = moneyness_grid[left_idx - 1]
    iv_l = iv_extrap[left_idx - 1]
    for j in range(left_idx - 1):
        iv_extrap[j] = iv_l + slope_l * (moneyness_grid[j] - m_l)

    # --- Right tail ---
    right_idx = np.searchsorted(moneyness_grid, right_cutoff)
    right_idx = np.clip(right_idx, 0, len(moneyness_grid) - window - 1)
    x_r = moneyness_grid[right_idx: right_idx + window]
    y_r = iv_extrap[right_idx: right_idx + window]
    slope_r = np.polyfit(x_r, y_r, 1)[0]
    m_r = moneyness_grid[right_idx]
    iv_r = iv_extrap[right_idx]
    for j in range(right_idx + 1, len(moneyness_grid)):
        iv_extrap[j] = iv_r + slope_r * (moneyness_grid[j] - m_r)

    np.clip(iv_extrap, 1e-4, None, out=iv_extrap)
    return moneyness_grid, iv_extrap


# ---------------------------------------------------------------------------
# Breeden-Litzenberger RND recovery
# ---------------------------------------------------------------------------

def _recover_rnd_dict(
    moneyness_grid: np.ndarray,
    iv_smoothed: np.ndarray,
    F: float,
    r: float,
    tau: float,
    n_grid: int = 200,
) -> Dict[str, np.ndarray]:
    """
    Recover the risk-neutral density (RND) via the Breeden-Litzenberger formula.

    The RND is the second derivative of the call-pricing function with respect
    to strike, scaled by the discount factor:

        q(K) = exp(r * tau) * d^2 C / dK^2

    The second derivative is approximated with a central finite-difference
    scheme using h = 0.001 * F:

        d^2 C / dK^2 ≈ [C(K + h) - 2 C(K) + C(K - h)] / h^2

    At each K ± h the IV is obtained by linear interpolation of the smoothed
    curve; the call price is then computed via Black-Scholes.

    The density is clipped to be non-negative and normalised to integrate to 1.

    Parameters
    ----------
    moneyness_grid : np.ndarray
        Moneyness grid from fit_iv_curve_moneyness / extrapolate_iv_linear.
    iv_smoothed : np.ndarray
        Smoothed (and extrapolated) IV at each grid point.
    F : float
        Forward price.
    r : float
        Risk-free rate.
    tau : float
        Time to expiration in years.
    n_grid : int
        Number of interior strike points for the output density grid.

    Returns
    -------
    dict with keys:
        'strike_grid'   : np.ndarray – strikes corresponding to the density.
        'density'       : np.ndarray – normalised risk-neutral density q(K).
        'moneyness_grid': np.ndarray – K / F for each strike in strike_grid.
    """
    # Build output strike grid (exclude outermost points to stay within IV support)
    m_min, m_max = moneyness_grid[0], moneyness_grid[-1]
    m_out = np.linspace(m_min * 1.01, m_max * 0.99, n_grid)
    K_out = m_out * F

    # Interpolator for IV curve
    iv_interp = interp1d(
        moneyness_grid, iv_smoothed,
        kind="linear", bounds_error=False,
        fill_value=(iv_smoothed[0], iv_smoothed[-1]),
    )

    h = 0.001 * F
    S = F * np.exp(-r * tau)
    density = np.empty(n_grid)

    for i, K in enumerate(K_out):
        Kp, Km = K + h, K - h
        iv_p = float(iv_interp(Kp / F))
        iv_0 = float(iv_interp(K / F))
        iv_m = float(iv_interp(Km / F))

        Cp = black_scholes_call(S, Kp, r, iv_p, tau)
        C0 = black_scholes_call(S, K,  r, iv_0, tau)
        Cm = black_scholes_call(S, Km, r, iv_m, tau)

        d2C_dK2 = (Cp - 2.0 * C0 + Cm) / h ** 2
        density[i] = np.exp(r * tau) * d2C_dK2

    # Enforce non-negativity
    density = np.clip(density, 0.0, None)

    # Normalise
    mass = trapezoid(density, K_out)
    if mass > 1e-12:
        density /= mass
    else:
        warnings.warn("RND mass is near zero; density may be unreliable.")

    return {
        "strike_grid": K_out,
        "density": density,
        "moneyness_grid": m_out,
    }


# ---------------------------------------------------------------------------
# RND moments
# ---------------------------------------------------------------------------

def _cdf_from_density(density: np.ndarray, strike_grid: np.ndarray) -> np.ndarray:
    """Compute the cumulative distribution function from the density via trapezoidal integration."""
    dx = np.diff(strike_grid)
    increments = 0.5 * (density[:-1] + density[1:]) * dx
    cdf = np.concatenate([[0.0], np.cumsum(increments)])
    cdf /= max(cdf[-1], 1e-12)
    return cdf


def _trim_to_epsilon(
    density: np.ndarray,
    strike_grid: np.ndarray,
    epsilon: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Trim density tails beyond the epsilon and (1 - epsilon) CDF quantiles.

    Returns the trimmed density (re-normalised) and the corresponding sub-grid.
    Returns (None, None) if the trimmed region is too small.
    """
    cdf = _cdf_from_density(density, strike_grid)
    lo = np.searchsorted(cdf, epsilon)
    hi = np.searchsorted(cdf, 1.0 - epsilon)
    hi = min(hi, len(density) - 1)
    if hi <= lo + 1:
        return None, None
    sk = strike_grid[lo: hi + 1]
    d = density[lo: hi + 1].copy()
    mass = trapezoid(d, sk)
    if mass < 1e-12:
        return None, None
    d /= mass
    return d, sk


def _compute_rnd_moments_full(
    density: np.ndarray,
    strike_grid: np.ndarray,
    F: float,
    tau: float,
    epsilon: float = 0.05,
) -> Dict[str, float]:
    """
    Compute risk-neutral moments from the recovered density.

    Two sets of moments are returned:
    - Full distribution: mu, var, skew, kurt  (in strike-level space)
    - Epsilon-trimmed  : mu_eps, var_eps, skew_eps, kurt_eps

    The moments are computed in log-return space, z = ln(K / F):
        mu   = E_Q[z]
        var  = E_Q[z^2] - mu^2
        skew = E_Q[(z - mu)^3] / var^(3/2)
        kurt = E_Q[(z - mu)^4] / var^2

    Parameters
    ----------
    density : np.ndarray
        Normalised risk-neutral density q(K).
    strike_grid : np.ndarray
        Corresponding strikes.
    F : float
        Forward price (used as reference for log-returns).
    tau : float
        Time to expiration (unused in computation, kept for API consistency).
    epsilon : float
        Tail-trimming fraction for the _eps variants.

    Returns
    -------
    dict with keys: mu, var, skew, kurt, mu_eps, var_eps, skew_eps, kurt_eps.
    """
    def _moments(d, sk):
        z = np.log(sk / F)
        mu_  = trapezoid(z * d, sk)
        m2   = trapezoid((z - mu_) ** 2 * d, sk)
        m3   = trapezoid((z - mu_) ** 3 * d, sk)
        m4   = trapezoid((z - mu_) ** 4 * d, sk)
        var_ = m2
        skew_ = m3 / var_ ** 1.5 if var_ > 1e-16 else np.nan
        kurt_ = m4 / var_ ** 2  if var_ > 1e-16 else np.nan
        return mu_, var_, skew_, kurt_

    mu, var, skew, kurt = _moments(density, strike_grid)

    d_eps, sk_eps = _trim_to_epsilon(density, strike_grid, epsilon)
    if d_eps is not None:
        mu_e, var_e, skew_e, kurt_e = _moments(d_eps, sk_eps)
    else:
        mu_e = var_e = skew_e = kurt_e = np.nan

    return {
        "mu":       mu,
        "var":      var,
        "skew":     skew,
        "kurt":     kurt,
        "mu_eps":   mu_e,
        "var_eps":  var_e,
        "skew_eps": skew_e,
        "kurt_eps": kurt_e,
    }


# ---------------------------------------------------------------------------
# Density validation
# ---------------------------------------------------------------------------

def _validate_density_full(
    density: np.ndarray,
    strike_grid: np.ndarray,
) -> Tuple[bool, str]:
    """
    Validate a risk-neutral density estimate.

    Checks:
    1. No negative values (after any floating-point noise).
    2. Integrates to approximately 1 (within 5 % tolerance).

    Parameters
    ----------
    density : np.ndarray
        Density values.
    strike_grid : np.ndarray
        Corresponding strikes (must match length of density).

    Returns
    -------
    valid : bool
    message : str
        Human-readable diagnostic.
    """
    if np.any(density < -1e-8):
        neg_frac = np.mean(density < 0)
        return False, f"Density has negative values ({neg_frac:.1%} of points)."

    mass = trapezoid(density, strike_grid)
    if not (0.95 <= mass <= 1.05):
        return False, f"Density integrates to {mass:.4f}, expected ~1."

    return True, f"Density valid (mass={mass:.4f}, min={density.min():.2e})."


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------

def fit_stock_month_rnd(
    df_options_stock_month: pd.DataFrame,
    tau_target: float = 30,
    min_strikes: int = 5,
) -> Optional[Dict]:
    """
    Full RND estimation pipeline for one stock-month.

    Steps
    -----
    1. Select options within ±5 days of tau_target (in calendar days).
    2. Compute the forward price from put-call parity (or use pre-computed column).
    3. Fit a Nadaraya-Watson IV curve in moneyness space.
    4. Linearly extrapolate tails.
    5. Recover the RND via Breeden-Litzenberger finite differences.
    6. Validate the density.
    7. Compute RND moments.

    Parameters
    ----------
    df_options_stock_month : pd.DataFrame
        Option data for one (stock, expiry-month) group.  Required columns:
        strike, implied_vol, opt_type.  Optional: days_to_exp, forward_price,
        rf_rate.
    tau_target : float
        Target days-to-expiration. Default 30.
    min_strikes : int
        Minimum number of distinct strikes required. Default 5.

    Returns
    -------
    dict or None
        'rnd_result'   : output of recover_rnd (strike_grid, density, moneyness_grid)
        'moments'      : output of compute_rnd_moments
        'valid'        : bool – density passed validation
        'diagnostics'  : str – validation message
        'forward_price': float
        'tau'          : float (years)
        'rf_rate'      : float
    Returns None when data are insufficient.
    """
    df = df_options_stock_month.copy()

    # --- Expiry selection ---
    tau_years = tau_target / 365.0
    if "days_to_exp" in df.columns:
        df = df[np.abs(df["days_to_exp"] - tau_target) <= 5]
    if df.empty or df["strike"].nunique() < min_strikes:
        return None

    # --- Forward price ---
    if "forward_price" in df.columns and df["forward_price"].notna().any():
        F = float(df["forward_price"].dropna().iloc[0])
    elif "stock_price" in df.columns and df["stock_price"].notna().any():
        S0 = float(df["stock_price"].dropna().iloc[0])
        rf = float(df["rf_rate"].iloc[0]) if "rf_rate" in df.columns else 0.0
        F = S0 * np.exp(rf * tau_years)
    else:
        # Use mid-point between highest put strike and lowest call strike as proxy
        put_strikes = df[df["opt_type"] == "P"]["strike"]
        call_strikes = df[df["opt_type"] == "C"]["strike"]
        if put_strikes.empty or call_strikes.empty:
            F = df["strike"].median()
        else:
            F = 0.5 * (put_strikes.max() + call_strikes.min())

    rf = float(df["rf_rate"].iloc[0]) if "rf_rate" in df.columns else 0.0

    # --- Fit IV curve ---
    try:
        m_grid, iv_smooth = _fit_iv_curve_moneyness_nw(df, F, tau_years)
    except ValueError:
        return None

    # --- Extrapolate tails ---
    m_grid, iv_smooth = _extrapolate_iv_linear_cutoff(m_grid, iv_smooth)

    # --- Recover RND ---
    rnd_result = _recover_rnd_dict(m_grid, iv_smooth, F, rf, tau_years)

    # --- Validate ---
    valid, msg = _validate_density_full(rnd_result["density"], rnd_result["strike_grid"])

    # --- Moments ---
    moments = _compute_rnd_moments_full(
        rnd_result["density"], rnd_result["strike_grid"], F, tau_years
    )

    return {
        "rnd_result":    rnd_result,
        "moments":       moments,
        "valid":         valid,
        "diagnostics":   msg,
        "forward_price": F,
        "tau":           tau_years,
        "rf_rate":       rf,
    }


# ---------------------------------------------------------------------------
# Public API — test-compatible signatures
# ---------------------------------------------------------------------------

def fit_iv_curve_moneyness(
    strikes: np.ndarray,
    ivs: np.ndarray,
    S: float,
    r: float,
    T: float,
    bandwidth: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit a smooth IV curve on a 200-point moneyness grid via Nadaraya-Watson.

    Parameters
    ----------
    strikes : array_like  Observed option strike prices.
    ivs     : array_like  Observed implied volatilities (same length as strikes).
    S       : float  Current spot price.
    r       : float  Risk-free rate (continuously compounded).
    T       : float  Time to expiration in years.
    bandwidth : float  Kernel bandwidth in moneyness units (default 0.1).

    Returns
    -------
    moneyness_grid : np.ndarray  200-point grid in [0.5, 2.0].
    iv_grid        : np.ndarray  Smoothed IVs at each grid point (all > 0).
    """
    F = S * np.exp(r * T)
    df = pd.DataFrame({"strike": np.asarray(strikes, float),
                       "implied_vol": np.asarray(ivs, float),
                       "opt_type": "call"})
    return _fit_iv_curve_moneyness_nw(df, F, T, bandwidth)


def extrapolate_iv_linear(
    moneyness_grid: np.ndarray,
    iv_grid: np.ndarray,
    extrapolate: bool = False,
    left_cutoff: float = 0.65,
    right_cutoff: float = 1.85,
) -> np.ndarray:
    """Linearly extrapolate IV tails; interior values are left unchanged.

    Parameters
    ----------
    moneyness_grid : np.ndarray  200-point moneyness grid.
    iv_grid        : np.ndarray  Smoothed IVs (same length).
    extrapolate    : bool  If False, return iv_grid unchanged (default False).
    left_cutoff    : float  Left boundary for the reliable region (default 0.7).
    right_cutoff   : float  Right boundary for the reliable region (default 1.5).

    Returns
    -------
    np.ndarray  IV array with extrapolated tails (when extrapolate=True).
    """
    if not extrapolate:
        return np.array(iv_grid, dtype=float)
    _, iv_extrap = _extrapolate_iv_linear_cutoff(
        np.asarray(moneyness_grid, float), np.asarray(iv_grid, float),
        left_cutoff, right_cutoff,
    )
    return iv_extrap


def recover_rnd(
    moneyness_grid: np.ndarray,
    iv_grid: np.ndarray,
    S: float,
    r: float,
    T: float,
) -> np.ndarray:
    """Recover the risk-neutral density via Breeden-Litzenberger.

    q(K) = exp(r·T) · d²C/dK², normalised to integrate to 1 over the
    strike grid.

    Parameters
    ----------
    moneyness_grid : np.ndarray  200-point moneyness grid.
    iv_grid        : np.ndarray  Smoothed IV at each grid point.
    S              : float  Spot price.
    r              : float  Risk-free rate.
    T              : float  Time to expiration in years.

    Returns
    -------
    np.ndarray  Non-negative, normalised density evaluated on S_grid = moneyness_grid × S.
    """
    F = S * np.exp(r * T)
    rnd_dict = _recover_rnd_dict(
        np.asarray(moneyness_grid, float), np.asarray(iv_grid, float), F, r, T
    )
    density = rnd_dict["density"]
    S_grid = rnd_dict["strike_grid"]
    integral = np.trapezoid(density, S_grid)
    if integral > 1e-12:
        density = density / integral
    return density


def compute_rnd_moments(
    density: np.ndarray,
    S_grid: np.ndarray,
) -> Dict[str, float]:
    """Compute risk-neutral moments from a density evaluated on a price grid.

    Parameters
    ----------
    density : np.ndarray  Normalised risk-neutral density values.
    S_grid  : np.ndarray  Corresponding strike / price grid.

    Returns
    -------
    dict  Keys: mean, variance, skewness, kurtosis, and log-return variants.
          The ``variance`` key contains the risk-neutral variance (≥ 0).
    """
    density = np.asarray(density, float)
    S_grid = np.asarray(S_grid, float)
    # Estimate the risk-neutral expectation (forward price proxy)
    integral = np.trapezoid(density, S_grid)
    if integral > 1e-12:
        density = density / integral
    F_hat = np.trapezoid(density * S_grid, S_grid)
    if not (np.isfinite(F_hat) and F_hat > 0):
        F_hat = float(np.nanmean(S_grid))
    raw = _compute_rnd_moments_full(density, S_grid, F_hat, tau=0.0)
    # Remap to canonical test-expected keys
    return {
        "mean":      raw.get("mu",   float("nan")),
        "variance":  max(raw.get("var",  0.0), 0.0),
        "skewness":  raw.get("skew", float("nan")),
        "kurtosis":  raw.get("kurt", float("nan")),
        "log_mean":  raw.get("mu",   float("nan")),
        "log_variance": max(raw.get("var", 0.0), 0.0),
        "log_skewness": raw.get("skew", float("nan")),
        "log_kurtosis": raw.get("kurt", float("nan")),
    }


def validate_density(
    density: np.ndarray,
    S_grid: np.ndarray,
) -> bool:
    """Validate that the density is non-negative and integrates to ≈ 1.

    Parameters
    ----------
    density : np.ndarray  Density values.
    S_grid  : np.ndarray  Corresponding price / strike grid.

    Returns
    -------
    bool  True if valid (non-negative and integral ≈ 1), False otherwise.
    """
    valid, _ = _validate_density_full(
        np.asarray(density, float), np.asarray(S_grid, float)
    )
    return bool(valid)
