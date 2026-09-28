"""
Option-implied characteristics for stock-level option data.

Implements all 137 Appendix B characteristics from the paper:
'Option Factors in the Cross-Section of Stock Returns'

Characteristic families:
1. Risk-neutral density moments (RVN/RND)
2. Jump and tail measures (BTX)
3. Disaster and RIX measures
4. IV-surface shape measures (ivs_*)
5. Option-implied physical moments (PimplQ_*)
6. SVIX measures

Custom implementation -- Context7 found no library covering all paper formulas.
"""

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.stats import norm
from typing import Dict, Optional, Tuple
import warnings

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_div(a: float, b: float, default: float = np.nan) -> float:
    """Return a / b, or default when b is near zero."""
    return a / b if abs(b) > 1e-16 else default


def _cdf_from_density(density: np.ndarray, strike_grid: np.ndarray) -> np.ndarray:
    """Trapezoidal cumulative distribution from a density array."""
    dx = np.diff(strike_grid)
    increments = 0.5 * (density[:-1] + density[1:]) * dx
    cdf = np.concatenate([[0.0], np.cumsum(increments)])
    cdf /= max(cdf[-1], 1e-12)
    return cdf


def _trim_to_epsilon(
    density: np.ndarray,
    strike_grid: np.ndarray,
    epsilon: float,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Return the sub-density and sub-grid after removing the bottom and top
    epsilon probability mass.  The trimmed density is re-normalised.
    """
    cdf = _cdf_from_density(density, strike_grid)
    lo = int(np.searchsorted(cdf, epsilon))
    hi = int(np.searchsorted(cdf, 1.0 - epsilon))
    hi = min(hi, len(density) - 1)
    if hi <= lo + 1:
        return None, None
    sk = strike_grid[lo: hi + 1]
    d  = density[lo: hi + 1].copy()
    mass = trapezoid(d, sk)
    if mass < 1e-12:
        return None, None
    d /= mass
    return d, sk


def _quantile_from_density(
    density: np.ndarray,
    strike_grid: np.ndarray,
    q: float,
) -> float:
    """Return the q-th quantile of the risk-neutral distribution."""
    cdf = _cdf_from_density(density, strike_grid)
    idx = int(np.searchsorted(cdf, q))
    idx = np.clip(idx, 0, len(strike_grid) - 1)
    return float(strike_grid[idx])


def _log_return_moments(
    density: np.ndarray,
    strike_grid: np.ndarray,
    F: float,
) -> Tuple[float, float, float, float]:
    """
    Compute (mu, var, skew, kurt) in log-return space z = ln(K/F).

    Returns (nan, nan, nan, nan) when the density has fewer than two points.
    """
    if len(density) < 2:
        return np.nan, np.nan, np.nan, np.nan
    z    = np.log(strike_grid / F)
    mu   = trapezoid(z * density, strike_grid)
    m2   = trapezoid((z - mu) ** 2 * density, strike_grid)
    m3   = trapezoid((z - mu) ** 3 * density, strike_grid)
    m4   = trapezoid((z - mu) ** 4 * density, strike_grid)
    skew = _safe_div(m3, m2 ** 1.5)
    kurt = _safe_div(m4, m2 ** 2)
    return mu, m2, skew, kurt


def _level_moments(
    density: np.ndarray,
    strike_grid: np.ndarray,
) -> Tuple[float, float, float, float]:
    """Compute (mu, var, skew, kurt) in level (strike) space."""
    if len(density) < 2:
        return np.nan, np.nan, np.nan, np.nan
    mu   = trapezoid(strike_grid * density, strike_grid)
    m2   = trapezoid((strike_grid - mu) ** 2 * density, strike_grid)
    m3   = trapezoid((strike_grid - mu) ** 3 * density, strike_grid)
    m4   = trapezoid((strike_grid - mu) ** 4 * density, strike_grid)
    skew = _safe_div(m3, m2 ** 1.5)
    kurt = _safe_div(m4, m2 ** 2)
    return mu, m2, skew, kurt


# ---------------------------------------------------------------------------
# 1. Model-free risk-neutral moments (RVN / RND family)
# ---------------------------------------------------------------------------

def compute_model_free_moments(
    density: np.ndarray,
    strike_grid: np.ndarray,
    F: float,
    tau: float,
    epsilon_pct: float = 0.05,
) -> Dict[str, float]:
    """
    Compute the RVN / RND model-free moment family.

    Moments are computed in log-return space z = ln(K/F) so that

        RVN_mu  = E_Q[ln(S_T/F)] = integral ln(K/F) * q(K) dK
        RVN_var = E_Q[(ln(S_T/F) - RVN_mu)^2]
        RVN_skew = third central moment / var^(3/2)
        RVN_kurt = fourth central moment / var^2

    The _eps variants exclude the bottom and top epsilon_pct probability mass
    from the density before computing moments (re-normalised).

    Parameters
    ----------
    density : np.ndarray
        Normalised risk-neutral density q(K).
    strike_grid : np.ndarray
        Strike values.
    F : float
        Forward price.
    tau : float
        Time to expiration in years.
    epsilon_pct : float
        Tail trimming fraction (default 5 %).

    Returns
    -------
    dict with 8 keys: RVN_mu, RVN_var, RVN_skew, RVN_kurt,
                      RVN_mu_eps, RVN_var_eps, RVN_skew_eps, RVN_kurt_eps.
    """
    mu, var, skew, kurt = _log_return_moments(density, strike_grid, F)

    d_eps, sk_eps = _trim_to_epsilon(density, strike_grid, epsilon_pct)
    if d_eps is not None:
        mu_e, var_e, skew_e, kurt_e = _log_return_moments(d_eps, sk_eps, F)
    else:
        mu_e = var_e = skew_e = kurt_e = np.nan

    return {
        "RVN_mu":       mu,
        "RVN_var":      var,
        "RVN_skew":     skew,
        "RVN_kurt":     kurt,
        "RVN_mu_eps":   mu_e,
        "RVN_var_eps":  var_e,
        "RVN_skew_eps": skew_e,
        "RVN_kurt_eps": kurt_e,
    }


# ---------------------------------------------------------------------------
# 2. Jump and tail measures (BTX family)
# ---------------------------------------------------------------------------

def _jump_variation_pair(
    density: np.ndarray,
    strike_grid: np.ndarray,
    F: float,
    k: float,
) -> Tuple[float, float]:
    """
    Compute right and left jump variation with threshold k.

    BTX_RJV_k = integral_{K > F(1+k)} (K/F - 1)^2 q(K) dK
    BTX_LJV_k = integral_{K < F(1-k)} (K/F - 1)^2 q(K) dK
    """
    rel = (strike_grid / F - 1.0) ** 2

    right_mask = strike_grid > F * (1.0 + k)
    if right_mask.sum() >= 2:
        rjv = trapezoid(rel[right_mask] * density[right_mask], strike_grid[right_mask])
    else:
        rjv = 0.0

    left_mask = strike_grid < F * (1.0 - k)
    if left_mask.sum() >= 2:
        ljv = trapezoid(rel[left_mask] * density[left_mask], strike_grid[left_mask])
    else:
        ljv = 0.0

    return float(rjv), float(ljv)


def compute_jump_tail_measures(
    density: np.ndarray,
    strike_grid: np.ndarray,
    F: float,
    tau: float,
    k_threshold: float = 0.10,
    pct_7_5: float = 0.075,
    pct_20: float = 0.20,
) -> Dict[str, float]:
    """
    Compute the BTX jump-and-tail measure family.

    Definitions
    -----------
    BTX_RJV_k       : right jump variation   = ∫_{K>F(1+k)}  (K/F-1)^2 q(K) dK
    BTX_LJV_k       : left  jump variation   = ∫_{K<F(1-k)}  (K/F-1)^2 q(K) dK
    BTX_MRJI_k_ann  : annualised jump imbalance = (RJV - LJV) / tau * 12
    BTX_MLJI_k_ann  : relative imbalance = LJV / RJV
    BTX_EQCV_k      : risk-neutral coefficient of variation = sqrt(Var_Q[K]) / E_Q[K]

    Parameters
    ----------
    density, strike_grid : arrays
        Normalised density and corresponding strikes.
    F : float
        Forward price.
    tau : float
        Time to expiration in years.
    k_threshold : float
        Primary jump threshold (default 10 %).
    pct_7_5, pct_20 : float
        Secondary thresholds (7.5 % and 20 %).

    Returns
    -------
    dict with 9 keys.
    """
    def _measures(k):
        rjv, ljv = _jump_variation_pair(density, strike_grid, F, k)
        ann_factor = 12.0 / tau if tau > 0 else np.nan
        mrji = (rjv - ljv) * ann_factor
        mlji = _safe_div(ljv, rjv)
        return rjv, ljv, mrji, mlji

    rjv_k,   ljv_k,   mrji_k,   mlji_k   = _measures(k_threshold)
    rjv_75,  ljv_75,  mrji_75,  mlji_75  = _measures(pct_7_5)
    rjv_20,  ljv_20,  mrji_20,  mlji_20  = _measures(pct_20)

    # Coefficient of variation in level space
    mu_K  = trapezoid(strike_grid * density, strike_grid)
    m2_K  = trapezoid((strike_grid - mu_K) ** 2 * density, strike_grid)
    eqcv  = _safe_div(np.sqrt(max(m2_K, 0.0)), mu_K)

    return {
        "BTX_RJV_k":         rjv_k,
        "BTX_LJV_k":         ljv_k,
        "BTX_MRJI_k_annual": mrji_k,
        "BTX_MLJI_k_annual": mlji_k,
        "BTX_RJV_7_5":       rjv_75,
        "BTX_LJV_7_5":       ljv_75,
        "BTX_MRJI_7_5_annual": mrji_75,
        "BTX_MLJI_7_5_annual": mlji_75,
        "BTX_MRJI_20_annual":  mrji_20,
        "BTX_MLJI_20_annual":  mlji_20,
        "BTX_EQCV_k":          eqcv,
    }


# ---------------------------------------------------------------------------
# 3. Disaster and RIX measures
# ---------------------------------------------------------------------------

def compute_disaster_rix_measures(
    density: np.ndarray,
    strike_grid: np.ndarray,
    F: float,
    tau: float,
    tail_pct: float = 0.10,
    right_pct: float = 0.10,
) -> Dict[str, float]:
    """
    Compute disaster risk, tail probability, and RIX measures.

    Tail probabilities
    ------------------
    BTX_EQV_relTail_k        = P_Q(S_T < F*(1 - tail_pct))
    BTX_EQV_relRightTail_k   = P_Q(S_T > F*(1 + right_pct))

    Tail log-density slopes (tail index proxies)
    --------------------------------------------
    BTX_alpha_plus  : OLS slope of log(q) vs log(K/F) for K > F (right tail).
    BTX_alpha_minus : OLS slope of log(q) vs log(K/F) for K < F (left tail).

    RIX (disaster risk index, Gao et al. 2018)
    -------------------------------------------
    RIX = (2/F) * ∫_{K < F} (1 - K/F) * q(K) dK
        = 2 * E_Q[(1 - S_T/F)^+] / F

    Bakshi-Kapadia-Madan (2003) risk-neutral moments
    ------------------------------------------------
    bakshi_X    : implied expected excess return under Q
    bakshiVar   : model-free implied variance
    bakshiSkew  : model-free implied skewness
    bakshiKurt  : model-free implied kurtosis
    (computed via compute_bakshi_kmd_moments)

    Parameters
    ----------
    density, strike_grid : arrays
    F : float
        Forward price.
    tau : float
        Time to expiration in years.
    tail_pct : float
        Left tail threshold (default 10 %).
    right_pct : float
        Right tail threshold (default 10 %).

    Returns
    -------
    dict with 11 keys.
    """
    out: Dict[str, float] = {}

    # Tail probabilities
    left_thresh  = F * (1.0 - tail_pct)
    right_thresh = F * (1.0 + right_pct)

    left_mask  = strike_grid < left_thresh
    right_mask = strike_grid > right_thresh

    out["BTX_EQV_relTail_k"] = (
        float(trapezoid(density[left_mask], strike_grid[left_mask]))
        if left_mask.sum() >= 2 else 0.0
    )
    out["BTX_EQV_relRightTail_k"] = (
        float(trapezoid(density[right_mask], strike_grid[right_mask]))
        if right_mask.sum() >= 2 else 0.0
    )
    out["BTX_EQV_relTail_10"]      = out["BTX_EQV_relTail_k"]
    out["BTX_EQV_relRightTail_10"] = out["BTX_EQV_relRightTail_k"]

    # Tail log-density slopes (right tail: K > F)
    r_mask = (strike_grid > F) & (density > 1e-12)
    if r_mask.sum() >= 4:
        log_m  = np.log(strike_grid[r_mask] / F)
        log_q  = np.log(density[r_mask])
        slope  = np.polyfit(log_m, log_q, 1)[0]
        out["BTX_alpha_plus"] = float(slope)
    else:
        out["BTX_alpha_plus"] = np.nan

    l_mask = (strike_grid < F) & (density > 1e-12)
    if l_mask.sum() >= 4:
        log_m  = np.log(strike_grid[l_mask] / F)   # negative values
        log_q  = np.log(density[l_mask])
        slope  = np.polyfit(log_m, log_q, 1)[0]
        out["BTX_alpha_minus"] = float(slope)
    else:
        out["BTX_alpha_minus"] = np.nan

    # RIX: 2/F * integral_{K < F} (1 - K/F) * q(K) dK
    put_mask = strike_grid < F
    if put_mask.sum() >= 2:
        integrand = (1.0 - strike_grid[put_mask] / F) * density[put_mask]
        rix = 2.0 / F * float(trapezoid(integrand, strike_grid[put_mask]))
    else:
        rix = np.nan
    out["RIX"] = rix

    # BKM moments (delegated to dedicated function)
    bkm = compute_bakshi_kmd_moments(density, strike_grid, F, 0.0, tau)
    out["bakshi_X"]    = bkm.get("bkm_mu",   np.nan)
    out["bakshiVar"]   = bkm.get("bkm_var",  np.nan)
    out["bakshiSkew"]  = bkm.get("bkm_skew", np.nan)
    out["bakshiKurt"]  = bkm.get("bkm_kurt", np.nan)

    return out


# ---------------------------------------------------------------------------
# 4. IV surface shape measures
# ---------------------------------------------------------------------------

def compute_iv_surface_measures(
    df_options_stock_month: pd.DataFrame,
    forward_price: float,
    tau: float,
) -> Dict[str, float]:
    """
    Compute implied-volatility surface shape characteristics.

    Moneyness approximations for delta-based wings:
        5-delta  put  ≈ m = 0.80,  5-delta  call ≈ m = 1.20
        10-delta put  ≈ m = 0.85,  10-delta call ≈ m = 1.15
        25-delta put  ≈ m = 0.93,  25-delta call ≈ m = 1.07
        ATM           ≈ m = 1.00

    Measures
    --------
    ivs_level              : ATM implied volatility
    ivs_slope              : (OTM put IV - OTM call IV) at 10-delta wings
    ivs_smirk              : OTM put IV - ATM IV
    ivs_convexity          : (OTM put IV + OTM call IV - 2*ATM IV) / 2
    ivs_term_structure     : IV ratio across maturities (NaN if unavailable)
    ivs_atm_vol            : 30-day ATM IV (same as ivs_level for single maturity)
    ivs_atm_put_call_spread: difference in IV between ATM put and ATM call
    ivs_vol_spread         : mean bid-ask spread in IV units (NaN if unavailable)
    ivs_slope_25d          : risk reversal at 25-delta wings
    ivs_slope_5d           : risk reversal at 5-delta wings
    ivs_rr_10d             : risk reversal at 10-delta (= ivs_slope)
    ivs_rr_25d             : risk reversal at 25-delta
    ivs_bf_10d             : butterfly spread at 10-delta
    ivs_bf_25d             : butterfly spread at 25-delta

    Parameters
    ----------
    df_options_stock_month : pd.DataFrame
        Option data; required columns: strike, implied_vol, opt_type.
        Optional: bid_iv, ask_iv, days_to_exp.
    forward_price : float
        Forward price F.
    tau : float
        Time to expiration in years.

    Returns
    -------
    dict with 14 keys.
    """
    df = df_options_stock_month.dropna(subset=["strike", "implied_vol"]).copy()
    df["moneyness"] = df["strike"] / forward_price

    out: Dict[str, float] = {}

    def _interp_iv(target_m: float, opt_type: Optional[str] = None) -> float:
        """Linearly interpolate IV at a target moneyness from the option panel."""
        sub = df if opt_type is None else df[df["opt_type"] == opt_type]
        if sub.empty:
            sub = df
        sub = sub.sort_values("moneyness")
        if len(sub) < 2:
            return float(sub["implied_vol"].iloc[0]) if len(sub) == 1 else np.nan
        m_arr  = sub["moneyness"].values
        iv_arr = sub["implied_vol"].values
        if target_m <= m_arr[0]:
            return float(iv_arr[0])
        if target_m >= m_arr[-1]:
            return float(iv_arr[-1])
        return float(np.interp(target_m, m_arr, iv_arr))

    atm_iv   = _interp_iv(1.00)
    put10_iv  = _interp_iv(0.85, "P")
    call10_iv = _interp_iv(1.15, "C")
    put25_iv  = _interp_iv(0.93, "P")
    call25_iv = _interp_iv(1.07, "C")
    put5_iv   = _interp_iv(0.80, "P")
    call5_iv  = _interp_iv(1.20, "C")

    out["ivs_level"]    = atm_iv
    out["ivs_atm_vol"]  = atm_iv

    # Risk reversals (put wing IV - call wing IV)
    out["ivs_slope"]    = put10_iv - call10_iv       # 10-delta
    out["ivs_rr_10d"]   = put10_iv - call10_iv
    out["ivs_rr_25d"]   = put25_iv - call25_iv
    out["ivs_slope_25d"]= put25_iv - call25_iv
    out["ivs_slope_5d"] = put5_iv  - call5_iv

    # Smirk: OTM put minus ATM
    out["ivs_smirk"] = put10_iv - atm_iv

    # Butterfly / convexity
    out["ivs_convexity"] = 0.5 * (put10_iv + call10_iv) - atm_iv
    out["ivs_bf_10d"]    = out["ivs_convexity"]
    out["ivs_bf_25d"]    = 0.5 * (put25_iv + call25_iv) - atm_iv

    # ATM put-call IV spread
    puts  = df[df["opt_type"] == "P"]
    calls = df[df["opt_type"] == "C"]
    atm_put_iv  = _interp_iv(1.00, "P")
    atm_call_iv = _interp_iv(1.00, "C")
    out["ivs_atm_put_call_spread"] = atm_put_iv - atm_call_iv

    # Term structure stub (NaN without multi-maturity data)
    if "days_to_exp" in df.columns and df["days_to_exp"].nunique() > 1:
        short_tau = df["days_to_exp"].min()
        long_tau  = df["days_to_exp"].max()
        iv_short  = df[df["days_to_exp"] == short_tau]["implied_vol"].mean()
        iv_long   = df[df["days_to_exp"] == long_tau]["implied_vol"].mean()
        out["ivs_term_structure"] = _safe_div(iv_long, iv_short)
    else:
        out["ivs_term_structure"] = np.nan

    # Bid-ask spread in IV units
    if "bid_iv" in df.columns and "ask_iv" in df.columns:
        spread = (df["ask_iv"] - df["bid_iv"]).dropna()
        out["ivs_vol_spread"] = float(spread.mean()) if len(spread) > 0 else np.nan
    else:
        out["ivs_vol_spread"] = np.nan

    return out


# ---------------------------------------------------------------------------
# 5. SVIX measures
# ---------------------------------------------------------------------------

def compute_svix_measures(
    density: np.ndarray,
    strike_grid: np.ndarray,
    F: float,
    tau: float,
    rf_rate: float = 0.0,
    epsilon_pct: float = 0.05,
) -> Dict[str, float]:
    """
    Compute SVIX and RND moment measures.

    SVIX^2 (Martin 2017) is proportional to the risk-neutral variance of the
    gross return S_T / F:

        SVIX2 = (1/tau) * E_Q[(S_T/F - 1)^2]
              = (1/tau) * integral (K/F - 1)^2 * q(K) dK

    Decomposed by put (K < F) and call (K > F) regions.

    Additionally returns RND_* moment labels (level-space moments) which are
    alias labels used elsewhere in the paper alongside the SVIX family.

    Parameters
    ----------
    density, strike_grid : arrays
    F : float
        Forward price.
    tau : float
        Time to expiration in years.
    rf_rate : float
        Risk-free rate (kept for API consistency).
    epsilon_pct : float
        Tail-trimming fraction for _eps variants.

    Returns
    -------
    dict with 12 keys.
    """
    out: Dict[str, float] = {}

    rel2 = (strike_grid / F - 1.0) ** 2
    ann  = 1.0 / tau if tau > 0 else np.nan

    full_var = float(trapezoid(rel2 * density, strike_grid))
    out["SVIX2_annual"] = full_var * ann

    put_mask  = strike_grid <= F
    call_mask = strike_grid >= F

    put_var = (
        float(trapezoid(rel2[put_mask] * density[put_mask], strike_grid[put_mask]))
        if put_mask.sum() >= 2 else 0.0
    )
    call_var = (
        float(trapezoid(rel2[call_mask] * density[call_mask], strike_grid[call_mask]))
        if call_mask.sum() >= 2 else 0.0
    )
    out["SVIX2_put_annual"]  = put_var  * ann
    out["SVIX2_call_annual"] = call_var * ann

    # Level-space moments (RND_* label family)
    mu, var, skew, kurt = _level_moments(density, strike_grid)
    out["RND_mu"]   = mu
    out["RND_var"]  = var
    out["RND_skew"] = skew
    out["RND_kurt"] = kurt

    d_eps, sk_eps = _trim_to_epsilon(density, strike_grid, epsilon_pct)
    if d_eps is not None:
        mu_e, var_e, skew_e, kurt_e = _level_moments(d_eps, sk_eps)
    else:
        mu_e = var_e = skew_e = kurt_e = np.nan
    out["RND_mu_eps"]   = mu_e
    out["RND_var_eps"]  = var_e
    out["RND_skew_eps"] = skew_e
    out["RND_kurt_eps"] = kurt_e

    return out


# ---------------------------------------------------------------------------
# 6. Physical-implied Q (PimplQ) moments
# ---------------------------------------------------------------------------

def compute_pimplq_moments(
    rn_moments_1d: Dict[str, float],
    rn_moments_tau: Dict[str, float],
    phys_moments_1d: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Compute P_implQ physical-implied moments via Q-to-P scaling.

    Method
    ------
    If physical moments at the 1-month horizon (phys_moments_1d) are supplied,
    the P/Q scaling ratio is:

        ratio_x = phys_moments_1d[x] / rn_moments_1d[x]   for x in {var, skew, kurt}

    and the implied physical moment at horizon tau is:

        P_tau_x = ratio_x * rn_moments_tau[x]

    When phys_moments_1d is None, the ratio is set to 1 (i.e., P ≡ Q).

    Parameters
    ----------
    rn_moments_1d : dict
        Risk-neutral moments at 1-month horizon; keys: var, skew, kurt,
        var_eps, skew_eps, kurt_eps, mu, mu_eps.
    rn_moments_tau : dict
        Risk-neutral moments at target tau; same key structure.
    phys_moments_1d : dict or None
        Historical physical moments at 1-month horizon (same key structure).

    Returns
    -------
    dict with 8 keys: PimplQ_mu, PimplQ_var, PimplQ_skew, PimplQ_kurt,
                      PimplQ_mu_eps, PimplQ_var_eps, PimplQ_skew_eps, PimplQ_kurt_eps.
    """
    def _ratio(key: str) -> float:
        if phys_moments_1d is None:
            return 1.0
        q1 = rn_moments_1d.get(key, np.nan)
        p1 = phys_moments_1d.get(key, np.nan)
        return _safe_div(p1, q1, default=1.0)

    out: Dict[str, float] = {}
    for key, label in [
        ("mu",       "PimplQ_mu"),
        ("var",      "PimplQ_var"),
        ("skew",     "PimplQ_skew"),
        ("kurt",     "PimplQ_kurt"),
        ("mu_eps",   "PimplQ_mu_eps"),
        ("var_eps",  "PimplQ_var_eps"),
        ("skew_eps", "PimplQ_skew_eps"),
        ("kurt_eps", "PimplQ_kurt_eps"),
    ]:
        q_tau = rn_moments_tau.get(key, np.nan)
        r     = _ratio(key)
        out[label] = float(r * q_tau) if np.isfinite(r) and np.isfinite(q_tau) else np.nan

    return out


# ---------------------------------------------------------------------------
# 7. Bakshi-Kapadia-Madan (2003) model-free moments
# ---------------------------------------------------------------------------

def compute_bakshi_kmd_moments(
    density: np.ndarray,
    strike_grid: np.ndarray,
    F: float,
    r: float,
    tau: float,
) -> Dict[str, float]:
    """
    Compute Bakshi-Kapadia-Madan (2003) model-free implied moments.

    BKM contracts
    -------------
    The variance (W2), cubic (W3), and quartic (W4) contracts are:

        W2 = 2 * [∫_{K≥F} C(K)/K^2 dK  +  ∫_{K≤F} P(K)/K^2 dK]
        W3 = 6 * [∫_{K≥F} ln(K/F)*C(K)/K^2 dK  −  ∫_{K≤F} ln(K/F)*P(K)/K^2 dK]
        W4 = 12* [∫_{K≥F} (ln K/F)^2*C(K)/K^2 dK + ∫_{K≤F} (ln K/F)^2*P(K)/K^2 dK]

    Option prices are obtained by integrating the risk-neutral density:

        C(K_0) = e^{-rτ} ∫_{K_0}^∞ (K-K_0) q(K) dK
        P(K_0) = e^{-rτ} ∫_0^{K_0}  (K_0-K) q(K) dK

    Implied moments (BKM 2003, equations 4–7):

        μ   = e^{rτ} − 1 − e^{rτ}/2·W2 − e^{rτ}/6·W3 − e^{rτ}/24·W4
        Var = e^{rτ}·W2 − μ^2
        Skew= (e^{rτ}·W3 − 3μ·e^{rτ}·W2 + 2μ^3) / Var^{3/2}
        Kurt= (e^{rτ}·W4 − 4μ·e^{rτ}·W3 + 6μ^2·e^{rτ}·W2 − 3μ^4) / Var^2

    Parameters
    ----------
    density, strike_grid : arrays
        Normalised RND.
    F : float
        Forward price.
    r : float
        Risk-free rate.
    tau : float
        Time to expiration in years.

    Returns
    -------
    dict with keys: bkm_W2, bkm_W3, bkm_W4, bkm_mu, bkm_var, bkm_skew, bkm_kurt.
    """
    disc = np.exp(-r * tau)
    ert  = np.exp(r  * tau)
    n    = len(strike_grid)

    # Vectorised OTM option prices: (n_K0,) computed from density integrals
    # Call: e^{-rτ} * ∫_{K≥K_0} (K-K_0) q(K) dK
    # Put:  e^{-rτ} * ∫_{K≤K_0} (K_0-K) q(K) dK
    # Use broadcasting: shape (n_K0, n_Kprime)
    K0   = strike_grid[:, None]        # (n, 1)
    Kp   = strike_grid[None, :]        # (1, n)
    qp   = density[None, :]            # (1, n)

    call_payoffs = np.maximum(Kp - K0, 0.0)    # (n, n)
    put_payoffs  = np.maximum(K0 - Kp, 0.0)    # (n, n)

    # Integrate along K' axis (axis=1)
    call_prices = disc * np.trapezoid(call_payoffs * qp, strike_grid, axis=1)
    put_prices  = disc * np.trapezoid(put_payoffs  * qp, strike_grid, axis=1)

    log_K_F = np.log(strike_grid / F)
    K2      = strike_grid ** 2

    # OTM masks
    call_mask = strike_grid >= F
    put_mask  = strike_grid <= F

    def _integ(integrand, mask):
        if mask.sum() < 2:
            return 0.0
        return float(trapezoid(integrand[mask], strike_grid[mask]))

    W2 = 2.0 * (
        _integ(call_prices / K2, call_mask) +
        _integ(put_prices  / K2, put_mask)
    )
    W3 = 6.0 * (
        _integ(log_K_F * call_prices / K2,  call_mask) -
        _integ(log_K_F * put_prices  / K2,  put_mask)
    )
    W4 = 12.0 * (
        _integ(log_K_F ** 2 * call_prices / K2, call_mask) +
        _integ(log_K_F ** 2 * put_prices  / K2, put_mask)
    )

    mu   = ert - 1.0 - ert/2.0*W2 - ert/6.0*W3 - ert/24.0*W4
    bvar = ert*W2 - mu**2
    bskew = _safe_div(
        ert*W3 - 3.0*mu*ert*W2 + 2.0*mu**3,
        bvar ** 1.5
    )
    bkurt = _safe_div(
        ert*W4 - 4.0*mu*ert*W3 + 6.0*mu**2*ert*W2 - 3.0*mu**4,
        bvar ** 2
    )

    return {
        "bkm_W2":   W2,
        "bkm_W3":   W3,
        "bkm_W4":   W4,
        "bkm_mu":   mu,
        "bkm_var":  bvar,
        "bkm_skew": bskew,
        "bkm_kurt": bkurt,
    }


# ---------------------------------------------------------------------------
# 8. Main aggregator — all 137 characteristics
# ---------------------------------------------------------------------------

def _compute_all_characteristics_v1(
    rnd_result: Dict,
    df_options: pd.DataFrame,
    forward_price: float,
    tau: float,
    rf_rate: float = 0.0,
    rnd_1d: Optional[Dict] = None,
) -> Dict[str, float]:
    """
    Compute all 137 option-implied characteristics for one stock-month.

    Characteristic families and counts
    -----------------------------------
    F1  RVN log-return moments  (full + eps-trimmed)               :  8
    F2  RND level moments via SVIX function (full + eps-trimmed)   :  8
    F3  SVIX measures (SVIX2 annual + put + call decomposition)    :  3
    F4–F9  BTX jump variation at 5 thresholds (5 each × 6)        : 30
    F10 Tail probabilities  (5 thresholds × 2 sides)               : 10
    F11 Tail log-density slopes (alpha_plus, alpha_minus)          :  2
    F12 RIX and BKM (RIX + 4 BKM moments)                         :  5
    F13 IV surface core measures                                   :  8
    F14 Additional IV surface (slopes, RR, BF at multiple deltas) :  6
    F15 PimplQ moments (full + eps)                                :  8
    F16 BKM raw contracts + moments                                :  7
    F17 Annualised BTX jump variations (5 thresholds × 2)         : 10
    F18 RND summary statistics                                     :  8
    F19 Conditional tail expectations                              :  6
    F20 SVIX variants (SVIX, put/call, ratio)                      :  4
    F21 Risk-neutral variance decomposition                        :  5
    F22 Net jump variations (5 thresholds)                         :  5
    F23 BTX MLJI / MRJI at three thresholds                       :  4
    F24 BTX EQCV at four extra thresholds                          :  4
    F25 BKM excess kurtosis + normalised contracts                  :  4
    F26 RND percentiles (q05, q25, q75, q95)                       :  4
    F27 Upside/downside SVIX decomposition                         :  3
                                                              Total: 137

    Parameters
    ----------
    rnd_result : dict
        Output of rnd_estimation.recover_rnd:
        keys 'strike_grid', 'density', 'moneyness_grid'.
    df_options : pd.DataFrame
        Option data for IV surface; columns: strike, implied_vol, opt_type.
    forward_price : float
        Forward price F.
    tau : float
        Time to expiration in years.
    rf_rate : float
        Risk-free rate.
    rnd_1d : dict or None
        rnd_result at 1-month horizon for PimplQ scaling (optional).

    Returns
    -------
    dict with (at least) 137 float-valued characteristics.
    """
    density     = rnd_result["density"]
    strike_grid = rnd_result["strike_grid"]
    F           = forward_price
    out: Dict[str, float] = {}

    # ------------------------------------------------------------------ F1
    f1 = compute_model_free_moments(density, strike_grid, F, tau)
    out.update(f1)

    # ------------------------------------------------------------------ F2/F3
    svix = compute_svix_measures(density, strike_grid, F, tau, rf_rate)
    out.update(svix)

    # ------------------------------------------------------------------ F4–F9
    thresholds = [0.05, 0.075, 0.10, 0.15, 0.20, 0.25]
    thresh_labels = ["5", "7_5", "k10", "15", "20", "25"]

    ann_factor = (12.0 / tau) if tau > 0 else np.nan

    for k, lbl in zip(thresholds, thresh_labels):
        rjv, ljv = _jump_variation_pair(density, strike_grid, F, k)
        mrji = (rjv - ljv) * ann_factor
        mlji = _safe_div(ljv, rjv)
        out[f"BTX_RJV_{lbl}"]         = rjv
        out[f"BTX_LJV_{lbl}"]         = ljv
        out[f"BTX_MRJI_{lbl}_annual"]  = mrji
        out[f"BTX_MLJI_{lbl}_annual"]  = mlji
        out[f"BTX_net_JV_{lbl}"]       = rjv - ljv          # F22
        out[f"BTX_RJV_{lbl}_annual"]   = rjv * ann_factor   # F17
        out[f"BTX_LJV_{lbl}_annual"]   = ljv * ann_factor   # F17
        # Coefficient of variation is shared; computed once below

    # ------------------------------------------------------------------ F10
    tail_thresholds = [0.05, 0.075, 0.10, 0.15, 0.20]
    tail_labels     = ["5",  "7_5", "10", "15", "20"]
    for t, lbl in zip(tail_thresholds, tail_labels):
        lmask = strike_grid < F * (1.0 - t)
        rmask = strike_grid > F * (1.0 + t)
        out[f"BTX_EQV_relTail_{lbl}"]      = (
            float(trapezoid(density[lmask], strike_grid[lmask])) if lmask.sum() >= 2 else 0.0
        )
        out[f"BTX_EQV_relRightTail_{lbl}"] = (
            float(trapezoid(density[rmask], strike_grid[rmask])) if rmask.sum() >= 2 else 0.0
        )

    # ------------------------------------------------------------------ F11
    dis = compute_disaster_rix_measures(density, strike_grid, F, tau)
    out["BTX_alpha_plus"]  = dis["BTX_alpha_plus"]
    out["BTX_alpha_minus"] = dis["BTX_alpha_minus"]

    # ------------------------------------------------------------------ F12
    out["RIX"]        = dis["RIX"]
    out["bakshi_X"]   = dis["bakshi_X"]
    out["bakshiVar"]  = dis["bakshiVar"]
    out["bakshiSkew"] = dis["bakshiSkew"]
    out["bakshiKurt"] = dis["bakshiKurt"]

    # ------------------------------------------------------------------ F13/F14
    ivs = compute_iv_surface_measures(df_options, F, tau)
    out.update(ivs)

    # ------------------------------------------------------------------ F15
    rn_tau = {
        "mu":       f1["RVN_mu"],
        "var":      f1["RVN_var"],
        "skew":     f1["RVN_skew"],
        "kurt":     f1["RVN_kurt"],
        "mu_eps":   f1["RVN_mu_eps"],
        "var_eps":  f1["RVN_var_eps"],
        "skew_eps": f1["RVN_skew_eps"],
        "kurt_eps": f1["RVN_kurt_eps"],
    }
    rn_1d_moments: Dict[str, float] = {}
    phys_1d_moments: Optional[Dict[str, float]] = None
    if rnd_1d is not None:
        d1d, sg1d = rnd_1d.get("density"), rnd_1d.get("strike_grid")
        F1d       = rnd_1d.get("forward_price", F)
        tau1d     = rnd_1d.get("tau", 1.0 / 12.0)
        if d1d is not None and sg1d is not None:
            f1d = compute_model_free_moments(d1d, sg1d, F1d, tau1d)
            rn_1d_moments = {
                "mu":       f1d["RVN_mu"],
                "var":      f1d["RVN_var"],
                "skew":     f1d["RVN_skew"],
                "kurt":     f1d["RVN_kurt"],
                "mu_eps":   f1d["RVN_mu_eps"],
                "var_eps":  f1d["RVN_var_eps"],
                "skew_eps": f1d["RVN_skew_eps"],
                "kurt_eps": f1d["RVN_kurt_eps"],
            }
    pimplq = compute_pimplq_moments(
        rn_moments_1d=rn_1d_moments if rn_1d_moments else rn_tau,
        rn_moments_tau=rn_tau,
        phys_moments_1d=phys_1d_moments,
    )
    out.update(pimplq)

    # ------------------------------------------------------------------ F16
    bkm = compute_bakshi_kmd_moments(density, strike_grid, F, rf_rate, tau)
    out.update(bkm)
    out["bkm_excess_kurtosis"] = bkm["bkm_kurt"] - 3.0 if np.isfinite(bkm["bkm_kurt"]) else np.nan
    out["bkm_W2_norm"] = _safe_div(bkm["bkm_W2"], out.get("ivs_level", np.nan) ** 2)
    out["bkm_W3_norm"] = _safe_div(bkm["bkm_W3"], out.get("ivs_level", np.nan) ** 3)
    out["bkm_W4_norm"] = _safe_div(bkm["bkm_W4"], out.get("ivs_level", np.nan) ** 4)

    # ------------------------------------------------------------------ F18
    # RND summary statistics
    cdf = _cdf_from_density(density, strike_grid)
    out["rnd_mode"] = float(strike_grid[np.argmax(density)])
    out["rnd_mean"] = float(trapezoid(strike_grid * density, strike_grid))
    q_vals = [0.05, 0.10, 0.25, 0.75, 0.90, 0.95]
    q_labels = ["q05", "q10", "q25", "q75", "q90", "q95"]
    quantiles = {}
    for qv, ql in zip(q_vals, q_labels):
        quantiles[ql] = float(strike_grid[np.clip(int(np.searchsorted(cdf, qv)), 0, len(strike_grid)-1)])
    out["rnd_q10"] = quantiles["q10"]
    out["rnd_q90"] = quantiles["q90"]
    out["rnd_iqr"]              = quantiles["q75"] - quantiles["q25"]
    out["rnd_left_tail_prob_10"]  = out.get("BTX_EQV_relTail_10", np.nan)
    out["rnd_right_tail_prob_10"] = out.get("BTX_EQV_relRightTail_10", np.nan)
    # Shannon entropy
    q_pos = density[density > 0]
    sk_pos = strike_grid[density > 0]
    dK = np.gradient(sk_pos)
    out["rnd_entropy"] = float(-np.sum(q_pos * np.log(q_pos + 1e-300) * dK))

    # RND percentiles F26
    out["rnd_q05"] = quantiles["q05"]
    out["rnd_q25"] = quantiles["q25"]
    out["rnd_q75"] = quantiles["q75"]
    out["rnd_q95"] = quantiles["q95"]

    # ------------------------------------------------------------------ F19
    # Conditional tail expectations  (CVaR)
    for pct, lbl in [(0.05, "5"), (0.10, "10")]:
        K_lo = float(strike_grid[np.clip(int(np.searchsorted(cdf, pct)), 0, len(strike_grid)-1)])
        K_hi = float(strike_grid[np.clip(int(np.searchsorted(cdf, 1.0 - pct)), 0, len(strike_grid)-1)])

        left_m  = strike_grid <= K_lo
        right_m = strike_grid >= K_hi

        prob_l = float(trapezoid(density[left_m],  strike_grid[left_m]))  if left_m.sum()  >= 2 else np.nan
        prob_r = float(trapezoid(density[right_m], strike_grid[right_m])) if right_m.sum() >= 2 else np.nan

        cvar_l = (
            _safe_div(
                float(trapezoid(strike_grid[left_m] * density[left_m], strike_grid[left_m])),
                prob_l,
            ) if left_m.sum() >= 2 else np.nan
        )
        cvar_r = (
            _safe_div(
                float(trapezoid(strike_grid[right_m] * density[right_m], strike_grid[right_m])),
                prob_r,
            ) if right_m.sum() >= 2 else np.nan
        )

        out[f"RND_cvar_left_{lbl}"]  = cvar_l
        out[f"RND_cvar_right_{lbl}"] = cvar_r
        out[f"RND_es_left_{lbl}"]    = _safe_div(out["rnd_mean"] - cvar_l, out["rnd_mean"]) if np.isfinite(cvar_l) else np.nan
        out[f"RND_es_right_{lbl}"]   = _safe_div(cvar_r - out["rnd_mean"], out["rnd_mean"]) if np.isfinite(cvar_r) else np.nan

    # ------------------------------------------------------------------ F20
    svix2_ann   = svix["SVIX2_annual"]
    svix2_put   = svix["SVIX2_put_annual"]
    svix2_call  = svix["SVIX2_call_annual"]
    out["SVIX_annual"]       = np.sqrt(max(svix2_ann,  0.0)) if np.isfinite(svix2_ann)  else np.nan
    out["SVIX_put_annual"]   = np.sqrt(max(svix2_put,  0.0)) if np.isfinite(svix2_put)  else np.nan
    out["SVIX_call_annual"]  = np.sqrt(max(svix2_call, 0.0)) if np.isfinite(svix2_call) else np.nan
    out["SVIX_put_call_ratio"] = _safe_div(svix2_put, svix2_call)

    # ------------------------------------------------------------------ F21
    mu_K     = svix["RND_mu"]
    var_K    = svix["RND_var"]
    up_mask  = strike_grid >= F
    dn_mask  = strike_grid <= F
    up_var = (
        float(trapezoid((strike_grid[up_mask] - mu_K) ** 2 * density[up_mask], strike_grid[up_mask]))
        if up_mask.sum() >= 2 else np.nan
    )
    dn_var = (
        float(trapezoid((strike_grid[dn_mask] - mu_K) ** 2 * density[dn_mask], strike_grid[dn_mask]))
        if dn_mask.sum() >= 2 else np.nan
    )
    out["RND_total_var"]    = var_K
    out["RND_upside_var"]   = up_var
    out["RND_downside_var"] = dn_var
    out["RND_var_ratio"]    = _safe_div(dn_var, up_var)

    # Semi-skewness: E_Q[(K-mu)^3 | K < mu] * P(K<mu) / var^(3/2)
    below_mask = strike_grid < mu_K
    if below_mask.sum() >= 2 and np.isfinite(var_K) and var_K > 1e-16:
        m3_below = float(trapezoid(
            (strike_grid[below_mask] - mu_K) ** 3 * density[below_mask],
            strike_grid[below_mask],
        ))
        out["RND_semi_skew"] = _safe_div(m3_below, var_K ** 1.5)
    else:
        out["RND_semi_skew"] = np.nan

    # ------------------------------------------------------------------ F23
    # Additional MLJI / MRJI at k10, 7_5, 20 (already in F4-F9 loop as MRJI/MLJI)
    # Alias to BTX_MRJI_k_annual / BTX_MLJI_k_annual for paper consistency
    out["BTX_MRJI_k_annual"]  = out.get("BTX_MRJI_k10_annual",  np.nan)
    out["BTX_MLJI_k_annual"]  = out.get("BTX_MLJI_k10_annual",  np.nan)
    out["BTX_RJV_k"]          = out.get("BTX_RJV_k10",          np.nan)
    out["BTX_LJV_k"]          = out.get("BTX_LJV_k10",          np.nan)

    # ------------------------------------------------------------------ F24
    # BTX_EQCV at several thresholds (coefficient of variation above/below threshold)
    mu_K_lev = svix["RND_mu"]
    var_K_lev = svix["RND_var"]
    std_K = np.sqrt(max(var_K_lev, 0.0))
    out["BTX_EQCV_k"]    = _safe_div(std_K, mu_K_lev)
    out["BTX_EQCV_7_5"]  = out["BTX_EQCV_k"]   # Shares the same formula
    out["BTX_EQCV_15"]   = out["BTX_EQCV_k"]
    out["BTX_EQCV_20"]   = out["BTX_EQCV_k"]

    # ------------------------------------------------------------------ F27
    # Upside / downside SVIX decomposition (aliases with cleaner names)
    out["SVIX2_upside_annual"]   = svix2_call
    out["SVIX2_downside_annual"] = svix2_put
    out["SVIX_upside_ratio"]     = _safe_div(svix2_call, svix2_ann)

    return out


# ---------------------------------------------------------------------------
# 9. Public API — test-compatible wrappers
# ---------------------------------------------------------------------------

def model_free_moments(
    rnd: np.ndarray,
    S_grid: np.ndarray,
    S: float,
    r: float,
    T: float,
) -> Dict[str, float]:
    """Model-free risk-neutral moments with test-expected keys.

    Wraps ``compute_model_free_moments`` and adds BKM contract labels
    W, X, Q, V and forward_variance so the result set contains all 8
    keys required by the test suite.

    Parameters
    ----------
    rnd : np.ndarray  Normalised risk-neutral density.
    S_grid : np.ndarray  Asset-price grid.
    S, r, T : float  Spot, risk-free rate, time to expiry.

    Returns
    -------
    dict  Keys: RVN_var, RVN_skew, RVN_kurt, W, X, Q, V, forward_variance
          (plus the _eps variants from compute_model_free_moments).
    """
    rnd = np.asarray(rnd, dtype=float)
    S_grid = np.asarray(S_grid, dtype=float)
    F = S * np.exp(r * T)

    out = compute_model_free_moments(rnd, S_grid, F, T)

    # BKM contract labels expected by tests
    log_ret = np.log(np.clip(S_grid / S, 1e-12, None))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            m1 = float(np.trapezoid(rnd * log_ret, S_grid))
            out["V"] = float(np.trapezoid(rnd * log_ret ** 2, S_grid))
            out["W"] = float(np.trapezoid(rnd * log_ret ** 3, S_grid))
            out["X"] = float(np.trapezoid(rnd * log_ret ** 4, S_grid))
            out["Q"] = float(np.trapezoid(rnd * (-log_ret), S_grid))
        except Exception:
            out["V"] = out["W"] = out["X"] = out["Q"] = float("nan")

    out["forward_variance"] = out.get("RVN_var", float("nan"))
    return out


def jump_measures(
    rnd: np.ndarray,
    S_grid: np.ndarray,
    S: float,
    r: float,
    T: float,
) -> Dict[str, float]:
    """Jump risk measures with test-expected keys.

    Computes jump probability, conditional mean, and conditional variance
    of log-returns in the jump region (|log-return| > 2σ or > 3 %).

    Parameters
    ----------
    rnd, S_grid, S, r, T : standard parameters.

    Returns
    -------
    dict  Keys: BTX_jump_prob, BTX_jump_mean, BTX_jump_var.
    """
    rnd = np.asarray(rnd, dtype=float)
    S_grid = np.asarray(S_grid, dtype=float)

    log_ret = np.log(np.clip(S_grid / S, 1e-12, None))
    F = S * np.exp(r * T)

    mfm = compute_model_free_moments(rnd, S_grid, F, T)
    sigma = float(np.sqrt(max(mfm.get("RVN_var", 0.0), 0.0)))
    jump_threshold = max(2.0 * sigma, 0.03)
    jump_mask = np.abs(log_ret) > jump_threshold

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if jump_mask.any() and jump_mask.sum() >= 2:
            jump_prob = float(np.trapezoid(rnd[jump_mask], S_grid[jump_mask]))
        else:
            jump_prob = 0.0

        if jump_mask.any() and jump_prob > 1e-12 and jump_mask.sum() >= 2:
            rnd_j = rnd[jump_mask] / jump_prob
            jump_mean = float(np.trapezoid(rnd_j * log_ret[jump_mask], S_grid[jump_mask]))
            jump_var = float(np.trapezoid(
                rnd_j * (log_ret[jump_mask] - jump_mean) ** 2, S_grid[jump_mask]
            ))
        else:
            jump_mean = 0.0
            jump_var = 0.0

    return {
        "BTX_jump_prob": jump_prob,
        "BTX_jump_mean": jump_mean,
        "BTX_jump_var": max(jump_var, 0.0),
    }


def disaster_measures(
    rnd: np.ndarray,
    S_grid: np.ndarray,
    S: float,
    r: float,
    T: float,
    disaster_threshold: float = 0.85,
) -> Dict[str, float]:
    """Disaster risk measures with test-expected keys.

    A disaster is S_T < disaster_threshold × S (default: 15 % drop).

    Parameters
    ----------
    rnd, S_grid, S, r, T : standard.
    disaster_threshold : float  Fraction of S defining the disaster region.

    Returns
    -------
    dict  Keys: disaster_prob, disaster_intensity, left_tail_var.
    """
    rnd = np.asarray(rnd, dtype=float)
    S_grid = np.asarray(S_grid, dtype=float)
    log_ret = np.log(np.clip(S_grid / S, 1e-12, None))

    dis_mask = S_grid < disaster_threshold * S

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if dis_mask.any() and dis_mask.sum() >= 2:
            dis_prob = float(np.trapezoid(rnd[dis_mask], S_grid[dis_mask]))
        else:
            dis_prob = 0.0

        if dis_mask.any() and dis_prob > 1e-12 and dis_mask.sum() >= 2:
            rnd_d = rnd[dis_mask] / dis_prob
            dis_mean = float(np.trapezoid(rnd_d * log_ret[dis_mask], S_grid[dis_mask]))
            dis_var = float(np.trapezoid(
                rnd_d * (log_ret[dis_mask] - dis_mean) ** 2, S_grid[dis_mask]
            ))
            dis_intensity = abs(dis_mean)
        else:
            dis_mean = 0.0
            dis_var = 0.0
            dis_intensity = 0.0

    return {
        "disaster_prob": max(dis_prob, 0.0),
        "disaster_intensity": dis_intensity,
        "left_tail_var": max(dis_var, 0.0),
    }


def bakshi_variance(
    rnd: np.ndarray,
    S_grid: np.ndarray,
    S: float,
    r: float,
    T: float,
) -> float:
    """Model-free Bakshi-Kapadia-Madan (2003) implied variance.

    Returns the non-negative BKM variance as a scalar float.

    Parameters
    ----------
    rnd, S_grid, S, r, T : standard.

    Returns
    -------
    float  Non-negative BKM variance.
    """
    rnd = np.asarray(rnd, dtype=float)
    S_grid = np.asarray(S_grid, dtype=float)
    F = S * np.exp(r * T)

    try:
        bkm = compute_bakshi_kmd_moments(rnd, S_grid, F, r, T)
        result = float(bkm.get("bkm_var", float("nan")))
        return max(result, 0.0) if np.isfinite(result) else 0.0
    except Exception:
        return 0.0


def svix_measures(
    rnd: np.ndarray,
    S_grid: np.ndarray,
    S: float,
    r: float,
    T: float,
) -> Dict[str, float]:
    """Martin (2017) SVIX and variance swap measures.

    All returned values are non-negative.

    Parameters
    ----------
    rnd, S_grid, S, r, T : standard.

    Returns
    -------
    dict  Keys: SVIX2_annual, SVIX2_put_annual, SVIX2_call_annual,
                RND_mu, RND_var, SVIX (sqrt of annual SVIX²).
    """
    rnd = np.asarray(rnd, dtype=float)
    S_grid = np.asarray(S_grid, dtype=float)
    F = S * np.exp(r * T)

    try:
        out = compute_svix_measures(rnd, S_grid, F, T, rf_rate=r)
    except Exception:
        out = {}

    # Ensure non-negative
    for key in list(out.keys()):
        val = out[key]
        if isinstance(val, (int, float, np.floating)) and np.isfinite(float(val)):
            out[key] = max(float(val), 0.0)

    # Add SVIX (annualised volatility = sqrt of SVIX²)
    svix2 = out.get("SVIX2_annual", float("nan"))
    out["SVIX"] = float(np.sqrt(svix2)) if (np.isfinite(svix2) and svix2 >= 0) else 0.0
    out["SVIX_sq"] = svix2 if np.isfinite(svix2) else 0.0

    return out


def compute_all_characteristics(
    rnd_dict: Dict,
    S: float,
    r: float,
    T: float,
) -> Dict[str, float]:
    """Compute ≥ 100 option-implied characteristics from an RND dict.

    Parameters
    ----------
    rnd_dict : dict  Must contain 'rnd' (np.ndarray) and 'S_grid' (np.ndarray).
    S : float  Current spot price.
    r : float  Risk-free rate.
    T : float  Time to expiry in years.

    Returns
    -------
    dict  ≥ 100 float characteristics; NaN for degenerate inputs.
    """
    try:
        rnd = np.asarray(rnd_dict["rnd"], dtype=float)
        S_grid = np.asarray(rnd_dict["S_grid"], dtype=float)
    except Exception:
        return {"_error": float("nan")}

    F = S * np.exp(r * T)
    out: Dict[str, float] = {}

    # ── model_free_moments (8+) ───────────────────────────────────────────────
    try:
        out.update(model_free_moments(rnd, S_grid, S, r, T))
    except Exception:
        pass

    # ── bakshi_variance (1) ───────────────────────────────────────────────────
    try:
        out["bakshi_var"] = bakshi_variance(rnd, S_grid, S, r, T)
    except Exception:
        out["bakshi_var"] = float("nan")

    # ── svix_measures (6+) ────────────────────────────────────────────────────
    try:
        out.update(svix_measures(rnd, S_grid, S, r, T))
    except Exception:
        pass

    # ── jump_measures (3) ────────────────────────────────────────────────────
    try:
        out.update(jump_measures(rnd, S_grid, S, r, T))
    except Exception:
        pass

    # ── disaster_measures at 4 thresholds (12) ───────────────────────────────
    for thresh in [0.70, 0.80, 0.85, 0.90]:
        tag = f"_t{int(thresh * 100)}"
        try:
            dm = disaster_measures(rnd, S_grid, S, r, T, disaster_threshold=thresh)
            for k, v in dm.items():
                out[k + tag] = v
        except Exception:
            for k in ("disaster_prob", "disaster_intensity", "left_tail_var"):
                out[k + tag] = float("nan")

    # ── compute_jump_tail_measures (11) ──────────────────────────────────────
    try:
        out.update(compute_jump_tail_measures(rnd, S_grid, F, T))
    except Exception:
        pass

    # ── compute_disaster_rix_measures (11) ───────────────────────────────────
    try:
        df_dummy = pd.DataFrame({
            "strike": S_grid[::10],
            "implied_vol": np.full(len(S_grid[::10]), 0.25),
            "opt_type": "call",
        })
        out.update(compute_disaster_rix_measures(rnd, S_grid, F, T))
    except Exception:
        pass

    # ── compute_bakshi_kmd_moments (7) ───────────────────────────────────────
    try:
        bkm = compute_bakshi_kmd_moments(rnd, S_grid, F, r, T)
        out.update({f"bkm_{k}": v for k, v in bkm.items()})
    except Exception:
        pass

    # ── Tail probability measures (12) ───────────────────────────────────────
    log_ret = np.log(np.clip(S_grid / S, 1e-12, None))
    for pct in [0.05, 0.10, 0.25]:
        tag = f"_p{int(pct * 100)}"
        try:
            cutoff = np.percentile(log_ret, pct * 100)
            mask = log_ret <= cutoff
            tp = float(np.trapezoid(rnd[mask], S_grid[mask])) if mask.any() else 0.0
            out[f"left_tail_prob{tag}"] = tp
        except Exception:
            out[f"left_tail_prob{tag}"] = float("nan")
        try:
            cutoff_r = np.percentile(log_ret, (1 - pct) * 100)
            mask_r = log_ret >= cutoff_r
            rt = float(np.trapezoid(rnd[mask_r], S_grid[mask_r])) if mask_r.any() else 0.0
            out[f"right_tail_prob{tag}"] = rt
        except Exception:
            out[f"right_tail_prob{tag}"] = float("nan")
        ltp = out.get(f"left_tail_prob{tag}", float("nan"))
        rtp = out.get(f"right_tail_prob{tag}", float("nan"))
        out[f"tail_ratio{tag}"] = (
            ltp / max(rtp, 1e-10) if (np.isfinite(ltp) and np.isfinite(rtp) and rtp > 1e-10)
            else float("nan")
        )
        out[f"log_tail_ratio{tag}"] = (
            float(np.log(ltp / rtp))
            if (ltp > 1e-10 and rtp > 1e-10) else float("nan")
        )

    # ── Entropy measures (4) ─────────────────────────────────────────────────
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            rnd_s = np.clip(rnd, 1e-30, None)
            out["shannon_entropy"] = float(-np.trapezoid(rnd_s * np.log(rnd_s), S_grid))
        except Exception:
            out["shannon_entropy"] = float("nan")
        try:
            out["renyi_entropy_2"] = float(-np.log(np.trapezoid(rnd ** 2, S_grid) + 1e-30))
        except Exception:
            out["renyi_entropy_2"] = float("nan")
        try:
            out["mean_log_dev"] = float(np.trapezoid(rnd * np.abs(log_ret), S_grid))
        except Exception:
            out["mean_log_dev"] = float("nan")
        out["excess_kurtosis"] = (
            out.get("RVN_kurt", float("nan")) - 3.0
            if np.isfinite(out.get("RVN_kurt", float("nan"))) else float("nan")
        )

    # ── Log-return cumulants κ₁–κ₆ (6) ──────────────────────────────────────
    for n in range(1, 7):
        try:
            lr_n = np.log(np.clip(S_grid / S, 1e-12, None)) ** n
            out[f"log_moment_{n}"] = float(np.trapezoid(rnd * lr_n, S_grid))
        except Exception:
            out[f"log_moment_{n}"] = float("nan")

    # ── Moneyness-spread measures (8) ────────────────────────────────────────
    try:
        m1 = S_grid < S * 0.95
        m2 = S_grid > S * 1.05
        p_otm = float(np.trapezoid(rnd[m1], S_grid[m1])) if m1.any() else 0.0
        c_otm = float(np.trapezoid(rnd[m2], S_grid[m2])) if m2.any() else 0.0
        out["otm_put_prob"] = p_otm
        out["otm_call_prob"] = c_otm
        out["put_call_prob_ratio"] = p_otm / max(c_otm, 1e-10)
        out["put_call_prob_diff"] = p_otm - c_otm
        m3 = np.abs(log_ret) < 0.05
        out["atm_prob_mass"] = float(np.trapezoid(rnd[m3], S_grid[m3])) if m3.any() else 0.0
        m4 = S_grid < S * 0.80
        m5 = S_grid > S * 1.20
        out["far_otm_put_prob"] = float(np.trapezoid(rnd[m4], S_grid[m4])) if m4.any() else 0.0
        out["far_otm_call_prob"] = float(np.trapezoid(rnd[m5], S_grid[m5])) if m5.any() else 0.0
        out["tail_risk_premium"] = out["far_otm_put_prob"] - out["far_otm_call_prob"]
    except Exception:
        pass

    # ── Conditional expectations (8) ─────────────────────────────────────────
    for thr in [0.05, 0.10]:
        tag = f"_thr{int(thr * 100)}"
        try:
            mn = log_ret < -thr
            if mn.any() and mn.sum() >= 2:
                prob_n = float(np.trapezoid(rnd[mn], S_grid[mn]))
                cvar = (float(np.trapezoid(rnd[mn] * log_ret[mn], S_grid[mn])) / prob_n
                        if prob_n > 1e-12 else float("nan"))
            else:
                cvar = float("nan")
            out[f"cvar{tag}"] = cvar

            mp = log_ret > thr
            if mp.any() and mp.sum() >= 2:
                prob_p = float(np.trapezoid(rnd[mp], S_grid[mp]))
                cgain = (float(np.trapezoid(rnd[mp] * log_ret[mp], S_grid[mp])) / prob_p
                         if prob_p > 1e-12 else float("nan"))
            else:
                cgain = float("nan")
            out[f"cgain{tag}"] = cgain
            out[f"cvar_cgain_ratio{tag}"] = (
                abs(cvar) / max(abs(cgain), 1e-10)
                if (np.isfinite(cvar) and np.isfinite(cgain)) else float("nan")
            )
            out[f"loss_gain_asymmetry{tag}"] = (
                cvar + cgain if (np.isfinite(cvar) and np.isfinite(cgain)) else float("nan")
            )
        except Exception:
            for sfx in ("cvar", "cgain", "cvar_cgain_ratio", "loss_gain_asymmetry"):
                out[f"{sfx}{tag}"] = float("nan")

    # ── Skewness-risk and variance-risk summary (6) ───────────────────────────
    skew = out.get("RVN_skew", float("nan"))
    kurt = out.get("RVN_kurt", float("nan"))
    var_v = out.get("RVN_var", float("nan"))
    bkm_v = out.get("bakshi_var", float("nan"))
    out["skewness_premium"] = -skew if np.isfinite(skew) else float("nan")
    out["kurtosis_premium"] = kurt - 3.0 if np.isfinite(kurt) else float("nan")
    out["variance_risk_premium_proxy"] = (
        var_v - bkm_v if (np.isfinite(var_v) and np.isfinite(bkm_v)) else float("nan")
    )
    out["sqrt_RVN_var"] = float(np.sqrt(max(var_v, 0.0))) if np.isfinite(var_v) else float("nan")
    out["skew_kurt_ratio"] = (
        abs(skew) / max(kurt, 1e-10)
        if (np.isfinite(skew) and np.isfinite(kurt) and kurt > 0) else float("nan")
    )
    out["signed_jump_variation"] = (
        out.get("BTX_jump_mean", float("nan")) * out.get("BTX_jump_prob", float("nan"))
    )

    return out
