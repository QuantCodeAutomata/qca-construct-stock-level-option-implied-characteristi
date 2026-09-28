"""
Data contracts and schemas for the option-implied characteristics project.

Defines the column requirements for equity and option DataFrames, the canonical
grouping of option-implied characteristics into research families, and validation
helpers that enforce those contracts at runtime.
"""

from typing import Dict, List

import pandas as pd


# ---------------------------------------------------------------------------
# Required column lists
# ---------------------------------------------------------------------------

REQUIRED_EQUITY_COLS: List[str] = [
    'permno',
    'date',
    'ret',
    'me',
    'sp500_flag',
]
"""Minimum columns that every equity panel DataFrame must contain."""

REQUIRED_OPTION_COLS: List[str] = [
    'permno',
    'date',
    'expiry_date',
    'strike',
    'opt_type',
    'mid_price',
    'implied_vol',
    'spot_price',
    'forward_price',
    'days_to_maturity',
]
"""Minimum columns that every option chain DataFrame must contain."""


# ---------------------------------------------------------------------------
# Characteristic families
# ---------------------------------------------------------------------------

CHARACTERISTIC_FAMILIES: Dict[str, List[str]] = {
    'model_free_moments': [
        # Risk-neutral moments derived without assuming a distributional model
        'RVN_mu',
        'RVN_var',
        'RVN_skew',
        'RVN_kurt',
        'RVN_mu_eps',
        'RVN_var_eps',
        'RVN_skew_eps',
        'RVN_kurt_eps',
    ],
    'jump_tail': [
        # Jump and tail-risk measures (Bollerslev-Todorov-Xu style)
        'BTX_RJV_k',          # Right jump variation, threshold k
        'BTX_LJV_k',          # Left jump variation, threshold k
        'BTX_MRJI_k_annual',  # Mean right jump intensity, annualised
        'BTX_MLJI_k_annual',  # Mean left jump intensity, annualised
        'BTX_RJV_7_5',        # Right jump variation, 7.5% threshold
        'BTX_LJV_7_5',        # Left jump variation, 7.5% threshold
        'BTX_MRJI_20_annual', # Mean right jump intensity, 20-day horizon
        'BTX_MLJI_20_annual', # Mean left jump intensity, 20-day horizon
        'BTX_EQCV_k',         # Equity crash-variance
    ],
    'disaster': [
        # Disaster-risk measures derived from option prices
        'BTX_EQV_relTail_k',       # Relative left-tail equity variance
        'BTX_EQV_relRightTail_k',  # Relative right-tail equity variance
        'BTX_EQV_relTail_10',      # Relative left-tail equity variance, 10% strike
        'BTX_EQV_relRightTail_10', # Relative right-tail equity variance, 10% strike
        'BTX_alpha_plus',           # Positive tail index
        'BTX_alpha_minus',          # Negative tail index
    ],
    'rix': [
        # Variance and higher-moment measures from Bakshi, Kapadia, Madan (2003)
        'RIX',          # Relative implied variance index
        'bakshi_X',     # BKM variance contract price
        'bakshiVar',    # BKM model-free variance
        'bakshiSkew',   # BKM model-free skewness
        'bakshiKurt',   # BKM model-free kurtosis
    ],
    'iv_surface': [
        # Parametric summaries of the implied-volatility surface
        'ivs_level',               # ATM implied-volatility level
        'ivs_slope',               # Term-structure slope
        'ivs_smirk',               # Moneyness skew (smirk)
        'ivs_convexity',           # Second-order moneyness curvature
        'ivs_term_structure',      # Short- vs long-dated IV spread
        'ivs_atm_vol',             # At-the-money implied volatility
        'ivs_atm_put_call_spread', # ATM put minus call IV spread
        'ivs_vol_spread',          # Vol-spread factor
    ],
    'oimp_moments': [
        # Option-implied moments computed via numerical integration of the RND
        'PimplQ_var',      # Option-implied risk-neutral variance
        'PimplQ_skew',     # Option-implied risk-neutral skewness
        'PimplQ_kurt',     # Option-implied risk-neutral kurtosis
        'PimplQ_var_eps',  # Standardised option-implied variance
        'PimplQ_skew_eps', # Standardised option-implied skewness
        'PimplQ_kurt_eps', # Standardised option-implied kurtosis
    ],
    'svix': [
        # Simple variance and risk-neutral distribution summaries (Martin 2017)
        'SVIX2_annual',     # Annualised SVIX squared
        'SVIX2_put_annual', # Put-leg annualised SVIX squared
        'SVIX2_call_annual',# Call-leg annualised SVIX squared
        'RND_mu',           # Risk-neutral distribution mean
        'RND_var',          # Risk-neutral distribution variance
        'RND_skew',         # Risk-neutral distribution skewness
        'RND_kurt',         # Risk-neutral distribution kurtosis
        'RND_mu_eps',       # Standardised RND mean
        'RND_var_eps',      # Standardised RND variance
        'RND_skew_eps',     # Standardised RND skewness
        'RND_kurt_eps',     # Standardised RND kurtosis
    ],
}
"""
Mapping of research family name to ordered list of characteristic column names.

Families follow the taxonomy used in Bali, Beckmeyer, Moerke, and Weigert (2023).
Each family can be extended independently as additional characteristics are
implemented.
"""


# ---------------------------------------------------------------------------
# Flat characteristic lists
# ---------------------------------------------------------------------------

ALL_OPTION_CHARS: List[str] = [
    char
    for chars in CHARACTERISTIC_FAMILIES.values()
    for char in chars
]
"""
Flat, ordered list of every option-implied characteristic name across all families.

Derived by concatenating the per-family lists in ``CHARACTERISTIC_FAMILIES`` in
insertion order.  Contains one entry per implemented characteristic (53 as of the
current family definitions; the target for full replication is 137).
"""

EQUITY_CHAR_COLS: List[str] = [f'char_{i:03d}' for i in range(1, 153)]
"""
Generic column names for the 152 equity characteristics used in cross-sectional
factor construction.

Names follow the pattern ``char_NNN`` (zero-padded to three digits).  In a full
replication these map to the 153-anomaly set of Chen and Zimmermann (2022), minus
the one characteristic reserved for the option-implied signal under study.
"""

FACTOR_UNIVERSE_COLS: List[str] = ['factor_name', 'date', 'return']
"""Column schema for the long-format factor-return panel."""


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_equity_data(df: pd.DataFrame) -> None:
    """
    Raise ``ValueError`` if *df* is missing any required equity column.

    Parameters
    ----------
    df:
        Equity panel DataFrame to validate.

    Raises
    ------
    ValueError
        Lists every missing column name in the error message.
    """
    missing: List[str] = [col for col in REQUIRED_EQUITY_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"Equity DataFrame is missing required columns: {missing}")


def validate_option_data(df: pd.DataFrame) -> None:
    """
    Raise ``ValueError`` if *df* is missing any required option column.

    Parameters
    ----------
    df:
        Option chain DataFrame to validate.

    Raises
    ------
    ValueError
        Lists every missing column name in the error message.
    """
    missing: List[str] = [col for col in REQUIRED_OPTION_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"Option DataFrame is missing required columns: {missing}")
