"""
Data loading utilities for option-implied characteristics project.
Supports Massive API for equity data, with synthetic data fallback for testing.

Replication assumptions documented:
- Risk-free rate: 3-month T-bill from FRED converted to monthly
- S&P 500 membership: approximated from market-cap ranking if historical data unavailable
- Option data: uses Massive API equity data as proxy; full option chain requires CBOE data
"""

import os
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def get_massive_client():
    """Get Massive REST client using MASSIVE_TOKEN env variable."""
    try:
        from massive import RESTClient
        api_key = os.getenv('MASSIVE_TOKEN')
        if api_key:
            return RESTClient(api_key=api_key)
        return None
    except ImportError:
        return None


def load_risk_free_rate(start: str = '2004-01-01', end: str = '2023-07-31') -> pd.Series:
    """
    Load 3-month Treasury bill rate from FRED as monthly risk-free rate.
    Falls back to synthetic constant if FRED unavailable.
    Returns monthly decimal rate (not annualized).

    Parameters
    ----------
    start:
        Start date string in ``YYYY-MM-DD`` format.
    end:
        End date string in ``YYYY-MM-DD`` format.

    Returns
    -------
    pd.Series
        Monthly risk-free rate in decimal form, indexed by month-end dates.
        Series name is ``'rf'``.
    """
    try:
        import pandas_datareader.data as web
        rf_annual = web.DataReader('TB3MS', 'fred', start, end)
        rf_monthly = rf_annual['TB3MS'] / 100 / 12  # annualized pct to monthly decimal
        rf_monthly = rf_monthly.resample('ME').last()
        rf_monthly.name = 'rf'
        return rf_monthly
    except Exception:
        warnings.warn('FRED unavailable. Using synthetic risk-free rate of 0.04/12 per month.')
        dates = pd.date_range(start, end, freq='ME')
        return pd.Series(0.04 / 12, index=dates, name='rf')


def load_french_factors(start: str = '2004-01-01', end: str = '2023-07-31') -> pd.DataFrame:
    """
    Load Kenneth French 5-factor model data plus momentum, short-term reversal,
    and long-term reversal. Returns 8-column DataFrame of monthly factor returns.

    Factors returned
    ----------------
    MKT, SMB, HML, RMW, CMA : Fama-French five factors
    MOM                      : Carhart momentum factor
    STREV                    : Short-term reversal
    LTREV                    : Long-term reversal

    Parameters
    ----------
    start:
        Start date string in ``YYYY-MM-DD`` format.
    end:
        End date string in ``YYYY-MM-DD`` format.

    Returns
    -------
    pd.DataFrame
        Monthly factor returns in decimal form, indexed by month-end dates.
    """
    try:
        import pandas_datareader.data as web
        ff5 = web.DataReader('F-F_Research_Data_5_Factors_2x3', 'famafrench', start, end)[0]
        mom = web.DataReader('F-F_Momentum_Factor', 'famafrench', start, end)[0]
        strev = web.DataReader('F-F_ST_Reversal_Factor', 'famafrench', start, end)[0]
        ltrev = web.DataReader('F-F_LT_Reversal_Factor', 'famafrench', start, end)[0]
        ff5 = ff5.drop(columns=['RF'], errors='ignore') / 100
        mom = mom / 100
        strev = strev / 100
        ltrev = ltrev / 100
        combined = ff5.join(mom, how='inner').join(strev, how='inner').join(ltrev, how='inner')
        combined.columns = ['MKT', 'SMB', 'HML', 'RMW', 'CMA', 'MOM', 'STREV', 'LTREV']
        combined.index = pd.to_datetime(combined.index.to_timestamp())
        return combined
    except Exception:
        warnings.warn('French factors unavailable. Generating synthetic benchmark factors.')
        dates = pd.date_range(start, end, freq='ME')
        rng = np.random.RandomState(42)
        factor_names = ['MKT', 'SMB', 'HML', 'RMW', 'CMA', 'MOM', 'STREV', 'LTREV']
        data = rng.normal(0.005, 0.04, (len(dates), 8))
        return pd.DataFrame(data, index=dates, columns=factor_names)


def generate_synthetic_option_data(
    n_stocks: int = 50,
    n_months: int = 234,
    n_strikes_per_stock: int = 10,
    start_date: str = '2004-02-01',
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic stock-level option data for testing and demonstration.

    Produces monthly option chain snapshots per stock with:
    - Multiple strikes around ATM
    - Both calls and puts
    - Implied volatility following a volatility smile

    Parameters
    ----------
    n_stocks:
        Number of synthetic stocks.
    n_months:
        Number of monthly periods to generate.
    n_strikes_per_stock:
        Number of strike prices per stock per period.
    start_date:
        First month-end date for the panel in ``YYYY-MM-DD`` format.
    random_state:
        Seed for ``numpy.random.RandomState`` to ensure reproducibility.

    Returns
    -------
    pd.DataFrame
        Long-format option chain with one row per (stock, date, strike, option type).
    """
    rng = np.random.RandomState(random_state)
    dates = pd.date_range(start_date, periods=n_months, freq='ME')
    stock_ids = [f'STOCK_{i:04d}' for i in range(n_stocks)]

    records = []
    for date in dates:
        for stock_id in stock_ids:
            spot = 100 * np.exp(rng.normal(0, 0.3))
            rf_daily = 0.04 / 252
            tau_days = 30
            forward = spot * np.exp(rf_daily * tau_days)

            # ATM vol with stock-specific level
            atm_vol = 0.2 + rng.uniform(-0.1, 0.2)

            # Generate strikes: moneyness grid around ATM
            moneyness_grid = np.linspace(0.80, 1.20, n_strikes_per_stock)
            strikes = forward * moneyness_grid

            for k, m in zip(strikes, moneyness_grid):
                # Volatility smile: skew + convexity
                skew = -0.05
                convexity = 0.03
                log_m = np.log(m)
                iv = atm_vol + skew * log_m + convexity * log_m**2 + rng.normal(0, 0.005)
                iv = max(0.05, iv)  # floor

                for opt_type in ['C', 'P']:
                    records.append({
                        'permno': stock_id,
                        'date': date,
                        'strike': round(k, 4),
                        'expiry_date': date + timedelta(days=tau_days),
                        'days_to_maturity': tau_days,
                        'opt_type': opt_type,
                        'implied_vol': iv,
                        'spot_price': spot,
                        'forward_price': forward,
                        'moneyness': m,
                        'mid_price': max(0.01, spot * 0.05 * rng.uniform(0.5, 1.5)),
                        'open_interest': int(rng.exponential(500)),
                        'bid_price': max(0.005, spot * 0.04 * rng.uniform(0.5, 1.5)),
                        'ask_price': max(0.01, spot * 0.06 * rng.uniform(0.5, 1.5)),
                    })

    df = pd.DataFrame(records)
    return df


def generate_synthetic_equity_data(
    n_stocks: int = 50,
    n_months: int = 235,  # Jan 2004 to Jul 2023
    start_date: str = '2004-01-01',
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic equity panel data for testing.
    Includes returns, market cap, S&P 500 flag, and 152 equity characteristics.

    Parameters
    ----------
    n_stocks:
        Number of synthetic stocks.
    n_months:
        Number of monthly periods to generate.
    start_date:
        First month-end date for the panel in ``YYYY-MM-DD`` format.
    random_state:
        Seed for ``numpy.random.RandomState`` to ensure reproducibility.

    Returns
    -------
    pd.DataFrame
        Long-format equity panel with one row per (stock, date).  Columns include
        ``permno``, ``date``, ``ret``, ``me``, ``sp500_flag``, and
        ``char_001`` through ``char_152``.
    """
    rng = np.random.RandomState(random_state)
    dates = pd.date_range(start_date, periods=n_months, freq='ME')
    stock_ids = [f'STOCK_{i:04d}' for i in range(n_stocks)]

    records = []
    # Generate random factor exposures for each stock
    n_chars = 152
    char_cols = [f'char_{i:03d}' for i in range(1, n_chars + 1)]

    stock_chars = {s: rng.normal(0, 1, n_chars) for s in stock_ids}
    stock_logme = {s: np.log(rng.lognormal(8, 1.5)) for s in stock_ids}

    for date in dates:
        for stock_id in stock_ids:
            base_ret = rng.normal(0.005, 0.05)
            me = np.exp(stock_logme[stock_id]) * np.exp(rng.normal(0, 0.05))
            chars = stock_chars[stock_id] + rng.normal(0, 0.1, n_chars)

            row: Dict = {
                'permno': stock_id,
                'date': date,
                'ret': base_ret,
                'me': me,
                'sp500_flag': int(rng.uniform() < 0.4),  # ~40% in S&P 500
            }
            for col, val in zip(char_cols, chars):
                row[col] = val
            records.append(row)

    return pd.DataFrame(records)


def load_sp500_history(start: str = '2004-01-01', end: str = '2023-07-31') -> pd.DataFrame:
    """
    Load historical S&P 500 constituent data.
    Returns DataFrame with columns: permno, date, sp500_flag.
    Uses synthetic data as substitute since real data requires CRSP.

    Replication assumption: Real implementation requires CRSP/Compustat point-in-time
    S&P 500 constituent history. This synthetic version approximates with
    random 40% selection stable across time.

    Parameters
    ----------
    start:
        Start date string in ``YYYY-MM-DD`` format.
    end:
        End date string in ``YYYY-MM-DD`` format.

    Returns
    -------
    pd.DataFrame
        Empty placeholder DataFrame.  Actual constituent data is integrated via
        ``generate_synthetic_equity_data`` for testing purposes.
    """
    warnings.warn(
        'Using synthetic S&P 500 history. '
        'Real replication requires CRSP/Compustat constituent data.'
    )
    return pd.DataFrame()  # Placeholder; actual data integrated via generate_synthetic_equity_data
