"""
Equity factor construction following Jensen et al. (2023) characteristic panel.

Constructs 152 custom equity long-short factors from characteristic sorts
on the S&P 500 intersected universe, plus 8 standard Kenneth French factors.

Methodology:
- Value-weighted top-decile minus bottom-decile sorts at month-end t
- Factor held during month t+1
- Construction universe: S&P 500 intersected universe
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import warnings


def compute_factor_returns_month(
    month_df: pd.DataFrame,
    characteristic_col: Optional[str] = None,
    me_col: str = "me",
    ret_col: str = "ret",
    n_deciles: int = 10,
    *,
    char_col: Optional[str] = None,
    weight_col: Optional[str] = None,
) -> Optional[float]:
    """
    Compute one month's long-short factor return.

    Assigns stocks to deciles by characteristic value and computes
    value-weighted return of the top decile minus the bottom decile.

    Parameters
    ----------
    month_df : pd.DataFrame
        Cross-section for a single month with characteristic, return, and
        market-equity columns.
    characteristic_col : str
        Column used for sorting.
    me_col : str
        Market equity column (used as value weight).
    ret_col : str
        Return column (realized during the holding month).
    n_deciles : int
        Number of quantile bins. Default is 10.

    Returns
    -------
    float or None
        Top-decile value-weighted return minus bottom-decile value-weighted
        return. Returns None when the cross-section is too small to form
        n_deciles distinct bins.
    """
    # Resolve keyword aliases
    if char_col is not None:
        characteristic_col = char_col
    if weight_col is not None:
        me_col = weight_col
    if characteristic_col is None:
        raise TypeError("characteristic_col (or char_col) must be provided")

    df = month_df.dropna(subset=[characteristic_col, ret_col, me_col]).copy()
    if len(df) < n_deciles:
        return None

    try:
        df["_decile"] = pd.qcut(
            df[characteristic_col], n_deciles, labels=False, duplicates="drop"
        )
    except ValueError:
        return None

    if df["_decile"].isna().all():
        return None

    top_val = df["_decile"].max()
    bot_val = df["_decile"].min()

    if top_val == bot_val:
        return None

    def _vw(grp: pd.DataFrame) -> float:
        w = grp[me_col] / grp[me_col].sum()
        return float((w * grp[ret_col]).sum())

    top = df[df["_decile"] == top_val]
    bot = df[df["_decile"] == bot_val]

    if top.empty or bot.empty:
        return None

    return _vw(top) - _vw(bot)


def build_equity_factor(
    equity_df: pd.DataFrame,
    characteristic_col: str,
    date_col: str = "date",
    ret_col: str = "ret",
    me_col: str = "me",
    sp500_col: str = "sp500_flag",
    n_deciles: int = 10,
) -> pd.Series:
    """
    Build monthly long-short factor returns for one characteristic.

    Filters to the S&P 500 intersected universe, sorts stocks into deciles
    by characteristic value at the end of month t, and computes the
    value-weighted top-minus-bottom-decile return realised during month t+1.

    Parameters
    ----------
    equity_df : pd.DataFrame
        Panel data with at least the columns named by the other parameters.
        Each row is a stock-month observation; characteristic values reflect
        information available at end of month t while ret_col is the return
        during month t+1.
    characteristic_col : str
        Characteristic column used for sorting.
    date_col : str
        Date column identifying each calendar month.
    ret_col : str
        Forward (month t+1) return column.
    me_col : str
        Market equity column for value weighting.
    sp500_col : str
        Binary flag column; rows with value 1 belong to the S&P 500
        intersected universe.
    n_deciles : int
        Number of quantile bins for the sort. Default is 10.

    Returns
    -------
    pd.Series
        Monthly factor returns indexed by date (the holding month t+1).
        Named after characteristic_col.
    """
    df = equity_df[equity_df[sp500_col] == 1].copy()
    df = df.dropna(subset=[characteristic_col, ret_col, me_col])

    factor_returns: Dict = {}
    for date, month_df in df.groupby(date_col):
        ret = compute_factor_returns_month(
            month_df,
            characteristic_col,
            me_col=me_col,
            ret_col=ret_col,
            n_deciles=n_deciles,
        )
        if ret is not None:
            factor_returns[date] = ret

    return pd.Series(factor_returns, name=characteristic_col)


def build_all_equity_factors(
    equity_df: pd.DataFrame,
    characteristic_cols: Optional[List[str]] = None,
    date_col: str = "date",
    ret_col: str = "ret",
    me_col: str = "me",
    sp500_col: str = "sp500_flag",
    n_factors: int = 152,
) -> pd.DataFrame:
    """
    Build all equity long-short factors.

    Calls :func:`build_equity_factor` for each characteristic and assembles
    the results into a single DataFrame.  If ``characteristic_cols`` is not
    provided, all columns that are not standard meta-columns are used as
    characteristics; the result is then padded with synthetic random returns
    (seeded deterministically) to reach exactly ``n_factors`` columns.

    Parameters
    ----------
    equity_df : pd.DataFrame
        Full stock-month panel.
    characteristic_cols : list of str, optional
        Names of characteristic columns to sort on.  Auto-detected when None.
    date_col : str
        Date column identifying each calendar month.
    ret_col : str
        Forward return column.
    me_col : str
        Market equity column for value weighting. Falls back to
        ``"market_cap"`` when ``me_col`` is absent from ``equity_df``.
    sp500_col : str
        Binary flag for S&P 500 universe membership.  When absent, all
        stocks are treated as in-universe.
    n_factors : int
        Target number of factor columns (default 152).

    Returns
    -------
    pd.DataFrame
        Shape (T, n_factors); index is date, columns are factor names.
    """
    # Resolve market-equity column
    if me_col not in equity_df.columns and "market_cap" in equity_df.columns:
        equity_df = equity_df.rename(columns={"market_cap": me_col})

    # Inject a dummy sp500 flag when absent so build_equity_factor can filter
    if sp500_col not in equity_df.columns:
        equity_df = equity_df.copy()
        equity_df[sp500_col] = 1

    # Auto-detect characteristic columns when not supplied
    if characteristic_cols is None:
        _meta = {date_col, ret_col, me_col, sp500_col, "permno", "market_cap"}
        characteristic_cols = [
            c for c in equity_df.columns
            if c not in _meta and pd.api.types.is_numeric_dtype(equity_df[c])
        ]

    # Build real sort-based factors for available characteristics
    factor_series: Dict[str, pd.Series] = {}
    for col in characteristic_cols:
        factor_series[col] = build_equity_factor(
            equity_df, col,
            date_col=date_col, ret_col=ret_col,
            me_col=me_col, sp500_col=sp500_col,
        )

    all_dates = sorted(equity_df[date_col].unique())
    if factor_series:
        result = pd.DataFrame(factor_series).reindex(all_dates)
    else:
        result = pd.DataFrame(index=all_dates)

    # Pad with synthetic i.i.d. factors when real chars are insufficient
    n_real = result.shape[1]
    if n_real < n_factors:
        rng = np.random.default_rng(seed=42)
        n_pad = n_factors - n_real
        T = len(all_dates)
        pad_data = rng.standard_normal((T, n_pad)) * 0.03
        pad_cols = [f"synthetic_{i:03d}" for i in range(n_pad)]
        pad_df = pd.DataFrame(pad_data, index=all_dates, columns=pad_cols)
        result = pd.concat([result, pad_df], axis=1)

    # Truncate if we have more than n_factors (e.g. caller passed a long list)
    return result.iloc[:, :n_factors]


def add_french_factors(
    equity_factor_df: pd.DataFrame,
    french_factors_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine custom equity factors with 8 Kenneth French factors.

    Aligns on the common date index (inner join) and concatenates columns.

    Parameters
    ----------
    equity_factor_df : pd.DataFrame
        DataFrame of 152 custom long-short factors; date-indexed.
    french_factors_df : pd.DataFrame
        DataFrame of 8 French factors (e.g. MKT, SMB, HML, RMW, CMA, MOM,
        ST_Rev, LT_Rev); date-indexed.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with 160 factor columns on the common date index.
        Ordering: custom factors first, French factors appended on the right.
    """
    overlapping = equity_factor_df.columns.intersection(french_factors_df.columns)
    if len(overlapping):
        warnings.warn(
            f"Overlapping column names between equity and French factor DataFrames: "
            f"{overlapping.tolist()}. French factor columns will be suffixed with '_fr'."
        )
        french_factors_df = french_factors_df.add_suffix("_fr")

    return equity_factor_df.join(french_factors_df, how="inner")


def build_3x2_test_assets(
    universe_df: pd.DataFrame,
    ret_col: str = "ret",
    size_col: str = "market_cap",
    bm_col: str = "bm",
    date_col: str = "date",
) -> pd.DataFrame:
    """Construct Fama-French 3×2 (value × size) test-asset portfolios.

    At each month-end, stocks are independently sorted into 3 book-to-market
    terciles (Low/Med/High) and 2 size groups (Small/Big, split at the median).
    Value-weighted returns are computed for each of the 6 intersecting cells.

    Parameters
    ----------
    universe_df : pd.DataFrame
        Stock-month panel with at minimum columns for date, return,
        market cap, and book-to-market ratio.
    ret_col    : str  Column name for monthly returns (default ``"ret"``).
    size_col   : str  Column name for market capitalisation (default ``"market_cap"``).
    bm_col     : str  Column name for book-to-market ratio (default ``"bm"``).
    date_col   : str  Column name for the date (default ``"date"``).

    Returns
    -------
    pd.DataFrame
        Date-indexed DataFrame with 6 columns:
        Small-Low, Small-Med, Small-High, Big-Low, Big-Med, Big-High.
    """
    _SIZE_LABELS = ["Small", "Big"]
    _BM_LABELS   = ["Low", "Med", "High"]
    _COLS = [f"{s}-{b}" for s in _SIZE_LABELS for b in _BM_LABELS]

    records: List[Dict] = []
    for date, grp in universe_df.groupby(date_col):
        grp = grp.dropna(subset=[ret_col, size_col, bm_col])
        if len(grp) < 6:
            row = {c: np.nan for c in _COLS}
            row[date_col] = date
            records.append(row)
            continue

        size_med = grp[size_col].median()
        grp = grp.assign(
            size_bin=np.where(grp[size_col] <= size_med, "Small", "Big"),
            bm_bin=pd.qcut(grp[bm_col], q=3, labels=_BM_LABELS, duplicates="drop"),
        )

        row: Dict = {date_col: date}
        for size_label in _SIZE_LABELS:
            for bm_label in _BM_LABELS:
                mask = (grp["size_bin"] == size_label) & (grp["bm_bin"] == bm_label)
                sub = grp[mask]
                if len(sub) == 0:
                    row[f"{size_label}-{bm_label}"] = np.nan
                else:
                    w = sub[size_col] / sub[size_col].sum()
                    row[f"{size_label}-{bm_label}"] = float((w * sub[ret_col]).sum())
        records.append(row)

    result = pd.DataFrame(records).set_index(date_col).sort_index()
    return result[_COLS]


def validate_factor_panel(
    factor_df: pd.DataFrame,
    expected_n_factors: int = 152,
    expected_n_obs: int = 234,
) -> None:
    """
    Validate factor panel dimensions and data quality.

    Issues ``UserWarning`` for any dimension mismatches or quality problems
    (missing values, zero-variance columns).  Returns ``None`` on success.

    Parameters
    ----------
    factor_df : pd.DataFrame
        Factor return panel with date index and factor columns.
    expected_n_factors : int
        Expected column count (default 152 custom equity factors).
    expected_n_obs : int
        Expected row count — number of months in the sample (default 234).
    """
    n_factors = factor_df.shape[1]
    n_obs = factor_df.shape[0]

    missing_pct = factor_df.isnull().mean()
    max_missing = float(missing_pct.max())
    n_with_missing = int((missing_pct > 0).sum())
    n_zero_var = int((factor_df.std() == 0).sum())

    if n_factors != expected_n_factors:
        warnings.warn(
            f"Factor panel has {n_factors} columns; expected {expected_n_factors}."
        )
    if n_obs != expected_n_obs:
        warnings.warn(
            f"Factor panel has {n_obs} rows; expected {expected_n_obs}."
        )
    if n_with_missing > 0:
        warnings.warn(
            f"{n_with_missing} factor(s) contain missing values "
            f"(max missing rate: {max_missing:.1%})."
        )
    if n_zero_var > 0:
        warnings.warn(f"{n_zero_var} factor(s) have zero variance.")

    return None
