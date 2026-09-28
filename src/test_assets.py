"""
Test asset construction via independent 3x2 characteristic-size sorts.

Follows Feng, Giglio, and Xiu (2020) conventions.
Six portfolios per characteristic: 3 characteristic bins x 2 size bins.
Universe: full intersected universe (not S&P 500 restricted).

Replication assumption: tercile characteristic splits and median size split,
value-weighted within portfolios, following Feng et al. (2020) conventions.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import warnings

# Labels for the 6 portfolios produced by a 3x2 sort.
# Naming convention: {size_label}_{char_label}
# size: S = small (below median ME), B = big (at or above median ME)
# char: L = bottom tercile, M = middle tercile, H = top tercile
_SIZE_LABELS = ["S", "B"]
_CHAR_LABELS = ["L", "M", "H"]


def _portfolio_col(characteristic_col: str, size_label: str, char_label: str) -> str:
    return f"{characteristic_col}_{size_label}_{char_label}"


def build_3x2_test_assets(
    equity_df: pd.DataFrame,
    characteristic_col: str,
    date_col: str = "date",
    ret_col: str = "ret",
    me_col: str = "me",
    weighting: str = "value",
    char_breaks: str = "tercile",
    size_breaks: str = "median",
) -> pd.DataFrame:
    """
    Build six portfolios via an independent 3x2 characteristic-size sort.

    At each month end, stocks in the full intersected universe are
    independently sorted into three characteristic bins and two size bins.
    The six intersections form the test portfolios whose returns are
    computed for the following month.

    Parameters
    ----------
    equity_df : pd.DataFrame
        Full intersected universe stock-month panel. Each row is a
        stock-month; the characteristic column reflects end-of-month
        information while ret_col is the forward month's return.
    characteristic_col : str
        Column used for the 3-way characteristic sort.
    date_col : str
        Column identifying the calendar month.
    ret_col : str
        Forward-month return column.
    me_col : str
        Market equity column used for value weighting and the size sort.
    weighting : str
        ``'value'`` for value-weighted portfolios (default) or ``'equal'``
        for equal-weighted portfolios.
    char_breaks : str
        Only ``'tercile'`` is supported (three equal-count bins).
    size_breaks : str
        Only ``'median'`` is supported (split at the cross-sectional median
        of market equity).

    Returns
    -------
    pd.DataFrame
        Shape (T, 6); date-indexed. Column names follow the pattern
        ``{characteristic_col}_{size}_{char}`` where size ∈ {S, B} and
        char ∈ {L, M, H}.
    """
    if char_breaks != "tercile":
        raise ValueError(f"char_breaks='{char_breaks}' is not supported; use 'tercile'.")
    if size_breaks != "median":
        raise ValueError(f"size_breaks='{size_breaks}' is not supported; use 'median'.")
    if weighting not in ("value", "equal"):
        raise ValueError(f"weighting='{weighting}' is not supported; use 'value' or 'equal'.")

    portfolio_cols = [
        _portfolio_col(characteristic_col, s, c)
        for s in _SIZE_LABELS
        for c in _CHAR_LABELS
    ]

    df = equity_df.dropna(subset=[characteristic_col, ret_col, me_col]).copy()

    records: List[Dict] = []
    for date, month_df in df.groupby(date_col):
        month_df = month_df.copy()

        # Independent characteristic tercile assignment
        try:
            month_df["_char_bin"] = pd.qcut(
                month_df[characteristic_col],
                3,
                labels=[0, 1, 2],
                duplicates="drop",
            )
        except ValueError:
            continue

        if month_df["_char_bin"].isna().all():
            continue

        # Independent size median split (0 = small, 1 = big)
        size_median = month_df[me_col].median()
        month_df["_size_bin"] = (month_df[me_col] >= size_median).astype(int)

        row: Dict = {date_col: date}
        for s_idx, s_label in enumerate(_SIZE_LABELS):
            for c_idx, c_label in enumerate(_CHAR_LABELS):
                mask = (month_df["_char_bin"] == c_idx) & (month_df["_size_bin"] == s_idx)
                group = month_df[mask]
                col_name = _portfolio_col(characteristic_col, s_label, c_label)

                if group.empty or group[me_col].sum() == 0:
                    row[col_name] = np.nan
                elif weighting == "value":
                    w = group[me_col] / group[me_col].sum()
                    row[col_name] = float((w * group[ret_col]).sum())
                else:
                    row[col_name] = float(group[ret_col].mean())

        records.append(row)

    if not records:
        return pd.DataFrame(columns=portfolio_cols)

    result = (
        pd.DataFrame(records)
        .set_index(date_col)
        .reindex(columns=portfolio_cols)
    )
    result.index.name = date_col
    return result


def build_all_test_assets(
    equity_df: pd.DataFrame,
    characteristic_cols: List[str],
    date_col: str = "date",
    ret_col: str = "ret",
    me_col: str = "me",
    weighting: str = "value",
) -> pd.DataFrame:
    """
    Build 3x2 test-asset portfolios for every characteristic.

    Calls :func:`build_3x2_test_assets` for each characteristic and
    concatenates the results into a wide DataFrame.

    Parameters
    ----------
    equity_df : pd.DataFrame
        Full intersected universe stock-month panel.
    characteristic_cols : list of str
        Characteristics to sort on; each produces 6 portfolio columns.
    date_col : str
        Date column.
    ret_col : str
        Forward-month return column.
    me_col : str
        Market equity column.
    weighting : str
        ``'value'`` (default) or ``'equal'``.

    Returns
    -------
    pd.DataFrame
        Shape (T, 6 * len(characteristic_cols)); date-indexed. Columns
        follow the naming convention ``{char}_{size}_{bin}``.
    """
    all_assets: List[pd.DataFrame] = []
    for col in characteristic_cols:
        portfolio_df = build_3x2_test_assets(
            equity_df,
            col,
            date_col=date_col,
            ret_col=ret_col,
            me_col=me_col,
            weighting=weighting,
        )
        all_assets.append(portfolio_df)

    if not all_assets:
        return pd.DataFrame()

    return pd.concat(all_assets, axis=1)


def compute_equal_weighted_test_return(test_asset_df: pd.DataFrame) -> pd.Series:
    """
    Equal-weight all test portfolios to produce a single return series.

    The target series used in the screening step is the simple cross-
    portfolio average, computed month-by-month while ignoring NaN columns.

    Parameters
    ----------
    test_asset_df : pd.DataFrame
        Wide DataFrame of test-asset returns as returned by
        :func:`build_all_test_assets` or :func:`build_3x2_test_assets`.
        Rows are months, columns are portfolio returns.

    Returns
    -------
    pd.Series
        Monthly equal-weighted average return across all test portfolios,
        indexed by date. Named ``'ew_target'``.
    """
    return test_asset_df.mean(axis=1).rename("ew_target")


def validate_test_assets(
    test_asset_df: pd.DataFrame,
    expected_avg_size: int = 369,
    size_col: Optional[str] = None,
    equity_df: Optional[pd.DataFrame] = None,
    date_col: str = "date",
    me_col: str = "me",
    characteristic_cols: Optional[List[str]] = None,
) -> Dict:
    """
    Validate test-asset panel dimensions and portfolio sizes.

    Checks that the panel is non-empty, reports the fraction of missing
    portfolio-months, and optionally computes average portfolio stock count
    when the underlying equity panel is provided.

    Parameters
    ----------
    test_asset_df : pd.DataFrame
        Wide panel of test-asset returns from :func:`build_all_test_assets`.
    expected_avg_size : int
        Target average number of stocks per portfolio-month (default 369,
        consistent with the Feng et al. 2020 sample).
    size_col : str, optional
        Not used directly; retained for API consistency.
    equity_df : pd.DataFrame, optional
        If supplied together with ``characteristic_cols``, the function
        computes the empirical average portfolio size and compares it to
        ``expected_avg_size``.
    date_col : str
        Date column in ``equity_df``.
    me_col : str
        Market equity column in ``equity_df``.
    characteristic_cols : list of str, optional
        Characteristics used to form the test assets. Required when
        ``equity_df`` is provided.

    Returns
    -------
    dict
        Keys:
        - ``n_portfolios`` (int): number of portfolio columns.
        - ``n_obs`` (int): number of monthly rows.
        - ``missing_pct`` (float): fraction of portfolio-months that are NaN.
        - ``avg_portfolio_size`` (float or None): mean stocks per portfolio-month
          (only populated when equity_df is provided).
        - ``expected_avg_size`` (int): target avg size.
        - ``avg_size_ok`` (bool or None): whether sizes match within 20%.
    """
    n_portfolios = test_asset_df.shape[1]
    n_obs = test_asset_df.shape[0]
    missing_pct = float(test_asset_df.isnull().mean().mean())

    avg_portfolio_size: Optional[float] = None
    avg_size_ok: Optional[bool] = None

    if equity_df is not None and characteristic_cols is not None:
        sizes: List[float] = []
        for char_col in characteristic_cols:
            df = equity_df.dropna(subset=[char_col]).copy()
            try:
                df["_char_bin"] = pd.qcut(
                    df.groupby(date_col)[char_col].transform(
                        lambda x: pd.qcut(x, 3, labels=False, duplicates="drop")
                    ),
                    3,
                    labels=False,
                    duplicates="drop",
                )
            except Exception:
                pass
            counts = df.groupby(date_col).size()
            # Approximate each of the 6 bins as 1/6 of the total
            sizes.extend((counts / 6).tolist())

        if sizes:
            avg_portfolio_size = float(np.mean(sizes))
            avg_size_ok = abs(avg_portfolio_size - expected_avg_size) / expected_avg_size <= 0.20

    if missing_pct > 0.10:
        warnings.warn(
            f"Test-asset panel has {missing_pct:.1%} missing portfolio-months."
        )
    if avg_size_ok is False:
        warnings.warn(
            f"Average portfolio size {avg_portfolio_size:.0f} deviates from "
            f"expected {expected_avg_size} by more than 20%."
        )

    return {
        "n_portfolios": n_portfolios,
        "n_obs": n_obs,
        "missing_pct": missing_pct,
        "avg_portfolio_size": avg_portfolio_size,
        "expected_avg_size": expected_avg_size,
        "avg_size_ok": avg_size_ok,
    }
