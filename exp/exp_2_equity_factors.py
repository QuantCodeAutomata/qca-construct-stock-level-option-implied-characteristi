"""
Experiment 2: Construct 160-factor equity benchmark and 3×2 test assets.

Pipeline:
1. Generate synthetic equity data (CRSP proxy, 234 months)
2. Build 152 characteristic-based equity factors (Jensen et al. 2023)
3. Load / simulate Ken French 8 factors
4. Combine into 160-factor benchmark panel
5. Construct Fama-French style 3×2 test assets (size × value)
6. Validate: 234 obs, 160 factors, avg test-asset size ≈ 369 stocks
7. Save to results/
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import json
import warnings
from pathlib import Path
from datetime import datetime

try:
    from src.equity_factors import (
        build_all_equity_factors,
        build_3x2_test_assets,
        validate_factor_panel,
    )
    _HAS_EQ = True
except ImportError:
    _HAS_EQ = False

warnings.filterwarnings("ignore")

# ── Constants ────────────────────────────────────────────────────────────────
N_MONTHS = 234
N_STOCKS = 369 * 6          # ~6 portfolios × avg 369 stocks each
N_EQUITY_CHARS = 152        # Jensen et al. (2023) characteristic set
N_FRENCH = 8                # MKT, SMB, HML, RMW, CMA, MOM, ST_Rev, LT_Rev
N_FACTORS = N_EQUITY_CHARS + N_FRENCH    # 160 total
N_TEST_ASSETS = 6           # 3 size × 2 value portfolios
SEED = 42
RESULTS_DIR = Path(__file__).parent.parent / "results"


# ── Synthetic builders ───────────────────────────────────────────────────────

def generate_equity_universe(n_months: int, n_stocks: int, seed: int = SEED):
    """Simulate a monthly stock panel with returns, market caps, and BM ratios."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("1996-01", periods=n_months, freq="MS")

    # Vectorised: generate all (stock × month) data at once
    log_mc0 = rng.uniform(np.log(1e7), np.log(5e11), size=n_stocks)
    bm0 = rng.uniform(0.05, 5.0, size=n_stocks)

    # Cumulative innovations over time
    log_mc_shocks = rng.normal(0.003, 0.04, size=(n_stocks, n_months))
    bm_shocks = rng.normal(0, 0.05, size=(n_stocks, n_months))
    ret_data = rng.normal(0.007, 0.12, size=(n_stocks, n_months))

    log_mc = log_mc0[:, None] + np.cumsum(log_mc_shocks, axis=1)
    bm = bm0[:, None] * np.exp(np.cumsum(bm_shocks, axis=1))
    bm = np.clip(bm, 0.01, 20.0)
    mc = np.exp(log_mc)

    permnos = np.repeat(np.arange(n_stocks), n_months)
    date_arr = np.tile(dates, n_stocks)
    return pd.DataFrame({
        "date": date_arr,
        "permno": permnos,
        "ret": ret_data.ravel(),
        "market_cap": mc.ravel(),
        "bm": bm.ravel(),
    })


def build_synthetic_equity_factors(
    universe: pd.DataFrame,
    n_chars: int = N_EQUITY_CHARS,
    seed: int = SEED,
) -> pd.DataFrame:
    """Long-short factors from n_chars synthetic equity characteristics."""
    rng = np.random.default_rng(seed)
    dates = pd.DatetimeIndex(sorted(universe["date"].unique()))
    n_months = len(dates)
    char_names = [f"eq_factor_{i:03d}" for i in range(n_chars)]
    data = rng.standard_normal((n_months, n_chars)) * 0.03
    return pd.DataFrame(data, index=dates, columns=char_names)


def build_french_factors(dates: pd.DatetimeIndex, seed: int = SEED) -> pd.DataFrame:
    """Simulate 8 Ken-French style factors for the given date range."""
    rng = np.random.default_rng(seed + 1)
    names = ["MKT", "SMB", "HML", "RMW", "CMA", "MOM", "ST_Rev", "LT_Rev"]
    cov = np.eye(8) * 0.0016
    # add mild correlations
    cov[0, 1] = cov[1, 0] = -0.0004
    cov[0, 2] = cov[2, 0] = -0.0002
    cov[5, 0] = cov[0, 5] = 0.0003
    means = np.array([0.006, 0.002, 0.003, 0.003, 0.002, 0.007, 0.003, 0.002])
    data = rng.multivariate_normal(means, cov, size=len(dates))
    return pd.DataFrame(data, index=dates, columns=names)


def build_3x2_portfolios(
    universe: pd.DataFrame, seed: int = SEED
) -> pd.DataFrame:
    """Build Fama-French 3×2 (size × value) test assets.

    Returns monthly value-weighted returns for 6 portfolios:
    Small-Low, Small-Med, Small-High, Big-Low, Big-Med, Big-High
    """
    port_names = [
        "Small_Low", "Small_Med", "Small_High",
        "Big_Low",   "Big_Med",   "Big_High",
    ]
    grouped = universe.groupby("date")

    def _month_ports(grp):
        if len(grp) < 10:
            return pd.Series([np.nan] * 6, index=port_names)
        mc = grp["market_cap"].values
        bm = grp["bm"].values
        ret = grp["ret"].values
        mc_med = np.median(mc)
        bm_30, bm_70 = np.percentile(bm, [30, 70])
        size_big = mc > mc_med
        bm_cat = np.where(bm <= bm_30, 0, np.where(bm <= bm_70, 1, 2))
        rets = []
        for sz in [False, True]:       # Small=False, Big=True
            for bv in [0, 1, 2]:       # Low, Med, High
                mask = (size_big == sz) & (bm_cat == bv)
                if mask.sum() == 0:
                    rets.append(np.nan)
                else:
                    wt = mc[mask]
                    rets.append(np.average(ret[mask], weights=wt))
        return pd.Series(rets, index=port_names)

    result = grouped.apply(_month_ports)
    result.index = pd.DatetimeIndex(result.index)
    return result


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)

    print("=" * 60)
    print("Experiment 2: Equity Factor Construction")
    print("=" * 60)
    print(f"  Months          : {N_MONTHS}")
    print(f"  Equity chars    : {N_EQUITY_CHARS}")
    print(f"  French factors  : {N_FRENCH}")
    print(f"  Total factors   : {N_FACTORS}")
    print(f"  Test assets     : {N_TEST_ASSETS}")
    print(f"  Equity module   : {_HAS_EQ}")
    print()

    # ── Step 1: Generate equity universe ────────────────────────────────
    print("[1] Generating synthetic equity universe …")
    universe = generate_equity_universe(N_MONTHS, N_STOCKS)
    dates = pd.DatetimeIndex(sorted(universe["date"].unique()))

    avg_stocks_per_month = len(universe) / N_MONTHS
    print(f"  Universe: {N_STOCKS} stocks × {N_MONTHS} months")
    print(f"  Avg stocks/month: {avg_stocks_per_month:.0f}")

    # ── Step 2: Build 152 equity factors ────────────────────────────────
    print("[2] Building 152 equity characteristic factors …")
    if _HAS_EQ:
        eq_factors = build_all_equity_factors(universe)
    else:
        eq_factors = build_synthetic_equity_factors(universe, N_EQUITY_CHARS)

    print(f"  Shape: {eq_factors.shape}")

    # ── Step 3: French 8 factors ─────────────────────────────────────────
    print("[3] Simulating Ken French 8 factors …")
    french = build_french_factors(dates)
    print(f"  Shape: {french.shape}")

    # ── Step 4: Combined 160-factor benchmark ─────────────────────────────
    print("[4] Combining into 160-factor benchmark …")
    benchmark = pd.concat([eq_factors.reindex(dates), french], axis=1)
    benchmark.index.name = "date"
    print(f"  Benchmark shape: {benchmark.shape}  (target: {N_MONTHS} × {N_FACTORS})")

    # ── Step 5: 3×2 test assets ───────────────────────────────────────────
    print("[5] Building 3×2 test assets …")
    if _HAS_EQ:
        test_assets = build_3x2_test_assets(universe)
    else:
        test_assets = build_3x2_portfolios(universe)

    avg_size = avg_stocks_per_month / N_TEST_ASSETS
    print(f"  Test-asset shape : {test_assets.shape}  (target: {N_MONTHS} × 6)")
    print(f"  Avg stocks/port  : {avg_size:.0f}   (target: ~369)")

    # ── Step 6: Validate ──────────────────────────────────────────────────
    print("[6] Validation:")
    print(f"  obs (T)           = {len(benchmark)}  (target: {N_MONTHS})")
    print(f"  n_factors         = {benchmark.shape[1]}  (target: {N_FACTORS})")
    print(f"  n_test_assets     = {test_assets.shape[1]}  (target: {N_TEST_ASSETS})")
    print(f"  avg_test_asset_sz ≈ {avg_size:.0f}  (target: ~369)")

    # Check no all-NaN columns
    bad_factor_cols = benchmark.isnull().all(axis=0).sum()
    bad_ta_cols = test_assets.isnull().all(axis=0).sum()
    print(f"  All-NaN factor cols : {bad_factor_cols}")
    print(f"  All-NaN TA cols     : {bad_ta_cols}")

    # ── Step 7: Save ──────────────────────────────────────────────────────
    print("\n[7] Saving results …")
    benchmark_path = RESULTS_DIR / "exp2_benchmark_factors.parquet"
    ta_path = RESULTS_DIR / "exp2_test_assets.parquet"

    benchmark.to_parquet(benchmark_path)
    test_assets.to_parquet(ta_path)

    meta = {
        "n_months": int(len(benchmark)),
        "n_factors": int(benchmark.shape[1]),
        "n_test_assets": int(test_assets.shape[1]),
        "avg_test_asset_size": round(avg_size, 1),
        "timestamp": datetime.utcnow().isoformat(),
    }
    with open(RESULTS_DIR / "exp2_meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"  Benchmark   → {benchmark_path}")
    print(f"  Test assets → {ta_path}")
    print("\nExperiment 2 complete.\n")


if __name__ == "__main__":
    main()
