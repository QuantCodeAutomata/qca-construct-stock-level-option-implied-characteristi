"""
Experiment 6: Robustness checks.

1. Maturity robustness: Re-run screening and joint tests for synthetic option
   factor panels constructed at maturities [1, 2, 3, 5, 10, 30, 60] days
   (or months, depending on convention).
2. Risk-free rate robustness: Compare TOPT (Treasury Overnight Funding Rate
   proxy) vs FRED T-bill rates (both synthetic here) as the discount rate.
3. Summarise results in a robustness table and save to results/.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import json
import warnings
from pathlib import Path
from datetime import datetime

from src.joint_tests import (
    grs_test,
    compute_max_sharpe,
    compare_sharpe_ratios,
    stewart_love_redundancy,
)

warnings.filterwarnings("ignore")

# ── Constants ────────────────────────────────────────────────────────────────
MATURITIES = [1, 2, 3, 5, 10, 30, 60]    # calendar days (synthetic proxy)
N_MONTHS = 234
N_OPTION_FACTORS = 15
N_EQUITY_FACTORS = 20     # keep small for robustness loops
SEED = 42
RESULTS_DIR = Path(__file__).parent.parent / "results"


# ── Synthetic data generators ────────────────────────────────────────────────

def _synthetic_option_factors(maturity: int, seed: int) -> pd.DataFrame:
    """Generate synthetic option factors whose IV varies with maturity."""
    rng = np.random.default_rng(seed + maturity)
    dates = pd.date_range("1996-01", periods=N_MONTHS, freq="MS")
    # Longer maturities → lower volatility in factor returns (term structure)
    vol_scale = 0.04 / np.sqrt(maturity / 21)   # normalise by ~1-month
    data = rng.standard_normal((N_MONTHS, N_OPTION_FACTORS)) * vol_scale
    # Inject a maturity-specific level shift
    level_shift = np.linspace(0.001, -0.001, N_OPTION_FACTORS) * (maturity / 30)
    data += level_shift[None, :]
    return pd.DataFrame(
        data, index=dates, columns=[f"char_{i:03d}" for i in range(N_OPTION_FACTORS)]
    )


def _synthetic_equity_factors(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 999)
    dates = pd.date_range("1996-01", periods=N_MONTHS, freq="MS")
    data = rng.standard_normal((N_MONTHS, N_EQUITY_FACTORS)) * 0.03
    return pd.DataFrame(
        data, index=dates, columns=[f"eq_{i:03d}" for i in range(N_EQUITY_FACTORS)]
    )


def _synthetic_rf(name: str, seed: int) -> pd.Series:
    """Simulate TOPT or FRED T-bill proxy."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("1996-01", periods=N_MONTHS, freq="MS")
    base = 0.04 / 12   # ~4% annualised / 12
    if name == "TOPT":
        rf = base + rng.normal(0, 0.0001, N_MONTHS)   # low noise overnight rate
    else:   # FRED 1-month T-bill
        rf = base + rng.normal(0, 0.0003, N_MONTHS)   # slightly more volatile
    return pd.Series(np.clip(rf, 0.0, None), index=dates, name=name)


# ── Robustness runners ───────────────────────────────────────────────────────

def run_maturity_robustness(equity_factors: pd.DataFrame) -> pd.DataFrame:
    """Run joint tests for each maturity; return summary DataFrame."""
    rows = []
    for mat in MATURITIES:
        print(f"  Maturity {mat:3d} days …", end="", flush=True)
        op = _synthetic_option_factors(mat, SEED)

        idx = op.index.intersection(equity_factors.index)
        op_a = op.loc[idx]
        eq_a = equity_factors.loc[idx]

        # GRS (use small benchmark subset for speed)
        try:
            K_bench = min(8, eq_a.shape[1])
            grs = grs_test(op_a, eq_a.iloc[:, :K_bench])
            grs_stat = grs["grs_stat"]
            grs_p = grs["p_value"]
        except Exception:
            grs_stat, grs_p = np.nan, np.nan

        # Sharpe comparison
        sharpe = compare_sharpe_ratios(eq_a, op_a)

        # Redundancy (use small subsets)
        n_op = min(8, op_a.shape[1])
        n_eq = min(10, eq_a.shape[1])
        try:
            red = stewart_love_redundancy(op_a.iloc[:, :n_op], eq_a.iloc[:, :n_eq])
            ri_opt = red["RI_options_given_equity"]
            ri_eq = red["RI_equity_given_options"]
        except Exception:
            ri_opt, ri_eq = np.nan, np.nan

        row = {
            "maturity_days": mat,
            "grs_stat": round(grs_stat, 4),
            "grs_p_value": round(grs_p, 4),
            "sr_equity": round(sharpe["sr_equity"], 4),
            "sr_option": round(sharpe["sr_option"], 4),
            "sr_combined": round(sharpe["sr_combined"], 4),
            "sr_improvement": round(sharpe["sr_improvement"], 4),
            "RI_option_given_equity": round(ri_opt, 4),
            "RI_equity_given_option": round(ri_eq, 4),
        }
        rows.append(row)
        print(f" GRS={grs_stat:.3f} (p={grs_p:.3f}), SR_imp={sharpe['sr_improvement']:+.2%}")

    return pd.DataFrame(rows)


def run_rf_robustness(equity_factors: pd.DataFrame) -> pd.DataFrame:
    """Compare TOPT vs FRED RF rate for Sharpe ratio computation."""
    op = _synthetic_option_factors(21, SEED)   # 21-day (1-month) maturity
    idx = op.index.intersection(equity_factors.index)
    op_a = op.loc[idx]
    eq_a = equity_factors.loc[idx]

    rows = []
    for rf_name in ["TOPT", "FRED"]:
        print(f"  RF proxy: {rf_name} …", end="", flush=True)
        rf = _synthetic_rf(rf_name, SEED).reindex(idx)
        sharpe = compare_sharpe_ratios(eq_a, op_a, rf_series=rf)
        row = {
            "rf_proxy": rf_name,
            "rf_mean_annual": round(float(rf.mean() * 12), 4),
            "sr_equity": round(sharpe["sr_equity"], 4),
            "sr_option": round(sharpe["sr_option"], 4),
            "sr_combined": round(sharpe["sr_combined"], 4),
            "sr_improvement": round(sharpe["sr_improvement"], 4),
        }
        rows.append(row)
        print(f" SR_equity={sharpe['sr_equity']:.4f}, SR_imp={sharpe['sr_improvement']:+.2%}")

    return pd.DataFrame(rows)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("Experiment 6: Robustness Checks")
    print("=" * 60)
    print(f"  Maturities tested : {MATURITIES}")
    print(f"  RF proxies        : TOPT, FRED")
    print()

    # Generate base equity factors once
    equity_factors = _synthetic_equity_factors(SEED)

    # ── Panel A: Maturity robustness ──────────────────────────────────────
    print("[Panel A] Maturity robustness:")
    mat_table = run_maturity_robustness(equity_factors)
    print("\n  Summary table:")
    print(mat_table.to_string(index=False))

    # ── Panel B: RF rate robustness ───────────────────────────────────────
    print("\n[Panel B] Risk-free rate robustness:")
    rf_table = run_rf_robustness(equity_factors)
    print("\n  Summary table:")
    print(rf_table.to_string(index=False))

    # ── Save ──────────────────────────────────────────────────────────────
    print("\n[Saving results …]")
    mat_path = RESULTS_DIR / "exp6_maturity_robustness.csv"
    rf_path = RESULTS_DIR / "exp6_rf_robustness.csv"

    mat_table.to_csv(mat_path, index=False)
    rf_table.to_csv(rf_path, index=False)

    meta = {
        "maturities_tested": MATURITIES,
        "rf_proxies": ["TOPT", "FRED"],
        "n_months": N_MONTHS,
        "timestamp": datetime.utcnow().isoformat(),
    }
    with open(RESULTS_DIR / "exp6_meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"  Maturity table → {mat_path}")
    print(f"  RF table       → {rf_path}")
    print("\nExperiment 6 complete.\n")


if __name__ == "__main__":
    main()
