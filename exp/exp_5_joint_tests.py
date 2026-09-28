"""
Experiment 5: Joint alpha tests (GRS), maximum Sharpe ratio comparisons,
and CCA redundancy analysis.

Pipeline:
1. Load screened option factors (exp_3) and benchmark equity factors (exp_2),
   or synthesise proxies.
2. Run GRS F-test for joint alpha = 0 of option factors vs equity benchmark.
3. Compare maximum Sharpe ratios: equity-only, option-only, combined.
4. Run CCA + Stewart-Love redundancy analysis.
5. Print and save all results.
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
    canonical_correlation_analysis,
    stewart_love_redundancy,
    run_joint_tests,
)
from src.cca_redundancy import full_redundancy_analysis

warnings.filterwarnings("ignore")

# ── Constants ────────────────────────────────────────────────────────────────
N_MONTHS = 234
N_OPTION_FACTORS = 15     # screened subset (from exp_3)
N_EQUITY_FACTORS = 160
SEED = 42
RESULTS_DIR = Path(__file__).parent.parent / "results"


# ── Data loading ─────────────────────────────────────────────────────────────

def _load_or_generate():
    """Return (option_factors, equity_factors, rf_series)."""
    bench_path = RESULTS_DIR / "exp2_benchmark_factors.parquet"
    sel_path = RESULTS_DIR / "exp3_selected_factors.json"
    op_path = RESULTS_DIR / "exp1_factor_panel.parquet"

    rng = np.random.default_rng(SEED)
    dates = pd.date_range("1996-01", periods=N_MONTHS, freq="MS")
    rf = pd.Series(0.0002 + rng.standard_normal(N_MONTHS) * 0.00005, index=dates)

    if bench_path.exists():
        equity_factors = pd.read_parquet(bench_path)
    else:
        equity_factors = pd.DataFrame(
            rng.standard_normal((N_MONTHS, N_EQUITY_FACTORS)) * 0.03,
            index=dates, columns=[f"eq_{i:03d}" for i in range(N_EQUITY_FACTORS)],
        )

    if sel_path.exists() and op_path.exists():
        with open(sel_path) as fh:
            selected = json.load(fh)["selected_factors"]
        all_op = pd.read_parquet(op_path)
        option_factors = all_op[selected].dropna(axis=1, how="all")
    else:
        n = N_OPTION_FACTORS
        data = rng.standard_normal((N_MONTHS, n)) * 0.04
        option_factors = pd.DataFrame(
            data, index=dates, columns=[f"char_{i:03d}" for i in range(n)]
        )

    return option_factors, equity_factors, rf


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("Experiment 5: Joint Tests")
    print("=" * 60)
    print()

    # ── Step 1: Load data ────────────────────────────────────────────────
    print("[1] Loading factor data …")
    option_factors, equity_factors, rf = _load_or_generate()

    # For GRS: option factors as test assets, equity factors as benchmark
    # Use a manageable equity subset to keep computation tractable
    K_bench = min(20, equity_factors.shape[1])
    eq_bench = equity_factors.iloc[:, :K_bench]

    idx = option_factors.index.intersection(eq_bench.index)
    op = option_factors.loc[idx]
    eq = eq_bench.loc[idx]
    rf_aligned = rf.reindex(idx).fillna(0.0)

    print(f"  Option factors  : {op.shape}")
    print(f"  Equity benchmark: {eq.shape}  (using first {K_bench} for GRS)")
    print(f"  Common obs      : {len(idx)}")

    # ── Step 2: GRS test ─────────────────────────────────────────────────
    print("\n[2] GRS joint alpha test …")
    try:
        grs = grs_test(test_asset_returns=op, benchmark_factor_returns=eq)
        print(f"  GRS statistic : {grs['grs_stat']:.4f}")
        print(f"  p-value       : {grs['p_value']:.4f}")
        print(f"  T={grs['T']}, N={grs['N']}, K={grs['K']}")
        n_sig_alpha = (np.abs(grs["alphas"]) > 2 * np.std(grs["alphas"])).sum()
        print(f"  Alphas >2σ    : {n_sig_alpha} / {grs['N']}")
    except Exception as e:
        grs = {"grs_stat": None, "p_value": None, "alphas": [], "T": 0, "N": 0, "K": 0}
        print(f"  GRS failed: {e}")

    # ── Step 3: Maximum Sharpe ratios ────────────────────────────────────
    print("\n[3] Maximum Sharpe ratio comparison …")

    # Use full equity factor set for Sharpe comparison
    eq_all = equity_factors.loc[idx]
    sharpe = compare_sharpe_ratios(eq_all, op, rf_series=rf_aligned)

    print(f"  SR equity-only  : {sharpe['sr_equity']:.4f}")
    print(f"  SR option-only  : {sharpe['sr_option']:.4f}")
    print(f"  SR combined     : {sharpe['sr_combined']:.4f}")
    print(f"  SR improvement  : {sharpe['sr_improvement']:+.2%}")

    # ── Step 4: CCA redundancy ────────────────────────────────────────────
    print("\n[4] CCA + Stewart-Love redundancy …")
    # Use small subsets for tractable CCA
    n_op = min(10, op.shape[1])
    n_eq = min(15, eq_all.shape[1])
    op_sub = op.iloc[:, :n_op]
    eq_sub = eq_all.iloc[:, :n_eq]

    cca_res = canonical_correlation_analysis(op_sub, eq_sub)
    rho = cca_res["canonical_correlations"]
    print(f"  Canonical correlations ({len(rho)} dimensions):")
    for i, r in enumerate(rho, 1):
        print(f"    ρ_{i} = {r:.4f}")

    redundancy = stewart_love_redundancy(X=op_sub, Y=eq_sub)
    print(f"\n  Stewart-Love Redundancy:")
    print(f"    RI(option | equity) = {redundancy['RI_options_given_equity']:.4f}")
    print(f"    RI(equity | option) = {redundancy['RI_equity_given_options']:.4f}")
    print(f"    Unique option share = {redundancy['unique_option_share']:.4f}")
    print(f"    Unique equity share = {redundancy['unique_equity_share']:.4f}")

    # Full redundancy via cca_redundancy module
    print("\n  Full redundancy analysis (cca_redundancy module):")
    full_red = full_redundancy_analysis(op_sub, eq_sub)
    print(f"    n_canonical_dims          = {full_red['n_canonical_dimensions']}")
    print(f"    RI_option_given_equity    = {full_red['RI_option_given_equity']:.4f}")
    print(f"    RI_equity_given_option    = {full_red['RI_equity_given_option']:.4f}")

    # ── Step 5: Save ──────────────────────────────────────────────────────
    print("\n[5] Saving results …")
    results = {
        "grs": {
            "grs_stat": float(grs["grs_stat"]) if grs["grs_stat"] is not None else None,
            "p_value": float(grs["p_value"]) if grs["p_value"] is not None else None,
            "T": grs["T"], "N": grs["N"], "K": grs["K"],
        },
        "sharpe": {k: round(float(v), 6) if v is not None else None
                   for k, v in sharpe.items()},
        "redundancy": {
            "RI_options_given_equity": round(redundancy["RI_options_given_equity"], 6),
            "RI_equity_given_options": round(redundancy["RI_equity_given_options"], 6),
            "unique_option_share": round(redundancy["unique_option_share"], 6),
            "unique_equity_share": round(redundancy["unique_equity_share"], 6),
        },
        "canonical_correlations": rho.tolist(),
        "timestamp": datetime.utcnow().isoformat(),
    }

    out_path = RESULTS_DIR / "exp5_joint_tests.json"
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)

    print(f"  Joint test results → {out_path}")
    print("\nExperiment 5 complete.\n")


if __name__ == "__main__":
    main()
