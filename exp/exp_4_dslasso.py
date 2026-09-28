"""
Experiment 4: DS-LASSO inference for screened option factors.

Double-selection LASSO (Belloni, Chernozhukov, Hansen 2014) provides
post-selection valid inference on the risk premium (lambda) of each
screened factor against a high-dimensional equity benchmark.

Pipeline:
1. Load screened option factors (exp_3) and benchmark equity factors (exp_2),
   or synthesise proxies.
2. For each screened factor, run DS-LASSO:
   a. LASSO of factor on all benchmark controls  → I_1
   b. LASSO of each test asset on all controls   → I_2
   c. Post-selection OLS with union I_1 ∪ I_2
3. Apply Benjamini-Hochberg and Bonferroni multiple-testing corrections.
4. Print results table: factor, lambda_s (bp/month), t-stat, BH flag.
5. Save to results/.
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
    from src.dslasso import (
        compute_covariance,
        select_I1,
        run_ds_lasso,
        apply_multiple_testing,
    )
    _HAS_DS = True
except ImportError:
    _HAS_DS = False

from sklearn.linear_model import LassoCV
import statsmodels.api as sm
from scipy.stats import t as t_dist

warnings.filterwarnings("ignore")

# ── Constants ────────────────────────────────────────────────────────────────
N_MONTHS = 234
LASSO_CV_FOLDS = 3          # 5 → 3 folds for speed
N_TEST_ASSETS = 6
N_MAX_FACTORS = 20         # cap screened factors to keep exp tractable
BH_ALPHA = 0.05
SEED = 42
RESULTS_DIR = Path(__file__).parent.parent / "results"


# ── Data loading ─────────────────────────────────────────────────────────────

def _load_or_generate():
    """Return (screened_factors, benchmark_factors, test_assets)."""
    bench_path = RESULTS_DIR / "exp2_benchmark_factors.parquet"
    ta_path = RESULTS_DIR / "exp2_test_assets.parquet"
    sel_path = RESULTS_DIR / "exp3_selected_factors.json"
    op_path = RESULTS_DIR / "exp1_factor_panel.parquet"

    rng = np.random.default_rng(SEED)
    dates = pd.date_range("1996-01", periods=N_MONTHS, freq="MS")

    if bench_path.exists() and ta_path.exists():
        benchmark = pd.read_parquet(bench_path)
        test_assets = pd.read_parquet(ta_path)
    else:
        benchmark = pd.DataFrame(
            rng.standard_normal((N_MONTHS, 160)) * 0.03,
            index=dates,
            columns=[f"eq_{i:03d}" for i in range(160)],
        )
        test_assets = pd.DataFrame(
            rng.standard_normal((N_MONTHS, N_TEST_ASSETS)) * 0.04,
            index=dates,
            columns=["Small_Low", "Small_Med", "Small_High",
                     "Big_Low", "Big_Med", "Big_High"],
        )

    if sel_path.exists() and op_path.exists():
        with open(sel_path) as fh:
            selected = json.load(fh)["selected_factors"]
        all_op = pd.read_parquet(op_path)
        screened = all_op[selected].dropna(axis=1, how="all")
    else:
        # Synthetic: 15 screened option factors with varied signal
        latent = rng.standard_normal((N_MONTHS, 3))
        n_screened = 15
        data = rng.standard_normal((N_MONTHS, n_screened)) * 0.04
        for i in range(min(5, n_screened)):
            data[:, i] += latent[:, i % 3] * 0.015
        screened = pd.DataFrame(
            data, index=dates,
            columns=[f"char_{i:03d}" for i in range(n_screened)],
        )

    return screened, benchmark, test_assets


# ── DS-LASSO implementation ───────────────────────────────────────────────────

def _residualise(y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Partial out linear projection of y on Z columns."""
    res = sm.OLS(y, sm.add_constant(Z)).fit()
    return res.resid


def _lasso_select(y: np.ndarray, X: np.ndarray, cv: int = LASSO_CV_FOLDS):
    """LassoCV feature selection; return bool mask of selected columns."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lasso = LassoCV(cv=cv, max_iter=10_000, n_jobs=1,
                        random_state=SEED).fit(X, y)
    return lasso.coef_ != 0.0


def _ds_lasso_single(
    factor_ret: np.ndarray,
    benchmark: np.ndarray,
    test_assets: np.ndarray,
    precomputed_mask2: np.ndarray | None = None,
) -> dict:
    """DS-LASSO for a single screened factor.

    Parameters
    ----------
    precomputed_mask2 : optional boolean mask for I_2 (shared across factors).

    Returns dict: lambda_s (bp), t_stat, p_value, n_selected_controls.
    """
    T, K = benchmark.shape
    N = test_assets.shape[1]

    # Step 1: LASSO of factor on benchmark (select I_1)
    mask1 = _lasso_select(factor_ret, benchmark)
    I1 = np.where(mask1)[0]

    # Step 2: LASSO of each test asset on benchmark (select I_2) — reuse if cached
    if precomputed_mask2 is not None:
        mask2 = precomputed_mask2
    else:
        mask2 = np.zeros(K, dtype=bool)
        for j in range(N):
            mask2 |= _lasso_select(test_assets[:, j], benchmark)
    I2 = np.where(mask2)[0]

    # Union
    I_union = np.union1d(I1, I2)

    # Step 3: Post-selection OLS — project test-asset mean on factor + controls
    ta_mean = test_assets.mean(axis=1)
    controls = benchmark[:, I_union] if len(I_union) > 0 else np.ones((T, 1))
    X_ols = np.column_stack([factor_ret, controls])
    res = sm.OLS(ta_mean, sm.add_constant(X_ols)).fit()

    # lambda_s is the coefficient on factor_ret
    lambda_s = res.params[1]
    t_stat = res.tvalues[1]
    p_value = res.pvalues[1]

    return {
        "lambda_s_raw": float(lambda_s),
        "lambda_s_bp": float(lambda_s * 10_000),   # annualised bp/month proxy
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "n_controls_I1": int(len(I1)),
        "n_controls_I2": int(len(I2)),
        "n_controls_union": int(len(I_union)),
    }


def _bh_correction(p_values: np.ndarray, alpha: float = BH_ALPHA) -> np.ndarray:
    """Benjamini-Hochberg FDR correction. Returns boolean significant mask."""
    m = len(p_values)
    ranks = np.argsort(p_values)
    bh_threshold = (ranks + 1) / m * alpha
    sorted_p = p_values[ranks]
    below = sorted_p <= bh_threshold
    # all ranks up to last True are significant
    if below.any():
        cutoff = np.where(below)[0][-1]
        sig_mask = np.zeros(m, dtype=bool)
        sig_mask[ranks[:cutoff + 1]] = True
    else:
        sig_mask = np.zeros(m, dtype=bool)
    return sig_mask


def _bonferroni_correction(p_values: np.ndarray, alpha: float = BH_ALPHA) -> np.ndarray:
    return p_values <= (alpha / len(p_values))


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("Experiment 4: DS-LASSO Inference")
    print("=" * 60)
    print(f"  DS module available : {_HAS_DS}")
    print(f"  LASSO CV folds      : {LASSO_CV_FOLDS}")
    print(f"  BH alpha            : {BH_ALPHA}")
    print()

    # ── Step 1: Load data ────────────────────────────────────────────────
    print("[1] Loading screened factors, benchmark, and test assets …")
    screened, benchmark, test_assets = _load_or_generate()

    idx = screened.index.intersection(benchmark.index).intersection(test_assets.index)
    screened = screened.loc[idx]
    benchmark = benchmark.loc[idx].fillna(0.0)
    test_assets = test_assets.loc[idx].fillna(0.0)

    F_mat = benchmark.values
    TA_mat = test_assets.values

    print(f"  Screened factors : {screened.shape}")
    print(f"  Benchmark        : {benchmark.shape}")
    print(f"  Test assets      : {test_assets.shape}")

    # Limit screened factors to keep experiment tractable
    factor_cols = list(screened.columns[:N_MAX_FACTORS])
    screened = screened[factor_cols]

    # ── Steps 2-3: DS-LASSO per factor ───────────────────────────────────
    print(f"\n[2/3] Running DS-LASSO for {len(factor_cols)} screened factors …")

    # Pre-compute I_2 mask (test-asset selection) once — shared across all factors
    print("  Pre-computing I₂ (LASSO of test assets on benchmark) …")
    K = F_mat.shape[1]
    mask2_shared = np.zeros(K, dtype=bool)
    for j in range(TA_mat.shape[1]):
        mask2_shared |= _lasso_select(TA_mat[:, j], F_mat)
    print(f"  I₂ selected {mask2_shared.sum()} benchmark columns")

    results = []
    for col in factor_cols:
        f_ret = screened[col].fillna(0.0).values
        if _HAS_DS:
            row = run_ds_lasso(f_ret, F_mat, TA_mat)
            # Normalise key: src.dslasso returns 'lambda_s' already in bp/month
            row["lambda_s_bp"] = row.pop("lambda_s", row.get("lambda_s_bp", float("nan")))
            row["factor"] = col
        else:
            row = _ds_lasso_single(f_ret, F_mat, TA_mat, precomputed_mask2=mask2_shared)
            row["factor"] = col
        results.append(row)
        print(f"  {col:15s}: λ={row['lambda_s_bp']:+7.2f} bp  t={row['t_stat']:+5.2f}  p={row['p_value']:.3f}")

    results_df = pd.DataFrame(results).set_index("factor")

    # ── Step 4: Multiple testing corrections ──────────────────────────────
    print("\n[4] Applying multiple-testing corrections …")
    p_vals = results_df["p_value"].values

    results_df["bh_significant"] = _bh_correction(p_vals)
    results_df["bonferroni_significant"] = _bonferroni_correction(p_vals)

    n_bh = results_df["bh_significant"].sum()
    n_bf = results_df["bonferroni_significant"].sum()
    print(f"  BH significant (q≤{BH_ALPHA})        : {n_bh} / {len(results_df)}")
    print(f"  Bonferroni significant (α/{len(results_df)}): {n_bf} / {len(results_df)}")

    # ── Print results table ───────────────────────────────────────────────
    print("\n  Full results table:")
    print(f"  {'Factor':15s}  {'λ_s (bp)':>10s}  {'t-stat':>8s}  {'p-val':>7s}  {'BH':>4s}  {'Bonf':>6s}")
    print("  " + "-" * 65)
    for factor, row in results_df.iterrows():
        bh_star = "✓" if row["bh_significant"] else " "
        bf_star = "✓" if row["bonferroni_significant"] else " "
        print(f"  {factor:15s}  {row['lambda_s_bp']:+10.2f}  {row['t_stat']:+8.2f}  "
              f"{row['p_value']:7.4f}  {bh_star:>4s}  {bf_star:>6s}")

    # ── Save ──────────────────────────────────────────────────────────────
    print("\n[5] Saving results …")
    out_path = RESULTS_DIR / "exp4_dslasso_results.csv"
    results_df.to_csv(out_path)

    meta = {
        "n_screened_factors": int(len(results_df)),
        "n_bh_significant": int(n_bh),
        "n_bonferroni_significant": int(n_bf),
        "bh_alpha": BH_ALPHA,
        "timestamp": datetime.utcnow().isoformat(),
    }
    with open(RESULTS_DIR / "exp4_meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"  DS-LASSO results → {out_path}")
    print("\nExperiment 4 complete.\n")


if __name__ == "__main__":
    main()
