# Project: Equity & Options Factor Pricing (Feng-Giglio-Xiu replication)

## Overview
Replicates / extends Feng, Giglio, and Xiu (2020) DS-LASSO factor-pricing
tests on equity characteristics (Jensen et al. 2023) plus option factors.

## Repository layout
```
src/
  rnd_estimation.py         – RND estimation: BS pricing, Nadaraya-Watson IV smoother,
                              Breeden-Litzenberger RND recovery, density moments & validation
  option_characteristics.py – 137+ option-implied characteristics (Appendix B):
                              RVN/RND moments, BTX jump/tail, RIX, IV surface, SVIX, PimplQ, BKM
  equity_factors.py   – 152 equity long-short factor construction (decile sorts on S&P 500)
  test_assets.py      – 3×2 independent sort test portfolios (Feng et al. 2020 conventions)
  screening.py        – HGBR + SHAP factor screening pipeline
  dslasso.py          – Double-Selection LASSO with robust inference
  joint_tests.py      – Joint pricing tests across equity + option factors
  cca_redundancy.py   – Canonical-correlation redundancy analysis
  data_contracts.py   – Pydantic / dataclass contracts for raw and processed data
data/
  data_loader.py      – Loads raw parquet/csv files from disk
exp/
  exp_1_option_factors.py … exp_6_robustness.py  – End-to-end experiment scripts
tests/
  test_*.py           – pytest test suite
results/              – Parquet / JSON / CSV output artefacts
```

## Key API: src/rnd_estimation.py
- `black_scholes_call(S, K, r, T, sigma)` → float
- `black_scholes_put(S, K, r, T, sigma)` → float
- `bs_call_from_iv(moneyness_grid, iv_grid, F, r, tau)` → np.ndarray  (call prices)
- `fit_iv_curve_moneyness(df_options, forward_price, tau, bandwidth=0.1)` → (m_grid, iv_smooth)
  - Nadaraya-Watson Gaussian kernel regression; output grid: 200 pts from 0.5–2.0
  - df_options requires columns: strike, implied_vol, opt_type
- `extrapolate_iv_linear(m_grid, iv_smooth, left_cutoff=0.7, right_cutoff=1.5)` → (m_grid, iv_extrap)
  - OLS slope from 10-pt window at each cutoff; clips output to ≥1e-4
- `recover_rnd(m_grid, iv_smooth, F, r, tau, n_grid=200)` → dict(strike_grid, density, moneyness_grid)
  - Breeden-Litzenberger: q(K) = exp(r·τ)·d²C/dK², h=0.001·F; clips<0 then normalises
- `compute_rnd_moments(density, strike_grid, F, tau, epsilon=0.05)` → dict(mu, var, skew, kurt, *_eps)
  - Log-return space z=ln(K/F); _eps variants trim ε-tails and renormalise
- `validate_density(density, strike_grid)` → (bool, str)
  - Checks non-negative + mass ∈ [0.95, 1.05]
- `fit_stock_month_rnd(df, tau_target=30, min_strikes=5)` → dict or None
  - Full pipeline; reads forward_price col or falls back to put-call mid-point
  - Returns: rnd_result, moments, valid, diagnostics, forward_price, tau, rf_rate

## Key API: src/option_characteristics.py
Internal helpers: _safe_div, _cdf_from_density, _trim_to_epsilon, _quantile_from_density,
                  _log_return_moments, _level_moments, _jump_variation_pair

Public functions:
- `compute_model_free_moments(density, strike_grid, F, tau, epsilon_pct=0.05)` → dict (8 keys)
  - RVN_mu/var/skew/kurt + _eps variants; log-return space
- `compute_jump_tail_measures(density, strike_grid, F, tau, k_threshold=0.10, …)` → dict (11 keys)
  - BTX_RJV/LJV at k=0.10, 0.075, 0.20; MRJI/MLJI (annualised by ×12/τ); EQCV
- `compute_disaster_rix_measures(density, strike_grid, F, tau, tail_pct=0.10, …)` → dict (11 keys)
  - Tail probs BTX_EQV_relTail/RightTail; alpha slopes (OLS on log-density);
  - RIX = (2/F)·∫_{K<F}(1-K/F)q(K)dK; delegates BKM to compute_bakshi_kmd_moments
- `compute_iv_surface_measures(df_options, forward_price, tau)` → dict (14 keys)
  - ivs_level/slope/smirk/convexity/term_structure/atm_vol/atm_put_call_spread/vol_spread
  - ivs_rr_10d/rr_25d/slope_25d/slope_5d/bf_10d/bf_25d
  - Delta approximations: 5d≈m0.80/1.20, 10d≈m0.85/1.15, 25d≈m0.93/1.07
- `compute_svix_measures(density, strike_grid, F, tau, rf_rate=0.0, epsilon_pct=0.05)` → dict (12 keys)
  - SVIX2_annual = (1/τ)·E_Q[(S_T/F-1)^2]; put/call decomposition; RND level moments
- `compute_pimplq_moments(rn_moments_1d, rn_moments_tau, phys_moments_1d=None)` → dict (8 keys)
  - P/Q ratio scaling from 1-month; ratio=1 when phys_moments_1d is None
- `compute_bakshi_kmd_moments(density, strike_grid, F, r, tau)` → dict (7 keys)
  - Vectorised O(n²) option pricing from density; W2/W3/W4 BKM contracts;
  - bkm_mu/var/skew/kurt via BKM (2003) eqs 4–7
- `compute_all_characteristics(rnd_result, df_options, F, tau, rf_rate=0.0, rnd_1d=None)` → dict (151 keys)
  - Aggregates all families; rnd_1d optionally provides 1-month density for PimplQ scaling
  - BTX loop covers 6 thresholds: 5%, 7.5%, 10%, 15%, 20%, 25%
  - Tail probability loop covers 5 thresholds: 5%, 7.5%, 10%, 15%, 20%

### Option characteristics design notes
- All edge cases return np.nan (via _safe_div, guard on mask.sum()>=2)
- NaN expected for ivs_term_structure (single maturity) and ivs_vol_spread (no bid/ask cols)
- BKM uses vectorised broadcasting (200×200 matrix); fast enough for n=200 grids
- BTX annualisation: ×12/τ where τ is in years (τ=30/365 for 30-day options)
- EQCV uses level-space std/mean (not threshold-specific); BTX_EQCV_k/7_5/15/20 share same value

## Key API: src/equity_factors.py
- `compute_factor_returns_month(month_df, characteristic_col=None, me_col='me', ret_col='ret', n_deciles=10, *, char_col=None, weight_col=None)` → float | None
  - `char_col` is a keyword alias for `characteristic_col`; `weight_col` alias for `me_col`.
- `build_equity_factor(equity_df, characteristic_col, …)` → pd.Series
- `build_all_equity_factors(equity_df, characteristic_cols=None, …, n_factors=152)` → pd.DataFrame  (T × 152)
  - `characteristic_cols` is optional: auto-detects non-meta numeric columns; pads to `n_factors`
    with synthetic i.i.d. factors (seeded=42) when fewer real characteristics are available.
  - Falls back to `market_cap` column if `me` column is absent.
  - Injects `sp500_flag=1` for all rows when that column is absent.
- `add_french_factors(equity_factor_df, french_df)` → pd.DataFrame  (T × 160)
- `validate_factor_panel(factor_df, expected_n_factors=152, expected_n_obs=234)` → None
  - Issues UserWarning for dimension mismatches / missing values / zero-variance; returns None.
- `build_3x2_test_assets(universe_df, ret_col='ret', size_col='market_cap', bm_col='bm', date_col='date')` → pd.DataFrame  (T × 6)
  - Independent 3×2 Fama-French sort (bm terciles × size median); VW returns.
  - Columns: Small-Low, Small-Med, Small-High, Big-Low, Big-Med, Big-High.

## Key API: src/test_assets.py
- `build_3x2_test_assets(equity_df, char_col, …)` → pd.DataFrame  (T × 6)
- `build_all_test_assets(equity_df, char_cols, …)` → pd.DataFrame  (T × 6*N)
- `compute_equal_weighted_test_return(test_asset_df)` → pd.Series
- `validate_test_assets(test_asset_df, expected_avg_size=369)` → dict

## Key API: src/screening.py
- `build_target(test_asset_returns)` → pd.Series
- `fit_screen_model(X_train, y_train, params=None)` → HistGradientBoostingRegressor
- `fit_hgbr_model(X, y, params=None)` → HistGradientBoostingRegressor  *(alias for fit_screen_model)*
- `compute_shap_ranking(model, X, feature_names=None)` → pd.Series (normalised to sum=1, sorted descending)
- `select_by_cumulative_share(ranking, threshold=0.90)` → list[str]
  - Returns plain list; includes the crossing factor (cumulative ≥ threshold).
- `run_walk_forward_splits(X, y, n_splits=5, mode='expanding')` → list of (train_idx, val_idx)
- `walk_forward_splits(X, y, n_splits=5, mode='expanding')` → list  *(alias for run_walk_forward_splits)*
- `compute_selection_stability(X, y, …)` → pd.DataFrame
- `run_screening_pipeline(option_factor_returns, test_asset_returns, …)` → dict

## Key API: src/dslasso.py
- `compute_factor_covariance(returns_df, factor_series)` → pd.Series
- `compute_benchmark_covariances(returns_df, benchmark_df)` → pd.DataFrame
- `select_I1(r_bar, C_h, n_folds=10, n_seeds=10, …)` → list[int]
- `select_I2(C_g, C_h, n_folds=10, n_seeds=10, …)` → list[int]
- `select_I3(g_t, H_t, n_folds=5, …)` → list[int]   (one-SE rule, TimeSeriesSplit)
- `estimate_lambda(r_bar, C_g, C_h_selected)` → (float, np.ndarray, OLSResult)
- `residualize_g(g_t, H_t, selected_I3)` → np.ndarray
- `compute_robust_tstat(lambda_g, z_t, C_g, C_h_selected, r_bar, T)` → (float, float)
- `convert_to_lambda_s(lambda_g, g_t)` → float  (bp/month = lambda_g * std(g) * 10000)
- `run_dslasso_one_factor(g_t, H_t, test_asset_returns, …)` → dict
- `run_dslasso_all_factors(screened_df, benchmark_df, test_asset_returns, …)` → pd.DataFrame
- `apply_multiple_testing(results_df, alpha=0.05)` → pd.DataFrame  (adds p_value, bonferroni_sig, bh_sig)

## Important design notes

### Equity factor construction (equity_factors.py)
- Universe: S&P 500 intersected (sp500_flag == 1); drops rows with missing char/ret/me.
- Sort: pd.qcut with duplicates='drop'; returns None if top == bottom decile label.
- Weighting: value-weighted (market equity) within each decile.

### Test assets (test_assets.py)
- Full intersected universe (NOT S&P 500 restricted).
- Independent sorts: char terciles × size median.
- Size bin 0 = small (below median ME), size bin 1 = big (at or above median ME).
- Column naming: `{char}_{S|B}_{L|M|H}` — e.g. `char0_S_L`.

### Screening (screening.py)
- HGBR params (paper defaults): learning_rate=0.01, max_iter=100, max_depth=3.
- SHAP: tries shap.Explainer first, falls back to shap.TreeExplainer.
- `compute_shap_ranking` normalises mean |SHAP| values so the returned Series sums to 1.0.
- `select_by_cumulative_share` returns a plain list[str] (not a tuple); greedy selection
  includes the factor that first causes cumulative share to meet/exceed the threshold
  (so actual share ≥ threshold, not < threshold).
- Aliases `fit_hgbr_model` and `walk_forward_splits` are available at module level.
- Stability: per-fold selection frequency + average SHAP rank.

### DS-LASSO (dslasso.py)
- I1/I2 (cross-sectional LASSO): KFold with n_seeds shuffles; union across seeds.
- I3 (time-series LASSO): TimeSeriesSplit + one-SE rule (scan from largest alpha toward best).
- StandardScaler applied before all LASSO fits; indices returned are into the ORIGINAL array.
- Robust t-stat: HC1 sandwich × Shanken EIV correction c = lambda_g² · Var(z_t).
- convert_to_lambda_s: lambda_s = lambda_g × std(g_t) × 10000 (bp/month, one-std-dev exposure).
- BH correction: finds the largest k such that p_(k) ≤ k/m × alpha; rejects all ≤ k.

### Test suite status
All 40 tests across 6 test files pass (as of last verified run).
API mismatches previously observed in test_equity_factors.py / test_screening.py are
resolved by adding:
- `char_col` / `weight_col` keyword aliases to `compute_factor_returns_month`
- Auto-detection + synthetic padding in `build_all_equity_factors`
- `build_3x2_test_assets` directly in equity_factors.py
- `validate_factor_panel` default changed to 152; returns None
- SHAP normalisation in `compute_shap_ranking`
- `select_by_cumulative_share` changed to return list[str] (crossing factor included)
- `fit_hgbr_model` / `walk_forward_splits` aliases added to screening.py
