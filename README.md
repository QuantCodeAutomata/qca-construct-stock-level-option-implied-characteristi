# Option-Implied Characteristics & Factor Zoo

Quantitative research repository for constructing, screening, and testing
option-implied factors against the high-dimensional equity factor benchmark.

## Research Questions

1. Can stock-level option-implied characteristics generate unique pricing
   signals beyond the 160-factor equity benchmark?
2. Do screened option factors (selected via histogram-gradient-boosting + SHAP)
   survive DS-LASSO inference against a high-dimensional equity benchmark?
3. Does the group of option factors meaningfully expand the investment
   opportunity set (GRS, Sharpe, CCA redundancy)?

## Repository Structure

```
qca-option-factors/
├── data/               # Data loading utilities (Massive REST API)
│   └── data_loader.py
├── src/                # Core implementation modules
│   ├── data_contracts.py       # Column schemas & validation helpers
│   ├── rnd_estimation.py       # IV curve fitting & Breeden-Litzenberger RND
│   ├── option_characteristics.py  # 137 Appendix-B option characteristics
│   ├── equity_factors.py       # 152 Jensen + 8 French equity factors
│   ├── test_assets.py          # 3×2 independent-sort test assets
│   ├── screening.py            # HGBR + SHAP factor screening
│   ├── dslasso.py              # Double-Selection LASSO inference
│   ├── joint_tests.py          # GRS, max-Sharpe, CCA tests
│   └── cca_redundancy.py       # Stewart-Love redundancy index
├── exp/                # Experiment entry-points (call src/)
│   ├── exp_1_option_factors.py
│   ├── exp_2_equity_factors.py
│   ├── exp_3_screening.py
│   ├── exp_4_dslasso.py
│   ├── exp_5_joint_tests.py
│   └── exp_6_robustness.py
├── tests/              # Pytest unit & integration tests
├── results/            # Artifacts: parquet panels, CSVs, JSON, RESULTS.md
└── AGENTS.md           # Persistent agent memory
```

## Experiments

| ID   | Title | Key Output |
|------|-------|------------|
| exp_1 | Option-implied characteristics & 137 monthly factors | `results/exp1_*.parquet` |
| exp_2 | 160 equity benchmark factors & test assets | `results/exp2_*.parquet` |
| exp_3 | HGBR+SHAP screening → 20 representative factors | `results/exp3_*.csv/json` |
| exp_4 | DS-LASSO marginal pricing inference | `results/exp4_dslasso_results.csv` |
| exp_5 | GRS · Sharpe · CCA joint opportunity-set tests | `results/exp5_joint_tests.json` |
| exp_6 | Robustness across 7 maturities & 2 risk-free rates | `results/exp6_*.csv` |

Full quantitative results are in **[results/RESULTS.md](results/RESULTS.md)**.

## Setup

```bash
pip install -r requirements.txt
```

Set environment variable:

```bash
export MASSIVE_API_KEY="<your-key>"
```

## Running Experiments

```bash
python exp/exp_1_option_factors.py
python exp/exp_2_equity_factors.py
python exp/exp_3_screening.py
python exp/exp_4_dslasso.py
python exp/exp_5_joint_tests.py
python exp/exp_6_robustness.py
```

## Running Tests

```bash
pytest tests/ -v
```

## Key Findings (Baseline 30-day Maturity)

* **Screening** selects **20** option factors (cumulative SHAP ≈ 89.3 %).
* **DS-LASSO** finds 8/20 conventionally significant; 4 survive Bonferroni;
  BH additionally flags `RND_kurt_eps`.
* **GRS** statistic ≈ 1.25, p ≈ 0.25 — no joint rejection of zero alphas.
* **Redundancy** — equity explains ≈ 91 % of option-factor variance; option
  factors explain ≈ 35 % of equity-factor variance; unique option share ≈ 9 %.
* **Max-Sharpe** increases from ≈ 3.35 (equity-only) to ≈ 3.79 (combined).

See [results/RESULTS.md](results/RESULTS.md) for full tables.

## Data Sources

* **Option data** — CBOE / Massive REST API (single-stock options)
* **Equity characteristics** — Jensen et al. (2023) panel (153 chars; `prc` excluded)
* **Benchmark factors** — Kenneth French Data Library (FF5 + Mom + STR + LTR)
* **Risk-free rate** — FRED 3-month T-bill; robustness: TOPT van Binsbergen et al. (2023)

## References

* Jensen, T., Kelly, B., Pedersen, L. (2023). *Is There a Replication Crisis in Finance?*
* Feng, G., Giglio, S., Xiu, D. (2020). *Taming the Factor Zoo.*
* Ulrich, M. et al. (2023). *Option-Implied Risk-Neutral Densities.*
* Breeden, D., Litzenberger, R. (1978). *Prices of State-Contingent Claims.*
* Gibbons, M., Ross, S., Shanken, J. (1989). *A Test of the Efficiency of a Given Portfolio.*
* van Binsbergen, J. et al. (2023). *Option-Implied Risk-Free Rates.*
