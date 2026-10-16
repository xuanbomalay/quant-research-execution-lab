# Quant Research and Execution Lab

A reproducible teaching and research-engineering portfolio: temporal data, strategy accounting, risk, NLP, machine learning, execution simulation, RL and derivatives.

## Verified evidence

- Ten teaching notebooks and a self-contained 00_run_all.ipynb; no placeholder lessons.
- 699 recorded checks across lessons 2–10 and 12 offline tests of the packaged reusable modules.
- Twelve fixed surviving issuers; 27,192 daily PIT feature rows; 33 filing events in the current text experiment.
- Five SB3 CPU PPO seeds and ten JAX TPU runs, 8,192 steps each; 15 exported checkpoints. TPU placement is verified using actual device arrays.
- Nested expanding-year evaluation (2019–2022): three inner alpha candidates, maturity purge and five-session embargo. 104 known trials retained as a lower bound.
- QuantLib and finite-difference cross-checks; 131,072 antithetic TPU MC terminal paths and 512 common discrete-hedging paths with named jump stress.

## Reproduce

Open 00_run_all.ipynb in Colab, select TPU, install pinned dependencies, restart once after core upgrades, then run top to bottom. JAX/libtpu must match the TPU runtime image. Statistics and SB3 use host CPU. Runtime metadata and source hashes are included. Full reruns fetch provider data and can take several minutes.

Offline: install NumPy, pandas, SciPy and scikit-learn, then run 'pip install -e . --no-deps' and 'python -m unittest discover -s tests -v'. GitHub Actions runs these offline component tests; it does not claim to retrain policies.

## Common-account development results

Returns are fractions; fees are cash units on initial capital 10,000. Daily Sharpe uses zero cash/risk-free rate. These already-observed diagnostics are not a pristine final test or live performance.

| model | net_return | annualized | max_drawdown | Sharpe_0rf | avg_gross | fees |
| --- | --- | --- | --- | --- | --- | --- |
| full | 0.117936 | 0.0575597 | -0.0573907 | 0.875269 | 0.41219 | 98.1002 |
| no_state_filter | 0.151882 | 0.0735599 | -0.116993 | 0.646407 | 0.681229 | 158.978 |
| no_text | 0.117856 | 0.0575218 | -0.0575347 | 0.873664 | 0.412311 | 98.0673 |
| no_ML | 0.0711169 | 0.0350894 | -0.0464232 | 0.791099 | 0.31231 | 60.7002 |
| no_RL | 0.0985489 | 0.0483131 | -0.0518131 | 0.837234 | 0.33133 | 85.5231 |
| equal_reference | 0.223318 | 0.106481 | -0.14293 | 0.859461 | 0.8 | 29.1705 |
| no_text_riskmatched | 0.117856 | 0.0575218 | -0.0575347 | 0.873664 | 0.412311 | 98.0673 |
| no_ML_riskmatched | 0.0711169 | 0.0350894 | -0.0464232 | 0.791099 | 0.31231 | 60.7002 |
| no_RL_riskmatched | 0.0985489 | 0.0483131 | -0.0518131 | 0.837234 | 0.33133 | 85.5231 |
| no_state_filter_riskmatched | 0.0820997 | 0.0404037 | -0.0617591 | 0.654437 | 0.352876 | 66.5467 |
| full_zero_cost | 0.127217 | 0.0619581 | -0.0559086 | 0.937754 | 0.412187 | 0 |
| full_10bp | 0.108731 | 0.0531794 | -0.0588705 | 0.812748 | 0.412192 | 192.812 |

## Material limitations

Survivor universe and coarse static industries; revised SEC/vendor data; stale disclosed shares; single-issuer text. Actual dictionary source: official 1993-2025 workbook. Microstructure source: **SYNTHETIC_MECHANISM_ONLY**; this validates mechanisms, not live HFT alpha. Option hedging is simulated and VIX is not tradable. Historical 2023–2026 had previously been displayed, and 2021–2022 reused in development; no unseen holdout is retroactively invented. Small PPO budgets and narrow actions do not establish profitability.

See the research report, course review, experiment ledger and negative-result table. Raw market prices, filings and lexicons are excluded; provider terms still apply. This AI-assisted teaching project evidences implemented research engineering, not independent original alpha or a hiring guarantee.
