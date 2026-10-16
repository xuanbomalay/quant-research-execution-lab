# Research report: a common-account quantitative research lab

## Question and contribution

Can a transparent combination of trend, accounting-based factors, filing text and supervised predictions improve a constrained portfolio relative to simple allocation, after a single common transaction-cost calculation? A separate experiment asks whether a PPO allocator adds anything within the same feasible sleeve space. Execution and derivative experiments audit mechanisms rather than being blended into the stock account. The contribution shown here is reproducible engineering and explicit falsification criteria, not a new claim of alpha.

## Data and availability

The stock exercise uses 12 predefined surviving US issuers over 2014–2022. Company facts retain accession, report end and the filed-date conservative availability bound. Latest publicly available share counts are mechanically adjusted for splits to estimate market capitalization; stale shares, taxonomy changes and restatements remain limitations. Price history is vendor-adjusted synthetic-unit accounting. Text events use actual issuer-linked filing PDFs and SEC metadata, conservatively scheduled after filed-day end plus processing delay. Exact acceptance fields are preserved. The current dictionary source is official 1993-2025 workbook. Single-company text and revised mirrors are not institutional PIT coverage.

## Validation design

Daily asset rows stay together. Forecast labels run from next open to the sixth subsequent open, and training labels must mature before the split. Nested expanding yearly folds in 2019–2022 select among Ridge alpha 1/10/100 only in the previous inner period; a five-session embargo is added. All 104 known trials, including seeds, ablations and costs, are retained as a lower bound. Outer-fold diagnostics are not a newly unobserved final holdout: later history had already been displayed or used in development. A forward evaluation remains outstanding.

## Accounting and constraints

Trend/factor/text/ML/RL target weights are combined on the same instruments before applying risk constraints and one self-financing ledger. Risk-asset budget is at most 0.8; per-stock cap 0.15, study-industry cap 0.35 and target turnover cap 0.25, with any compulsory risk override logged. Costs use 3bp commission and 2bp slippage on each traded leg. Borrow and funding are modeled in the separate long-short demonstration. Adjusted units, zero cash yield, no taxes and no endogenous market impact are explicit assumptions.

## Main observed development results

Fractions denote returns/drawdowns; costs are cash units. The table preserves every ablation and the zero/10bp cost sensitivity. Removed sleeves are not renormalized, so exposure changes are visible. Risk-matched comparisons use a past-validation scale frozen before the development diagnostic; future realized risk need not match exactly.

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

## Statistical evidence and rejection

Paired circular block bootstrap uses 5/21/63-session blocks on common dates. These intervals are descriptive under development reuse and cannot establish independent replication. DSR is shown for the known-trial lower bound and multiples thereof using daily Sharpe, skewness and non-excess kurtosis. The independent-trial/iid-return approximation is materially imperfect because candidates correlate and returns depend serially. No result is relabeled significant by selecting a favorable assumption.

| index | hypothesis | delta_net_return | observed_result | reject_reason_or_limit |
| --- | --- | --- | --- | --- |
| 0 | full improves net return vs no_text | 7.96733e-05 | positive_descriptive_difference_only | development data already observed; no independent final holdout; exposure and costs differ |
| 1 | full improves net return vs no_ML | 0.0468189 | positive_descriptive_difference_only | development data already observed; no independent final holdout; exposure and costs differ |
| 2 | full improves net return vs no_RL | 0.0193869 | positive_descriptive_difference_only | development data already observed; no independent final holdout; exposure and costs differ |
| 3 | full improves net return vs no_state_filter | -0.0339461 | not_supported_in_this_window | development data already observed; no independent final holdout; exposure and costs differ |
| 4 | full improves net return vs equal_reference | -0.105382 | not_supported_in_this_window | development data already observed; no independent final holdout; exposure and costs differ |

## Reinforcement learning audit

Five SB3 CPU seeds and ten actual JAX TPU runs (five per reward variant), 8192 steps per run, use the same cost-constrained transition. Normalization is fitted only in training; future returns are absent from the state. Artificial episode ends bootstrap, true data ends do not. Five-seed median validation Sharpe selected SB3_CPU_lambda_0.05. Reports retain all seeds, average gross exposure and underperformance counts; training success is not economic success. Checkpoints are shipped for inspection, not as a production trading policy.

## Independent execution and option reports

The official LOBSTER endpoint was unavailable. Source kind is SYNTHETIC_MECHANISM_ONLY; queue and latency results are explicitly mechanism-only synthetic tests. Touch is not a fill, cancellations do not optimistically advance the queue, and incomplete execution is retained. This is not evidence of live HFT capability or a basis for transplanting intraday costs into daily stocks.

BSM prices and Greeks match QuantLib and finite differences. CRR refinement and 131072 actual TPU Monte Carlo terminal paths independently check pricing. Discrete delta hedges use 512 common risk-neutral GBM paths and a named -20% jump stress, explicit cash interest, fees and terminal settlement. PnL is per option in dollars, not a return on zero initial capital. Cboe VIX and SPX are compared on the same 30-calendar-day, 365-day annualization basis as an ex-post variance-premium proxy. No real historical option-chain backtest is claimed.

## Reproduction and next falsification

Run the TPU Colab master after dependency installation and a restart. Source hash: f665bf8a293d3359b1d9ba061a2a9971c2e4632e1f0b3db304bae7fd90c9fe1d. There are 699 recorded lesson checks and 12 offline packaged-module tests. All raw market/filing/lexicon data are excluded from redistribution. Next, define one independent hypothesis, freeze the design and collect new forward observations or expand to an appropriately licensed point-in-time universe. Until then this is an AI-assisted teaching/research-engineering portfolio, not independent original alpha or live investment performance.
