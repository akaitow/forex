# FX Currency Strength: Momentum or Mean Reversion?

A reproducible Python case study testing whether the four strongest major currencies on one trading day continue to outperform on the next.

## Executive finding

Across 1,298 next-day observations, the prior day’s top-four currencies produced an average next-day strength score of **-0.0086%**. The 95% moving-block bootstrap confidence interval was **[-0.0179%, +0.0003%]**. The estimated $p_0 \rightarrow p_1$ slope was **-0.141** with a Newey–West $p$-value of **0.001**.

**Conclusion:** the sample rejects a momentum interpretation and shows a modest one-day mean-reversion relationship: continuation occurred on **46.6%** of days, and the $p_0 \rightarrow p_1$ slope was statistically negative. The effect is small, not directly investable, and not evidence of a profitable strategy.

## What the project demonstrates

- Constructs all 28 unique pairs from eight major currencies.
- Converts pair-level log returns into an equal-weighted currency-strength index.
- Separates a zero-sum accounting identity from an empirical predictability test.
- Uses a moving-block bootstrap, sign test, and Newey–West regression inference.
- Audits triangular cross-rate coherence and repeats the test with only seven USD legs.
- Communicates the result through distribution, relationship, stability, and robustness visuals.

## Repository contents

- `fx_currency_strength_case_study.ipynb` — executed, reader-facing analysis.
- `data/fx_prices_2021-08-01_2026-08-01.csv` — frozen Yahoo Finance input snapshot.
- `requirements.txt` — Python dependencies.
- `scripts/build_case_study.py` — deterministic notebook builder.

## Run locally

```bash
python -m pip install -r requirements.txt
jupyter nbconvert --execute --to notebook --inplace fx_currency_strength_case_study.ipynb
```

The notebook reads the frozen CSV first. If it is absent, it downloads the configured range through `yfinance` and recreates the cache.

## Method in one line

`28 FX closes` → `daily log returns` → `8 strength scores` → `top four on day t` → `same currencies on day t+1`

## Caveats

The index is not directly investable; transaction costs are omitted; Yahoo Finance closes may not reflect a common institutional fixing time; and non-significance does not establish market efficiency or universal independence.

## Data source

Yahoo Finance data accessed through [`yfinance`](https://ranaroussi.github.io/yfinance/). For research and portfolio demonstration only—not investment advice.
