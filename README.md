# Machine-Learning-Alpha-Models
Developed a regime-based quantitative trading strategy for U.S. equity index futures. Enhanced trade selection using XGBoost. Improved 4-year out-of-sample portfolio Sharpe ratio from 0.92 to 1.37 through walk-forward validated machine learning filter.

# Phase 2

## Abstract

This study extends the prior rule-based alpha framework (Phase 1) by integrating XGBoost as a machine learning (ML) entry filter on v1 and v4 models. We verify whether ML-based filtering can statistically improve the out-of-sample (OOS) Sharpe ratio of the rule-based variants on U.S. equity index futures.

**Main Results:**

- 50/50 capital-split portfolio Walk-Forward 4-year Sharpe improved from **0.917** (rules only) to **1.365** (full ML)
- v1 + XGBoost: Sharpe 0.750 → 1.060 (+41%), MDD -48.2% → -31.1% (16.5%p improvement)
- v4 + XGBoost: Sharpe 0.694 → 0.985 (+42%), MDD comparable
- Robustness verified across 4 checks (hyperparameter, random seed, leave-one-out, random label)
- **Authenticity score: 0.81** (81% of improvement attributed to genuine ML learning)

**Verified Methodology:**

- Walk-forward cross-validation with yearly re-training 
- Separate training models (v1 ML, v4 ML) to preserve distinct entry mechanisms
- Fixed hyperparameters without tuning (overfitting prevention)

---

## 1. Introduction

### 1.1 Motivation

Phase 1 established OOS Sharpe 0.96+ on U.S. equity index futures using rule-based v1 and v4 models with Variance Ratio (VR) regime classification. This Phase 2 investigates the following research question:

> Can ML entry filtering provide statistically significant alpha enhancement on top of validated rule-based models, without compromising the robustness properties demonstrated in Phase 1?

### 1.2 Approach

We treat the ML filter as a **binary classifier on trade-level outcomes**. Each rule-generated entry signal is evaluated by an XGBoost model trained on historical trade outcomes; only entries with predicted profit probability above a threshold are executed.

This design preserves the rule-based logic as the signal generator and uses ML as a selectivity layer.
---

## 2. Dataset

| Item | Value |
|------|-------|
| Source | Rule-based v1 and v4 trades from Phase 1 |
| Markets | ES, NQ, YM, RTY |
| Total Trades | 2,402 (v1: 1,302; v4: 1,100) |
| Features per Entry | 30 |
| In-Sample (IS) | 2018-2023 (1,687 trades) |
| Out-of-Sample (OOS) | 2024-2026 (715 trades) |
| Label | y = 1 if trade_return > 0 else 0 |

### 2.1 Feature Categories

- **VR Regime**: vr_16, vr_30, short_vr, long_vr, vr_score, regime5 one-hot encoded
- **Volatility**: atr, atr_norm, bbw, bbw_percentile, is_expansion, walk_up/down, rolling_std_20, rolling_std_100
- **Price Position & Momentum**: bb_position, dist_ma_norm, mom_5_norm, mom_10_norm, mom_20_norm
- **Time**: hour_of_day, day_of_week
- **Categorical**: product (one-hot), entry_type (one-hot)

All features are captured at the moment of entry, ensuring no look-ahead bias.

---

## 3. Methodology

### 3.1 ML Model Specification

XGBoost gradient boosting classifier with fixed hyperparameters:

| Parameter | Value |
|-----------|-------|
| n_estimators | 200 |
| max_depth | 4 |
| learning_rate | 0.05 |
| subsample | 0.8 |
| colsample_bytree | 0.8 |
| Threshold | 0.60 (entry accepted when P(profit) ≥ 0.60) |

Feature normalization: StandardScaler fitted on training data.

### 3.2 Separate Training Architecture

v1 and v4 trades are trained as **independent models**:

- v1 ML trained on v1 trades only
- v4 ML trained on v4 trades only

This preserves the distinct entry mechanisms of each variant (v1: general-purpose; v4: selective on high-volatility).

### 3.3 Walk-Forward Cross-Validation

4-fold walk-forward with **expanding training window**:

| Fold | Training Period | Test Period |
|------|-----------------|-------------|
| 1 | 2018 ~ 2021 | 2022 |
| 2 | 2018 ~ 2022 | 2023 |
| 3 | 2018 ~ 2023 | 2024 |
| 4 | 2018 ~ 2024 | 2025 |

Yearly retraining simulates production deployment. Five random seeds [10, 100, 1000, 10000, 100000] are averaged for each fold.

---

## 4. Results

### 4.1 Standalone Performance (Walk-Forward 4-Year, Test 2022-2025)

#### v1 Standalone (5-seed averaged)

| Configuration | Trades | Win Rate | Sharpe | MDD | 4Y Return | CAGR |
|---------------|-------:|---------:|-------:|----:|----------:|-----:|
| v1 Rule baseline | 701 | 56.5% | +0.750 | -48.2% | +107.61% | +20.11% |
| v1 + XGBoost | 290 | 60.1% | +1.060 | -31.1% | +127.11% | +22.56% |

ML filter improvement on v1: Sharpe **+0.310 (+41%)**, MDD **17.1%p reduction**, return **+19.50%p**.

#### v4 Standalone (5-seed averaged)

| Configuration | Trades | Win Rate | Sharpe | MDD | 4Y Return | CAGR |
|---------------|-------:|---------:|-------:|----:|----------:|-----:|
| v4 Rule baseline | 546 | 52.2% | +0.694 | -27.6% | +80.97% | +16.03% |
| v4 + XGBoost | 207 | 58.7% | +0.985 | -28.6% | +99.86% | +18.93% |

ML filter improvement on v4: Sharpe **+0.291 (+42%)**, return **+18.89%p**. MDD comparable.

### 4.2 50/50 Capital-Split Portfolio

Total capital of 100 is split into v1 sub-account (50) and v4 sub-account (50). Portfolio Sharpe computed over time-sorted combined trades.

| Configuration | Sharpe | MDD | 4Y Return | CAGR |
|---------------|-------:|----:|----------:|-----:|
| Baseline (v1 rule + v4 rule) | +0.917 | -35.5% | +94.29% | +18.12% |
| Hybrid (v1 rule + v4 ML) | +1.064 | -31.4% | +103.73% | +19.52% |
| **Full ML (v1 ML + v4 ML)** | **+1.365** | **-28.9%** | **+113.49%** | **+20.79%** |

Adopted configuration: **Full ML 50/50**.

---

## 5. Robustness

Four robustness checks performed on the v1 ML + v4 ML combined Sharpe (5-seed averaged).

| Test | Mean Sharpe | Std | Pass Criterion |
|------|------------:|----:|----------------|
| Hyperparameter (5 configurations) | 1.514 | 0.258 | std < 0.30 ✓ |
| Random Seed (5 seeds) | 1.446 | 0.203 | std < 0.25 ✓ |
| Leave-One-Out (per product) | 1.217 | 0.296 | std < 0.30 ✓ |
| Random Label (negative control) | 0.887 | 0.364 | Effect disappears ✓ |

### 5.1 Authenticity Score

The random label control measures the statistical effect of any threshold-based trade reduction (selecting only high-probability trades shrinks the sample, which alone shifts Sharpe). Comparison of real-label and random-label Sharpe:

$$
\text{Authenticity} = \frac{\text{Sharpe}_{\text{real}} - \text{Sharpe}_{\text{random}}}{\text{Sharpe}_{\text{real}}} = \frac{1.446 - 0.887}{1.446} = 0.81
$$

**81% of the improvement is attributed to genuine ML learning**; 19% is the statistical artifact of threshold-based sample reduction.

### 5.2 Interpretation

- **Hyperparameter robustness** (std 0.258): No specific configuration is required; the result is stable across reasonable settings.
- **Random seed robustness** (std 0.203): Performance is not seed-dependent.
- **Leave-one-out robustness** (std 0.296): The alpha is not driven by any single market.
- **Random label collapse** (Sharpe 0.887): When labels are shuffled, the model still produces a Sharpe above zero due to threshold-based sample reduction; the real-label gap (+0.559) measures the genuine ML contribution.

---

## 6. Forward Validation

A true forward test was conducted using training data through 2025-04-03 and an unseen test period 2025-04-04 to 2026-06-10 (14 months). This test uses **all available data prior to the forward period**, simulating a single-shot deployment decision.

| Configuration | Trades | Win Rate | Sharpe | 14M Return | MDD |
|---------------|-------:|---------:|-------:|-----------:|----:|
| v1 Rule baseline | 179 | 63.1% | +2.101 | +98.67% | -22.9% |
| v1 + XGBoost | 63 | 62.2% | +1.230 | +30.07% | -20.4% |
| v4 Rule baseline | 148 | 52.7% | +1.414 | +48.99% | -23.7% |
| v4 + XGBoost | 55 | 60.2% | +1.738 | +44.50% | -23.6% |

### 6.1 Forward Validation Seed Distribution

#### v1 + XGBoost
| Seed | Trades | Sharpe | 14M Return | MDD |
|-----:|-------:|-------:|-----------:|----:|
| 10 | 61 | +1.526 | +36.34% | -15.0% |
| 100 | 64 | +1.474 | +37.61% | -18.9% |
| 1,000 | 67 | +1.300 | +33.26% | -24.3% |
| 10,000 | 66 | +0.443 | +8.62% | -24.2% |
| 100,000 | 59 | +1.410 | +34.51% | -19.5% |

#### v4 + XGBoost
| Seed | Trades | Sharpe | 14M Return | MDD |
|-----:|-------:|-------:|-----------:|----:|
| 10 | 51 | +1.863 | +47.39% | -23.1% |
| 100 | 56 | +1.540 | +39.12% | -24.6% |
| 1,000 | 63 | +1.667 | +44.04% | -25.9% |
| 10,000 | 50 | +2.240 | +58.56% | -19.8% |
| 100,000 | 53 | +1.379 | +33.37% | -24.6% |

---

## 7. Discussion

### 7.1 ML as a Selectivity Layer

The ML filter does not replace rule-based logic; it operates as a selectivity layer **within** rule-defined entries. The rule layer generates a candidate trade set; the ML layer ranks and filters. This separation preserves the interpretability of Phase 1 rules while adding statistical refinement.

### 7.2 Methodological Sensitivity

The walk-forward methodology with yearly retraining is **essential** for production-grade evaluation. ML models trained on historical data and applied without retraining over multi-year periods show diminished performance due to dataset shift. The standard practice in deployed quantitative systems is periodic retraining, which the walk-forward framework simulates.

### 7.3 Authenticity Score Interpretation

Threshold-based filtering alone produces a sample-selection bias: by selecting trades with predicted probability above 0.60, the model is selecting trades with above-median outcomes, which inflates measured Sharpe even with shuffled labels (Sharpe 0.887). The authenticity score (0.81) separates this statistical artifact from genuine ML learning, providing an honest measure of ML contribution.

### 7.4 Asset Class Constraint Preserved

Phase 2 inherits Phase 1's asset class limitation. ML enhancement does not generalize the alpha to commodities, currencies, or other markets where rule-based alpha failed. The ML model learns patterns within rule-generated trades, not new asset class generalization.

### 7.5 v1 versus v4 Forward Behavior

Forward validation (Section 6) shows v1 ML underperformed v1 rule in the specific 14-month window. This is consistent with the v1 baseline Sharpe of 2.101 in that period — when the rule baseline is exceptionally strong, the ML filter rejects profitable trades. v4 ML maintained consistent improvement (Sharpe 1.738 vs 1.414 baseline) across the same period, reflecting v4's more selective rule structure.

---

## 8. Limitations

- Walk-forward Sharpe 1.365 is based on Test 2022-2025 (4 years). Performance during real-time periodic retraining may differ from historical simulation.
- Forward validation period (14 months) is limited; longer forward periods are required for stronger validation.
- Hyperparameters were fixed without tuning to prevent overfitting; tuning may improve metrics but risks degrading OOS performance.
- ML training data limited to 8.3 years of trade history; longer history would improve robustness.

---

## 9. Conclusion

Phase 2 verifies that XGBoost entry filtering on rule-based v1 and v4 models statistically improves OOS Sharpe ratio on U.S. equity index futures. The 50/50 capital-split portfolio achieves walk-forward Sharpe **1.365** with authenticity score **0.81**, confirming genuine ML learning beyond statistical artifacts of threshold-based sample selection.

### 9.1 Adopted Model

**Full ML 50/50 Portfolio** (v1 ML + v4 ML, 50/50 capital split):

| Metric | Value |
|--------|------:|
| Walk-Forward 4-Year Sharpe | **+1.365** |
| MDD | **-28.9%** |
| Cumulative Return (4Y) | **+113.49%** |
| CAGR | **+20.79%** |
| Authenticity Score | **0.81** |
| Implementation | XGBoost, fixed hyperparameters, no tuning |

### 9.2 Production Deployment Requirements

- **Yearly retraining**: Required for sustained performance
- **Same asset class scope**: U.S. equity index futures only (ES, NQ, YM, RTY)
- **Same timeframe**: 4-hour bars
- **Fixed threshold 0.60**: No per-deployment tuning

---

## 10. Future Research

### A. Real-Time Deployment Validation
- Paper trading with yearly retraining workflow
- Tracking of authenticity score in deployment

### B. Macro Layer Integration (Phase 3)
- Add daily-timeframe regime classifier (e.g., AER) as macro filter
- Multi-timeframe architecture: daily macro + 4-hour trading

### C. Market Expansion
- Test ML enhancement on global equity indices (DAX, FTSE, ESTX50, KOSPI200, NK225)
- Investigate whether ML filtering generalizes where Phase 1 alpha was not tested

### D. Alternative ML Architectures
- LightGBM, CatBoost direct comparison
- Neural network baselines for nonlinear feature interaction

---

## 11. Code Structure

```
project/
├── README.md                       # This document
├── feature_engineering.py          # Trade extraction + 30 features
├── ml_step1_logistic.py            # Logistic Model_1 (local)
├── ml_step2_xgboost.py             # XGBoost Model_2 (local)
├── ml_step3_final_model.py         # XGBoost Separated_v1_v4
├── ml_step3_robustness.py          # 4 robustness checks
└── forward_validation.py           # 14-month forward test
```

**Execution:**
```
# Feature extraction
python feature_engineering.py

# Walk-forward evaluation
python ml_step3_final_model.py

# Robustness checks
python ml_step3_robustness.py

python ml_step3_d_check.py (local)

# Portfolio Execution

> Only executed in local file.
```

---

## 12. References

1. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794.

2. López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.

3. Bailey, D. H., & López de Prado, M. (2014). The deflated Sharpe ratio: Correcting for selection bias, backtest overfitting, and non-normality. *Journal of Portfolio Management*, 40(5), 94–107.

4. Tashman, L. J. (2000). Out-of-sample tests of forecasting accuracy: An analysis and review. *International Journal of Forecasting*, 16(4), 437–450.

---

## 13. Acknowledgments

This study extends the rule-based alpha framework verified in Phase 1. The authenticity score methodology (random label control) provides a rigorous separation of genuine ML learning from threshold-based sample selection artifacts, addressing a common source of false positives in ML-augmented trading systems.

The Walk-Forward Sharpe of 1.365 with authenticity 0.81 represents a statistically defensible enhancement to the Phase 1 baseline, suitable for production deployment under the documented constraints.

---
**Version**: 2.0 (as of 2026-06) 

**Status**: Phase 1 (rule-based) completed. Phase 2 (ML enhancement) completed.

**New Model**: Project 3 (multi-timeframe macro layer - advanced state classifier) planned.

