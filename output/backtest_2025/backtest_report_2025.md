# Moneyball Dojo — 2025 Season Backtest Results

Generated: 2026-02-14

## Methodology

- **Training Data**: 2022-2024 seasons (7,283 games)
- **Test Data**: 2025 full season (2,430 games)
- **Method**: Walk-forward backtesting
  - Models trained ONLY on 2022-2024 data
  - Rolling stats computed using only games before each prediction date
  - Team stats from 2024 as baseline (simulating real pre-season conditions)
- **Models**: XGBoost classifiers/regressors
- **No data leakage**: 2025 data was never used in training

---

## Moneyline

- **Overall Accuracy**: 64.7% (2426 games)
- **AUC-ROC**: 0.7040

### Accuracy by Confidence Tier

| Tier | Accuracy | Record |
|------|----------|--------|
| STRONG | 72.9% | 960/1317 |
| MODERATE | 56.0% | 301/537 |
| LEAN | 53.3% | 192/360 |
| PASS | 55.2% | 117/212 |

---

## Over/Under

- **MAE**: 3.381 runs
- **Total games**: 2426

### Line Accuracy

| Line | Accuracy |
|------|----------|
| 7.5 | 62.2% |
| 8.0 | 60.5% |
| 8.5 | 61.3% |
| 9.0 | 63.1% |
| 9.5 | 64.1% |

---

## Run Line

- **Overall Accuracy**: 66.3% (2426 games)
- **AUC-ROC**: 0.6602

### Accuracy by Confidence Tier

| Tier | Accuracy | Record |
|------|----------|--------|
| STRONG | 73.4% | 956/1303 |
| MODERATE | 62.1% | 300/483 |
| LEAN | 57.0% | 227/398 |
| PASS | 51.6% | 125/242 |

---

## F5 Moneyline

- **Overall Accuracy**: 64.6% (2426 games)
- **AUC-ROC**: 0.7092

### Accuracy by Confidence Tier

| Tier | Accuracy | Record |
|------|----------|--------|
| STRONG | 69.5% | 1242/1787 |
| MODERATE | 55.0% | 175/318 |
| LEAN | 38.2% | 29/76 |
| PASS | 49.4% | 121/245 |

---

## NRFI

- **Overall Accuracy**: 53.5% (2429 games)
- **AUC-ROC**: 0.5535

### Accuracy by Confidence Tier

| Tier | Accuracy | Record |
|------|----------|--------|
| STRONG | 54.5% | 103/189 |
| MODERATE | 56.4% | 494/876 |
| LEAN | 51.5% | 702/1364 |

---

## Monthly Breakdown (Moneyline)

| Month | Games | Accuracy | STRONG Acc | STRONG Games |
|-------|-------|----------|------------|-------------|
| 2025-03 | 63 | 71.4% | 81.0% | 42 |
| 2025-04 | 389 | 67.6% | 76.0% | 208 |
| 2025-05 | 411 | 65.9% | 73.8% | 221 |
| 2025-06 | 397 | 63.2% | 73.2% | 224 |
| 2025-07 | 370 | 61.6% | 72.4% | 192 |
| 2025-08 | 422 | 60.0% | 68.8% | 224 |
| 2025-09 | 374 | 69.3% | 71.8% | 206 |

---

*This backtest is based on historical data. Past performance does not guarantee future results.*
*Not financial advice. Gamble responsibly.*