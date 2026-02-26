# Moneyball Dojo — 2025 Season Backtest Results

Generated: 2026-02-26

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

- **Overall Accuracy**: 52.9% (2425 games)
- **AUC-ROC**: 0.5533

### Accuracy by Confidence Tier

| Tier | Accuracy | Record |
|------|----------|--------|
| STRONG | 58.7% | 323/550 |
| MODERATE | 53.0% | 376/709 |
| LEAN | 50.5% | 379/750 |
| PASS | 49.3% | 205/416 |

---

## Over/Under

- **MAE**: 3.640 runs
- **Total games**: 2425

### Line Accuracy

| Line | Accuracy |
|------|----------|
| 7.5 | 54.6% |
| 8.0 | 49.3% |
| 8.5 | 50.8% |
| 9.0 | 56.1% |
| 9.5 | 59.4% |

---

## Run Line

- **Overall Accuracy**: 64.5% (2425 games)
- **AUC-ROC**: 0.5606

### Accuracy by Confidence Tier

| Tier | Accuracy | Record |
|------|----------|--------|
| STRONG | 67.3% | 998/1482 |
| MODERATE | 62.6% | 305/487 |
| LEAN | 60.3% | 185/307 |
| PASS | 51.7% | 77/149 |

---

## F5 Moneyline

- **Overall Accuracy**: 54.5% (2425 games)
- **AUC-ROC**: 0.5765

### Accuracy by Confidence Tier

| Tier | Accuracy | Record |
|------|----------|--------|
| STRONG | 59.9% | 656/1096 |
| MODERATE | 51.4% | 343/667 |
| LEAN | 50.6% | 89/176 |
| PASS | 47.9% | 233/486 |

---

## NRFI

- **Overall Accuracy**: 56.5% (2429 games)
- **AUC-ROC**: 0.5842

### Accuracy by Confidence Tier

| Tier | Accuracy | Record |
|------|----------|--------|
| STRONG | 60.1% | 401/667 |
| MODERATE | 57.8% | 451/780 |
| LEAN | 53.0% | 521/982 |

---

## Monthly Breakdown (Moneyline)

| Month | Games | Accuracy | STRONG Acc | STRONG Games |
|-------|-------|----------|------------|-------------|
| 2025-03 | 62 | 45.2% | 54.5% | 11 |
| 2025-04 | 389 | 55.3% | 64.5% | 110 |
| 2025-05 | 411 | 52.1% | 50.6% | 81 |
| 2025-06 | 397 | 51.9% | 56.7% | 90 |
| 2025-07 | 370 | 49.7% | 48.8% | 84 |
| 2025-08 | 422 | 54.5% | 65.0% | 100 |
| 2025-09 | 374 | 55.1% | 64.9% | 74 |

---

*This backtest is based on historical data. Past performance does not guarantee future results.*
*Not financial advice. Gamble responsibly.*