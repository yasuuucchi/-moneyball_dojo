# Moneyball Dojo — Verified Track Record

## 2025 Season Results (STRONG Picks Only)

All results below are from our **walk-forward backtest** on the full 2025 MLB season.
Models were trained on 2022-2024 data only. No lookahead bias. No cherry-picking.

---

## Performance Summary

| Market | Record | Win Rate | ROI | Games |
|--------|--------|----------|-----|-------|
| Moneyline | 960W - 357L | **72.9%** | **+39.2%** | 1317 |
| Run Line | 956W - 347L | **73.4%** | **+40.1%** | 1303 |
| Over/Under | 671W - 303L | **68.9%** | **+31.5%** | 974 |
| F5 Moneyline | 1242W - 545L | **69.5%** | **+32.7%** | 1787 |

> **Combined STRONG picks: 3829W - 1552L (71.2%) | ROI: +35.8%**

---

## Moneyline — STRONG Picks Detail

- **Overall model accuracy (all tiers):** 64.7% on 2426 games
- **AUC-ROC:** 0.7040
- **STRONG picks accuracy:** 72.9% (960/1317)
- **STRONG picks ROI:** +39.2% (at standard -110 juice)

### Monthly Trend

| Month | Record | Win Rate | ROI |
|-------|--------|----------|-----|
| 2025-03 | 34W-8L | 81.0% | +54.5% |
| 2025-04 | 158W-50L | 76.0% | +45.0% |
| 2025-05 | 163W-58L | 73.8% | +40.8% |
| 2025-06 | 164W-60L | 73.2% | +39.8% |
| 2025-07 | 139W-53L | 72.4% | +38.2% |
| 2025-08 | 154W-70L | 68.8% | +31.2% |
| 2025-09 | 148W-58L | 71.8% | +37.2% |

## Run Line — STRONG Picks Detail

- **Overall model accuracy (all tiers):** 66.3% on 2426 games
- **AUC-ROC:** 0.6602
- **STRONG picks accuracy:** 73.4% (956/1303)
- **STRONG picks ROI:** +40.1% (at standard -110 juice)

### Monthly Trend

| Month | Record | Win Rate | ROI |
|-------|--------|----------|-----|
| 2025-03 | 30W-7L | 81.1% | +54.8% |
| 2025-04 | 142W-61L | 70.0% | +33.5% |
| 2025-05 | 181W-56L | 76.4% | +45.8% |
| 2025-06 | 162W-62L | 72.3% | +38.1% |
| 2025-07 | 133W-54L | 71.1% | +35.8% |
| 2025-08 | 154W-63L | 71.0% | +35.5% |
| 2025-09 | 154W-44L | 77.8% | +48.5% |

## Over/Under — STRONG Picks Detail

- **Overall MAE:** 3.381 runs on 2426 games
- **STRONG picks accuracy:** 68.9% (671/974)
- **STRONG picks ROI:** +31.5% (at standard -110 juice)

### Monthly Trend

| Month | Record | Win Rate | ROI |
|-------|--------|----------|-----|
| 2025-03 | 14W-4L | 77.8% | +48.5% |
| 2025-04 | 115W-43L | 72.8% | +39.0% |
| 2025-05 | 122W-68L | 64.2% | +22.6% |
| 2025-06 | 104W-44L | 70.3% | +34.2% |
| 2025-07 | 112W-50L | 69.1% | +32.0% |
| 2025-08 | 99W-38L | 72.3% | +38.0% |
| 2025-09 | 105W-56L | 65.2% | +24.5% |

## F5 Moneyline — STRONG Picks Detail

- **Overall model accuracy (all tiers):** 64.6% on 2426 games
- **AUC-ROC:** 0.7092
- **STRONG picks accuracy:** 69.5% (1242/1787)
- **STRONG picks ROI:** +32.7% (at standard -110 juice)

### Monthly Trend

| Month | Record | Win Rate | ROI |
|-------|--------|----------|-----|
| 2025-03 | 40W-13L | 75.5% | +44.1% |
| 2025-04 | 200W-77L | 72.2% | +37.8% |
| 2025-05 | 216W-89L | 70.8% | +35.2% |
| 2025-06 | 206W-88L | 70.1% | +33.8% |
| 2025-07 | 178W-91L | 66.2% | +26.3% |
| 2025-08 | 200W-105L | 65.6% | +25.2% |
| 2025-09 | 202W-82L | 71.1% | +35.8% |

---

## Methodology

- **Algorithm:** XGBoost (gradient boosted decision trees)
- **Training data:** 2022-2024 MLB seasons (7,283 games)
- **Test data:** 2025 full season (2,426 games)
- **Walk-forward validation:** Rolling stats use only pre-game data
- **No data leakage:** 2025 data was never used in training
- **Confidence tiers:** STRONG picks require model probability significantly away from 50%
- **ROI calculation:** Assumes standard -110 American odds (1.909 decimal)

### What Makes a STRONG Pick?

Our AI model assigns confidence tiers based on the strength of its signal:

| Tier | Criteria | 2025 ML Accuracy |
|------|----------|-----------------|
| STRONG | High conviction signal | 72.9% (1317 games) |
| MODERATE | Moderate conviction | 56.0% (537 games) |
| LEAN | Slight edge detected | 53.3% (360 games) |
| PASS | No actionable edge | 55.2% (212 games) |

We only publish STRONG picks — maximizing quality over quantity.

---

*Last updated: 2026-02-15*
*Past performance does not guarantee future results. Not financial advice.*