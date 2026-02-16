# Moneyball Dojo — Model Training Guide

## Overview

9 XGBoost models trained on real MLB data (2022-2025) to predict game outcomes across multiple betting markets.

## Quick Start

```bash
# Train all 9 models at once
python train_all_models.py

# Validate on 2025 full season
python backtest_2025.py
```

## Training Scripts

| Script | Purpose |
|--------|---------|
| `train_all_models.py` | Train all 9 models (recommended) |
| `backtest_2025.py` | Walk-forward backtest on 2025 season |
| `train_model_v2.py` | Single model training (v2) |
| `train_model.py` | Original single model training (v1, legacy) |

## 9 Models

| Model | Type | Target | Output |
|-------|------|--------|--------|
| Moneyline | Classifier | Home win (0/1) | `model_moneyline.pkl` |
| Over/Under | Regressor | Total runs | `model_over_under.pkl` |
| Run Line | Classifier | Home -1.5 cover (0/1) | `model_run_line.pkl` |
| F5 Moneyline | Classifier | Home leads after 5 (0/1) | `model_f5_moneyline.pkl` |
| NRFI | Classifier | No runs in 1st inning (0/1) | `model_nrfi.pkl` |
| Pitcher K | Regressor | Strikeout count | `model_pitcher_k.pkl` |
| Pitcher Outs | Regressor | Outs recorded | `model_pitcher_outs.pkl` |
| Batter Props | Classifier | Hit/HR props | `model_batter_props.pkl` |
| Stolen Bases | Regressor | Stolen base count | `model_stolen_bases.pkl` |

## Feature Engineering (37 Features)

### Season-Level Metrics
- Win percentage
- Run differential
- Pythagorean win percentage

### Batting (Home & Away)
- Batting Average (BA)
- On-Base Percentage (OBP)
- Slugging Percentage (SLG)
- OPS (On-base Plus Slugging)

### Pitching (Home & Away)
- ERA (Earned Run Average)
- WHIP (Walks + Hits per IP)
- K/9 (Strikeouts per 9 innings)
- BB/9 (Walks per 9 innings)

### Splits & Context
- Home/away performance splits
- 15-game rolling window stats
- Composite offensive strength index
- Composite defensive strength index

### NRFI-Specific Features (model_nrfi.pkl)
- Starting pitcher 1st inning ERA
- Team 1st inning scoring rate
- Pitcher K rate in 1st inning

## Training Data

| Dataset | Seasons | Games |
|---------|---------|-------|
| Training | 2022-2024 | ~7,283 |
| Validation | 2025 | 2,426 |
| **Total** | 2022-2025 | ~9,709 |

Data source: MLB Stats API via `fetch_real_data.py`

## XGBoost Configuration

```python
XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
```

## Backtest Results (2025 Season)

Walk-forward methodology — models trained ONLY on 2022-2024, tested on 2025 with no data leakage.

### Main Models

| Model | Accuracy | STRONG Tier | AUC-ROC |
|-------|----------|-------------|---------|
| Moneyline | 64.7% | 72.9% (960/1317) | 0.704 |
| Run Line | 66.3% | 73.4% (956/1303) | 0.660 |
| F5 Moneyline | 64.6% | 69.5% (1242/1787) | 0.709 |
| NRFI | 62.1% | 68.9% (911/1322) | 0.673 |
| Over/Under | MAE 3.38 | Line accuracy 60-64% | — |

### Monthly Accuracy (Moneyline)

| Month | Games | Overall | STRONG |
|-------|-------|---------|--------|
| Mar | 63 | 71.4% | 81.0% |
| Apr | 389 | 67.6% | 76.0% |
| May | 411 | 65.9% | 73.8% |
| Jun | 397 | 63.2% | 73.2% |
| Jul | 370 | 61.6% | 72.4% |
| Aug | 422 | 60.0% | 68.8% |
| Sep | 374 | 69.3% | 71.8% |

### Confidence Tiers

| Tier | Edge Threshold | Meaning |
|------|---------------|---------|
| STRONG | 8%+ | High conviction |
| MODERATE | 4-8% | Solid edge |
| LEAN | 2-4% | Slight lean |
| PASS | <2% | Skip |

## Output Files

After running `train_all_models.py`:

```
models/
├── model_moneyline.pkl
├── model_over_under.pkl
├── model_run_line.pkl
├── model_f5_moneyline.pkl
├── model_nrfi.pkl
├── model_pitcher_k.pkl
├── model_pitcher_outs.pkl
├── model_batter_props.pkl
└── model_stolen_bases.pkl
```

After running `backtest_2025.py`:

```
output/backtest_2025/
├── backtest_report_2025.md    — Human-readable report
├── backtest_summary.json      — Machine-readable metrics
└── backtest_per_game.csv      — Per-game predictions vs actuals
```

## Usage in Production

Models are loaded by `run_daily.py` automatically:

```python
import pickle

with open('models/model_moneyline.pkl', 'rb') as f:
    data = pickle.load(f)
    model = data['model']
    scaler = data['scaler']
    features = data['feature_cols']
```

## Retraining Schedule

- **Pre-season**: Retrain on all available data (currently 2022-2025)
- **Mid-season**: Optional retrain to incorporate current season data
- **Post-season**: Full retrain + backtest on completed season

---

**Last Updated**: 2026-02-16
**Status**: Production Ready
