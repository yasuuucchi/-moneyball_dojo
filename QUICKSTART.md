# Moneyball Dojo - Quick Start Guide

Get up and running in 5 minutes.

## Installation

```bash
pip install -r requirements.txt
```

## Run the Daily Pipeline

```bash
python run_daily.py
```

This single command:
1. Fetches today's MLB schedule from MLB Stats API
2. Loads 9 trained XGBoost models
3. Generates 37-feature vectors for each game
4. Predicts across all markets (Moneyline, O/U, Run Line, F5, NRFI, Props)
5. Calculates edge vs. implied odds
6. Generates English + Japanese digests + Twitter post
7. Saves everything to `output/YYYYMMDD/`

### Output Files

```
output/YYYYMMDD/
├── digest_EN_YYYY-MM-DD.md    → Copy-paste to Substack
├── digest_JA_YYYY-MM-DD.md    → Copy-paste to note.com
├── twitter_YYYY-MM-DD.txt     → Copy-paste to X/Twitter
├── predictions.csv             → Google Sheets import
└── predictions_YYYY-MM-DD.json → Full prediction data
```

## Model Training

```bash
# Retrain all 9 models
python train_all_models.py

# Validate on 2025 season (2,426 games)
python backtest_2025.py

# Generate track record page
python generate_track_record.py
```

## Key Scripts

| Script | Purpose |
|--------|---------|
| `run_daily.py` | Main pipeline — run this daily |
| `train_all_models.py` | Train all 9 XGBoost models |
| `backtest_2025.py` | Full season backtest validation |
| `generate_track_record.py` | Performance page for Substack |
| `daily_digest_generator.py` | Digest template engine |
| `sheets_schema_v2.py` | Google Sheets event-log schema |

## Confidence Tiers

| Tier | Edge | Emoji |
|------|------|-------|
| STRONG | 8%+ | 🔥 |
| MODERATE | 4-8% | 👍 |
| LEAN | 2-4% | → |
| PASS | <2% | ⏸ |

## Daily Workflow (90 seconds)

```
1. python run_daily.py           ← generates everything
2. Open output/YYYYMMDD/
3. Copy digest_EN → paste to Substack → Publish
4. Copy twitter → paste to X → Post
5. (Weekly) Copy digest_JA → paste to note.com → Publish
```

## Backtest Results

| Model | Accuracy | STRONG Tier |
|-------|----------|-------------|
| Moneyline | 64.7% | 72.9% |
| Run Line | 66.3% | 73.4% |
| F5 Moneyline | 64.6% | 69.5% |
| NRFI | 62.1% | 68.9% |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| ModuleNotFoundError | `pip install -r requirements.txt` |
| No games found | Off-season or no games scheduled for today |
| Model file missing | Run `python train_all_models.py` |
| Google Sheets not updating | Check credentials in `.env` or GitHub Secrets |

## Next Steps

1. Read [`README.md`](README.md) for full documentation
2. Read [`ROADMAP.md`](ROADMAP.md) for project timeline
3. Read [`SETUP_GUIDE.md`](SETUP_GUIDE.md) for platform setup (Substack, X, etc.)
4. Read [`README_TRAIN_MODEL.md`](README_TRAIN_MODEL.md) for model training details
