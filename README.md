# Moneyball Dojo — AI-Powered MLB Predictions

> Built by a Japanese AI engineer in Tokyo. 9 models. 37 features. Data > gut feelings.

[![Daily Predictions](https://github.com/yasuuucchi/-moneyball_dojo/actions/workflows/moneyball_daily.yml/badge.svg)](https://github.com/yasuuucchi/-moneyball_dojo/actions)

## What is Moneyball Dojo?

A production-ready MLB prediction system that:
- Runs **9 XGBoost models** across multiple betting markets
- Generates daily bilingual content (English for Substack, Japanese for note.com)
- Publishes via GitHub Actions with zero manual intervention on the ML side
- Tracks all predictions in Google Sheets for transparent performance logging

## Backtest Results (2025 Season — 2,426 Games)

Models were trained on 2022-2024 data and tested on the full 2025 season with **zero data leakage**.

| Model | Overall Accuracy | STRONG Tier | AUC-ROC |
|-------|-----------------|-------------|---------|
| **Moneyline** | 64.7% | 72.9% (960/1317) | 0.704 |
| **Run Line (-1.5)** | 66.3% | 73.4% (956/1303) | 0.660 |
| **F5 Moneyline** | 64.6% | 69.5% (1242/1787) | 0.709 |
| **NRFI** | 62.1% | 68.9% (911/1322) | 0.673 |
| **Over/Under** | MAE 3.38 | Line accuracy 60-64% | — |

Full backtest report: [`output/backtest_2025/backtest_report_2025.md`](output/backtest_2025/backtest_report_2025.md)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full daily pipeline (fetches data, predicts, generates digest)
python run_daily.py

# Output goes to output/YYYYMMDD/
#   ├── digest_EN_YYYY-MM-DD.md   (Substack)
#   ├── digest_JA_YYYY-MM-DD.md   (note.com)
#   ├── twitter_YYYY-MM-DD.txt    (X/Twitter)
#   ├── predictions.csv            (Google Sheets)
#   └── predictions_YYYY-MM-DD.json (full data)
```

## 9 Trained Models

```
models/
├── model_moneyline.pkl      — Win/loss prediction (main)
├── model_over_under.pkl     — Total runs Over/Under
├── model_run_line.pkl       — -1.5 spread
├── model_f5_moneyline.pkl   — First 5 innings
├── model_nrfi.pkl           — No Run First Inning (62% accuracy)
├── model_pitcher_k.pkl      — Pitcher strikeouts
├── model_pitcher_outs.pkl   — Pitcher outs
├── model_batter_props.pkl   — Batter props
└── model_stolen_bases.pkl   — Stolen base props
```

## 37 Features

Each game prediction uses a 37-dimensional feature vector:

- **Season stats**: Win%, run differential, Pythagorean W%
- **Batting**: BA, OBP, SLG, OPS (home & away)
- **Pitching**: ERA, WHIP, K/9, BB/9 (home & away)
- **Splits**: Home/away performance splits
- **Rolling**: 15-game rolling window stats
- **Composite**: Offensive/defensive strength indices

## Daily Pipeline (`run_daily.py`)

```
06:00  MLB Stats API → fetch today's schedule + pitchers
06:05  Generate 37-feature vectors per game
06:10  Run 9 XGBoost models → predictions + confidence tiers
06:15  Calculate edge (model probability vs. implied odds)
06:20  Generate Daily Digest (EN + JA markdown + Twitter)
06:25  Upload predictions to Google Sheets
06:30  Done — ready to publish
```

### Confidence Tiers

| Tier | Edge | Meaning |
|------|------|---------|
| STRONG | 8%+ | High conviction — best bets |
| MODERATE | 4-8% | Solid edge |
| LEAN | 2-4% | Slight lean |
| PASS | <2% | Too close to call |

## Project Structure

```
moneyball_dojo/
├── Core Pipeline
│   ├── run_daily.py                  ← Main entry point (9 models)
│   ├── train_all_models.py           ← Train all 9 models
│   ├── backtest_2025.py              ← Full season validation
│   └── generate_track_record.py      ← Performance page generator
│
├── Data Pipeline
│   ├── fetch_real_data.py            ← MLB Stats API
│   ├── fetch_player_data.py          ← Player-level data
│   ├── fetch_nrfi_data.py            ← NRFI-specific data
│   └── fetch_game_props_data.py      ← Props data
│
├── Content Generation
│   ├── daily_digest_generator.py     ← Digest templates (EN + JA)
│   └── article_generator_template.py ← Claude API article prompts
│
├── Database
│   ├── sheets_schema_v2.py           ← Google Sheets event-log schema
│   └── update_daily_results.py       ← Results tracking
│
├── Legacy/Reference
│   ├── data_pipeline_demo.py         ← Original demo pipeline
│   ├── train_model.py                ← v1 model training
│   └── train_model_v2.py             ← v2 model training
│
├── Articles (articles/)
│   ├── 01_introducing_moneyball_dojo_{EN,JA}.md
│   ├── 02_how_our_ai_model_works_{EN,JA}.md
│   ├── 03_2025_season_review_{EN,JA}.md
│   └── 04_2026_division_predictions_{EN,JA}.md
│
├── Output (output/)
│   ├── backtest_2025/                ← Backtest results
│   └── YYYYMMDD/                     ← Daily predictions
│
├── Infra
│   ├── .github/workflows/            ← GitHub Actions
│   └── requirements.txt
│
└── Docs
    ├── ROADMAP.md
    ├── SETUP_GUIDE.md
    ├── QUICKSTART.md
    ├── README_TRAIN_MODEL.md
    ├── claude_pro_digest_prompt.md
    └── claude_pro_project_instructions.md
```

## GitHub Actions

The workflow (`.github/workflows/moneyball_daily.yml`) runs automatically:

1. **Fetch data** — MLB Stats API for today's games
2. **Generate predictions** — All 9 models
3. **Create digest** — EN + JA markdown + Twitter post
4. **Upload to Sheets** — Append predictions to Google Sheets
5. **Notify** — Create GitHub Issue / Slack alert

### Required Secrets

| Secret | Purpose |
|--------|---------|
| `CLAUDE_API_KEY` | Article generation (optional) |
| `GOOGLE_SHEETS_CREDENTIALS` | Sheets API service account JSON (base64) |
| `GOOGLE_SHEETS_ID` | Target spreadsheet ID |
| `SLACK_WEBHOOK` | Notifications (optional) |

## Model Training

```bash
# Retrain all 9 models on latest data
python train_all_models.py

# Run backtest on 2025 season
python backtest_2025.py

# Generate track record page
python generate_track_record.py
```

See [`README_TRAIN_MODEL.md`](README_TRAIN_MODEL.md) for detailed training documentation.

## Content Platforms

| Platform | Language | Frequency | Content |
|----------|----------|-----------|---------|
| **Substack** | English | Daily | Full digest + analysis |
| **note.com** | Japanese | Weekly | Japanese digest |
| **X/Twitter** | English | Daily | Top 5 picks (280 chars) |
| **Google Sheets** | — | Daily | Raw prediction data |

## Cost

| Item | Monthly |
|------|---------|
| Claude Pro | $20 |
| Claude API (Haiku) | $5-15 |
| Domain | $1 |
| GitHub Actions / Sheets / Substack | $0 |
| **Total** | **$26-36** |

## License

MIT License

---

**Last Updated:** February 2026
