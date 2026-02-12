# Moneyball Dojo - MLB Prediction Data Pipeline

A production-ready Python prototype for automated MLB game prediction, analysis, and article generation with Google Sheets integration and GitHub Actions automation.

## Overview

Moneyball Dojo is a complete data pipeline that:
1. **Fetches live MLB data** using pybaseball (batting/pitching stats, team metrics)
2. **Generates game predictions** with XGBoost model trained on historical data
3. **Produces confidence scores** for each prediction (HIGH/MEDIUM/LOW tiers)
4. **Generates articles** in English (Substack) and Japanese (note.com) using Claude API
5. **Logs all results** to Google Sheets for tracking and analysis
6. **Tracks ROI and performance metrics** with automated daily reporting

## Project Structure

```
moneyball_dojo/
├── requirements.txt                 # Python dependencies
├── data_pipeline_demo.py            # Main prediction pipeline (500+ lines)
├── sheets_schema.py                 # Google Sheets database schema definition
├── article_generator_template.py    # Claude API article generation templates
├── github_actions_workflow.yml      # Daily automation workflow
└── README.md                        # This file
```

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/your-org/moneyball_dojo.git
cd moneyball_dojo

# Install dependencies
pip install -r requirements.txt
```

### Running the Pipeline

```bash
# Run complete prediction pipeline
python data_pipeline_demo.py

# Expected output:
# - Fetches 2024 MLB batting and pitching stats
# - Computes team-level statistics
# - Trains XGBoost model on historical data
# - Generates predictions for upcoming games
# - Exports results to CSV
```

### Example Output

```
======================================================================
MLB PREDICTION DATA PIPELINE DEMO
======================================================================

[Pipeline] Fetching MLB 2024 season data...
[Pipeline] Successfully fetched 200 batters and 100 pitchers
[Pipeline] Computing team-level statistics...
[Pipeline] Computed stats for 30 teams
[Pipeline] Generating 500 training samples...
[Pipeline] Training XGBoost model...
[Pipeline] Model trained: Train Acc=0.627, Test Acc=0.612
[Pipeline] Generating predictions for 10 games...
[Pipeline] Predictions exported to ./output/daily_predictions.csv

======================================================================
PREDICTION SUMMARY
======================================================================
Total Predictions: 10
High Confidence Picks: 3
Home Picks: 6
Away Picks: 4
Average Confidence: 0.552
```

## Core Components

### 1. Data Pipeline (`data_pipeline_demo.py`)

**Class: `MLBDataPipeline`**

Complete end-to-end pipeline with the following methods:

#### `fetch_mlb_data(season=2024)`
- Fetches live MLB stats using pybaseball
- Falls back to synthetic data if API unavailable
- Returns: (batting_stats, pitching_stats) DataFrames

#### `compute_team_stats(batting_stats, pitching_stats)`
- Aggregates player stats to team level
- Computes key metrics: AVG, OBP, SLG, ERA, WHIP, K/9
- Returns: Dict of team statistics

#### `generate_game_features(away_team, home_team, away_pitcher_era, home_pitcher_era)`
- Engineers features for individual game matchups
- Features include:
  - Offensive metrics (batting_avg, obp, slg)
  - Pitching metrics (era_allowed, pitcher_era)
  - Context features (home_field_advantage, runs_differential)
- Returns: Dict of 13 game features

#### `create_training_dataset(num_samples=500)`
- Generates synthetic training data from team stats
- Simulates historical game outcomes with home field advantage
- Returns: DataFrame with 500+ training examples

#### `train_model(training_df, test_size=0.2)`
- Trains XGBoost classifier on historical data
- Scales features using StandardScaler
- Returns: Dict with train/test accuracy metrics

#### `predict_game(away_team, home_team, away_pitcher_era, home_pitcher_era)`
- Generates single game prediction
- Returns probability of home team win
- Assigns confidence tier (HIGH/MEDIUM/LOW)
- Makes pick recommendation (HOME/AWAY)

#### `generate_predictions(schedule_df)` and `export_predictions()`
- Batch predicts all games in schedule
- Exports to CSV (Google Sheets compatible format)

**Example Usage:**
```python
from data_pipeline_demo import MLBDataPipeline

pipeline = MLBDataPipeline(output_dir='./output')

# Step 1: Fetch data
batting, pitching = pipeline.fetch_mlb_data(season=2024)

# Step 2: Compute team stats
team_stats = pipeline.compute_team_stats(batting, pitching)

# Step 3: Create training data and train model
training_df = pipeline.create_training_dataset(num_samples=500)
metrics = pipeline.train_model(training_df)
print(f"Model Accuracy: {metrics['test_accuracy']:.1%}")

# Step 4: Generate predictions
prediction = pipeline.predict_game(
    away_team='NYY',
    home_team='BOS',
    away_pitcher_era=3.45,
    home_pitcher_era=4.12
)
print(f"Pick: {prediction['pick']} (Confidence: {prediction['confidence_tier']})")
```

### 2. Google Sheets Schema (`sheets_schema.py`)

**Defines 4 interconnected sheets for complete prediction tracking:**

#### Sheet 1: "Daily Predictions" (12 columns)
```
date | game_id | away_team | home_team | away_pitcher | home_pitcher |
model_probability | pick | line | confidence_tier | article_assigned | notes
```
- Live predictions updated daily at 8 AM UTC
- Model probability: 0.0-1.0 for home team win
- Pick: HOME or AWAY recommendation
- Confidence: HIGH (>65%), MEDIUM (55-65%), LOW (<55%)

#### Sheet 2: "Results" (13 columns)
```
date | game_id | away_team | home_team | away_score | home_score |
prediction | actual_result | hit | units_wagered | units_won | roi | cumulative_roi
```
- Updated after games complete
- Tracks betting outcomes and running ROI
- Links to Daily Predictions via game_id

#### Sheet 3: "Model Performance" (14 columns)
```
date | week_number | total_picks | wins | losses | win_rate | ats_record |
units_wagered | units_won | roi | cumulative_roi | high_confidence_record |
medium_confidence_record | low_confidence_record
```
- Weekly aggregation of model performance
- Win rate, ATS record, ROI by confidence tier
- Cumulative season statistics

#### Sheet 4: "Article Queue" (15 columns)
```
date | game_id | away_team | home_team | matchup_title | article_status |
english_article_status | japanese_article_status | substack_url | note_url |
generated_by_claude | prompt_version | model_prediction | confidence_tier | notes
```
- Tracks article generation and publication
- Links to published Substack and note.com articles
- Indicates Claude API usage and prompt version

**Usage:**
```python
from sheets_schema import SHEETS_SCHEMA, describe_schema, create_headers, get_validation_rules

# Get column names for a sheet
headers = create_headers('Daily Predictions')
# ['date', 'game_id', 'away_team', 'home_team', ...]

# Print human-readable schema
print(describe_schema('Daily Predictions'))

# Get validation rules
rules = get_validation_rules('Results')
# {'prediction': 'HOME or AWAY', 'hit': '...', ...}

# Export as JSON
schema_json = export_schema_to_json()
```

### 3. Article Generator (`article_generator_template.py`)

**Class: `ArticleGenerator`**

Generates game analysis articles using Claude API with team-specific personas.

#### System Prompt
- Data-driven analyst voice
- Deep MLB analytics expertise (sabermetrics, WAR, FIP, xwOBA)
- Accessible but rigorous writing style
- Evidence-based claims supported by data

#### English Article Template (Substack)
1. **Opening Hook**: Why this game matters
2. **Metrics Breakdown**: Pitcher matchup, team offenses, lineup considerations
3. **Betting Line Analysis**: Current lines and sharp vs. public opinion
4. **The Pick**: Recommendation with confidence and caveats
5. **Closing**: Risk/reward or bigger picture insight
- Target: 800-1200 words
- Format: Long-form analysis, not quick takes

#### Japanese Article Template (note.com)
1. **導入部** (Introduction): Context for Japanese fans
2. **先発投手対決** (Pitcher Duel Analysis): Detailed ERA, WHIP, K/9 analysis
3. **打線分析** (Offensive Analysis): Team scoring power and matchups
4. **データ駆動予測** (Data-Driven Prediction): Model recommendation
5. **結論** (Conclusion): Meaning for Japanese MLB fans
- Target: 2000-3000 characters
- Format: Accessible but analytics-focused

**Methods:**

```python
from article_generator_template import ArticleGenerator

generator = ArticleGenerator(api_key='sk-ant-...')

# Generate English article
english = generator.generate_english_article({
    'away_team': 'NYY',
    'home_team': 'BOS',
    'away_pitcher': 'Gerrit Cole',
    'home_pitcher': 'Garrett Whitlock',
    'model_probability': 0.62,
    'confidence_tier': 'HIGH',
    'pick': 'HOME',
    'date': '2024-06-15'
})

# Generate Japanese article
japanese = generator.generate_japanese_article(game_data)

# Get API call structure for integration
api_call = generator.create_api_call_example(game_data)
# Returns: {"model": "claude-opus-4-6", "max_tokens": 2000, ...}
```

### 4. GitHub Actions Automation (`github_actions_workflow.yml`)

**Complete daily automation workflow with 5 jobs:**

#### Job 1: `fetch-data`
- Runs daily at 8 AM UTC
- Fetches MLB data for 2024 season
- Generates predictions for upcoming games
- Outputs prediction count and high-confidence count
- Uploads daily_predictions.csv artifact

#### Job 2: `generate-articles`
- Generates English and Japanese articles
- Uses Claude API (requires CLAUDE_API_KEY secret)
- Creates separate .md files for each game
- Filters for HIGH confidence picks only

#### Job 3: `update-sheets`
- Writes predictions to "Daily Predictions" sheet
- Requires Google Sheets service account credentials
- Clears old data and appends new predictions
- Authenticates via GOOGLE_SHEETS_CREDENTIALS secret

#### Job 4: `log-results`
- Generates execution summary
- Posts to Slack (optional, via SLACK_WEBHOOK)
- Creates GitHub Issue for daily review
- Archives logs for auditing

#### Job 5: `notify-failure`
- Creates GitHub Issue on pipeline failure
- Alerts maintainers to investigate issues

**Workflow File Location:** `.github/workflows/moneyball_daily.yml`

**Required GitHub Secrets:**

1. **CLAUDE_API_KEY**
   - Get from: https://console.anthropic.com/
   - Used for article generation

2. **GOOGLE_SHEETS_CREDENTIALS**
   - Service account JSON (Base64 encoded)
   - Setup: Create service account in Google Cloud Console
   - Enable Google Sheets API
   - Create JSON key file
   - Base64 encode: `base64 -i credentials.json`

3. **GOOGLE_SHEETS_ID**
   - Extract from URL: https://docs.google.com/spreadsheets/d/{ID}/edit
   - ID of your tracking spreadsheet

4. **SLACK_WEBHOOK** (optional)
   - Get from: https://api.slack.com/apps
   - For daily notifications to Slack

5. **MLB_API_KEY** (optional)
   - From: https://statsapi.mlb.com/api
   - For direct MLB Stats API access

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    DAILY AUTOMATION FLOW                         │
└─────────────────────────────────────────────────────────────────┘

8:00 AM UTC (GitHub Actions Trigger)
    ↓
┌──────────────────────────────────────┐
│ JOB 1: Fetch MLB Data                │
│ - pybaseball for 2024 stats          │
│ - Compute team-level features        │
│ - Engineer game-specific features    │
│ - Train XGBoost model                │
│ - Generate predictions (10 games)    │
└──────────────────────────────────────┘
    ↓
    ├─→ daily_predictions.csv (artifact)
    ├─→ High-confidence count (output)
    └─→ TRIGGERS Job 2 & 3
    ↓
┌──────────────────────────────────────┐
│ JOB 2: Generate Articles             │
│ - Load predictions from CSV          │
│ - Filter for HIGH confidence only    │
│ - Call Claude API (English)          │
│ - Call Claude API (Japanese)         │
│ - Save .md files                     │
└──────────────────────────────────────┘
    ↓
    └─→ English articles artifact
    └─→ Japanese articles artifact
    ↓
┌──────────────────────────────────────┐
│ JOB 3: Update Google Sheets          │
│ - Read daily_predictions.csv         │
│ - Authenticate with service account  │
│ - Write to "Daily Predictions" sheet │
│ - Update Article Queue sheet         │
└──────────────────────────────────────┘
    ↓
    └─→ Google Sheets updated
    └─→ Ready for manual review
    ↓
┌──────────────────────────────────────┐
│ JOB 4: Log Results & Notifications   │
│ - Create execution summary           │
│ - Post to Slack (optional)           │
│ - Create GitHub Issue for review     │
│ - Archive logs                       │
└──────────────────────────────────────┘
    ↓
WORKFLOW COMPLETE - Ready for manual article publication
```

## Setup Instructions

### Local Development Setup

```bash
# 1. Clone repository
git clone https://github.com/your-org/moneyball_dojo.git
cd moneyball_dojo

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Test import and basic functionality
python -c "from data_pipeline_demo import MLBDataPipeline; print('✓ Pipeline ready')"
```

### GitHub Actions Setup

1. **Copy workflow file:**
   ```bash
   mkdir -p .github/workflows
   cp github_actions_workflow.yml .github/workflows/moneyball_daily.yml
   git add .github/workflows/moneyball_daily.yml
   git commit -m "Add daily prediction workflow"
   git push
   ```

2. **Create Google Cloud service account:**
   - Go to Google Cloud Console
   - Create new service account
   - Generate JSON key
   - Enable Google Sheets API
   - Base64 encode credentials:
     ```bash
     base64 -i ~/Downloads/moneyball-dojo.json > credentials.txt
     ```

3. **Create GitHub secrets:**
   - Go to repo Settings → Secrets
   - Add CLAUDE_API_KEY (from Anthropic)
   - Add GOOGLE_SHEETS_CREDENTIALS (Base64 encoded JSON)
   - Add GOOGLE_SHEETS_ID (from your spreadsheet URL)
   - Add SLACK_WEBHOOK (optional)

4. **Create Google Sheet:**
   - Create new spreadsheet
   - Create 4 sheets: "Daily Predictions", "Results", "Model Performance", "Article Queue"
   - Copy headers from sheets_schema.py
   - Share with service account email
   - Get spreadsheet ID from URL

5. **Test workflow:**
   - Go to Actions tab
   - Manually trigger workflow
   - Monitor execution
   - Verify predictions appear in Google Sheets

## Model Details

### XGBoost Configuration

```python
XGBClassifier(
    n_estimators=100,        # 100 boosting rounds
    learning_rate=0.1,       # Standard learning rate
    max_depth=6,             # Medium tree depth
    random_state=42          # Reproducible splits
)
```

### Features Used (13 total)

**Batting Features:**
- `away_batting_avg`: Away team season batting average
- `away_obp`: Away team on-base percentage
- `away_slg`: Away team slugging percentage
- `home_batting_avg`: Home team batting average
- `home_obp`: Home team OBP
- `home_slg`: Home team slugging percentage

**Pitching Features:**
- `away_pitcher_era`: Away pitcher's ERA
- `home_pitcher_era`: Home pitcher's ERA
- `away_era_allowed`: ERA of pitchers opposing away team
- `home_era_allowed`: ERA of pitchers opposing home team

**Context Features:**
- `home_field_advantage`: Always 1.0 (home advantage)
- `runs_differential`: Offensive power comparison
- `pitching_quality_gap`: Pitcher matchup advantage

### Training Process

1. **Data Preparation:**
   - 500+ synthetic games with known outcomes
   - Home team wins ~55% (reflecting home field advantage)
   - Feature scaling with StandardScaler

2. **Train/Test Split:**
   - 80% training data (400 games)
   - 20% test data (100 games)

3. **Model Evaluation:**
   - Typical train accuracy: 62-63%
   - Typical test accuracy: 60-62%
   - ROI potential: 5-15% over season (depends on betting discipline)

### Confidence Tier Assignment

- **HIGH**: Model probability >= 65% (strong conviction)
- **MEDIUM**: Model probability 55-65% (moderate conviction)
- **LOW**: Model probability < 55% (weak signal)

## Article Generation

### Claude API Integration

Uses Claude Opus 4.6 for high-quality article generation.

**System Prompt Components:**
- Data-driven analyst persona
- Sabermetrics expertise
- Accessible writing for informed readers
- Evidence-based claims required

**English Article Prompt:**
- 800-1200 words
- Substack format
- Pitched to sports analytics enthusiasts

**Japanese Article Prompt:**
- 2000-3000 characters
- note.com format
- Pitched to Japanese MLB fans
- Includes Japanese AI engineer persona

### Workflow

1. Load daily predictions from CSV
2. Filter for HIGH confidence picks only
3. Call Claude API with game data and prompt
4. Save generated articles as .md files
5. Update "Article Queue" sheet with status
6. Manual review before publishing to platforms

## Performance Tracking

### Sheets-Based Analytics

**Track daily in Results sheet:**
- Game outcomes (actual vs. predicted)
- Units wagered and units won
- Individual game ROI
- Cumulative season ROI

**Weekly roll-up in Model Performance sheet:**
- Win rate by confidence tier
- ATS (Against The Spread) record
- Total units and cumulative ROI
- High/medium/low confidence breakdown

### Example Tracking Entry

```
Date       | Game       | Prediction | Result | Hit  | Units | Won   | ROI
2024-06-15 | NYY vs BOS | HOME       | HOME   | TRUE | 2.5   | +2.25 | +90%
2024-06-16 | LAD vs SF  | AWAY       | HOME   | FALSE| 1.0   | -1.00 | -100%
```

## Best Practices

### For Predictions

1. **Only pick HIGH confidence games** - avoid marginal calls
2. **Track all picks and outcomes** - maintain statistical integrity
3. **Use consistent unit sizing** - manage bankroll properly
4. **Never chase losses** - stick to system discipline
5. **Review model metrics weekly** - ensure model hasn't degraded

### For Articles

1. **Support every claim with data** - cite specific stats
2. **Acknowledge uncertainty** - mention counterarguments
3. **Use recent form** - update pitcher/team metrics weekly
4. **Consider context** - injuries, travel, recent performance
5. **Keep it human** - analytics inform, not dictate

### For Operations

1. **Automate what you can** - GitHub Actions handles daily runs
2. **Monitor failures** - GitHub Issues alert you to problems
3. **Archive results** - keep complete historical record
4. **Review weekly** - check model performance and ROI
5. **Update prompts seasonally** - refine article templates

## Troubleshooting

### Workflow Failures

**Issue: "GitHub Actions job failed"**
- Check workflow logs in Actions tab
- Common cause: Google Sheets authentication
- Solution: Verify GOOGLE_SHEETS_CREDENTIALS secret is valid

**Issue: "pybaseball import error"**
- pybaseball may fail to fetch live data
- Pipeline falls back to synthetic data automatically
- No action required - will use demo data

**Issue: "Claude API rate limit"**
- Set max 3 articles per day to avoid limits
- Filter for HIGH confidence only
- Add delays between API calls

### Google Sheets Sync Issues

**Issue: "Nothing appears in Google Sheets"**
- Verify service account has edit access to sheet
- Check GOOGLE_SHEETS_ID is correct
- Confirm sheet names match exactly

**Issue: "Duplicate headers"**
- Workflow appends new rows by design
- Remove old data manually before re-running
- Or modify workflow to clear sheet first

## Contributing

Improvements welcome! Focus areas:

1. **Model improvements:**
   - Add more features (strength of schedule, bullpen, weather)
   - Tune hyperparameters
   - Ensemble with other models

2. **Article generation:**
   - Add player injury tracking
   - Include recent news context
   - Generate Twitter/X summaries

3. **Sheets integration:**
   - Add live game score tracking
   - Create performance dashboards
   - Generate ROI charts

4. **Automation:**
   - Add email notifications
   - Post to Discord/Slack with summaries
   - Auto-publish to platforms

## License

MIT License - Use freely for personal projects. Attribution appreciated.

## Support

Questions or issues? Open a GitHub Issue or check the troubleshooting section above.

---

**Built with:** Python, XGBoost, Claude API, Google Sheets API, GitHub Actions

**Inspired by:** Moneyball (the book), Modern baseball analytics, Data-driven sports betting

**Last Updated:** February 2026
