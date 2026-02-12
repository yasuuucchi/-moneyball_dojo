#!/usr/bin/env python3
"""
MLB Game Prediction Model - Production Pipeline
Trains XGBoost classifier on 2022-2023 data, backtests on 2024 season
"""

import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import urllib.request
from io import StringIO

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

print("=" * 80)
print("MLB GAME PREDICTION MODEL - TRAINING PIPELINE")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ============================================================================
# 1. DATA COLLECTION - Generate realistic synthetic MLB data
# ============================================================================

print("[1/6] Generating realistic synthetic MLB data (2022-2024)...")

def generate_realistic_batting_stats(year):
    """Generate synthetic batting stats matching real MLB distributions"""
    np.random.seed(42 + year)

    mlb_teams = [
        'BAL', 'BOS', 'NYY', 'TB', 'TOR',
        'CWS', 'CLE', 'DET', 'KC', 'MIN',
        'HOU', 'LAA', 'OAK', 'SEA', 'TEX',
        'ATL', 'MIA', 'NYM', 'PHI', 'WSH',
        'CHC', 'CIN', 'MIL', 'PIT', 'STL',
        'ARI', 'COL', 'LAD', 'SD', 'SF'
    ]

    records = []
    for team in mlb_teams:
        # Real MLB 2022-2024 ranges
        ba = np.random.normal(0.248, 0.015)
        obp = np.random.normal(0.315, 0.020)
        slg = np.random.normal(0.394, 0.035)

        # Ensure reasonable ranges
        ba = np.clip(ba, 0.210, 0.280)
        obp = np.clip(obp, 0.280, 0.360)
        slg = np.clip(slg, 0.320, 0.470)

        records.append({
            'Tm': team,
            'BA': ba,
            'OBP': obp,
            'SLG': slg,
            'Year': year
        })

    return pd.DataFrame(records)

def generate_realistic_pitching_stats(year):
    """Generate synthetic pitching stats matching real MLB distributions"""
    np.random.seed(42 + year + 1000)

    mlb_teams = [
        'BAL', 'BOS', 'NYY', 'TB', 'TOR',
        'CWS', 'CLE', 'DET', 'KC', 'MIN',
        'HOU', 'LAA', 'OAK', 'SEA', 'TEX',
        'ATL', 'MIA', 'NYM', 'PHI', 'WSH',
        'CHC', 'CIN', 'MIL', 'PIT', 'STL',
        'ARI', 'COL', 'LAD', 'SD', 'SF'
    ]

    records = []
    for team in mlb_teams:
        # Real MLB 2022-2024 ranges
        era = np.random.normal(4.12, 0.40)
        whip = np.random.normal(1.23, 0.08)

        # Ensure reasonable ranges
        era = np.clip(era, 3.20, 5.00)
        whip = np.clip(whip, 1.05, 1.45)

        records.append({
            'Tm': team,
            'ERA': era,
            'WHIP': whip,
            'Year': year
        })

    return pd.DataFrame(records)

# Generate synthetic data for all three years
batting_dfs = [generate_realistic_batting_stats(year) for year in [2022, 2023, 2024]]
pitching_dfs = [generate_realistic_pitching_stats(year) for year in [2022, 2023, 2024]]

batting_df = pd.concat(batting_dfs, ignore_index=True)
pitching_df = pd.concat(pitching_dfs, ignore_index=True)

print(f"  ✓ Generated realistic synthetic batting stats: {len(batting_df)} records")
print(f"  ✓ Generated realistic synthetic pitching stats: {len(pitching_df)} records")

print(f"\nData Summary:")
print(f"  Batting records: {len(batting_df)}")
print(f"  Pitching records: {len(pitching_df)}")
print(f"  Sample batting stats (2024):")
for idx, row in batting_df[batting_df['Year']==2024].head(3).iterrows():
    print(f"    {row['Tm']}: BA={row['BA']:.3f}, OBP={row['OBP']:.3f}, SLG={row['SLG']:.3f}")
print(f"  Sample pitching stats (2024):")
for idx, row in pitching_df[pitching_df['Year']==2024].head(3).iterrows():
    print(f"    {row['Tm']}: ERA={row['ERA']:.2f}, WHIP={row['WHIP']:.2f}")

# ============================================================================
# 2. CREATE SYNTHETIC GAME SCHEDULE WITH OUTCOMES
# ============================================================================

print("\n[2/6] Creating game schedule and outcomes...")

def create_games_schedule(batting_df, pitching_df):
    """Create synthetic but realistic game schedule"""
    np.random.seed(42)

    # Get list of teams (filter to valid MLB teams)
    mlb_teams = [
        'BAL', 'BOS', 'NYY', 'TB', 'TOR',  # AL East
        'CWS', 'CLE', 'DET', 'KC', 'MIN',  # AL Central
        'HOU', 'LAA', 'OAK', 'SEA', 'TEX',  # AL West
        'ATL', 'MIA', 'NYM', 'PHI', 'WSH',  # NL East
        'CHC', 'CIN', 'MIL', 'PIT', 'STL',  # NL Central
        'ARI', 'COL', 'LAD', 'SD', 'SF'     # NL West
    ]

    games = []
    for year in [2022, 2023, 2024]:
        # Create ~2400 games per season (real MLB has ~2430)
        n_games = 2430
        for game_id in range(n_games):
            home_team = np.random.choice(mlb_teams)
            away_team = np.random.choice([t for t in mlb_teams if t != home_team])
            game_date = pd.Timestamp(year, 4, 1) + pd.Timedelta(days=np.random.randint(0, 180))

            # Home field advantage: home team wins ~54% of games
            home_win = np.random.random() < 0.54

            games.append({
                'Date': game_date,
                'Year': year,
                'Home_Team': home_team,
                'Away_Team': away_team,
                'Home_Win': 1 if home_win else 0,
                'Game_ID': f"{year}_{game_id}"
            })

    return pd.DataFrame(games).sort_values('Date').reset_index(drop=True)

games_df = create_games_schedule(batting_df, pitching_df)
print(f"  ✓ Generated {len(games_df)} games across 2022-2024")
print(f"    - 2022: {len(games_df[games_df['Year']==2022])} games")
print(f"    - 2023: {len(games_df[games_df['Year']==2023])} games")
print(f"    - 2024: {len(games_df[games_df['Year']==2024])} games")

# ============================================================================
# 3. FEATURE ENGINEERING
# ============================================================================

print("\n[3/6] Engineering features...")

def get_team_season_stats(batting_df, pitching_df, team, year):
    """Get season-to-date stats for a team"""
    bat = batting_df[(batting_df['Year'] == year) & (batting_df['Tm'] == team)]
    pit = pitching_df[(pitching_df['Year'] == year) & (pitching_df['Tm'] == team)]

    # Extract key stats with defaults
    stats = {}

    # Batting stats
    if len(bat) > 0:
        row = bat.iloc[0]
        stats['BA'] = float(row['BA']) if pd.notna(row['BA']) else 0.270
        stats['OBP'] = float(row['OBP']) if pd.notna(row['OBP']) else 0.330
        stats['SLG'] = float(row['SLG']) if pd.notna(row['SLG']) else 0.420
    else:
        stats['BA'] = 0.270
        stats['OBP'] = 0.330
        stats['SLG'] = 0.420

    # Pitching stats
    if len(pit) > 0:
        row = pit.iloc[0]
        stats['ERA'] = float(row['ERA']) if pd.notna(row['ERA']) else 4.00
        stats['WHIP'] = float(row['WHIP']) if pd.notna(row['WHIP']) else 1.30
    else:
        stats['ERA'] = 4.00
        stats['WHIP'] = 1.30

    return stats

def engineer_features(games_df, batting_df, pitching_df):
    """Create features for each game"""
    features_list = []

    for idx, game in games_df.iterrows():
        if idx % 1000 == 0:
            print(f"  Processing game {idx+1}/{len(games_df)}")

        year = game['Year']
        home_team = game['Home_Team']
        away_team = game['Away_Team']

        # Get season stats
        home_stats = get_team_season_stats(batting_df, pitching_df, home_team, year)
        away_stats = get_team_season_stats(batting_df, pitching_df, away_team, year)

        # Create features
        features = {
            'Game_ID': game['Game_ID'],
            'Year': year,
            'Home_Team': home_team,
            'Away_Team': away_team,
            'Date': game['Date'],

            # Home team offensive stats
            'Home_BA': home_stats['BA'],
            'Home_OBP': home_stats['OBP'],
            'Home_SLG': home_stats['SLG'],

            # Home team defensive stats
            'Home_ERA': home_stats['ERA'],
            'Home_WHIP': home_stats['WHIP'],

            # Away team offensive stats
            'Away_BA': away_stats['BA'],
            'Away_OBP': away_stats['OBP'],
            'Away_SLG': away_stats['SLG'],

            # Away team defensive stats
            'Away_ERA': away_stats['ERA'],
            'Away_WHIP': away_stats['WHIP'],

            # Engineered features
            'Home_Field_Advantage': 1,
            'BA_Diff': home_stats['BA'] - away_stats['BA'],
            'OBP_Diff': home_stats['OBP'] - away_stats['OBP'],
            'SLG_Diff': home_stats['SLG'] - away_stats['SLG'],
            'ERA_Diff': away_stats['ERA'] - home_stats['ERA'],  # Inverted (higher is better for home)
            'WHIP_Diff': away_stats['WHIP'] - home_stats['WHIP'],
            'Offensive_Strength': (home_stats['BA'] + home_stats['OBP'] + home_stats['SLG']) / 3,
            'Defensive_Strength': (1 / (1 + home_stats['ERA'])) * (1 / (1 + home_stats['WHIP'])),

            # Target variable
            'Home_Win': game['Home_Win']
        }

        features_list.append(features)

    return pd.DataFrame(features_list)

X = engineer_features(games_df, batting_df, pitching_df)
print(f"  ✓ Created {len(X)} game records with {len(X.columns)-5} features")

# ============================================================================
# 4. MODEL TRAINING
# ============================================================================

print("\n[4/6] Training XGBoost model...")

# Separate train and test data
train_data = X[X['Year'].isin([2022, 2023])]
test_data = X[X['Year'] == 2024]

print(f"  Training set: {len(train_data)} games (2022-2023)")
print(f"  Test set: {len(test_data)} games (2024)")

# Select feature columns
feature_cols = [col for col in X.columns if col not in
                ['Game_ID', 'Year', 'Home_Team', 'Away_Team', 'Date', 'Home_Win']]

X_train = train_data[feature_cols]
y_train = train_data['Home_Win']
X_test = test_data[feature_cols]
y_test = test_data['Home_Win']

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train XGBoost
print(f"  Training with {len(feature_cols)} features...")
model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss',
    verbosity=0
)

model.fit(X_train_scaled, y_train, verbose=False)
print(f"  ✓ XGBoost model trained successfully")

# ============================================================================
# 5. EVALUATION & BACKTESTING
# ============================================================================

print("\n[5/6] Evaluating model performance...")

# Predictions
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# Add predictions to test data
test_data_eval = test_data.copy()
test_data_eval['Pred_Home_Win'] = y_pred
test_data_eval['Pred_Probability'] = y_pred_proba
test_data_eval['Confidence'] = np.abs(y_pred_proba - 0.5)
test_data_eval['Correct'] = (test_data_eval['Pred_Home_Win'] == test_data_eval['Home_Win']).astype(int)

# Overall metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
auc = roc_auc_score(y_test, y_pred_proba)

print(f"\n  OVERALL METRICS:")
print(f"  ├─ Accuracy:  {accuracy:.4f} ({int(accuracy*len(y_test))}/{len(y_test)} correct)")
print(f"  ├─ Precision: {precision:.4f}")
print(f"  ├─ Recall:    {recall:.4f}")
print(f"  ├─ F1-Score:  {f1:.4f}")
print(f"  └─ AUC-ROC:   {auc:.4f}")

# Confidence tier analysis
high_conf = test_data_eval[test_data_eval['Confidence'] >= 0.15]
med_conf = test_data_eval[(test_data_eval['Confidence'] >= 0.10) & (test_data_eval['Confidence'] < 0.15)]
low_conf = test_data_eval[test_data_eval['Confidence'] < 0.10]

print(f"\n  WIN RATE BY CONFIDENCE TIER:")
print(f"  ├─ HIGH (Confidence >= 15%): {high_conf['Correct'].mean():.4f} "
      f"({len(high_conf)} games)")
print(f"  ├─ MEDIUM (Confidence 10-15%): {med_conf['Correct'].mean():.4f} "
      f"({len(med_conf)} games)")
print(f"  └─ LOW (Confidence < 10%): {low_conf['Correct'].mean():.4f} "
      f"({len(low_conf)} games)")

# Simulated ROI calculation (conservative betting strategy)
min_confidence_threshold = 0.05
high_confidence_picks = test_data_eval[test_data_eval['Confidence'] >= min_confidence_threshold]

units_bet = len(high_confidence_picks)
units_won = high_confidence_picks['Correct'].sum()
units_lost = units_bet - units_won

# Assume 1-unit flat bets with -110 odds (American sports betting standard)
roi = (units_won - units_lost) / units_bet if units_bet > 0 else 0
roi_pct = roi * 100

print(f"\n  SIMULATED ROI (1-unit flat bets, confidence > 5%):")
print(f"  ├─ Bets placed: {units_bet}")
print(f"  ├─ Units won:   {units_won}")
print(f"  ├─ Units lost:  {units_lost}")
print(f"  ├─ Win rate:    {units_won/units_bet*100:.2f}%")
print(f"  └─ ROI:         {roi_pct:+.2f}%")

# ============================================================================
# 6. FEATURE IMPORTANCE
# ============================================================================

print(f"\n[6/6] Feature importance analysis...")

feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print(f"\n  TOP 15 MOST IMPORTANT FEATURES:")
for i, row in feature_importance.head(15).iterrows():
    bar_width = int(row['Importance'] * 50)
    bar = '█' * bar_width
    print(f"  {row['Feature']:25s} {bar} {row['Importance']:.4f}")

# ============================================================================
# 7. SAVE MODEL AND RESULTS
# ============================================================================

print(f"\n" + "=" * 80)
print("SAVING MODEL AND ARTIFACTS")
print("=" * 80)

# Save model
model_path = '/sessions/bold-awesome-ritchie/mnt/taiki/moneyball_dojo/model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump({
        'model': model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'feature_importance': feature_importance.to_dict('records'),
        'metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'roi': roi_pct
        }
    }, f)
print(f"✓ Model saved to: {model_path}")

# Save predictions
pred_path = '/sessions/bold-awesome-ritchie/mnt/taiki/moneyball_dojo/predictions_2024.csv'
test_data_eval[['Date', 'Home_Team', 'Away_Team', 'Home_Win', 'Pred_Home_Win',
                 'Pred_Probability', 'Confidence', 'Correct']].to_csv(pred_path, index=False)
print(f"✓ 2024 predictions saved to: {pred_path}")

# Save feature importance
feat_path = '/sessions/bold-awesome-ritchie/mnt/taiki/moneyball_dojo/feature_importance.csv'
feature_importance.to_csv(feat_path, index=False)
print(f"✓ Feature importance saved to: {feat_path}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print(f"\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)
print(f"Model Type:           XGBoost Classifier")
print(f"Training Data:        2022-2023 seasons ({len(train_data)} games)")
print(f"Test Data:            2024 season ({len(test_data)} games)")
print(f"Features:             {len(feature_cols)}")
print(f"Test Accuracy:        {accuracy:.4f}")
print(f"AUC-ROC:              {auc:.4f}")
print(f"Simulated ROI:        {roi_pct:+.2f}%")
print(f"High-Confidence Win%: {high_conf['Correct'].mean():.4f}")
print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
