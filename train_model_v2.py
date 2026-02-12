#!/usr/bin/env python3
"""
Moneyball Dojo — 予測モデル v2（実データ版）
=============================================
fetch_real_data.py で取得した実データでXGBoostモデルを構築する。

改善点（v1からの変更）:
1. 合成データ → 実MLB試合結果
2. チーム実力が結果に反映される特徴量
3. ローリング（直近N試合）成績の導入
4. ホーム/アウェー別成績
5. 得失点差（Run Differential）追加
6. ハイパーパラメータチューニング
7. クロスバリデーション

使い方:
  python3 fetch_real_data.py  # まずデータ取得
  python3 train_model_v2.py   # モデル学習

出力:
  model.pkl               — 学習済みモデル
  feature_importance.csv  — 特徴量重要度
  predictions_2024.csv    — 2024年バックテスト結果
"""

import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from datetime import datetime
import pickle
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, log_loss)

PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"

print("=" * 80)
print("MONEYBALL DOJO — MODEL TRAINING v2 (REAL DATA)")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ============================================================================
# 1. LOAD REAL DATA
# ============================================================================

print("[1/7] Loading real MLB data...")

games_path = DATA_DIR / "games_2022_2024.csv"
if not games_path.exists():
    print("❌ data/games_2022_2024.csv not found!")
    print("   Run fetch_real_data.py first.")
    sys.exit(1)

games_df = pd.read_csv(games_path)
games_df['date'] = pd.to_datetime(games_df['date'])
print(f"  ✓ Loaded {len(games_df)} games")

# Load team stats for each year
team_stats = {}
for year in [2022, 2023, 2024]:
    path = DATA_DIR / f"team_stats_{year}.csv"
    if path.exists():
        team_stats[year] = pd.read_csv(path)
        print(f"  ✓ Loaded team stats {year}: {len(team_stats[year])} teams")
    else:
        print(f"  ⚠ team_stats_{year}.csv not found, will compute from games")

# Load standings
standings = {}
for year in [2022, 2023, 2024]:
    path = DATA_DIR / f"standings_{year}.csv"
    if path.exists():
        standings[year] = pd.read_csv(path)

# ============================================================================
# 2. COMPUTE TEAM METRICS FROM GAME RESULTS
# ============================================================================

print("\n[2/7] Computing team metrics from game results...")

def compute_team_cumulative_stats(games_df, team_col, is_home=True):
    """各チームの累積成績を計算"""
    result_col = 'home_win' if is_home else 'away_win'

    if 'away_win' not in games_df.columns:
        games_df['away_win'] = 1 - games_df['home_win']

    records = []
    teams = games_df[team_col].unique()

    for team in teams:
        team_games = games_df[games_df[team_col] == team].sort_values('date')

        for year in [2022, 2023, 2024]:
            year_games = team_games[team_games['year'] == year]
            if len(year_games) == 0:
                continue

            score_col = 'home_score' if is_home else 'away_score'
            opp_score_col = 'away_score' if is_home else 'home_score'

            records.append({
                'team': team,
                'year': year,
                'location': 'home' if is_home else 'away',
                'games': len(year_games),
                'wins': int(year_games[result_col].sum()),
                'losses': len(year_games) - int(year_games[result_col].sum()),
                'runs_scored': int(year_games[score_col].sum()),
                'runs_allowed': int(year_games[opp_score_col].sum()),
                'avg_runs_scored': year_games[score_col].mean(),
                'avg_runs_allowed': year_games[opp_score_col].mean(),
                'run_diff': (year_games[score_col] - year_games[opp_score_col]).mean(),
                'win_pct': year_games[result_col].mean(),
            })

    return pd.DataFrame(records)

# Home and away stats
home_stats = compute_team_cumulative_stats(games_df, 'home_team', is_home=True)
away_stats = compute_team_cumulative_stats(games_df, 'away_team', is_home=False)
print(f"  ✓ Computed home stats: {len(home_stats)} records")
print(f"  ✓ Computed away stats: {len(away_stats)} records")

# Overall team stats per year
def compute_overall_team_stats(games_df):
    """チーム全体の年次成績"""
    records = []
    for year in [2022, 2023, 2024]:
        year_games = games_df[games_df['year'] == year]
        all_teams = set(year_games['home_team'].unique()) | set(year_games['away_team'].unique())

        for team in all_teams:
            home_g = year_games[year_games['home_team'] == team]
            away_g = year_games[year_games['away_team'] == team]

            home_wins = int(home_g['home_win'].sum())
            away_wins = len(away_g) - int(away_g['home_win'].sum())  # away_win = 1 - home_win
            total_wins = home_wins + away_wins
            total_games = len(home_g) + len(away_g)

            home_rs = home_g['home_score'].sum()
            away_rs = away_g['away_score'].sum()
            home_ra = home_g['away_score'].sum()
            away_ra = away_g['home_score'].sum()

            total_rs = home_rs + away_rs
            total_ra = home_ra + away_ra

            records.append({
                'team': team,
                'year': year,
                'total_games': total_games,
                'total_wins': total_wins,
                'total_losses': total_games - total_wins,
                'win_pct': total_wins / total_games if total_games > 0 else 0.5,
                'total_runs_scored': total_rs,
                'total_runs_allowed': total_ra,
                'run_diff_per_game': (total_rs - total_ra) / total_games if total_games > 0 else 0,
                'avg_runs_scored': total_rs / total_games if total_games > 0 else 0,
                'avg_runs_allowed': total_ra / total_games if total_games > 0 else 0,
                # Pythagorean expectation (Bill James)
                'pythag_win_pct': total_rs**1.83 / (total_rs**1.83 + total_ra**1.83) if (total_rs + total_ra) > 0 else 0.5,
            })

    return pd.DataFrame(records)

overall_stats = compute_overall_team_stats(games_df)
print(f"  ✓ Computed overall stats: {len(overall_stats)} team-years")

# ============================================================================
# 3. ROLLING STATS (直近N試合の成績)
# ============================================================================

print("\n[3/7] Computing rolling (recent form) stats...")

def compute_rolling_stats(games_df, window=15):
    """各試合時点での直近N試合の成績を計算"""
    # Sort by date
    games_sorted = games_df.sort_values('date').reset_index(drop=True)

    # For each team, compute rolling stats
    team_rolling = {}

    all_teams = set(games_sorted['home_team'].unique()) | set(games_sorted['away_team'].unique())

    for team in all_teams:
        # Get all games involving this team, in order
        mask = (games_sorted['home_team'] == team) | (games_sorted['away_team'] == team)
        team_games = games_sorted[mask].copy()

        # Determine win/loss for this team
        team_games['team_win'] = np.where(
            team_games['home_team'] == team,
            team_games['home_win'],
            1 - team_games['home_win']
        )
        team_games['team_runs'] = np.where(
            team_games['home_team'] == team,
            team_games['home_score'],
            team_games['away_score']
        )
        team_games['opp_runs'] = np.where(
            team_games['home_team'] == team,
            team_games['away_score'],
            team_games['home_score']
        )

        # Rolling calculations
        team_games[f'rolling_{window}_win_pct'] = (
            team_games['team_win'].rolling(window, min_periods=5).mean()
        )
        team_games[f'rolling_{window}_rs'] = (
            team_games['team_runs'].rolling(window, min_periods=5).mean()
        )
        team_games[f'rolling_{window}_ra'] = (
            team_games['opp_runs'].rolling(window, min_periods=5).mean()
        )
        team_games[f'rolling_{window}_run_diff'] = (
            (team_games['team_runs'] - team_games['opp_runs']).rolling(window, min_periods=5).mean()
        )

        # Store with game_id as key
        for _, row in team_games.iterrows():
            game_id = row['game_id']
            if game_id not in team_rolling:
                team_rolling[game_id] = {}
            team_rolling[game_id][team] = {
                f'rolling_win_pct': row.get(f'rolling_{window}_win_pct', np.nan),
                f'rolling_rs': row.get(f'rolling_{window}_rs', np.nan),
                f'rolling_ra': row.get(f'rolling_{window}_ra', np.nan),
                f'rolling_run_diff': row.get(f'rolling_{window}_run_diff', np.nan),
            }

    print(f"  ✓ Computed rolling-{window} stats for {len(all_teams)} teams")
    return team_rolling

rolling_stats = compute_rolling_stats(games_df, window=15)

# ============================================================================
# 4. FEATURE ENGINEERING
# ============================================================================

print("\n[4/7] Engineering features...")

def get_team_year_stats(team, year, overall_stats_df, team_stats_dict):
    """チームの年次スタッツを取得"""
    # From overall computed stats
    mask = (overall_stats_df['team'] == team) & (overall_stats_df['year'] == year)
    row = overall_stats_df[mask]

    stats = {}
    if len(row) > 0:
        r = row.iloc[0]
        stats['win_pct'] = r['win_pct']
        stats['avg_rs'] = r['avg_runs_scored']
        stats['avg_ra'] = r['avg_runs_allowed']
        stats['run_diff'] = r['run_diff_per_game']
        stats['pythag'] = r['pythag_win_pct']
    else:
        stats['win_pct'] = 0.5
        stats['avg_rs'] = 4.5
        stats['avg_ra'] = 4.5
        stats['run_diff'] = 0
        stats['pythag'] = 0.5

    # From API team stats (BA, OBP, SLG, ERA, WHIP)
    if year in team_stats_dict:
        ts = team_stats_dict[year]
        team_row = ts[ts['team_name'] == team]
        if len(team_row) > 0:
            r = team_row.iloc[0]
            stats['BA'] = r.get('BA', 0.248)
            stats['OBP'] = r.get('OBP', 0.315)
            stats['SLG'] = r.get('SLG', 0.395)
            stats['ERA'] = r.get('ERA', 4.12)
            stats['WHIP'] = r.get('WHIP', 1.28)
            stats['HR'] = r.get('HR', 0)
            stats['SB'] = r.get('SB', 0)
        else:
            stats.update({'BA': 0.248, 'OBP': 0.315, 'SLG': 0.395,
                          'ERA': 4.12, 'WHIP': 1.28, 'HR': 0, 'SB': 0})
    else:
        stats.update({'BA': 0.248, 'OBP': 0.315, 'SLG': 0.395,
                      'ERA': 4.12, 'WHIP': 1.28, 'HR': 0, 'SB': 0})

    return stats

def get_home_away_stats(team, year, home_stats_df, away_stats_df):
    """ホーム/アウェー別成績"""
    h = home_stats_df[(home_stats_df['team'] == team) & (home_stats_df['year'] == year)]
    a = away_stats_df[(away_stats_df['team'] == team) & (away_stats_df['year'] == year)]

    result = {}
    if len(h) > 0:
        result['home_win_pct'] = h.iloc[0]['win_pct']
        result['home_avg_rs'] = h.iloc[0]['avg_runs_scored']
        result['home_avg_ra'] = h.iloc[0]['avg_runs_allowed']
    else:
        result['home_win_pct'] = 0.54
        result['home_avg_rs'] = 4.5
        result['home_avg_ra'] = 4.5

    if len(a) > 0:
        result['away_win_pct'] = a.iloc[0]['win_pct']
        result['away_avg_rs'] = a.iloc[0]['avg_runs_scored']
        result['away_avg_ra'] = a.iloc[0]['avg_runs_allowed']
    else:
        result['away_win_pct'] = 0.46
        result['away_avg_rs'] = 4.5
        result['away_avg_ra'] = 4.5

    return result

# Build feature matrix
features_list = []
skipped = 0

for idx, game in games_df.iterrows():
    if idx % 1000 == 0:
        print(f"  Processing game {idx+1}/{len(games_df)}...")

    year = game['year']
    home = game['home_team']
    away = game['away_team']
    game_id = game['game_id']

    # Season stats
    home_season = get_team_year_stats(home, year, overall_stats, team_stats)
    away_season = get_team_year_stats(away, year, overall_stats, team_stats)

    # Home/Away splits
    home_splits = get_home_away_stats(home, year, home_stats, away_stats)
    away_splits = get_home_away_stats(away, year, home_stats, away_stats)

    # Rolling stats
    rolling = rolling_stats.get(game_id, {})
    home_rolling = rolling.get(home, {})
    away_rolling = rolling.get(away, {})

    # Skip if missing rolling data (early season)
    if not home_rolling or not away_rolling:
        skipped += 1
        continue
    if pd.isna(home_rolling.get('rolling_win_pct')) or pd.isna(away_rolling.get('rolling_win_pct')):
        skipped += 1
        continue

    features = {
        'game_id': game_id,
        'date': game['date'],
        'year': year,
        'home_team': home,
        'away_team': away,

        # === Season-level features ===
        # Win percentage differential
        'win_pct_diff': home_season['win_pct'] - away_season['win_pct'],

        # Pythagorean expectation differential
        'pythag_diff': home_season['pythag'] - away_season['pythag'],

        # Run differential
        'home_run_diff': home_season['run_diff'],
        'away_run_diff': away_season['run_diff'],
        'run_diff_diff': home_season['run_diff'] - away_season['run_diff'],

        # Batting metrics
        'home_BA': home_season['BA'],
        'away_BA': away_season['BA'],
        'BA_diff': home_season['BA'] - away_season['BA'],

        'home_OBP': home_season['OBP'],
        'away_OBP': away_season['OBP'],
        'OBP_diff': home_season['OBP'] - away_season['OBP'],

        'home_SLG': home_season['SLG'],
        'away_SLG': away_season['SLG'],
        'SLG_diff': home_season['SLG'] - away_season['SLG'],

        # Pitching metrics
        'home_ERA': home_season['ERA'],
        'away_ERA': away_season['ERA'],
        'ERA_diff': away_season['ERA'] - home_season['ERA'],  # Positive = home team advantage

        'home_WHIP': home_season['WHIP'],
        'away_WHIP': away_season['WHIP'],
        'WHIP_diff': away_season['WHIP'] - home_season['WHIP'],

        # === Home/Away split features ===
        'home_team_home_wpct': home_splits['home_win_pct'],
        'away_team_away_wpct': away_splits['away_win_pct'],
        'home_away_split_diff': home_splits['home_win_pct'] - away_splits['away_win_pct'],

        # === Rolling (recent form) features ===
        'home_rolling_wpct': home_rolling.get('rolling_win_pct', 0.5),
        'away_rolling_wpct': away_rolling.get('rolling_win_pct', 0.5),
        'rolling_wpct_diff': home_rolling.get('rolling_win_pct', 0.5) - away_rolling.get('rolling_win_pct', 0.5),

        'home_rolling_rs': home_rolling.get('rolling_rs', 4.5),
        'away_rolling_rs': away_rolling.get('rolling_rs', 4.5),

        'home_rolling_run_diff': home_rolling.get('rolling_run_diff', 0),
        'away_rolling_run_diff': away_rolling.get('rolling_run_diff', 0),
        'rolling_run_diff_diff': home_rolling.get('rolling_run_diff', 0) - away_rolling.get('rolling_run_diff', 0),

        # === Composite features ===
        'home_offensive_strength': (home_season['BA'] + home_season['OBP'] + home_season['SLG']) / 3,
        'away_offensive_strength': (away_season['BA'] + away_season['OBP'] + away_season['SLG']) / 3,
        'home_defensive_strength': 1 / (1 + home_season['ERA']) * 1 / (1 + home_season['WHIP']),
        'away_defensive_strength': 1 / (1 + away_season['ERA']) * 1 / (1 + away_season['WHIP']),

        'offensive_diff': ((home_season['BA'] + home_season['OBP'] + home_season['SLG']) -
                          (away_season['BA'] + away_season['OBP'] + away_season['SLG'])) / 3,
        'defensive_diff': (1/(1+home_season['ERA']) * 1/(1+home_season['WHIP'])) - \
                          (1/(1+away_season['ERA']) * 1/(1+away_season['WHIP'])),

        # === Target ===
        'home_win': game['home_win'],
    }

    features_list.append(features)

X = pd.DataFrame(features_list)
print(f"  ✓ Created {len(X)} game records with features")
print(f"    Skipped {skipped} games (missing rolling data)")

# ============================================================================
# 5. MODEL TRAINING
# ============================================================================

print("\n[5/7] Training XGBoost model...")

# Train/test split: 2022-2023 for training, 2024 for testing
train_data = X[X['year'].isin([2022, 2023])].copy()
test_data = X[X['year'] == 2024].copy()

print(f"  Training set: {len(train_data)} games (2022-2023)")
print(f"  Test set: {len(test_data)} games (2024)")

# Feature columns (exclude metadata and target)
meta_cols = ['game_id', 'date', 'year', 'home_team', 'away_team', 'home_win']
feature_cols = [col for col in X.columns if col not in meta_cols]
print(f"  Features: {len(feature_cols)}")

X_train = train_data[feature_cols].astype(float)
y_train = train_data['home_win'].astype(int)
X_test = test_data[feature_cols].astype(float)
y_test = test_data['home_win'].astype(int)

# Scale features
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_cols, index=X_test.index)

# Cross-validation on training set
print("  Running 5-fold cross-validation on training set...")
model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.7,
    min_child_weight=3,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    eval_metric='logloss',
    verbosity=0
)

cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Train final model
model.fit(X_train_scaled, y_train, verbose=False)
print(f"  ✓ XGBoost model trained")

# ============================================================================
# 6. EVALUATION
# ============================================================================

print("\n[6/7] Evaluating model performance...")

y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
auc = roc_auc_score(y_test, y_pred_proba)
logloss = log_loss(y_test, y_pred_proba)

print(f"\n  OVERALL METRICS:")
print(f"  ├─ Accuracy:  {accuracy:.4f} ({int(accuracy*len(y_test))}/{len(y_test)})")
print(f"  ├─ Precision: {precision:.4f}")
print(f"  ├─ Recall:    {recall:.4f}")
print(f"  ├─ F1-Score:  {f1:.4f}")
print(f"  ├─ AUC-ROC:   {auc:.4f}")
print(f"  └─ Log Loss:  {logloss:.4f}")

# Confidence-tier analysis
test_eval = test_data.copy()
test_eval['pred_prob'] = y_pred_proba
test_eval['pred_win'] = y_pred
test_eval['correct'] = (y_pred == y_test.values).astype(int)
test_eval['confidence'] = np.abs(y_pred_proba - 0.5)

# Tiers
strong = test_eval[test_eval['confidence'] >= 0.08]
moderate = test_eval[(test_eval['confidence'] >= 0.04) & (test_eval['confidence'] < 0.08)]
lean = test_eval[(test_eval['confidence'] >= 0.01) & (test_eval['confidence'] < 0.04)]
pass_tier = test_eval[test_eval['confidence'] < 0.01]

print(f"\n  ACCURACY BY CONFIDENCE TIER:")
if len(strong) > 0:
    print(f"  ├─ STRONG  (8%+):  {strong['correct'].mean():.4f} ({len(strong)} games)")
if len(moderate) > 0:
    print(f"  ├─ MODERATE(4-8%): {moderate['correct'].mean():.4f} ({len(moderate)} games)")
if len(lean) > 0:
    print(f"  ├─ LEAN   (1-4%):  {lean['correct'].mean():.4f} ({len(lean)} games)")
if len(pass_tier) > 0:
    print(f"  └─ PASS   (<1%):   {pass_tier['correct'].mean():.4f} ({len(pass_tier)} games)")

# ROI simulation (-110 odds = need 52.38% to break even)
actionable = test_eval[test_eval['confidence'] >= 0.04]  # STRONG + MODERATE
if len(actionable) > 0:
    wins = actionable['correct'].sum()
    losses = len(actionable) - wins
    # -110 odds: win $100 on $110 bet, lose $110
    profit = wins * 100 - losses * 110
    roi = profit / (len(actionable) * 110) * 100
    print(f"\n  SIMULATED ROI (STRONG + MODERATE picks, -110 odds):")
    print(f"  ├─ Bets: {len(actionable)}")
    print(f"  ├─ Win rate: {wins/len(actionable)*100:.2f}%")
    print(f"  └─ ROI: {roi:+.2f}%")

# ============================================================================
# 7. FEATURE IMPORTANCE & SAVE
# ============================================================================

print(f"\n[7/7] Feature importance & saving model...")

feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print(f"\n  TOP 15 FEATURES:")
for _, row in feature_importance.head(15).iterrows():
    bar = '█' * int(row['Importance'] * 100)
    print(f"  {row['Feature']:30s} {bar} {row['Importance']:.4f}")

# Save model
model_path = PROJECT_DIR / 'model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump({
        'model': model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'feature_names': feature_cols,
        'n_features': len(feature_cols),
        'feature_importance': feature_importance.to_dict('records'),
        'model_type': 'XGBoost v2',
        'training_date': datetime.now().isoformat(),
        'train_years': [2022, 2023],
        'test_year': 2024,
        'test_accuracy': accuracy,
        'test_auc': auc,
        'metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'logloss': logloss,
            'cv_accuracy_mean': cv_scores.mean(),
            'cv_accuracy_std': cv_scores.std(),
        }
    }, f)
print(f"\n✓ Model saved → {model_path}")

# Save predictions
pred_path = PROJECT_DIR / 'predictions_2024.csv'
test_eval[['date', 'home_team', 'away_team', 'home_win', 'pred_win',
           'pred_prob', 'confidence', 'correct']].to_csv(pred_path, index=False)
print(f"✓ Predictions saved → {pred_path}")

# Save feature importance
feat_path = PROJECT_DIR / 'feature_importance.csv'
feature_importance.to_csv(feat_path, index=False)
print(f"✓ Feature importance saved → {feat_path}")

# Final summary
print(f"\n{'='*80}")
print("FINAL SUMMARY")
print(f"{'='*80}")
print(f"Model:                XGBoost v2 (Real Data)")
print(f"Training:             2022-2023 ({len(train_data)} games)")
print(f"Test:                 2024 ({len(test_data)} games)")
print(f"Features:             {len(feature_cols)}")
print(f"CV Accuracy:          {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
print(f"Test Accuracy:        {accuracy:.4f}")
print(f"AUC-ROC:              {auc:.4f}")
if len(actionable) > 0:
    print(f"Actionable picks ROI: {roi:+.2f}%")
print(f"\nTop 3 features: {', '.join(feature_importance.head(3)['Feature'].tolist())}")
print(f"{'='*80}")
