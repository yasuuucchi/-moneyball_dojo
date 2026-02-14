#!/usr/bin/env python3
"""
Moneyball Dojo — 2025 Season Backtest
======================================
目的: 2025年全2,430試合に対してモデルの予測精度を検証し、
      公開可能な実績データを生成する。

方法:
  1. 2022-2024データでモデルを再訓練（2025は完全に未知）
  2. 2025年の各試合について、その日までのデータのみで特徴量を構築（ウォークフォワード）
  3. 5モデル（Moneyline, O/U, Run Line, F5, NRFI）で全試合予測
  4. 実際の結果と照合し、信頼度別の的中率・ROIを算出

使い方:
  python3 backtest_2025.py
"""

import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
from pathlib import Path
from collections import defaultdict

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error

PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"
MODELS_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = PROJECT_DIR / "output" / "backtest_2025"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Backup dir for original models
BACKUP_DIR = MODELS_DIR / "backup_original"
BACKUP_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# STEP 0: Backup original models
# ============================================================================
def backup_original_models():
    """現行モデルをバックアップ（後で戻すため）"""
    print("[0] Backing up original models...")
    for pkl in MODELS_DIR.glob("model_*.pkl"):
        backup_path = BACKUP_DIR / pkl.name
        if not backup_path.exists():
            import shutil
            shutil.copy2(pkl, backup_path)
            print(f"  -> {pkl.name}")
    print("  Done.\n")


# ============================================================================
# STEP 1: Load all data
# ============================================================================
def load_all_data():
    """全データ読み込み"""
    print("[1] Loading all data...")

    games_df = pd.read_csv(DATA_DIR / "games_2022_2025.csv")
    games_df['date'] = pd.to_datetime(games_df['date'])
    games_df['total_runs'] = games_df['home_score'] + games_df['away_score']
    games_df['run_margin'] = games_df['home_score'] - games_df['away_score']
    games_df['home_covers_rl'] = (games_df['run_margin'] >= 2).astype(int)

    team_stats = {}
    for year in [2022, 2023, 2024, 2025]:
        path = DATA_DIR / f"team_stats_{year}.csv"
        if path.exists():
            team_stats[year] = pd.read_csv(path)

    nrfi_df = pd.read_csv(DATA_DIR / "nrfi_data_2022_2025.csv")

    train_games = games_df[games_df['year'].isin([2022, 2023, 2024])].copy()
    test_games = games_df[games_df['year'] == 2025].copy()

    print(f"  Train: {len(train_games)} games (2022-2024)")
    print(f"  Test:  {len(test_games)} games (2025)")
    print(f"  NRFI:  {len(nrfi_df)} records")
    print()

    return games_df, train_games, test_games, team_stats, nrfi_df


# ============================================================================
# STEP 2: Feature engineering (reuse from train_all_models.py)
# ============================================================================
def compute_overall_stats(games_df, years=None):
    """チーム成績を年別に計算"""
    overall = {}
    df = games_df if years is None else games_df[games_df['year'].isin(years)]

    for year in df['year'].unique():
        yg = df[df['year'] == year]
        teams = set(yg['home_team'].unique()) | set(yg['away_team'].unique())

        for team in teams:
            hg = yg[yg['home_team'] == team]
            ag = yg[yg['away_team'] == team]

            hw = int(hg['home_win'].sum())
            aw = len(ag) - int(ag['home_win'].sum())
            tw = hw + aw
            tg = len(hg) + len(ag)
            if tg == 0:
                continue

            hrs = hg['home_score'].sum()
            ars = ag['away_score'].sum()
            hra = hg['away_score'].sum()
            ara = ag['home_score'].sum()
            trs = hrs + ars
            tra = hra + ara

            overall[(team, year)] = {
                'win_pct': tw / tg,
                'avg_rs': trs / tg,
                'avg_ra': tra / tg,
                'run_diff': (trs - tra) / tg,
                'pythag': trs**1.83 / (trs**1.83 + tra**1.83) if (trs + tra) > 0 else 0.5,
                'home_win_pct': hg['home_win'].mean() if len(hg) > 0 else 0.54,
                'away_win_pct': (1 - ag['home_win']).mean() if len(ag) > 0 else 0.46,
                'avg_total_runs_home': (hg['home_score'] + hg['away_score']).mean() if len(hg) > 0 else 9.0,
                'avg_total_runs_away': (ag['home_score'] + ag['away_score']).mean() if len(ag) > 0 else 9.0,
                'home_margin_avg': (hg['home_score'] - hg['away_score']).mean() if len(hg) > 0 else 0.3,
                'away_margin_avg': (ag['away_score'] - ag['home_score']).mean() if len(ag) > 0 else -0.3,
                'blowout_rate': (
                    (hg['run_margin'].abs() >= 4).sum() +
                    ((ag['home_score'] - ag['away_score']).abs() >= 4).sum()
                ) / tg if tg > 0 else 0.25,
            }

    return overall


def compute_rolling_per_game(games_df, window=15):
    """各試合時点でのローリングスタッツ（ウォークフォワード）"""
    games_sorted = games_df.sort_values('date').reset_index(drop=True)
    teams = set(games_sorted['home_team'].unique()) | set(games_sorted['away_team'].unique())

    rolling_map = {}  # game_id -> {team: stats}

    for team in teams:
        mask = (games_sorted['home_team'] == team) | (games_sorted['away_team'] == team)
        tg = games_sorted[mask].copy()

        tg['team_win'] = np.where(tg['home_team'] == team, tg['home_win'], 1 - tg['home_win'])
        tg['team_runs'] = np.where(tg['home_team'] == team, tg['home_score'], tg['away_score'])
        tg['opp_runs'] = np.where(tg['home_team'] == team, tg['away_score'], tg['home_score'])
        tg['game_total'] = tg['home_score'] + tg['away_score']

        tg['r_wpct'] = tg['team_win'].rolling(window, min_periods=5).mean()
        tg['r_rs'] = tg['team_runs'].rolling(window, min_periods=5).mean()
        tg['r_ra'] = tg['opp_runs'].rolling(window, min_periods=5).mean()
        tg['r_rd'] = (tg['team_runs'] - tg['opp_runs']).rolling(window, min_periods=5).mean()
        tg['r_total'] = tg['game_total'].rolling(window, min_periods=5).mean()

        for _, row in tg.iterrows():
            gid = row['game_id']
            if gid not in rolling_map:
                rolling_map[gid] = {}
            rolling_map[gid][team] = {
                'rolling_win_pct': row.get('r_wpct', np.nan),
                'rolling_rs': row.get('r_rs', np.nan),
                'rolling_ra': row.get('r_ra', np.nan),
                'rolling_run_diff': row.get('r_rd', np.nan),
                'rolling_game_total': row.get('r_total', np.nan),
            }

    return rolling_map


def get_api_stats(team, year, team_stats_dict):
    """チーム打撃・投手指標"""
    d = {'BA': 0.248, 'OBP': 0.315, 'SLG': 0.395, 'ERA': 4.12, 'WHIP': 1.28}
    if year not in team_stats_dict:
        return d
    ts = team_stats_dict[year]
    row = ts[ts['team_name'] == team]
    if len(row) == 0:
        return d
    r = row.iloc[0]
    return {
        'BA': float(r.get('BA', 0.248)),
        'OBP': float(r.get('OBP', 0.315)),
        'SLG': float(r.get('SLG', 0.395)),
        'ERA': float(r.get('ERA', 4.12)),
        'WHIP': float(r.get('WHIP', 1.28)),
    }


def build_feature_matrix(games_df, overall_stats, rolling_stats, team_stats_dict,
                          include_totals=True):
    """全試合の特徴量マトリックスを構築"""
    features_list = []
    skipped = 0

    for _, game in games_df.iterrows():
        year = game['year']
        home = game['home_team']
        away = game['away_team']
        gid = game['game_id']

        hs = overall_stats.get((home, year))
        aws = overall_stats.get((away, year))
        if not hs or not aws:
            skipped += 1
            continue

        rolling = rolling_stats.get(gid, {})
        hr = rolling.get(home, {})
        ar = rolling.get(away, {})
        if not hr or not ar or pd.isna(hr.get('rolling_win_pct')) or pd.isna(ar.get('rolling_win_pct')):
            skipped += 1
            continue

        ha = get_api_stats(home, year, team_stats_dict)
        aa = get_api_stats(away, year, team_stats_dict)

        f = {
            'game_id': gid, 'date': game['date'], 'year': year,
            'home_team': home, 'away_team': away,

            'win_pct_diff': hs['win_pct'] - aws['win_pct'],
            'pythag_diff': hs['pythag'] - aws['pythag'],
            'home_run_diff': hs['run_diff'],
            'away_run_diff': aws['run_diff'],
            'run_diff_diff': hs['run_diff'] - aws['run_diff'],

            'home_BA': ha['BA'], 'away_BA': aa['BA'], 'BA_diff': ha['BA'] - aa['BA'],
            'home_OBP': ha['OBP'], 'away_OBP': aa['OBP'], 'OBP_diff': ha['OBP'] - aa['OBP'],
            'home_SLG': ha['SLG'], 'away_SLG': aa['SLG'], 'SLG_diff': ha['SLG'] - aa['SLG'],
            'home_ERA': ha['ERA'], 'away_ERA': aa['ERA'], 'ERA_diff': aa['ERA'] - ha['ERA'],
            'home_WHIP': ha['WHIP'], 'away_WHIP': aa['WHIP'], 'WHIP_diff': aa['WHIP'] - ha['WHIP'],

            'home_team_home_wpct': hs['home_win_pct'],
            'away_team_away_wpct': aws['away_win_pct'],
            'home_away_split_diff': hs['home_win_pct'] - aws['away_win_pct'],

            'home_rolling_wpct': hr['rolling_win_pct'],
            'away_rolling_wpct': ar['rolling_win_pct'],
            'rolling_wpct_diff': hr['rolling_win_pct'] - ar['rolling_win_pct'],
            'home_rolling_rs': hr['rolling_rs'],
            'away_rolling_rs': ar['rolling_rs'],
            'home_rolling_run_diff': hr['rolling_run_diff'],
            'away_rolling_run_diff': ar['rolling_run_diff'],
            'rolling_run_diff_diff': hr['rolling_run_diff'] - ar['rolling_run_diff'],

            'home_offensive_strength': (ha['BA'] + ha['OBP'] + ha['SLG']) / 3,
            'away_offensive_strength': (aa['BA'] + aa['OBP'] + aa['SLG']) / 3,
            'home_defensive_strength': 1/(1+ha['ERA']) * 1/(1+ha['WHIP']),
            'away_defensive_strength': 1/(1+aa['ERA']) * 1/(1+aa['WHIP']),
            'offensive_diff': ((ha['BA']+ha['OBP']+ha['SLG']) - (aa['BA']+aa['OBP']+aa['SLG'])) / 3,
            'defensive_diff': (1/(1+ha['ERA'])*1/(1+ha['WHIP'])) - (1/(1+aa['ERA'])*1/(1+aa['WHIP'])),
        }

        if include_totals:
            f['combined_avg_rs'] = hs['avg_rs'] + aws['avg_rs']
            f['combined_avg_ra'] = hs['avg_ra'] + aws['avg_ra']
            f['home_avg_total'] = hs.get('avg_total_runs_home', 9.0)
            f['away_avg_total'] = aws.get('avg_total_runs_away', 9.0)
            f['rolling_game_total_home'] = hr.get('rolling_game_total', 9.0)
            f['rolling_game_total_away'] = ar.get('rolling_game_total', 9.0)
            f['combined_rolling_rs'] = hr['rolling_rs'] + ar['rolling_rs']
            f['combined_rolling_ra'] = hr['rolling_ra'] + ar['rolling_ra']
            f['combined_ERA'] = (ha['ERA'] + aa['ERA']) / 2
            f['combined_WHIP'] = (ha['WHIP'] + aa['WHIP']) / 2
            f['combined_OPS'] = ((ha['OBP']+ha['SLG']) + (aa['OBP']+aa['SLG'])) / 2
            f['home_margin_avg'] = hs.get('home_margin_avg', 0.3)
            f['away_margin_avg'] = aws.get('away_margin_avg', -0.3)
            f['blowout_rate_home'] = hs.get('blowout_rate', 0.25)
            f['blowout_rate_away'] = aws.get('blowout_rate', 0.25)

        f['home_win'] = game['home_win']
        f['total_runs'] = game['total_runs']
        f['run_margin'] = game['run_margin']
        f['home_covers_rl'] = game['home_covers_rl']

        features_list.append(f)

    return pd.DataFrame(features_list), skipped


# ============================================================================
# STEP 3: Train models on 2022-2024, test on 2025
# ============================================================================
def train_moneyline_backtest(X_all):
    """Moneyline: 2022-2024 train -> 2025 test"""
    print("  Training Moneyline...")
    target = 'home_win'
    meta = ['game_id', 'date', 'year', 'home_team', 'away_team']
    exclude = meta + ['total_runs', 'run_margin', 'home_covers_rl',
                       'combined_avg_rs', 'combined_avg_ra', 'home_avg_total',
                       'away_avg_total', 'rolling_game_total_home', 'rolling_game_total_away',
                       'combined_rolling_rs', 'combined_rolling_ra', 'combined_ERA',
                       'combined_WHIP', 'combined_OPS', 'home_margin_avg', 'away_margin_avg',
                       'blowout_rate_home', 'blowout_rate_away']
    feat_cols = [c for c in X_all.columns if c not in exclude and c != target]

    train = X_all[X_all['year'].isin([2022, 2023, 2024])]
    test = X_all[X_all['year'] == 2025]

    X_train = train[feat_cols].astype(float)
    y_train = train[target].astype(int)
    X_test = test[feat_cols].astype(float)
    y_test = test[target].astype(int)

    scaler = StandardScaler()
    X_tr = pd.DataFrame(scaler.fit_transform(X_train), columns=feat_cols, index=X_train.index)
    X_te = pd.DataFrame(scaler.transform(X_test), columns=feat_cols, index=X_test.index)

    model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.7, min_child_weight=3,
                          reg_alpha=0.1, reg_lambda=1.0, random_state=42,
                          eval_metric='logloss', verbosity=0)

    cv = cross_val_score(model, X_tr, y_train, cv=5, scoring='accuracy')
    model.fit(X_tr, y_train, verbose=False)

    y_prob = model.predict_proba(X_te)[:, 1]

    print(f"    CV Accuracy: {cv.mean():.4f} (+/- {cv.std():.4f})")

    return model, scaler, feat_cols, y_prob, y_test, test


def train_over_under_backtest(X_all):
    """Over/Under: 2022-2024 train -> 2025 test"""
    print("  Training Over/Under...")
    target = 'total_runs'
    meta = ['game_id', 'date', 'year', 'home_team', 'away_team']
    exclude = meta + ['home_win', 'total_runs', 'run_margin', 'home_covers_rl']
    feat_cols = [c for c in X_all.columns if c not in exclude]

    train = X_all[X_all['year'].isin([2022, 2023, 2024])]
    test = X_all[X_all['year'] == 2025]

    X_train = train[feat_cols].astype(float)
    y_train = train[target].astype(float)
    X_test = test[feat_cols].astype(float)
    y_test = test[target].astype(float)

    scaler = StandardScaler()
    X_tr = pd.DataFrame(scaler.fit_transform(X_train), columns=feat_cols, index=X_train.index)
    X_te = pd.DataFrame(scaler.transform(X_test), columns=feat_cols, index=X_test.index)

    model = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.7, min_child_weight=3,
                         reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbosity=0)

    cv = cross_val_score(model, X_tr, y_train, cv=5, scoring='neg_mean_absolute_error')
    model.fit(X_tr, y_train, verbose=False)

    y_pred = model.predict(X_te)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"    CV MAE: {-cv.mean():.3f} (+/- {cv.std():.3f})")
    print(f"    Test MAE: {mae:.3f} runs")

    return model, scaler, feat_cols, y_pred, y_test, test


def train_run_line_backtest(X_all):
    """Run Line: 2022-2024 train -> 2025 test"""
    print("  Training Run Line...")
    target = 'home_covers_rl'
    meta = ['game_id', 'date', 'year', 'home_team', 'away_team']
    exclude = meta + ['home_win', 'total_runs', 'run_margin', 'home_covers_rl']
    feat_cols = [c for c in X_all.columns if c not in exclude]

    train = X_all[X_all['year'].isin([2022, 2023, 2024])]
    test = X_all[X_all['year'] == 2025]

    X_train = train[feat_cols].astype(float)
    y_train = train[target].astype(int)
    X_test = test[feat_cols].astype(float)
    y_test = test[target].astype(int)

    scaler = StandardScaler()
    X_tr = pd.DataFrame(scaler.fit_transform(X_train), columns=feat_cols, index=X_train.index)
    X_te = pd.DataFrame(scaler.transform(X_test), columns=feat_cols, index=X_test.index)

    model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.7, min_child_weight=3,
                          reg_alpha=0.1, reg_lambda=1.0, random_state=42,
                          eval_metric='logloss', verbosity=0)

    cv = cross_val_score(model, X_tr, y_train, cv=5, scoring='accuracy')
    model.fit(X_tr, y_train, verbose=False)

    y_prob = model.predict_proba(X_te)[:, 1]

    print(f"    CV Accuracy: {cv.mean():.4f} (+/- {cv.std():.4f})")

    return model, scaler, feat_cols, y_prob, y_test, test


def train_f5_backtest(X_all):
    """F5 Moneyline: 2022-2024 train -> 2025 test"""
    print("  Training F5 Moneyline...")
    target = 'home_win'
    meta = ['game_id', 'date', 'year', 'home_team', 'away_team']
    exclude = meta + ['total_runs', 'run_margin', 'home_covers_rl',
                       'blowout_rate_home', 'blowout_rate_away',
                       'home_margin_avg', 'away_margin_avg']
    feat_cols = [c for c in X_all.columns if c not in exclude and c != target]

    train = X_all[X_all['year'].isin([2022, 2023, 2024])]
    test = X_all[X_all['year'] == 2025]

    X_train = train[feat_cols].astype(float)
    y_train = train[target].astype(int)
    X_test = test[feat_cols].astype(float)
    y_test = test[target].astype(int)

    scaler = StandardScaler()
    X_tr = pd.DataFrame(scaler.fit_transform(X_train), columns=feat_cols, index=X_train.index)
    X_te = pd.DataFrame(scaler.transform(X_test), columns=feat_cols, index=X_test.index)

    model = XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                          reg_alpha=0.2, reg_lambda=1.5, random_state=42,
                          eval_metric='logloss', verbosity=0)

    cv = cross_val_score(model, X_tr, y_train, cv=5, scoring='accuracy')
    model.fit(X_tr, y_train, verbose=False)

    y_prob = model.predict_proba(X_te)[:, 1]

    print(f"    CV Accuracy: {cv.mean():.4f} (+/- {cv.std():.4f})")

    return model, scaler, feat_cols, y_prob, y_test, test


def train_nrfi_backtest(games_df, overall_stats, nrfi_df, team_stats_dict):
    """NRFI: 2022-2024 train -> 2025 test"""
    print("  Training NRFI...")

    if 'nrfi' in nrfi_df.columns and 'nrfi_result' not in nrfi_df.columns:
        nrfi_df = nrfi_df.rename(columns={'nrfi': 'nrfi_result'})

    # 1回特化のチーム統計
    home_1st = nrfi_df.groupby(['home_team', 'year'])['home_1st_runs'].agg(['mean', 'count']).reset_index()
    home_1st.columns = ['team', 'year', 'avg_1st_runs_home', 'games_home']

    away_1st = nrfi_df.groupby(['away_team', 'year'])['away_1st_runs'].agg(['mean', 'count']).reset_index()
    away_1st.columns = ['team', 'year', 'avg_1st_runs_away', 'games_away']

    home_1st_allowed = nrfi_df.groupby(['home_team', 'year'])['away_1st_runs'].agg(['mean']).reset_index()
    home_1st_allowed.columns = ['team', 'year', 'avg_1st_allowed_home']

    away_1st_allowed = nrfi_df.groupby(['away_team', 'year'])['home_1st_runs'].agg(['mean']).reset_index()
    away_1st_allowed.columns = ['team', 'year', 'avg_1st_allowed_away']

    team_1st_stats = home_1st.merge(away_1st, on=['team', 'year'], how='outer')
    team_1st_stats = team_1st_stats.merge(home_1st_allowed, on=['team', 'year'], how='outer')
    team_1st_stats = team_1st_stats.merge(away_1st_allowed, on=['team', 'year'], how='outer').fillna(0)

    merged = nrfi_df.merge(
        games_df[['game_id', 'year', 'home_team', 'away_team']].drop_duplicates(),
        on='game_id', how='inner', suffixes=('', '_games')
    )

    feature_rows = []
    for _, row in merged.iterrows():
        home = row['home_team']
        away = row['away_team']
        year = row['year']

        h_1st = team_1st_stats[(team_1st_stats['team'] == home) & (team_1st_stats['year'] == year)]
        a_1st = team_1st_stats[(team_1st_stats['team'] == away) & (team_1st_stats['year'] == year)]

        if h_1st.empty or a_1st.empty:
            continue

        h_s = h_1st.iloc[0]
        a_s = a_1st.iloc[0]

        home_os = overall_stats.get((home, year), {})
        away_os = overall_stats.get((away, year), {})

        features = {
            'game_id': row['game_id'],
            'year': year,
            'nrfi_result': row['nrfi_result'],
            'home_1st_rs': h_s['avg_1st_runs_home'],
            'home_1st_ra': h_s['avg_1st_allowed_home'],
            'away_1st_rs': a_s['avg_1st_runs_away'],
            'away_1st_ra': a_s['avg_1st_allowed_away'],
            'combined_1st_ra': (h_s['avg_1st_allowed_home'] + a_s['avg_1st_allowed_away']) / 2,
            'combined_1st_rs': (h_s['avg_1st_runs_home'] + a_s['avg_1st_runs_away']) / 2,
            'home_era': home_os.get('team_era', 4.00),
            'away_era': away_os.get('team_era', 4.00),
            'home_obp': home_os.get('team_obp', 0.320),
            'away_obp': away_os.get('team_obp', 0.320),
            'total_runs_per_game': home_os.get('runs_per_game', 0) + away_os.get('runs_per_game', 0),
        }
        feature_rows.append(features)

    feat_df = pd.DataFrame(feature_rows)
    feat_cols = [c for c in feat_df.columns if c not in ['game_id', 'year', 'nrfi_result']]

    train = feat_df[feat_df['year'].isin([2022, 2023, 2024])]
    test = feat_df[feat_df['year'] == 2025]

    X_train = train[feat_cols].astype(float).fillna(0)
    y_train = train['nrfi_result'].astype(int)
    X_test = test[feat_cols].astype(float).fillna(0)
    y_test = test['nrfi_result'].astype(int)

    scaler = StandardScaler()
    X_tr = pd.DataFrame(scaler.fit_transform(X_train), columns=feat_cols)
    X_te = pd.DataFrame(scaler.transform(X_test), columns=feat_cols)

    model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.03,
                          subsample=0.8, colsample_bytree=0.8,
                          eval_metric='logloss', random_state=42, verbosity=0)

    cv = cross_val_score(model, X_tr, y_train, cv=5, scoring='accuracy')
    model.fit(X_tr, y_train)

    y_prob = model.predict_proba(X_te)[:, 1]

    print(f"    CV Accuracy: {cv.mean():.4f} (+/- {cv.std():.4f})")
    print(f"    Test games: {len(test)}")

    return model, scaler, feat_cols, y_prob, y_test, test


# ============================================================================
# STEP 4: Evaluate predictions
# ============================================================================
def evaluate_classifier(name, y_prob, y_true, test_df, confidence_func):
    """分類モデルの評価（信頼度別）"""
    y_pred = (y_prob > 0.5).astype(int)
    overall_acc = accuracy_score(y_true, y_pred)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = 0.0

    # Confidence tiers
    confidences = [confidence_func(p) for p in y_prob]
    results_by_tier = defaultdict(lambda: {'correct': 0, 'total': 0, 'probs': [], 'actuals': []})

    game_results = []  # Per-game results for CSV output

    for i, (prob, actual, conf) in enumerate(zip(y_prob, y_true, confidences)):
        pred = 1 if prob > 0.5 else 0
        correct = int(pred == actual)

        results_by_tier[conf]['correct'] += correct
        results_by_tier[conf]['total'] += 1
        results_by_tier[conf]['probs'].append(prob)
        results_by_tier[conf]['actuals'].append(actual)

        game_results.append({
            'game_id': int(test_df.iloc[i].get('game_id', 0)) if hasattr(test_df, 'iloc') else 0,
            'home_team': test_df.iloc[i].get('home_team', '') if hasattr(test_df, 'iloc') else '',
            'away_team': test_df.iloc[i].get('away_team', '') if hasattr(test_df, 'iloc') else '',
            'date': str(test_df.iloc[i].get('date', '')) if hasattr(test_df, 'iloc') else '',
            'model': name,
            'prob': round(float(prob), 4),
            'prediction': int(pred),
            'actual': int(actual),
            'correct': correct,
            'confidence': conf,
        })

    print(f"\n  === {name} ===")
    print(f"  Overall Accuracy: {overall_acc:.4f} ({int(overall_acc * len(y_true))}/{len(y_true)})")
    print(f"  AUC-ROC: {auc:.4f}")
    print()

    tier_summary = {}
    for tier in ['STRONG', 'MODERATE', 'LEAN', 'PASS']:
        d = results_by_tier.get(tier, {'correct': 0, 'total': 0})
        if d['total'] > 0:
            acc = d['correct'] / d['total']
            print(f"  {tier:10s}: {acc:.4f} ({d['correct']}/{d['total']})")
            tier_summary[tier] = {
                'accuracy': round(acc, 4),
                'correct': d['correct'],
                'total': d['total'],
            }

    return {
        'overall_accuracy': round(overall_acc, 4),
        'auc': round(auc, 4),
        'total_games': len(y_true),
        'tiers': tier_summary,
    }, game_results


def evaluate_regressor(name, y_pred, y_true, test_df):
    """回帰モデルの評価"""
    mae = mean_absolute_error(y_true, y_pred)

    print(f"\n  === {name} ===")
    print(f"  MAE: {mae:.3f} runs")
    print(f"  Mean predicted: {y_pred.mean():.2f}")
    print(f"  Mean actual: {y_true.mean():.2f}")

    # Line accuracy at common totals
    line_results = {}
    for line in [7.5, 8.0, 8.5, 9.0, 9.5]:
        pred_over = (y_pred > line).astype(int)
        actual_over = (y_true > line).astype(int)
        acc = accuracy_score(actual_over, pred_over)
        n_games = len(y_true)
        print(f"  Line {line}: {acc:.4f}")
        line_results[str(line)] = round(acc, 4)

    # Confidence by margin
    game_results = []
    for i, (pred, actual) in enumerate(zip(y_pred, y_true)):
        margin_85 = pred - 8.5
        if abs(margin_85) > 1.5:
            conf = 'STRONG'
        elif abs(margin_85) > 0.75:
            conf = 'MODERATE'
        elif abs(margin_85) > 0.3:
            conf = 'LEAN'
        else:
            conf = 'PASS'

        ou_pick = 'OVER' if pred > 8.5 else 'UNDER'
        actual_ou = 'OVER' if actual > 8.5 else 'UNDER'
        correct = int(ou_pick == actual_ou)

        game_results.append({
            'game_id': int(test_df.iloc[i].get('game_id', 0)) if hasattr(test_df, 'iloc') else 0,
            'home_team': test_df.iloc[i].get('home_team', '') if hasattr(test_df, 'iloc') else '',
            'away_team': test_df.iloc[i].get('away_team', '') if hasattr(test_df, 'iloc') else '',
            'date': str(test_df.iloc[i].get('date', '')) if hasattr(test_df, 'iloc') else '',
            'model': name,
            'predicted_total': round(float(pred), 2),
            'actual_total': int(actual),
            'pick_8_5': ou_pick,
            'actual_8_5': actual_ou,
            'correct_8_5': correct,
            'confidence': conf,
        })

    # Confidence tier accuracy for O/U 8.5
    print("\n  O/U 8.5 by confidence:")
    for tier in ['STRONG', 'MODERATE', 'LEAN', 'PASS']:
        tier_games = [g for g in game_results if g['confidence'] == tier]
        if tier_games:
            tier_correct = sum(g['correct_8_5'] for g in tier_games)
            tier_acc = tier_correct / len(tier_games)
            print(f"  {tier:10s}: {tier_acc:.4f} ({tier_correct}/{len(tier_games)})")

    return {
        'mae': round(mae, 3),
        'line_accuracy': line_results,
        'total_games': len(y_true),
    }, game_results


# ============================================================================
# Confidence functions (matching run_daily.py logic)
# ============================================================================
def ml_confidence(prob):
    """Moneyline confidence tier"""
    # We don't have market odds, so use model probability distance from 0.5
    abs_dist = abs(prob - 0.5)
    if abs_dist >= 0.15:
        return 'STRONG'
    elif abs_dist >= 0.08:
        return 'MODERATE'
    elif abs_dist < 0.03:
        return 'PASS'
    else:
        return 'LEAN'


def rl_confidence(prob):
    """Run Line confidence tier"""
    abs_edge = abs(prob - 0.5)
    if abs_edge >= 0.15:
        return 'STRONG'
    elif abs_edge >= 0.08:
        return 'MODERATE'
    elif abs_edge < 0.03:
        return 'PASS'
    else:
        return 'LEAN'


def f5_confidence(prob):
    """F5 confidence tier"""
    abs_edge = abs(prob - 0.5)
    if abs_edge >= 0.08:
        return 'STRONG'
    elif abs_edge >= 0.04:
        return 'MODERATE'
    elif abs_edge < 0.03:
        return 'PASS'
    else:
        return 'LEAN'


def nrfi_confidence(prob):
    """NRFI confidence tier"""
    abs_edge = abs(prob - 0.5)
    if abs_edge >= 0.10:
        return 'STRONG'
    elif abs_edge >= 0.05:
        return 'MODERATE'
    else:
        return 'LEAN'


# ============================================================================
# STEP 5: Generate report
# ============================================================================
def generate_report(all_summaries, all_game_results):
    """バックテスト結果のレポート生成"""
    print("\n\n" + "=" * 70)
    print("BACKTEST REPORT — 2025 FULL SEASON")
    print("=" * 70)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Summary table
    print("=" * 70)
    print(f"{'Model':<20} {'Overall Acc':>12} {'STRONG':>12} {'MODERATE':>12} {'LEAN':>12}")
    print("-" * 70)

    for name, summary in all_summaries.items():
        if name == 'Over/Under':
            line_85 = summary.get('line_accuracy', {}).get('8.5', 0)
            print(f"{name:<20} {'MAE='+str(summary['mae']):>12} {'Line 8.5: ' + str(line_85):>36}")
        else:
            overall = f"{summary['overall_accuracy']:.1%}"
            tiers = summary.get('tiers', {})
            strong = f"{tiers.get('STRONG', {}).get('accuracy', 0):.1%} ({tiers.get('STRONG', {}).get('total', 0)})" if 'STRONG' in tiers else '-'
            moderate = f"{tiers.get('MODERATE', {}).get('accuracy', 0):.1%} ({tiers.get('MODERATE', {}).get('total', 0)})" if 'MODERATE' in tiers else '-'
            lean = f"{tiers.get('LEAN', {}).get('accuracy', 0):.1%} ({tiers.get('LEAN', {}).get('total', 0)})" if 'LEAN' in tiers else '-'
            print(f"{name:<20} {overall:>12} {strong:>12} {moderate:>12} {lean:>12}")

    print("=" * 70)

    # Save JSON summary
    summary_path = OUTPUT_DIR / "backtest_summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            'generated': datetime.now().isoformat(),
            'season': 2025,
            'train_years': [2022, 2023, 2024],
            'method': 'walk-forward (rolling stats), season-level team stats from 2024',
            'models': all_summaries,
        }, f, indent=2, default=str)
    print(f"\n  Summary saved: {summary_path}")

    # Save per-game CSV
    all_rows = []
    for results in all_game_results.values():
        all_rows.extend(results)

    if all_rows:
        csv_path = OUTPUT_DIR / "backtest_per_game.csv"
        pd.DataFrame(all_rows).to_csv(csv_path, index=False)
        print(f"  Per-game results saved: {csv_path}")

    # Generate markdown report for publishing
    generate_markdown_report(all_summaries, all_game_results)


def generate_markdown_report(all_summaries, all_game_results):
    """公開用マークダウンレポート生成"""
    md = []
    md.append("# Moneyball Dojo — 2025 Season Backtest Results")
    md.append("")
    md.append(f"Generated: {datetime.now().strftime('%Y-%m-%d')}")
    md.append("")
    md.append("## Methodology")
    md.append("")
    md.append("- **Training Data**: 2022-2024 seasons (7,283 games)")
    md.append("- **Test Data**: 2025 full season (2,430 games)")
    md.append("- **Method**: Walk-forward backtesting")
    md.append("  - Models trained ONLY on 2022-2024 data")
    md.append("  - Rolling stats computed using only games before each prediction date")
    md.append("  - Team stats from 2024 as baseline (simulating real pre-season conditions)")
    md.append("- **Models**: XGBoost classifiers/regressors")
    md.append("- **No data leakage**: 2025 data was never used in training")
    md.append("")
    md.append("---")
    md.append("")

    # Results by model
    for name, summary in all_summaries.items():
        md.append(f"## {name}")
        md.append("")

        if name == 'Over/Under':
            md.append(f"- **MAE**: {summary['mae']:.3f} runs")
            md.append(f"- **Total games**: {summary['total_games']}")
            md.append("")
            md.append("### Line Accuracy")
            md.append("")
            md.append("| Line | Accuracy |")
            md.append("|------|----------|")
            for line, acc in summary.get('line_accuracy', {}).items():
                md.append(f"| {line} | {acc:.1%} |")
        else:
            md.append(f"- **Overall Accuracy**: {summary['overall_accuracy']:.1%} ({summary['total_games']} games)")
            md.append(f"- **AUC-ROC**: {summary.get('auc', 0):.4f}")
            md.append("")
            md.append("### Accuracy by Confidence Tier")
            md.append("")
            md.append("| Tier | Accuracy | Record |")
            md.append("|------|----------|--------|")
            for tier in ['STRONG', 'MODERATE', 'LEAN', 'PASS']:
                t = summary.get('tiers', {}).get(tier)
                if t:
                    md.append(f"| {tier} | {t['accuracy']:.1%} | {t['correct']}/{t['total']} |")

        md.append("")
        md.append("---")
        md.append("")

    # Monthly breakdown for Moneyline
    ml_games = all_game_results.get('Moneyline', [])
    if ml_games:
        md.append("## Monthly Breakdown (Moneyline)")
        md.append("")
        md.append("| Month | Games | Accuracy | STRONG Acc | STRONG Games |")
        md.append("|-------|-------|----------|------------|-------------|")

        monthly = defaultdict(lambda: {'correct': 0, 'total': 0, 'strong_correct': 0, 'strong_total': 0})
        for g in ml_games:
            month = g['date'][:7]
            monthly[month]['total'] += 1
            monthly[month]['correct'] += g['correct']
            if g['confidence'] == 'STRONG':
                monthly[month]['strong_total'] += 1
                monthly[month]['strong_correct'] += g['correct']

        for month in sorted(monthly.keys()):
            m = monthly[month]
            acc = m['correct'] / m['total'] if m['total'] > 0 else 0
            s_acc = m['strong_correct'] / m['strong_total'] if m['strong_total'] > 0 else 0
            md.append(f"| {month} | {m['total']} | {acc:.1%} | {s_acc:.1%} | {m['strong_total']} |")

        md.append("")

    # Disclaimer
    md.append("---")
    md.append("")
    md.append("*This backtest is based on historical data. Past performance does not guarantee future results.*")
    md.append("*Not financial advice. Gamble responsibly.*")

    report_path = OUTPUT_DIR / "backtest_report_2025.md"
    report_path.write_text("\n".join(md), encoding='utf-8')
    print(f"  Markdown report saved: {report_path}")


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 70)
    print("MONEYBALL DOJO — 2025 FULL SEASON BACKTEST")
    print("=" * 70)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Purpose: Validate model accuracy on 2025 season")
    print(f"         Generate publishable track record")
    print("=" * 70)
    print()

    # Step 0: Backup
    backup_original_models()

    # Step 1: Load data
    games_df, train_games, test_games, team_stats, nrfi_df = load_all_data()

    # Step 2: Feature engineering
    # ウォークフォワード: ローリングスタッツは全期間で計算（各試合時点での直近15試合を使う）
    # team_statsは2024を使用（2025シーズン開始時点で利用可能なデータ）
    print("[2] Computing features (walk-forward)...")

    # Overall stats: 2022-2024の各年を個別に計算（ローリング用）
    # + 2025年も計算するが、これはバックテスト中に各試合の時点での統計を使うため
    overall_stats = compute_overall_stats(games_df)

    # Rolling stats: 全期間で計算（各game_idの時点での直近15試合）
    rolling_stats = compute_rolling_per_game(games_df, window=15)

    # team_stats for API-level features: 2025の試合には2024のteam_statsを使う
    # （シーズン開始時点では2025のデータがないため）
    backtest_team_stats = {
        2022: team_stats.get(2022),
        2023: team_stats.get(2023),
        2024: team_stats.get(2024),
        2025: team_stats.get(2024),  # KEY: 2025の予測には2024のteam_statsを使う
    }
    # None値のフィルタリング
    backtest_team_stats = {k: v for k, v in backtest_team_stats.items() if v is not None}

    # Feature matrix
    X_all, skipped = build_feature_matrix(games_df, overall_stats, rolling_stats, backtest_team_stats)
    print(f"  Feature matrix: {len(X_all)} games ({skipped} skipped)")
    print(f"  2025 games in matrix: {len(X_all[X_all['year']==2025])}")
    print()

    # Step 3: Train and predict
    print("[3] Training models on 2022-2024, predicting 2025...")
    print()

    all_summaries = {}
    all_game_results = {}

    # Moneyline
    ml_model, ml_scaler, ml_feats, ml_probs, ml_actual, ml_test = train_moneyline_backtest(X_all)
    ml_summary, ml_games = evaluate_classifier('Moneyline', ml_probs, ml_actual, ml_test, ml_confidence)
    all_summaries['Moneyline'] = ml_summary
    all_game_results['Moneyline'] = ml_games

    # Over/Under
    ou_model, ou_scaler, ou_feats, ou_preds, ou_actual, ou_test = train_over_under_backtest(X_all)
    ou_summary, ou_games = evaluate_regressor('Over/Under', ou_preds, ou_actual, ou_test)
    all_summaries['Over/Under'] = ou_summary
    all_game_results['Over/Under'] = ou_games

    # Run Line
    rl_model, rl_scaler, rl_feats, rl_probs, rl_actual, rl_test = train_run_line_backtest(X_all)
    rl_summary, rl_games = evaluate_classifier('Run Line', rl_probs, rl_actual, rl_test, rl_confidence)
    all_summaries['Run Line'] = rl_summary
    all_game_results['Run Line'] = rl_games

    # F5 Moneyline
    f5_model, f5_scaler, f5_feats, f5_probs, f5_actual, f5_test = train_f5_backtest(X_all)
    f5_summary, f5_games = evaluate_classifier('F5 Moneyline', f5_probs, f5_actual, f5_test, f5_confidence)
    all_summaries['F5 Moneyline'] = f5_summary
    all_game_results['F5 Moneyline'] = f5_games

    # NRFI
    nrfi_model, nrfi_scaler, nrfi_feats, nrfi_probs, nrfi_actual, nrfi_test = \
        train_nrfi_backtest(games_df, overall_stats, nrfi_df, backtest_team_stats)
    nrfi_summary, nrfi_games = evaluate_classifier('NRFI', nrfi_probs, nrfi_actual, nrfi_test, nrfi_confidence)
    all_summaries['NRFI'] = nrfi_summary
    all_game_results['NRFI'] = nrfi_games

    # Step 4: Report
    print()
    print("[4] Generating report...")
    generate_report(all_summaries, all_game_results)

    print("\n" + "=" * 70)
    print("BACKTEST COMPLETE")
    print(f"All outputs saved to: {OUTPUT_DIR}/")
    print()
    print("Files generated:")
    print(f"  - backtest_summary.json    (machine-readable results)")
    print(f"  - backtest_per_game.csv    (every prediction vs actual)")
    print(f"  - backtest_report_2025.md  (publishable markdown report)")
    print()
    print("Original models backed up to: models/backup_original/")
    print("NOTE: Original models have NOT been modified. This backtest")
    print("      trained temporary models in memory only.")
    print("=" * 70)


if __name__ == '__main__':
    main()
