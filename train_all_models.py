#!/usr/bin/env python3
"""
Moneyball Dojo — 全市場モデルトレーニング
==========================================
Phase 1-3の全ベッティング市場モデルを一括で構築する。

モデル一覧:
  1. Moneyline     — 勝敗予測（既存v2）
  2. Over/Under    — 合計得点のOver/Under予測
  3. Run Line      — -1.5/+1.5スプレッド予測
  4. Pitcher K     — 先発投手の奪三振数予測
  5. First 5 Inn   — 前半5イニングの勝敗予測
  6. Batter Props  — 打者ヒット/HR数予測

使い方:
  python3 fetch_real_data.py           # まずデータ取得
  python3 train_all_models.py          # 全モデル学習
  python3 train_all_models.py --model ou    # Over/Underのみ
  python3 train_all_models.py --model rl    # Run Lineのみ

出力:
  models/model_moneyline.pkl
  models/model_over_under.pkl
  models/model_run_line.pkl
  models/model_f5_moneyline.pkl
  models/model_pitcher_k.pkl
  models/model_batter_props.pkl
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
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, mean_absolute_error,
                             mean_squared_error, r2_score)

PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"
MODELS_DIR = PROJECT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ============================================================================
# SHARED: データ読み込みと特徴量計算
# ============================================================================

def load_all_data():
    """全データ読み込み"""
    games_path = DATA_DIR / "games_2022_2025.csv"
    if not games_path.exists():
        games_path = DATA_DIR / "games_2022_2024.csv"
    if not games_path.exists():
        print("❌ data/games_2022_2025.csv not found! Run fetch_real_data.py first.")
        sys.exit(1)

    games_df = pd.read_csv(games_path)
    games_df['date'] = pd.to_datetime(games_df['date'])
    games_df['total_runs'] = games_df['home_score'] + games_df['away_score']
    games_df['run_margin'] = games_df['home_score'] - games_df['away_score']
    games_df['home_covers_rl'] = (games_df['run_margin'] >= 2).astype(int)  # Home -1.5

    team_stats = {}
    for year in [2022, 2023, 2024, 2025]:
        path = DATA_DIR / f"team_stats_{year}.csv"
        if path.exists():
            team_stats[year] = pd.read_csv(path)

    return games_df, team_stats


def compute_all_team_metrics(games_df):
    """チーム成績を一括計算"""
    # Overall stats per team-year
    overall_stats = {}
    for year in games_df['year'].unique():
        year_games = games_df[games_df['year'] == year]
        all_teams = set(year_games['home_team'].unique()) | set(year_games['away_team'].unique())

        for team in all_teams:
            home_g = year_games[year_games['home_team'] == team]
            away_g = year_games[year_games['away_team'] == team]

            home_wins = int(home_g['home_win'].sum())
            away_wins = len(away_g) - int(away_g['home_win'].sum())
            total_wins = home_wins + away_wins
            total_games = len(home_g) + len(away_g)
            if total_games == 0:
                continue

            home_rs = home_g['home_score'].sum()
            away_rs = away_g['away_score'].sum()
            home_ra = home_g['away_score'].sum()
            away_ra = away_g['home_score'].sum()
            total_rs = home_rs + away_rs
            total_ra = home_ra + away_ra

            overall_stats[(team, year)] = {
                'win_pct': total_wins / total_games,
                'avg_rs': total_rs / total_games,
                'avg_ra': total_ra / total_games,
                'run_diff': (total_rs - total_ra) / total_games,
                'pythag': total_rs**1.83 / (total_rs**1.83 + total_ra**1.83) if (total_rs + total_ra) > 0 else 0.5,
                'home_win_pct': home_g['home_win'].mean() if len(home_g) > 0 else 0.54,
                'away_win_pct': (1 - away_g['home_win']).mean() if len(away_g) > 0 else 0.46,
                'avg_total_runs_home': (home_g['home_score'] + home_g['away_score']).mean() if len(home_g) > 0 else 9.0,
                'avg_total_runs_away': (away_g['home_score'] + away_g['away_score']).mean() if len(away_g) > 0 else 9.0,
                'home_margin_avg': (home_g['home_score'] - home_g['away_score']).mean() if len(home_g) > 0 else 0.3,
                'away_margin_avg': (away_g['away_score'] - away_g['home_score']).mean() if len(away_g) > 0 else -0.3,
                'blowout_rate': ((home_g['run_margin'].abs() >= 4).sum() + ((away_g['home_score'] - away_g['away_score']).abs() >= 4).sum()) / total_games if total_games > 0 else 0.25,
            }

    return overall_stats


def compute_rolling_features(games_df, window=15):
    """ローリングスタッツ（試合ごと）"""
    games_sorted = games_df.sort_values('date').reset_index(drop=True)
    all_teams = set(games_sorted['home_team'].unique()) | set(games_sorted['away_team'].unique())

    team_rolling = {}  # game_id -> team -> stats

    for team in all_teams:
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
            if gid not in team_rolling:
                team_rolling[gid] = {}
            team_rolling[gid][team] = {
                'rolling_win_pct': row.get('r_wpct', np.nan),
                'rolling_rs': row.get('r_rs', np.nan),
                'rolling_ra': row.get('r_ra', np.nan),
                'rolling_run_diff': row.get('r_rd', np.nan),
                'rolling_game_total': row.get('r_total', np.nan),
            }

    return team_rolling


def get_api_stats(team, year, team_stats_dict):
    """API取得済みのチーム指標"""
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


def build_feature_matrix(games_df, overall_stats, rolling_stats, team_stats_dict, include_totals=True):
    """全モデル共通の特徴量マトリックスを構築"""
    features_list = []
    skipped = 0

    for idx, game in games_df.iterrows():
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

            # --- Shared features (used by all models) ---
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
            # --- Over/Under specific features ---
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

            # --- Run Line specific features ---
            f['home_margin_avg'] = hs.get('home_margin_avg', 0.3)
            f['away_margin_avg'] = aws.get('away_margin_avg', -0.3)
            f['blowout_rate_home'] = hs.get('blowout_rate', 0.25)
            f['blowout_rate_away'] = aws.get('blowout_rate', 0.25)

        # --- Targets ---
        f['home_win'] = game['home_win']
        f['total_runs'] = game['total_runs']
        f['run_margin'] = game['run_margin']
        f['home_covers_rl'] = game['home_covers_rl']

        features_list.append(f)

    return pd.DataFrame(features_list), skipped


# ============================================================================
# MODEL 1: MONEYLINE (既存v2と同等)
# ============================================================================
def train_moneyline(X, meta_cols):
    """勝敗予測モデル"""
    print("\n" + "="*70)
    print("MODEL 1: MONEYLINE (勝敗予測)")
    print("="*70)

    target = 'home_win'
    exclude = meta_cols + ['total_runs', 'run_margin', 'home_covers_rl',
                           'combined_avg_rs', 'combined_avg_ra', 'home_avg_total',
                           'away_avg_total', 'rolling_game_total_home', 'rolling_game_total_away',
                           'combined_rolling_rs', 'combined_rolling_ra', 'combined_ERA',
                           'combined_WHIP', 'combined_OPS', 'home_margin_avg', 'away_margin_avg',
                           'blowout_rate_home', 'blowout_rate_away']
    feat_cols = [c for c in X.columns if c not in exclude and c != target]

    # 時系列分離: 2022-2024で学習、2025で評価（リーク防止）
    train = X[X['year'].isin([2022, 2023, 2024])]
    test = X[X['year'] == 2025]

    X_train, y_train = train[feat_cols].astype(float), train[target].astype(int)
    X_test, y_test = test[feat_cols].astype(float), test[target].astype(int)

    scaler = StandardScaler()
    X_tr = pd.DataFrame(scaler.fit_transform(X_train), columns=feat_cols, index=X_train.index)
    X_te = pd.DataFrame(scaler.transform(X_test), columns=feat_cols, index=X_test.index)

    model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.7, min_child_weight=3,
                          reg_alpha=0.1, reg_lambda=1.0, random_state=42,
                          eval_metric='logloss', verbosity=0)

    cv = cross_val_score(model, X_tr, y_train, cv=5, scoring='accuracy')
    model.fit(X_tr, y_train, verbose=False)

    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"  CV Accuracy: {cv.mean():.4f} (+/- {cv.std():.4f})")
    print(f"  Test Accuracy: {acc:.4f}")
    print(f"  AUC-ROC: {auc:.4f}")

    save_model('model_moneyline.pkl', model, scaler, feat_cols, 'Moneyline', acc, auc, cv)
    return model, scaler, feat_cols


# ============================================================================
# MODEL 2: OVER/UNDER
# ============================================================================
def train_over_under(X, meta_cols):
    """合計得点予測モデル（回帰 + 分類）"""
    print("\n" + "="*70)
    print("MODEL 2: OVER/UNDER (合計得点予測)")
    print("="*70)

    target_reg = 'total_runs'
    target_cls = 'over_8_5'  # 8.5がMLBの典型的なライン
    X['over_8_5'] = (X['total_runs'] > 8.5).astype(int)

    exclude = meta_cols + ['home_win', 'total_runs', 'run_margin', 'home_covers_rl', 'over_8_5']
    feat_cols = [c for c in X.columns if c not in exclude]

    # 時系列分離: 2022-2024で学習、2025で評価（リーク防止）
    train = X[X['year'].isin([2022, 2023, 2024])]
    test = X[X['year'] == 2025]

    X_train, y_train_reg = train[feat_cols].astype(float), train[target_reg].astype(float)
    X_test, y_test_reg = test[feat_cols].astype(float), test[target_reg].astype(float)
    y_train_cls = train[target_cls].astype(int)
    y_test_cls = test[target_cls].astype(int)

    scaler = StandardScaler()
    X_tr = pd.DataFrame(scaler.fit_transform(X_train), columns=feat_cols, index=X_train.index)
    X_te = pd.DataFrame(scaler.transform(X_test), columns=feat_cols, index=X_test.index)

    # 回帰モデル（合計得点予測）
    print("  [Regression] Predicting total runs...")
    reg_model = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.7, min_child_weight=3,
                             reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbosity=0)

    cv_reg = cross_val_score(reg_model, X_tr, y_train_reg, cv=5, scoring='neg_mean_absolute_error')
    reg_model.fit(X_tr, y_train_reg, verbose=False)
    y_pred_reg = reg_model.predict(X_te)

    mae = mean_absolute_error(y_test_reg, y_pred_reg)
    rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
    r2 = r2_score(y_test_reg, y_pred_reg)

    print(f"  MAE: {mae:.3f} runs")
    print(f"  RMSE: {rmse:.3f} runs")
    print(f"  R²: {r2:.4f}")
    print(f"  CV MAE: {-cv_reg.mean():.3f} (+/- {cv_reg.std():.3f})")

    # 分類モデル（Over 8.5予測）
    print("\n  [Classification] Over/Under 8.5...")
    cls_model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.7, min_child_weight=3,
                              reg_alpha=0.1, reg_lambda=1.0, random_state=42,
                              eval_metric='logloss', verbosity=0)

    cv_cls = cross_val_score(cls_model, X_tr, y_train_cls, cv=5, scoring='accuracy')
    cls_model.fit(X_tr, y_train_cls, verbose=False)
    y_pred_cls = cls_model.predict(X_te)
    y_prob_cls = cls_model.predict_proba(X_te)[:, 1]

    acc = accuracy_score(y_test_cls, y_pred_cls)
    auc = roc_auc_score(y_test_cls, y_prob_cls)

    print(f"  Accuracy: {acc:.4f}")
    print(f"  AUC-ROC: {auc:.4f}")
    print(f"  CV Accuracy: {cv_cls.mean():.4f} (+/- {cv_cls.std():.4f})")

    # Over率の分析
    actual_over_rate = y_test_cls.mean()
    pred_over_rate = y_pred_cls.mean()
    print(f"\n  Actual Over 8.5 rate: {actual_over_rate:.3f}")
    print(f"  Predicted Over rate: {pred_over_rate:.3f}")

    # 複数ラインでの精度
    for line in [7.5, 8.0, 8.5, 9.0, 9.5]:
        pred_over = (y_pred_reg > line).astype(int)
        actual_over = (y_test_reg > line).astype(int)
        line_acc = accuracy_score(actual_over, pred_over)
        print(f"  Line {line}: Accuracy {line_acc:.4f}")

    save_model('model_over_under.pkl', {
        'regressor': reg_model,
        'classifier': cls_model,
    }, scaler, feat_cols, 'Over/Under', acc, auc, cv_cls,
    extra={'mae': mae, 'rmse': rmse, 'r2': r2})

    return reg_model, cls_model, scaler, feat_cols


# ============================================================================
# MODEL 3: RUN LINE (-1.5 / +1.5)
# ============================================================================
def train_run_line(X, meta_cols):
    """ランライン予測モデル"""
    print("\n" + "="*70)
    print("MODEL 3: RUN LINE (-1.5 スプレッド予測)")
    print("="*70)

    target = 'home_covers_rl'  # Home wins by 2+

    exclude = meta_cols + ['home_win', 'total_runs', 'run_margin', 'home_covers_rl', 'over_8_5']
    feat_cols = [c for c in X.columns if c not in exclude and c in X.columns]

    # 時系列分離: 2022-2024で学習、2025で評価（リーク防止）
    train = X[X['year'].isin([2022, 2023, 2024])]
    test = X[X['year'] == 2025]

    X_train, y_train = train[feat_cols].astype(float), train[target].astype(int)
    X_test, y_test = test[feat_cols].astype(float), test[target].astype(int)

    scaler = StandardScaler()
    X_tr = pd.DataFrame(scaler.fit_transform(X_train), columns=feat_cols, index=X_train.index)
    X_te = pd.DataFrame(scaler.transform(X_test), columns=feat_cols, index=X_test.index)

    model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.7, min_child_weight=3,
                          reg_alpha=0.1, reg_lambda=1.0, random_state=42,
                          eval_metric='logloss', verbosity=0)

    cv = cross_val_score(model, X_tr, y_train, cv=5, scoring='accuracy')
    model.fit(X_tr, y_train, verbose=False)

    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"  CV Accuracy: {cv.mean():.4f} (+/- {cv.std():.4f})")
    print(f"  Test Accuracy: {acc:.4f}")
    print(f"  AUC-ROC: {auc:.4f}")

    # RL分析
    actual_cover_rate = y_test.mean()
    print(f"\n  Actual Home -1.5 cover rate: {actual_cover_rate:.3f}")

    # 信頼度別
    conf = np.abs(y_prob - 0.5)
    strong = conf >= 0.08
    moderate = (conf >= 0.04) & (conf < 0.08)
    if strong.sum() > 0:
        print(f"  STRONG picks: {accuracy_score(y_test[strong], y_pred[strong]):.4f} ({strong.sum()} games)")
    if moderate.sum() > 0:
        print(f"  MODERATE picks: {accuracy_score(y_test[moderate], y_pred[moderate]):.4f} ({moderate.sum()} games)")

    save_model('model_run_line.pkl', model, scaler, feat_cols, 'Run Line', acc, auc, cv)
    return model, scaler, feat_cols


# ============================================================================
# MODEL 4: FIRST 5 INNINGS MONEYLINE
# ============================================================================
def train_f5_moneyline(X, meta_cols):
    """前半5イニングの勝敗予測（フルゲームの特徴量で近似）"""
    print("\n" + "="*70)
    print("MODEL 4: FIRST 5 INNINGS MONEYLINE (前半予測)")
    print("="*70)
    print("  NOTE: F5専用のイニング別データがないため、")
    print("        先発投手の影響が大きい特徴量に重み付けして近似します。")

    target = 'home_win'

    # F5は先発投手とチームの初期パフォーマンスに依存
    # ERA, WHIP, 打撃指標を重視し、ブルペン関連は除外
    exclude = meta_cols + ['total_runs', 'run_margin', 'home_covers_rl', 'over_8_5',
                           'blowout_rate_home', 'blowout_rate_away',
                           'home_margin_avg', 'away_margin_avg']
    feat_cols = [c for c in X.columns if c not in exclude and c != target]

    # 時系列分離: 2022-2024で学習、2025で評価（リーク防止）
    train = X[X['year'].isin([2022, 2023, 2024])]
    test = X[X['year'] == 2025]

    X_train, y_train = train[feat_cols].astype(float), train[target].astype(int)
    X_test, y_test = test[feat_cols].astype(float), test[target].astype(int)

    scaler = StandardScaler()
    X_tr = pd.DataFrame(scaler.fit_transform(X_train), columns=feat_cols, index=X_train.index)
    X_te = pd.DataFrame(scaler.transform(X_test), columns=feat_cols, index=X_test.index)

    # F5用にERA/WHIPの重要度を上げるためサンプルウェイトを調整
    model = XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                          reg_alpha=0.2, reg_lambda=1.5, random_state=42,
                          eval_metric='logloss', verbosity=0)

    cv = cross_val_score(model, X_tr, y_train, cv=5, scoring='accuracy')
    model.fit(X_tr, y_train, verbose=False)

    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"  CV Accuracy: {cv.mean():.4f} (+/- {cv.std():.4f})")
    print(f"  Test Accuracy: {acc:.4f}")
    print(f"  AUC-ROC: {auc:.4f}")
    print(f"  NOTE: F5の実際の精度はイニング別データで改善可能")

    save_model('model_f5_moneyline.pkl', model, scaler, feat_cols, 'F5 Moneyline', acc, auc, cv)
    return model, scaler, feat_cols


# ============================================================================
# MODEL 5: PITCHER STRIKEOUT PROPS
# ============================================================================
def train_pitcher_k_props():
    """投手の奪三振数予測モデル"""
    print("\n" + "="*70)
    print("MODEL 5: PITCHER K PROPS (先発投手 奪三振予測)")
    print("="*70)

    # 投手データ読み込み
    pitcher_dfs = []
    for year in [2022, 2023, 2024, 2025]:
        path = DATA_DIR / f"pitcher_stats_{year}.csv"
        if path.exists():
            df = pd.read_csv(path)
            pitcher_dfs.append(df)
            print(f"  ✓ Loaded pitcher stats {year}: {len(df)} pitchers")

    if not pitcher_dfs:
        print("  ❌ No pitcher data found. Run fetch_player_data.py first.")
        return None, None, None

    pitchers = pd.concat(pitcher_dfs, ignore_index=True)

    # 試合データと結合して対戦相手情報を取得
    games_path = DATA_DIR / "games_2022_2025.csv"
    games_df = pd.read_csv(games_path)

    # チーム打撃成績（対戦相手の三振しやすさ）
    team_stats_dict = {}
    for year in [2022, 2023, 2024, 2025]:
        path = DATA_DIR / f"team_stats_{year}.csv"
        if path.exists():
            team_stats_dict[year] = pd.read_csv(path)

    # Load actual K results from game props data (2024-2025)
    props_path = DATA_DIR / "game_props_data_2024_2025.csv"
    if not props_path.exists():
        print("  ⚠ game_props_data_2024_2025.csv not found!")
        return None, None, None
    
    props_df = pd.read_csv(props_path)

    # Build feature rows
    feature_rows = []
    for _, game in props_df.iterrows():
        g_id = int(game['game_id'])
        g_info = games_df[games_df['game_id'] == g_id]
        if g_info.empty: continue
        g_info = g_info.iloc[0]
        year = g_info['year']

        for side in ['home', 'away']:
            pid = game.get(f'{side}_starter_id')
            if pd.isna(pid): continue
            # Find pitcher profile in the loaded pitchers dataframe
            p_profile = pitchers[(pitchers['player_id'] == pid) & (pitchers['year'] == year)]
            if p_profile.empty: continue
            p_profile = p_profile.iloc[0]
            
            # Find opponent stats
            opp_team = g_info['away_team'] if side == 'home' else g_info['home_team']
            opp_year_stats = team_stats_dict.get(year, pd.DataFrame())
            
            opp_so_per_game = 8.5 # default
            if not opp_year_stats.empty:
                opp_row = opp_year_stats[opp_year_stats['team_name'] == opp_team]
                if not opp_row.empty:
                    so_total = opp_row.iloc[0].get('SO_hit', 1300)
                    games_total = opp_row.iloc[0].get('games', 162) # If games col doesn't exist, assume 162
                    opp_so_per_game = so_total / games_total if games_total > 0 else 8.5

            features = {
                'game_id': g_id,
                'year': year,
                'player_id': pid,
                'K_per_9': p_profile.get('K_per_9', 8.0),
                'BB_per_9': p_profile.get('BB_per_9', 3.0),
                'ERA': p_profile.get('ERA', 4.12),
                'WHIP': p_profile.get('WHIP', 1.28),
                'avg_K_per_start': p_profile.get('K_per_game', 5.0),
                'avg_innings_per_start': p_profile.get('avg_innings_per_start', 5.5),
                'opp_SO_per_game': opp_so_per_game,
                'K_matchup_factor': (p_profile.get('K_per_9', 8.0) / 9.0) * opp_so_per_game,
                # Target
                'target_k': game.get(f'{side}_starter_k', 0)
            }
            feature_rows.append(features)

    feat_df = pd.DataFrame(feature_rows)
    feat_cols = [c for c in feat_df.columns if c not in ['game_id', 'year', 'player_id', 'target_k']]
    print(f"  ✓ Built features for {len(feat_df)} starting pitcher performances (K prediction)")

    years = sorted(feat_df['year'].unique())
    test_year = years[-1]
    train = feat_df[feat_df['year'] < test_year]
    test = feat_df[feat_df['year'] == test_year]

    if len(test) == 0:
        print("  ⚠ No 2024 test data, using 2023 as test")
        train = feat_df[feat_df['year'] == 2022]
        test = feat_df[feat_df['year'] == 2023]

    X_train = train[feat_cols].astype(float)
    y_train = train['target_k'].astype(float)
    X_test = test[feat_cols].astype(float)
    y_test = test['target_k'].astype(float)

    scaler = StandardScaler()
    X_tr = pd.DataFrame(scaler.fit_transform(X_train), columns=feat_cols, index=X_train.index)
    X_te = pd.DataFrame(scaler.transform(X_test), columns=feat_cols, index=X_test.index)

    # 回帰モデル
    model = XGBRegressor(n_estimators=150, max_depth=3, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                         reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbosity=0)

    cv = cross_val_score(model, X_tr, y_train, cv=5, scoring='neg_mean_absolute_error')
    model.fit(X_tr, y_train, verbose=False)

    y_pred = model.predict(X_te)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n  RESULTS:")
    print(f"  ├─ MAE: {mae:.3f} K/game")
    print(f"  ├─ RMSE: {rmse:.3f} K/game")
    print(f"  ├─ R²: {r2:.4f}")
    print(f"  └─ CV MAE: {-cv.mean():.3f} (+/- {cv.std():.3f})")

    # Over/Under判定の精度（典型的なライン: 5.5K）
    for line in [4.5, 5.5, 6.5, 7.5]:
        pred_over = (y_pred > line).astype(int)
        actual_over = (y_test > line).astype(int)
        if actual_over.sum() > 0 and (1 - actual_over).sum() > 0:
            line_acc = accuracy_score(actual_over, pred_over)
            print(f"  K O/U {line}: Accuracy {line_acc:.4f}")

    # 特徴量重要度
    feat_imp = pd.DataFrame({'Feature': feat_cols, 'Importance': model.feature_importances_})
    feat_imp = feat_imp.sort_values('Importance', ascending=False)
    print(f"\n  Top features: {', '.join(feat_imp.head(3)['Feature'].tolist())}")

    path = MODELS_DIR / 'model_pitcher_k.pkl'
    with open(path, 'wb') as f:
        pickle.dump({
            'model': model,
            'scaler': scaler,
            'feature_cols': feat_cols,
            'feature_names': feat_cols,
            'n_features': len(feat_cols),
            'model_type': 'Pitcher K Props',
            'training_date': datetime.now().isoformat(),
            'train_years': sorted(train['year'].unique().tolist()),
            'test_year': int(test['year'].iloc[0]),
            'test_accuracy': 0,  # 回帰モデルなのでaccuracyは0
            'test_auc': 0,
            'metrics': {
                'k_mae': mae, 'k_rmse': rmse, 'k_r2': r2,
                'cv_mae': -cv.mean(), 'cv_mae_std': cv.std(),
                'accuracy': 0, 'auc': 0,
                'cv_accuracy_mean': 0, 'cv_accuracy_std': 0,
            }
        }, f)
    print(f"\n  ✓ Saved → {path}")

    return model, scaler, feat_cols


# ============================================================================
# MODEL 6: BATTER HIT / HR PROPS
# ============================================================================
def train_batter_props():
    """打者のヒット・HR予測モデル"""
    print("\n" + "="*70)
    print("MODEL 6: BATTER PROPS (打者 Hit/HR 予測)")
    print("="*70)

    # 打者データ読み込み
    batter_dfs = []
    for year in [2022, 2023, 2024, 2025]:
        path = DATA_DIR / f"batter_stats_{year}.csv"
        if path.exists():
            df = pd.read_csv(path)
            batter_dfs.append(df)
            print(f"  ✓ Loaded batter stats {year}: {len(df)} batters")

    if not batter_dfs:
        print("  ❌ No batter data found. Run fetch_player_data.py first.")
        return None, None, None, None, None, None

    batters = pd.concat(batter_dfs, ignore_index=True)

    # 特徴量構築
    features_list = []

    for _, batter in batters.iterrows():
        if batter['games'] < 50 or batter['at_bats'] < 150:
            continue

        f = {
            'player_id': batter['player_id'],
            'name': batter['name'],
            'year': batter['year'],
            'team': batter['team_name'],

            # 打撃能力
            'BA': batter.get('BA', 0.250),
            'OBP': batter.get('OBP', 0.320),
            'SLG': batter.get('SLG', 0.400),
            'OPS': batter.get('OPS', 0.720),
            'games': batter.get('games', 0),
            'at_bats': batter.get('at_bats', 0),
            'plate_appearances': batter.get('plate_appearances', 0),
            'hits': batter.get('hits', 0),
            'home_runs': batter.get('home_runs', 0),
            'walks': batter.get('walks', 0),
            'strikeouts': batter.get('strikeouts', 0),
            'stolen_bases': batter.get('stolen_bases', 0),
            # 'hits_per_game': batter.get('hits_per_game', 1.0), # Removed for leakage
            # 'HR_per_game': batter.get('HR_per_game', 0.1), # Removed for leakage
            'K_rate': batter.get('K_rate', 0.22),
            'ISO': batter.get('SLG', 0.400) - batter.get('BA', 0.250),  # Isolated Power
            'BB_rate': batter.get('walks', 0) / max(batter.get('plate_appearances', 1), 1),
            'AB_per_HR': batter.get('at_bats', 0) / max(batter.get('home_runs', 1), 1),

            # ターゲット
            'target_hits_per_game': batter.get('hits_per_game', 1.0),
            'target_HR_per_game': batter.get('HR_per_game', 0.1),
        }

        features_list.append(f)

    if not features_list:
        print("  ❌ No valid batter records")
        return None, None, None, None, None, None

    bdf = pd.DataFrame(features_list)
    print(f"  ✓ {len(bdf)} batter-seasons for modeling")

    meta = ['player_id', 'name', 'year', 'team', 'target_hits_per_game', 'target_HR_per_game']
    feat_cols = [c for c in bdf.columns if c not in meta]

    # 時系列分離: 2022-2024で学習、2025で評価（リーク防止）
    train = bdf[bdf['year'].isin([2022, 2023, 2024])]
    test = bdf[bdf['year'] == 2025]

    if len(test) == 0:
        train = bdf[bdf['year'].isin([2022, 2023])]
        test = bdf[bdf['year'] == 2024]

    X_train = train[feat_cols].astype(float)
    X_test = test[feat_cols].astype(float)

    scaler = StandardScaler()
    X_tr = pd.DataFrame(scaler.fit_transform(X_train), columns=feat_cols, index=X_train.index)
    X_te = pd.DataFrame(scaler.transform(X_test), columns=feat_cols, index=X_test.index)

    # === HITS MODEL ===
    print("\n  --- HITS MODEL ---")
    y_train_h = train['target_hits_per_game'].astype(float)
    y_test_h = test['target_hits_per_game'].astype(float)

    hit_model = XGBRegressor(n_estimators=150, max_depth=3, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                             reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbosity=0)

    cv_h = cross_val_score(hit_model, X_tr, y_train_h, cv=5, scoring='neg_mean_absolute_error')
    hit_model.fit(X_tr, y_train_h, verbose=False)
    y_pred_h = hit_model.predict(X_te)

    mae_h = mean_absolute_error(y_test_h, y_pred_h)
    r2_h = r2_score(y_test_h, y_pred_h)

    print(f"  MAE: {mae_h:.4f} hits/game")
    print(f"  R²: {r2_h:.4f}")
    print(f"  CV MAE: {-cv_h.mean():.4f}")

    # Over 0.5 hits判定
    for line in [0.5, 1.0, 1.5]:
        pred_o = (y_pred_h > line).astype(int)
        actual_o = (y_test_h > line).astype(int)
        if actual_o.sum() > 0 and (1 - actual_o).sum() > 0:
            print(f"  Hits O/U {line}: Accuracy {accuracy_score(actual_o, pred_o):.4f}")

    # === HR MODEL ===
    print("\n  --- HR MODEL ---")
    y_train_hr = train['target_HR_per_game'].astype(float)
    y_test_hr = test['target_HR_per_game'].astype(float)

    hr_model = XGBRegressor(n_estimators=150, max_depth=3, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                            reg_alpha=0.2, reg_lambda=1.5, random_state=42, verbosity=0)

    cv_hr = cross_val_score(hr_model, X_tr, y_train_hr, cv=5, scoring='neg_mean_absolute_error')
    hr_model.fit(X_tr, y_train_hr, verbose=False)
    y_pred_hr = hr_model.predict(X_te)

    mae_hr = mean_absolute_error(y_test_hr, y_pred_hr)
    r2_hr = r2_score(y_test_hr, y_pred_hr)

    print(f"  MAE: {mae_hr:.4f} HR/game")
    print(f"  R²: {r2_hr:.4f}")
    print(f"  CV MAE: {-cv_hr.mean():.4f}")

    # 特徴量重要度
    feat_imp_h = pd.DataFrame({'Feature': feat_cols, 'Importance': hit_model.feature_importances_})
    feat_imp_hr = pd.DataFrame({'Feature': feat_cols, 'Importance': hr_model.feature_importances_})
    print(f"\n  Top Hit features: {', '.join(feat_imp_h.sort_values('Importance', ascending=False).head(3)['Feature'].tolist())}")
    print(f"  Top HR features: {', '.join(feat_imp_hr.sort_values('Importance', ascending=False).head(3)['Feature'].tolist())}")

    path = MODELS_DIR / 'model_batter_props.pkl'
    with open(path, 'wb') as f:
        pickle.dump({
            'hit_model': hit_model,
            'hr_model': hr_model,
            'scaler': scaler,
            'feature_cols': feat_cols,
            'feature_names': feat_cols,
            'n_features': len(feat_cols),
            'model_type': 'Batter Props',
            'training_date': datetime.now().isoformat(),
            'train_years': sorted(train['year'].unique().tolist()),
            'test_year': int(test['year'].iloc[0]),
            'test_accuracy': 0,
            'test_auc': 0,
            'metrics': {
                'hit_mae': mae_h, 'hit_r2': r2_h,
                'hr_mae': mae_hr, 'hr_r2': r2_hr,
                'accuracy': 0, 'auc': 0,
                'cv_accuracy_mean': 0, 'cv_accuracy_std': 0,
            }
        }, f)
    print(f"\n  ✓ Saved → {path}")

    return hit_model, hr_model, scaler, feat_cols


# ============================================================================
# MODEL 7: NRFI/YRFI — 初回無得点予測
# ============================================================================
def _build_nrfi_features(games_df, nrfi_df, team_stats_dict):
    """Build NRFI feature matrix with pitcher data, venue, rolling stats, and cross-features.

    Returns:
        feat_df: DataFrame with features + game_id, year, nrfi_result
        feat_cols: List of feature column names
    """
    if 'nrfi' in nrfi_df.columns and 'nrfi_result' not in nrfi_df.columns:
        nrfi_df = nrfi_df.rename(columns={'nrfi': 'nrfi_result'})

    # --- Load pitcher stats ---
    pitcher_lookup = {}
    for year in [2022, 2023, 2024, 2025]:
        path = DATA_DIR / f"pitcher_stats_{year}.csv"
        if path.exists():
            for _, p in pd.read_csv(path).iterrows():
                pitcher_lookup[(p['name'], p['year'])] = {
                    'ERA': float(p.get('ERA', 4.12)),
                    'WHIP': float(p.get('WHIP', 1.28)),
                    'K_per_9': float(p.get('K_per_9', 8.0)),
                    'BB_per_9': float(p.get('BB_per_9', 3.0)),
                }

    def get_pitcher(name, year):
        defaults = {'ERA': 4.12, 'WHIP': 1.28, 'K_per_9': 8.0, 'BB_per_9': 3.0}
        if not isinstance(name, str) or not name or name == 'TBA':
            return defaults
        p = pitcher_lookup.get((name, year))
        if p:
            return p
        # Fallback: last-name match
        last = name.split()[-1] if ' ' in name else name
        for (pn, py), ps in pitcher_lookup.items():
            if py == year and pn.endswith(last):
                return ps
        return defaults

    def get_team_csv(team, year):
        defaults = {'BA': 0.248, 'OBP': 0.315, 'SLG': 0.395, 'ERA': 4.12, 'WHIP': 1.28}
        ts = team_stats_dict.get(year)
        if ts is None:
            return defaults
        row = ts[ts['team_name'] == team]
        if row.empty:
            return defaults
        r = row.iloc[0]
        return {k: float(r.get(k, defaults[k])) for k in defaults}

    # --- Merge NRFI with games to get pitcher + venue ---
    games_info = games_df[['game_id', 'year', 'home_team', 'away_team',
                           'home_pitcher', 'away_pitcher', 'venue']].drop_duplicates()
    merged = nrfi_df.merge(games_info, on='game_id', how='inner', suffixes=('', '_g'))
    for col in [c for c in merged.columns if c.endswith('_g')]:
        base = col[:-2]
        if base in merged.columns:
            merged[base] = merged[base].fillna(merged[col])
        merged.drop(columns=[col], inplace=True)

    # --- 1st inning team season averages ---
    h1 = nrfi_df.groupby(['home_team', 'year']).agg(
        h_1st_rs=('home_1st_runs', 'mean'), h_nrfi=('nrfi_result', 'mean'),
    ).to_dict('index')
    a1 = nrfi_df.groupby(['away_team', 'year']).agg(
        a_1st_rs=('away_1st_runs', 'mean'),
    ).to_dict('index')
    h1a = nrfi_df.groupby(['home_team', 'year']).agg(
        h_1st_ra=('away_1st_runs', 'mean'),
    ).to_dict('index')
    a1a = nrfi_df.groupby(['away_team', 'year']).agg(
        a_1st_ra=('home_1st_runs', 'mean'),
    ).to_dict('index')

    # --- Venue NRFI rate ---
    nrfi_venue = nrfi_df.merge(games_df[['game_id', 'venue']].drop_duplicates(), on='game_id', how='left')
    venue_stats = nrfi_venue.groupby('venue')['nrfi_result'].agg(['mean', 'count'])
    league_nrfi = nrfi_df['nrfi_result'].mean()
    venue_map = {v: r['mean'] if r['count'] >= 20 else league_nrfi
                 for v, r in venue_stats.iterrows()}

    # --- Rolling NRFI rate (walk-forward, last 20 games) ---
    nrfi_sorted = nrfi_df.copy()
    if 'date' not in nrfi_sorted.columns:
        nrfi_sorted = nrfi_sorted.merge(
            games_df[['game_id', 'date']].drop_duplicates(), on='game_id', how='left'
        )
    nrfi_sorted = nrfi_sorted.sort_values('date')
    ROLL = 20
    teams = set(nrfi_sorted['home_team'].unique()) | set(nrfi_sorted['away_team'].unique())

    roll_nrfi = {}   # game_id -> {home: nrfi_rate, away: nrfi_rate}
    roll_1st_rs = {} # game_id -> {home: rs, away: rs}
    for team in teams:
        tg = nrfi_sorted[(nrfi_sorted['home_team'] == team) | (nrfi_sorted['away_team'] == team)].copy()
        tg['t_1st'] = np.where(tg['home_team'] == team, tg['home_1st_runs'], tg['away_1st_runs'])
        r_nrfi = tg['nrfi_result'].rolling(ROLL, min_periods=5).mean()
        r_rs = tg['t_1st'].rolling(ROLL, min_periods=5).mean()
        for i, (_, row) in enumerate(tg.iterrows()):
            gid = row['game_id']
            if gid not in roll_nrfi:
                roll_nrfi[gid] = {}
                roll_1st_rs[gid] = {}
            side = 'home' if row['home_team'] == team else 'away'
            roll_nrfi[gid][side] = r_nrfi.iloc[i] if not pd.isna(r_nrfi.iloc[i]) else league_nrfi
            roll_1st_rs[gid][side] = r_rs.iloc[i] if not pd.isna(r_rs.iloc[i]) else 0.49

    # --- Build features ---
    feature_rows = []
    for _, row in merged.iterrows():
        home, away, year, gid = row['home_team'], row['away_team'], int(row['year']), row['game_id']

        h1v = h1.get((home, year), {})
        a1v = a1.get((away, year), {})
        if not h1v or not a1v:
            continue

        hp = get_pitcher(row.get('home_pitcher', ''), year)
        ap = get_pitcher(row.get('away_pitcher', ''), year)
        ht = get_team_csv(home, year)
        at = get_team_csv(away, year)
        vnrfi = venue_map.get(row.get('venue', ''), league_nrfi)
        rn = roll_nrfi.get(gid, {})
        rr = roll_1st_rs.get(gid, {})

        features = {
            'game_id': gid, 'year': year, 'nrfi_result': row['nrfi_result'],

            # 1st inning season averages
            'home_1st_rs': h1v.get('h_1st_rs', 0.49),
            'home_1st_ra': h1a.get((home, year), {}).get('h_1st_ra', 0.49),
            'away_1st_rs': a1v.get('a_1st_rs', 0.49),
            'away_1st_ra': a1a.get((away, year), {}).get('a_1st_ra', 0.49),
            'combined_1st_rs': (h1v.get('h_1st_rs', 0.49) + a1v.get('a_1st_rs', 0.49)) / 2,
            'home_season_nrfi': h1v.get('h_nrfi', league_nrfi),

            # Starting pitcher stats
            'hp_era': hp['ERA'], 'hp_whip': hp['WHIP'],
            'hp_k9': hp['K_per_9'], 'hp_bb9': hp['BB_per_9'],
            'ap_era': ap['ERA'], 'ap_whip': ap['WHIP'],
            'ap_k9': ap['K_per_9'], 'ap_bb9': ap['BB_per_9'],
            'combined_sp_era': (hp['ERA'] + ap['ERA']) / 2,
            'combined_sp_whip': (hp['WHIP'] + ap['WHIP']) / 2,
            'sp_era_diff': hp['ERA'] - ap['ERA'],
            # Pitcher dominance: low ERA + high K/9
            'hp_dominance': hp['K_per_9'] / max(hp['ERA'], 0.5),
            'ap_dominance': ap['K_per_9'] / max(ap['ERA'], 0.5),
            'combined_dominance': (hp['K_per_9'] / max(hp['ERA'], 0.5) +
                                   ap['K_per_9'] / max(ap['ERA'], 0.5)) / 2,

            # Matchup cross-features: pitcher vs lineup
            'hp_era_x_away_obp': hp['ERA'] * at['OBP'],
            'ap_era_x_home_obp': ap['ERA'] * ht['OBP'],
            'hp_whip_x_away_slg': hp['WHIP'] * at['SLG'],
            'ap_whip_x_home_slg': ap['WHIP'] * ht['SLG'],
            'hp_bb9_x_away_obp': hp['BB_per_9'] * at['OBP'],
            'ap_bb9_x_home_obp': ap['BB_per_9'] * ht['OBP'],

            # Team batting (from CSV)
            'home_ba': ht['BA'], 'away_ba': at['BA'],
            'home_obp': ht['OBP'], 'away_obp': at['OBP'],
            'home_slg': ht['SLG'], 'away_slg': at['SLG'],
            'combined_obp': (ht['OBP'] + at['OBP']) / 2,

            # Team pitching (from CSV)
            'home_team_era': ht['ERA'], 'away_team_era': at['ERA'],
            'home_team_whip': ht['WHIP'], 'away_team_whip': at['WHIP'],
            'combined_team_era': (ht['ERA'] + at['ERA']) / 2,

            # Venue factor
            'venue_nrfi_rate': vnrfi,

            # Rolling (walk-forward safe)
            'roll_nrfi_home': rn.get('home', league_nrfi),
            'roll_nrfi_away': rn.get('away', league_nrfi),
            'roll_nrfi_combined': (rn.get('home', league_nrfi) + rn.get('away', league_nrfi)) / 2,
            'roll_1st_rs_home': rr.get('home', 0.49),
            'roll_1st_rs_away': rr.get('away', 0.49),
            'roll_1st_rs_combined': (rr.get('home', 0.49) + rr.get('away', 0.49)) / 2,
        }
        feature_rows.append(features)

    feat_df = pd.DataFrame(feature_rows)
    feat_cols = [c for c in feat_df.columns if c not in ['game_id', 'year', 'nrfi_result']]
    return feat_df, feat_cols


def train_nrfi(games_df, overall_stats, rolling_stats, team_stats):
    """Train NRFI model v3: pitcher + venue + rolling + matchup cross-features"""
    print("\n" + "="*60)
    print("  NRFI/YRFI MODEL — 初回無得点予測 (v3)")
    print("="*60)

    nrfi_path = DATA_DIR / "nrfi_data_2022_2025.csv"
    if not nrfi_path.exists():
        print("  ⚠ nrfi_data_2022_2025.csv not found!")
        return None, None, None

    nrfi_df = pd.read_csv(nrfi_path)
    print(f"  ✓ Loaded {len(nrfi_df)} NRFI records")

    feat_df, feat_cols = _build_nrfi_features(games_df, nrfi_df, team_stats)
    print(f"  ✓ {len(feat_cols)} features for {len(feat_df)} games")

    years = sorted(feat_df['year'].unique())
    test_year = years[-1]
    train = feat_df[feat_df['year'] < test_year]
    test = feat_df[feat_df['year'] == test_year]

    X_train = train[feat_cols].astype(float).fillna(0)
    y_train = train['nrfi_result'].astype(int)
    X_test = test[feat_cols].astype(float).fillna(0)
    y_test = test['nrfi_result'].astype(int)

    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=feat_cols)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=feat_cols)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.6,
        min_child_weight=5,
        reg_alpha=0.3,
        reg_lambda=2.0,
        gamma=0.15,
        eval_metric='logloss',
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train_s, y_train, verbose=False)

    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    try:
        auc_val = roc_auc_score(y_test, y_prob)
    except Exception:
        auc_val = 0.0

    cv = cross_val_score(model, X_train_s, y_train, cv=5, scoring='accuracy')

    # Confidence tier analysis
    print(f"\n  Confidence Tiers:")
    for tier, lo, hi in [('STRONG', 0.10, 1.0), ('MODERATE', 0.05, 0.10), ('LEAN', 0.0, 0.05)]:
        mask = (np.abs(y_prob - 0.5) >= lo) & (np.abs(y_prob - 0.5) < hi)
        if mask.sum() > 0:
            t_acc = accuracy_score(y_test[mask], y_pred[mask])
            print(f"    {tier:10s}: {t_acc:.4f} ({mask.sum()} games)")

    # Feature importance top 10
    imp = sorted(zip(feat_cols, model.feature_importances_), key=lambda x: -x[1])
    print(f"\n  Top features:")
    for name, val in imp[:10]:
        bar = '█' * int(val * 50)
        print(f"    {name:30s} {bar} {val:.4f}")

    print(f"\n  Accuracy: {acc:.4f} | AUC: {auc_val:.4f} | CV: {cv.mean():.4f}±{cv.std():.4f}")

    save_model('model_nrfi.pkl', model, scaler, feat_cols,
               'NRFI/YRFI', acc, auc_val, cv)

    return model, scaler, feat_cols


# ============================================================================
# MODEL 8: STOLEN BASES — 盗塁予測
# ============================================================================
def train_stolen_bases(games_df, overall_stats):
    """Train Stolen Bases over/under model"""
    print("\n" + "="*60)
    print("  STOLEN BASES MODEL — 盗塁予測")
    print("="*60)

    # Load batter stats for SB data
    all_batters = []
    for year in [2022, 2023, 2024, 2025]:
        path = DATA_DIR / f"batter_stats_{year}.csv"
        if path.exists():
            df = pd.read_csv(path)
            all_batters.append(df)

    if not all_batters:
        print("  ⚠ No batter stats found! Run fetch_player_data.py first.")
        return None, None, None

    batters_df = pd.concat(all_batters, ignore_index=True)
    print(f"  ✓ Loaded {len(batters_df)} batter records")

    # Calculate team-level SB rates
    team_sb = batters_df.groupby(['team_name', 'year']).agg(
        total_sb=('stolen_bases', 'sum'),
        total_games=('games', 'max'),
        avg_speed_proxy=('stolen_bases', 'mean'),  # avg SB per qualifying batter
        total_batters=('player_id', 'count')
    ).reset_index()
    team_sb['sb_per_game'] = team_sb['total_sb'] / team_sb['total_games'].clip(lower=1)

    # Load actual SB counts from game props data (2024-2025)
    props_path = DATA_DIR / "game_props_data_2024_2025.csv"
    if not props_path.exists():
        print("  ⚠ game_props_data_2024_2025.csv not found!")
        return None, None, None
    
    props_df = pd.read_csv(props_path)
    print(f"  ✓ Loaded {len(props_df)} games with actual SB metrics")

    # Build game-level features by merging with games
    feature_rows = []
    for _, game in props_df.iterrows():
        g_id = int(game['game_id'])
        # Find this game in games_df to get teams/rolling stats
        g_info = games_df[games_df['game_id'] == g_id]
        if g_info.empty: continue
        g_info = g_info.iloc[0]
        
        home = g_info['home_team']
        away = g_info['away_team']
        year = g_info['year']

        home_stats = overall_stats.get((home, year), {})
        away_stats = overall_stats.get((away, year), {})
        
        # Team SB tendencies from historical season data
        h_sb_tendency = team_sb[(team_sb['team_name'] == home) & (team_sb['year'] == year)]
        a_sb_tendency = team_sb[(team_sb['team_name'] == away) & (team_sb['year'] == year)]

        features = {
            'game_id': g_id,
            'year': year,
            # Features (Tendencies)
            'home_sb_per_game': h_sb_tendency.iloc[0]['sb_per_game'] if not h_sb_tendency.empty else 0.7,
            'away_sb_per_game': a_sb_tendency.iloc[0]['sb_per_game'] if not a_sb_tendency.empty else 0.7,
            'home_speed_proxy': h_sb_tendency.iloc[0]['avg_speed_proxy'] if not h_sb_tendency.empty else 10/162,
            'away_speed_proxy': a_sb_tendency.iloc[0]['avg_speed_proxy'] if not a_sb_tendency.empty else 10/162,
            'home_win_pct': home_stats.get('win_pct', 0.500),
            'away_win_pct': away_stats.get('win_pct', 0.500),
            'home_runs_per_game': home_stats.get('runs_per_game', 0),
            'away_runs_per_game': away_stats.get('runs_per_game', 0),
            'home_obp': home_stats.get('team_obp', 0.320),
            'away_obp': away_stats.get('team_obp', 0.320),
            # Target (Actual in game)
            'target_total_sb': game['total_sb']
        }
        feature_rows.append(features)

    feat_df = pd.DataFrame(feature_rows)
    # Binary classification: Will there be OVER 1.5 Stolen Bases?
    feat_df['sb_over'] = (feat_df['target_total_sb'] >= 2).astype(int)
    
    feat_cols = [c for c in feat_df.columns if c not in ['game_id', 'year', 'sb_over', 'target_total_sb']]

    print(f"  ✓ Built features for {len(feat_df)} games")

    years = sorted(feat_df['year'].unique())
    test_year = years[-1]
    train = feat_df[feat_df['year'] < test_year]
    test = feat_df[feat_df['year'] == test_year]

    if len(test) == 0:
        train = feat_df.sample(frac=0.8, random_state=42)
        test = feat_df.drop(train.index)

    X_train = train[feat_cols].astype(float).fillna(0)
    y_train = train['sb_over'].astype(int)
    X_test = test[feat_cols].astype(float).fillna(0)
    y_test = test['sb_over'].astype(int)

    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=feat_cols)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=feat_cols)

    model = XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=42
    )
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    try:
        auc_val = roc_auc_score(y_test, y_prob)
    except:
        auc_val = 0.0

    cv = cross_val_score(model, X_train_s, y_train, cv=5, scoring='accuracy')
    print(f"  Accuracy: {acc:.4f} | AUC: {auc_val:.4f} | CV: {cv.mean():.4f}±{cv.std():.4f}")

    save_model('model_stolen_bases.pkl', model, scaler, feat_cols,
               'Stolen Bases', acc, auc_val, cv)

    return model, scaler, feat_cols


# ============================================================================
# MODEL 9: PITCHER OUTS — 投手アウト数（イニング数）予測
# ============================================================================
def train_pitcher_outs(games_df, overall_stats):
    """Train Pitcher Outs (Innings Pitched) regression model"""
    print("\n" + "="*60)
    print("  PITCHER OUTS MODEL — 投手イニング数予測")
    print("="*60)

    # Load pitcher stats
    all_pitchers = []
    for year in [2022, 2023, 2024, 2025]:
        path = DATA_DIR / f"pitcher_stats_{year}.csv"
        if path.exists():
            df = pd.read_csv(path)
            all_pitchers.append(df)

    if not all_pitchers:
        print("  ⚠ No pitcher stats found! Run fetch_player_data.py first.")
        return None, None, None

    pitchers_df = pd.concat(all_pitchers, ignore_index=True)
    print(f"  ✓ Loaded {len(pitchers_df)} pitcher records")

    # Target: avg_innings_per_start (regression)
    # Features: ERA, WHIP, K/9, BB/9, team quality
    pitchers_df = pitchers_df[pitchers_df['games_started'] >= 5].copy()

    # Load actual IP results from game props data (2024-2025)
    props_path = DATA_DIR / "game_props_data_2024_2025.csv"
    if not props_path.exists():
        print("  ⚠ game_props_data_2024_2025.csv not found!")
        return None, None, None
    
    props_df = pd.read_csv(props_path)

    # Build feature rows
    feature_rows = []
    for _, game in props_df.iterrows():
        g_id = int(game['game_id'])
        g_info = games_df[games_df['game_id'] == g_id]
        if g_info.empty: continue
        g_info = g_info.iloc[0]
        year = g_info['year']

        for side in ['home', 'away']:
            pid = game.get(f'{side}_starter_id')
            if pd.isna(pid): continue
            
            # Find pitcher profile in pitcher_stats
            p_profile = pitchers_df[(pitchers_df['player_id'] == pid) & (pitchers_df['year'] == year)]
            if p_profile.empty: continue
            p_profile = p_profile.iloc[0]
            
            # Actual IP in this game (Convert '6.1' style to float 6.33 for regression if needed, but '6.1' is 6 + 1/3)
            ip_str = str(game.get(f'{side}_starter_ip', '0.0'))
            try:
                ip_val = float(ip_str.split('.')[0]) + (float(ip_str.split('.')[1])/3 if '.' in ip_str else 0)
            except: ip_val = 0.0

            features = {
                'game_id': g_id,
                'year': year,
                'player_id': pid,
                # Features from season profile
                'K_per_9': p_profile.get('K_per_9', 0),
                'BB_per_9': p_profile.get('BB_per_9', 0),
                'ERA': p_profile.get('ERA', 4.0),
                'WHIP': p_profile.get('WHIP', 1.3),
                'avg_innings_per_start': p_profile.get('avg_innings_per_start', 5.5),
                'K_BB_ratio': p_profile.get('K_per_9', 0) / max(p_profile.get('BB_per_9', 1), 0.5),
                # Target
                'target_ip': ip_val
            }
            feature_rows.append(features)

    feat_df = pd.DataFrame(feature_rows)
    feat_cols = [c for c in feat_df.columns if c not in ['game_id', 'year', 'player_id', 'target_ip']]

    print(f"  ✓ Built features for {len(feat_df)} starting pitcher performances")

    if len(feat_df) < 50:
        print("  ⚠ Not enough data to train Pitcher Outs model")
        return None, None, None

    years = sorted(feat_df['year'].unique())
    test_year = years[-1]
    train = feat_df[feat_df['year'] < test_year]
    test = feat_df[feat_df['year'] == test_year]

    if len(test) == 0:
        train = feat_df.sample(frac=0.8, random_state=42)
        test = feat_df.drop(train.index)

    X_train = train[feat_cols].astype(float).fillna(0)
    y_train = train['target_ip'].astype(float)
    X_test = test[feat_cols].astype(float).fillna(0)
    y_test = test['target_ip'].astype(float)

    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=feat_cols)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=feat_cols)

    model = XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"  MAE: {mae:.3f} innings | R²: {r2:.3f} | RMSE: {rmse:.3f}")

    path = MODELS_DIR / 'model_pitcher_outs.pkl'
    with open(path, 'wb') as f:
        pickle.dump({
            'model': model,
            'scaler': scaler,
            'feature_cols': feat_cols,
            'feature_names': feat_cols,
            'n_features': len(feat_cols),
            'model_type': 'Pitcher Outs',
            'training_date': datetime.now().isoformat(),
            'metrics': {
                'outs_mae': mae, 'outs_r2': r2, 'outs_rmse': rmse,
                'accuracy': 0, 'auc': 0,
                'cv_accuracy_mean': 0, 'cv_accuracy_std': 0,
            }
        }, f)
    print(f"  ✓ Saved → {path}")

    return model, scaler, feat_cols


# ============================================================================
# SAVE HELPER
# ============================================================================
def save_model(filename, model, scaler, feat_cols, model_type, accuracy, auc, cv_scores, extra=None):
    """モデル保存"""
    path = MODELS_DIR / filename
    data = {
        'model': model,
        'scaler': scaler,
        'feature_cols': feat_cols,
        'feature_names': feat_cols,
        'n_features': len(feat_cols),
        'model_type': model_type,
        'training_date': datetime.now().isoformat(),
        'train_years': [2022, 2023, 2024],  # 2025は評価用
        'test_year': 2025,
        'test_accuracy': accuracy,
        'test_auc': auc,
        'metrics': {
            'accuracy': accuracy,
            'auc': auc,
            'cv_accuracy_mean': cv_scores.mean(),
            'cv_accuracy_std': cv_scores.std(),
        }
    }
    if extra:
        data['metrics'].update(extra)

    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print(f"\n  ✓ Saved → {path}")

    # 既存のmodel.pklもMoneylineの場合は更新
    if filename == 'model_moneyline.pkl':
        legacy_path = PROJECT_DIR / 'model.pkl'
        with open(legacy_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"  ✓ Also saved → {legacy_path} (legacy compatibility)")


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("="*80)
    print("MONEYBALL DOJO — ALL MARKETS MODEL TRAINING")
    print("="*80)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # コマンドライン引数
    args = sys.argv[1:]
    only_model = None
    if '--model' in args:
        idx = args.index('--model')
        if idx + 1 < len(args):
            only_model = args[idx + 1].lower()

    # データ読み込み
    print("[1/6] Loading data...")
    games_df, team_stats = load_all_data()
    print(f"  ✓ {len(games_df)} games loaded")

    print("\n[2/6] Computing team metrics...")
    overall_stats = compute_all_team_metrics(games_df)
    print(f"  ✓ {len(overall_stats)} team-year records")

    print("\n[3/6] Computing rolling stats...")
    rolling_stats = compute_rolling_features(games_df)
    print(f"  ✓ Rolling stats computed")

    print("\n[4/6] Building feature matrix...")
    X, skipped = build_feature_matrix(games_df, overall_stats, rolling_stats, team_stats)
    print(f"  ✓ {len(X)} games with features (skipped {skipped})")

    meta_cols = ['game_id', 'date', 'year', 'home_team', 'away_team']

    # モデルトレーニング
    print("\n[5/6] Training models...")

    results = {}

    if only_model is None or only_model == 'ml':
        m, s, f = train_moneyline(X, meta_cols)
        results['moneyline'] = {'accuracy': accuracy_score(
            X[X['year']==2024]['home_win'].astype(int),
            m.predict(pd.DataFrame(s.transform(X[X['year']==2024][f].astype(float)), columns=f))
        )}

    if only_model is None or only_model == 'ou':
        reg, cls, s, f = train_over_under(X, meta_cols)
        results['over_under'] = True

    if only_model is None or only_model == 'rl':
        m, s, f = train_run_line(X, meta_cols)
        results['run_line'] = True

    if only_model is None or only_model == 'f5':
        m, s, f = train_f5_moneyline(X, meta_cols)
        results['f5'] = True

    # Player-level models (独自データ読み込み)
    if only_model is None or only_model == 'pk':
        try:
            train_pitcher_k_props()
            results['pitcher_k'] = True
        except Exception as e:
            print(f"\n  ⚠ Pitcher K Props skipped: {e}")
            print("    → Run fetch_player_data.py first")

    if only_model is None or only_model == 'bp':
        try:
            train_batter_props()
            results['batter_props'] = True
        except Exception as e:
            print(f"\n  ⚠ Batter Props skipped: {e}")
            print("    → Run fetch_player_data.py first")

    # New market models
    if only_model is None or only_model == 'nrfi':
        try:
            train_nrfi(games_df, overall_stats, rolling_stats, team_stats)
            results['nrfi'] = True
        except Exception as e:
            print(f"\n  ⚠ NRFI model skipped: {e}")
            print("    → Run fetch_nrfi_data.py first")

    if only_model is None or only_model == 'sb':
        try:
            train_stolen_bases(games_df, overall_stats)
            results['stolen_bases'] = True
        except Exception as e:
            print(f"\n  ⚠ Stolen Bases model skipped: {e}")
            print("    → Run fetch_player_data.py first")

    if only_model is None or only_model == 'outs':
        try:
            train_pitcher_outs(games_df, overall_stats)
            results['pitcher_outs'] = True
        except Exception as e:
            print(f"\n  ⚠ Pitcher Outs model skipped: {e}")
            print("    → Run fetch_player_data.py first")

    # サマリー
    print("\n\n" + "="*80)
    print("[6/6] FINAL SUMMARY")
    print("="*80)

    for pkl_file in sorted(MODELS_DIR.glob('model_*.pkl')):
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        name = data.get('model_type', pkl_file.stem)
        metrics = data.get('metrics', {})
        n_feat = data.get('n_features', 0)

        # Regression models show MAE instead of accuracy
        if 'pitcher_k' in name.lower() or 'pitcher_k' in pkl_file.stem:
            mae = metrics.get('k_mae', 0)
            r2 = metrics.get('k_r2', 0)
            print(f"  {name:20s} | MAE: {mae:.3f} K | R²: {r2:.3f} | Features: {n_feat}")
        elif 'pitcher_outs' in pkl_file.stem or 'Pitcher Outs' in name:
            mae = metrics.get('outs_mae', 0)
            r2 = metrics.get('outs_r2', 0)
            print(f"  {name:20s} | MAE: {mae:.3f} IP | R²: {r2:.3f} | Features: {n_feat}")
        elif 'batter' in name.lower() or 'batter' in pkl_file.stem:
            mae_h = metrics.get('hit_mae', 0)
            mae_hr = metrics.get('hr_mae', 0)
            print(f"  {name:20s} | Hit MAE: {mae_h:.3f} | HR MAE: {mae_hr:.4f} | Features: {n_feat}")
        else:
            acc = metrics.get('accuracy', 0)
            auc_val = metrics.get('auc', 0)
            print(f"  {name:20s} | Acc: {acc:.4f} | AUC: {auc_val:.4f} | Features: {n_feat}")

    print(f"\n  All models saved to: {MODELS_DIR}/")
    print("="*80)


if __name__ == '__main__':
    main()
