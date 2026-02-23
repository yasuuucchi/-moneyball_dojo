"""
Moneyball Dojo — Monte Carlo Season Simulator
===============================================
XGBoostモデルを使って2026 MLBシーズンを10,000回シミュレーション。
出力:
  - 各チームの優勝確率（ワールドシリーズ / リーグ / ディビジョン）
  - プレーオフ進出確率（12チーム）
  - 予想勝率と勝利数の分布

使い方:
  python season_simulator.py                    # フルシミュレーション
  python season_simulator.py --sims 1000        # シミュレーション回数指定
  python season_simulator.py --output results   # 出力先指定
"""

import os
import sys
import json
import pickle
import logging
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np

# ========================================================
# 設定
# ========================================================
PROJECT_DIR = Path(__file__).parent
MODELS_DIR = PROJECT_DIR / "models"
DATA_DIR = PROJECT_DIR / "data"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

# ========================================================
# MLB ディビジョン定義
# ========================================================
DIVISIONS = {
    'AL East': ['New York Yankees', 'Baltimore Orioles', 'Toronto Blue Jays',
                'Boston Red Sox', 'Tampa Bay Rays'],
    'AL Central': ['Cleveland Guardians', 'Minnesota Twins', 'Kansas City Royals',
                   'Detroit Tigers', 'Chicago White Sox'],
    'AL West': ['Houston Astros', 'Seattle Mariners', 'Texas Rangers',
                'Los Angeles Angels', 'Athletics'],
    'NL East': ['Atlanta Braves', 'Philadelphia Phillies', 'New York Mets',
                'Washington Nationals', 'Miami Marlins'],
    'NL Central': ['Milwaukee Brewers', 'Chicago Cubs', 'St. Louis Cardinals',
                   'Cincinnati Reds', 'Pittsburgh Pirates'],
    'NL West': ['Los Angeles Dodgers', 'San Diego Padres', 'Arizona Diamondbacks',
                'San Francisco Giants', 'Colorado Rockies'],
}

LEAGUE_DIVISIONS = {
    'AL': ['AL East', 'AL Central', 'AL West'],
    'NL': ['NL East', 'NL Central', 'NL West'],
}

ALL_TEAMS = [team for div_teams in DIVISIONS.values() for team in div_teams]


def get_team_division(team):
    for div, teams in DIVISIONS.items():
        if team in teams:
            return div
    return None


def get_team_league(team):
    div = get_team_division(team)
    if div and div.startswith('AL'):
        return 'AL'
    elif div and div.startswith('NL'):
        return 'NL'
    return None


# ========================================================
# 2026 MLB スケジュール生成（162試合）
# ========================================================
def generate_season_schedule():
    """
    MLBスケジュールルールに基づく162試合スケジュールを生成。
    - 同ディビジョン: 各チーム13試合（ホーム6-7、アウェー6-7）= 52試合
    - 同リーグ他ディビジョン: 各チーム6-7試合 = 64試合
    - インターリーグ: 各チーム3-4試合 = 46試合
    """
    logger.info("Generating 162-game season schedule...")
    schedule = []

    # 同ディビジョン: 13試合ずつ（7ホーム + 6アウェー or 6+7）
    for div_name, teams in DIVISIONS.items():
        for i, team_a in enumerate(teams):
            for j, team_b in enumerate(teams):
                if i >= j:
                    continue
                # 13試合: 7ホーム + 6アウェー (交互)
                for g in range(7):
                    schedule.append({'home_team': team_a, 'away_team': team_b})
                for g in range(6):
                    schedule.append({'home_team': team_b, 'away_team': team_a})

    # 同リーグ他ディビジョン: 各対戦6-7試合
    for league, divs in LEAGUE_DIVISIONS.items():
        for d1_idx, div1_name in enumerate(divs):
            for d2_idx, div2_name in enumerate(divs):
                if d1_idx >= d2_idx:
                    continue
                for team_a in DIVISIONS[div1_name]:
                    for team_b in DIVISIONS[div2_name]:
                        # 7試合: 4ホーム + 3アウェー
                        for g in range(4):
                            schedule.append({'home_team': team_a, 'away_team': team_b})
                        for g in range(3):
                            schedule.append({'home_team': team_b, 'away_team': team_a})

    # インターリーグ: 各対戦3試合（自然なライバル + ローテーション）
    al_teams = [t for d in LEAGUE_DIVISIONS['AL'] for t in DIVISIONS[d]]
    nl_teams = [t for d in LEAGUE_DIVISIONS['NL'] for t in DIVISIONS[d]]

    # 全AL vs NL組み合わせからサンプル
    interleague_pairs = []
    for al in al_teams:
        for nl in nl_teams:
            interleague_pairs.append((al, nl))

    # チームごとのインターリーグ試合数を追跡
    il_counts = defaultdict(int)
    target_il = 46  # 各チーム46試合

    np.random.seed(2026)
    np.random.shuffle(interleague_pairs)

    for al, nl in interleague_pairs:
        if il_counts[al] >= target_il or il_counts[nl] >= target_il:
            continue
        games_to_add = min(3, target_il - il_counts[al], target_il - il_counts[nl])
        home_games = games_to_add // 2 + (games_to_add % 2)
        away_games = games_to_add - home_games
        for g in range(home_games):
            schedule.append({'home_team': al, 'away_team': nl})
        for g in range(away_games):
            schedule.append({'home_team': nl, 'away_team': al})
        il_counts[al] += games_to_add
        il_counts[nl] += games_to_add

    # 各チームの試合数を確認・調整
    game_counts = defaultdict(int)
    for g in schedule:
        game_counts[g['home_team']] += 1
        game_counts[g['away_team']] += 1

    logger.info(f"  Schedule generated: {len(schedule)} total games")
    for team in ALL_TEAMS:
        count = game_counts.get(team, 0)
        if count != 162:
            logger.debug(f"  {team}: {count} games (target: 162)")

    return schedule


# ========================================================
# モデル読み込み & 特徴量生成
# ========================================================
def load_moneyline_model():
    """Moneylineモデルとスケーラーを読み込み"""
    logger.info("Loading moneyline model...")
    model_path = MODELS_DIR / 'model_moneyline.pkl'
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        sys.exit(1)

    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    logger.info(f"  ✓ Moneyline model loaded ({len(model_data['feature_cols'])} features)")
    return model_data


def load_team_data():
    """チームデータを読み込み"""
    logger.info("Loading team data...")

    # 試合データ
    games_path = DATA_DIR / "games_2022_2025.csv"
    if not games_path.exists():
        games_path = DATA_DIR / "games_2022_2024.csv"
    games_df = pd.read_csv(games_path)
    games_df['date'] = pd.to_datetime(games_df['date'])

    # チームスタッツ（最新年）
    team_stats = {}
    for year in [2022, 2023, 2024, 2025]:
        path = DATA_DIR / f"team_stats_{year}.csv"
        if path.exists():
            team_stats[year] = pd.read_csv(path)

    latest_year = max(team_stats.keys())
    logger.info(f"  ✓ {len(games_df)} games, team stats through {latest_year}")
    return games_df, team_stats, latest_year


def compute_team_overall_stats(games_df, year):
    """指定年のチーム全体成績"""
    year_games = games_df[games_df['year'] == year]
    all_teams = set(year_games['home_team'].unique()) | set(year_games['away_team'].unique())

    records = {}
    for team in all_teams:
        home_g = year_games[year_games['home_team'] == team]
        away_g = year_games[year_games['away_team'] == team]

        home_wins = int(home_g['home_win'].sum())
        away_wins = len(away_g) - int(away_g['home_win'].sum())
        total_wins = home_wins + away_wins
        total_games = len(home_g) + len(away_g)

        home_rs = home_g['home_score'].sum()
        away_rs = away_g['away_score'].sum()
        home_ra = home_g['away_score'].sum()
        away_ra = away_g['home_score'].sum()

        total_rs = home_rs + away_rs
        total_ra = home_ra + away_ra

        win_pct = total_wins / total_games if total_games > 0 else 0.5
        avg_rs = total_rs / total_games if total_games > 0 else 4.5
        avg_ra = total_ra / total_games if total_games > 0 else 4.5
        run_diff = (total_rs - total_ra) / total_games if total_games > 0 else 0
        pythag = total_rs**1.83 / (total_rs**1.83 + total_ra**1.83) if (total_rs + total_ra) > 0 else 0.5

        home_win_pct = home_g['home_win'].mean() if len(home_g) > 0 else 0.54
        away_win_pct = (1 - away_g['home_win']).mean() if len(away_g) > 0 else 0.46
        home_avg_rs = home_g['home_score'].mean() if len(home_g) > 0 else 4.5
        away_avg_rs = away_g['away_score'].mean() if len(away_g) > 0 else 4.5

        records[team] = {
            'win_pct': win_pct, 'avg_rs': avg_rs, 'avg_ra': avg_ra,
            'run_diff': run_diff, 'pythag': pythag,
            'home_win_pct': home_win_pct, 'away_win_pct': away_win_pct,
            'home_avg_rs': home_avg_rs, 'away_avg_rs': away_avg_rs,
        }

    return records


def compute_team_rolling_stats(games_df, window=15):
    """直近N試合のローリングスタッツ"""
    games_sorted = games_df.sort_values('date').reset_index(drop=True)
    all_teams = set(games_sorted['home_team'].unique()) | set(games_sorted['away_team'].unique())

    latest_rolling = {}
    for team in all_teams:
        mask = (games_sorted['home_team'] == team) | (games_sorted['away_team'] == team)
        team_games = games_sorted[mask].copy()

        team_games['team_win'] = np.where(
            team_games['home_team'] == team, team_games['home_win'], 1 - team_games['home_win']
        )
        team_games['team_runs'] = np.where(
            team_games['home_team'] == team, team_games['home_score'], team_games['away_score']
        )
        team_games['opp_runs'] = np.where(
            team_games['home_team'] == team, team_games['away_score'], team_games['home_score']
        )

        recent = team_games.tail(window)
        if len(recent) >= 5:
            latest_rolling[team] = {
                'rolling_win_pct': recent['team_win'].mean(),
                'rolling_rs': recent['team_runs'].mean(),
                'rolling_ra': recent['opp_runs'].mean(),
                'rolling_run_diff': (recent['team_runs'] - recent['opp_runs']).mean(),
            }
        else:
            latest_rolling[team] = {
                'rolling_win_pct': 0.5, 'rolling_rs': 4.5,
                'rolling_ra': 4.5, 'rolling_run_diff': 0.0,
            }

    return latest_rolling


def get_api_team_stats(team_name, year, team_stats_dict):
    """チーム打撃・投手指標"""
    defaults = {'BA': 0.248, 'OBP': 0.315, 'SLG': 0.395, 'ERA': 4.12, 'WHIP': 1.28,
                'HR': 0, 'SB': 0, 'OPS': 0.710}
    if year not in team_stats_dict:
        return defaults
    ts = team_stats_dict[year]
    team_row = ts[ts['team_name'] == team_name]
    if len(team_row) == 0:
        return defaults
    r = team_row.iloc[0]
    return {
        'BA': float(r.get('BA', 0.248)), 'OBP': float(r.get('OBP', 0.315)),
        'SLG': float(r.get('SLG', 0.395)), 'ERA': float(r.get('ERA', 4.12)),
        'WHIP': float(r.get('WHIP', 1.28)), 'HR': int(r.get('HR', 0)),
        'SB': int(r.get('SB', 0)), 'OPS': float(r.get('OPS', 0.710)),
    }


def build_game_features(home, away, overall_stats, rolling_stats, team_stats_dict, latest_year):
    """37特徴量を生成"""
    home_season = overall_stats.get(home, {
        'win_pct': 0.5, 'avg_rs': 4.5, 'avg_ra': 4.5, 'run_diff': 0,
        'pythag': 0.5, 'home_win_pct': 0.54, 'away_win_pct': 0.46,
        'home_avg_rs': 4.5, 'away_avg_rs': 4.5,
    })
    away_season = overall_stats.get(away, {
        'win_pct': 0.5, 'avg_rs': 4.5, 'avg_ra': 4.5, 'run_diff': 0,
        'pythag': 0.5, 'home_win_pct': 0.54, 'away_win_pct': 0.46,
        'home_avg_rs': 4.5, 'away_avg_rs': 4.5,
    })
    home_api = get_api_team_stats(home, latest_year, team_stats_dict)
    away_api = get_api_team_stats(away, latest_year, team_stats_dict)
    home_rolling = rolling_stats.get(home, {
        'rolling_win_pct': 0.5, 'rolling_rs': 4.5, 'rolling_ra': 4.5, 'rolling_run_diff': 0
    })
    away_rolling = rolling_stats.get(away, {
        'rolling_win_pct': 0.5, 'rolling_rs': 4.5, 'rolling_ra': 4.5, 'rolling_run_diff': 0
    })

    features = {
        'win_pct_diff': home_season['win_pct'] - away_season['win_pct'],
        'pythag_diff': home_season['pythag'] - away_season['pythag'],
        'home_run_diff': home_season['run_diff'],
        'away_run_diff': away_season['run_diff'],
        'run_diff_diff': home_season['run_diff'] - away_season['run_diff'],
        'home_BA': home_api['BA'], 'away_BA': away_api['BA'],
        'BA_diff': home_api['BA'] - away_api['BA'],
        'home_OBP': home_api['OBP'], 'away_OBP': away_api['OBP'],
        'OBP_diff': home_api['OBP'] - away_api['OBP'],
        'home_SLG': home_api['SLG'], 'away_SLG': away_api['SLG'],
        'SLG_diff': home_api['SLG'] - away_api['SLG'],
        'home_ERA': home_api['ERA'], 'away_ERA': away_api['ERA'],
        'ERA_diff': away_api['ERA'] - home_api['ERA'],
        'home_WHIP': home_api['WHIP'], 'away_WHIP': away_api['WHIP'],
        'WHIP_diff': away_api['WHIP'] - home_api['WHIP'],
        'home_team_home_wpct': home_season['home_win_pct'],
        'away_team_away_wpct': away_season['away_win_pct'],
        'home_away_split_diff': home_season['home_win_pct'] - away_season['away_win_pct'],
        'home_rolling_wpct': home_rolling['rolling_win_pct'],
        'away_rolling_wpct': away_rolling['rolling_win_pct'],
        'rolling_wpct_diff': home_rolling['rolling_win_pct'] - away_rolling['rolling_win_pct'],
        'home_rolling_rs': home_rolling['rolling_rs'],
        'away_rolling_rs': away_rolling['rolling_rs'],
        'home_rolling_run_diff': home_rolling['rolling_run_diff'],
        'away_rolling_run_diff': away_rolling['rolling_run_diff'],
        'rolling_run_diff_diff': home_rolling['rolling_run_diff'] - away_rolling['rolling_run_diff'],
        'home_offensive_strength': (home_api['BA'] + home_api['OBP'] + home_api['SLG']) / 3,
        'away_offensive_strength': (away_api['BA'] + away_api['OBP'] + away_api['SLG']) / 3,
        'home_defensive_strength': 1 / (1 + home_api['ERA']) * 1 / (1 + home_api['WHIP']),
        'away_defensive_strength': 1 / (1 + away_api['ERA']) * 1 / (1 + away_api['WHIP']),
        'offensive_diff': ((home_api['BA'] + home_api['OBP'] + home_api['SLG']) -
                          (away_api['BA'] + away_api['OBP'] + away_api['SLG'])) / 3,
        'defensive_diff': (1/(1+home_api['ERA']) * 1/(1+home_api['WHIP'])) -
                         (1/(1+away_api['ERA']) * 1/(1+away_api['WHIP'])),
    }
    return features


# ========================================================
# 全マッチアップの勝率を事前計算
# ========================================================
def precompute_win_probabilities(model_data, overall_stats, rolling_stats, team_stats_dict, latest_year):
    """全30x30チームの対戦勝率をXGBoostで事前計算"""
    logger.info("Precomputing win probabilities for all matchups...")

    model = model_data['model']
    scaler = model_data['scaler']
    feature_cols = model_data['feature_cols']

    win_probs = {}

    for home in ALL_TEAMS:
        for away in ALL_TEAMS:
            if home == away:
                continue

            features = build_game_features(
                home, away, overall_stats, rolling_stats, team_stats_dict, latest_year
            )

            feat_df = pd.DataFrame([features])
            for col in set(feature_cols) - set(feat_df.columns):
                feat_df[col] = 0.0
            feat_df = feat_df[feature_cols].astype(float)
            feat_scaled = scaler.transform(feat_df)

            prob = model.predict_proba(pd.DataFrame(feat_scaled, columns=feature_cols))[0][1]
            win_probs[(home, away)] = float(prob)

    logger.info(f"  ✓ {len(win_probs)} matchup probabilities computed")
    return win_probs


# ========================================================
# モンテカルロ シーズンシミュレーション
# ========================================================
def simulate_season(schedule, win_probs, rng):
    """1シーズンをシミュレーション。各チームの勝利数を返す"""
    wins = defaultdict(int)
    losses = defaultdict(int)

    for game in schedule:
        home = game['home_team']
        away = game['away_team']
        prob = win_probs.get((home, away), 0.5)

        if rng.random() < prob:
            wins[home] += 1
            losses[away] += 1
        else:
            wins[away] += 1
            losses[home] += 1

    return dict(wins), dict(losses)


def determine_playoffs(wins):
    """
    2024+ MLBプレーオフ形式: 各リーグ6チーム = 計12チーム
    - ディビジョン優勝3チーム
    - ワイルドカード3チーム（ディビジョン優勝以外で最高勝率3チーム）
    """
    playoffs = {'AL': [], 'NL': []}

    for league, divs in LEAGUE_DIVISIONS.items():
        div_winners = []
        non_winners = []

        for div_name in divs:
            teams = DIVISIONS[div_name]
            best_team = max(teams, key=lambda t: wins.get(t, 0))
            div_winners.append(best_team)
            for t in teams:
                if t != best_team:
                    non_winners.append(t)

        # ワイルドカード: 非優勝チームから上位3チーム
        non_winners.sort(key=lambda t: wins.get(t, 0), reverse=True)
        wild_cards = non_winners[:3]

        playoffs[league] = div_winners + wild_cards

    return playoffs


def simulate_playoff_bracket(playoffs, win_probs, rng):
    """
    プレーオフをシミュレーション。
    ワイルドカードラウンド → ディビジョンシリーズ → リーグCS → ワールドシリーズ
    簡略化: 各ラウンドを単一の勝率ベース判定（7試合制を考慮）
    """
    def series_prob(team_a, team_b, games=7):
        """BO7（またはBO5）のシリーズ勝率を計算"""
        # ホームアドバンテージは上位シードに
        p = win_probs.get((team_a, team_b), 0.5)
        # 7試合制のシリーズ勝率（近似）
        wins_needed = (games + 1) // 2
        # 累積二項分布で計算
        prob_a_wins = 0
        from math import comb
        for total_games in range(wins_needed, games + 1):
            # team_aが正好total_games試合目で勝つ（最後の試合を勝つ）
            prev_games = total_games - 1
            prev_wins = wins_needed - 1
            prob_a_wins += comb(prev_games, prev_wins) * (p ** wins_needed) * ((1-p) ** (total_games - wins_needed))
        return prob_a_wins

    def play_series(team_a, team_b, games, rng):
        """シリーズをシミュレーション"""
        p = series_prob(team_a, team_b, games)
        return team_a if rng.random() < p else team_b

    results = {}

    for league in ['AL', 'NL']:
        teams = playoffs[league]
        if len(teams) < 6:
            results[f'{league}_champion'] = teams[0] if teams else None
            continue

        # Seed by wins (div winners first, then WC)
        # ワイルドカードラウンド (Best of 3)
        # #3 div winner vs #3 WC, #2 WC vs #1 WC → 上位2チームは bye なし
        # 2024形式: #1シード bye, #2 vs #WC3, #3 vs #WC2, #WC1 vs #4...
        # 簡略化: ディビジョン優勝1位=bye, 残り4チームでWCラウンド
        seed = sorted(teams[:3], key=lambda t: win_probs.get((t, teams[0]), 0.5), reverse=True)
        wc = sorted(teams[3:], key=lambda t: sum(win_probs.get((t, opp), 0.5) for opp in ALL_TEAMS if opp != t), reverse=True)

        # WC Round (Best of 3)
        wc_winner1 = play_series(seed[1], wc[2], 3, rng)
        wc_winner2 = play_series(seed[2], wc[1], 3, rng)
        wc_winner3 = play_series(wc[0], seed[0], 3, rng)  # simplified

        # ALDS/NLDS (Best of 5)
        ds_teams = [seed[0], wc_winner1, wc_winner2, wc_winner3]
        ds_winner1 = play_series(ds_teams[0], ds_teams[3], 5, rng)
        ds_winner2 = play_series(ds_teams[1], ds_teams[2], 5, rng)

        # ALCS/NLCS (Best of 7)
        league_champ = play_series(ds_winner1, ds_winner2, 7, rng)
        results[f'{league}_champion'] = league_champ

    # World Series (Best of 7)
    al_champ = results.get('AL_champion')
    nl_champ = results.get('NL_champion')
    if al_champ and nl_champ:
        p = series_prob(al_champ, nl_champ, 7)
        ws_winner = al_champ if rng.random() < p else nl_champ
        results['world_series_champion'] = ws_winner
    else:
        results['world_series_champion'] = al_champ or nl_champ

    return results


def run_simulation(n_sims=10000, seed=2026):
    """メインシミュレーション実行"""
    logger.info(f"=== Moneyball Dojo Season Simulator ===")
    logger.info(f"Running {n_sims:,} simulations...")

    # データ読み込み
    model_data = load_moneyline_model()
    games_df, team_stats, latest_year = load_team_data()

    # スタッツ計算
    overall_stats = compute_team_overall_stats(games_df, latest_year)
    rolling_stats = compute_team_rolling_stats(games_df, window=15)

    # 勝率事前計算
    win_probs = precompute_win_probabilities(
        model_data, overall_stats, rolling_stats, team_stats, latest_year
    )

    # スケジュール生成
    schedule = generate_season_schedule()

    # シミュレーション実行
    rng = np.random.default_rng(seed)

    # カウンター
    division_wins = defaultdict(int)
    playoff_appearances = defaultdict(int)
    league_championships = defaultdict(int)
    world_series_wins = defaultdict(int)
    win_totals = defaultdict(list)

    for sim in range(n_sims):
        if (sim + 1) % 1000 == 0:
            logger.info(f"  Simulation {sim + 1:,}/{n_sims:,}...")

        # レギュラーシーズン
        wins, losses = simulate_season(schedule, win_probs, rng)

        for team in ALL_TEAMS:
            win_totals[team].append(wins.get(team, 0))

        # ディビジョン優勝
        for div_name, teams in DIVISIONS.items():
            div_winner = max(teams, key=lambda t: wins.get(t, 0))
            division_wins[div_winner] += 1

        # プレーオフ
        playoffs = determine_playoffs(wins)
        for league_teams in playoffs.values():
            for team in league_teams:
                playoff_appearances[team] += 1

        # ポストシーズン
        results = simulate_playoff_bracket(playoffs, win_probs, rng)

        for league in ['AL', 'NL']:
            champ = results.get(f'{league}_champion')
            if champ:
                league_championships[champ] += 1

        ws_winner = results.get('world_series_champion')
        if ws_winner:
            world_series_wins[ws_winner] += 1

    # 結果集計
    logger.info("Compiling results...")
    results_data = {}

    for team in ALL_TEAMS:
        wins_arr = np.array(win_totals[team])
        results_data[team] = {
            'division': get_team_division(team),
            'league': get_team_league(team),
            'mean_wins': round(float(wins_arr.mean()), 1),
            'median_wins': int(np.median(wins_arr)),
            'std_wins': round(float(wins_arr.std()), 1),
            'win_range': f"{int(wins_arr.min())}-{int(wins_arr.max())}",
            'p10_wins': int(np.percentile(wins_arr, 10)),
            'p90_wins': int(np.percentile(wins_arr, 90)),
            'playoff_pct': round(playoff_appearances[team] / n_sims * 100, 1),
            'division_pct': round(division_wins[team] / n_sims * 100, 1),
            'league_pct': round(league_championships[team] / n_sims * 100, 1),
            'ws_pct': round(world_series_wins[team] / n_sims * 100, 1),
        }

    return results_data, n_sims


def format_results(results_data, n_sims):
    """結果を見やすく表示"""
    print(f"\n{'='*80}")
    print(f"MONEYBALL DOJO — 2026 MLB SEASON SIMULATION ({n_sims:,} simulations)")
    print(f"{'='*80}\n")

    # ワールドシリーズ優勝確率 Top 10
    print("## World Series Championship Probability\n")
    print(f"{'Rank':<5} {'Team':<30} {'WS %':>8} {'Pennant %':>10} {'Playoff %':>10} {'Avg W':>8}")
    print("-" * 75)

    sorted_teams = sorted(results_data.items(), key=lambda x: x[1]['ws_pct'], reverse=True)
    for rank, (team, data) in enumerate(sorted_teams, 1):
        print(f"{rank:<5} {team:<30} {data['ws_pct']:>7.1f}% {data['league_pct']:>9.1f}% "
              f"{data['playoff_pct']:>9.1f}% {data['mean_wins']:>7.1f}")

    # ディビジョン別
    print(f"\n{'='*80}")
    print("## Division Breakdown\n")

    for div_name in ['AL East', 'AL Central', 'AL West', 'NL East', 'NL Central', 'NL West']:
        print(f"\n### {div_name}")
        print(f"{'Team':<30} {'Avg W':>7} {'Div %':>8} {'PO %':>8} {'WS %':>8} {'Range (10-90)':>15}")
        print("-" * 75)

        div_teams = [(t, results_data[t]) for t in DIVISIONS[div_name]]
        div_teams.sort(key=lambda x: x[1]['mean_wins'], reverse=True)

        for team, data in div_teams:
            print(f"{team:<30} {data['mean_wins']:>6.1f} {data['division_pct']:>7.1f}% "
                  f"{data['playoff_pct']:>7.1f}% {data['ws_pct']:>7.1f}% "
                  f"{data['p10_wins']:>5}-{data['p90_wins']:<5}")

    return sorted_teams


def save_results(results_data, n_sims, output_dir=None):
    """結果をJSONとMarkdownで保存"""
    if output_dir is None:
        output_dir = PROJECT_DIR / "output" / "simulation"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON出力
    json_output = {
        'generated': datetime.utcnow().isoformat(),
        'simulations': n_sims,
        'model': 'XGBoost Moneyline v1',
        'data_through': '2025 season',
        'teams': results_data,
    }

    json_path = output_dir / "season_simulation_2026.json"
    with open(json_path, 'w') as f:
        json.dump(json_output, f, indent=2)
    logger.info(f"  ✓ JSON saved: {json_path}")

    # Markdown出力（記事用）
    sorted_teams = sorted(results_data.items(), key=lambda x: x[1]['ws_pct'], reverse=True)

    md_lines = [
        f"# 2026 MLB Season Simulation Results",
        f"",
        f"*{n_sims:,} Monte Carlo simulations using XGBoost win probability model trained on 2022-2025 data.*",
        f"",
        f"---",
        f"",
        f"## World Series Championship Probability",
        f"",
        f"| Rank | Team | WS % | Pennant % | Playoff % | Avg Wins |",
        f"|------|------|------|-----------|-----------|----------|",
    ]

    for rank, (team, data) in enumerate(sorted_teams, 1):
        md_lines.append(
            f"| {rank} | **{team}** | {data['ws_pct']:.1f}% | "
            f"{data['league_pct']:.1f}% | {data['playoff_pct']:.1f}% | "
            f"{data['mean_wins']:.1f} |"
        )

    md_lines.extend([
        f"",
        f"---",
        f"",
        f"## Division Predictions",
        f"",
    ])

    for div_name in ['AL East', 'AL Central', 'AL West', 'NL East', 'NL Central', 'NL West']:
        md_lines.extend([
            f"### {div_name}",
            f"",
            f"| Team | Avg W-L | Div % | Playoff % | WS % | Win Range (10th-90th) |",
            f"|------|---------|-------|-----------|------|----------------------|",
        ])

        div_teams = [(t, results_data[t]) for t in DIVISIONS[div_name]]
        div_teams.sort(key=lambda x: x[1]['mean_wins'], reverse=True)

        for team, data in div_teams:
            avg_losses = round(162 - data['mean_wins'], 1)
            md_lines.append(
                f"| **{team}** | {data['mean_wins']:.1f}-{avg_losses:.1f} | "
                f"{data['division_pct']:.1f}% | {data['playoff_pct']:.1f}% | "
                f"{data['ws_pct']:.1f}% | {data['p10_wins']}-{data['p90_wins']} |"
            )

        md_lines.append(f"")

    md_lines.extend([
        f"---",
        f"",
        f"*Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')} | "
        f"Model: XGBoost Moneyline | Data: 2022-2025 MLB seasons | "
        f"Simulations: {n_sims:,}*",
    ])

    md_path = output_dir / "season_simulation_2026.md"
    with open(md_path, 'w') as f:
        f.write('\n'.join(md_lines))
    logger.info(f"  ✓ Markdown saved: {md_path}")

    return json_path, md_path


# ========================================================
# メイン
# ========================================================
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Moneyball Dojo Season Simulator')
    parser.add_argument('--sims', type=int, default=10000, help='Number of simulations')
    parser.add_argument('--seed', type=int, default=2026, help='Random seed')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    args = parser.parse_args()

    results_data, n_sims = run_simulation(n_sims=args.sims, seed=args.seed)
    sorted_teams = format_results(results_data, n_sims)
    json_path, md_path = save_results(results_data, n_sims, args.output)

    print(f"\n✓ Results saved to:")
    print(f"  JSON: {json_path}")
    print(f"  Markdown: {md_path}")
