"""
Moneyball Dojo - Daily Orchestrator v3 (全市場対応)
====================================================
毎日これを1回実行するだけで:
1. 本日の試合データをMLB Stats APIから取得
2. 保存済みデータ + ローリングスタッツで特徴量を生成
3. 全6モデルで予測:
   - Moneyline (勝敗)
   - Over/Under (合計得点)
   - Run Line (-1.5スプレッド)
   - First 5 Innings (前半5回勝敗)
   - Pitcher K Props (先発投手の奪三振数)
   - Batter Props (打者のヒット/HR数)
4. エッジ（モデル確率 vs Log5ベースライン）を計算
5. Substack用の英語Daily Digestを生成（Markdown）
6. note用の日本語Daily Digestを生成（Markdown）
7. Google Sheets用のCSVを出力
8. Twitter/X投稿用のサマリーを出力

使い方:
  python3 run_daily.py              # 本日の予測を生成
  python3 run_daily.py 2026-03-27   # 指定日の予測を生成

必要パッケージ:
  pip3 install MLB-StatsAPI pandas scikit-learn xgboost gspread
"""

import os
import sys
import pickle
import json
import time
import logging
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

from ensemble_wrapper import EnsembleWrapper  # noqa: F401 — needed for pickle deserialization

# Venue classification constants (must match train_all_models.py)
_DOME_STADIUMS = {
    'Tropicana Field', 'Rogers Centre', 'loanDepot park', 'LoanDepot park',
    'Marlins Park', 'Minute Maid Park', 'Globe Life Field',
    'American Family Field', 'Chase Field', 'T-Mobile Park',
}
_HIGH_ALTITUDE = {'Coors Field': 5280}

# MLB Stats API
try:
    import statsapi
    STATSAPI_AVAILABLE = True
except ImportError:
    STATSAPI_AVAILABLE = False

# Google Sheets（オプション）
try:
    import gspread
    from google.oauth2.service_account import Credentials as ServiceAccountCredentials
    SHEETS_AVAILABLE = True
except ImportError:
    SHEETS_AVAILABLE = False

# Anthropic API（オプション）
try:
    from article_generator_template import ArticleGenerator
    ARTICLE_GEN_AVAILABLE = True
except ImportError:
    ARTICLE_GEN_AVAILABLE = False

# The Odds API（実オッズ取得、オプション）
ODDS_API_KEY = os.environ.get('ODDS_API_KEY', '')

# ========================================================
# ロギング設定
# ========================================================
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

# ========================================================
# 設定
# ========================================================
PROJECT_DIR = Path(__file__).parent
MODELS_DIR = PROJECT_DIR / "models"
DATA_DIR = PROJECT_DIR / "data"
CREDENTIALS_PATH = PROJECT_DIR / "credentials.json"
SPREADSHEET_NAME = "Moneyball Dojo DB"

# リトライ設定
MAX_RETRIES = 3
RETRY_DELAYS = [2, 4, 8]  # seconds


# ========================================================
# 1. 全モデル読み込み
# ========================================================
def load_all_models():
    """全学習済みモデルを読み込む"""
    print("[1/9] Loading all trained models...")

    models = {}

    model_files = {
        'moneyline': MODELS_DIR / 'model_moneyline.pkl',
        'over_under': MODELS_DIR / 'model_over_under.pkl',
        'run_line': MODELS_DIR / 'model_run_line.pkl',
        'f5_moneyline': MODELS_DIR / 'model_f5_moneyline.pkl',
        'pitcher_k': MODELS_DIR / 'model_pitcher_k.pkl',
        'batter_props': MODELS_DIR / 'model_batter_props.pkl',
        'nrfi': MODELS_DIR / 'model_nrfi.pkl',
        'stolen_bases': MODELS_DIR / 'model_stolen_bases.pkl',
        'pitcher_outs': MODELS_DIR / 'model_pitcher_outs.pkl',
    }

    for name, path in model_files.items():
        if path.exists():
            with open(path, 'rb') as f:
                models[name] = pickle.load(f)
            print(f"  ✓ {name}")
        else:
            print(f"  ⚠ {name} not found — skipping")

    if 'moneyline' not in models:
        # 旧model.pklを試す
        legacy = PROJECT_DIR / "model.pkl"
        if legacy.exists():
            with open(legacy, 'rb') as f:
                models['moneyline'] = pickle.load(f)
            print(f"  ✓ moneyline (legacy model.pkl)")
        else:
            print("  ❌ No moneyline model found. Run train_all_models.py first.")
            sys.exit(1)

    print(f"  → {len(models)} models loaded")
    return models


# ========================================================
# 2. 当日の試合スケジュール取得
# ========================================================
def get_todays_schedule(target_date):
    """MLB Stats APIから当日の試合スケジュールを取得"""
    print(f"[2/9] Fetching schedule for {target_date}...")

    if not STATSAPI_AVAILABLE:
        print("  ❌ statsapi not installed. Run: pip3 install MLB-StatsAPI")
        sys.exit(1)

    try:
        sched = statsapi.schedule(
            start_date=target_date,
            end_date=target_date,
            sportId=1
        )
    except Exception as e:
        print(f"  ❌ MLB API error: {e}")
        print("  → Check your internet connection")
        sys.exit(1)

    games = []
    for game in sched:
        game_type = game.get('game_type', '')
        if game_type not in ('R', 'S', 'E', 'W'):  # Regular, Spring, Exhibition, WBC
            continue

        games.append({
            'game_id': game.get('game_id', ''),
            'date': target_date,
            'home_team': game.get('home_name', ''),
            'away_team': game.get('away_name', ''),
            'home_pitcher': game.get('home_probable_pitcher', 'TBA'),
            'away_pitcher': game.get('away_probable_pitcher', 'TBA'),
            'venue': game.get('venue_name', ''),
            'game_type': game_type,
            'status': game.get('status', ''),
        })

    print(f"  ✓ Found {len(games)} games for {target_date}")
    return games


# ========================================================
# 2b. The Odds API — 実オッズ取得（オプション）
# ========================================================
TEAM_NAME_MAP_ODDS = {
    'New York Yankees': 'New York Yankees', 'Boston Red Sox': 'Boston Red Sox',
    'Tampa Bay Rays': 'Tampa Bay Rays', 'Baltimore Orioles': 'Baltimore Orioles',
    'Toronto Blue Jays': 'Toronto Blue Jays', 'New York Mets': 'New York Mets',
    'Atlanta Braves': 'Atlanta Braves', 'Washington Nationals': 'Washington Nationals',
    'Philadelphia Phillies': 'Philadelphia Phillies', 'Miami Marlins': 'Miami Marlins',
    'Los Angeles Dodgers': 'Los Angeles Dodgers', 'San Diego Padres': 'San Diego Padres',
    'San Francisco Giants': 'San Francisco Giants', 'Arizona Diamondbacks': 'Arizona Diamondbacks',
    'Colorado Rockies': 'Colorado Rockies', 'Milwaukee Brewers': 'Milwaukee Brewers',
    'Chicago Cubs': 'Chicago Cubs', 'St. Louis Cardinals': 'St. Louis Cardinals',
    'Pittsburgh Pirates': 'Pittsburgh Pirates', 'Cincinnati Reds': 'Cincinnati Reds',
    'Houston Astros': 'Houston Astros', 'Los Angeles Angels': 'Los Angeles Angels',
    'Oakland Athletics': 'Oakland Athletics', 'Seattle Mariners': 'Seattle Mariners',
    'Texas Rangers': 'Texas Rangers', 'Kansas City Royals': 'Kansas City Royals',
    'Minnesota Twins': 'Minnesota Twins', 'Chicago White Sox': 'Chicago White Sox',
    'Detroit Tigers': 'Detroit Tigers', 'Cleveland Guardians': 'Cleveland Guardians',
}

def _american_to_implied(odds: int) -> float:
    """Convert American odds to implied probability (no-vig)."""
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    else:
        return 100 / (odds + 100)

def fetch_market_odds() -> dict:
    """Fetch real moneyline odds from The Odds API.

    Returns dict: { (home_team, away_team): {'home_implied': float, 'away_implied': float, 'source': str} }
    Returns empty dict if API key not set or request fails.
    """
    if not ODDS_API_KEY:
        return {}

    try:
        import requests
    except ImportError:
        return {}

    url = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"
    params = {
        'apiKey': ODDS_API_KEY,
        'regions': 'us',
        'markets': 'h2h',
        'oddsFormat': 'american',
    }

    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                odds_map = {}
                for game in data:
                    home = game.get('home_team', '')
                    away = game.get('away_team', '')
                    # Use consensus (average across books) for stability
                    home_odds_list = []
                    away_odds_list = []
                    for book in game.get('bookmakers', []):
                        h2h = next((m for m in book['markets'] if m['key'] == 'h2h'), None)
                        if h2h:
                            for outcome in h2h['outcomes']:
                                if outcome['name'] == home:
                                    home_odds_list.append(outcome['price'])
                                elif outcome['name'] == away:
                                    away_odds_list.append(outcome['price'])
                    if home_odds_list and away_odds_list:
                        # Consensus implied probability (average of all books, then remove vig)
                        home_imp = np.mean([_american_to_implied(o) for o in home_odds_list])
                        away_imp = np.mean([_american_to_implied(o) for o in away_odds_list])
                        total_imp = home_imp + away_imp
                        # Remove vig (normalize to sum to 1.0)
                        home_imp_nv = home_imp / total_imp
                        away_imp_nv = away_imp / total_imp
                        odds_map[(home, away)] = {
                            'home_implied': round(home_imp_nv, 4),
                            'away_implied': round(away_imp_nv, 4),
                            'source': 'market',
                            'n_books': len(home_odds_list),
                        }
                logger.info(f"Fetched market odds for {len(odds_map)} games ({len(data)} total from API)")
                return odds_map
            elif resp.status_code == 401:
                logger.warning("Odds API: invalid API key")
                return {}
            elif resp.status_code == 429:
                logger.warning("Odds API: rate limited")
            else:
                logger.warning(f"Odds API: HTTP {resp.status_code}")
        except Exception as e:
            logger.warning(f"Odds API attempt {attempt+1} failed: {e}")
        if attempt < 2:
            time.sleep(2 ** attempt)

    return {}


# ========================================================
# 3. 保存済みデータ読み込み
# ========================================================
def load_historical_data():
    """data/ディレクトリの保存済みデータを読み込む"""
    print("[3/9] Loading historical data...")

    # 試合データ
    games_path = DATA_DIR / "games_2022_2025.csv"
    if not games_path.exists():
        games_path = DATA_DIR / "games_2022_2024.csv"
    
    if not games_path.exists():
        print(f"  ❌ data/games_2022_2025.csv not found!")
        print("  → Run: python3 fetch_real_data.py")
        sys.exit(1)

    games_df = pd.read_csv(games_path)
    games_df['date'] = pd.to_datetime(games_df['date'])
    print(f"  ✓ Loaded {len(games_df)} historical games")

    # チームスタッツ
    team_stats = {}
    for year in [2022, 2023, 2024, 2025, 2026]:
        path = DATA_DIR / f"team_stats_{year}.csv"
        if path.exists():
            team_stats[year] = pd.read_csv(path)

    latest_year = max(team_stats.keys()) if team_stats else 2024
    print(f"  ✓ Team stats: {sorted(team_stats.keys())}")

    # 投手データ（K props用）
    pitcher_stats = {}
    for year in [2022, 2023, 2024, 2025, 2026]:
        path = DATA_DIR / f"pitcher_stats_{year}.csv"
        if path.exists():
            pitcher_stats[year] = pd.read_csv(path)

    if pitcher_stats:
        latest_pitcher_year = max(pitcher_stats.keys())
        print(f"  ✓ Pitcher stats: {sorted(pitcher_stats.keys())}")
    else:
        latest_pitcher_year = None
        print("  ⚠ No pitcher stats — K Props will be skipped")

    # 打者データ（batter props用）
    batter_stats = {}
    for year in [2022, 2023, 2024, 2025, 2026]:
        path = DATA_DIR / f"batter_stats_{year}.csv"
        if path.exists():
            batter_stats[year] = pd.read_csv(path)

    if batter_stats:
        latest_batter_year = max(batter_stats.keys())
        print(f"  ✓ Batter stats: {sorted(batter_stats.keys())}")
    else:
        latest_batter_year = None
        print("  ⚠ No batter stats — Batter Props will be skipped")

    return games_df, team_stats, latest_year, pitcher_stats, latest_pitcher_year, batter_stats, latest_batter_year


# ========================================================
# 4. チームスタッツ計算
# ========================================================
def compute_team_overall_stats(games_df, year):
    """指定年のチーム全体成績を計算.

    NOTE: In live daily usage, games_df contains only completed games (up to yesterday),
    so this computes stats from past data only — no future data leakage.
    Do NOT use this for backtesting where games_df contains the full season.
    """
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
    """各チームの直近N試合のローリングスタッツ"""
    games_sorted = games_df.sort_values('date').reset_index(drop=True)
    all_teams = set(games_sorted['home_team'].unique()) | set(games_sorted['away_team'].unique())

    latest_rolling = {}

    for team in all_teams:
        mask = (games_sorted['home_team'] == team) | (games_sorted['away_team'] == team)
        team_games = games_sorted[mask].copy()

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
    """API取得済みのチーム打撃・投手指標"""
    defaults = {'BA': 0.248, 'OBP': 0.315, 'SLG': 0.395, 'ERA': 4.12, 'WHIP': 1.28, 'HR': 0, 'SB': 0, 'OPS': 0.710}

    if year not in team_stats_dict:
        return defaults

    ts = team_stats_dict[year]
    team_row = ts[ts['team_name'] == team_name]
    if len(team_row) == 0:
        return defaults

    r = team_row.iloc[0]
    return {
        'BA': float(r.get('BA', 0.248)),
        'OBP': float(r.get('OBP', 0.315)),
        'SLG': float(r.get('SLG', 0.395)),
        'ERA': float(r.get('ERA', 4.12)),
        'WHIP': float(r.get('WHIP', 1.28)),
        'HR': int(r.get('HR', 0)),
        'SB': int(r.get('SB', 0)),
        'OPS': float(r.get('OPS', 0.710)),
    }


# ========================================================
# 5. 特徴量生成 + 全モデル予測
# ========================================================
def _get_sp_stats(pitcher_name, year, pitcher_stats_dict):
    """Get starting pitcher individual stats with fallback defaults."""
    defaults = {'ERA': 4.12, 'WHIP': 1.28, 'K_per_9': 8.0, 'BB_per_9': 3.0}
    if not isinstance(pitcher_name, str) or not pitcher_name or pitcher_name == 'TBA':
        return defaults
    if year not in pitcher_stats_dict:
        return defaults
    ps = pitcher_stats_dict[year]
    match = ps[ps['name'] == pitcher_name]
    if len(match) == 0:
        last_name = pitcher_name.split()[-1]
        match = ps[ps['name'].str.contains(last_name, case=False, na=False)]
    if len(match) == 0:
        return defaults
    p = match.iloc[0]
    return {
        'ERA': float(p.get('ERA', 4.12)),
        'WHIP': float(p.get('WHIP', 1.28)),
        'K_per_9': float(p.get('K_per_9', 8.0)),
        'BB_per_9': float(p.get('BB_per_9', 3.0)),
    }


def build_game_features(home, away, overall_stats, rolling_stats, team_stats_dict, latest_year,
                        home_pitcher=None, away_pitcher=None, pitcher_stats_dict=None, **kwargs):
    """1試合分のゲームレベル特徴量を生成（先発投手+球場+休養日+移動距離）"""
    home_season = overall_stats[home]
    away_season = overall_stats[away]
    home_api = get_api_team_stats(home, latest_year, team_stats_dict)
    away_api = get_api_team_stats(away, latest_year, team_stats_dict)
    home_rolling = rolling_stats.get(home, {'rolling_win_pct': 0.5, 'rolling_rs': 4.5, 'rolling_ra': 4.5, 'rolling_run_diff': 0})
    away_rolling = rolling_stats.get(away, {'rolling_win_pct': 0.5, 'rolling_rs': 4.5, 'rolling_ra': 4.5, 'rolling_run_diff': 0})

    # Starting pitcher individual stats
    hp = _get_sp_stats(home_pitcher, latest_year, pitcher_stats_dict or {})
    ap = _get_sp_stats(away_pitcher, latest_year, pitcher_stats_dict or {})

    features = {
        # Season-level
        'win_pct_diff': home_season['win_pct'] - away_season['win_pct'],
        'pythag_diff': home_season['pythag'] - away_season['pythag'],
        'home_run_diff': home_season['run_diff'],
        'away_run_diff': away_season['run_diff'],
        'run_diff_diff': home_season['run_diff'] - away_season['run_diff'],
        # Batting
        'home_BA': home_api['BA'], 'away_BA': away_api['BA'],
        'BA_diff': home_api['BA'] - away_api['BA'],
        'home_OBP': home_api['OBP'], 'away_OBP': away_api['OBP'],
        'OBP_diff': home_api['OBP'] - away_api['OBP'],
        'home_SLG': home_api['SLG'], 'away_SLG': away_api['SLG'],
        'SLG_diff': home_api['SLG'] - away_api['SLG'],
        # Pitching
        'home_ERA': home_api['ERA'], 'away_ERA': away_api['ERA'],
        'ERA_diff': away_api['ERA'] - home_api['ERA'],
        'home_WHIP': home_api['WHIP'], 'away_WHIP': away_api['WHIP'],
        'WHIP_diff': away_api['WHIP'] - home_api['WHIP'],
        # Home/Away splits
        'home_team_home_wpct': home_season['home_win_pct'],
        'away_team_away_wpct': away_season['away_win_pct'],
        'home_away_split_diff': home_season['home_win_pct'] - away_season['away_win_pct'],
        # Rolling
        'home_rolling_wpct': home_rolling['rolling_win_pct'],
        'away_rolling_wpct': away_rolling['rolling_win_pct'],
        'rolling_wpct_diff': home_rolling['rolling_win_pct'] - away_rolling['rolling_win_pct'],
        'home_rolling_rs': home_rolling['rolling_rs'],
        'away_rolling_rs': away_rolling['rolling_rs'],
        'home_rolling_run_diff': home_rolling['rolling_run_diff'],
        'away_rolling_run_diff': away_rolling['rolling_run_diff'],
        'rolling_run_diff_diff': home_rolling['rolling_run_diff'] - away_rolling['rolling_run_diff'],
        # Composite
        'home_offensive_strength': (home_api['BA'] + home_api['OBP'] + home_api['SLG']) / 3,
        'away_offensive_strength': (away_api['BA'] + away_api['OBP'] + away_api['SLG']) / 3,
        'home_defensive_strength': 1 / (1 + home_api['ERA']) * 1 / (1 + home_api['WHIP']),
        'away_defensive_strength': 1 / (1 + away_api['ERA']) * 1 / (1 + away_api['WHIP']),
        'offensive_diff': ((home_api['BA'] + home_api['OBP'] + home_api['SLG']) -
                          (away_api['BA'] + away_api['OBP'] + away_api['SLG'])) / 3,
        'defensive_diff': (1/(1+home_api['ERA']) * 1/(1+home_api['WHIP'])) -
                         (1/(1+away_api['ERA']) * 1/(1+away_api['WHIP'])),
        # Starting pitcher individual features
        'sp_home_era': hp['ERA'], 'sp_away_era': ap['ERA'],
        'sp_era_diff': ap['ERA'] - hp['ERA'],
        'sp_home_whip': hp['WHIP'], 'sp_away_whip': ap['WHIP'],
        'sp_whip_diff': ap['WHIP'] - hp['WHIP'],
        'sp_home_k9': hp['K_per_9'], 'sp_away_k9': ap['K_per_9'],
        'sp_k9_diff': hp['K_per_9'] - ap['K_per_9'],
        'sp_home_bb9': hp['BB_per_9'], 'sp_away_bb9': ap['BB_per_9'],
        'sp_bb9_diff': ap['BB_per_9'] - hp['BB_per_9'],
        'sp_home_era_x_opp_obp': hp['ERA'] * away_api['OBP'],
        'sp_away_era_x_opp_obp': ap['ERA'] * home_api['OBP'],
        'sp_home_dominance': hp['K_per_9'] / max(hp['ERA'], 0.5),
        'sp_away_dominance': ap['K_per_9'] / max(ap['ERA'], 0.5),
        'sp_dominance_diff': (hp['K_per_9'] / max(hp['ERA'], 0.5)) - (ap['K_per_9'] / max(ap['ERA'], 0.5)),
        'sp_combined_era': (hp['ERA'] + ap['ERA']) / 2,
        'sp_combined_whip': (hp['WHIP'] + ap['WHIP']) / 2,
        # Park Factor & venue features
        'park_factor': kwargs.get('park_factor', 1.0),
        'is_dome': kwargs.get('is_dome', 0),
        'altitude': kwargs.get('altitude', 0),
        # Rest days & travel (defaults for prediction time)
        'home_rest_days': kwargs.get('home_rest_days', 1),
        'away_rest_days': kwargs.get('away_rest_days', 1),
        'rest_diff': kwargs.get('home_rest_days', 1) - kwargs.get('away_rest_days', 1),
        'away_travel_miles': kwargs.get('away_travel_miles', 0),
        'home_travel_miles': kwargs.get('home_travel_miles', 0),
        'travel_diff': kwargs.get('away_travel_miles', 0) - kwargs.get('home_travel_miles', 0),
    }

    return features, home_season, away_season, home_api, away_api, home_rolling, away_rolling


def predict_with_model(model_data, features, feature_cols_override=None):
    """汎用モデル予測関数"""
    model = model_data['model']
    scaler = model_data['scaler']
    feature_cols = feature_cols_override or model_data['feature_cols']

    feat_df = pd.DataFrame([features])

    # 不足/余分列の調整
    missing = set(feature_cols) - set(feat_df.columns)
    for col in missing:
        feat_df[col] = 0.0
    extra = set(feat_df.columns) - set(feature_cols)
    if extra:
        feat_df = feat_df.drop(columns=list(extra))

    feat_df = feat_df[feature_cols].astype(float)
    feat_scaled = scaler.transform(feat_df)
    feat_scaled_df = pd.DataFrame(feat_scaled, columns=feature_cols)

    return model, feat_scaled_df


def build_ou_extra_features(home_api, away_api, home_season, away_season, home_rolling, away_rolling):
    """Over/Under用の追加特徴量"""
    return {
        'combined_avg_rs': (home_season['avg_rs'] + away_season['avg_rs']) / 2,
        'combined_avg_ra': (home_season['avg_ra'] + away_season['avg_ra']) / 2,
        'total_expected_runs': home_season['avg_rs'] + away_season['avg_rs'],
        'combined_ERA': (home_api['ERA'] + away_api['ERA']) / 2,
        'combined_WHIP': (home_api['WHIP'] + away_api['WHIP']) / 2,
        'combined_BA': (home_api['BA'] + away_api['BA']) / 2,
        'combined_OBP': (home_api['OBP'] + away_api['OBP']) / 2,
        'combined_SLG': (home_api['SLG'] + away_api['SLG']) / 2,
        'combined_OPS': (home_api['OPS'] + away_api['OPS']) / 2,
        'scoring_environment': (home_season['avg_rs'] + away_season['avg_rs'] +
                               home_season['avg_ra'] + away_season['avg_ra']) / 4,
        'pitching_quality': -(home_api['ERA'] + away_api['ERA']) / 2,
        'hitting_power': (home_api['SLG'] + away_api['SLG']) / 2,
        'combined_rolling_rs': (home_rolling['rolling_rs'] + away_rolling['rolling_rs']) / 2,
        'combined_rolling_ra': (home_rolling['rolling_ra'] + away_rolling['rolling_ra']) / 2,
        'rolling_total_runs': home_rolling['rolling_rs'] + away_rolling['rolling_rs'],
    }


def generate_all_predictions(games, models, games_df, team_stats_dict, latest_year,
                              pitcher_stats, latest_pitcher_year,
                              batter_stats, latest_batter_year,
                              market_odds=None):
    """全6モデルで全試合の予測を生成"""
    print("[5/9] Generating predictions with all models...")

    # Spring Training shrinkage: regress probabilities toward 0.5
    # ST games use non-standard lineups, short pitching, split squads — model signal is weaker
    ST_SHRINK = 0.5  # multiply deviation from 0.5 by this factor for game_type='S'

    def st_adjust(prob, game_type):
        """Shrink probability toward 0.5 for Spring Training games."""
        if game_type == 'S':
            return 0.5 + (prob - 0.5) * ST_SHRINK
        return prob

    overall_stats = compute_team_overall_stats(games_df, latest_year)
    rolling_stats = compute_team_rolling_stats(games_df, window=15)

    # Compute park factors from historical data
    games_df_copy = games_df.copy()
    games_df_copy['total_runs'] = games_df_copy['home_score'] + games_df_copy['away_score']
    pf_cache = {}
    for yr in games_df_copy['year'].unique():
        yg = games_df_copy[games_df_copy['year'] == yr]
        league_avg = yg['total_runs'].mean()
        if league_avg == 0:
            continue
        for venue in yg['venue'].dropna().unique():
            vg = yg[yg['venue'] == venue]
            n = len(vg)
            raw_pf = vg['total_runs'].mean() / league_avg
            shrink = min(n, 80) / 80
            pf_cache[(venue, yr)] = 1.0 + (raw_pf - 1.0) * shrink

    all_predictions = []
    skipped = 0

    for game in games:
        home = game['home_team']
        away = game['away_team']

        if home not in overall_stats or away not in overall_stats:
            print(f"  ⚠ Unknown team: {home} or {away} — skipping")
            skipped += 1
            continue

        # Park factor for this venue
        venue = game.get('venue', '')
        pf = pf_cache.get((venue, latest_year), 1.0)

        # ゲームレベル特徴量（先発投手+球場+休養日+移動距離含む）
        features, home_season, away_season, home_api, away_api, home_rolling, away_rolling = \
            build_game_features(home, away, overall_stats, rolling_stats, team_stats_dict, latest_year,
                                home_pitcher=game.get('home_pitcher'),
                                away_pitcher=game.get('away_pitcher'),
                                pitcher_stats_dict=pitcher_stats if pitcher_stats else {},
                                park_factor=pf,
                                is_dome=1 if venue in _DOME_STADIUMS else 0,
                                altitude=_HIGH_ALTITUDE.get(venue, 0) / 5280)

        pred = {**game}

        # --- MONEYLINE ---
        if 'moneyline' in models:
            try:
                ml_model, ml_feat = predict_with_model(models['moneyline'], features)
                home_prob = ml_model.predict_proba(ml_feat)[0][1]
                home_prob = st_adjust(home_prob, game.get('game_type', 'R'))
                pred['ml_pick'] = 'HOME' if home_prob > 0.5 else 'AWAY'

                # Store probability of the PICKED team winning
                pick_prob = home_prob if pred['ml_pick'] == 'HOME' else 1 - home_prob
                pred['ml_prob'] = round(float(pick_prob), 4)

                # Edge = model probability - market/baseline probability
                # Use REAL market odds if available (The Odds API), else Log5 fallback
                mkt = (market_odds or {}).get((home, away))
                if mkt:
                    home_baseline = mkt['home_implied']
                    pred['ml_odds_source'] = 'market'
                    pred['ml_n_books'] = mkt.get('n_books', 0)
                else:
                    # Log5 fallback (no market data)
                    hw = home_season['win_pct']
                    aw = away_season['win_pct']
                    denom = hw + aw - 2 * hw * aw
                    home_baseline = (hw - hw * aw) / denom if denom != 0 else 0.5
                    home_baseline = max(0.25, min(0.75, home_baseline))
                    pred['ml_odds_source'] = 'log5'
                pick_baseline = home_baseline if pred['ml_pick'] == 'HOME' else 1 - home_baseline
                pred['ml_implied'] = round(float(pick_baseline), 4)
                pred['ml_edge'] = round(float(pick_prob - pick_baseline), 4)

                abs_edge = abs(pred['ml_edge'])
                if abs_edge >= 0.08:
                    pred['ml_confidence'] = 'STRONG'
                elif abs_edge >= 0.04:
                    pred['ml_confidence'] = 'MODERATE'
                elif abs(home_prob - 0.5) < 0.03:
                    pred['ml_confidence'] = 'PASS'
                else:
                    pred['ml_confidence'] = 'LEAN'

                if pred['ml_confidence'] == 'PASS':
                    pred['ml_pick'] = 'PASS'
            except Exception as e:
                pred['ml_prob'] = 0.5
                pred['ml_pick'] = 'N/A'
                pred['ml_edge'] = 0.0
                pred['ml_confidence'] = 'N/A'

        # --- OVER/UNDER ---
        if 'over_under' in models:
            try:
                ou_data = models['over_under']
                ou_extra = build_ou_extra_features(home_api, away_api, home_season, away_season, home_rolling, away_rolling)
                ou_features = {**features, **ou_extra}

                # Regression model for total runs
                reg_model = ou_data['regressor']  # regressor
                reg_scaler = ou_data['scaler']
                reg_feat_cols = ou_data['feature_cols']

                reg_df = pd.DataFrame([ou_features])
                for col in set(reg_feat_cols) - set(reg_df.columns):
                    reg_df[col] = 0.0
                reg_df = reg_df[reg_feat_cols].astype(float)
                reg_scaled = reg_scaler.transform(reg_df)
                pred_total = reg_model.predict(pd.DataFrame(reg_scaled, columns=reg_feat_cols))[0]

                pred['ou_predicted_total'] = round(float(pred_total), 2)

                # Classify over/under at common lines
                for line in [7.5, 8.0, 8.5, 9.0, 9.5]:
                    margin = pred_total - line
                    if abs(margin) > 1.5:
                        conf = 'STRONG'
                    elif abs(margin) > 0.75:
                        conf = 'MODERATE'
                    elif abs(margin) > 0.3:
                        conf = 'LEAN'
                    else:
                        conf = 'PASS'
                    side = 'OVER' if margin > 0 else 'UNDER'
                    pred[f'ou_{line}_pick'] = side if conf != 'PASS' else 'PASS'
                    pred[f'ou_{line}_confidence'] = conf
                    pred[f'ou_{line}_margin'] = round(float(margin), 2)

            except Exception as e:
                pred['ou_predicted_total'] = None

        # --- RUN LINE ---
        if 'run_line' in models:
            try:
                rl_model, rl_feat = predict_with_model(models['run_line'], features)
                prob = rl_model.predict_proba(rl_feat)[0][1]
                prob = st_adjust(prob, game.get('game_type', 'R'))
                pred['rl_covers_prob'] = round(float(prob), 4)
                pred['rl_pick'] = 'HOME -1.5' if prob > 0.5 else 'AWAY +1.5'

                # Historical base rate: home team covers -1.5 ~35.7% of the time.
                # Confidence reflects deviation from that base rate, not from 0.5.
                RL_BASE_RATE = 0.357
                abs_edge = abs(prob - RL_BASE_RATE)
                if abs_edge >= 0.12:
                    pred['rl_confidence'] = 'STRONG'
                elif abs_edge >= 0.06:
                    pred['rl_confidence'] = 'MODERATE'
                elif abs_edge < 0.02:
                    pred['rl_confidence'] = 'PASS'
                else:
                    pred['rl_confidence'] = 'LEAN'

                if pred['rl_confidence'] == 'PASS':
                    pred['rl_pick'] = 'PASS'
            except Exception as e:
                pred['rl_covers_prob'] = 0.5
                pred['rl_pick'] = 'N/A'
                pred['rl_confidence'] = 'N/A'

        # --- FIRST 5 INNINGS ---
        if 'f5_moneyline' in models:
            try:
                f5_model, f5_feat = predict_with_model(models['f5_moneyline'], features)
                prob = f5_model.predict_proba(f5_feat)[0][1]
                prob = st_adjust(prob, game.get('game_type', 'R'))
                pred['f5_prob'] = round(float(prob), 4)
                pred['f5_pick'] = 'HOME' if prob > 0.5 else 'AWAY'

                abs_edge = abs(prob - 0.5)
                if abs_edge >= 0.08:
                    pred['f5_confidence'] = 'STRONG'
                elif abs_edge >= 0.04:
                    pred['f5_confidence'] = 'MODERATE'
                elif abs_edge < 0.03:
                    pred['f5_confidence'] = 'PASS'
                else:
                    pred['f5_confidence'] = 'LEAN'

                if pred['f5_confidence'] == 'PASS':
                    pred['f5_pick'] = 'PASS'
            except Exception as e:
                pred['f5_prob'] = 0.5
                pred['f5_pick'] = 'N/A'
                pred['f5_confidence'] = 'N/A'

        # --- PITCHER K PROPS ---
        if 'pitcher_k' in models and latest_pitcher_year:
            pitcher_df = pitcher_stats[latest_pitcher_year]

            for side, pitcher_name_key in [('home', 'home_pitcher'), ('away', 'away_pitcher')]:
                pitcher_name = game.get(pitcher_name_key, 'TBA')
                if pitcher_name == 'TBA' or not pitcher_name:
                    pred[f'{side}_pitcher_k_pred'] = None
                    continue

                try:
                    pk_data = models['pitcher_k']
                    pk_model = pk_data['model']
                    pk_scaler = pk_data['scaler']
                    pk_feat_cols = pk_data['feature_cols']

                    # 投手データをマッチ
                    p_row = pitcher_df[pitcher_df['name'].str.contains(pitcher_name.split()[-1], case=False, na=False)]
                    if len(p_row) == 0:
                        pred[f'{side}_pitcher_k_pred'] = None
                        continue

                    p = p_row.iloc[0]
                    pk_features = {}
                    for col in pk_feat_cols:
                        if col in p.index:
                            pk_features[col] = float(p[col])
                        elif col == 'K_BB_ratio':
                            k9 = float(p.get('K_per_9', 8.0))
                            bb9 = float(p.get('BB_per_9', 3.0))
                            pk_features[col] = k9 / bb9 if bb9 > 0 else 3.0
                        elif col == 'hits_per_9':
                            ip = float(p.get('innings', 100))
                            ha = float(p.get('hits_allowed', 150))
                            pk_features[col] = ha * 9 / ip if ip > 0 else 9.0
                        else:
                            pk_features[col] = 0.0

                    pk_df = pd.DataFrame([pk_features])[pk_feat_cols].astype(float)
                    pk_scaled = pk_scaler.transform(pk_df)
                    k_pred = pk_model.predict(pd.DataFrame(pk_scaled, columns=pk_feat_cols))[0]

                    pred[f'{side}_pitcher_k_pred'] = round(float(k_pred), 2)
                    pred[f'{side}_pitcher_name'] = pitcher_name

                    # K Props lines
                    for line in [4.5, 5.5, 6.5, 7.5]:
                        margin = k_pred - line
                        if abs(margin) > 1.5:
                            conf = 'STRONG'
                        elif abs(margin) > 0.75:
                            conf = 'MODERATE'
                        elif abs(margin) > 0.3:
                            conf = 'LEAN'
                        else:
                            conf = 'PASS'
                        side_pick = 'OVER' if margin > 0 else 'UNDER'
                        pred[f'{side}_k_{line}_pick'] = side_pick if conf != 'PASS' else 'PASS'
                        pred[f'{side}_k_{line}_conf'] = conf

                except Exception:
                    pred[f'{side}_pitcher_k_pred'] = None

        # --- BATTER PROPS ---
        if 'batter_props' in models and latest_batter_year:
            # バッターは多すぎるので、各チーム上位3打者のみ予測
            batter_df = batter_stats[latest_batter_year]

            for side, team_name in [('home', home), ('away', away)]:
                try:
                    bp_data = models['batter_props']
                    hit_model = bp_data['hit_model']   # hit model
                    hr_model = bp_data.get('hr_model')  # hr model
                    bp_scaler = bp_data['scaler']
                    bp_feat_cols = bp_data['feature_cols']

                    # チームの打者を取得
                    team_batters = batter_df[batter_df['team_name'].str.contains(team_name.split()[-1], case=False, na=False)]
                    if len(team_batters) == 0:
                        continue

                    # OPS上位3人
                    top_batters = team_batters.nlargest(3, 'OPS') if 'OPS' in team_batters.columns else team_batters.head(3)

                    batter_preds = []
                    for _, b in top_batters.iterrows():
                        bp_features = {}
                        for col in bp_feat_cols:
                            if col in b.index:
                                bp_features[col] = float(b[col])
                            elif col == 'ISO':
                                bp_features[col] = float(b.get('SLG', 0.395)) - float(b.get('BA', 0.248))
                            elif col == 'BB_rate':
                                pa = float(b.get('plate_appearances', 500))
                                bb = float(b.get('walks', 50))
                                bp_features[col] = bb / pa if pa > 0 else 0.08
                            elif col == 'AB_per_HR':
                                ab = float(b.get('at_bats', 500))
                                hr = float(b.get('home_runs', 15))
                                bp_features[col] = ab / hr if hr > 0 else 40
                            else:
                                bp_features[col] = 0.0

                        bp_df = pd.DataFrame([bp_features])[bp_feat_cols].astype(float)
                        bp_scaled = bp_scaler.transform(bp_df)
                        bp_scaled_df = pd.DataFrame(bp_scaled, columns=bp_feat_cols)

                        hit_pred = hit_model.predict(bp_scaled_df)[0]
                        hr_pred = hr_model.predict(bp_scaled_df)[0] if hr_model else 0

                        batter_preds.append({
                            'name': b.get('name', 'Unknown'),
                            'hits_pred': round(float(hit_pred), 3),
                            'hr_pred': round(float(hr_pred), 4),
                        })

                    pred[f'{side}_batter_preds'] = batter_preds

                except Exception:
                    pass

        # --- NRFI/YRFI (v2: with pitcher + park factors) ---
        if 'nrfi' in models:
            try:
                nrfi_data = models['nrfi']
                nrfi_model = nrfi_data['model']
                nrfi_scaler = nrfi_data['scaler']
                nrfi_feat_cols = nrfi_data['feature_cols']

                # 先発投手データの取得
                def _get_pitcher_stats(pitcher_name, year, pitcher_stats_dict):
                    defaults = {'ERA': 4.12, 'WHIP': 1.28, 'K_per_9': 8.5}
                    if not isinstance(pitcher_name, str) or not pitcher_name or pitcher_name == 'TBA':
                        return defaults
                    if year not in pitcher_stats_dict:
                        return defaults
                    ps = pitcher_stats_dict[year]
                    match = ps[ps['name'] == pitcher_name]
                    if len(match) == 0:
                        last_name = pitcher_name.split()[-1]
                        match = ps[ps['name'].str.contains(last_name, case=False, na=False)]
                    if len(match) == 0:
                        return defaults
                    p = match.iloc[0]
                    return {
                        'ERA': float(p.get('ERA', 4.12)),
                        'WHIP': float(p.get('WHIP', 1.28)),
                        'K_per_9': float(p.get('K_per_9', 8.5)),
                    }

                # 投手データ取得
                hp_name = game.get('home_pitcher', 'TBA')
                ap_name = game.get('away_pitcher', 'TBA')
                hp = _get_pitcher_stats(hp_name, latest_year, pitcher_stats if pitcher_stats else {})
                ap = _get_pitcher_stats(ap_name, latest_year, pitcher_stats if pitcher_stats else {})

                # Store pitcher stats in pred for article generation
                pred['home_pitcher_era'] = hp['ERA']
                pred['home_pitcher_whip'] = hp['WHIP']
                pred['home_pitcher_k9'] = hp['K_per_9']
                pred['away_pitcher_era'] = ap['ERA']
                pred['away_pitcher_whip'] = ap['WHIP']
                pred['away_pitcher_k9'] = ap['K_per_9']

                nrfi_features = {
                    # 1回特化チーム統計（リアルタイムでは簡易推定）
                    'home_1st_rs': home_season['avg_rs'] * 0.11,
                    'home_1st_ra': home_season['avg_ra'] * 0.11,
                    'away_1st_rs': away_season['avg_rs'] * 0.11,
                    'away_1st_ra': away_season['avg_ra'] * 0.11,
                    'combined_1st_ra': (home_season['avg_ra'] + away_season['avg_ra']) * 0.055,
                    'combined_1st_rs': (home_season['avg_rs'] + away_season['avg_rs']) * 0.055,
                    # 先発投手
                    'home_starter_era': hp['ERA'],
                    'away_starter_era': ap['ERA'],
                    'home_starter_whip': hp['WHIP'],
                    'away_starter_whip': ap['WHIP'],
                    'home_starter_k9': hp['K_per_9'],
                    'away_starter_k9': ap['K_per_9'],
                    'starter_era_diff': hp['ERA'] - ap['ERA'],
                    'starter_whip_diff': hp['WHIP'] - ap['WHIP'],
                    'combined_starter_era': (hp['ERA'] + ap['ERA']) / 2,
                    # 球場（リアルタイムではリーグ平均を使用）
                    'venue_nrfi_rate': 0.511,
                    # チーム統計
                    'home_era': home_api['ERA'],
                    'away_era': away_api['ERA'],
                    'home_obp': home_api['OBP'],
                    'away_obp': away_api['OBP'],
                    'total_runs_per_game': home_season['avg_rs'] + away_season['avg_rs'],
                }

                nrfi_df = pd.DataFrame([nrfi_features])
                for col in set(nrfi_feat_cols) - set(nrfi_df.columns):
                    nrfi_df[col] = 0.0
                nrfi_df = nrfi_df[nrfi_feat_cols].astype(float)
                nrfi_scaled = nrfi_scaler.transform(nrfi_df)
                prob = nrfi_model.predict_proba(pd.DataFrame(nrfi_scaled, columns=nrfi_feat_cols))[0][1]
                prob = st_adjust(prob, game.get('game_type', 'R'))

                pred['nrfi_prob'] = round(float(prob), 4)
                pred['nrfi_pick'] = 'NRFI' if prob > 0.5 else 'YRFI'

                abs_edge = abs(prob - 0.5)
                if abs_edge >= 0.10:
                    pred['nrfi_confidence'] = 'STRONG'
                elif abs_edge >= 0.05:
                    pred['nrfi_confidence'] = 'MODERATE'
                else:
                    pred['nrfi_confidence'] = 'LEAN'
            except Exception:
                pred['nrfi_pick'] = 'N/A'

        # --- STOLEN BASES ---
        if 'stolen_bases' in models:
            try:
                sb_data = models['stolen_bases']
                sb_model = sb_data['model']
                sb_scaler = sb_data['scaler']
                sb_feat_cols = sb_data['feature_cols']

                sb_features = {
                    'home_sb_per_game': home_season.get('sb_per_game', 0.7),
                    'away_sb_per_game': away_season.get('sb_per_game', 0.7),
                    'combined_sb_per_game': home_season.get('sb_per_game', 0.7) + away_season.get('sb_per_game', 0.7),
                    'home_speed_proxy': home_api.get('stolen_bases', 10) / 162,
                    'away_speed_proxy': away_api.get('stolen_bases', 10) / 162,
                    'home_win_pct': home_season['win_pct'],
                    'away_win_pct': away_season['win_pct'],
                    'home_runs_per_game': home_season['avg_rs'],
                    'away_runs_per_game': away_season['avg_rs'],
                    'home_obp': home_api['OBP'],
                    'away_obp': away_api['OBP'],
                }

                sb_df = pd.DataFrame([sb_features])
                for col in set(sb_feat_cols) - set(sb_df.columns):
                    sb_df[col] = 0.0
                sb_df = sb_df[sb_feat_cols].astype(float)
                sb_scaled = sb_scaler.transform(sb_df)
                prob = sb_model.predict_proba(pd.DataFrame(sb_scaled, columns=sb_feat_cols))[0][1]
                prob = st_adjust(prob, game.get('game_type', 'R'))

                pred['sb_prob'] = round(float(prob), 4)
                pred['sb_pick'] = 'OVER' if prob > 0.5 else 'UNDER'
                pred['sb_confidence'] = 'STRONG' if abs(prob-0.5) > 0.1 else 'MODERATE' if abs(prob-0.5) > 0.05 else 'LEAN'
            except Exception:
                pred['sb_pick'] = 'N/A'

        # --- PITCHER OUTS ---
        if 'pitcher_outs' in models and latest_pitcher_year:
            pitch_df = pitcher_stats[latest_pitcher_year]
            for side, p_name_key in [('home', 'home_pitcher'), ('away', 'away_pitcher')]:
                p_name = game.get(p_name_key, 'TBA')
                if p_name == 'TBA': continue
                
                try:
                    out_data = models['pitcher_outs']
                    out_model = out_data['model']
                    out_scaler = out_data['scaler']
                    out_feat_cols = out_data['feature_cols']

                    p_row = pitch_df[pitch_df['name'].str.contains(p_name.split()[-1], case=False, na=False)]
                    if len(p_row) == 0: continue
                    p = p_row.iloc[0]

                    out_features = {col: float(p.get(col, 0)) for col in out_feat_cols}
                    # Add team context
                    team_stat = home_season if side == 'home' else away_season
                    out_features['team_win_pct'] = team_stat['win_pct']
                    out_features['team_era'] = team_stat['avg_ra'] # Proxy
                    out_features['team_runs_per_game'] = team_stat['avg_rs']

                    out_df = pd.DataFrame([out_features])
                    for col in set(out_feat_cols) - set(out_df.columns):
                        out_df[col] = 0.0
                    out_df = out_df[out_feat_cols].astype(float)
                    out_scaled = out_scaler.transform(out_df)
                    outs_pred = out_model.predict(pd.DataFrame(out_scaled, columns=out_feat_cols))[0]
                    
                    pred[f'{side}_outs_pred'] = round(float(outs_pred), 2)
                    # Convert to innings format (e.g. 5.66 -> 5.2)
                    full_inns = int(outs_pred)
                    frac = outs_pred - full_inns
                    if frac < 0.15: outs_label = f"{full_inns}.0"
                    elif frac < 0.45: outs_label = f"{full_inns}.1"
                    else: outs_label = f"{full_inns}.2"
                    pred[f'{side}_outs_label'] = outs_label
                except Exception:
                    pass

        # 追加メタデータ
        pred['home_win_pct'] = round(float(home_season['win_pct']), 3)
        pred['away_win_pct'] = round(float(away_season['win_pct']), 3)
        pred['home_rolling_form'] = round(float(home_rolling['rolling_win_pct']), 3)
        pred['away_rolling_form'] = round(float(away_rolling['rolling_win_pct']), 3)

        all_predictions.append(pred)

    # Moneylineエッジでソート
    all_predictions.sort(key=lambda x: abs(x.get('ml_edge', 0)), reverse=True)

    print(f"  ✓ Generated {len(all_predictions)} full predictions (skipped {skipped})")

    # サマリー
    if all_predictions and 'ml_confidence' in all_predictions[0]:
        strong = sum(1 for p in all_predictions if p.get('ml_confidence') == 'STRONG')
        moderate = sum(1 for p in all_predictions if p.get('ml_confidence') == 'MODERATE')
        lean = sum(1 for p in all_predictions if p.get('ml_confidence') == 'LEAN')
        passed = sum(1 for p in all_predictions if p.get('ml_confidence') == 'PASS')
        print(f"  → ML: STRONG {strong} | MODERATE {moderate} | LEAN {lean} | PASS {passed}")

    return all_predictions


# ========================================================
# 6. Digest生成
# ========================================================
def edge_display(edge):
    """エッジ表示フォーマット"""
    if edge > 0.08:
        return f"+{edge*100:.1f}% 🔥"
    elif edge > 0.04:
        return f"+{edge*100:.1f}% 👍"
    elif edge > 0:
        return f"+{edge*100:.1f}%"
    elif edge > -0.03:
        return f"{edge*100:.1f}%"
    else:
        return f"{edge*100:.1f}% ⚠️"


def generate_english_digest(predictions, target_date, models_loaded):
    """Substack用英語ダイジェスト — 全市場対応"""
    print("[6/9] Generating English digest (Substack)...")

    if not predictions:
        return f"# Moneyball Dojo Daily Digest — {target_date}\n\nNo games scheduled today.\n"

    md = []
    md.append(f"# Moneyball Dojo Daily Digest — {target_date}")
    md.append("")
    active_models = len(models_loaded)
    md.append(f"*{len(predictions)} games analyzed across {active_models} AI models. Here's what the numbers say.*")
    md.append("")

    # Spring Training banner
    is_spring_training = any(p.get('game_type') == 'S' for p in predictions)
    if is_spring_training:
        md.append("> **Spring Training Notice:** Probabilities are **shrunk 50% toward the mean** to reflect")
        md.append("> non-standard lineups, split squads, and short pitching stints. Treat all picks with extra caution.")
        md.append("> Treat all picks as directional signals, not high-conviction bets.")
        md.append("")

    md.append("---")
    md.append("")

    # === MONEYLINE TABLE ===
    actionable_ml = [p for p in predictions if p.get('ml_pick', 'N/A') not in ('PASS', 'N/A')]

    md.append("## Moneyline Picks")
    md.append("")
    # Determine edge source label from first actionable prediction
    _edge_src = 'Market' if any(p.get('ml_odds_source') == 'market' for p in predictions) else 'Log5'
    md.append(f"| Matchup | Pick | Win Prob | Edge vs {_edge_src} | Confidence |")
    md.append("|---------|------|----------|------|------------|")
    for p in predictions:
        matchup = f"{p['away_team']} @ {p['home_team']}"
        pick = p.get('ml_pick', 'N/A')
        prob = p.get('ml_prob', 0.5)
        e = p.get('ml_edge', 0)
        conf = p.get('ml_confidence', 'N/A')
        pick_d = f"**{pick}**" if conf in ('STRONG', 'MODERATE') else pick
        md.append(f"| {matchup} | {pick_d} | {prob*100:.1f}% | {edge_display(e)} | {conf} |")
    md.append("")

    # === OVER/UNDER ===
    if any(p.get('ou_predicted_total') for p in predictions):
        md.append("## Over/Under")
        md.append("")
        md.append("| Matchup | Predicted Total | O/U 8.5 | Margin | Confidence |")
        md.append("|---------|----------------|---------|--------|------------|")
        for p in predictions:
            if p.get('ou_predicted_total') is None:
                continue
            matchup = f"{p['away_team']} @ {p['home_team']}"
            total = p['ou_predicted_total']
            pick_85 = p.get('ou_8.5_pick', 'N/A')
            margin_85 = p.get('ou_8.5_margin', 0)
            conf_85 = p.get('ou_8.5_confidence', 'N/A')
            md.append(f"| {matchup} | {total:.1f} | {pick_85} | {margin_85:+.1f} | {conf_85} |")
        md.append("")

    # === RUN LINE ===
    rl_actionable = [p for p in predictions if p.get('rl_confidence') in ('STRONG', 'MODERATE')]
    if rl_actionable:
        md.append("## Run Line (-1.5)")
        md.append("")
        md.append("| Matchup | Pick | Home Cover Prob | Confidence |")
        md.append("|---------|------|-----------------|------------|")
        for p in rl_actionable:
            matchup = f"{p['away_team']} @ {p['home_team']}"
            pick = p.get('rl_pick', 'N/A')
            prob = p.get('rl_covers_prob', 0.357)
            conf = p.get('rl_confidence', 'N/A')
            md.append(f"| {matchup} | {pick} | {prob*100:.1f}% | {conf} |")
        md.append("")

    # === F5 MONEYLINE ===
    if any(p.get('f5_prob') for p in predictions):
        md.append("## First 5 Innings")
        md.append("")
        md.append("| Matchup | Pick | Win Prob | Confidence |")
        md.append("|---------|------|----------|------------|")
        for p in predictions:
            if p.get('f5_pick', 'N/A') == 'N/A':
                continue
            matchup = f"{p['away_team']} @ {p['home_team']}"
            pick = p.get('f5_pick', 'N/A')
            prob = p.get('f5_prob', 0.5)
            conf = p.get('f5_confidence', 'N/A')
            md.append(f"| {matchup} | {pick} | {prob*100:.1f}% | {conf} |")
        md.append("")

    # === NRFI/YRFI ===
    if any(p.get('nrfi_pick') for p in predictions):
        md.append("## NRFI/YRFI (No Runs First Inning)")
        md.append("")
        md.append("| Matchup | Pick | Prob | Confidence |")
        md.append("|---------|------|------|------------|")
        for p in predictions:
            if p.get('nrfi_pick', 'N/A') == 'N/A':
                continue
            matchup = f"{p['away_team']} @ {p['home_team']}"
            pick = p.get('nrfi_pick', 'N/A')
            prob = p.get('nrfi_prob', 0.5)
            conf = p.get('nrfi_confidence', 'N/A')
            md.append(f"| {matchup} | {pick} | {prob*100:.1f}% | {conf} |")
        md.append("")

    # === STOLEN BASES ===
    if any(p.get('sb_pick') for p in predictions):
        md.append("## Stolen Base Props (Team Over/Under)")
        md.append("")
        md.append("| Matchup | Pick | Prob | Confidence |")
        md.append("|---------|------|------|------------|")
        for p in predictions:
            if p.get('sb_pick', 'N/A') == 'N/A':
                continue
            matchup = f"{p['away_team']} @ {p['home_team']}"
            pick = p.get('sb_pick', 'N/A')
            prob = p.get('sb_prob', 0.5)
            conf = p.get('sb_confidence', 'N/A')
            md.append(f"| {matchup} | {pick} | {prob*100:.1f}% | {conf} |")
        md.append("")

    # === PITCHER K PROPS ===
    has_k = any(p.get('home_pitcher_k_pred') or p.get('away_pitcher_k_pred') for p in predictions)
    if has_k:
        md.append("## Pitcher Strikeout Props")
        md.append("")
        md.append("| Pitcher | Team | Predicted K | O/U 5.5 | O/U 6.5 |")
        md.append("|---------|------|-------------|---------|---------|")
        for p in predictions:
            for side in ['home', 'away']:
                k_pred = p.get(f'{side}_pitcher_k_pred')
                if k_pred is None:
                    continue
                name = p.get(f'{side}_pitcher_name', p.get(f'{side}_pitcher', 'TBA'))
                team = p[f'{side}_team']
                pick_55 = p.get(f'{side}_k_5.5_pick', 'N/A')
                pick_65 = p.get(f'{side}_k_6.5_pick', 'N/A')
                md.append(f"| {name} | {team} | {k_pred:.1f} K | {pick_55} | {pick_65} |")
        md.append("")

    # === BATTER PROPS ===
    has_batter = any(p.get('home_batter_preds') or p.get('away_batter_preds') for p in predictions)
    if has_batter:
        md.append("## Batter Props (Top Hitters)")
        md.append("")
        md.append("| Player | Team | Pred Hits/G | Pred HR/G |")
        md.append("|--------|------|-------------|-----------|")
        seen_batters = set()
        for p in predictions:
            for side in ['home', 'away']:
                batters = p.get(f'{side}_batter_preds', [])
                team = p[f'{side}_team']
                for b in batters:
                    key = (b['name'], team)
                    if key in seen_batters:
                        continue
                    seen_batters.add(key)
                    md.append(f"| {b['name']} | {team} | {b['hits_pred']:.2f} | {b['hr_pred']:.3f} |")
        md.append("")

    # === FEATURED PICK ===
    best = actionable_ml[0] if actionable_ml else None
    if best:
        md.append(f"## Featured: {best['away_team']} @ {best['home_team']}")
        md.append("")
        md.append(f"**Moneyline: {best['ml_pick']}** ({best['ml_prob']*100:.1f}%, edge {edge_display(best['ml_edge'])})")

        if best.get('ou_predicted_total'):
            md.append(f"**Over/Under:** Predicted total {best['ou_predicted_total']:.1f} runs")
        if best.get('rl_pick') and best['rl_pick'] != 'PASS':
            md.append(f"**Run Line:** {best['rl_pick']} ({best['rl_covers_prob']*100:.1f}%)")
        if best.get('f5_pick') and best['f5_pick'] != 'PASS':
            md.append(f"**F5 Moneyline:** {best['f5_pick']} ({best['f5_prob']*100:.1f}%)")
        md.append("")

    # トラックレコード要約
    track_record_path = PROJECT_DIR / "output" / "track_record_data.json"
    if track_record_path.exists():
        try:
            with open(track_record_path, 'r') as f:
                tr = json.load(f)
            sp = tr.get('strong_picks', {})
            md.append("## 2025 Backtested Track Record (STRONG Picks)")
            md.append("")
            md.append("| Market | Win Rate | ROI | Games |")
            md.append("|--------|----------|-----|-------|")
            for name in ['Moneyline', 'Run Line', 'Over/Under', 'F5 Moneyline']:
                s = sp.get(name)
                if s:
                    md.append(f"| {name} | {s['accuracy']:.1%} | {s['roi']:+.1f}% | {s['total']} |")
            c = tr.get('combined', {})
            if c:
                md.append(f"\n> Combined: **{c['accuracy']:.1%}** win rate, **{c['roi']:+.1f}%** ROI across {c['total']} picks")
            md.append("")
        except Exception:
            pass

    # フッター
    md.append("---")
    md.append("")
    md.append(f"*{active_models} AI models. {len(predictions)} games. Real data. No gut feelings.*")
    md.append("*Not financial advice. Gamble responsibly.*")
    md.append("")
    md.append("**Subscribe** to get daily picks before first pitch.")

    content = "\n".join(md)
    print(f"  ✓ English digest: {len(content)} characters")
    return content


def generate_japanese_digest(predictions, target_date, models_loaded):
    """note.com用日本語ダイジェスト — 全市場対応"""
    print("[7/9] Generating Japanese digest (note.com)...")

    if not predictions:
        return f"# Moneyball Dojo デイリーダイジェスト — {target_date}\n\n本日の試合はありません。\n"

    confidence_ja = {'STRONG': '🔥 強気', 'MODERATE': '👍 中程度', 'LEAN': '→ 傾向', 'PASS': '⏸ 見送り', 'N/A': '-'}
    active_models = len(models_loaded)

    md = []
    md.append(f"# Moneyball Dojo デイリーダイジェスト — {target_date}")
    md.append("")
    md.append(f"こんにちは、Moneyball Dojoです。本日は{len(predictions)}試合を{active_models}つのAIモデルで分析しました。")
    md.append("")

    # Spring Training banner
    is_spring_training = any(p.get('game_type') == 'S' for p in predictions)
    if is_spring_training:
        md.append("> **【春季キャンプ期間中】** 本モデルはレギュラーシーズンデータ（2022-2025年）で学習しています。")
        md.append("> 春季トレーニングは通常と異なるオーダー・投手起用のため、予測は参考値としてご利用ください。")
        md.append("")

    md.append("---")
    md.append("")

    # === MONEYLINE ===
    md.append("## マネーライン予測")
    md.append("")
    _edge_src_ja = 'Market' if any(p.get('ml_odds_source') == 'market' for p in predictions) else 'Log5'
    md.append(f"| 対戦 | 予測 | 勝率 | Edge vs {_edge_src_ja} | 信頼度 |")
    md.append("|------|------|------|--------|--------|")
    for p in predictions:
        matchup = f"{p['away_team']} @ {p['home_team']}"
        pick = p.get('ml_pick', 'N/A')
        prob = p.get('ml_prob', 0.5)
        e = p.get('ml_edge', 0)
        conf = confidence_ja.get(p.get('ml_confidence', 'N/A'), 'N/A')
        md.append(f"| {matchup} | {pick} | {prob*100:.1f}% | {edge_display(e)} | {conf} |")
    md.append("")

    # === OVER/UNDER ===
    if any(p.get('ou_predicted_total') for p in predictions):
        md.append("## 合計得点 (Over/Under)")
        md.append("")
        md.append("| 対戦 | 予想合計 | O/U 8.5 | マージン | 信頼度 |")
        md.append("|------|---------|---------|---------|--------|")
        for p in predictions:
            if p.get('ou_predicted_total') is None:
                continue
            matchup = f"{p['away_team']} @ {p['home_team']}"
            total = p['ou_predicted_total']
            pick = p.get('ou_8.5_pick', 'N/A')
            margin = p.get('ou_8.5_margin', 0)
            conf = confidence_ja.get(p.get('ou_8.5_confidence', 'N/A'), '-')
            md.append(f"| {matchup} | {total:.1f} | {pick} | {margin:+.1f} | {conf} |")
        md.append("")

    # === RUN LINE ===
    rl_actionable_ja = [p for p in predictions if p.get('rl_confidence') in ('STRONG', 'MODERATE')]
    if rl_actionable_ja:
        md.append("## ランライン (-1.5)")
        md.append("")
        md.append("| 対戦 | 予測 | ホームカバー確率 | 信頼度 |")
        md.append("|------|------|-----------------|--------|")
        for p in rl_actionable_ja:
            matchup = f"{p['away_team']} @ {p['home_team']}"
            pick = p.get('rl_pick', 'N/A')
            prob = p.get('rl_covers_prob', 0.357)
            conf = confidence_ja.get(p.get('rl_confidence', 'N/A'), '-')
            md.append(f"| {matchup} | {pick} | {prob*100:.1f}% | {conf} |")
        md.append("")

    # === NRFI/YRFI ===
    if any(p.get('nrfi_pick') for p in predictions):
        md.append("## NRFI/YRFI (1回裏表の無得点予測)")
        md.append("")
        md.append("| 対戦 | 予測 | 確率 | 信頼度 |")
        md.append("|------|------|------|--------|")
        for p in predictions:
            if p.get('nrfi_pick', 'N/A') == 'N/A':
                continue
            matchup = f"{p['away_team']} @ {p['home_team']}"
            pick = p.get('nrfi_pick', 'N/A')
            prob = p.get('nrfi_prob', 0.5)
            conf = confidence_ja.get(p.get('nrfi_confidence', 'N/A'), '-')
            md.append(f"| {matchup} | {pick} | {prob*100:.1f}% | {conf} |")
        md.append("")

    # === PITCHER K PROPS ===
    has_k = any(p.get('home_pitcher_k_pred') or p.get('away_pitcher_k_pred') for p in predictions)
    if has_k:
        md.append("## 投手奪三振予測")
        md.append("")
        md.append("| 投手名 | チーム | 予測K | O/U 5.5 | O/U 6.5 |")
        md.append("|--------|--------|-------|---------|---------|")
        for p in predictions:
            for side in ['home', 'away']:
                k_pred = p.get(f'{side}_pitcher_k_pred')
                if k_pred is None:
                    continue
                name = p.get(f'{side}_pitcher_name', p.get(f'{side}_pitcher', 'TBA'))
                team = p[f'{side}_team']
                pick_55 = p.get(f'{side}_k_5.5_pick', '-')
                pick_65 = p.get(f'{side}_k_6.5_pick', '-')
                md.append(f"| {name} | {team} | {k_pred:.1f} K | {pick_55} | {pick_65} |")
        md.append("")

    # === BATTER PROPS ===
    has_batter = any(p.get('home_batter_preds') or p.get('away_batter_preds') for p in predictions)
    if has_batter:
        md.append("## 打者プロップス（主力打者）")
        md.append("")
        md.append("| 選手名 | チーム | 予測H/G | 予測HR/G |")
        md.append("|--------|--------|---------|----------|")
        seen_batters_ja = set()
        for p in predictions:
            for side in ['home', 'away']:
                batters = p.get(f'{side}_batter_preds', [])
                team = p[f'{side}_team']
                for b in batters:
                    key = (b['name'], team)
                    if key in seen_batters_ja:
                        continue
                    seen_batters_ja.add(key)
                    md.append(f"| {b['name']} | {team} | {b['hits_pred']:.2f} | {b['hr_pred']:.3f} |")
        md.append("")

    # トラックレコード要約
    track_record_path = PROJECT_DIR / "output" / "track_record_data.json"
    if track_record_path.exists():
        try:
            with open(track_record_path, 'r') as f:
                tr = json.load(f)
            sp = tr.get('strong_picks', {})
            name_ja = {'Moneyline': 'ML', 'Run Line': 'RL', 'Over/Under': 'O/U', 'F5 Moneyline': 'F5'}
            md.append("## 2025バックテスト実績（STRONGピック限定）")
            md.append("")
            md.append("| マーケット | 勝率 | ROI | 試合数 |")
            md.append("|-----------|------|-----|--------|")
            for name in ['Moneyline', 'Run Line', 'Over/Under', 'F5 Moneyline']:
                s = sp.get(name)
                if s:
                    md.append(f"| {name_ja.get(name, name)} | {s['accuracy']:.1%} | {s['roi']:+.1f}% | {s['total']} |")
            c = tr.get('combined', {})
            if c:
                md.append(f"\n> 全体: **{c['accuracy']:.1%}** 勝率、**{c['roi']:+.1f}%** ROI（{c['total']}ピック）")
            md.append("")
        except Exception:
            pass

    # フッター
    md.append("---")
    md.append("")
    md.append(f"*東京のAIエンジニアが開発。{active_models}モデル × XGBoost。データで勝負する。*")
    md.append("*投資アドバイスではありません。責任あるプレイを。*")

    content = "\n".join(md)
    print(f"  ✓ Japanese digest: {len(content)} characters")
    return content


def generate_twitter_post(predictions, target_date, models_loaded):
    """Twitter/X投稿用サマリー"""
    print("[8/9] Generating Twitter summary...")

    actionable = [p for p in predictions if p.get('ml_confidence') in ('STRONG', 'MODERATE')]
    active_models = len(models_loaded)

    lines = []
    lines.append(f"🎯 Moneyball Dojo AI Picks — {target_date}")
    lines.append(f"({active_models} models, {len(predictions)} games)")
    lines.append("")

    if not actionable:
        lines.append("No high-confidence picks today. Sometimes the best bet is no bet.")
    else:
        for p in actionable[:5]:
            emoji = "🔥" if p.get('ml_confidence') == 'STRONG' else "👍"
            lines.append(f"{emoji} {p['away_team']}@{p['home_team']}: {p.get('ml_pick', '?')} ({p.get('ml_prob', 0.5)*100:.0f}%)")

            # Over/Under
            if p.get('ou_predicted_total'):
                pick_85 = p.get('ou_8.5_pick', '')
                if pick_85 and pick_85 != 'PASS':
                    lines.append(f"   O/U: {p['ou_predicted_total']:.1f} ({pick_85} 8.5)")

    lines.append("")
    lines.append(f"Full analysis → [Substack link]")
    lines.append("#MLB #SportsBetting #AIpicks #MoneyballDojo")

    content = "\n".join(lines)
    print(f"  ✓ Twitter post: {len(content)} characters")
    return content


# ========================================================
# 7. 保存
# ========================================================
def save_outputs(predictions, en_digest, ja_digest, twitter_post, target_date):
    """全出力をファイルに保存"""
    print("[9/9] Saving outputs...")

    OUTPUT_DIR = PROJECT_DIR / "output" / target_date.replace("-", "")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # CSV（Sheets用）— ネストされた辞書をフラット化
    csv_data = []
    for p in predictions:
        row = {}
        for k, v in p.items():
            if isinstance(v, (list, dict)):
                row[k] = json.dumps(v, ensure_ascii=False)
            elif isinstance(v, (np.integer,)):
                row[k] = int(v)
            elif isinstance(v, (np.floating,)):
                row[k] = float(v)
            else:
                row[k] = v
        csv_data.append(row)

    csv_path = OUTPUT_DIR / "predictions.csv"
    pd.DataFrame(csv_data).to_csv(csv_path, index=False)
    print(f"  ✓ CSV → {csv_path}")

    # 英語Digest
    en_path = OUTPUT_DIR / f"digest_EN_{target_date}.md"
    en_path.write_text(en_digest, encoding='utf-8')
    print(f"  ✓ English Digest → {en_path}")

    # 日本語Digest
    ja_path = OUTPUT_DIR / f"digest_JA_{target_date}.md"
    ja_path.write_text(ja_digest, encoding='utf-8')
    print(f"  ✓ Japanese Digest → {ja_path}")

    # Twitter投稿
    tw_path = OUTPUT_DIR / f"twitter_{target_date}.txt"
    tw_path.write_text(twitter_post, encoding='utf-8')
    print(f"  ✓ Twitter → {tw_path}")

    # JSON（全データ）
    def convert_numpy(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    clean_predictions = []
    for p in predictions:
        clean_predictions.append({k: convert_numpy(v) for k, v in p.items()})

    json_path = OUTPUT_DIR / f"predictions_{target_date}.json"
    json_path.write_text(json.dumps(clean_predictions, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"  ✓ JSON → {json_path}")

    # === output/latest/ — always overwrite with the most recent run ===
    LATEST_DIR = PROJECT_DIR / "output" / "latest"
    if LATEST_DIR.exists():
        import shutil
        shutil.rmtree(LATEST_DIR)
    LATEST_DIR.mkdir(parents=True, exist_ok=True)

    (LATEST_DIR / "POST_TO_SUBSTACK.md").write_text(en_digest, encoding='utf-8')
    (LATEST_DIR / "POST_TO_NOTE.md").write_text(ja_digest, encoding='utf-8')
    (LATEST_DIR / "POST_TO_X.txt").write_text(twitter_post, encoding='utf-8')
    pd.DataFrame(csv_data).to_csv(LATEST_DIR / "predictions.csv", index=False)
    (LATEST_DIR / "predictions.json").write_text(
        json.dumps(clean_predictions, indent=2, ensure_ascii=False), encoding='utf-8')
    (LATEST_DIR / "README.txt").write_text(
        f"Moneyball Dojo - Latest Predictions ({target_date})\n"
        f"=================================================\n\n"
        f"POST_TO_SUBSTACK.md  -> Substack (English)\n"
        f"POST_TO_NOTE.md      -> note.com (Japanese)\n"
        f"POST_TO_X.txt        -> Twitter/X\n\n"
        f"predictions.csv      -> Google Sheets\n"
        f"predictions.json     -> Full data (API/archive)\n\n"
        f"Generated by: python3 run_daily.py\n",
        encoding='utf-8')
    print(f"  ✓ latest/ → {LATEST_DIR}/")

    return OUTPUT_DIR


def upload_to_sheets(predictions, target_date, model_version="v3"):
    """
    Google Sheets v2 スキーマに基づき予測データを append する。
    認証: credentials.json または GOOGLE_SHEETS_CREDENTIALS 環境変数
    対象: GOOGLE_SHEETS_ID 環境変数 or SPREADSHEET_NAME
    """
    if not SHEETS_AVAILABLE:
        print("  ⚠ gspread未インストール → pip install gspread google-auth")
        return False

    # 認証
    gc = None
    try:
        # 1) 環境変数からJSON認証情報（GitHub Actions用）
        creds_json = os.environ.get('GOOGLE_SHEETS_CREDENTIALS')
        if creds_json:
            import json as _json
            creds_dict = _json.loads(creds_json)
            credentials = ServiceAccountCredentials.from_service_account_info(
                creds_dict,
                scopes=['https://www.googleapis.com/auth/spreadsheets'],
            )
            gc = gspread.authorize(credentials)
        # 2) credentials.json ファイル（ローカル用）
        elif CREDENTIALS_PATH.exists():
            gc = gspread.service_account(filename=str(CREDENTIALS_PATH))
        else:
            print("  ⚠ 認証情報なし（credentials.json / GOOGLE_SHEETS_CREDENTIALS）→ Sheets連携スキップ")
            return False
    except Exception as e:
        print(f"  ⚠ Sheets認証失敗: {e}")
        return False

    # スプレッドシートを開く
    try:
        sheet_id = os.environ.get('GOOGLE_SHEETS_ID')
        if sheet_id:
            sh = gc.open_by_key(sheet_id)
        else:
            sh = gc.open(SPREADSHEET_NAME)
    except Exception as e:
        print(f"  ⚠ Spreadsheet open failed: {e}")
        return False

    now_iso = datetime.utcnow().isoformat() + "Z"
    errors = []

    # --- Sheet: Daily Predictions (メインの予測データ) ---
    try:
        ws = _get_or_create_worksheet(sh, 'Daily Predictions')
        rows = []
        for p in predictions:
            rows.append([
                target_date,
                str(p.get('game_id', '')),
                str(p.get('away_team', '')),
                str(p.get('home_team', '')),
                str(p.get('away_pitcher', 'TBA')),
                str(p.get('home_pitcher', 'TBA')),
                str(round(float(p.get('ml_prob', 0)), 4)),
                str(p.get('ml_pick', '')),
                str(round(float(p.get('ml_edge', 0)), 4)),
                str(p.get('ml_confidence', '')),
                str(p.get('ou_predicted_total', '')),
                str(p.get('rl_pick', '')),
                str(p.get('rl_confidence', '')),
                str(p.get('f5_pick', '')),
                str(p.get('f5_confidence', '')),
                str(p.get('nrfi_pick', '')),
                str(p.get('nrfi_confidence', '')),
                str(p.get('sb_pick', '')),
                now_iso,
            ])
        _batch_append_with_retry(ws, rows)
        print(f"  ✓ Daily Predictions → {len(rows)} rows")
    except Exception as e:
        errors.append(f"Daily Predictions: {e}")

    # --- Sheet: predictions (v2 スキーマ: append-only event log) ---
    try:
        ws_pred = _get_or_create_worksheet(sh, 'predictions')
        pred_rows = []
        for p in predictions:
            pred_rows.append([
                str(p.get('game_id', '')),
                f"{model_version}_{target_date.replace('-', '_')}",
                str(round(float(p.get('ml_prob', 0)), 4)),
                str(p.get('ou_predicted_total', '')),
                str(round(float(p.get('ml_edge', 0)), 4)),
                str(p.get('ml_confidence', '')),
                str(p.get('ml_pick', '')),
                f"Edge={p.get('ml_edge', 0)*100:+.1f}%",
                now_iso,
            ])
        _batch_append_with_retry(ws_pred, pred_rows)
        print(f"  ✓ predictions (v2 log) → {len(pred_rows)} rows")
    except Exception as e:
        errors.append(f"predictions: {e}")

    if errors:
        for err in errors:
            print(f"  ⚠ Sheets error: {err}")
        return False

    return True


def _get_or_create_worksheet(sh, name):
    """ワークシートを取得。なければ作成する。"""
    try:
        return sh.worksheet(name)
    except gspread.exceptions.WorksheetNotFound:
        return sh.add_worksheet(title=name, rows=1000, cols=26)


def _batch_append_with_retry(ws, rows, max_retries=MAX_RETRIES):
    """gspread batch append with retry logic."""
    for attempt in range(max_retries):
        try:
            ws.append_rows(rows, value_input_option='USER_ENTERED')
            return
        except Exception as e:
            if attempt < max_retries - 1:
                delay = RETRY_DELAYS[attempt]
                logger.warning(f"Sheets append failed (attempt {attempt+1}): {e}. Retrying in {delay}s...")
                time.sleep(delay)
            else:
                raise


# ========================================================
# Slack通知
# ========================================================
def _slack_post(webhook_url, payload):
    """Slack webhookにPOSTする。リトライ付き。"""
    try:
        import requests
    except ImportError:
        return False
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(webhook_url, json=payload, timeout=10)
            if resp.status_code == 200:
                return True
        except Exception:
            pass
        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_DELAYS[attempt])
    return False


def _notify_success(predictions, target_date, models, en_digest, ja_digest, twitter_post):
    """パイプライン成功時にSlackへ投稿用コンテンツを送信する。"""
    slack_webhook = os.environ.get('SLACK_WEBHOOK')
    if not slack_webhook:
        return

    # --- Message 1: サマリー + X投稿(コピペ用) ---
    strong = [p for p in predictions if p.get('ml_confidence') == 'STRONG']
    top = predictions[0] if predictions else None

    summary_lines = [
        f"*Moneyball Dojo {target_date}*",
        f"{len(predictions)} games | {len(models)} models | STRONG {len(strong)}",
        "",
    ]
    if top:
        summary_lines.append(
            f"Top: {top['away_team']} @ {top['home_team']} "
            f"→ {top.get('ml_pick', '?')} {top.get('ml_prob', 0)*100:.0f}%"
        )
        summary_lines.append("")

    summary_lines.append("*--- POST TO X (copy below) ---*")
    summary_lines.append(f"```{twitter_post}```")

    _slack_post(slack_webhook, {"text": "\n".join(summary_lines)})

    # --- Message 2: Substack記事 (EN) ---
    # Slack text limit ~40k chars, but keep it reasonable
    en_trimmed = en_digest[:3900]
    if len(en_digest) > 3900:
        en_trimmed += "\n\n... (truncated, full version in output/latest/POST_TO_SUBSTACK.md)"
    _slack_post(slack_webhook, {
        "text": f"*--- POST TO SUBSTACK (copy below) ---*\n```{en_trimmed}```"
    })

    # --- Message 3: note記事 (JA) ---
    ja_trimmed = ja_digest[:3900]
    if len(ja_digest) > 3900:
        ja_trimmed += "\n\n... (truncated, full version in output/latest/POST_TO_NOTE.md)"
    _slack_post(slack_webhook, {
        "text": f"*--- POST TO note.com (copy below) ---*\n```{ja_trimmed}```"
    })

    logger.info("Slack success notifications sent (3 messages)")


def _notify_error(step_name: str, error: Exception, target_date: str):
    """パイプラインエラーを Slack / GitHub Issues に通知する。"""
    error_msg = f"[{step_name}] {type(error).__name__}: {error}"
    logger.error(error_msg)
    logger.error(traceback.format_exc())

    # Slack通知（環境変数あれば）
    slack_webhook = os.environ.get('SLACK_WEBHOOK')
    if slack_webhook:
        try:
            import requests
            payload = {
                "text": f"🚨 Moneyball Dojo Pipeline Error — {target_date}\n"
                        f"Step: {step_name}\n"
                        f"Error: {error_msg}\n"
                        f"```{traceback.format_exc()[-500:]}```"
            }
            for attempt in range(MAX_RETRIES):
                try:
                    resp = requests.post(slack_webhook, json=payload, timeout=10)
                    if resp.status_code == 200:
                        logger.info("Slack notification sent")
                        break
                except Exception:
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_DELAYS[attempt])
        except ImportError:
            pass

    # エラーログファイル
    error_dir = PROJECT_DIR / "output" / target_date.replace("-", "")
    error_dir.mkdir(parents=True, exist_ok=True)
    error_path = error_dir / "pipeline_errors.log"
    with open(error_path, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"Time: {datetime.utcnow().isoformat()}Z\n")
        f.write(f"Step: {step_name}\n")
        f.write(f"Error: {error_msg}\n")
        f.write(traceback.format_exc())


def _run_step_with_retry(step_name, func, *args, max_retries=MAX_RETRIES, **kwargs):
    """パイプラインステップをリトライ付きで実行する。"""
    last_error = None
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                delay = RETRY_DELAYS[attempt]
                logger.warning(f"[{step_name}] failed (attempt {attempt+1}): {e}. Retrying in {delay}s...")
                time.sleep(delay)
            else:
                logger.error(f"[{step_name}] failed after {max_retries} attempts")
    raise last_error


# ========================================================
# メイン実行
# ========================================================
def main():
    target_date = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%Y-%m-%d")
    pipeline_errors = []

    print("=" * 70)
    print("MONEYBALL DOJO — DAILY PREDICTION PIPELINE v4 (FULL AUTOMATION)")
    print(f"Date: {target_date}")
    print("=" * 70)
    print()

    # 1. 全モデル読み込み
    models = load_all_models()
    print()

    # 2. 当日スケジュール取得
    games = get_todays_schedule(target_date)
    print()

    if not games:
        print("❌ No games found. Exiting.")
        return

    # 3. 過去データ読み込み
    games_df, team_stats, latest_year, pitcher_stats, latest_pitcher_year, batter_stats, latest_batter_year = load_historical_data()
    print()

    # 4. 予測ステップ表示
    print("[4/9] Computing team metrics & rolling stats...")
    print(f"  ✓ Using {latest_year} as reference season")
    print()

    # 4b. 実オッズ取得（The Odds API）
    market_odds = {}
    if ODDS_API_KEY:
        print("[4b/9] Fetching market odds from The Odds API...")
        try:
            market_odds = fetch_market_odds()
            if market_odds:
                print(f"  ✓ Market odds: {len(market_odds)} games (edge = model vs market)")
            else:
                print("  ⚠ No market odds returned — using Log5 baseline")
        except Exception as e:
            print(f"  ⚠ Odds API error: {e} — using Log5 baseline")
    else:
        print("[4b/9] No ODDS_API_KEY — edge calculated vs Log5 baseline")

    # 5. 全モデル予測
    predictions = generate_all_predictions(
        games, models, games_df, team_stats, latest_year,
        pitcher_stats, latest_pitcher_year,
        batter_stats, latest_batter_year,
        market_odds=market_odds,
    )
    print()

    if not predictions:
        print("❌ No predictions generated. Exiting.")
        return

    # 6. 英語Digest（テンプレート版 — フォールバック用）
    en_digest = generate_english_digest(predictions, target_date, models)
    print()

    # 7. 日本語Digest（テンプレート版 — フォールバック用）
    ja_digest = generate_japanese_digest(predictions, target_date, models)
    print()

    # 8. Twitter投稿
    twitter_post = generate_twitter_post(predictions, target_date, models)
    print()

    # ========================================================
    # [NEW] Anthropic API による記事生成（テンプレート版を上書き）
    # ========================================================
    if ARTICLE_GEN_AVAILABLE:
        print("[API] Generating AI-powered articles via Anthropic API...")
        try:
            generator = ArticleGenerator()
            if generator.api_available:
                api_en, api_ja = _run_step_with_retry(
                    "Anthropic API Article Generation",
                    generator.generate_daily_digest,
                    predictions, target_date, len(models),
                )
                if api_en:
                    en_digest = api_en
                    print(f"  ✓ API English digest: {len(en_digest)} characters")
                if api_ja:
                    ja_digest = api_ja
                    print(f"  ✓ API Japanese digest: {len(ja_digest)} characters")
            else:
                print("  ⚠ Anthropic API key not set — using template digests")
        except Exception as e:
            pipeline_errors.append(("Anthropic API", e))
            _notify_error("Anthropic API Article Generation", e, target_date)
            print(f"  ⚠ API article generation failed: {e} — using template fallback")
    else:
        print("[API] article_generator_template not available — using template digests")
    print()

    # 9. 保存
    output_dir = save_outputs(predictions, en_digest, ja_digest, twitter_post, target_date)
    print()

    # ========================================================
    # [NEW] Google Sheets v2 スキーマ自動書き込み
    # ========================================================
    print("[SHEETS] Uploading to Google Sheets (v2 schema)...")
    try:
        sheets_ok = _run_step_with_retry(
            "Google Sheets Upload",
            upload_to_sheets,
            predictions, target_date,
            max_retries=MAX_RETRIES,
        )
        if not sheets_ok:
            print("  ⚠ Sheets upload skipped (no credentials)")
    except Exception as e:
        pipeline_errors.append(("Google Sheets", e))
        _notify_error("Google Sheets Upload", e, target_date)
        print(f"  ⚠ Sheets upload failed: {e}")
    print()

    # ========================================================
    # Slack成功通知（投稿用コンテンツ付き）
    # ========================================================
    print("[SLACK] Sending predictions to Slack...")
    try:
        _notify_success(predictions, target_date, models, en_digest, ja_digest, twitter_post)
        print("  ✓ Slack notifications sent")
    except Exception as e:
        print(f"  ⚠ Slack notification failed: {e}")
    print()

    # ========================================================
    # パイプラインサマリー
    # ========================================================
    print("=" * 70)
    if pipeline_errors:
        print(f"⚠️  PIPELINE COMPLETE WITH {len(pipeline_errors)} ERROR(S)")
        for step, err in pipeline_errors:
            print(f"   ❌ {step}: {err}")
    else:
        print("✅ DAILY PIPELINE v4 COMPLETE — FULL AUTOMATION")
    print(f"   Models used: {', '.join(models.keys())}")
    print(f"   Games analyzed: {len(predictions)}")
    print(f"   All outputs saved to: {output_dir}/")
    print(f"   Quick access:  output/latest/")
    print()

    if not pipeline_errors:
        print("🤖 FULLY AUTOMATED — No manual steps needed!")
        print("   Slack → コピペして投稿するだけ")
        print("   output/latest/POST_TO_SUBSTACK.md → Substack")
        print("   output/latest/POST_TO_NOTE.md     → note.com")
        print("   output/latest/POST_TO_X.txt       → X")
    else:
        print("📋 FALLBACK STEPS:")
        print(f"   1. output/latest/POST_TO_SUBSTACK.md → Substack")
        print(f"   2. output/latest/POST_TO_NOTE.md     → note.com")
        print(f"   3. output/latest/POST_TO_X.txt       → X")
    print("=" * 70)


if __name__ == '__main__':
    main()
