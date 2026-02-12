#!/usr/bin/env python3
"""
Moneyball Dojo — 実MLBデータ取得スクリプト
==========================================
実際のMLBデータをCSVに保存する。

使い方:
  pip3 install MLB-StatsAPI pandas
  python3 fetch_real_data.py              # 過去データ (2022-2024) を取得
  python3 fetch_real_data.py --update     # 今シーズンのデータを追加/更新
  python3 fetch_real_data.py --season 2025  # 指定シーズンのみ取得

出力:
  data/games_2022_2024.csv    — 過去の全試合結果
  data/games_YYYY.csv         — 指定シーズンの試合結果
  data/team_stats_YYYY.csv    — チーム成績
  data/standings_YYYY.csv     — スタンディング
"""

import statsapi
import pandas as pd
import time
import sys
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def fetch_all_games(year):
    """1シーズン全試合の結果を取得"""
    print(f"  Fetching {year} schedule...")

    start_date = f"{year}-02-20"  # Spring Training含む
    end_date = f"{year}-10-05"

    # 今年の場合は今日までに制限
    today = datetime.now()
    if year == today.year:
        end_date = today.strftime("%Y-%m-%d")

    sched = statsapi.schedule(
        start_date=start_date,
        end_date=end_date,
        sportId=1
    )

    games = []
    for game in sched:
        # レギュラーシーズンのみ
        if game.get('game_type', '') != 'R':
            continue
        # 完了した試合のみ
        if game.get('status', '') not in ('Final', 'Game Over', 'Completed Early'):
            continue

        games.append({
            'game_id': game['game_id'],
            'date': game['game_date'],
            'year': year,
            'home_team': game.get('home_name', ''),
            'home_abbr': game.get('home_id', ''),
            'away_team': game.get('away_name', ''),
            'away_abbr': game.get('away_id', ''),
            'home_score': game.get('home_score', 0),
            'away_score': game.get('away_score', 0),
            'home_win': 1 if game.get('home_score', 0) > game.get('away_score', 0) else 0,
            'winning_team': game.get('winning_team', ''),
            'losing_team': game.get('losing_team', ''),
            'home_pitcher': game.get('home_probable_pitcher', ''),
            'away_pitcher': game.get('away_probable_pitcher', ''),
            'venue': game.get('venue_name', ''),
        })

    print(f"    ✓ {len(games)} regular season games")
    return games


def fetch_team_stats(year):
    """チーム成績を取得（勝敗、得失点）"""
    print(f"  Fetching {year} standings...")

    standings = statsapi.standings_data(
        leagueId='103,104',
        season=year
    )

    teams = []
    for div_id, div_data in standings.items():
        for team in div_data['teams']:
            teams.append({
                'year': year,
                'team_name': team['name'],
                'team_id': team.get('team_id', ''),
                'division': div_data['div_name'],
                'wins': team['w'],
                'losses': team['l'],
                'win_pct': team['w'] / (team['w'] + team['l']) if (team['w'] + team['l']) > 0 else 0,
                'gb': team.get('gb', '-'),
            })

    print(f"    ✓ {len(teams)} teams")
    return teams


def fetch_team_batting_pitching(year):
    """各チームの打撃・投手指標を取得"""
    print(f"  Fetching {year} team batting/pitching stats...")

    team_stats = []
    teams_data = statsapi.get('teams', {'sportId': 1, 'season': year})

    for team in teams_data.get('teams', []):
        team_id = team['id']
        team_name = team['name']
        team_abbr = team.get('abbreviation', '')

        try:
            raw = statsapi.get('team_stats', {
                'teamId': team_id,
                'stats': 'season',
                'group': 'hitting,pitching',
                'season': year,
                'gameType': 'R'
            })

            team_record = {
                'year': year,
                'team_id': team_id,
                'team_name': team_name,
                'team_abbr': team_abbr,
            }

            for stat_group in raw.get('stats', []):
                group_name = stat_group.get('group', {}).get('displayName', '')
                if stat_group.get('splits'):
                    stats = stat_group['splits'][0].get('stat', {})

                    if group_name == 'hitting':
                        team_record['BA'] = float(stats.get('avg', '.000').replace('.', '0.', 1)) if isinstance(stats.get('avg'), str) else stats.get('avg', 0)
                        team_record['OBP'] = float(stats.get('obp', '.000').replace('.', '0.', 1)) if isinstance(stats.get('obp'), str) else stats.get('obp', 0)
                        team_record['SLG'] = float(stats.get('slg', '.000').replace('.', '0.', 1)) if isinstance(stats.get('slg'), str) else stats.get('slg', 0)
                        team_record['R'] = stats.get('runs', 0)
                        team_record['H'] = stats.get('hits', 0)
                        team_record['HR'] = stats.get('homeRuns', 0)
                        team_record['BB_hit'] = stats.get('baseOnBalls', 0)
                        team_record['SO_hit'] = stats.get('strikeOuts', 0)
                        team_record['SB'] = stats.get('stolenBases', 0)
                        team_record['OPS'] = stats.get('ops', 0)

                    elif group_name == 'pitching':
                        team_record['ERA'] = float(stats.get('era', '0.00')) if isinstance(stats.get('era'), str) else stats.get('era', 0)
                        team_record['WHIP'] = float(stats.get('whip', '0.00')) if isinstance(stats.get('whip'), str) else stats.get('whip', 0)
                        team_record['SO_pitch'] = stats.get('strikeOuts', 0)
                        team_record['BB_pitch'] = stats.get('baseOnBalls', 0)
                        team_record['RA'] = stats.get('runs', 0)
                        team_record['SV'] = stats.get('saves', 0)

            team_stats.append(team_record)
            time.sleep(0.3)

        except Exception as e:
            print(f"    ⚠ Error for {team_name}: {e}")
            continue

    print(f"    ✓ {len(team_stats)} teams with full stats")
    return team_stats


def fetch_initial_data():
    """初回セットアップ: 2022-2025の過去データ取得"""
    print("[1/3] Fetching game results (2022-2025)...")
    all_games = []
    for year in [2022, 2023, 2024, 2025]:
        games = fetch_all_games(year)
        all_games.extend(games)
        time.sleep(1)

    games_df = pd.DataFrame(all_games)
    games_path = DATA_DIR / "games_2022_2025.csv"
    games_df.to_csv(games_path, index=False)
    print(f"  ✓ Saved {len(games_df)} games → {games_path}")
    print()

    print("[2/3] Fetching team standings...")
    for year in [2022, 2023, 2024, 2025]:
        teams = fetch_team_stats(year)
        teams_df = pd.DataFrame(teams)
        path = DATA_DIR / f"standings_{year}.csv"
        teams_df.to_csv(path, index=False)
        print(f"  ✓ Saved → {path}")
        time.sleep(1)
    print()

    print("[3/3] Fetching team batting & pitching stats...")
    for year in [2022, 2023, 2024, 2025]:
        stats = fetch_team_batting_pitching(year)
        stats_df = pd.DataFrame(stats)
        path = DATA_DIR / f"team_stats_{year}.csv"
        stats_df.to_csv(path, index=False)
        print(f"  ✓ Saved → {path}")
        time.sleep(1)


def fetch_season_update(year=None):
    """シーズン途中のデータ更新（指定シーズンのみ）"""
    if year is None:
        year = datetime.now().year

    print(f"[UPDATE] Fetching {year} season data...")
    print()

    # 試合データ
    print(f"[1/3] Fetching {year} game results...")
    games = fetch_all_games(year)
    if games:
        games_df = pd.DataFrame(games)
        path = DATA_DIR / f"games_{year}.csv"
        games_df.to_csv(path, index=False)
        print(f"  ✓ Saved {len(games_df)} games → {path}")

        # games_2022_2025.csv を更新（2022-2025 + 今年分をマージ）
        base_path = DATA_DIR / "games_2022_2025.csv"
        # 旧ファイルがある場合はリネームして使う
        legacy_path = DATA_DIR / "games_2022_2024.csv"
        
        if base_path.exists():
            base_df = pd.read_csv(base_path)
        elif legacy_path.exists():
            print(f"  Note: Migrating {legacy_path} to {base_path}...")
            base_df = pd.read_csv(legacy_path)
        else:
            base_df = pd.DataFrame()

        if not base_df.empty:
            # 古い同年データを除去してマージ
            base_df = base_df[base_df['year'] != year]
            merged = pd.concat([base_df, games_df], ignore_index=True)
            # 全データファイルを更新
            merged.to_csv(base_path, index=False)
            print(f"  ✓ Merged all data → {base_path} ({len(merged)} total games)")
    else:
        print(f"  ⚠ No games found for {year} yet")
    print()

    # スタンディング
    print(f"[2/3] Fetching {year} standings...")
    try:
        teams = fetch_team_stats(year)
        if teams:
            teams_df = pd.DataFrame(teams)
            path = DATA_DIR / f"standings_{year}.csv"
            teams_df.to_csv(path, index=False)
            print(f"  ✓ Saved → {path}")
    except Exception as e:
        print(f"  ⚠ Standings not available yet: {e}")
    print()

    # チームスタッツ
    print(f"[3/3] Fetching {year} team batting & pitching stats...")
    try:
        stats = fetch_team_batting_pitching(year)
        if stats:
            stats_df = pd.DataFrame(stats)
            path = DATA_DIR / f"team_stats_{year}.csv"
            stats_df.to_csv(path, index=False)
            print(f"  ✓ Saved → {path}")
    except Exception as e:
        print(f"  ⚠ Team stats not available yet: {e}")


def main():
    print("=" * 60)
    print("MONEYBALL DOJO — MLB REAL DATA FETCH")
    print("=" * 60)
    print()

    # コマンドライン引数の処理
    args = sys.argv[1:]

    if '--update' in args:
        # シーズン更新モード
        year = datetime.now().year
        # --season YYYY が指定されていればそれを使用
        if '--season' in args:
            idx = args.index('--season')
            if idx + 1 < len(args):
                year = int(args[idx + 1])
        fetch_season_update(year)

    elif '--season' in args:
        # 指定シーズンのみ取得
        idx = args.index('--season')
        if idx + 1 < len(args):
            year = int(args[idx + 1])
            fetch_season_update(year)
        else:
            print("Error: --season requires a year argument")
            sys.exit(1)

    else:
        # 初回セットアップ（デフォルト）
        fetch_initial_data()

    print()
    print("=" * 60)
    print("✅ DATA FETCH COMPLETE")
    print(f"   All files saved to: {DATA_DIR}/")
    print()
    print("Next steps:")
    print("  - Initial setup: python3 train_model_v2.py")
    print("  - Daily update:  python3 fetch_real_data.py --update")
    print("  - Run pipeline:  python3 run_daily.py")
    print("=" * 60)


if __name__ == '__main__':
    main()
