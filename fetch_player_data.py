#!/usr/bin/env python3
"""
Moneyball Dojo — 選手レベルデータ取得
=====================================
投手K Props / 打者Hit・HR Props のモデル構築に必要な個人データを取得。

使い方:
  pip3 install MLB-StatsAPI pandas
  python3 fetch_player_data.py              # 2022-2024の選手データ
  python3 fetch_player_data.py --update     # 今シーズン更新

出力:
  data/pitcher_stats_YYYY.csv   — 投手の年間成績（先発のみ）
  data/pitcher_gamelogs_YYYY.csv — 投手の試合別成績
  data/batter_stats_YYYY.csv    — 打者の年間成績
  data/batter_gamelogs_YYYY.csv  — 打者の試合別成績
"""

import statsapi
import pandas as pd
import time
import sys
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def get_roster_player_ids(year):
    """全チームのロスターから選手IDを取得"""
    print(f"  Getting {year} rosters...")
    teams_data = statsapi.get('teams', {'sportId': 1, 'season': year})

    pitchers = []
    batters = []

    for team in teams_data.get('teams', []):
        team_id = team['id']
        team_name = team['name']
        team_abbr = team.get('abbreviation', '')

        try:
            roster = statsapi.get('team_roster', {
                'teamId': team_id,
                'season': year,
                'rosterType': 'fullSeason'
            })

            for player in roster.get('roster', []):
                pid = player['person']['id']
                name = player['person']['fullName']
                pos = player.get('position', {}).get('abbreviation', '')

                if pos in ('P', 'SP', 'RP'):
                    pitchers.append({
                        'player_id': pid, 'name': name, 'position': pos,
                        'team_name': team_name, 'team_abbr': team_abbr, 'year': year
                    })
                elif pos == 'TWP':
                    # Two-way players go to both lists
                    pitchers.append({
                        'player_id': pid, 'name': name, 'position': pos,
                        'team_name': team_name, 'team_abbr': team_abbr, 'year': year
                    })
                    batters.append({
                        'player_id': pid, 'name': name, 'position': pos,
                        'team_name': team_name, 'team_abbr': team_abbr, 'year': year
                    })
                else:
                    batters.append({
                        'player_id': pid, 'name': name, 'position': pos,
                        'team_name': team_name, 'team_abbr': team_abbr, 'year': year
                    })

            time.sleep(0.2)
        except Exception as e:
            print(f"    ⚠ Error for {team_name}: {e}")

    print(f"    ✓ {len(pitchers)} pitchers, {len(batters)} batters")
    return pitchers, batters


def fetch_pitcher_season_stats(pitchers, year):
    """先発投手のシーズン成績"""
    print(f"  Fetching {year} pitcher season stats...")

    records = []
    seen = set()

    for p in pitchers:
        pid = p['player_id']
        if pid in seen:
            continue
        seen.add(pid)

        try:
            stats = statsapi.player_stat_data(
                pid, group="pitching", type="yearByYear",
                sportId=1
            )

            # Each entry in stats['stats'] is one season:
            # {type: 'yearByYear', group: 'pitching', season: '2024', stats: {gamesPlayed: ..., era: ...}}
            for stat_entry in stats.get('stats', []):
                entry_group = stat_entry.get('group', '')
                entry_season = str(stat_entry.get('season', ''))

                if entry_group == 'pitching' and entry_season == str(year):
                    s = stat_entry.get('stats', {})
                    if not isinstance(s, dict):
                        continue

                    gs = int(s.get('gamesStarted', 0))
                    if gs < 5:  # 先発5試合未満は除外
                        continue

                    records.append({
                        'player_id': pid,
                        'name': p['name'],
                        'team_name': p['team_name'],
                        'year': year,
                        'games': int(s.get('gamesPlayed', 0)),
                        'games_started': gs,
                        'innings': float(s.get('inningsPitched', '0').replace(' ', '')) if isinstance(s.get('inningsPitched'), str) else float(s.get('inningsPitched', 0)),
                        'wins': int(s.get('wins', 0)),
                        'losses': int(s.get('losses', 0)),
                        'ERA': float(s.get('era', '0')) if s.get('era') else 0,
                        'WHIP': float(s.get('whip', '0')) if s.get('whip') else 0,
                        'strikeouts': int(s.get('strikeOuts', 0)),
                        'walks': int(s.get('baseOnBalls', 0)),
                        'hits_allowed': int(s.get('hits', 0)),
                        'home_runs_allowed': int(s.get('homeRuns', 0)),
                        'K_per_9': float(s.get('strikeoutsPer9Inn', '0')) if s.get('strikeoutsPer9Inn') else 0,
                        'BB_per_9': float(s.get('walksPer9Inn', '0')) if s.get('walksPer9Inn') else 0,
                        'K_per_game': int(s.get('strikeOuts', 0)) / gs if gs > 0 else 0,
                        'avg_innings_per_start': float(s.get('inningsPitched', '0').replace(' ', '')) / gs if gs > 0 and isinstance(s.get('inningsPitched'), str) else float(s.get('inningsPitched', 0)) / gs if gs > 0 else 0,
                    })
                    break  # Found the year, no need to continue

            time.sleep(0.15)
        except Exception as e:
            continue

    print(f"    ✓ {len(records)} qualifying starting pitchers")
    return records


def fetch_batter_season_stats(batters, year):
    """打者のシーズン成績"""
    print(f"  Fetching {year} batter season stats...")

    records = []
    seen = set()

    for b in batters:
        pid = b['player_id']
        if pid in seen:
            continue
        seen.add(pid)

        try:
            stats = statsapi.player_stat_data(
                pid, group="hitting", type="yearByYear",
                sportId=1
            )

            for stat_entry in stats.get('stats', []):
                entry_group = stat_entry.get('group', '')
                entry_season = str(stat_entry.get('season', ''))

                if entry_group == 'hitting' and entry_season == str(year):
                    s = stat_entry.get('stats', {})
                    if not isinstance(s, dict):
                        continue

                    g = int(s.get('gamesPlayed', 0))
                    ab = int(s.get('atBats', 0))
                    if g < 30 or ab < 100:  # 30試合/100打席未満は除外
                        continue

                    records.append({
                        'player_id': pid,
                        'name': b['name'],
                        'team_name': b['team_name'],
                        'year': year,
                        'games': g,
                        'at_bats': ab,
                        'plate_appearances': int(s.get('plateAppearances', 0)),
                        'hits': int(s.get('hits', 0)),
                        'doubles': int(s.get('doubles', 0)),
                        'triples': int(s.get('triples', 0)),
                        'home_runs': int(s.get('homeRuns', 0)),
                        'rbi': int(s.get('rbi', 0)),
                        'walks': int(s.get('baseOnBalls', 0)),
                        'strikeouts': int(s.get('strikeOuts', 0)),
                        'stolen_bases': int(s.get('stolenBases', 0)),
                        'BA': float(s.get('avg', '.000').replace('.', '0.', 1)) if isinstance(s.get('avg'), str) else float(s.get('avg', 0)),
                        'OBP': float(s.get('obp', '.000').replace('.', '0.', 1)) if isinstance(s.get('obp'), str) else float(s.get('obp', 0)),
                        'SLG': float(s.get('slg', '.000').replace('.', '0.', 1)) if isinstance(s.get('slg'), str) else float(s.get('slg', 0)),
                        'OPS': float(s.get('ops', '.000').replace('.', '0.', 1)) if isinstance(s.get('ops'), str) else float(s.get('ops', 0)),
                        'hits_per_game': int(s.get('hits', 0)) / g if g > 0 else 0,
                        'HR_per_game': int(s.get('homeRuns', 0)) / g if g > 0 else 0,
                                'K_rate': int(s.get('strikeOuts', 0)) / int(s.get('plateAppearances', 1)) if int(s.get('plateAppearances', 1)) > 0 else 0,
                            })

            time.sleep(0.15)
        except Exception as e:
            continue

    print(f"    ✓ {len(records)} qualifying batters")
    return records


def fetch_data_for_year(year):
    """指定年のデータを取得"""
    print(f"\n{'='*50}")
    print(f"Fetching {year} player data")
    print(f"{'='*50}")

    # ロスター取得
    pitchers, batters = get_roster_player_ids(year)

    # 投手成績
    pitcher_stats = fetch_pitcher_season_stats(pitchers, year)
    if pitcher_stats:
        df = pd.DataFrame(pitcher_stats)
        path = DATA_DIR / f"pitcher_stats_{year}.csv"
        df.to_csv(path, index=False)
        print(f"  ✓ Saved → {path}")

    # 打者成績
    batter_stats = fetch_batter_season_stats(batters, year)
    if batter_stats:
        df = pd.DataFrame(batter_stats)
        path = DATA_DIR / f"batter_stats_{year}.csv"
        df.to_csv(path, index=False)
        print(f"  ✓ Saved → {path}")

    return pitcher_stats, batter_stats


def main():
    print("="*60)
    print("MONEYBALL DOJO — PLAYER DATA FETCH")
    print("="*60)

    args = sys.argv[1:]

    if '--update' in args:
        year = datetime.now().year
        if '--season' in args:
            idx = args.index('--season')
            if idx + 1 < len(args):
                year = int(args[idx + 1])
        fetch_data_for_year(year)
    elif '--season' in args:
        idx = args.index('--season')
        if idx + 1 < len(args):
            year = int(args[idx + 1])
            fetch_data_for_year(year)
    else:
        # 初回: 2022-2025
        for year in [2022, 2023, 2024, 2025]:
            fetch_data_for_year(year)
            time.sleep(2)

    print(f"\n{'='*60}")
    print("✅ PLAYER DATA FETCH COMPLETE")
    print(f"   Files saved to: {DATA_DIR}/")
    print()
    print("Next: python3 train_all_models.py")
    print("="*60)


if __name__ == '__main__':
    main()
