#!/usr/bin/env python3
"""
Moneyball Dojo — NRFI/YRFI Data Fetcher
=======================================
Fetch 1st inning scores for all historical games to train the NRFI model.
Uses statsapi.get('game') → liveData.linescore.innings[0] for accurate data.

Usage:
  python3 fetch_nrfi_data.py
  python3 fetch_nrfi_data.py --limit 100   # Test with first 100 games
"""

import statsapi
import pandas as pd
import time
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
GAMES_FILE = DATA_DIR / "games_2022_2025.csv"
OUTPUT_FILE = DATA_DIR / "nrfi_data_2022_2025.csv"

def fetch_nrfi_data(limit=None):
    if not GAMES_FILE.exists():
        print(f"❌ Games file not found: {GAMES_FILE}")
        return

    print(f"Reading games from {GAMES_FILE}...")
    games_df = pd.read_csv(GAMES_FILE)

    if limit:
        games_df = games_df.head(limit)

    # Start fresh (overwrite existing)
    print(f"Target: {len(games_df)} games to process.")

    records = []
    count = 0
    errors = 0

    print("Starting fetch loop (Ctrl+C to stop safely)...")

    try:
        for idx, game in games_df.iterrows():
            game_id = int(game['game_id'])

            try:
                # Use raw game endpoint for accurate linescore data
                data = statsapi.get('game', {'gamePk': game_id})
                linescore = data.get('liveData', {}).get('linescore', {})
                innings = linescore.get('innings', [])

                if not innings:
                    errors += 1
                    continue

                first_inning = innings[0]
                home_1st = int(first_inning.get('home', {}).get('runs', 0))
                away_1st = int(first_inning.get('away', {}).get('runs', 0))
                total_1st = home_1st + away_1st
                nrfi = 1 if total_1st == 0 else 0

                record = {
                    'game_id': game_id,
                    'date': game['date'],
                    'year': game['year'],
                    'home_team': game['home_team'],
                    'away_team': game['away_team'],
                    'home_1st_runs': home_1st,
                    'away_1st_runs': away_1st,
                    'total_1st_runs': total_1st,
                    'nrfi_result': nrfi  # 1=NRFI (No Run), 0=YRFI (Yes Run)
                }
                records.append(record)
                count += 1

                label = 'NRFI' if nrfi else 'YRFI'
                print(f"\r[{count}/{len(games_df)}] {game['date']} {game['away_team']}@{game['home_team']}: 1st={total_1st} ({label})", end="")

                # Save batch every 200 games
                if len(records) >= 200:
                    df = pd.DataFrame(records)
                    if count <= 200:
                        # First batch: write with header
                        df.to_csv(OUTPUT_FILE, index=False)
                    else:
                        df.to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
                    records = []

            except KeyboardInterrupt:
                print("\n\nStopping safely...")
                break
            except Exception as e:
                errors += 1
                continue

    except KeyboardInterrupt:
        print("\nStopping...")

    # Save remaining
    if records:
        df = pd.DataFrame(records)
        if count <= len(records):
            df.to_csv(OUTPUT_FILE, index=False)
        else:
            df.to_csv(OUTPUT_FILE, mode='a', header=False, index=False)

    print(f"\n\nDone. Processed: {count}, Errors: {errors}")

    # Show stats
    if OUTPUT_FILE.exists():
        result_df = pd.read_csv(OUTPUT_FILE)
        nrfi_rate = result_df['nrfi_result'].mean()
        print(f"NRFI Rate: {nrfi_rate:.1%} ({int(result_df['nrfi_result'].sum())}/{len(result_df)})")
    
    print(f"Data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    limit = None
    if '--limit' in sys.argv:
        idx = sys.argv.index('--limit')
        if idx + 1 < len(sys.argv):
            limit = int(sys.argv[idx + 1])
    
    fetch_nrfi_data(limit=limit)
