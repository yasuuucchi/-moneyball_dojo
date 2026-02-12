#!/usr/bin/env python3
import statsapi
import pandas as pd
import time
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

DATA_DIR = Path(__file__).parent / "data"
GAMES_FILE = DATA_DIR / "games_2022_2025.csv"
OUTPUT_FILE = DATA_DIR / "game_props_data_2024_2025.csv"

def get_game_props(game_id):
    try:
        data = statsapi.get('game', {'gamePk': game_id})
        live_data = data.get('liveData', {})
        boxscore = live_data.get('boxscore', {})
        linescore = live_data.get('linescore', {})
        
        # NRFI/YRFI
        innings = linescore.get('innings', [])
        nrfi = 1
        if innings:
            first = innings[0]
            if int(first.get('home', {}).get('runs', 0)) + int(first.get('away', {}).get('runs', 0)) > 0:
                nrfi = 0
        
        # Stolen Bases
        home_sb = boxscore.get('teams', {}).get('home', {}).get('teamStats', {}).get('batting', {}).get('stolenBases', 0)
        away_sb = boxscore.get('teams', {}).get('away', {}).get('teamStats', {}).get('batting', {}).get('stolenBases', 0)
        
        # Pitcher Outs (Starter)
        pitcher_stats = {}
        for side in ['home', 'away']:
            p_list = boxscore.get('teams', {}).get(side, {}).get('pitchers', [])
            if p_list:
                starter_id = p_list[0]
                p_data = boxscore.get('teams', {}).get(side, {}).get('players', {}).get(f'ID{starter_id}', {})
                p_stats = p_data.get('stats', {}).get('pitching', {})
                pitcher_stats[f'{side}_starter_id'] = starter_id
                pitcher_stats[f'{side}_starter_ip'] = p_stats.get('inningsPitched', '0.0')
                pitcher_stats[f'{side}_starter_k'] = p_stats.get('strikeOuts', 0)
        
        return {
            'game_id': game_id,
            'nrfi': nrfi,
            'total_sb': home_sb + away_sb,
            **pitcher_stats
        }
    except Exception as e:
        return None

def main():
    if not GAMES_FILE.exists():
        print("Games file not found")
        return

    df = pd.read_csv(GAMES_FILE)
    # Focus on 2024-2025 for now to get high-quality recent data quickly
    df = df[df['year'] >= 2024].copy()
    
    print(f"Processing {len(df)} games from 2024-2025...")
    
    results = []
    processed = 0
    
    # Use ThreadPool to speed up API calls
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_game = {executor.submit(get_game_props, int(row['game_id'])): row for _, row in df.iterrows()}
        
        for future in as_completed(future_to_game):
            res = future.result()
            processed += 1
            if res:
                results.append(res)
            
            if processed % 50 == 0:
                print(f"Progress: {processed}/{len(df)}")
                # Periodically save
                pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)

    pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)
    print(f"Done. Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
