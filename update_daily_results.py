
import os
import sys
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import statsapi

# Google Sheets用（オプション）
try:
    import gspread
    SHEETS_AVAILABLE = True
except ImportError:
    SHEETS_AVAILABLE = False

PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"
GAMES_PATH = DATA_DIR / "games_2022_2025.csv"
NRFI_PATH = DATA_DIR / "nrfi_data_2022_2025.csv"
CREDENTIALS_PATH = PROJECT_DIR / "credentials.json"
SPREADSHEET_NAME = "Moneyball Dojo DB"

def update_game_results(target_date):
    """MLB APIから指定日の結果を取得し、CSVに追記する"""
    print(f"--- Updating results for {target_date} ---")
    
    # 1. 試合スケジュールとスコアを取得
    try:
        sched = statsapi.schedule(start_date=target_date, end_date=target_date)
    except Exception as e:
        print(f"❌ Error fetching from MLB API: {e}")
        return []

    if not sched:
        print(f"No games found for {target_date}.")
        return []

    # 2. 既存のデータを読み込み
    if GAMES_PATH.exists():
        games_df = pd.read_csv(GAMES_PATH)
        existing_ids = set(games_df['game_id'].astype(int).tolist())
    else:
        print(f"⚠ {GAMES_PATH} not found. Creating new.")
        games_df = pd.DataFrame()
        existing_ids = set()

    new_rows = []
    nrfi_rows = []

    for game in sched:
        g_id = int(game['game_id'])
        if game['status'] != 'Final':
            print(f"  Skipping Game {g_id}: Status is {game['status']}")
            continue
        
        if g_id in existing_ids:
            print(f"  Skipping Game {g_id}: Already exists in CSV")
            continue

        home_score = int(game.get('home_score', 0))
        away_score = int(game.get('away_score', 0))
        home_win = 1 if home_score > away_score else 0
        
        # メインのゲームデータ
        row = {
            'game_id': g_id,
            'date': target_date,
            'year': int(target_date.split('-')[0]),
            'home_team': game.get('home_name'),
            'home_abbr': '', # APIからは直接取れない場合がある
            'away_team': game.get('away_name'),
            'away_abbr': '',
            'home_score': home_score,
            'away_score': away_score,
            'home_win': home_win,
            'winning_team': game.get('home_name') if home_win else game.get('away_name'),
            'losing_team': game.get('away_name') if home_win else game.get('home_name'),
            'home_pitcher': game.get('home_probable_pitcher', 'Unknown'),
            'away_pitcher': game.get('away_probable_pitcher', 'Unknown'),
            'venue': game.get('venue_name', 'Unknown')
        }
        new_rows.append(row)

        # NRFIデータ（イニングスコア取得）
        try:
            game_data = statsapi.get('game', {'gamePk': g_id})
            linescore = game_data.get('liveData', {}).get('linescore', {})
            innings = linescore.get('innings', [])
            if innings:
                f_inn = innings[0]
                h_1st = int(f_inn.get('home', {}).get('runs', 0))
                a_1st = int(f_inn.get('away', {}).get('runs', 0))
                nrfi_rows.append({
                    'game_id': g_id,
                    'date': target_date,
                    'year': int(target_date.split('-')[0]),
                    'home_team': row['home_team'],
                    'away_team': row['away_team'],
                    'home_1st_runs': h_1st,
                    'away_1st_runs': a_1st,
                    'total_1st_runs': h_1st + a_1st,
                    'nrfi_result': 1 if (h_1st + a_1st) == 0 else 0
                })
        except:
            pass

    # 3. CSVに保存
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        new_df.to_csv(GAMES_PATH, mode='a', header=not GAMES_PATH.exists(), index=False)
        print(f"✅ Added {len(new_rows)} games to {GAMES_PATH.name}")
        
        if nrfi_rows and NRFI_PATH.exists():
            pd.DataFrame(nrfi_rows).to_csv(NRFI_PATH, mode='a', header=False, index=False)
            print(f"✅ Added {len(nrfi_rows)} NRFI records to {NRFI_PATH.name}")
    else:
        print("No new results to add to CSV.")

    return new_rows

def sync_with_sheets(results, target_date):
    """Google Sheetsのresultsシートを更新する"""
    if not SHEETS_AVAILABLE or not results:
        return

    if not CREDENTIALS_PATH.exists():
        print("⚠ credentials.json not found. Skipping Sheets sync.")
        return

    try:
        gc = gspread.service_account(filename=str(CREDENTIALS_PATH))
        sh = gc.open(SPREADSHEET_NAME)
        
        # 1. resultsシートに実績を追記
        try:
            ws_res = sh.worksheet('results')
            for r in results:
                # 簡易的な実績行（実際には予測データと結合するのが理想）
                ws_res.append_row([
                    target_date,
                    r['game_id'],
                    r['away_team'],
                    r['home_team'],
                    f"{r['away_score']}-{r['home_score']}",
                    "HOME" if r['home_win'] else "AWAY"
                ])
            print(f"✅ Updated 'results' sheet in Google Sheets.")
        except:
            print("⚠ 'results' sheet not found in Google Sheets.")

    except Exception as e:
        print(f"❌ Sheets sync failed: {e}")

if __name__ == "__main__":
    # 引数があればその日、なければ昨日のデータを取得
    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        target = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    new_data = update_game_results(target)
    sync_with_sheets(new_data, target)
