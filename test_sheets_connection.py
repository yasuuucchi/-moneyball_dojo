"""
Moneyball Dojo — Google Sheets接続テスト
=========================================
このスクリプトをローカルで実行して、Sheets APIが正しく設定されているか確認します。

使い方:
  pip install gspread
  python test_sheets_connection.py
"""

import gspread
from pathlib import Path

CREDENTIALS_PATH = Path(__file__).parent / "credentials.json"
SPREADSHEET_NAME = "Moneyball Dojo DB"


def main():
    print("=" * 50)
    print("MONEYBALL DOJO — Sheets接続テスト")
    print("=" * 50)
    print()

    # 1. credentials.json確認
    if not CREDENTIALS_PATH.exists():
        print("❌ credentials.json が見つかりません")
        print(f"   期待するパス: {CREDENTIALS_PATH}")
        return False
    print("✓ credentials.json 発見")

    # 2. 接続
    try:
        gc = gspread.service_account(filename=str(CREDENTIALS_PATH))
        print("✓ Google認証成功")
    except Exception as e:
        print(f"❌ Google認証失敗: {e}")
        return False

    # 3. スプレッドシートを開く
    try:
        sh = gc.open(SPREADSHEET_NAME)
        print(f"✓ スプレッドシート '{sh.title}' に接続")
        print(f"  ID: {sh.id}")
    except gspread.SpreadsheetNotFound:
        print(f"❌ '{SPREADSHEET_NAME}' が見つかりません")
        print("  → Google Sheetsで 'Moneyball Dojo DB' を作成してください")
        print("  → サービスアカウントのメールに編集権限を共有してください")
        return False
    except Exception as e:
        print(f"❌ スプレッドシートを開けません: {e}")
        return False

    # 4. シート一覧
    worksheets = sh.worksheets()
    print(f"  既存シート: {[ws.title for ws in worksheets]}")

    # 5. 必要なシートを作成（なければ）
    required_sheets = {
        'predictions': ['date', 'game_id', 'away_team', 'home_team', 'win_prob',
                        'market_odds', 'implied_prob', 'edge', 'pick', 'confidence'],
        'results': ['date', 'game_id', 'away_team', 'home_team', 'pick',
                    'confidence', 'actual_winner', 'correct', 'edge'],
        'daily_kpi': ['date', 'total_games', 'strong_picks', 'moderate_picks',
                      'correct', 'accuracy', 'roi'],
    }

    existing_titles = [ws.title for ws in worksheets]

    for sheet_name, headers in required_sheets.items():
        if sheet_name not in existing_titles:
            ws = sh.add_worksheet(title=sheet_name, rows=1000, cols=len(headers))
            ws.append_row(headers)
            print(f"  ✓ シート '{sheet_name}' を作成（ヘッダー付き）")
        else:
            print(f"  ✓ シート '{sheet_name}' は既に存在")

    # 6. デフォルトの Sheet1 を削除（あれば）
    try:
        default_sheet = sh.worksheet("Sheet1")
        if len(sh.worksheets()) > 1:
            sh.del_worksheet(default_sheet)
            print("  ✓ デフォルトの 'Sheet1' を削除")
    except gspread.WorksheetNotFound:
        pass

    # 7. テスト書き込み
    try:
        ws = sh.worksheet('predictions')
        test_row = ['2026-02-09', 'TEST_001', 'NYY', 'BOS', '0.55',
                    '-110', '0.524', '0.026', 'HOME', 'LEAN']
        ws.append_row(test_row)
        print("  ✓ テストデータの書き込み成功")

        # テストデータを削除
        all_values = ws.get_all_values()
        if len(all_values) > 1:
            ws.delete_rows(len(all_values))
            print("  ✓ テストデータを削除")
    except Exception as e:
        print(f"  ⚠ 書き込みテスト失敗: {e}")
        print("  → 編集権限を確認してください")
        return False

    print()
    print("=" * 50)
    print("✅ 全テスト合格！Sheets連携の準備完了です。")
    print("=" * 50)
    return True


if __name__ == '__main__':
    main()
