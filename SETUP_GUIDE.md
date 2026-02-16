# Moneyball Dojo - セットアップガイド

## ステップ1: ドメイン＆アカウント登録（今日中に実施）

### ドメイン登録
1. **Namecheap**（https://www.namecheap.com）または **Google Domains** にアクセス
2. `moneyballdojo.com` を検索
3. 代替案: `moneyballdojo.io`（テック感がある）
4. 費用: .comで年間約$10-15

### Twitter/X アカウント
1. https://twitter.com/signup にアクセス
2. `@MoneyballDojo` を登録
3. プロフィール案: "東京発・AI駆動のMLB予測 | 日本人AIエンジニアが開発 | 勘より数字で勝負"
4. 初日にアプローチを説明するスレッドを固定投稿

### Substack アカウント
1. https://substack.com にアクセス
2. パブリケーション作成: `moneyballdojo.substack.com`
3. 後から: カスタムドメイン `moneyballdojo.com` を接続（Substackはこれをサポート）
4. 無料プランで即開始、有料購読者500人以上で有料プラン（$12.99/月）を追加

### note.com アカウント
1. https://note.com にアクセス
2. Moneyball Dojoブランディングでアカウント作成
3. プロフィール: "東京在住のAIエンジニアが、機械学習でMLBを本気で予測する。NPB×MLB×AIの交差点から。"
4. 無料記事でスタート、後から有料マガジン（¥800/月）を追加

---

## ステップ2: GitHubリポジトリ

```bash
# リポジトリ作成
gh repo create moneyball-dojo --private --description "AI駆動のMLB/NFL予測システム"

# クローンしてファイルを追加
git clone https://github.com/YOUR_USERNAME/moneyball-dojo.git
cd moneyball-dojo

# プロトタイプファイルをコピー
cp -r /path/to/moneyball_dojo/* .

# シークレット追加（GitHub Settings > Secrets > Actions）
# - CLAUDE_API_KEY: Anthropic APIキー
# - SHEETS_CREDENTIALS: Google サービスアカウント JSON（base64エンコード済み）
```

### 必要なGitHub Secrets:
| シークレット名 | 説明 | 取得方法 |
|---------------|------|---------|
| `CLAUDE_API_KEY` | Anthropic APIキー | https://console.anthropic.com |
| `SHEETS_CREDENTIALS` | Google Sheets サービスアカウント | GCP Console > IAM > サービスアカウント |
| `SUBSTACK_EMAIL` | Substackログインメール | 登録時のメールアドレス |
| `SUBSTACK_PASSWORD` | Substackパスワード | 登録時のパスワード |

---

## ステップ3: Google Sheets データベース

1. Google Cloud Console（https://console.cloud.google.com）にアクセス
2. 新規プロジェクト作成: "Moneyball Dojo"
3. Google Sheets API を有効化
4. サービスアカウント作成:
   - IAM と管理 > サービスアカウント > 作成
   - JSONキーファイルをダウンロード
5. Google スプレッドシート作成:
   - 名前: "Moneyball Dojo - Predictions 2026"
   - サービスアカウントのメールアドレスに編集権限を付与
6. イベントログ方式のスプレッドシートを作成（`sheets_schema_v2.py` 参照）

### gspread 動作テスト:
```python
import gspread
from google.oauth2.service_account import Credentials

creds = Credentials.from_service_account_file('credentials.json',
    scopes=['https://www.googleapis.com/auth/spreadsheets'])
client = gspread.authorize(creds)

sheet = client.open('Moneyball Dojo - Predictions 2026')
ws = sheet.worksheet('Daily Predictions')
ws.append_row(['2026-03-28', 'NYY', 'BOS', 'Cole', 'Sale', 0.62, 'HOME', 'HIGH'])
print("テスト行の追加成功！")
```

---

## ステップ4: Claude API セットアップ

1. https://console.anthropic.com にアクセス
2. APIキーを作成
3. 以下で動作確認:

```python
import anthropic

client = anthropic.Anthropic(api_key="your-key")

response = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=300,
    messages=[{"role": "user", "content": "NYY vs BOSの100字のMLB試合プレビューを書いて。データ重視で簡潔に。"}]
)
print(response.content[0].text)
```

### コスト見積もり:
- Haiku Batch API: 約$0.005/記事
- 10記事/日 × 180日 = シーズンあたり$9
- 予算: 最初は月$20の支出上限を設定

---

## ステップ5: プレシーズン・コンテンツ計画

### プレシーズン記事（✅ 全4本作成済み — articles/フォルダ）:
- [x] 01: "Introducing Moneyball Dojo" — オリジン・ストーリー
- [x] 02: "How Our AI Model Works" — 方法論の解説
- [x] 03: "2025 Season Review" — データ分析
- [x] 04: "2026 Division Predictions" — 地区別予測

### 開幕日（2026年3月27日）:
- [ ] 初日の全予測と詳細分析
- [ ] Twitterスレッド: "AIが選ぶ開幕日の予測"
- [ ] note.com: "開幕日予測：AIが見る2026年MLBシーズン"

### 日次運用:
- [ ] 朝: 予測を生成（自動化）
- [ ] 昼: Substack公開（自動化または手動）
- [ ] 夕: 結果更新、成績をツイート
- [ ] 週次: パフォーマンスレビュー + 深掘り分析記事

---

## チェックリスト

- [ ] moneyballdojo.com を登録
- [ ] Twitter/Xで @MoneyballDojo を作成
- [ ] moneyballdojo.substack.com を作成
- [ ] note.com アカウントを作成
- [ ] GitHub プライベートリポジトリをセットアップ
- [ ] Google Cloud プロジェクト + Sheets API をセットアップ
- [ ] Google スプレッドシートを作成（sheets_schema_v2.py参照）
- [ ] Claude APIキーを取得
- [ ] `python run_daily.py` でパイプラインをテスト
- [x] プレシーズン記事4本作成済み（articles/フォルダ）
- [ ] ロゴをデザイン（確定版）
- [ ] Twitter での発信を開始（投稿開始）
