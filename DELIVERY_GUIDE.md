# Moneyball Dojo 配信プラットフォームガイド

> **目的**: GitHub Actions → Beehiiv/Buffer → 読者 の完全自動配信パイプラインの構築手順
> **最終更新**: 2026年2月24日

---

## なぜ Beehiiv なのか（Substack との比較）

| 項目 | Beehiiv | Substack |
|------|---------|---------|
| **プラットフォーム手数料** | **0%** | **10%**（年$10万売上で$1万の損失） |
| **API** | **オープンAPI** → 完全自動投稿可能 | 限定的 → 自動化困難 |
| **カスタムドメイン** | 無料プランから対応 | 有料プランのみ |
| **A/Bテスト** | 件名、送信時間のA/Bテスト | なし |
| **コホート分析** | 購読者のセグメント分析 | 基本的な統計のみ |
| **マネタイズ** | Stripe直接連携（手数料なし） | Stripe経由だが10%上乗せ |
| **無料プラン** | 2,500購読者まで | 無制限（ただし10%取られる） |

**結論**: 自動化パイプラインにはBeehiivが圧倒的に有利。

---

## Part 1: Beehiiv セットアップ（10分）

### Step 1: アカウント作成
1. beehiiv.com にアクセス → "Start for free" をクリック
2. ニュースレター名: **Moneyball Dojo**
3. URL: `moneyballdojo.beehiiv.com`（後でカスタムドメインに変更可能）
4. カテゴリ: Sports / Data & Analytics

### Step 2: プロフィール設定
- ニュースレター説明文（英語）:

```
AI-powered MLB predictions from a 9-model XGBoost ensemble.
Daily picks with edge analysis. Backtested: 64.7% accuracy, 72.9% STRONG-tier win rate (+15.3% ROI).
Built by a Japanese AI engineer in Tokyo.
```

- ロゴ: MoneyballDojoのロゴをアップロード
- カラースキーム: ダークブルー (#1e3a5f) + ゴールド (#d4af37)

### Step 3: ウェルカムメール設定
1. Settings → Automations → Welcome Email
2. `WELCOME_EMAIL_EN.md` の内容をコピペ
3. 「Send test email」で確認

### Step 4: API キー取得（自動投稿用）
1. Settings → Integrations → API
2. "Generate new API key" をクリック
3. スコープ: `posts:write`, `posts:read`
4. API キーをコピー → GitHub Secrets に保存: `BEEHIIV_API_KEY`
5. Publication ID もコピー → GitHub Secrets に保存: `BEEHIIV_PUB_ID`

---

## Part 2: Buffer セットアップ（X自動投稿）

### なぜ Buffer か
- 月$6で X への自動投稿をスケジュール可能
- APIがシンプル → GitHub Actions から直接叩ける
- X の ToS に準拠（直接API投稿より安全）

### Step 1: アカウント作成
1. buffer.com にアクセス → 無料プラン or Essentials ($6/月)
2. X アカウント（@MoneyballDojo）を連携

### Step 2: 投稿スケジュール設定
1. Publishing → Posting Schedule
2. 以下のスケジュールを設定:
   - 毎日 8:00 AM ET（米国東部時間）→ Daily Digest の要約
   - 毎日 12:00 PM ET → 本日のトップピック
   - 毎日 6:00 PM ET → 試合結果速報（夜間バッチ）

### Step 3: API アクセストークン取得
1. Settings → API & Integrations
2. Access Token をコピー
3. GitHub Secrets に保存: `BUFFER_ACCESS_TOKEN`

---

## Part 3: GitHub Actions への統合

### 必要な Secrets（リポジトリの Settings → Secrets）

| Secret 名 | 値 | 用途 |
|-----------|-----|------|
| `BEEHIIV_API_KEY` | Beehiiv API キー | 記事自動投稿 |
| `BEEHIIV_PUB_ID` | Beehiiv Publication ID | 記事自動投稿 |
| `BUFFER_ACCESS_TOKEN` | Buffer アクセストークン | X 自動投稿 |
| `CLAUDE_API_KEY` | Anthropic API キー | 記事生成 |
| `GOOGLE_SHEETS_CREDENTIALS` | GCP サービスアカウント JSON | Sheets書き込み |
| `GOOGLE_SHEETS_ID` | スプレッドシートID | Sheets書き込み |
| `SLACK_WEBHOOK` | Slack Webhook URL（任意） | 通知 |

### run_daily.py への追加（概要）

`run_daily.py` の末尾に以下のステップを追加する:

```python
# === Beehiiv 自動投稿 ===
import requests

def publish_to_beehiiv(title, content_html, pub_id, api_key):
    """Beehiiv API で記事を自動投稿"""
    url = f"https://api.beehiiv.com/v2/publications/{pub_id}/posts"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "title": title,
        "subtitle": "Daily AI-Powered MLB Predictions",
        "content": [{"type": "html", "html": content_html}],
        "status": "confirmed",  # 即時公開
        "send_to": "all"        # 全購読者にメール送信
    }
    response = requests.post(url, json=payload, headers=headers)
    return response.status_code == 201


# === Buffer 自動投稿 ===
def post_to_buffer(text, access_token, profile_id):
    """Buffer API で X に自動投稿をスケジュール"""
    url = "https://api.bufferapp.com/1/updates/create.json"
    payload = {
        "access_token": access_token,
        "profile_ids[]": profile_id,
        "text": text,
        "now": False,  # スケジュールに従って投稿
    }
    response = requests.post(url, data=payload)
    return response.status_code == 200
```

### ワークフローの更新（概要）

`github_actions_workflow.yml` の `daily-pipeline` ジョブの環境変数に追加:

```yaml
env:
  BEEHIIV_API_KEY: ${{ secrets.BEEHIIV_API_KEY }}
  BEEHIIV_PUB_ID: ${{ secrets.BEEHIIV_PUB_ID }}
  BUFFER_ACCESS_TOKEN: ${{ secrets.BUFFER_ACCESS_TOKEN }}
```

---

## Part 4: 情報の流れ（完成形）

```
毎朝 06:00 UTC（GitHub Actions トリガー）
│
├─ MLB Stats API → データ取得
├─ XGBoost 9モデル → 予測生成
├─ Claude API (Haiku) → EN/JA 記事生成
│
├─→ Beehiiv API → 英語 Digest 自動パブリッシュ → 全購読者にメール配信
├─→ Buffer API → X 投稿スケジュール（1日3回）
├─→ Google Sheets → 予測ログ自動追記
│
└─ 15:00 UTC（結果バッチ）
   ├─ MLB Stats API → 試合結果取得
   ├─ Google Sheets → 勝率・ROI 更新
   └─ Buffer API → 結果速報 X 投稿

Taiki の手動作業:
  - note.com に日本語版を週1投稿（15分/週）
  - Manus で X エンゲージメント（30分/週）
```

---

## Part 5: note.com の運用

note.com は公式APIがないため、完全自動化は不可。

### 運用方法（週1回、15分）

**Option A: 手動コピペ**
1. GitHub の `output/YYYYMMDD/digest_JA_*.md` を開く
2. note.com の新規投稿にMarkdownをコピペ
3. 投稿ボタンを押す

**Option B: Manus で自動化**
1. 毎週月曜にManusに指示:
   「GitHubのoutput/から最新の日本語Digestを取得して、note.comに投稿して」
2. Manusがブラウザ操作で投稿を完了

### note.com の投稿ガイドライン
- タイトル: ベッティング用語を避ける（「予測」「分析」を使う）
- タグ: #MLB #大谷翔平 #データ分析 #AI #野球
- 投稿時間: 日本時間 12:00 or 20:00
- 頻度: 週1回（月曜 or 木曜）

---

## トラブルシューティング

| 問題 | 解決策 |
|------|--------|
| Beehiiv API が 401 エラー | API キーの有効期限を確認。Settings → API で再生成 |
| Buffer に投稿されない | X アカウントの再連携が必要な場合あり。buffer.com で確認 |
| 記事が空で投稿される | Claude API の応答を確認。フォールバックテンプレートが作動しているか |
| note.com にログインできない | Manus のセッションが切れている場合は再ログイン |
