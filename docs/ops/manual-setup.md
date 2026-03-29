# Manual Setup

> CEOが手動でセットアップする必要がある項目の一覧。
> AIチームでは実行できない、認証・外部サービス設定をまとめる。

---

## 1. GitHub Secrets

リポジトリの Settings → Secrets and variables → Actions で以下を設定:

| Secret Name | 用途 | 取得方法 |
|---|---|---|
| `CLAUDE_API_KEY` | Claude API（記事生成） | [Anthropic Console](https://console.anthropic.com/) → API Keys |
| `GOOGLE_SHEETS_CREDENTIALS` | Google Sheets API認証（JSON） | GCP Console → Service Account → JSON キー |
| `GOOGLE_SHEETS_ID` | Google Sheetsのスプレッドシート ID | シートURLの `/d/` と `/edit` の間の文字列 |
| `SLACK_WEBHOOK_URL` | Slack通知 | Slack App設定 → Incoming Webhooks → Webhook URL |

### GOOGLE_SHEETS_CREDENTIALS の設定手順

1. [Google Cloud Console](https://console.cloud.google.com/) にアクセス
2. プロジェクトを作成（または既存を使用）
3. Google Sheets API を有効化
4. サービスアカウントを作成
5. JSON キーをダウンロード
6. JSON の内容全体を GitHub Secret の値として貼り付け
7. Google Sheets でそのサービスアカウントのメールアドレスを「編集者」として共有

---

## 2. Claude Desktop Cowork Scheduled Tasks

Claude Desktop のスケジュールタスク機能で以下を設定。
詳細は `docs/ops/scheduled-tasks/` ディレクトリを参照。

| Task Name | Schedule | 内容 |
|---|---|---|
| Morning Brief | 毎朝 (Desktop起動時) | 予測サマリー確認、公開判断 |
| Publish Triage | Morning Brief後 | 公開モード判定、X投稿準備 |
| Weekly Audit | 毎週月曜 | 週間精度レビュー、Substack記事生成 |
| Anthropic Radar | 毎週日曜 | Anthropic新機能チェック |

**注意**: これらは Desktop が起動中のときのみ実行される。
日次予測パイプライン（Layer 1）には影響しない。

---

## 3. note.com Account

### 初期セットアップ

1. [note.com](https://note.com/) でアカウント作成
2. プロフィール設定:
   - 名前: Moneyball Dojo
   - 自己紹介: AIとセイバーメトリクスでMLBを予測する（CLAUDE.mdのブランドメッセージ参照）
   - アイコン: X と統一
3. 初回記事の投稿（`articles/` ディレクトリの JA 版 01-06）

### 投稿ポリシー

- EN記事の日次ミラーではない（`publishing-policy.md` 参照）
- 週1-2本、JA独自価値のコンテンツ
- 日本人選手の分析を積極的に取り入れる

---

## 4. Google Sheets Database

### テンプレートからの作成

1. 新しい Google Spreadsheet を作成
2. `sheets_schema_v2.py` に定義されたスキーマに従ってシートを作成:
   - `Predictions`: 日次予測データ
   - `Results`: 試合結果
   - `Accuracy`: モデル精度追跡
   - `Daily_Digest`: ダイジェスト生成用
3. スプレッドシートIDをメモ（GitHub Secretに設定）
4. サービスアカウントのメールアドレスを「編集者」として共有

### シート構造

詳細は `sheets_schema_v2.py` を参照。主要カラム:

```
Predictions: date, game_id, away, home, model, prediction, confidence, tier, timestamp
Results:     date, game_id, away, home, away_score, home_score, winner
Accuracy:    date, model, total, correct, accuracy, strong_total, strong_correct, strong_accuracy
```

---

## 5. Slack Workspace + Webhook

### セットアップ手順

1. [Slack](https://slack.com/) でワークスペースを作成（または既存を使用）
2. チャンネルを作成:
   - `#moneyball-predictions` — 日次予測通知
   - `#moneyball-alerts` — エラー・障害通知
   - `#moneyball-weekly` — 週次レポート
3. Slack App を作成:
   - [api.slack.com/apps](https://api.slack.com/apps) → Create New App
   - Incoming Webhooks を有効化
   - `#moneyball-predictions` チャンネルにWebhook URLを生成
4. Webhook URL を GitHub Secret (`SLACK_WEBHOOK_URL`) に設定

### 通知フォーマット

```json
{
  "channel": "#moneyball-predictions",
  "text": "Today's Predictions (2026-03-29): 15 games, 4 STRONG picks",
  "attachments": [
    {
      "color": "#36a64f",
      "title": "STRONG Picks",
      "text": "NYY over BOS (72%), LAD over SF (69%)"
    }
  ]
}
```

---

## Setup Checklist

完了したらチェック:

- [ ] GitHub Secrets: `CLAUDE_API_KEY`
- [ ] GitHub Secrets: `GOOGLE_SHEETS_CREDENTIALS`
- [ ] GitHub Secrets: `GOOGLE_SHEETS_ID`
- [ ] GitHub Secrets: `SLACK_WEBHOOK_URL`
- [ ] Claude Desktop Cowork タスク設定
- [ ] note.com アカウント作成 + プロフィール設定
- [ ] note.com 初回記事投稿（JA版 01-06）
- [ ] Google Sheets スプレッドシート作成
- [ ] Google Sheets サービスアカウント共有
- [ ] Slack ワークスペース + チャンネル作成
- [ ] Slack Incoming Webhook 設定
