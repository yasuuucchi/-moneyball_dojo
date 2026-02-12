# Moneyball Dojo マスターロードマップ

> **ルール**: 🤖 = Claudeが実行済み/実行可能 | 👤 = Taikiがやる必要あり | ⏳ = 未着手

---

## 現在地（2026年2月8日）

```
MLBシーズン開幕: 2026年3月27日 → 残り約7週間
```

---

## Phase 0: 基盤構築（今週中）

### 👤 Taikiのタスク（これだけやってください）

| # | タスク | 所要時間 | 手順 |
|---|--------|---------|------|
| 1 | moneyballdojo.com ドメイン取得 | 5分 | Namecheap.comで検索→購入（年$10-15） |
| 2 | @MoneyballDojo をX/Twitterで作成 | 5分 | twitter.com/signup → ハンドル入力 |
| 3 | moneyballdojo.substack.com 作成 | 10分 | substack.com → Start Writing → URL設定 |
| 4 | note.com アカウント作成 | 5分 | note.com → 新規登録 |
| 5 | Anthropic APIキー取得 | 5分 | console.anthropic.com → APIキー作成 |
| 6 | Google Cloud プロジェクト作成 | 15分 | 下記の詳細手順を参照 |

**合計: 約45分の作業です。**

#### Google Cloud 詳細手順:
1. https://console.cloud.google.com にアクセス
2. 新しいプロジェクト作成 → 名前: "Moneyball Dojo"
3. 左メニュー → APIとサービス → ライブラリ → "Google Sheets API" を検索 → 有効にする
4. 左メニュー → IAMと管理 → サービスアカウント → 作成
   - 名前: moneyball-dojo-bot
   - 役割: 編集者
5. 作成したアカウントをクリック → キー → 新しいキーを追加 → JSON → ダウンロード
6. Google Sheets で新しいスプレッドシートを作成
   - 名前: "Moneyball Dojo - Predictions 2026"
   - 共有 → サービスアカウントのメールアドレス（xxx@xxx.iam.gserviceaccount.com）を追加 → 編集者権限

### 🤖 Claudeが実行済みのタスク

| # | タスク | 状態 |
|---|--------|------|
| ✅ | 戦略ドキュメント作成 | 完了 |
| ✅ | ChatGPT Proレビュー反映 | 完了 |
| ✅ | data_pipeline_demo.py（予測パイプライン） | 動作確認済み |
| ✅ | sheets_schema_v2.py（イベントログDB設計） | 動作確認済み |
| ✅ | daily_digest_generator.py（日次Digest生成） | 動作確認済み |
| ✅ | github_actions_workflow.yml | 作成済み |
| ✅ | Google Sheets テンプレート(.xlsx) | 作成済み → Sheetsにインポート可能 |
| ✅ | 第1回記事ドラフト（英語/日本語） | 作成済み |
| ✅ | セットアップガイド（日本語） | 作成済み |

---

## Phase 1: プレシーズン（2月中旬〜3月上旬）

### 👤 Taikiのタスク

| # | タスク | タイミング |
|---|--------|-----------|
| 1 | 第1回記事をSubstackに投稿 | Phase 0完了後すぐ |
| 2 | 第1回記事をnote.comに投稿 | 同上 |
| 3 | Twitterで初投稿（自己紹介スレッド） | 同上 |
| 4 | ロゴをSubstack/note/Xに設定 | 記事投稿と同時 |
| 5 | Google SheetsにテンプレートXLSXをインポート | Phase 0完了後 |

### 🤖 Claudeが実行するタスク

| # | タスク | 内容 | 状態 |
|---|--------|------|------|
| ✅ | 実データでMLモデル学習 | XGBoost学習済み (52.84%精度, +5.69% ROI) | 完了 |
| ✅ | プレシーズン記事 4本作成 | 全8ファイル（EN+JA各4本） | 完了 |
| ✅ | 自動化スクリプト完成版 | run_daily.py 動作確認済み | 完了 |
| ✅ | Twitter投稿テンプレート作成 | run_daily.pyに内蔵 | 完了 |

---

## Phase 2: スプリングトレーニング（3月上旬〜中旬）

### 👤 Taikiのタスク

| # | タスク |
|---|--------|
| 1 | 毎日生成されたDigestをSubstackにコピペ投稿（90秒） |
| 2 | 週1回note.comに日本語版を投稿 |
| 3 | Twitterで毎日1-2回投稿（テンプレート使用） |

### 🤖 Claudeが実行するタスク

| # | タスク |
|---|--------|
| 1 | スプリングトレーニングの予測データ生成 |
| 2 | モデルの精度調整（バックテスト結果を元に） |
| 3 | パフォーマンスダッシュボード作成 |

---

## Phase 3: MLB開幕（3月27日〜）

### 日次ルーティン

```
毎朝自動（GitHub Actions / Claudeが実行）:
  06:00  データ取得（pybaseball）
  06:05  特徴量エンジニアリング
  06:10  XGBoostで全試合予測
  06:15  エッジ計算（モデル確率 vs マーケットオッズ）
  06:20  Daily Digest Markdown生成
  06:25  Google Sheetsに予測ログ追記
  06:30  完成したMarkdownをTaikiに通知

Taikiの作業（90秒）:
  → SubstackにMarkdownをコピペ → 投稿ボタン
  → Twitterに本日のトップピックを投稿

夜（自動）:
  結果記録 → Sheets更新 → ROI/勝率再計算
```

---

## Phase 4: 有料化 & NFL拡張（4月〜）

| タイミング | アクション |
|-----------|----------|
| 無料購読者500人達成時 | 有料プラン開始（$12.99/月） |
| 6月頃 | NFLモデル開発開始 |
| 9月 | NFLシーズン開幕 → デュアルスポーツ化 |

---

## ファイル一覧（現在の成果物）

```
moneyball_dojo/
├── 📊 コアシステム
│   ├── data_pipeline_demo.py          ✅ 予測パイプライン
│   ├── sheets_schema_v2.py            ✅ DB設計（イベントログ方式）
│   ├── daily_digest_generator.py      ✅ Digest生成（英語+日本語）
│   └── article_generator_template.py  ✅ Claude API記事テンプレート
│
├── 📝 記事
│   ├── articles/01_introducing_moneyball_dojo_EN.md  ✅ 第1回（英語）
│   └── articles/01_introducing_moneyball_dojo_JA.md  ✅ 第1回（日本語）
│
├── 🗄️ データベース
│   └── Moneyball_Dojo_Database_Template.xlsx  ✅ Sheets用テンプレート
│
├── ⚙️ インフラ
│   ├── github_actions_workflow.yml     ✅ 自動実行ワークフロー
│   └── requirements.txt               ✅ 依存パッケージ
│
├── 📋 ドキュメント
│   ├── SETUP_GUIDE.md                 ✅ セットアップ手順（日本語）
│   └── ROADMAP.md                     ✅ このファイル
│
└── 📂 出力
    └── output/daily_predictions.csv   ✅ サンプル予測出力
```

---

## 💰 コスト見積もり

| 項目 | 月額 | 備考 |
|------|------|------|
| Claude Pro | $20 | 既に契約済み |
| Claude API (Haiku Batch) | $5-15 | 記事生成用 |
| ドメイン | $1 | 年$12を月割り |
| GitHub Actions | $0 | 無料枠で十分 |
| Google Sheets | $0 | 無料 |
| Substack | $0 | プラットフォーム無料 |
| **合計** | **$26-36/月** | |

---

## 🎯 成功指標

| 指標 | 3ヶ月後目標 | 6ヶ月後目標 | 1年後目標 |
|------|-----------|-----------|----------|
| 無料購読者 | 1,000 | 5,000 | 15,000 |
| 有料購読者 | - | 150 | 450 |
| 月収 | $0 | $1,950 | $5,850 |
| モデル勝率 | 55%+ | 57%+ | 58%+ |
| X フォロワー | 500 | 2,000 | 5,000 |
