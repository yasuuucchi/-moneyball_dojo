# Moneyball Dojo マスターロードマップ

> **ルール**: ✅ = 完了 | 🔄 = 進行中 | ⏳ = 未着手 | 👤 = Taikiの作業

---

## 現在地（2026年2月16日）

```
MLBシーズン開幕: 2026年3月27日 → 残り約5.5週間
```

### 完了済みの主要マイルストーン

| 日付 | マイルストーン |
|------|---------------|
| 2/8 | 初回コミット：予測パイプライン、DB設計、Digest生成、GitHub Actions |
| 2/12 | GitHub Actions ワークフロー修正完了 |
| 2/13 | run_daily.py に予測パイプライン統合 |
| 2/15 | 2025年フルシーズン バックテスト（2,426試合） |
| 2/15 | 全モデル 2022-2025 データで再学習 |
| 2/15 | NRFIモデル大改善（53% → 62%、STRONG 69%） |
| 2/15 | トラックレコードページ生成スクリプト追加 |

---

## Phase 0: 基盤構築 ✅完了

### 👤 Taikiのタスク（これだけやってください）

| # | タスク | 所要時間 | 状態 |
|---|--------|---------|------|
| 1 | moneyballdojo.com ドメイン取得 | 5分 | ⏳ |
| 2 | @MoneyballDojo をX/Twitterで作成 | 5分 | ⏳ |
| 3 | moneyballdojo.substack.com 作成 | 10分 | ⏳ |
| 4 | note.com アカウント作成 | 5分 | ⏳ |
| 5 | Anthropic APIキー取得 | 5分 | ⏳ |
| 6 | Google Cloud プロジェクト作成 | 15分 | ⏳ |

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
   - 共有 → サービスアカウントのメールアドレスを追加 → 編集者権限

### ✅ Claudeが完了したタスク

| タスク | 状態 |
|--------|------|
| 戦略ドキュメント作成 | ✅ |
| ChatGPT Proレビュー反映 | ✅ |
| data_pipeline_demo.py（予測パイプライン） | ✅ |
| sheets_schema_v2.py（イベントログDB設計） | ✅ |
| daily_digest_generator.py（日次Digest生成） | ✅ |
| github_actions_workflow.yml（自動実行） | ✅ |
| Google Sheets テンプレート(.xlsx) | ✅ |
| セットアップガイド（日本語） | ✅ |

---

## Phase 1: プレシーズン ✅完了

### ✅ ML基盤

| タスク | 結果 |
|--------|------|
| XGBoostモデル学習（初回） | 52.84%精度, +5.69% ROI |
| **全9モデル再学習（2022-2025データ）** | ✅ 2/15完了 |
| **2025フルシーズンバックテスト** | ✅ 2,426試合で検証済み |
| **NRFIモデル大改善** | ✅ 53%→62%（+9pt）、STRONG 69% |
| **トラックレコード生成スクリプト** | ✅ generate_track_record.py |

### バックテスト結果サマリー（2025年 2,426試合）

| モデル | 精度 | STRONG精度 | AUC-ROC |
|--------|------|-----------|---------|
| **Moneyline** | 64.7% | 72.9% (960/1317) | 0.704 |
| **Run Line** | 66.3% | 73.4% (956/1303) | 0.660 |
| **F5 Moneyline** | 64.6% | 69.5% (1242/1787) | 0.709 |
| **NRFI** | 62.1% | 68.9% (911/1322) | 0.673 |
| **Over/Under** | MAE 3.38 | ライン精度 60-64% | — |

### ✅ コンテンツ

| タスク | 状態 |
|--------|------|
| プレシーズン記事4本 × EN+JA = 8ファイル | ✅ |
| 自動化スクリプト run_daily.py | ✅ |
| Twitter投稿テンプレート | ✅ |
| Claude Proダイジェストプロンプト | ✅ |

#### 記事一覧（articles/）

| # | タイトル | EN | JA |
|---|---------|----|----|
| 01 | Introducing Moneyball Dojo | ✅ | ✅ |
| 02 | How Our AI Model Works | ✅ | ✅ |
| 03 | 2025 Season Review | ✅ | ✅ |
| 04 | 2026 Division Predictions | ✅ | ✅ |

---

## Phase 2: Substackローンチ 🔄進行中

### 👤 Taikiのタスク

| # | タスク | 状態 |
|---|--------|------|
| 1 | Substackアカウント作成＆初期設定 | ⏳ |
| 2 | 紹介記事（01）をSubstackに投稿 | ⏳ |
| 3 | 技術解説記事（02）を投稿 | ⏳ |
| 4 | 2025レビュー記事（03）を投稿 | ⏳ |
| 5 | 2026予測記事（04）を投稿 | ⏳ |
| 6 | note.comに日本語版4本を投稿 | ⏳ |
| 7 | Twitterで初投稿（自己紹介スレッド） | ⏳ |
| 8 | ロゴをSubstack/note/Xに設定 | ⏳ |

### 🔄 Claudeのタスク

| # | タスク | 状態 |
|---|--------|------|
| 1 | Substackランディングコンテンツ最適化 | ⏳ |
| 2 | バックテスト実績を記事に反映 | ⏳ |
| 3 | サンプルダイジェスト（開幕日3/27想定）の品質向上 | ⏳ |
| 4 | メール購読者向けウェルカムメール文案 | ⏳ |

---

## Phase 3: スプリングトレーニング（3月上旬〜中旬）

### 日次ルーティン

```
毎朝自動（GitHub Actions）:
  06:00  MLB Stats APIからデータ取得
  06:05  37特徴量ベクトル生成
  06:10  9モデルで全試合予測
  06:15  エッジ計算（モデル確率 vs 暗示オッズ）
  06:20  Daily Digest生成（EN + JA + Twitter）
  06:25  Google Sheetsに予測ログ追記
  06:30  完成したMarkdownをTaikiに通知

Taikiの作業（90秒）:
  → SubstackにMarkdownをコピペ → 投稿ボタン
  → Twitterに本日のトップピックを投稿

夜（自動）:
  結果記録 → Sheets更新 → ROI/勝率再計算
```

### Claudeのタスク

| # | タスク | 状態 |
|---|--------|------|
| 1 | スプリングトレーニング予測データ生成 | ⏳ |
| 2 | モデル精度調整（リアルタイム結果で補正） | ⏳ |
| 3 | パフォーマンスダッシュボード作成 | ⏳ |

---

## Phase 4: MLB開幕（3月27日〜）

- 毎日9モデルでフル予測
- Substack毎日投稿（EN）
- note.com週1投稿（JA）
- Twitter毎日2-3回投稿
- Google Sheetsで実績トラッキング

---

## Phase 5: 有料化 & 拡張（5月〜）

| タイミング | アクション |
|-----------|----------|
| 無料購読者500人達成時 | 有料プラン開始（$12.99/月） |
| 6月頃 | NFLモデル開発開始 |
| 9月 | NFLシーズン開幕 → デュアルスポーツ化 |

---

## 学習済みモデル（9本）

```
models/
├── model_moneyline.pkl      ← 勝敗予測（メイン）
├── model_over_under.pkl     ← 合計得点 Over/Under
├── model_run_line.pkl       ← -1.5 スプレッド
├── model_f5_moneyline.pkl   ← 前半5イニング
├── model_nrfi.pkl           ← 初回無得点（改善済み 62%）
├── model_pitcher_k.pkl      ← 投手奪三振
├── model_pitcher_outs.pkl   ← 投手アウト数
├── model_batter_props.pkl   ← 打者プロップ
└── model_stolen_bases.pkl   ← 盗塁プロップ
```

---

## ファイル一覧

```
moneyball_dojo/
├── 📊 コアシステム
│   ├── run_daily.py                     ✅ メインパイプライン（9モデル統合）
│   ├── train_all_models.py              ✅ 全モデル一括学習
│   ├── backtest_2025.py                 ✅ 2025年バックテスト
│   ├── generate_track_record.py         ✅ トラックレコード生成
│   ├── data_pipeline_demo.py            ✅ 予測パイプラインデモ
│   ├── daily_digest_generator.py        ✅ Digest生成（英語+日本語）
│   ├── article_generator_template.py    ✅ Claude API記事テンプレート
│   └── sheets_schema_v2.py             ✅ DB設計（イベントログ方式）
│
├── 🧠 モデル学習
│   ├── train_model.py                   ✅ 単体モデル学習（v1）
│   ├── train_model_v2.py                ✅ 単体モデル学習（v2）
│   └── train_all_models.py              ✅ 全9モデル一括学習
│
├── 📡 データ取得
│   ├── fetch_real_data.py               ✅ MLB Stats API
│   ├── fetch_player_data.py             ✅ 選手データ
│   ├── fetch_nrfi_data.py               ✅ NRFI用データ
│   └── fetch_game_props_data.py         ✅ プロップデータ
│
├── 📝 記事（8ファイル）
│   └── articles/
│       ├── 01_introducing_moneyball_dojo_{EN,JA}.md
│       ├── 02_how_our_ai_model_works_{EN,JA}.md
│       ├── 03_2025_season_review_{EN,JA}.md
│       └── 04_2026_division_predictions_{EN,JA}.md
│
├── 📂 出力
│   └── output/
│       ├── backtest_2025/               ✅ バックテスト結果
│       ├── 20260327/                    ✅ サンプルダイジェスト
│       └── daily_predictions.csv        ✅ 予測CSV
│
├── ⚙️ インフラ
│   ├── .github/workflows/               ✅ GitHub Actions
│   ├── requirements.txt                 ✅ 依存パッケージ
│   └── .gitignore                       ✅
│
├── 📋 ドキュメント
│   ├── ROADMAP.md                       ✅ このファイル
│   ├── README.md                        ✅ プロジェクト概要
│   ├── README_TRAIN_MODEL.md            ✅ モデル学習手順
│   ├── SETUP_GUIDE.md                   ✅ セットアップ手順
│   ├── QUICKSTART.md                    ✅ クイックスタート
│   ├── claude_pro_digest_prompt.md      ✅ Claude Pro用プロンプト
│   └── claude_pro_project_instructions.md ✅ Claude Pro設定
│
└── 🗄️ データ
    ├── data/                            ✅ 学習データ（CSV）
    ├── models/                          ✅ 学習済みモデル（9本）
    └── Moneyball_Dojo_Database_Template.xlsx ✅ Sheetsテンプレート
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
