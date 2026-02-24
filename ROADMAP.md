# Moneyball Dojo マスターロードマップ

> **ルール**: ✅ = 完了 | 🔄 = 進行中 | ⏳ = 未着手 | 👤 = Taikiの作業

---

## 現在地（2026年2月24日）

```
WBC開幕: 2026年3月5日 → 残り9日
MLBシーズン開幕: 2026年3月27日 → 残り約4.5週間
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
| 2/17 | コンテンツカレンダー＆Xフォローリスト作成 |
| 2/22 | 10,000回モンテカルロ・シーズンシミュレーション完成 |
| 2/22 | 記事05（WS/プレーオフ予想）＆記事06（WBC 2026プレビュー）作成 |
| 2/24 | 記事05, 06の日本語版作成。全12記事完成 |
| 2/24 | マルチバーティカル展開戦略策定（WNBA, Polymarket, eSports, DFS） |
| 2/24 | Beehiiv移行 + Buffer連携の配信ガイド作成 |

---

## Phase 0: 基盤構築 ✅完了

### 👤 Taikiのタスク（これだけやってください）

| # | タスク | 所要時間 | 状態 |
|---|--------|---------|------|
| 1 | moneyballdojo.com ドメイン取得 | 5分 | ⏳ |
| 2 | @MoneyballDojo をX/Twitterで作成 | 5分 | ⏳ |
| 3 | **Beehiiv アカウント作成**（Substack より推奨） | 10分 | ⏳ |
| 4 | note.com アカウント作成 | 5分 | ⏳ |
| 5 | Anthropic APIキー取得 | 5分 | ⏳ |
| 6 | Google Cloud プロジェクト作成 | 15分 | ⏳ |

**合計: 約45分の作業です。**

> **Substack → Beehiiv に変更した理由：**
> - プラットフォーム手数料 0%（Substackは売上の10%）
> - オープンAPI → GitHub Actionsから完全自動投稿が可能
> - コホート分析、A/Bテストなど高度な分析機能
> - 詳細は `DELIVERY_GUIDE.md` を参照

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
| マルチバーティカル市場分析 | ✅ |
| Beehiiv移行 + Buffer連携ガイド | ✅ |
| ウェルカムメール文案（EN + JA） | ✅ |

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
| **モンテカルロ・シーズンシミュレーション** | ✅ 10,000回、全30チーム |

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
| プレシーズン記事6本 × EN + JA = 12ファイル | ✅ |
| 自動化スクリプト run_daily.py | ✅ |
| Twitter投稿テンプレート | ✅ |
| Claude Proダイジェストプロンプト | ✅ |
| X フォローリスト（80+アカウント） | ✅ |
| コンテンツカレンダー（6週間分） | ✅ |

#### 記事一覧（articles/）

| # | タイトル | EN | JA |
|---|---------|----|----|
| 01 | Introducing Moneyball Dojo | ✅ | ✅ |
| 02 | How Our AI Model Works | ✅ | ✅ |
| 03 | 2025 Season Review | ✅ | ✅ |
| 04 | 2026 Division Predictions | ✅ | ✅ |
| 05 | 2026 Championship Predictions (WS/Playoff) | ✅ | ✅ |
| 06 | WBC 2026 Preview | ✅ | ✅ |

---

## Phase 2: Beehiivローンチ 🔄進行中

### 👤 Taikiのタスク

| # | タスク | 状態 |
|---|--------|------|
| 1 | Beehiivアカウント作成＆初期設定（`DELIVERY_GUIDE.md` 参照） | ⏳ |
| 2 | Buffer アカウント作成 + X連携 | ⏳ |
| 3 | 紹介記事（01）をBeehiivに投稿 | ⏳ |
| 4 | 記事02〜06を順次投稿 | ⏳ |
| 5 | note.comに日本語版6本を投稿 | ⏳ |
| 6 | Twitterで初投稿（自己紹介スレッド） | ⏳ |
| 7 | ロゴをBeehiiv/note/Xに設定 | ⏳ |
| 8 | ウェルカムメールをBeehiivに設定 | ⏳ |

### ✅ Claudeのタスク

| # | タスク | 状態 |
|---|--------|------|
| 1 | 配信プラットフォームガイド（DELIVERY_GUIDE.md） | ✅ |
| 2 | バックテスト実績を記事に反映 | ✅（記事04, 05にシミュレーションデータ統合済み） |
| 3 | サンプルダイジェスト（開幕日3/27想定）の品質向上 | ⏳ |
| 4 | メール購読者向けウェルカムメール文案 | ✅ |

---

## Phase 3: WBC + スプリングトレーニング（3月上旬〜中旬）

### WBC期間（3/5〜3/17）

```
WBC期間の特別コンテンツ:
  - 毎日のWBC試合予測（国際大会用調整モデル）
  - 選手パフォーマンス追跡
  - WBC結果振り返り記事（3/17頃）
```

### 日次ルーティン（開幕後）

```
毎朝自動（GitHub Actions）:
  06:00  MLB Stats APIからデータ取得
  06:05  37特徴量ベクトル生成
  06:10  9モデルで全試合予測
  06:15  エッジ計算（モデル確率 vs 暗示オッズ）
  06:20  Daily Digest生成（EN + JA + Twitter）
  06:25  Google Sheetsに予測ログ追記
  06:30  Beehiiv API → 自動パブリッシュ
  06:30  Buffer API → X自動投稿スケジュール

Taikiの作業: ゼロ（完全自動化）

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
- Beehiiv毎日投稿（EN）→ 完全自動
- note.com週1投稿（JA）→ Manus or 手動
- X毎日2-3回投稿 → Buffer経由で自動
- Google Sheetsで実績トラッキング → 完全自動

---

## Phase 5: マルチバーティカル展開（5月〜）

> **戦略:** MLBで構築した「データ取得 → XGBoost → LLM記事生成 → 自動配信」パイプラインを
> 他の市場に水平展開する。詳細は `MULTI_VERTICAL_STRATEGY.md` を参照。

### 展開ロードマップ

| 時期 | バーティカル | おすすめ度 | 理由 |
|------|------------|-----------|------|
| **5月** | **WNBA プレイヤープロップ** | **S** | MLB開幕と同時期。モデル流用が最も容易。市場の非効率性が高い |
| **6月** | **Polymarket/Kalshi アービトラージ** | **A** | 確率出力パイプラインをそのまま再利用。高い課金意欲 |
| **7月** | **eスポーツ（CS2/LoL）** | **A** | 若年層市場。PandaScore APIで無料データ取得可能 |
| **9月** | **DFS最適化ラインナップ** | **B** | NFL/MLB並行でDFS需要が最大化する時期 |

### 各バーティカルの展開方法

```
共通パイプライン（再利用可能）:
  ┌─────────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
  │ データ取得   │ →  │ XGBoost  │ →  │ Claude   │ →  │ Beehiiv  │
  │ (API変更)   │    │ (特徴量  │    │ API      │    │ API      │
  │             │    │  変更)   │    │ (EN+JA)  │    │ (自動)   │
  └─────────────┘    └──────────┘    └──────────┘    └──────────┘

WNBA:  wehoop API → バスケ特徴量 → Claude → Beehiiv
Poly:  Polymarket API → 確率特徴量 → Claude → Beehiiv
eSpo:  PandaScore API → ゲーム特徴量 → Claude → Beehiiv
DFS:   DraftKings + MLB API → 最適化 → Claude → Beehiiv
```

### 有料化スケジュール

| タイミング | アクション |
|-----------|----------|
| 無料購読者500人達成時 | 有料プラン開始（$14.99/月） |
| WNBA追加時 | マルチスポーツバンドル（$19.99/月） |
| 3バーティカル以上 | プレミアムバンドル（$29.99/月） |

---

## 学習済みモデル（9本 → 将来15+本）

```
models/
├── 🔵 MLB（稼働中）
│   ├── model_moneyline.pkl      ← 勝敗予測（メイン）
│   ├── model_over_under.pkl     ← 合計得点 Over/Under
│   ├── model_run_line.pkl       ← -1.5 スプレッド
│   ├── model_f5_moneyline.pkl   ← 前半5イニング
│   ├── model_nrfi.pkl           ← 初回無得点（改善済み 62%）
│   ├── model_pitcher_k.pkl      ← 投手奪三振
│   ├── model_pitcher_outs.pkl   ← 投手アウト数
│   ├── model_batter_props.pkl   ← 打者プロップ
│   └── model_stolen_bases.pkl   ← 盗塁プロップ
│
├── 🟡 WNBA（5月〜開発）
│   ├── model_wnba_spread.pkl    ← スプレッド予測
│   ├── model_wnba_props.pkl     ← プレイヤープロップ
│   └── model_wnba_totals.pkl   ← トータル予測
│
└── 🟠 Polymarket（6月〜開発）
    └── model_event_probability.pkl ← 事象確率予測
```

---

## ファイル一覧

```
moneyball_dojo/
├── 📊 コアシステム
│   ├── run_daily.py                     ✅ メインパイプライン（9モデル統合）
│   ├── train_all_models.py              ✅ 全モデル一括学習
│   ├── season_simulator.py              ✅ モンテカルロ・シミュレーション
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
├── 📝 記事（12ファイル）
│   └── articles/
│       ├── 01_introducing_moneyball_dojo_{EN,JA}.md
│       ├── 02_how_our_ai_model_works_{EN,JA}.md
│       ├── 03_2025_season_review_{EN,JA}.md
│       ├── 04_2026_division_predictions_{EN,JA}.md
│       ├── 05_2026_championship_predictions_{EN,JA}.md
│       └── 06_wbc_2026_preview_{EN,JA}.md
│
├── 📂 出力
│   └── output/
│       ├── backtest_2025/               ✅ バックテスト結果
│       ├── simulation/                  ✅ シーズンシミュレーション結果
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
│   ├── CONTENT_CALENDAR.md              ✅ コンテンツカレンダー
│   ├── X_FOLLOW_LIST.md                 ✅ X フォローリスト
│   ├── DELIVERY_GUIDE.md                ✅ 配信プラットフォームガイド
│   ├── MULTI_VERTICAL_STRATEGY.md       ✅ マルチバーティカル展開戦略
│   ├── ARCHITECTURE.md                  ✅ システムアーキテクチャ
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

### MLBのみ（Phase 2-4）

| 項目 | 月額 | 備考 |
|------|------|------|
| Claude Pro | $20 | 既に契約済み |
| Claude API (Haiku) | $5-15 | 記事生成用 |
| ドメイン | $1 | 年$12を月割り |
| GitHub Actions | $0 | 無料枠で十分 |
| Google Sheets | $0 | 無料 |
| Beehiiv | $0 | 無料プラン（2,500購読者まで） |
| Buffer | $6 | X自動投稿 |
| **合計** | **$32-42/月** | |

### マルチバーティカル展開後（Phase 5）

| 項目 | 月額 | 備考 |
|------|------|------|
| Claude API (追加分) | $10-20 | 複数バーティカルの記事生成 |
| Beehiiv Scale | $0-49 | 購読者数に応じてアップグレード |
| WNBA API | $0 | wehoop（無料） |
| PandaScore | $0 | 無料枠で開始 |
| **追加合計** | **$10-69/月** | |

---

## 🎯 成功指標

| 指標 | 3ヶ月後目標 | 6ヶ月後目標 | 1年後目標 |
|------|-----------|-----------|----------|
| 無料購読者（全バーティカル合計） | 1,000 | 5,000 | 20,000 |
| 有料購読者 | - | 200 | 600 |
| 月収 | $0 | $3,000 | $9,000+ |
| バーティカル数 | 1 (MLB) | 3 (MLB+WNBA+Poly) | 4+ |
| モデル勝率（MLB） | 55%+ | 57%+ | 58%+ |
| X フォロワー | 500 | 2,000 | 5,000 |
