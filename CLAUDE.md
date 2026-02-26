# CLAUDE.md — Moneyball Dojo (MLB事業)

> **このファイルはAIエージェントへのオリエンテーション文書。**
> 新しいセッションを開始したら、まずここを読む。

---

## あなたは誰か

あなたは **Dojo Labs** の技術チーム（CTO部門）。
Dojo Labs は東京在住の個人が運営するAI予測事業グループ。
このリポジトリは Dojo Labs の **最初の事業: Moneyball Dojo（MLB予測）** のコードベース。

## 組織構造（Hub & Spoke モデル）

```
Dojo Labs (親ブランド)
├── Moneyball Dojo  ← このリポ。MLB予測。最優先事業。
├── Poly Dojo       ← Polymarket予測。2番目の事業。別リポ（未作成）。
├── WNBA Dojo       ← WNBA予測。3番目。5月開幕に合わせて開発。
└── eSports Dojo    ← eスポーツ予測。4番目。後回し。
```

**ブランド戦略:**
- 各事業は独立したアカウント（Substack/X/note）を持つ
- 「Dojo」が共通ブランド。クロスプロモは親アカウント経由
- 日本語圏はHub（Dojo Labs）で統合配信

## このリポの状態

**完成度: 95%。コード作業は基本的に完了。**

### 完成済み
- XGBoost 9モデル全て訓練済み（2022-2025データ）
- 2025年バックテスト完了（2,426試合）
  - ⚠ 旧精度（64.7%/72.9%）はrolling statsデータリークを含む水増し値
  - リーク修正+キャリブレーション後: ML 57.8% (test split), 要backtest再実行
- 10,000回モンテカルロ・シーズンシミュレーション
- 記事12本（6本 x EN+JA）: 01-06
- GitHub Actions 3ワークフロー（予測/結果更新/モデル再学習）
- run_daily.py: データ取得→予測→記事生成→Sheets書込の完全パイプライン
- 配信ガイド（DELIVERY_GUIDE.md）、ウェルカムメール、マルチバーティカル戦略書

### 完了済み（ユーザーの手動作業）
- [x] Substackに記事を投稿済み
- [x] X (@MoneyballDojo) で自己紹介スレッド投稿済み

### 未完了（ユーザーの手動作業）
- [ ] note.comに日本語版を投稿（6本）
- [ ] GitHub Secrets 設定（CLAUDE_API_KEY, GOOGLE_SHEETS_*）
- [ ] Spring Training 予測配信の開始

### 未完了（次のコード作業）
- [ ] Beehiiv API連携（DELIVERY_GUIDE.md参照。Substackの代替として推奨）
- [ ] Buffer API連携（X自動投稿）
- [ ] n8n ワークフロー構築（Phone CEO アーキテクチャ）
- [ ] 記事07: WBC結果振り返り（3/17以降に作成）
- [ ] 記事08: Opening Day全カード予測（3/20頃に作成）

## Phone CEO アーキテクチャ（目標）

ユーザー（CEO）はスマホからSlack/Telegramで操作するだけ。
PCを開かなくても全事業が回る状態を目指す。

```
スマホ (Slack/Telegram)
  ↕ コマンド & 通知
n8n (Hetzner VPS, $7/月)
  ↕ ワークフロー実行
GitHub Actions + Claude API + Beehiiv API + Google Sheets
```

## 技術スタック

- **ML**: XGBoost, scikit-learn, pandas
- **データ**: MLB Stats API (statsapi.mlb.com)
- **記事生成**: Claude API (Haiku)
- **配信**: Substack（暫定） → Beehiiv（推奨移行先）
- **X投稿**: Buffer ($6/月)
- **DB**: Google Sheets (sheets_schema_v2.py)
- **CI/CD**: GitHub Actions
- **自動化**: n8n（未導入、計画中）

## 重要な設計判断の記録

1. **Substack vs Beehiiv**: Beehiivを推奨（手数料0%、API柔軟）。ただしSubstackは既に作成済み。移行は購読者が増えてからでもOK。
2. **Manus AI**: 現時点では不要。n8n + cron + Claude API で十分。月収$10,000超えたら再検討。
3. **事業間の関係**: 「競争」ではなく「比較」。各事業が独立P&Lを持ち、週次で横比較レポートを生成。
4. **最速マネタイズ**: Polymarket（通年、今日から可能） > MLB（3/27開幕） > WNBA（5月） > eSports（後回し）

## ファイル構成の要点

```
重要ファイル:
  run_daily.py              ← メインパイプライン。これが全てを動かす
  train_all_models.py       ← 9モデル一括学習
  season_simulator.py       ← モンテカルロ・シミュレーション
  daily_digest_generator.py ← Digest生成（EN + JA）

ドキュメント:
  ROADMAP.md                ← マスターロードマップ（フェーズ別計画）
  CONTENT_CALENDAR.md       ← 投稿スケジュール
  DELIVERY_GUIDE.md         ← Beehiiv + Buffer セットアップ手順
  MULTI_VERTICAL_STRATEGY.md ← 4事業の展開計画（WNBA, Poly, eSports, DFS）
  WELCOME_EMAIL.md          ← 新規購読者向けメール（EN + JA）
  ARCHITECTURE.md           ← システム全体像（mermaid図）
  X_FOLLOW_LIST.md          ← X フォロー戦略（80+アカウント）

記事 (articles/):
  01-06 の EN + JA = 12ファイル

モデル (models/):
  9本の .pkl ファイル（全て2022-2025データで学習済み）
```

## コミュニケーションルール

- ユーザーは日本語で話す。日本語で返す。
- コードのコメントは英語。
- 記事は EN + JA の両方を必ず作る。
- 進捗報告はユーザーからAIにではなく、AIがユーザーに報告する。
- 計画より実行を優先。「出しながら直す」。

## 今のフェーズ

**Phase 2: ローンチ準備** → 開幕(3/27)までに配信体制を確立する。

最優先: 記事をSubstackとnote.comに投稿して、最初の読者を獲得すること。
コード作業より配信が先。
