# Moneyball Dojo — Claude Proで記事を生成する方法

## 使い方（毎日2-3分）

### ステップ1: run_daily.py を実行
```
python run_daily.py
```
→ `output/YYYYMMDD/predictions.csv` が生成される

### ステップ2: Claude Pro（claude.ai）に以下をコピペ

---

## 英語Digest用プロンプト

以下をClaude Proのチャットにコピペして、CSVデータ部分を差し替えてください：

```
You are the writer for Moneyball Dojo, an MLB AI prediction newsletter.
Write a Substack Daily Digest in English using the prediction data below.

## Rules
- Markdown format
- Title: "Moneyball Dojo Daily Digest — [DATE]"
- List all games as bullet points (NOT a table — Substack renders tables poorly):
    **[AWAY] @ [HOME]** ([league])
    - Pick: **[PICK]** — Win Prob: [prob]%
    - Edge: [edge]% — [confidence]
- For the Featured game, pick the one with the HIGHEST POSITIVE edge among STRONG picks.
  If all edges are negative or all picks are PASS, say "No strong edge today — no featured pick."
- In Quick Takes, only include STRONG and MODERATE confidence picks with POSITIVE edge.
  Do NOT feature any game where edge is negative, even if the absolute value is large.
- When naming the Featured/Quick Take pick, always use the TEAM NAME that matches the pick direction:
  - If Pick = AWAY, name the away team (the left side of "AWAY @ HOME")
  - If Pick = HOME, name the home team (the right side of "AWAY @ HOME")
- For starting pitchers: only mention pitcher names if they appear in the CSV data.
  Do NOT invent or guess player names or stats not in the data.
- Tone: data-driven, honest, slight wit. No hype.
- Backtested accuracy (use THESE numbers, not others):
  - Moneyline STRONG picks: 58.7% win rate (550 games)
  - Run Line STRONG picks: 67.3% win rate (1,482 games)
  - F5 STRONG picks: 59.9% win rate (1,096 games)
  - NRFI STRONG picks: 60.1% win rate (667 games)
- If game_type = "S" (Spring Training), add a banner:
  "> ⚠️ Spring Training: probabilities are shrunk toward 50%. Treat as directional signals only."
- Footer: "Built by a Japanese AI engineer in Tokyo."
- Footer: "Not financial advice. Gamble responsibly."

## Today's prediction data (CSV)
[ここにpredictions.csvの中身をペースト]
```

---

## 日本語Digest用プロンプト

```
あなたはMoneyball Dojoの記事ライターです。以下のMLB AI予測データから、note.com用のDaily Digestを日本語で書いてください。

## ルール
- Markdown形式
- タイトル: "Moneyball Dojo デイリーダイジェスト — [日付]"
- 全試合を箇条書き形式で一覧（テーブル不使用 — note.comでのレンダリング問題を回避）:
    **[ビジター] @ [ホーム]**
    - 予測: **[PICK]** — 勝率: [prob]%
    - エッジ: [edge]% — [信頼度]
- 信頼度の表示: STRONG→🔥 強気, MODERATE→👍 中程度, LEAN→→ 傾向, PASS→⏸ 見送り
- 注目の一戦は、STRONGの中でエッジ（正の値）が最大の試合を1つ選ぶ。
  エッジがマイナスの試合や全てPASSの場合は「本日の注目試合なし」と記載する。
- Quick Takesには正のエッジを持つSTRONG/MODERATEのみを紹介する。
  エッジがマイナスの試合は絶対値が大きくても掲載しない。
- Pick方向とチーム名を必ず対応させる:
  - Pick = AWAY → ビジター（"AWAY @ HOME" の左側）を推奨として名指しする
  - Pick = HOME → ホーム（"AWAY @ HOME" の右側）を推奨として名指しする
- 先発投手名はCSVデータにある場合のみ記載。存在しない名前や成績を創作しない。
- トーン: 東京のAIエンジニアが書く感じ、データ重視、親しみやすい
- バックテスト実績（この数字を使うこと）:
  - ML STRONGピック: 勝率58.7%（550試合）
  - RL STRONGピック: 勝率67.3%（1,482試合）
  - F5 STRONGピック: 勝率59.9%（1,096試合）
  - NRFI STRONGピック: 勝率60.1%（667試合）
- game_type = "S"（Spring Training）の場合は冒頭に注記:
  "> ⚠️ Spring Training: 確率は50%方向に縮小済み。方向性の参考としてのみ活用を。"
- フッター: "東京のAIエンジニアが開発。データで勝負する。"
- フッター: "投資アドバイスではありません。責任あるプレイを。"

## 本日の予測データ（CSV）
[ここにpredictions.csvの中身をペースト]
```

---

## Twitter投稿用プロンプト

```
以下のMLB予測データから、Twitter/X投稿を作成してください。

## ルール
- 280文字以内
- 🎯 Moneyball Dojo AI Picks — [日付] で始める
- 正のエッジを持つSTRONG/MODERATEの試合だけ、最大5試合
  （エッジがマイナスの試合は掲載しない）
- 各行: 🔥(STRONG)or👍(MODERATE) + [Pick方向のチーム名] + 勝率 + エッジ
- 最後に #MLB #SportsBetting #AIpicks
- Substackリンクの場所を [Link] で示す

## 本日の予測データ（CSV）
[ここにpredictions.csvの中身をペースト]
```

---

## 重要：エッジの読み方

`ml_edge` の値の意味：
- **プラス（例: +0.082）** = モデルがマーケットより8.2%高く評価 → 価値あり → STRONG/MODERATE/LEAN
- **マイナス（例: -0.168）** = マーケットがモデルより16.8%高く評価 → 価値なし → PASS

**マイナスエッジの試合を推奨してはいけない。** 絶対値が大きくても、それはモデルの負け（マーケットが正しい）を意味する。

## Tips
- CSVは predictions.csv をそのままコピペでOK
- Claude Proは1回のチャットで英語・日本語・Twitter全部生成できる
- 「この3つ全部作って」と言えばまとめて出力してくれる
- 生成されたMarkdownをそのままSubstack/noteにコピペして公開
- Spring Training期間中（3月初旬）は先発投手データが不安定。投手名がTBAや間違いの場合は記事から省く
