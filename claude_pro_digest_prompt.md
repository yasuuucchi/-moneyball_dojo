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
あなたはMoneyball Dojoの記事ライターです。以下のMLB AI予測データから、Substack用のDaily Digestを英語で書いてください。

## ルール
- Markdown形式
- タイトル: "Moneyball Dojo Daily Digest — [日付]"
- 全試合の表（Matchup / Pick / Win Prob / Edge / Confidence）
- エッジ最大の試合を1つ「Featured」として深掘り（3-4段落）
- STRONG/MODERATEの試合を「Quick Takes」で紹介
- トーン: データ重視、正直、少しウィットを効かせる
- フッターに "Built by a Japanese AI engineer in Tokyo" を入れる
- "Not financial advice. Gamble responsibly." を入れる

## 本日の予測データ（CSV）
[ここにpredictions.csvの中身をペースト]
```

---

## 日本語Digest用プロンプト

```
あなたはMoneyball Dojoの記事ライターです。以下のMLB AI予測データから、note.com用のDaily Digestを日本語で書いてください。

## ルール
- Markdown形式
- タイトル: "Moneyball Dojo デイリーダイジェスト — [日付]"
- 全試合の表（対戦 / 予測 / 勝率 / エッジ / 信頼度）
- 信頼度の日本語: STRONG→🔥 強気, MODERATE→👍 中程度, LEAN→→ 傾向, PASS→⏸ 見送り
- エッジ最大の試合を「注目の一戦」として深掘り（3-4段落）
- トーン: 東京のAIエンジニアが書く感じ、データ重視、親しみやすい
- フッターに "東京のAIエンジニアが開発。データで勝負する。" を入れる
- "投資アドバイスではありません。責任あるプレイを。" を入れる

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
- STRONG/MODERATEの試合だけ、最大5試合
- 各行: 🔥(STRONG)or👍(MODERATE) + 対戦 + Pick + 勝率 + エッジ
- 最後に #MLB #SportsBetting #AIpicks
- Substackリンクの場所を [Link] で示す

## 本日の予測データ（CSV）
[ここにpredictions.csvの中身をペースト]
```

---

## Tips
- CSVは predictions.csv をそのままコピペでOK
- Claude Proは1回のチャットで英語・日本語・Twitter全部生成できる
- 「この3つ全部作って」と言えばまとめて出力してくれる
- 生成されたMarkdownをそのままSubstack/noteにコピペして公開
