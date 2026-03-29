# Audience & Offer — Moneyball Dojo

## Primary Audience (EN)

### ペルソナ

1. **Data-Curious MLB Fan**
   - MLBを観ているが、セイバーメトリクスは表面的にしか知らない
   - FanGraphsを見たことはあるが、使いこなせていない
   - 「データで野球を見る」ことに興味がある
   - 求めていること: わかりやすいデータ分析、教育的コンテンツ

2. **Transparent Picks Seeker**
   - スポーツベッティングに参加しているが、AI picksアカウントの不透明さに不信感
   - "80% win rate"の主張を信じていない
   - 検証可能なトラックレコードを求めている
   - 求めていること: 事前ログ、事後監査、正直な成績報告

3. **Sabermetrics Enthusiast**
   - Bill James, Tom Tangoの著作を読んでいる層
   - 自分でもデータ分析をしたい/している
   - モデルの中身に興味がある
   - 求めていること: 技術的な深掘り、モデルアーキテクチャの説明

4. **AI/ML Practitioner**
   - MLエンジニア、データサイエンティスト
   - スポーツ予測をケーススタディとして見ている
   - 求めていること: 特徴量設計、モデル選択、キャリブレーションの実践例

### 共通点

全ペルソナに共通するのは「**透明性への渇望**」。
既存のサービスが見せないものを見たい人たち。

---

## Primary Audience (JA)

### ペルソナ

1. **大谷ファン / 日本人選手ファン**
   - MLBを大谷翔平経由で見始めた
   - データ分析の視点は新鮮に感じる
   - 求めていること: 日本人選手の試合予測、わかりやすい解説

2. **データ好きの野球ファン**
   - NPBのデータは見ているが、MLBのセイバーメトリクスは英語の壁がある
   - wOBA, FIPなどの概念を日本語で学びたい
   - 求めていること: セイバーメトリクスの日本語解説、MLBデータの読み方

3. **AIと野球の交差点に興味がある人**
   - テック系のバックグラウンド
   - 「AIで予測」というコンセプトに興味
   - 求めていること: 技術記事、モデルの仕組み

### 日本語市場の特性

- MLB予測をデータドリブンで日本語配信するサービスはほぼ皆無
- 競合ゼロのニッチ
- 大谷翔平効果で日本のMLB関心は過去最高水準
- note.comはテック系・分析系コンテンツとの親和性が高い

---

## Value Proposition

### EN

> "The only MLB prediction service that logs every pick before the game and audits every result after."

### JA

> 「全予測を試合前にログし、全結果を試合後に監査する、唯一のMLB予測サービス」

### 展開

| 要素 | 内容 |
|------|------|
| **What** | MLB全マーケットのAI予測（ML, RL, F5, NRFI, Totals, etc.） |
| **How** | XGBoostモデル + セイバーメトリクス特徴量 + 週次改善 |
| **Why different** | 事前ログ + 事後監査 + 全プロセス公開 |
| **For whom** | データに興味があるMLBファン、透明性を求めるベッター |

---

## Reader Journey

### Stage 1: Discovery（発見）

```
X投稿を見る → "おっ、予測が事前にログされてる"
```

- X投稿にGoogle Sheetsリンクを添付
- 「logged before first pitch」のフレーズで透明性を示す
- リプライで結果を報告（当たりも外れも）

### Stage 2: Exploration（探索）

```
Substackに来る → 無料記事を読む
```

- Weekly Audit: "何がズレたか"の分析に惹かれる
- モデル解説記事: "こんなに中身を見せるサービスは珍しい"
- 過去のトラックレコードをSheetsで検証できる

### Stage 3: Subscription（購読）

```
メール登録 → 毎週のAuditが届く
```

- 登録のトリガー: 「来週のAuditも読みたい」
- Welcome Email で期待値を設定（WELCOME_EMAIL.md参照）
- 初期は全て無料。信頼構築フェーズ

### Stage 4: Trust Building（信頼構築）

```
毎週のAuditを受け取る → モデルの改善を見守る
```

- 継続的に正直な結果報告を受け取る
- モデルが改善していく過程を一緒に体験
- 「このサービスは信頼できる」という確信が育つ

### Stage 5: Conversion（将来）

```
有料コンテンツへの移行
```

- 前提: Stage 4が十分に成熟してから
- 有料の価値: "raw data"ではなく"意思決定の圧縮"
- 無料で十分価値を感じた人が「もっと」を求める構造

---

## Acquisition Channels

### Primary

| チャネル | 役割 | KPI |
|---------|------|-----|
| **X (@MoneyballDojo)** | 発見の入口。Daily signals配信 | フォロワー数、エンゲージメント率 |
| **Substack** | 信頼構築の場。Weekly Audit配信 | 購読者数、開封率 |

### Secondary

| チャネル | 役割 | KPI |
|---------|------|-----|
| **note.com** | 日本語市場の開拓 | PV、スキ数、フォロワー数 |
| **Reddit (r/baseball, r/sportsbook)** | コミュニティでの信頼獲得 | 投稿へのupvote、コメント |
| **GitHub** | 技術者コミュニティへのリーチ | Stars、Forks |

### Cross-Promotion

- Dojo Labs傘下の他事業（Poly Dojo, WNBA Dojoなど）からのクロスプロモ
- 「同じチームが他のスポーツ/マーケットも予測している」というストーリー
