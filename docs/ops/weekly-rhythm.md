# Weekly Rhythm

> Moneyball Dojo の週間運用リズム。
> 曜日ごとの定型タスクと、自動/手動の区分を定義する。

---

## Weekly Overview

| Day | Automatic (Layer 1) | Agentic (Layer 2) | CEO Action |
|---|---|---|---|
| **Mon** | 予測 + 結果 + モデル再学習 | Weekly Review生成, Weekly Audit記事作成 | Weekly Audit確認 |
| **Tue** | 予測 + 結果 | Publish Triage, Weekly Audit公開 | Substack公開確認 |
| **Wed** | 予測 + 結果 | Publish Triage | — |
| **Thu** | 予測 + 結果 | Publish Triage | — |
| **Fri** | 予測 + 結果 | Publish Triage, Weekend Preview準備 | — |
| **Sat** | 予測 + 結果 | Lighter Triage | — |
| **Sun** | 予測 + 結果 | Lighter Triage, Anthropic Radar | Radar確認（任意） |

---

## Monday: 週の始まり

### Automatic (Layer 1)
- `00:00 UTC` — モデル再学習 (`weekly-retrain.yml`)
  - 直近の全データで9モデルを再学習
  - 新しい `.pkl` ファイルを git push
- `08:00 UTC` — 日次予測（通常通り）
- `15:00 UTC` — 結果更新（通常通り）

### Agentic (Layer 2)
- **Weekly Review生成**
  - 先週の全予測の精度集計
  - モデル別・ティア別の的中率
  - 特筆すべきトレンドの抽出
- **Weekly Audit記事作成**
  - Substack向けフラッグシップ記事の下書き
  - 精度データ + 分析 + 来週の展望

### CEO
- Weekly Review をSlackで確認
- Weekly Audit の下書きを確認（火曜の公開前）

---

## Tuesday: Weekly Audit 公開日

### Agentic (Layer 2)
- **Weekly Audit公開**
  - Substackに記事を公開
  - X で告知スレッドを投稿
  - note.com 向けJA版の準備（必要に応じて）
- 通常の Publish Triage

### CEO
- Substack公開を確認
- 読者反応をモニタリング（任意）

---

## Wednesday - Thursday: 通常運用

### Agentic (Layer 2)
- Morning Brief → Publish Triage → X投稿
- 特別なタスクなし。安定運用。

---

## Friday: 週末準備

### Agentic (Layer 2)
- 通常の Publish Triage
- **Weekend Preview準備**
  - 土日の注目カード整理
  - 週末向けコンテンツの下書き

---

## Saturday - Sunday: ライトモード

### Agentic (Layer 2)
- **Lighter Content**: 深い分析記事は出さない
  - Weekend Preview の配信
  - 試合結果のレキャップのみ
- **Sunday: Anthropic Radar**
  - 週次のAnthropic新機能チェック
  - 運用改善提案の作成（`docs/ops/anthropic-feature-watch.md` 参照）

---

## Flagship Content: Weekly Audit

**Moneyball Dojo の看板コンテンツ。毎週月曜/火曜にSubstackで公開。**

### 構成

1. **Week in Review**: 先週の予測精度サマリー
2. **Model Performance**: モデル別の詳細分析
3. **Tier Effectiveness**: STRONG / MODERATE / LEAN の的中率比較
4. **Notable Calls**: 的中した好予測 & 外れた予測の振り返り
5. **Week Ahead**: 来週の注目カード・展望

### 品質基準

- データに基づく客観的分析
- 良い結果も悪い結果も正直に報告
- 読者が予測の信頼性を判断できる情報を提供

---

## Weekly Anthropic Radar

**内部レビュー。公開しない。**

- Anthropicの最新アップデートをチェック
- 運用に影響する変更を特定
- 分類: Immediate adopt / Experiment candidate / Skip / Monitor
- 詳細は `docs/ops/anthropic-feature-watch.md` 参照
