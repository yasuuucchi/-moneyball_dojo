# Monthly Rhythm

> 月次の戦略サイクル。各週にテーマを設定し、月末にCEOサマリーを提出する。

---

## Monthly Overview

| Week | Theme | Lead | Output |
|---|---|---|---|
| Week 1 | Strategy Review | Mika + Thompson + Godin | 戦略レビューメモ |
| Week 2 | Model Deep Dive | Silver + James + Tanaka | 精度・キャリブレーションレポート |
| Week 3 | Content & Growth Experiment | Thompson + Godin + Kanda | 実験結果レポート |
| Week 4 | Planning & CEO Summary | Mika | 月次CEOサマリーメモ |

---

## Week 1: Strategy Review

**ポジショニングと方向性の確認。**

### レビュー項目

- **Positioning**: 競合との差別化は維持されているか
  - 他のMLB予測サービスとの比較
  - 読者から見た Moneyball Dojo の強み/弱み
- **Competitors**: 新しい競合の出現、既存競合の動向
  - 無料/有料の予測サービスをスキャン
  - 差別化ポイントの更新
- **Content Design**: コンテンツフォーマットは最適か
  - 読者のエンゲージメント（開封率、クリック率）
  - フォーマット変更の検討
- **Reader Funnel**: 読者獲得ファネルの健全性
  - X → Substack の導線
  - note.com の読者動向
  - 離脱率の確認
- **Paid Readiness**: 有料化の準備状況
  - 無料読者数の推移
  - 有料化に必要な最低読者数の見積もり
  - 有料コンテンツの候補

### Output
- 戦略レビューメモ（1ページ）
- アクションアイテム（あれば）

---

## Week 2: Model Performance Deep Dive

**モデルの健全性を徹底検証する月次の深掘り分析。**

### レビュー項目

- **Accuracy Trends**: 月間の的中率推移
  - 全体精度、モデル別精度
  - 週ごとの変動パターン
  - 月初 vs 月末の比較（データ蓄積の効果）
- **Calibration**: 予測確率の信頼性
  - ECE (Expected Calibration Error)
  - Isotonic Regression による補正効果
  - 信頼度区間ごとの的中率
- **Tier Effectiveness**: ティアシステムの有効性
  - STRONG > MODERATE > LEAN の順序が保たれているか
  - 各ティアのサンプルサイズは十分か
  - ティア閾値の調整が必要か
- **Feature Importance**: 特徴量の重要度変化
  - 月間で重要度が変化した特徴量
  - 新しい特徴量の候補

### Output
- モデルパフォーマンスレポート
- 再学習・調整の推奨事項

---

## Week 3: Content & Growth Experiment Review

**実験の振り返りと次の実験計画。**

### レビュー項目

- **Running Experiments**: 進行中の実験の中間/最終結果
  - X投稿の最適時間帯テスト
  - コンテンツフォーマットのA/Bテスト
  - ハッシュタグ戦略の効果
- **Growth Metrics**: 成長指標の月次レビュー
  - Substack 購読者数の推移
  - X フォロワー数の推移
  - note.com PV数の推移
  - エンゲージメント率の変化
- **New Experiment Design**: 来月の実験計画
  - 仮説の設定
  - 測定方法の定義
  - 成功/失敗の基準

### Output
- 実験結果レポート
- 来月の実験計画（1-2個に絞る）

---

## Week 4: Planning & CEO Summary

**月の締めくくり。来月の計画とCEOへの報告。**

### CEO Summary Memo

以下のフォーマットで月次サマリーを作成:

```
## Moneyball Dojo — Monthly Summary (YYYY年MM月)

### Numbers
- 予測試合数: NNN
- 全体精度: XX.X%
- STRONG精度: XX.X%
- Substack購読者: NNN (+NN)
- Xフォロワー: NNN (+NN)
- コスト: $X.XX/月

### Wins
- [今月の成果 1]
- [今月の成果 2]

### Concerns
- [懸念事項があれば]

### Next Month Focus
- [来月の最優先事項]
- [来月の実験計画]

### Decisions Needed
- [CEOの判断が必要な事項]
```

### Planning

- 来月のコンテンツカレンダー更新
- 来月の実験計画確定
- モデル改善のロードマップ更新
- リソース（コスト）の見通し確認

---

## Monthly Cadence in Context

```
MLB Season Timeline:
  Mar: Spring Training → Opening Day (3/27)
  Apr-Sep: Regular Season (日次予測フル稼働)
  Oct: Postseason (特別コンテンツ)
  Nov-Feb: Offseason (モデル改善、来季準備)

月次サイクルはシーズン中 (Apr-Oct) に最も重要。
オフシーズン (Nov-Feb) はモデル改善と戦略立案に集中。
```
