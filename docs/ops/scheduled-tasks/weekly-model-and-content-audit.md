# Weekly Model and Content Audit

## 概要

| 項目 | 内容 |
|------|------|
| **目的** | 予測品質、信頼度ティア、コンテンツ整合性の週次レビュー |
| **実行方法** | Claude Desktop Cowork スケジュールタスク |
| **頻度** | 毎週月曜 11:00 AM ET |
| **推奨モデル** | Opus（精密な分析） |
| **対象フォルダ** | `output/`, `articles/`, `docs/` |

## タスクプロンプト（コピペ用）

```
Perform the weekly Moneyball Dojo audit:

## Part 1: Prediction Quality
- Review this week's prediction outputs in output/daily/
- Count: total picks, strong/moderate/lean distribution
- If results are available, calculate: accuracy by tier, accuracy by market
- Flag any calibration concerns (model says X% but actual rate is Y%)

## Part 2: Content Review
- Check latest articles/ for accuracy claims consistency
- Verify no inflated accuracy numbers (reference: docs/defaults.md brand section)
- Check EN/JA consistency

## Part 3: Improvement Items
- What went wrong this week?
- What worked well?
- Specific recommendations for next week

Output format:
## Weekly Audit — Week of [DATE]
### Prediction Summary
[table: market, picks, strong%, accuracy if known]
### Content Check
[pass/fail items]
### Recommendations
[numbered list, max 5]
### Decision Log Updates Needed
[any new decisions to record]

Save to output/weekly/audit-[DATE].md
```

## 成功基準

- 全マーケット（ML, RL, F5, NRFI）のピック数・ティア分布が集計されること
- 精度の水増し表記がないか検出できること
- EN/JA 記事間の不整合が検出されること

## レビューポイント

- キャリブレーション分析の精度
- コンテンツ内の精度数値が実測値と一致しているか
