# Monthly Strategy Review

## 概要

| 項目 | 内容 |
|------|------|
| **目的** | ポジショニング、競合、コンテンツ、グロース、有料化の月次戦略レビュー |
| **実行方法** | Claude Desktop Cowork スケジュールタスク |
| **頻度** | 毎月第1月曜 3:00 PM ET |
| **推奨モデル** | Opus（精密な分析） |
| **対象フォルダ** | `docs/strategy/`, `output/weekly/`, `docs/decision-log.md` |

## タスクプロンプト（コピペ用）

```
Perform the monthly Moneyball Dojo strategy review:

## 1. Positioning Check
- Review docs/strategy/brand-positioning.md — still accurate?
- Any competitor moves that require response?

## 2. Track Record
- Aggregate this month's prediction performance from weekly audits
- Overall accuracy by market and tier
- Trend vs previous months

## 3. Content Strategy
- Review docs/strategy/content-architecture.md
- What content performed well? What didn't?
- Substack subscriber growth (if data available)
- note.com engagement (if data available)

## 4. Growth
- X follower growth
- Newsletter subscriber trend
- What distribution experiments to try next month?

## 5. Paid Readiness
- Is there enough track record for paid content?
- What would paid subscribers get?
- Recommended: wait / soft launch / full launch

## 6. CEO Memo
- 3 things going well
- 3 things to improve
- 1 decision needed from CEO (if any)
- Default if no CEO response

Save to output/monthly/strategy-review-[YYYY-MM].md
Update docs/decision-log.md if new decisions emerge.
```

## 成功基準

- 全6セクションが網羅されていること
- 週次監査からの数値が正確に集計されていること
- CEO メモが簡潔で意思決定に役立つこと

## レビューポイント

- 有料化判定の閾値が適切か
- 競合分析が表面的でなく実用的か
- 前月比のトレンド分析が正しいか
