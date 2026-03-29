# Daily Publish Triage

## 概要

| 項目 | 内容 |
|------|------|
| **目的** | 本日の配信モード（publish / caution / hold）を判定 |
| **実行方法** | Claude Desktop Cowork スケジュールタスク |
| **頻度** | 毎日 10:30 AM ET |
| **推奨モデル** | Sonnet |
| **対象フォルダ** | `output/`, `models/` |

## タスクプロンプト（コピペ用）

```
Determine today's Moneyball Dojo publish mode:

Check:
1. Did today's predictions generate successfully? (output/ files exist and are valid)
2. Are there any model warnings or anomalies? (extreme probabilities, missing games)
3. Are there any known issues flagged in recent morning briefs?

Decision matrix:
- PUBLISH: Predictions generated normally, no anomalies → proceed with normal publishing
- CAUTION: Minor anomalies (missing 1-2 games, some extreme probs) → publish with disclaimer, notify CEO
- HOLD: Major failures (no predictions, >50% games missing, model errors) → do NOT publish, escalate to CEO

Output format:
## Publish Triage — [DATE]
- **Mode**: PUBLISH / CAUTION / HOLD
- **Reason**: [1 sentence]
- **Games covered**: [X/Y]
- **Anomalies**: [list or "none"]
- **CEO notification**: not needed / sent / URGENT

Save to output/daily/publish-triage-[DATE].md
```

## 成功基準

- 配信モードが正しく判定されること
- 異常検出の閾値が適切であること
- HOLD 判定時に CEO へ確実にエスカレーションされること

## レビューポイント

- CAUTION と HOLD の境界が適切か
- 極端な確率値の検出が機能しているか
