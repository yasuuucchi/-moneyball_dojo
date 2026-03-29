# Weekly Anthropic Radar

## 概要

| 項目 | 内容 |
|------|------|
| **目的** | このプロジェクトに関連する Anthropic プラットフォームの更新を追跡 |
| **実行方法** | Claude Desktop Cowork スケジュールタスク |
| **頻度** | 毎週月曜 2:00 PM ET |
| **推奨モデル** | Sonnet |
| **対象** | Web 検索 + `docs/ops/anthropic-feature-watch.md` |

## タスクプロンプト（コピペ用）

```
Perform the weekly Anthropic platform radar for Moneyball Dojo:

Check for updates to:
1. Claude general release notes
2. Claude Code (settings, hooks, subagents, memory)
3. Claude Code on the web
4. Cowork / scheduled tasks
5. Claude in Chrome
6. Computer Use

For each finding, classify:
- ADOPT NOW: directly improves our ops, no cost
- EXPERIMENT: worth testing
- SKIP: not relevant
- MONITOR: interesting, not ready

Output format:
## Anthropic Radar — Week of [DATE]
### Changes Found
[numbered list with classification]
### Recommended Actions
[what to do this week]
### No Action Needed
[items checked with no relevant changes]

Keep to 1 page. Only include items relevant to Moneyball Dojo operations.
Save to output/weekly/anthropic-radar-[DATE].md
Update docs/ops/anthropic-feature-watch.md if needed.
```

## 成功基準

- 主要な Anthropic プラットフォーム変更が漏れなく検出されること
- 分類（ADOPT NOW / EXPERIMENT / SKIP / MONITOR）が適切であること
- Moneyball Dojo の運用に無関係な情報でノイズが発生しないこと

## レビューポイント

- ADOPT NOW の判定が的確か（すぐ導入すべきものを見逃していないか）
- `docs/ops/anthropic-feature-watch.md` が最新状態に保たれているか
