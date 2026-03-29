# Daily Morning Brief

## 概要

| 項目 | 内容 |
|------|------|
| **目的** | 朝のブリーフィング — 昨日の実行結果、出力ファイル、未解決課題、配信状態を確認 |
| **実行方法** | Claude Desktop Cowork スケジュールタスク |
| **頻度** | 毎日 10:15 AM ET（GitHub Actions 予測実行 08:00 UTC の後） |
| **推奨モデル** | Sonnet（高速・コスト効率） |
| **対象フォルダ** | `output/`, `docs/`, `.github/` |

## タスクプロンプト（コピペ用）

```
Review today's Moneyball Dojo status:
1. Check output/ for today's prediction files — were they generated successfully?
2. Check if any GitHub Actions runs failed (look at .github/ recent logs if available)
3. Read docs/decision-log.md for any recent decisions
4. Summarize in this format:

## Morning Brief — [DATE]
- **Pipeline**: OK / FAILED / PARTIAL
- **Games today**: [count]
- **Strong picks**: [count] across [markets]
- **Open issues**: [list if any]
- **Publish mode**: publish / caution / hold
- **CEO action needed**: yes/no — [what if yes]

Keep it under 20 lines. Save to output/daily/morning-brief-[DATE].md
```

## 成功基準

- ブリーフが生成されること
- 偽陽性（問題がないのに問題ありと報告）がないこと
- publish mode がパイプラインの実際の状態を正確に反映していること

## レビューポイント

- パイプライン障害が正しく検出されているか
- publish mode のロジックが正しいか
