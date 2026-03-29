# On-Demand Release Review

## 概要

| 項目 | 内容 |
|------|------|
| **目的** | PR、重要なドキュメント変更、ワークフロー変更のリリース前レビュー |
| **実行方法** | オンデマンド（スケジュールではない） |
| **頻度** | 必要に応じて |
| **推奨モデル** | Opus（精密なレビュー） |
| **対象** | 現在の PR または変更セット |

## タスクプロンプト（コピペ用）

```
Review the pending changes for Moneyball Dojo release:

Check:
1. Does any change touch models/ or training logic? → Flag for Silver + James review
2. Does any change touch .github/workflows/? → Flag for Hightower review
3. Does any change touch prediction logic in run_daily.py? → Flag for Silver + Tanaka review
4. Are there any secrets or credentials exposed?
5. Do docs match the code changes?
6. Is the decision-log updated if a policy/default changed?

Output format:
## Release Review
- **Risk level**: LOW / MEDIUM / HIGH
- **Reviewers needed**: [names]
- **Blocking issues**: [list or "none"]
- **Non-blocking suggestions**: [list]
- **Decision log update needed**: yes/no
- **CEO approval needed**: yes/no — [reason]

Approve / Request Changes / Block
```

## 成功基準

- セキュリティリスク（シークレット漏洩）が確実に検出されること
- 適切なレビュアーが正しくアサインされること
- ブロッキング / ノンブロッキングの区別が的確であること

## レビューポイント

- モデル変更時に Silver + James の両名がアサインされるか
- ワークフロー変更時に Hightower がアサインされるか
- credentials / secrets の検出が漏れなく機能するか
