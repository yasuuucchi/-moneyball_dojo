# Automation Architecture

> Moneyball Dojo の自動化は3層構造で設計されている。
> 上位レイヤーが落ちても、下位レイヤーは独立して動作する。

---

## Overview: 3-Layer Architecture

```
+================================================================+
|  Layer 3: Human-in-the-Loop (CEO via Slack / Claude Code App)  |
|  判断: 公開承認、ブランド変更、コスト追加                          |
+================================================================+
        |  エスカレーション（publish hold / cost / brand）
        v
+================================================================+
|  Layer 2: Agentic Ops (Claude Desktop Cowork Scheduled Tasks)  |
|  条件: Desktop が起動中のときのみ実行                              |
|  タスク: morning brief, publish triage, weekly audit, radar     |
+================================================================+
        |  読み取り・分析（予測結果、モデル精度、Anthropic更新）
        v
+================================================================+
|  Layer 1: Deterministic Core (run_daily.py + GitHub Actions)   |
|  無条件: LLM不要。毎日確実に動く。                                 |
|  Cron: 08:00 UTC 予測 / 15:00 UTC 結果 / Mon 00:00 UTC 再学習  |
+================================================================+
        |
        v
  [MLB Stats API] --> [Google Sheets] --> [Slack Webhook]
```

---

## Layer 1: Deterministic Core

**原則: LLMが使えなくても、予測パイプラインは止まらない。**

| Schedule (UTC) | Workflow | 処理内容 |
|---|---|---|
| 毎日 08:00 | `daily-predictions.yml` | run_daily.py → 予測生成 → Google Sheets 書込 → Slack通知 |
| 毎日 15:00 | `daily-results.yml` | 結果取得 → 精度更新 → Google Sheets 更新 → Slack通知 |
| 月曜 00:00 | `weekly-retrain.yml` | train_all_models.py → 9モデル再学習 → git push |

### 構成要素

- **run_daily.py**: メインパイプライン。MLB Stats API からデータ取得 → XGBoost 推論 → Google Sheets 書込
- **GitHub Actions**: cron トリガー。サーバーレス実行
- **Google Sheets**: データベース。予測・結果・精度を一元管理
- **Slack Webhook**: 実行結果の通知。成功/失敗を即座にCEOへ

### 障害時の挙動

- MLB Stats API ダウン → エラー通知、翌日リトライ
- Google Sheets API エラー → ローカルCSV保存、次回同期
- GitHub Actions 障害 → Slack通知、手動実行可能

---

## Layer 2: Agentic Ops

**原則: 日次予測パイプラインではない。分析・整理・提案を行う知的レイヤー。**

| Task | Schedule | 内容 |
|---|---|---|
| Morning Brief | 毎朝 (Desktop起動時) | 本日の予測サマリー、注目カード、公開判断の下書き |
| Publish Triage | Morning Brief後 | 公開モード判定 (publish / caution / hold) |
| Weekly Audit | 毎週月曜 | 週間精度レビュー、モデル健全性チェック、Substack記事生成 |
| Anthropic Radar | 毎週 | Anthropic新機能チェック、運用改善提案 |

### 重要な制約

- **Desktop が閉じていれば実行されない** — それで構わない
- Layer 1 が確実に動いているため、Layer 2 の不在は致命的ではない
- Layer 2 の出力は「提案」であり、最終判断は Layer 3 (CEO) が行う

---

## Layer 3: Human-in-the-Loop

**原則: CEOは判断だけする。作業はしない。**

### CEOが判断するもの

- `caution` / `hold` モードの公開判断
- ブランドメッセージの変更
- 新しいコスト（API、SaaS）の承認
- 有料化タイミングの決定

### CEOの操作手段

- **Slack**: 通知受信、簡易承認（リアクション）
- **Claude Code App (スマホ)**: 詳細な指示、戦略相談
- **Claude Desktop**: 複雑な作業の実行指示

### フロー例: 通常の1日

```
08:00 UTC  [Layer 1] run_daily.py 自動実行 → 予測生成 → Sheets書込 → Slack通知
           CEO: Slack通知を確認（スマホ）

09:00      [Layer 2] Morning Brief → 本日の注目カード整理 → Publish Triage
           → publish モード → 自動公開準備

10:00      [Layer 3] CEO: 公開内容をSlackで確認 → 問題なければスルー
           → X投稿・Substack更新が実行される

15:00 UTC  [Layer 1] 結果取得 → 精度更新 → Slack通知
           CEO: 的中率をSlackで確認
```

### フロー例: 問題発生時

```
08:00 UTC  [Layer 1] run_daily.py 実行 → 予測生成完了 → Slack通知

09:00      [Layer 2] Publish Triage → モデル精度が閾値以下を検出
           → caution モード設定 → CEOにSlack通知

10:00      [Layer 3] CEO: cautionの理由を確認
           → 「今日は予測公開を見送り」と判断 → hold に変更
           → 分析記事のみ公開
```

---

## Design Principles

1. **下位レイヤーは上位に依存しない**: Layer 1 は単独で完結する
2. **上位レイヤーは価値を追加する**: Layer 2 は分析、Layer 3 は判断
3. **障害は上に伝播しない**: Desktop が落ちても予測は出る
4. **コストは最小**: GitHub Actions (無料枠) + Google Sheets (無料) + Slack (無料)
