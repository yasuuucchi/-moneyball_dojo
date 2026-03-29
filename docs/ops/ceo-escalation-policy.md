# CEO Escalation Policy

> AIチームが自律的に判断できる範囲と、CEOの承認が必要な範囲を定義する。

---

## Must Escalate（CEOに必ず確認）

以下の項目は、AIチームが独自に判断・実行してはならない。

### 1. Pricing / Monetization
- 有料プラン・有料機能の導入タイミング
- 価格設定の変更
- 無料→有料への移行判断

### 2. Publishing Platform Changes
- Substack から他プラットフォームへの移行
- 新しい配信チャネルの追加（既存: Substack, note.com, X）
- 既存チャネルの廃止

### 3. Brand Message Overhaul
- ブランド名の変更（Moneyball Dojo, Dojo Labs）
- タグラインやポジショニングの大幅変更
- ロゴ・ビジュアルアイデンティティの変更
- ターゲット読者層の変更

### 4. Additional Costs
- 月額 $0 を超える新しい API 契約
- 新しい SaaS ツールの導入
- 有料プランへのアップグレード
- VPS・ホスティング費用の発生

### 5. Publish Hold/Caution Override
- `hold` モード（重大問題）の解除判断
- `caution` モードでの公開強行判断
- 公開済みコンテンツの取り下げ

### 6. Security / Permissions / Destructive Changes
- API キー・シークレットの変更
- リポジトリの公開/非公開設定変更
- データの不可逆削除
- 外部サービスとの新しい認証連携

---

## Don't Escalate（AIチームが自律的に実行）

以下は、CEOの承認なしにチームが判断・実行してよい。

- **ドキュメント整理**: README更新、CLAUDE.md改善、コメント追加
- **サブエージェント調整**: プロンプト改善、出力フォーマット変更
- **Hook改善**: pre-commit, pre-push などの開発フック調整
- **Weekly Radar 結果のコンパイル**: Anthropic更新のまとめ作成
- **既存デフォルトに従った運用**: `docs/defaults.md` に定義済みの判断基準に沿った行動
- **コード品質改善**: リファクタリング、テスト追加、型ヒント追加
- **バグ修正**: 既存機能の不具合修正（機能変更を伴わないもの）
- **定期レポート生成**: Weekly Audit、精度レポートの自動生成

---

## Precedent-Based Proposal Format

エスカレーション時は、以下のフォーマットで提案する。
過去の決定を参照することで、CEOの判断負荷を最小化する。

```
## エスカレーション: [タイトル]

### Previous Decision
- 決定番号: DL-NNN
- 内容: [過去の関連する決定を引用]
- 日付: YYYY-MM-DD

### Current Similarity
- [今回の状況が過去の決定とどう類似しているか]
- [共通する判断基準は何か]

### Default Proposal
- [推奨アクション]
- [想定されるコスト/リスク]
- [期待される効果]

### Exception Reason (if any)
- [過去の決定から逸脱する場合、その理由]
- [逸脱によるリスクと対策]
```

### 例

```
## エスカレーション: The Odds API 有料プラン検討

### Previous Decision
- 決定番号: DL-003
- 内容: 「月額コストは $0 を維持。無料枠で運用する」
- 日付: 2026-03-15

### Current Similarity
- The Odds API 無料枠 (500 req/月) で運用中
- 今回も外部API費用に関する判断

### Default Proposal
- 無料枠の範囲内で継続運用（リクエスト数を最適化）
- コスト: $0/月
- 効果: 現状維持、機能制限あり

### Exception Reason
- 無料枠では全試合のオッズ取得が困難
- 有料プラン ($10/月) で全試合カバー可能
- 精度向上の可能性あり（オッズデータは強力な特徴量）
```

---

## Decision Log

エスカレーションの結果は `docs/decision-log.md` に記録する。

| ID | Date | Topic | Decision | Rationale |
|---|---|---|---|---|
| DL-001 | 2026-03-XX | Substack優先 | Substackで配信確立を最優先 | 実績作りが先 |
| DL-002 | 2026-03-XX | n8n→Claude Code移行 | /schedule + Computer Useに移行 | コスト削減 |
| ... | ... | ... | ... | ... |
