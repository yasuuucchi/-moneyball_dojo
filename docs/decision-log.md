# Decision Log - Moneyball Dojo

> **目的**: プロジェクトに関する重要な意思決定を記録し、将来の判断に一貫性を持たせるための公式記録。
> 新しいセッションで判断に迷ったら、まずここを参照する。
> 各エントリは「なぜそう決めたか」「次も同じ状況ならどうするか」を明示する。

---

### DL-001: Substack優先
- **Date**: 2026-03-01
- **Owner**: CEO
- **Decision**: Substackで配信確立を最優先とする。Beehiiv等への移行は購読者基盤ができてから検討する。
- **Why**: まず実績を作ることが最重要。プラットフォーム最適化は後回し。Substackは無料で始められ、ニュースレター配信の実績がある。
- **Applies when**: 配信プラットフォームの選定・移行を検討するとき。
- **Default next time**: Substackを使い続ける。移行の議論はしない。
- **Revisit trigger**: 有料購読者が100名を超えた、またはSubstackの機能制限がビジネスに影響し始めたとき。

---

### DL-002: n8n → Claude Code自動化に移行
- **Date**: 2026-03-01
- **Owner**: CEO + Hightower
- **Decision**: n8n + VPS構成を廃止し、Claude Codeの /schedule + Dispatch + Computer Use で自動化する。
- **Why**: VPSコスト（$7/月）削減、セットアップの大幅簡素化、Claude Codeエコシステムへの統一。/scheduleでクラウドcron実行が可能になった。
- **Applies when**: 自動化インフラの設計・変更を検討するとき。
- **Default next time**: Claude Code機能（/schedule, Dispatch）を第一候補とする。外部ツール導入前にClaude Codeで実現可能か確認。
- **Revisit trigger**: Claude Codeの /schedule に重大な制限が見つかった、またはタスク数がProプランの上限に達したとき。

---

### DL-003: Spring Training予測は非公開
- **Date**: 2026-02-27
- **Owner**: CEO + Silver
- **Decision**: Spring Training期間の予測は内部テストのみに留め、公開しない。開幕（3/27）から本番配信を開始する。
- **Why**: Spring Trainingはレギュラーシーズンとデータ特性が大きく異なる（主力選手の出場制限、マイナーリーガーの混在）。精度保証ができない状態で配信するとブランド毀損リスクがある。
- **Applies when**: プレシーズン・オールスター休暇など、通常のレギュラーシーズンと異なる期間の予測配信を検討するとき。
- **Default next time**: レギュラーシーズン以外の予測は非公開。内部検証のみ。
- **Revisit trigger**: Spring Trainingデータでの精度が60%を安定的に超えた場合。

---

### DL-004: WBC予測はコンテンツのみ
- **Date**: 2026-02-27
- **Owner**: Silver
- **Decision**: WBC（ワールド・ベースボール・クラシック）に関してはモデル予測を行わず、コンテンツ（記事・解説）のみを提供する。
- **Why**: WBCのデータはMLBレギュラーシーズンと構造が異なり、訓練データに含まれていない。モデル予測を出すとハルシネーション（根拠のない予測）リスクが高い。
- **Applies when**: MLB以外の国際大会・特殊イベントの予測を検討するとき。
- **Default next time**: 訓練データに含まれないイベントではモデル予測を出さない。
- **Revisit trigger**: WBC/国際大会の十分な過去データが蓄積され、専用モデルを訓練できる状態になったとき。

---

### DL-005: Computer Use = fallback only
- **Date**: 2026-03-01
- **Owner**: CEO
- **Decision**: Computer Use（GUI自動化）は主軸にしない。deterministic core（API・スクリプト）を維持し、Computer UseはAPI未提供のサービスに対するfallbackとして使用する。
- **Why**: GUI自動化はブラウザのDOM変更・レイアウト変更で壊れやすい。信頼性の高い運用にはdeterministicな処理を優先すべき。
- **Applies when**: 新しい自動化タスクの実装方法を選定するとき。
- **Default next time**: まずAPIの有無を調査。APIがあればAPI使用。なければComputer Use。
- **Revisit trigger**: Computer Useの信頼性が大幅に向上し、APIと同等の安定性が確認されたとき。

---

### DL-006: データリーク修正後の精度を正として採用
- **Date**: 2026-02-25
- **Owner**: Silver
- **Decision**: rolling stats + overall_statsのデータリークを修正し、修正後の精度を公式値として採用する。旧精度（64.7%/72.9%）は使用禁止。
- **Why**: 旧精度はoverall_statsの事後データ混入とrolling statsのshift未適用により水増しされていた。正確な精度は以下の通り:
  - ML: 全体52.9% / STRONG 58.7% (550試合)
  - RL: 全体64.5% / STRONG 67.3% (1,482試合)
  - F5: 全体54.5% / STRONG 59.9% (1,096試合)
  - NRFI: 全体56.5% / STRONG 60.1% (667試合)
- **Applies when**: 精度に関する記事・マーケティング資料・外部コミュニケーション全般。
- **Default next time**: 常にリーク修正後の数値を使用する。旧数値を引用しない。
- **Revisit trigger**: 2025シーズン全データでの再バックテスト完了時に数値を更新。

---

### DL-007: 過大な勝率主張を避ける
- **Date**: 2026-03-01
- **Owner**: CEO + Thompson
- **Decision**: 精度や勝率に関して過大な主張をしない。透明性・事前公開・事後監査を前面に出す。
- **Why**: スポーツ予測業界は過大広告が蔓延しており、差別化のためにも誠実なポジショニングが重要。信頼構築が長期的なブランド価値につながる。
- **Applies when**: 記事・X投稿・ランディングページ・購読者向けコミュニケーション全般。
- **Default next time**: 「research desk」ポジショニング。具体的数値は必ずバックテスト結果に基づく。「保証」「確実」等の表現は使わない。
- **Revisit trigger**: なし。これは恒久的なブランド方針。

---

### DL-008: Slack = control plane / notification sink
- **Date**: 2026-03-01
- **Owner**: Hightower
- **Decision**: SlackはClaude実行面（execution plane）ではなく、通知先（notification sink）およびコントロールプレーンとして使用する。
- **Why**: Slackでの実行はレート制限・メッセージ長制限・エラーハンドリングの複雑さがある。通知と監視に特化させることで信頼性を確保。
- **Applies when**: Slack連携の設計・拡張を検討するとき。
- **Default next time**: Slackには結果通知・アラートのみを送信。処理実行はGitHub Actions / /scheduleで行う。
- **Revisit trigger**: Slack Bot APIが大幅に改善され、長時間タスクの実行に適した機能が追加されたとき。

---

### DL-009: note.comはEN mirrorにしない
- **Date**: 2026-03-01
- **Owner**: Kanda
- **Decision**: note.comの日本語記事は英語記事の単純翻訳（mirror）にせず、日本語圏独自の価値を提供する。
- **Why**: 日本の野球ファンはMLBへの関心の角度が異なる（日本人選手中心、NPBとの比較など）。単純翻訳では価値が薄い。
- **Applies when**: 日本語コンテンツの企画・作成時。
- **Default next time**: EN記事を「ベース」として使いつつ、JA版は独自の切り口・補足情報を追加する。
- **Revisit trigger**: note.comの読者フィードバックに基づいて方針を微調整。

---

### DL-010: 追加APIコスト前提の設計をdefaultにしない
- **Date**: 2026-03-01
- **Owner**: CEO
- **Decision**: Proプラン内で最適化することをデフォルトとし、追加APIコストが発生する設計を安易に採用しない。
- **Why**: 事業初期段階で固定費を最小化する。収益が安定してからスケールに応じてコストを増やす。
- **Applies when**: 新機能・自動化・外部サービス連携の設計時。
- **Default next time**: まずProプラン内で実現可能な方法を検討。追加コストが必要な場合はCEO承認を得る。
- **Revisit trigger**: 月間収益がAPIコストの10倍を超えたとき。
