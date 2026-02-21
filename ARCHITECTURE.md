# Moneyball Dojo — システム全体像

> 凡例: 🟢 自動化済み　🟡 要実装/選択中　🔴 手動/未解決

---

```mermaid
flowchart TD

    %% =========================================
    %% TRIGGERS
    %% =========================================
    T1(["⏰ 毎朝 6:00\nGitHub Actions"])
    T2(["⏰ 毎夜 15:00\nGitHub Actions"])
    T3(["⏰ 毎週月曜\nGitHub Actions"])

    %% =========================================
    %% 1. 予測パイプライン（自動）
    %% =========================================
    subgraph PIPELINE ["🟢 予測パイプライン（完全自動）"]
        direction TB
        API["📡 MLB Stats API\nデータ取得"]
        FEAT["🔢 特徴量エンジニアリング\n37変数"]
        MODELS["🧠 9モデル同時予測\nMoneyline / Run Line / F5\nNRFI / Over-Under\nPitcher K / Batter Props\nStolen Bases / Pitcher Outs"]
        EDGE["📊 エッジ計算\nモデル確率 vs 市場オッズ\n→ STRONG / MODERATE 分類"]

        API --> FEAT --> MODELS --> EDGE
    end

    %% =========================================
    %% 2. コンテンツ生成（最重要選択肢）
    %% =========================================
    subgraph CONTENT ["🟡 コンテンツ生成（ここが選択肢）"]
        direction LR

        subgraph OPTB ["Option B: Manus"]
            B1["🤖 Manus\nブラウザ操作型 AI Agent\n✅ 自然な英語/日本語\n✅ Web検索で文脈補完\n❌ スケジューラではない\n❌ 手動トリガー必要\n💰 クレジット制（高め）"]
        end

        subgraph OPTA ["⭐ Option A: Claude API（推奨）"]
            A1["⚡ Claude API (Haiku)\n✅ GitHub Actions内で全自動\n✅ 英語品質◎\n✅ 日本語 → 英語翻訳OK\n✅ 月 $5-15 と安い\n✅ 既存コードに統合済み\n❌ Web閲覧は不可"]
        end
    end

    %% =========================================
    %% 3. 生成コンテンツ
    %% =========================================
    subgraph OUTPUT ["📄 生成コンテンツ"]
        direction LR
        EN["🇺🇸 英語 Digest\nSubstack 用 Markdown"]
        JA["🇯🇵 日本語 Digest\nnote.com 用 Markdown"]
        TW["🐦 X 投稿文\n英語ピック 3-5本"]
        CSV["📊 予測 CSV\nトラッキング用"]
    end

    %% =========================================
    %% 4. 配信（改善ポイント）
    %% =========================================
    subgraph DIST ["🟡 配信（ここが今の課題）"]
        direction TB

        SUB["📧 Substack\n🟡 メール投稿機能で自動化可\n→ GitHub Actions がメール送信\n→ 自動 Publish 設定で完結\n追加コスト $0"]

        NOTE["📝 note.com\n🔴 公式 API なし\n→ Manus がブラウザ操作\n→ または週 1 手動（15分）\n＊日本語圏は頻度低くてもOK"]

        XAUTO["🐦 X / Twitter\n🟡 Buffer 等で自動スケジュール\n→ GitHub Actions → Buffer API\n→ Buffer → X に自動投稿\n月 $6 〜"]

        SHEETS["📊 Google Sheets\n🟢 完全自動\n実績トラッキング・ROI計算"]
    end

    %% =========================================
    %% 5. エンゲージメント（最難関）
    %% =========================================
    subgraph ENGAGE ["🔴 エンゲージメント（最難関・Manus の出番）"]
        direction LR
        WATCH["👀 X トレンド監視\nフォロー中アカウントの投稿を\nManus がブラウズ\n（@FanGraphs @JeffPassan 等）"]
        QRT["✍️ 引用 RT 文生成\n文脈を読んで AI が英語生成\nManus が実行"]
        ENGAGE_POST["📤 X に投稿\n⚠️ ToS グレーゾーン注意\n頻度は低め推奨"]

        WATCH --> QRT --> ENGAGE_POST
    end

    %% =========================================
    %% 結果フィードバックループ
    %% =========================================
    subgraph FEEDBACK ["🟢 フィードバックループ（自動）"]
        RESULT["🏆 試合結果取得\nMLB Stats API"]
        UPDATE["📈 モデル精度更新\n週次再学習"]
        TRACK["✅ 実績トラッキング\n勝率・ROI 自動計算"]

        RESULT --> TRACK
        RESULT --> UPDATE
    end

    %% =========================================
    %% 接続
    %% =========================================
    T1 --> PIPELINE
    T2 --> FEEDBACK
    T3 --> UPDATE

    EDGE --> CONTENT
    EDGE --> CSV

    OPTA -->|"推奨ルート"| OUTPUT
    OPTB -.->|"手動起動時"| OUTPUT

    EN --> SUB
    JA --> NOTE
    TW --> XAUTO
    CSV --> SHEETS

    ENGAGE_POST --> XAUTO
    TRACK --> SHEETS

    %% =========================================
    %% スタイル
    %% =========================================
    classDef green fill:#16a34a,color:#fff,stroke:#15803d,stroke-width:2px
    classDef yellow fill:#d97706,color:#fff,stroke:#b45309,stroke-width:2px
    classDef red fill:#dc2626,color:#fff,stroke:#b91c1c,stroke-width:2px
    classDef blue fill:#1d4ed8,color:#fff,stroke:#1e40af,stroke-width:2px
    classDef gray fill:#6b7280,color:#fff,stroke:#4b5563,stroke-width:2px

    class PIPELINE,SHEETS,FEEDBACK green
    class CONTENT,DIST,SUB,XAUTO yellow
    class ENGAGE,NOTE red
    class OPTA blue
    class OPTB gray
```

---

## 結論：何を使うべきか

| 役割 | ツール | 理由 |
|------|--------|------|
| **コンテンツ生成** | **Claude API** | 安い・全自動・既存コードに統合済み |
| **Substack 配信** | **メール投稿機能** | APIなしで自動化可・追加コスト $0 |
| **X 毎日投稿** | **Buffer ($6/月)** | 最も安全・確実 |
| **note.com 投稿** | **Manus** | APIなし → ブラウザ操作が唯一の自動化手段 |
| **X エンゲージメント** | **Manus（週数回）** | Web閲覧 + 文脈理解が必要 → Manus の得意領域 |
| **実績トラッキング** | **GitHub Actions** | 既に自動 |

## Manus をどう使うか（具体的）

Manus はスケジューラではなく **「お願いすると動いてくれる賢いエージェント」**。

```
毎週月曜（30分）:
  Manus に「今週の note.com 記事を投稿して」と指示
  → Manus が note.com にログイン → 記事コピペ → 投稿

週 2-3 回:
  Manus に「@FanGraphs や @JeffPassan の最新ツイートを見て、
  MoneyballDojo として引用RTして」と指示
  → Manus が X をブラウズ → 文脈に合う引用RT を投稿
```

**Taiki の実質作業時間: 週 30-60 分**（完全ゼロは現実的に困難）
