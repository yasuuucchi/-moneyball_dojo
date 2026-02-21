# Moneyball Dojo v4 — 完全自動化セットアップガイド

> **対象**: 非エンジニアでも絶対にミスらない
> **所要時間**: 約30〜40分
> **必要なもの**: ブラウザだけ（Anthropic / Google Cloud / GitHub アカウントは取得済み前提）

---

## まず全体像を理解する

GitHub Actions が**毎朝自動**で予測を実行するために、4つの「鍵」を GitHub に登録します。
鍵を登録するだけで、あとは全部自動で動きます。

```
あなたがやること = 「鍵を4つコピペして GitHub に貼る」だけ
```

| # | 鍵の名前 | 何に使う | 必須？ |
|---|---------|---------|--------|
| 1 | `CLAUDE_API_KEY` | AI が記事を自動生成する | ✅ 必須 |
| 2 | `GOOGLE_SHEETS_CREDENTIALS` | Sheets に予測を自動書き込む | ✅ 必須 |
| 3 | `GOOGLE_SHEETS_ID` | どのスプレッドシートに書くか | ✅ 必須 |
| 4 | `SLACK_WEBHOOK` | エラー時に Slack 通知する | 🟡 推奨 |

---

## ステップ 1: Anthropic API キーを取得する（5分）

### 1-1. Anthropic Console を開く
1. ブラウザのアドレスバーに **`console.anthropic.com`** と入力して Enter
2. ログインする（アカウントは取得済み）

### 1-2. API キーを作成する
1. ログイン後の画面で、**左側のメニュー** を見る
2. **「API Keys」** という項目をクリック
3. 画面右上の **「Create Key」** ボタン（青or緑のボタン）をクリック
4. 「Key name」に **`Moneyball Dojo`** と入力
5. **「Create Key」** ボタンを押す
6. 画面にキーが表示される。**`sk-ant-`** で始まる長い文字列です

> ⚠️ **超重要**: このキーは **この画面を閉じたら二度と見れません**。
> 今すぐ以下をやってください：
> 1. キーの右側の **コピーボタン**（📋アイコン）をクリック
> 2. **メモ帳**（WindowsならNotepad、MacならTextEdit）を開く
> 3. **Ctrl+V**（MacはCmd+V）で貼り付けて保存しておく

### 1-3. GitHub にこの鍵を登録する
1. ブラウザの**新しいタブ**で GitHub のリポジトリを開く
   - URL: `https://github.com/yasuuucchi/-moneyball_dojo`
2. ページ上部にタブが並んでいる（Code, Issues, Pull requests...）
3. 一番右のほうにある **「Settings」**（⚙️ 歯車マーク）をクリック

> もし Settings が見つからない場合、タブが画面に収まりきっていません。
> 「...」をクリックすると隠れているタブが出てきます。

4. Settings ページの **左側メニュー** を下にスクロール
5. **「Secrets and variables」** という項目を見つけてクリック
6. すると下に展開されるので **「Actions」** をクリック
7. 画面右上の緑のボタン **「New repository secret」** をクリック
8. 2つの入力欄が出てくる:
   - **Name 欄**: `CLAUDE_API_KEY` と入力（大文字・アンダースコアに注意。コピペ推奨）
   - **Secret 欄**: メモ帳に保存した `sk-ant-...` のキーを貼り付け
9. **「Add secret」** ボタンをクリック

**確認方法**: ページに `CLAUDE_API_KEY` と表示されたら成功（中身は「***」で隠されています）

✅ **ステップ 1 完了！**

---

## ステップ 2: Google Sheets の設定（20分）

ここが一番長いですが、画面のスクショを頭に浮かべながら一つずつやれば大丈夫です。

### 2-1. Google Cloud でプロジェクトを作る

1. ブラウザで **`console.cloud.google.com`** を開く
2. ログインする
3. ページ上部に **プロジェクト名**（例:「My First Project」）が表示されている → **これをクリック**
4. ポップアップウィンドウが出る
5. ポップアップの **右上** にある **「新しいプロジェクト」** をクリック
6. 入力する:
   - プロジェクト名: **`Moneyball Dojo`**
   - 場所: そのまま（変更不要）
7. **「作成」** ボタンをクリック
8. 10秒ほど待つ
9. 画面右上の **🔔 ベルアイコン** をクリック
10. 「プロジェクト "Moneyball Dojo" を作成しました」というメッセージが出る
11. そのメッセージの中の **「プロジェクトを選択」** をクリック

> これで「Moneyball Dojo」プロジェクトの中に入りました。
> ページ上部のプロジェクト名が「Moneyball Dojo」になっていれば OK。

### 2-2. Google Sheets API を有効にする

1. 画面 **左上** のハンバーガーメニュー（**☰** 三本線のアイコン）をクリック
2. メニューの中から **「API とサービス」** を探す（上のほうにあります）
3. さらに展開されたメニューから **「ライブラリ」** をクリック
4. 大きな検索バーが出る
5. 検索バーに **`Google Sheets API`** と入力して Enter
6. 検索結果に **「Google Sheets API」**（Google のアイコン付き）が出る → **クリック**
7. 青い **「有効にする」** ボタンをクリック
8. 数秒待つと「API が有効になりました」と表示される

### 2-3. サービスアカウントを作る

> **サービスアカウント** = ロボット専用の Google アカウント。
> このロボットが Google Sheets に書き込みを行います。

1. 画面 **左上** のハンバーガーメニュー（☰）をクリック
2. **「API とサービス」** → **「認証情報」** をクリック
3. 画面上部の **「+ 認証情報を作成」** ボタンをクリック
4. ドロップダウンメニューが出る → **「サービス アカウント」** を選択
5. 入力する:
   - サービス アカウント名: **`moneyball-bot`**
   - サービス アカウント ID: 自動で入る（触らないでOK）
   - 説明: 空欄でOK
6. **「作成して続行」** ボタンをクリック
7. 次の画面「このサービス アカウントにプロジェクトへのアクセスを許可する」
   - **何も選ばなくてOK** → **「続行」** をクリック
8. 次の画面「ユーザーにこのサービス アカウントへのアクセスを許可」
   - **何も入力しなくてOK** → **「完了」** をクリック

### 2-4. JSON キーファイルをダウンロードする

1. 「認証情報」ページに戻っているはず
2. ページ下部の **「サービス アカウント」** セクションを見る
3. **`moneyball-bot@moneyball-dojo-xxxxx.iam.gserviceaccount.com`** というメールが表示されている
4. **このメールアドレスをクリック**
5. サービスアカウントの詳細ページに移動する
6. 画面上部のタブから **「キー」** タブをクリック
7. **「鍵を追加」** ボタンをクリック → **「新しい鍵を作成」** を選択
8. キーのタイプ: **「JSON」が選択されていることを確認**（デフォルトで JSON になっているはず）
9. **「作成」** をクリック
10. 📥 **ファイルが自動でダウンロードされる**

> ファイル名は `moneyball-dojo-xxxxx-xxxxxxxxxx.json` のような名前です。
> ダウンロードフォルダに入っています。
>
> ⚠️ **このファイルはパスワードと同じくらい大切です**。他人に渡さないでください。

### 2-5. JSON の中身を GitHub に登録する

1. ダウンロードした JSON ファイルを探す
   - Windows: 「ダウンロード」フォルダ
   - Mac: 「ダウンロード」フォルダ
2. ファイルを **右クリック**
3. **「プログラムから開く」** → **「メモ帳」**（Windows）/ **「テキストエディット」**（Mac）
4. ファイルの中身が表示される。こんな感じ:
   ```
   {
     "type": "service_account",
     "project_id": "moneyball-dojo-xxxxx",
     "private_key_id": "...",
     ...
   }
   ```
5. **Ctrl+A**（Mac: Cmd+A）で **全選択**
6. **Ctrl+C**（Mac: Cmd+C）で **コピー**
7. GitHub のリポジトリの **Settings → Secrets and variables → Actions** に行く
   - 直接URL: `https://github.com/yasuuucchi/-moneyball_dojo/settings/secrets/actions`
8. **「New repository secret」** をクリック
9. 入力する:
   - **Name**: `GOOGLE_SHEETS_CREDENTIALS`（コピペ推奨）
   - **Secret**: さっきコピーした JSON の **中身をまるごと** 貼り付け（Ctrl+V）
10. **「Add secret」** をクリック

### 2-6. スプレッドシートを作る

1. ブラウザの新しいタブで **`sheets.google.com`** を開く
2. **「+」空白のスプレッドシート** をクリック（左上にある）
3. 左上の「無題のスプレッドシート」をクリックして、名前を **`Moneyball Dojo DB`** に変更
4. 画面下部を見る → **「シート1」** というタブがある
5. 「シート1」を **右クリック** → **「名前を変更」** → **`Daily Predictions`** と入力して Enter
6. 「Daily Predictions」タブの右にある **「＋」ボタン** をクリック
7. 新しく追加されたシートを **右クリック** → **「名前を変更」** → **`predictions`** と入力して Enter

> これでシートが2つある状態:
> - `Daily Predictions`（メインの予測データ用）
> - `predictions`（v2 event log 用）

### 2-7. サービスアカウントにスプレッドシートを共有する

> ロボット（サービスアカウント）がこのスプレッドシートに書き込めるように、
> ロボットのメールアドレスに「編集者」権限を与えます。

1. ステップ 2-4 でダウンロードした JSON ファイルを **もう一度メモ帳で開く**
2. ファイルの中から **`"client_email":`** という行を探す（Ctrl+F で検索できます）
3. こんな感じで書いてある:
   ```
   "client_email": "moneyball-bot@moneyball-dojo-xxxxx.iam.gserviceaccount.com",
   ```
4. **ダブルクォーテーション（"..."）の中のメールアドレスだけ** をコピー
   - つまり `moneyball-bot@moneyball-dojo-xxxxx.iam.gserviceaccount.com` の部分
5. Google Sheets（`Moneyball Dojo DB`）の画面に戻る
6. 画面右上の **「共有」** ボタン（緑のボタン）をクリック
7. 「ユーザーやグループを追加」という入力欄に **さっきコピーしたメールアドレスを貼り付け**
8. 右側のプルダウンが **「編集者」** になっていることを確認
9. **「送信」** をクリック
10. 「このユーザーは Google アカウントを持っていません。通知を送信しますか？」→ **「とにかく共有」** をクリック

### 2-8. スプレッドシート ID を GitHub に登録する

1. Google Sheets の **アドレスバー（URL）** を見る
2. URL はこんな形:
   ```
   https://docs.google.com/spreadsheets/d/1AbCdEfGhIjKlMnOpQrStUvWxYz123456789/edit
   ```
3. **`/d/` と `/edit` の間** にある長い文字列を選択してコピー
   - 上の例だと `1AbCdEfGhIjKlMnOpQrStUvWxYz123456789` の部分
   - **`/d/` や `/edit` は含めないでください**

> コツ: `/d/` の直後にカーソルを置いて、`/edit` の手前まで選択 → Ctrl+C

4. GitHub Secrets に戻る
   - `https://github.com/yasuuucchi/-moneyball_dojo/settings/secrets/actions`
5. **「New repository secret」** をクリック
6. 入力する:
   - **Name**: `GOOGLE_SHEETS_ID`（コピペ推奨）
   - **Secret**: さっきコピーしたスプレッドシート ID
7. **「Add secret」** をクリック

✅ **ステップ 2 完了！**

---

## ステップ 3: Slack 通知の設定（5分）

### 3-1. Slack App を作る
1. ブラウザで **`api.slack.com/apps`** を開く
2. **「Create New App」** ボタン（緑）をクリック
3. **「From scratch」** を選択
4. 入力する:
   - App Name: **`Moneyball Dojo Bot`**
   - Pick a workspace: あなたの Slack ワークスペースをドロップダウンから選択
5. **「Create App」** をクリック

### 3-2. Webhook を有効にする
1. App の設定画面に飛ぶ
2. 左メニューの **「Incoming Webhooks」** をクリック
3. 画面右上のスイッチ（トグル）を **「On」** にする
4. ページを下にスクロール
5. **「Add New Webhook to Workspace」** ボタンをクリック
6. 通知を送りたいチャンネルを選ぶ（例: `#moneyball-dojo` や `#general`）
7. **「許可する」** をクリック
8. ページが戻り、Webhook URL が表示される
   - `https://hooks.slack.com/services/T.../B.../...` という長いURL
9. URL の右の **「Copy」** をクリック

### 3-3. GitHub に登録する
1. GitHub Secrets に戻る
2. **「New repository secret」** をクリック
3. 入力する:
   - **Name**: `SLACK_WEBHOOK`
   - **Secret**: さっきコピーした URL
4. **「Add secret」** をクリック

✅ **ステップ 3 完了！**

---

## ステップ 4: 最終確認（3分）

### 4-1. 登録した Secrets を確認する

GitHub の Settings → Secrets → Actions ページに、**以下の 4 つ**が全て表示されていれば OK:

- [x] `CLAUDE_API_KEY`
- [x] `GOOGLE_SHEETS_CREDENTIALS`
- [x] `GOOGLE_SHEETS_ID`
- [x] `SLACK_WEBHOOK`

> 各シークレットの中身は「***」で隠されているのが正常です。

### 4-2. テスト実行する

1. GitHub リポジトリのトップページに戻る
2. 上部タブの **「Actions」** をクリック
3. 左メニューから **「Moneyball Dojo Daily Predictions」** をクリック
4. 右側の **「Run workflow」** ボタンをクリック
5. ドロップダウンが出る:
   - **Branch**: そのまま（デフォルト）
   - **Action to run**: `predict`
   - **Run in test mode**: `false`
6. 緑の **「Run workflow」** ボタンをクリック

### 4-3. 結果を確認する

1. ページをリロードすると、**黄色い丸印**（実行中）のワークフローが表示される
2. クリックして進捗を見る
3. 全部 **✅ 緑のチェック** になったら成功！

**確認ポイント**:
- 📊 Google Sheets の「Moneyball Dojo DB」を開く → 予測データが入っている
- 💬 Slack のチャンネルを見る → 通知が来ている
- 📁 Actions ページの下部 → 「Artifacts」に `daily-predictions` がある

---

## これで完了！今後の日常

### 何もしなくて OK — 全自動で動きます

| 時間 (UTC) | 日本時間 | 何が起こるか |
|-----------|---------|------------|
| 毎日 08:00 | 17:00 | 予測生成 → AI記事生成 → Sheets書き込み → Slack通知 |
| 毎日 15:00 | 翌00:00 | 試合結果を自動取得 → データ更新 |
| 毎週月曜 00:00 | 09:00 | モデル自動再学習 |

### エラーが起きたら

自分で気づく必要はありません:
- **Slack** に通知が来る
- **GitHub Issues** に自動で Issue が作られる
- エラーログは `output/YYYYMMDD/pipeline_errors.log` に保存される

---

## よくあるトラブルと対処法

| 症状 | 原因 | 対処法 |
|------|------|--------|
| Actions が動かない | Secret の名前が違う | Settings → Secrets で名前を確認。大文字・アンダースコアは正確に |
| Sheets に書き込めない | 共有設定を忘れた | ステップ 2-7 をもう一度やる |
| 記事が生成されない | Anthropic の残高がない | `console.anthropic.com` → Billing で残高確認 |
| Slack 通知が来ない | Webhook URL を間違えた | ステップ 3-2 で新しい URL を取り直す |
| 予測が0件 | シーズンオフ or API障害 | MLB のシーズン中（3月末〜10月）のみ予測が生成される |

---

## Secret 名のコピペ用（ミス防止）

以下をそのままコピーして Name 欄に貼ってください:

```
CLAUDE_API_KEY
```

```
GOOGLE_SHEETS_CREDENTIALS
```

```
GOOGLE_SHEETS_ID
```

```
SLACK_WEBHOOK
```
