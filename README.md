# KEIBA LAB Publisher Ver.2

Keiba AI Mobile / Dashboardが保存した `.keiba` Prediction Snapshotを読み取り、会場別note完成原稿とレース別X投稿を作る配信専用アプリです。予想処理は呼ばず、Snapshotの能力・順位・能力帯・印・妙味・各評価を変更しません。

## 起動

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## 最短の当日運用

1. `.keiba` を開き、会場を選ぶ。
2. 固定冒頭文・任意の「主のひとこと」を確認し、note原稿をコピーまたはMarkdown保存する。
3. noteで公開し、会場固定URLをPublisherへ登録する。
4. X接続確認、全投稿プレビュー、dry-run、1投稿テストの順に確認する。
5. 選択レースを投稿開始する。1件目の後は5/10/15分の間隔で予約される。

noteについて、一般公開された公式の投稿APIを確認できなかったため、非公式APIやブラウザ自動操作は実装していません。全文コピー、Markdown、プレーンテキストを使用してください。

## X公式API

実装はX API v2のユーザーコンテキストを使用します。

- 接続確認: `GET https://api.x.com/2/users/me`
- Create Post: `POST https://api.x.com/2/tweets`
- 認証: OAuth 1.0a User Context（Consumer Key/Secret + User Access Token/Secret）

X Developer ConsoleでProject/Appを作成し、User authentication settingsでRead and write権限を有効にしてください。権限変更後はアクセストークンを再生成します。Xの料金・利用上限は契約とDeveloper Console表示に従います（料金体系は変更され得るため、運用開始日に必ず確認してください）。

環境変数または `.streamlit/secrets.toml` に次を設定します。値は画面・ソース・Gitに保存しません。

```toml
X_API_KEY = "..."
X_API_SECRET = "..."
X_ACCESS_TOKEN = "..."
X_ACCESS_TOKEN_SECRET = "..."
```

想定ユーザー名（初期値 `keiba_lab_ai`）と `/2/users/me` のusernameが一致しなければ投稿を拒否します。投稿前に会場note URLが必要です。成功時はpost ID、時刻、race_id、本文SHA-256、アカウント、note URLを保存し、`race_id × account × post_type` で二重投稿を防ぎます。失敗時は本文hash、HTTP status、エラー、retry可否を記録し、他レースの状態を維持します。

## Publisher state

schema v2は生成/手動修正版のnote・X本文、固定冒頭文、主のひとこと、会場note URL、公開モード、無料race_id、予約時刻、投稿状態、投稿履歴、X post IDを保存します。元Snapshot本体は格納・変更せず、SHA-256で同じ予想ファイルか照合します。schema v1は読込時にv2へ移行します。

## 公開モード

- 全レース無料
- 1R無料＋全レースnote（無料race_idを指定）

これは公開区分だけで、予想内容は変更しません。有料決済APIは未実装です。

## テスト

```bash
python -m pytest -q
```

認証情報なしで全テストを実行できます。HTTPはモックし、本番投稿は行いません。
