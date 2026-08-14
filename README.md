# KEIBA LAB Publisher

Keiba AI Dashboard が保存した `.keiba` Prediction Snapshotを読み取り、会場別note原稿・レース別X投稿文・投稿作業状態へ変換する配信用アプリです。

## 重要な境界

- Publisherは予想しません。`predict_jra` / `predict_nar` その他の評価計算を呼びません。
- `.keiba` はDashboardの正本 `core/prediction_snapshot.py` の完全コピーで検証します。
- Snapshotは読み取り専用です。能力値、能力順位、能力帯、今回評価順位、印、妙味ありを変更しません。
- note/Xの文面は保存済みのSnapshot事実だけから生成します。結果や未来オッズは読み込みません。
- 騎手名正規化は文面表示だけに適用し、Snapshotの騎手評価や文字列を書き換えません。
- Publisher Ver.1は自動投稿しません。Xは将来の公式API接続境界だけを用意しています。noteの非公式API・ブラウザ自動操作も行いません。

## 起動

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

1. Dashboardで保存した `.keiba` をアップロードします。
2. 会場・レースを選択して保存予想を確認します。
3. `note原稿` / `X投稿` タブで原稿をプレビュー・編集します。
4. 会場別note URL、X対象、予約時刻を設定します。
5. `Publisher作業状態を保存` からJSONを保存します。

新馬戦除外は初期ONです。レース名または保存済みクラス情報に `新馬` / `メイクデビュー` が明記されたレースだけを除外します。

## Publisher保存JSON

Prediction Snapshot本体は格納・変更せず、元 `.keiba` のSHA-256、Snapshot SHA-256、予想事実SHA-256、race_id一覧、原稿、手動編集、note URL、投稿対象、状態、予約、投稿履歴を保存します。読み込み時に元Snapshotの識別情報と照合します。

投稿状態は `未生成`、`原稿生成済`、`note URL登録済`、`X投稿準備完了`、`投稿済`、`投稿失敗` です。同一 `race_id × Xアカウント × 投稿種別` の投稿済み履歴がある場合は再投稿を拒否します。

将来の無料/有料区分に備え、状態schemaには公開用の印・短評と、詳細分析用の予約セクションを分離して保持します。Ver.1では全原稿を無料公開用として生成し、販売処理は行いません。

## テスト

```bash
python -m pytest -q
```

同梱実データfixtureは2026-08-09 JRAの札幌・新潟・中京、計32レースです。NARは同一schemaのテストfixtureでも検証します。
