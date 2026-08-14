# KEIBA LAB Publisher Ver.3

保存済み `.keiba` Prediction Snapshotを、全レースnote原稿・手動X投稿文・公開記録・結果リプへ変換する公開運用アシスタントです。

Publisherは予想しません。能力値、能力順位、今回評価、能力帯、◎○▲△☆、妙味、騎手、保存済み考察材料を再計算・補正・変更しません。Dashboard側の予想ロジックも変更しません。

## 起動

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

X API、API Key、Access Token、OAuth設定は不要です。Xへの投稿は完成文をコピーして手動で行います。

## 運用手順

1. `.keiba` を読み込む。
2. 今日の運用モードを選ぶ。
   - 🏢 通常平日：全レースnote＋全レースX原稿
   - 🏃 忙しい日：全レースnote＋手動選択した無料1R
   - 🌴 休日：全レースnote＋手動選択した無料1R
3. 固定冒頭文と任意の「主のひとこと」を確認し、note原稿をコピーまたはMarkdown保存する。
4. note公開後、当日の会場note URLを入力する。
5. URL込み280文字以内のX原稿をコピーし、手動投稿後に「X投稿済みにする」を押す。
6. 正式結果JSONがある場合だけ読み込み、結果リプ案と「主の結果ひとこと」を作る。
7. Publisher stateを保存する。

## 公開候補

候補表示は、Snapshotに既にある◎の能力/今回順位、考察材料数、妙味、能力1位と◎の関係だけを使います。これは予想評価ではなく、公開コンテンツとして説明しやすいレースの提案です。最終選択は常にユーザーが行います。

## 正式結果JSON

`.keiba` 内の `mobile_snapshot.result_file` に正式結果があれば利用します。空の場合は「結果データ未取得」です。別JSONも読み込めます。

```json
{
  "race_id": "202644081402",
  "results": [
    {"horse_no": "16", "rank": "1", "horse_name": "ガーリッシュ"}
  ],
  "payoffs": {
    "wide": [{"combination": "3-16", "payout": 410}]
  }
}
```

着順・配当はJSONに明記された値だけを表示します。組合せ、配当、的中を推測しません。

## state v3

次を保存します。

- 運用モード、無料公開race_id
- 生成note、手動修正版note、固定冒頭文、主のひとこと
- 生成X、手動修正版X、会場note URL
- 手動公開記録、本文SHA-256、無料公開区分
- 正式結果データ、結果リプ、主の結果ひとこと

Ver.1/2 stateは読込時にv3へ移行します。旧X API履歴は `legacy_x_api_history` に隔離し、新しい手動公開記録と混在させません。

## テスト

```bash
python -m pytest -q
```

本番API・ブラウザ操作なしで実行できます。
