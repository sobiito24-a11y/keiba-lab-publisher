# KEIBA LAB Publisher Ver.2 実装報告

## 1. 既存処理の監査

既存Publisherは `.keiba` の正規ローダー、予想事実SHA-256、会場別note/X下書き、騎手名正規化、state schema v1、`race_id × account × post_type` の二重投稿境界を持っていました。`posting.py` は無効化スタブで、予約実行・X API・post ID/HTTP失敗履歴は未実装でした。

Dashboard/同梱旧コードには `prediction_result.ai_race_review`、`mobile_snapshot.market_compare.race_summary`、JRA/NAR notebookの展開考察がありました。いずれも保存済みSnapshot材料として参照可能ですが、予測関数を呼ぶ動的再生成はPublisherの境界を越えるため不使用です。Ver.2は既存Snapshotの `horses` と最終 `mark` を正本にした文章レイヤーとして実装しました。

## 2. 改善内容

- 指定の固定冒頭文、日付/会場見出し、タイトル自動生成
- ◎○▲を必ず比較する2～4段落、能力1位≠◎、能力上位の印下げ、展開/コース/距離/近走/状態/斤量/調教/妙味の選択的文章化
- 本命欄と妙味欄（妙味は買い推奨と断定しない）
- 最終印整合、今回順位、内部ラベル、禁止語、X 280文字の公開直前検査
- レース番号・能力構図・材料に応じた複数構文
- 会場固定note URL、全レース無料/1R無料モード、主のひとこと、固定ポスト文
- 生成原稿と手動修正版を別保存

## 3. X API・運用

OAuth 1.0a User Contextで `GET /2/users/me` と `POST /2/tweets` を使用します。想定username不一致、認証なし、note URLなし、二重投稿は拒否します。dry-run、明示確認付き1投稿テスト、全件プレビュー後の一括開始、5/10/15分間隔、指定時刻予約を実装しました。予約監視は30秒ごとで、Publisherを起動したままにする必要があります。

成功履歴はpost ID、投稿日時、race_id、本文SHA-256、アカウント、note URLを保存します。失敗履歴はエラー、HTTP status、retry可否も保存し、他レースを継続します。

## 4. note公開

一般公開された公式投稿APIを確認できなかったため自動公開は行いません。非公式API・ブラウザ自動操作も不使用です。全文コピー、Markdown、プレーンテキスト保存を用意しました。

## 5. 実大井データ監査

添付 `2026-08-14_NAR_大井.keiba` の大井1R～10Rで生成しました。

- 印不一致: 0
- 今回順位矛盾: 0
- 騎手誤表記: 0
- 内部ラベル残存: 0
- X最大換算文字数: 210/280
- note URL挿入: 10/10
- 二重投稿拒否: 10/10
- Prediction Snapshot署名: 生成前後一致
- 考察最大ペア類似度: 0.807（馬名差替えの完全一致なし）

完成原稿は `examples/2026-08-14_大井_note_v2.md`、X全10件は `examples/2026-08-14_大井_x_v2.json`、機械監査値は `examples/2026-08-14_大井_audit_v2.json` に同梱しています。

## 6. テスト

`44 passed`。JRA/NAR、複数会場、固定冒頭、◎○▲、能力1位≠◎、妙味有無、騎手正規化、最終印、X文字数、URL、認証、dry-run、成功/失敗履歴、二重投稿、state再読込、Snapshot不変を確認しました。予想関数の呼出しは0回です。

## 7. 変更ファイル

- `.gitignore`
- `README.md`
- `app.py`
- `publisher/content.py`
- `publisher/jockey.py`
- `publisher/posting.py`
- `publisher/state.py`
- `requirements.txt`
- `tests/test_content.py`
- `tests/test_v2.py`
- `scripts/audit_ooi_v2.py`
- `examples/2026-08-14_大井_note_v2.md`
- `examples/2026-08-14_大井_x_v2.json`
- `examples/2026-08-14_大井_audit_v2.json`

Git commit / pushは実施していません。Prediction Snapshot変更は0です。
