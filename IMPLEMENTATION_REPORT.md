# KEIBA LAB Publisher 実装報告

## 探索結果

既存レース考察ロジックは **B：一部再利用可能** でした。

- `.keiba` 読込・schema v1・SHA-256・重複race_id検証はDashboardの `core/prediction_snapshot.py` を完全コピーで再利用。
- 保存済み `prediction_result.ai_race_review` と `mobile_snapshot.market_compare.race_summary` はSnapshot内の既存考察データとして参照可能。
- 既存の動的レース考察生成関数は再予想相当の再生成を避けるためPublisherから呼ばない。
- Publisherのnote/X短文化は、保存済み馬データを読む決定的なテンプレートとして新設。
- 騎手正規化はDashboard `market_compare.py` の既存正規化を優先し、明確な2文字省略だけPublisher表示層で安全に補完。

## 構成

- `app.py`: Streamlit UI
- `publisher/snapshot.py`: Dashboard正本ローダーの呼出、Snapshot不変fingerprint
- `publisher/content.py`: 会場別note・レース別X・新馬戦明記判定
- `publisher/jockey.py`: 表示専用の騎手同一性判定
- `publisher/state.py`: Publisher JSON、URL、対象、予約、状態、履歴、二重投稿防止
- `publisher/posting.py`: 無効化された将来のX公式API境界
- `shared_dashboard_core/core`: Snapshot正本読込のためのDashboard core完全コピー
- `examples`: 実Snapshotから生成した3会場noteと32レースX原稿
- `tests`: 回帰テストと実 `.keiba` fixture

## 実データ確認

添付 `2026-08-09_JRA_all_venues.keiba` をSHA-256検証付きで読み込みました。

- JRA 3会場
- 中京10レース
- 新潟11レース
- 札幌11レース
- 合計32レース
- `新馬` / `メイクデビュー` の明記による除外: 0レース
- note原稿: 3件
- X原稿: 32件、全件固有文

NARの実 `.keiba` は添付されていないため、実データを装っていません。NARはschema v1に適合する有効なNAR Snapshot fixtureで、正本ローダー・大井会場・note/X生成を確認しました。

## 不変性

Publisher読込・原稿生成・騎手表示正規化・作業状態保存の前後で、次を含む予想事実fingerprintが一致しました。

- 能力値
- 能力順位
- 能力帯
- AI今回評価順位
- ◎○▲△☆
- 妙味あり

Publisherソースは `predict_jra` / `predict_nar` / `apply_prediction_logic` を呼びません。Publisher JSONはPrediction Snapshot本体を含まず、元ファイル・Snapshot・予想事実のSHA-256識別情報だけを保持します。

## テスト

```text
36 passed in 11.29s
```

Streamlitヘッドレス起動も確認済みです。Mobile/Dashboard本体のファイル変更、commit、pushはいずれも0件です。

