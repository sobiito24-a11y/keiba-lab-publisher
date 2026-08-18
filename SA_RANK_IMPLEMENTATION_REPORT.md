# KEIBA LAB Publisher 注目度S/A 実装レポート

## 対象

Publisher Ver.3だけを変更した。Keiba AI Mobile、Keiba AI Dashboard、Prediction Snapshot、`.keiba` 内部形式、予想・印・能力・今回評価・買い目ロジックは変更していない。

## 実装

- `publisher/confidence.py` を追加し、JRA/NAR固定基準を純粋な読み取り処理として実装。
- 内部ランクは `SS / S / A / 対象外`、公開ランクは `S / A / 非表示`。
- `show_public_ss=false` のため内部SSは公開Sへマッピング。
- 騎手の継続/乗替は構造化された `jockey_change` を優先。
- `plus_materials` 内の説明用「印」「今回順位」は除外。
- note各レースにS/A表示。Sには説明と保存済み注意材料を追加。
- note冒頭にS一覧を追加。Aは各レース欄のみ。
- XはSだけ「🐴 注目度S」を表示し、280文字以内へ既存短縮処理を維持。
- 忙しい日/休日の候補は内部SS→S→A→対象外の順。ただし自動確定しない。
- Publisher stateへ `show_public_ss` とrace_id別 `confidence_ranks` を追加。旧state v3も既定値補完で読める。
- UIの公開状況表へ小さな「注目度」列と、検証中の注記を追加。

## 実データ確認

### JRA 2026-08-16（32R）

- 内部S：2R
- A：20R
- 対象外：10R
- 例：新潟1R シュネーバレン＝公開S
- 例：中京1R ラホーヤアイズ＝公開A
- 例：中京7R リリージョワ＝非表示

### NAR 2026-08-17（20R）

- 内部SS：1R（大井4R パープルフォッグ、公開S）
- S：4R
- A：15R
- 例：大井2R ニチリン＝公開S
- 例：大井1R スマートビアンカ＝公開A

全52Rのnote/X生成、X 280文字以内、Snapshot署名不変、state往復を確認した。

## テスト

- 新規 `tests/test_confidence.py` にJRA/NAR各4分類、SS公開S、note、X、運用モード、state、Snapshot不変のテストを追加。
- 実行環境にpytestがなく、外部依存追加も許可されなかったためpytestコマンドは未実行。
- 代替としてPythonコンパイル、28項目の直接assert、実JRA/NAR 52Rの生成・不変監査を実施。

## 監査

- Prediction Snapshot変更件数：0
- 予想再計算回数：0
- 能力値変更件数：0
- 今回評価変更件数：0
- 印変更件数：0
- 元 `.keiba` 変更件数：0
- commit：0
- push：0

