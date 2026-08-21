from __future__ import annotations

import hashlib
from datetime import datetime, timedelta
from typing import Any, Iterable, Mapping

from .content import text


READING_STATUSES = ("未投稿", "下書き済み", "公開済み")
DEFAULT_DUPLICATE_WINDOW_DAYS = 30
DEFAULT_READING_TAGS = ("KEIBALAB", "AI競馬予想", "競馬予想")


class ReadingArticleError(ValueError):
    pass


class DuplicateReadingThemeError(ReadingArticleError):
    pass


READING_THEMES: dict[str, dict[str, str]] = {
    "能力1位と◎が違うのはなぜ？": {
        "title": "能力1位と◎が違うのはなぜ？ KEIBA LABの予想の見方",
        "body": """## 能力1位と◎が違うのはなぜ？

KEIBA LABでは、能力値だけで◎を決めていません。

能力値は、これまでの走りや指数、相手関係などから見た「その馬が持っている力」を見るための軸です。

一方で◎は、そのレースで走る条件まで含めた最終判断です。距離、コース、展開、近走の流れ、騎手、調教、相手関係などを重ねて見たときに、能力1位ではない馬を本命にすることがあります。

つまり、能力1位は「地力の評価」、◎は「今回の条件で狙う中心馬」という位置づけです。

能力1位と◎が違うレースは、能力だけで決めない分、どの条件を重く見たのかを考える材料になります。

KEIBA LABでは、予想をレース前の時点で保存し、あとから見返せる形で検証しています。数値で確認できない実績は記事内に載せず、保存済みの材料をもとに考え方を整理していきます。""",
    },
    "KEIBA LABの「能力値」とは？": {
        "title": "KEIBA LABの「能力値」とは？ 予想を見る前に知っておきたい基準",
        "body": """## KEIBA LABの「能力値」とは？

KEIBA LABの能力値は、各馬を同じレース内で比較するための基準です。

過去の走り、指数、相手関係、距離やコースへの対応など、保存済みの材料をもとに全頭を並べて見ます。

ただし、能力値はそのまま買い目を決める数字ではありません。能力が高くても、今回の条件が合わない可能性はあります。逆に、能力順位が少し下でも、条件がかみ合えば評価を上げることがあります。

大事なのは、能力値を「馬の土台」として見ることです。

◎や注目度を見る前に能力順位を確認すると、AIがどの馬を地力上位と見ているのかが分かりやすくなります。

この記事では考え方を説明しています。的中率や回収率などの実績値は、根拠となる検証データを確認できる場合だけ掲載します。""",
    },
    "今回評価とは？": {
        "title": "KEIBA LABの「今回評価」とは？ 能力値との違い",
        "body": """## 今回評価とは？

今回評価は、そのレースで走る条件を含めた総合評価です。

能力値が「その馬の地力」を見るための軸だとすれば、今回評価は「今日この条件でどうか」を見るための軸です。

距離、コース、馬場、展開、近走の流れ、騎手や調教など、保存済みの材料を重ねて、今回のレースで評価し直します。

能力値と今回評価がそろって高い馬は分かりやすい中心候補です。一方で、能力値は高いのに今回評価が伸びない馬、能力値以上に今回評価が上がる馬もいます。

そうしたズレを見ることで、単純な能力順では見えないレースのポイントが出てきます。

KEIBA LABでは、予想時点の評価を保存し、あとから振り返れる形にしています。根拠のない成績数値は使わず、保存済みの材料をもとに説明します。""",
    },
    "注目度Sはどう決めている？": {
        "title": "注目度Sはどう決めている？ KEIBA LABの補助評価について",
        "body": """## 注目度Sはどう決めている？

KEIBA LABの注目度Sは、買い推奨や的中保証ではありません。

保存済みのPrediction Snapshotにある材料を確認し、◎の能力評価、今回評価、条件材料、マイナス材料などが複数そろっているかを見る補助ラベルです。

注目度Sは「AIが見た材料が比較的そろっているレース」として扱います。だからこそ、Sが付いていても必ず当たるわけではありませんし、Sではないレースにも面白い材料はあります。

使い方としては、全レースの中でどこを重点的に読むかを決める目印に近いです。

KEIBA LABでは、注目度も保存済みの予想材料から表示します。あとから都合よく変えたり、結果に合わせて評価を書き換えたりしないことを大事にしています。""",
    },
    "妙味馬とは？": {
        "title": "KEIBA LABの「妙味馬」とは？ 買い推奨ではなく比較材料",
        "body": """## 妙味馬とは？

KEIBA LABで表示する妙味馬は、買い推奨そのものではありません。

保存済みの評価材料と市場評価を比べたときに、見直す余地がありそうな馬を「相手候補として注目したい」という意味で出しています。

本命とは違い、妙味馬は軸にするための印ではありません。能力、今回評価、展開、距離やコース適性などの材料を見て、人気や評価とのズレを考えるための入口です。

予想を見るときは、◎を中心にしつつ、妙味馬がどの材料で拾われているのかを確認すると読みやすくなります。

実績値や回収率を語る場合は、保存済みの検証データを確認できるときだけ掲載します。根拠がない数字は作りません。""",
    },
    "JRAと地方競馬で評価方法が違う理由": {
        "title": "JRAと地方競馬で評価方法が違う理由",
        "body": """## JRAと地方競馬で評価方法が違う理由

JRAと地方競馬では、使える材料やレースの性質が同じではありません。

JRAでは調教、コース替わり、距離適性、相手関係など、比較しやすい材料が多くあります。

地方競馬では、競馬場ごとの特徴、出走間隔、同じ相手との再戦、コース経験などが大きな材料になることがあります。調教データの扱いもJRAと同じにはできません。

そのため、KEIBA LABではJRAと地方競馬を同じ物差しだけで評価しないようにしています。

大事なのは、どちらが上という話ではなく、それぞれのレースで確認できる材料を無理なく使うことです。

予想記事では、保存済みの材料をもとに、能力値と今回評価の違いが読み取れるようにしています。""",
    },
    "実際の予想結果を検証して分かったこと": {
        "title": "予想結果の検証で大事にしていること",
        "body": """## 予想結果の検証で大事にしていること

KEIBA LABでは、予想をレース前の時点で保存し、あとから振り返れる形にしています。

検証で大事なのは、当たったか外れたかだけを見ることではありません。

◎の選び方、能力1位とのズレ、今回評価の理由、妙味馬の拾い方、注目度S/Aの出方などを、結果と照らし合わせて確認します。

ただし、根拠となる検証データが確認できない段階で、的中率や回収率の数字を出すことはしません。

数字を出すなら、対象期間、対象レース、集計条件が確認できる状態にしてからです。

この記事では、まず検証の見方を整理しています。結果に合わせて予想を変えず、保存済みの予想をそのまま見返すことを大切にしています。""",
    },
    "AI競馬予想をどう検証しているか": {
        "title": "AI競馬予想をどう検証しているか KEIBA LABの振り返り方",
        "body": """## AI競馬予想をどう検証しているか

AI競馬予想は、出した予想を残しておくことが大切です。

結果を見たあとに理由を足したり、印を変えたりすると、何が良くて何が悪かったのか分からなくなります。

KEIBA LABでは、Prediction Snapshotとして予想時点のデータを保存し、その内容をもとにnote本文を作ります。

検証では、◎の成績だけでなく、能力1位、今回評価1位、注目度S/A、妙味馬などを分けて見ます。

ただし、確認できる集計データがない数字は記事に載せません。的中率や回収率を出す場合は、保存済みの検証データを根拠にします。

まずは予想を変えずに残すこと。そのうえで、どの評価がどの場面で機能したのかを丁寧に見ていきます。""",
    },
}


def reading_theme_options() -> tuple[str, ...]:
    return tuple(READING_THEMES)


def generate_reading_article(
    theme: str | None = None,
    *,
    existing_articles: Iterable[Mapping[str, Any]] = (),
    now: datetime | None = None,
    duplicate_window_days: int = DEFAULT_DUPLICATE_WINDOW_DAYS,
) -> dict[str, Any]:
    created = now or datetime.now()
    chosen = theme or next_available_theme(existing_articles, now=created, duplicate_window_days=duplicate_window_days)
    if chosen not in READING_THEMES:
        raise ReadingArticleError("未対応の記事テーマです。")
    ensure_theme_not_recent(chosen, existing_articles, now=created, duplicate_window_days=duplicate_window_days)
    template = READING_THEMES[chosen]
    created_at = created.isoformat(timespec="seconds")
    return {
        "id": _article_id(chosen, created_at),
        "theme": chosen,
        "title": template["title"],
        "body": template["body"].rstrip() + "\n",
        "tags": list(DEFAULT_READING_TAGS),
        "status": "未投稿",
        "created_at": created_at,
    }


def generate_reading_candidates(
    count: int = 3,
    *,
    existing_articles: Iterable[Mapping[str, Any]] = (),
    now: datetime | None = None,
    duplicate_window_days: int = DEFAULT_DUPLICATE_WINDOW_DAYS,
) -> list[dict[str, Any]]:
    created = now or datetime.now()
    articles = list(existing_articles)
    result = []
    for theme_name in reading_theme_options():
        if len(result) >= count:
            break
        if is_theme_recent(theme_name, articles, now=created, duplicate_window_days=duplicate_window_days):
            continue
        article = generate_reading_article(theme_name, existing_articles=articles, now=created, duplicate_window_days=duplicate_window_days)
        result.append(article)
        articles.append(article)
    return result


def ensure_theme_not_recent(
    theme: str,
    articles: Iterable[Mapping[str, Any]],
    *,
    now: datetime | None = None,
    duplicate_window_days: int = DEFAULT_DUPLICATE_WINDOW_DAYS,
) -> None:
    if is_theme_recent(theme, articles, now=now, duplicate_window_days=duplicate_window_days):
        raise DuplicateReadingThemeError(f"直近{duplicate_window_days}日以内に同じテーマの記事があります。")


def is_theme_recent(
    theme: str,
    articles: Iterable[Mapping[str, Any]],
    *,
    now: datetime | None = None,
    duplicate_window_days: int = DEFAULT_DUPLICATE_WINDOW_DAYS,
) -> bool:
    current = now or datetime.now()
    threshold = current - timedelta(days=duplicate_window_days)
    for article in articles:
        if text(article.get("theme")) != theme:
            continue
        created_at = _parse_datetime(text(article.get("created_at")))
        if created_at and created_at >= threshold:
            return True
    return False


def next_available_theme(
    articles: Iterable[Mapping[str, Any]],
    *,
    now: datetime | None = None,
    duplicate_window_days: int = DEFAULT_DUPLICATE_WINDOW_DAYS,
) -> str:
    for theme in reading_theme_options():
        if not is_theme_recent(theme, articles, now=now, duplicate_window_days=duplicate_window_days):
            return theme
    raise DuplicateReadingThemeError(f"直近{duplicate_window_days}日以内に生成済みでないテーマがありません。")


def update_reading_article(article: Mapping[str, Any], *, title: str, body: str, tags: Iterable[str], status: str) -> dict[str, Any]:
    if status not in READING_STATUSES:
        raise ReadingArticleError("読み物記事のステータスが不正です。")
    updated = dict(article)
    updated["title"] = text(title)
    updated["body"] = text(body).rstrip() + "\n"
    updated["tags"] = [text(tag) for tag in tags if text(tag)]
    updated["status"] = status
    if not updated["title"] or not updated["body"].strip():
        raise ReadingArticleError("タイトルと本文は空にできません。")
    return updated


def _article_id(theme: str, created_at: str) -> str:
    return hashlib.sha1(f"{theme}|{created_at}".encode("utf-8")).hexdigest()[:12]


def _parse_datetime(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None
