from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import streamlit as st

from .dashboard_cards import (
    RaceCard,
    filtered_summary_counts,
    format_strategy_score,
    prepare_race_cards,
    today_best_five,
)
from .prediction_history import build_prediction_history_zip, history_zip_file_name
from .summary_loader import summary_date, summary_venues


DETAIL_PAGE = "pages/3_Race_Detail.py"


def apply_mobile_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {max-width: 760px; padding-top: 1rem; padding-bottom: 3rem;}
        h1 {font-size: 1.8rem !important;}
        h2 {font-size: 1.45rem !important;}
        h3 {font-size: 1.2rem !important;}
        div[data-testid="stMetric"] {background: #f7f8fa; border-radius: 12px; padding: .55rem;}
        div[data-testid="stMetricValue"] {font-size: 1.35rem;}
        .race-meta {color: #5f6368; font-size: .88rem; margin-top: -.4rem;}
        .race-ticket {font-size: 1.04rem; font-weight: 700; margin: .45rem 0;}
        @media (max-width: 640px) {
          .block-container {padding-left: .75rem; padding-right: .75rem;}
          div[data-testid="stHorizontalBlock"] {gap: .45rem;}
          button[kind="secondary"], button[kind="primary"] {min-height: 2.75rem;}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_best_five(
    summaries: Iterable[tuple[str, Mapping[str, Any], str | Path]],
) -> None:
    st.subheader("今日のBEST5")
    summary_list = list(summaries)
    venues = _combined_venues(summary for _source, summary, _analysis_dir in summary_list)
    selected_venue = ""
    if venues:
        selected = st.selectbox("BEST5会場", ("すべて", *venues), key="best5_venue_filter")
        selected_venue = "" if selected == "すべて" else selected

    cards = today_best_five(summary_list, venue=selected_venue)
    if not cards:
        st.info("今日のBUY対象データはありません。")
        return
    for rank, card in enumerate(cards, start=1):
        render_buy_card(card, key_prefix=f"best-{rank}", rank=rank)


def render_summary_dashboard(
    title: str,
    caption: str,
    summary: Mapping[str, Any],
    analysis_dir: str | Path,
    *,
    source: str,
    show_heading: bool = True,
) -> None:
    if show_heading:
        st.divider()
        st.subheader(title)
        st.caption(caption)
    date = summary_date(summary)
    if date:
        st.write(f"対象日：{date.replace('-', '/')}")

    venues = summary_venues(summary)
    selected_venue = ""
    if venues:
        selected = st.selectbox("開催場", ("すべて", *venues), key=f"{source}-venue-filter")
        selected_venue = "" if selected == "すべて" else selected
    sort_label = st.radio("表示順", ("おすすめ順", "レース順"), horizontal=True, key=f"{source}-sort-mode")
    sort_mode = "race" if sort_label == "レース順" else "score"

    buy_count, hold_count, skip_count = filtered_summary_counts(summary, selected_venue)
    buy_col, hold_col, skip_col = st.columns(3)
    buy_col.metric("BUY", f"{buy_count}R")
    hold_col.metric("HOLD", f"{hold_count}R")
    skip_col.metric("SKIP", f"{skip_count}R")

    if venues:
        st.caption("開催場：" + " / ".join(venues))
    st.download_button(
        "本日の予想履歴を一括ダウンロード",
        data=build_prediction_history_zip(summary, analysis_dir, source=source),
        file_name=history_zip_file_name(summary, source=source),
        mime="application/zip",
        use_container_width=True,
    )

    best_cards = prepare_race_cards(summary, analysis_dir, source=source, decision="buy", venue=selected_venue)[:5]
    if best_cards:
        st.markdown("#### BEST5")
        for index, card in enumerate(best_cards, start=1):
            render_hold_row(card, key_prefix=f"{source}-best-{index}")

    buy_cards = prepare_race_cards(summary, analysis_dir, source=source, decision="buy", venue=selected_venue, sort_mode=sort_mode)
    st.markdown("#### BUY")
    if not buy_cards:
        st.caption("BUY対象レースはありません。")
    for index, card in enumerate(buy_cards):
        render_buy_card(card, key_prefix=f"{source}-buy-{index}")

    hold_cards = prepare_race_cards(summary, analysis_dir, source=source, decision="hold", venue=selected_venue, sort_mode=sort_mode)
    with st.expander(f"HOLD {hold_count}R", expanded=False):
        if not hold_cards:
            st.caption("HOLD対象レースはありません。")
        for index, card in enumerate(hold_cards):
            render_hold_row(card, key_prefix=f"{source}-hold-{index}")

    st.caption(f"SKIP {skip_count}R（件数のみ表示）")


def render_buy_card(card: RaceCard, *, key_prefix: str, rank: int | None = None) -> None:
    with st.container(border=True):
        prefix = f"#{rank} " if rank is not None else ""
        st.markdown(f"### {prefix}{card.venue} {card.race_number}")
        st.caption(
            f"{card.source} ・ 発走 {card.post_time} ・ "
            f"strategy_score {format_strategy_score(card.strategy_score)}"
        )
        st.markdown(f"**{card.ticket}**")
        roi_col, rank_col = st.columns(2)
        roi_col.metric("期待回収率", card.roi)
        rank_col.metric("投資ランク", card.investment_rank)

        st.markdown("**買う理由**")
        st.write(f"条件一致：{card.condition_match}")
        st.write(f"採用戦略：{card.adopted_strategy}")
        st.write(f"期待回収率：{card.roi}")
        st.write(f"strategy_score：{format_strategy_score(card.strategy_score)}")
        if card.buy_reasons:
            st.caption("BUYになった理由")
            for reason in card.buy_reasons:
                st.write(f"- {reason}")
        else:
            st.caption("BUYになった理由：Summaryに記録がありません。")

        if card.horses:
            for horse in card.horses:
                trust = f"  \n{horse.trust_summary}" if horse.trust_summary else ""
                st.markdown(
                    f"**{horse.mark} {horse.number} {horse.name}**  \n"
                    f"AI点 **{horse.ai_score}** ・ 能力評価 **{horse.ability}**{trust}"
                )
        else:
            st.caption("対象馬データは詳細JSONまたはSummaryにありません。")

        _render_detail_button(card, key=f"{key_prefix}-{card.race_id or 'race'}")


def render_hold_row(card: RaceCard, *, key_prefix: str) -> None:
    with st.container(border=True):
        st.markdown(f"**{card.venue} {card.race_number}**　発走 {card.post_time}")
        st.caption(
            f"{card.ticket} ・ strategy_score {format_strategy_score(card.strategy_score)} ・ 期待回収率 {card.roi}"
        )
        _render_detail_button(card, key=f"{key_prefix}-{card.race_id or 'race'}")


def _render_detail_button(card: RaceCard, *, key: str) -> None:
    if not card.detail_available:
        st.button("詳細データなし", key=key, disabled=True, use_container_width=True)
        return
    if st.button("詳細を見る", key=key, use_container_width=True):
        st.session_state.dashboard_detail_path = card.detail_path
        st.session_state.dashboard_detail_title = f"{card.venue} {card.race_number}"
        st.session_state.dashboard_detail_source = card.source
        st.switch_page(DETAIL_PAGE)


def _combined_venues(summaries: Iterable[Mapping[str, Any]]) -> tuple[str, ...]:
    values: list[str] = []
    for summary in summaries:
        values.extend(summary_venues(summary))
    return tuple(dict.fromkeys(value for value in values if value))
