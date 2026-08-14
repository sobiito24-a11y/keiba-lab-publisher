from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from typing import Any

import streamlit as st

from .dashboard_batch import BatchPredictionError, UploadedSource, predict_uploaded_sources
from .models import PredictionResult
from .prediction_snapshot import (
    KeibaSnapshotError,
    keiba_bytes,
    keiba_file_name,
    load_keiba,
    restore_prediction_result,
    subset_event_snapshot,
    update_user_selection,
)


RenderMobileResult = Callable[[PredictionResult], None]


DASHBOARD_CSS = """
<style>
  .block-container {max-width: 1180px; padding-top: 1rem; padding-bottom: 3rem;}
  .ka-dashboard-intro {color:#475467; margin-top:-.35rem; margin-bottom:1rem;}
  .ka-snapshot-note {padding:.65rem .8rem; border-radius:8px; background:#f0fdf4;
    border:1px solid #bbf7d0; color:#166534; margin:.5rem 0 1rem;}
  @media (max-width: 640px) {
    .block-container {padding-left:.7rem; padding-right:.7rem;}
    button[kind="secondary"], button[kind="primary"] {min-height:2.7rem;}
  }
</style>
"""


def render_dashboard(render_mobile_result: RenderMobileResult) -> None:
    st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)
    _init_state()
    st.title("Keiba AI Dashboard")
    st.markdown(
        '<div class="ka-dashboard-intro">Mobileと同じ予想を全レース一括作成し、開催単位で保存・復元します。</div>',
        unsafe_allow_html=True,
    )

    action = st.radio(
        "操作",
        ("新規一括予想", "保存した予想を開く"),
        horizontal=True,
        label_visibility="collapsed",
        key="dashboard_action",
    )
    if action == "新規一括予想":
        _render_new_prediction()
    else:
        _render_saved_prediction_open()

    event = st.session_state.get("dashboard_event_snapshot")
    if isinstance(event, Mapping) and event.get("races"):
        st.divider()
        _render_event(event, render_mobile_result)


def _init_state() -> None:
    st.session_state.setdefault("dashboard_event_snapshot", None)
    st.session_state.setdefault("dashboard_batch_report", None)
    st.session_state.setdefault("dashboard_selected_race_id", "")
    st.session_state.setdefault("dashboard_rendered_race_id", "")
    st.session_state.setdefault("dashboard_loaded_signature", "")


def _render_new_prediction() -> None:
    st.subheader("新規一括予想")
    st.caption(
        "JRA/NARのHTMLを複数選択、または収集ZIPを複数投入できます。"
        "ファイル名ではなくcanonical・race_id・HTML内容で自動分類します。"
    )
    uploads = st.file_uploader(
        "HTML / ZIPを投入",
        type=["html", "htm", "zip"],
        accept_multiple_files=True,
        key="dashboard_batch_uploads",
    )
    if not st.button(
        "一括予想データ作成",
        type="primary",
        use_container_width=True,
        disabled=not uploads,
    ):
        _render_last_batch_report()
        return
    progress = st.progress(0)
    status = st.empty()

    def on_progress(current: int, total: int, label: str) -> None:
        progress.progress(current / max(1, total))
        status.write(f"{current}/{total}R 予想中：{label}")

    try:
        sources = [UploadedSource(item.name, item.getvalue()) for item in uploads or []]
        with st.spinner("Mobile予想をレース単位で実行しています…"):
            report = predict_uploaded_sources(sources, progress=on_progress)
    except BatchPredictionError as exc:
        progress.empty()
        status.empty()
        st.error(str(exc))
        return
    progress.progress(1.0)
    status.write("一括予想データ作成が完了しました。")
    st.session_state.dashboard_event_snapshot = report.event_snapshot
    st.session_state.dashboard_batch_report = report
    st.session_state.dashboard_selected_race_id = ""
    st.session_state.dashboard_rendered_race_id = ""
    st.success(f"{report.predicted_race_count}RのPrediction Snapshotを作成しました。")
    _render_last_batch_report()


def _render_last_batch_report() -> None:
    report = st.session_state.get("dashboard_batch_report")
    if report is None:
        return
    st.caption(
        f"入力 {report.input_file_count}件 / HTML {report.html_file_count}件 / "
        f"認識 {report.recognized_file_count}件 / 予想 {report.predicted_race_count}R / "
        f"スキップ {report.skipped_race_count}R"
    )
    if report.warnings or report.errors:
        with st.expander("分類・予想の詳細", expanded=False):
            for message in report.warnings:
                st.warning(message)
            for message in report.errors:
                st.error(message)


def _render_saved_prediction_open() -> None:
    st.subheader("保存した予想を開く")
    st.caption(".keiba内のPrediction Snapshotを直接表示します。HTML再解析・予想再計算は行いません。")
    uploaded = st.file_uploader(
        ".keibaファイルを選択",
        type=["keiba"],
        accept_multiple_files=False,
        key="dashboard_saved_prediction",
    )
    if uploaded is None:
        return
    payload = uploaded.getvalue()
    signature = f"{uploaded.name}:{len(payload)}:{hashlib.sha256(payload).hexdigest()}"
    if st.session_state.dashboard_loaded_signature == signature:
        return
    try:
        event = load_keiba(payload)
    except KeibaSnapshotError as exc:
        st.error(str(exc))
        return
    st.session_state.dashboard_event_snapshot = event
    st.session_state.dashboard_batch_report = None
    st.session_state.dashboard_selected_race_id = ""
    st.session_state.dashboard_rendered_race_id = ""
    st.session_state.dashboard_loaded_signature = signature
    st.success(f"保存時点の予想 {len(event['races'])}Rを読み込みました（再計算なし）。")


def _render_event(event: Mapping[str, Any], render_mobile_result: RenderMobileResult) -> None:
    races = list(event.get("races") or [])
    selected = _render_race_navigation(races)
    if not selected:
        return
    race_id = str(selected.get("race_id") or "")
    if st.session_state.dashboard_rendered_race_id != race_id:
        _clear_mobile_selection_widgets()
        st.session_state.dashboard_rendered_race_id = race_id
    result = restore_prediction_result(selected)
    st.session_state.prediction_result = result
    st.markdown(
        '<div class="ka-snapshot-note">選択レースのMobile PredictionResultを表示中。保存ファイル読込時は再計算しません。</div>',
        unsafe_allow_html=True,
    )
    render_mobile_result(result)

    # Do not rebuild a loaded historical snapshot on ordinary display. Only
    # the explicit Mobile user-selection action is allowed to update it.
    before_selection = (
        ((selected.get("mobile_snapshot") or {}).get("market_compare") or {}).get("user_selection")
        or {}
    )
    after_selection = (
        ((getattr(result, "debug_info", {}) or {}).get("market_compare") or {}).get("user_selection")
        or {}
    )
    updated = (
        update_user_selection(event, race_id, after_selection)
        if dict(after_selection) != dict(before_selection)
        else dict(event)
    )
    st.session_state.dashboard_event_snapshot = updated
    _render_save_buttons(updated, selected)


def _render_race_navigation(races: list[Mapping[str, Any]]) -> dict[str, Any] | None:
    st.subheader("開催・レース選択")
    dates = sorted({str(item.get("date") or "日付不明") for item in races})
    race_date = st.selectbox("開催日", dates, key="dashboard_date")
    date_races = [item for item in races if str(item.get("date") or "日付不明") == race_date]

    modes = sorted({str(item.get("race_mode") or "") for item in date_races})
    mode_labels = {"jra": "JRA", "nar": "NAR"}
    mode = st.radio(
        "区分",
        modes,
        horizontal=True,
        format_func=lambda value: mode_labels.get(value, value.upper()),
        key=f"dashboard_race_mode_{race_date}",
    )
    mode_races = [item for item in date_races if str(item.get("race_mode") or "") == mode]

    venues = list(dict.fromkeys(str(item.get("venue") or "会場不明") for item in mode_races))
    venue = st.radio(
        "会場",
        venues,
        horizontal=True,
        key=f"dashboard_venue_{race_date}_{mode}",
    )
    venue_races = [item for item in mode_races if str(item.get("venue") or "会場不明") == venue]
    venue_races.sort(key=_race_number_key)

    valid_ids = {str(item.get("race_id") or "") for item in venue_races}
    current = st.session_state.dashboard_selected_race_id
    if current not in valid_ids:
        current = str(venue_races[0].get("race_id") or "") if venue_races else ""
        st.session_state.dashboard_selected_race_id = current
    columns = st.columns(4)
    for index, race in enumerate(venue_races):
        race_id = str(race.get("race_id") or "")
        label = str(race.get("race_number") or f"{index + 1}R")
        button_type = "primary" if race_id == current else "secondary"
        if columns[index % 4].button(
            label,
            key=f"dashboard-race-{race_id}",
            type=button_type,
            use_container_width=True,
        ):
            st.session_state.dashboard_selected_race_id = race_id
            current = race_id
    if not current:
        st.info("選択できるレースがありません。")
        return None
    return next((dict(item) for item in venue_races if str(item.get("race_id") or "") == current), None)


def _render_save_buttons(event: Mapping[str, Any], selected: Mapping[str, Any]) -> None:
    st.divider()
    st.subheader("Prediction Snapshot保存")
    date = str(selected.get("date") or "")
    mode = str(selected.get("race_mode") or "")
    venue = str(selected.get("venue") or "")
    venue_event = subset_event_snapshot(event, race_date=date, race_mode=mode, venue=venue)
    venue_col, all_col = st.columns(2)
    venue_col.download_button(
        "この開催を保存",
        data=keiba_bytes(venue_event),
        file_name=keiba_file_name(venue_event),
        mime="application/zip",
        use_container_width=True,
    )
    all_col.download_button(
        "全会場まとめて保存",
        data=keiba_bytes(event),
        file_name=keiba_file_name(event),
        mime="application/zip",
        use_container_width=True,
    )
    st.caption("HTMLは含めず、表示復元用Prediction Snapshotだけを保存します。")


def _clear_mobile_selection_widgets() -> None:
    for key in (
        "market_user_horses",
        "market_user_reason",
        "market_user_ticket",
        "save_market_user_selection",
    ):
        st.session_state.pop(key, None)


def _race_number_key(race: Mapping[str, Any]) -> tuple[int, str]:
    text = str(race.get("race_number") or "")
    digits = "".join(character for character in text if character.isdigit())
    return (int(digits) if digits else 999, str(race.get("race_id") or ""))
