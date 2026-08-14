from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping

import streamlit as st

from publisher.content import group_by_venue, note_article, race_number, short_commentary, x_post
from publisher.snapshot import LoadedPrediction, load_prediction
from publisher.state import DuplicatePostError, dump_state, load_state, new_state, record_post


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _event_date(snapshot: Mapping[str, Any]) -> str:
    dates = [_text(race.get("date")) for race in snapshot.get("races") or [] if _text(race.get("date"))]
    return dates[0] if dates else _text(snapshot.get("scope", {}).get("dates", [""])[0])


def _mode_label(snapshot: Mapping[str, Any]) -> str:
    modes = sorted({_text(race.get("race_mode")).upper() for race in snapshot.get("races") or []})
    return " / ".join(mode for mode in modes if mode)


def _get_loaded(data: bytes) -> LoadedPrediction:
    digest_key = f"loaded:{hash(data)}:{len(data)}"
    if st.session_state.get("loaded_key") != digest_key:
        st.session_state.loaded_prediction = load_prediction(data)
        st.session_state.loaded_key = digest_key
        st.session_state.publisher_state = new_state(st.session_state.loaded_prediction.source_info)
    return st.session_state.loaded_prediction


def _ensure_drafts(state: dict[str, Any], snapshot: Mapping[str, Any], *, exclude_debut: bool) -> dict[str, list[dict[str, Any]]]:
    grouped = group_by_venue(snapshot, exclude_debut=exclude_debut)
    race_date = _event_date(snapshot)
    for venue, races in grouped.items():
        state.setdefault("note_drafts", {}).setdefault(venue, note_article(venue, races, race_date))
        for race in races:
            race_id = _text(race.get("race_id"))
            url = state.setdefault("note_urls", {}).get(venue, "")
            state.setdefault("x_drafts", {}).setdefault(race_id, x_post(race, url))
            state.setdefault("x_targets", {}).setdefault(race_id, True)
            state.setdefault("race_status", {}).setdefault(race_id, "原稿生成済")
    return grouped


def _race_preview(race: Mapping[str, Any]) -> None:
    st.subheader(f"{_text(race.get('venue'))}{race_number(race)} {_text(race.get('race_name'))}")
    facts = [
        _text(race.get("surface")),
        f"{_text(race.get('distance'))}m" if _text(race.get("distance")) else "",
        f"{_text(race.get('head_count'))}頭" if _text(race.get("head_count")) else "",
        f"race_id {_text(race.get('race_id'))}",
    ]
    st.caption("｜".join(item for item in facts if item))
    st.info(short_commentary(race))
    rows = []
    for horse in race.get("horses") or []:
        if _text(horse.get("mark")):
            rows.append(
                {
                    "印": horse.get("mark"),
                    "馬": f"{_text(horse.get('horse_no'))} {_text(horse.get('horse_name'))}",
                    "能力": horse.get("ability_band"),
                    "能力順位": horse.get("ability_rank"),
                    "今回評価": horse.get("current_evaluation_rank"),
                    "オッズ": horse.get("odds_at_prediction"),
                    "妙味": "あり" if horse.get("value_signal") else "",
                }
            )
    if rows:
        st.dataframe(rows, hide_index=True, use_container_width=True)
    else:
        st.warning("保存Snapshotに印がありません。印を推測生成しません。")


def _note_tab(state: dict[str, Any], venue: str, races: list[dict[str, Any]], race_date: str) -> None:
    url = st.text_input("この会場のnote URL", value=state["note_urls"].get(venue, ""), key=f"note-url-{venue}")
    if url != state["note_urls"].get(venue, ""):
        state["note_urls"][venue] = url
        for race in races:
            race_id = _text(race.get("race_id"))
            state["x_drafts"][race_id] = x_post(race, url)
            state["race_status"][race_id] = "X投稿準備完了" if url else "原稿生成済"
    draft = st.text_area("会場別note原稿（編集可）", value=state["note_drafts"].get(venue, ""), height=520, key=f"note-{venue}")
    state["note_drafts"][venue] = draft
    st.download_button(
        "Markdownを保存",
        draft.encode("utf-8"),
        file_name=f"{race_date}_{venue}_note.md",
        mime="text/markdown",
        use_container_width=True,
    )
    st.download_button(
        "プレーンテキストを保存",
        draft.encode("utf-8"),
        file_name=f"{race_date}_{venue}_note.txt",
        mime="text/plain",
        use_container_width=True,
    )


def _x_tab(state: dict[str, Any], venue: str, races: list[dict[str, Any]]) -> None:
    col1, col2 = st.columns(2)
    if col1.button("全選択", use_container_width=True, key=f"all-{venue}"):
        for race in races:
            state["x_targets"][_text(race.get("race_id"))] = True
    if col2.button("全解除", use_container_width=True, key=f"none-{venue}"):
        for race in races:
            state["x_targets"][_text(race.get("race_id"))] = False
    state["x_account"] = st.text_input("Xアカウント（将来の二重投稿照合用）", value=state.get("x_account", ""))
    st.caption("Ver.1はプレビューと手動投稿用です。Xへの自動送信は無効です。")
    for race in races:
        race_id = _text(race.get("race_id"))
        header = f"{venue}{race_number(race)}"
        state["x_targets"][race_id] = st.checkbox(
            f"{header}を投稿対象にする",
            value=bool(state["x_targets"].get(race_id, True)),
            key=f"target-{race_id}",
        )
        draft = st.text_area(
            f"{header} X投稿文（編集可）",
            value=state["x_drafts"].get(race_id, ""),
            height=230,
            key=f"x-{race_id}",
        )
        state["x_drafts"][race_id] = draft
        schedule = st.text_input(
            f"{header} 予約時刻（任意・例 09:30）",
            value=state.setdefault("schedules", {}).get(race_id, ""),
            key=f"schedule-{race_id}",
        )
        state["schedules"][race_id] = schedule
        posted_col, failed_col = st.columns(2)
        if posted_col.button("手動投稿済として記録", key=f"posted-{race_id}", use_container_width=True):
            try:
                record_post(state, race_id, state.get("x_account", ""), "x", "投稿済")
                st.success(f"{header}を投稿済として記録しました。")
            except DuplicatePostError as exc:
                st.warning(str(exc))
        if failed_col.button("投稿失敗を記録", key=f"failed-{race_id}", use_container_width=True):
            record_post(state, race_id, state.get("x_account", ""), "x", "投稿失敗", message="手動記録")
            st.warning(f"{header}の投稿失敗を記録しました。")


def _history_tab(state: dict[str, Any], races: list[dict[str, Any]]) -> None:
    rows = []
    for race in races:
        race_id = _text(race.get("race_id"))
        rows.append(
            {
                "race_id": race_id,
                "レース": f"{_text(race.get('venue'))}{race_number(race)}",
                "状態": state.get("race_status", {}).get(race_id, "未生成"),
                "X対象": bool(state.get("x_targets", {}).get(race_id)),
                "予約": state.get("schedules", {}).get(race_id, ""),
            }
        )
    st.dataframe(rows, hide_index=True, use_container_width=True)
    history = state.get("post_history") or []
    if history:
        st.dataframe(history, hide_index=True, use_container_width=True)
    else:
        st.caption("投稿履歴はまだありません。")


def main() -> None:
    st.set_page_config(page_title="KEIBA LAB Publisher", page_icon="📝", layout="wide")
    st.title("KEIBA LAB Publisher")
    st.caption("保存済みPrediction Snapshotを再計算せず、note/X原稿へ変換します。")
    uploaded = st.file_uploader("予想ファイルを開く", type=["keiba"])
    if uploaded is None:
        st.info("Keiba AI Dashboardで保存した .keiba ファイルを選択してください。")
        return
    try:
        loaded = _get_loaded(uploaded.getvalue())
    except Exception as exc:
        st.error(f".keibaを開けません: {exc}")
        return
    snapshot = loaded.snapshot
    st.success(f"{_event_date(snapshot)}｜{_mode_label(snapshot)}｜{len(snapshot.get('races') or [])}レースを復元しました。再予想は行っていません。")

    state: dict[str, Any] = st.session_state.publisher_state
    state_file = st.file_uploader("Publisher作業状態を開く（任意）", type=["json"], key="state-upload")
    if state_file is not None and st.session_state.get("state-file") != state_file.name:
        try:
            state = load_state(state_file.getvalue(), source_info=loaded.source_info)
            st.session_state.publisher_state = state
            st.session_state["state-file"] = state_file.name
            st.success("Publisher作業状態を復元しました。")
        except Exception as exc:
            st.error(f"Publisher保存ファイルを開けません: {exc}")

    exclude = st.toggle("新馬戦を除外", value=bool(state.get("exclude_debut", True)))
    if exclude != state.get("exclude_debut"):
        state["exclude_debut"] = exclude
        state["note_drafts"] = {}
        state["x_drafts"] = {}
    grouped = _ensure_drafts(state, snapshot, exclude_debut=exclude)
    if not grouped:
        st.warning("掲載対象レースがありません。")
        return

    venues = list(grouped)
    venue = st.radio("会場", venues, horizontal=True)
    races = grouped[venue]
    labels = [race_number(race) for race in races]
    selected_label = st.radio("レース", labels, horizontal=True)
    selected = races[labels.index(selected_label)]
    _race_preview(selected)

    tab_note, tab_x, tab_history = st.tabs(["note原稿", "X投稿", "投稿履歴"])
    with tab_note:
        _note_tab(state, venue, races, _event_date(snapshot))
    with tab_x:
        _x_tab(state, venue, races)
    with tab_history:
        _history_tab(state, races)

    loaded.assert_unchanged()
    state_name = f"{_event_date(snapshot)}_publisher.json"
    st.download_button(
        "Publisher作業状態を保存",
        dump_state(state),
        file_name=state_name,
        mime="application/json",
        use_container_width=True,
    )
    with st.expander("読込監査情報"):
        st.json(loaded.source_info)
        st.write("Prediction Snapshotは読み取り専用です。能力値・順位・能力帯・印・妙味を変更していません。")


if __name__ == "__main__":
    main()
