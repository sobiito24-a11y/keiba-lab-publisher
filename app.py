from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Any, Mapping

import streamlit as st

from publisher.content import DEFAULT_NOTE_INTRO, DEFAULT_PINNED_POST, group_by_venue, note_article, note_title, race_number, short_commentary, x_post, x_weighted_length
from publisher.posting import PostingError, XApiClient, XCredentials, post_to_x, validate_post_prerequisites, verify_account
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
        generated = note_article(venue, races, race_date, intro=state.get("note_intro") or DEFAULT_NOTE_INTRO, owner_comments=state.get("owner_comments") or {})
        state.setdefault("note_generated", {})[venue] = generated
        state.setdefault("note_drafts", {}).setdefault(venue, generated)
        for race in races:
            race_id = _text(race.get("race_id"))
            url = state.setdefault("note_urls", {}).get(venue, "")
            generated_x = x_post(race, url)
            state.setdefault("x_generated", {})[race_id] = generated_x
            state.setdefault("x_drafts", {}).setdefault(race_id, generated_x)
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
    st.text_input("noteタイトル", value=note_title(venue, race_date), disabled=True)
    intro = st.text_area("固定冒頭文（設定・編集可）", value=state.get("note_intro") or DEFAULT_NOTE_INTRO, height=300, key=f"intro-{venue}")
    if intro != (state.get("note_intro") or DEFAULT_NOTE_INTRO):
        state["note_intro"] = intro
        regenerated = note_article(venue, races, race_date, intro=intro, owner_comments=state.get("owner_comments") or {})
        state["note_generated"][venue] = regenerated
        state["note_drafts"][venue] = regenerated
    url = st.text_input("この会場のnote URL", value=state["note_urls"].get(venue, ""), key=f"note-url-{venue}")
    if url != state["note_urls"].get(venue, ""):
        state["note_urls"][venue] = url
        for race in races:
            race_id = _text(race.get("race_id"))
            state["x_drafts"][race_id] = x_post(race, url)
            state["race_status"][race_id] = "X投稿準備完了" if url else "原稿生成済"
    with st.expander("【主のひとこと】をレース別に追加", expanded=False):
        changed = False
        for race in races:
            race_id = _text(race.get("race_id"))
            old = state.setdefault("owner_comments", {}).get(race_id, "")
            new = st.text_input(f"{venue}{race_number(race)}", value=old, key=f"owner-{race_id}")
            if new != old:
                state["owner_comments"][race_id] = new
                changed = True
        if changed:
            regenerated = note_article(venue, races, race_date, intro=state.get("note_intro") or DEFAULT_NOTE_INTRO, owner_comments=state["owner_comments"])
            state["note_generated"][venue] = regenerated
            state["note_drafts"][venue] = regenerated
    draft = st.text_area("会場別note完成原稿（全文コピー・編集可）", value=state["note_drafts"].get(venue, ""), height=620, key=f"note-{venue}")
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


def _credentials() -> XCredentials:
    values = dict(os.environ)
    try:
        values.update(dict(st.secrets))
    except Exception:
        pass
    return XCredentials.from_mapping(values)


def _post_one(state: dict[str, Any], race: Mapping[str, Any], client: XApiClient, account: str, *, dry_run: bool) -> None:
    race_id = _text(race.get("race_id"))
    body = state["x_drafts"][race_id]
    venue = _text(race.get("venue"))
    ensure_url = state.get("note_urls", {}).get(venue, "")
    validate_post_prerequisites(note_url=ensure_url, account=account)
    from publisher.state import ensure_can_post
    ensure_can_post(state, race_id, account, "x_race")
    post_id = post_to_x(client, body=body, dry_run=dry_run)
    if not dry_run:
        record_post(state, race_id, account, "x_race", "投稿済", body=body, post_id=post_id, note_url=ensure_url)


@st.fragment(run_every="30s")
def _scheduled_worker(state: dict[str, Any], venue: str, races: list[dict[str, Any]]) -> None:
    if not state.get("scheduler_active"):
        return
    try:
        client = XApiClient(_credentials())
        verify_account(client, state["expected_x_account"])
        now = datetime.now()
        for race in races:
            race_id = _text(race.get("race_id")); raw = state.get("schedules", {}).get(race_id, "")
            if not raw: continue
            try: due = datetime.fromisoformat(raw) if "T" in raw else datetime.combine(now.date(), datetime.strptime(raw, "%H:%M").time())
            except ValueError: continue
            if due <= now:
                try: _post_one(state, race, client, state["x_account"], dry_run=False)
                except DuplicatePostError: pass
                except Exception as exc: record_post(state, race_id, state["x_account"], "x_race", "投稿失敗", body=state["x_drafts"][race_id], note_url=state["note_urls"].get(venue, ""), message=str(exc), http_status=getattr(exc, "http_status", None), retryable=getattr(exc, "retryable", False))
    except Exception as exc:
        st.error(f"予約投稿を確認できません: {exc}")


def _x_tab(state: dict[str, Any], venue: str, races: list[dict[str, Any]]) -> None:
    col1, col2 = st.columns(2)
    if col1.button("全選択", use_container_width=True, key=f"all-{venue}"):
        for race in races:
            state["x_targets"][_text(race.get("race_id"))] = True
    if col2.button("全解除", use_container_width=True, key=f"none-{venue}"):
        for race in races:
            state["x_targets"][_text(race.get("race_id"))] = False
    state["expected_x_account"] = st.text_input("想定Xアカウント", value=state.get("expected_x_account", "keiba_lab_ai")).lstrip("@")
    client = None
    if st.button("X接続確認", use_container_width=True):
        try:
            client = XApiClient(_credentials())
            me = verify_account(client, state["expected_x_account"])
            state["x_account"] = me["username"]
            st.success(f"接続済み：@{me['username']}")
        except Exception as exc:
            st.error(str(exc))
    if state.get("x_account"):
        st.success(f"確認済みアカウント：@{state['x_account']}")
    mode = st.radio("投稿モード", ["手動投稿", "予約投稿"], horizontal=True)
    interval = st.selectbox("一括投稿の間隔", [5, 10, 15], index=[5, 10, 15].index(int(state.get("posting_interval_minutes", 10))))
    state["posting_interval_minutes"] = interval
    st.caption("認証情報は環境変数・Streamlit Secretsからのみ読み込み、画面には表示しません。まずdry-run、次に1投稿テストを行ってください。")
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
        st.caption(f"X換算 {x_weighted_length(draft)} / 280文字")
        schedule = st.text_input(
            f"{header} 予約時刻（任意・例 09:30）",
            value=state.setdefault("schedules", {}).get(race_id, ""),
            key=f"schedule-{race_id}",
        )
        state["schedules"][race_id] = schedule
        test_col, post_col = st.columns(2)
        if test_col.button("dry-run", key=f"dry-{race_id}", use_container_width=True):
            try:
                _post_one(state, race, XApiClient(_credentials()), state.get("x_account") or state["expected_x_account"], dry_run=True)
                st.success("dry-run成功（Xへは送信していません）")
            except Exception as exc: st.error(str(exc))
        confirm = st.checkbox(f"{header}の1投稿テストを許可", key=f"confirm-{race_id}")
        if post_col.button("1投稿テスト", key=f"post-{race_id}", use_container_width=True, disabled=not confirm or not state.get("x_account")):
            try:
                client = XApiClient(_credentials())
                verify_account(client, state["expected_x_account"])
                _post_one(state, race, client, state["x_account"], dry_run=False)
                st.success(f"{header}を投稿しました。")
            except Exception as exc:
                status = getattr(exc, "http_status", None)
                record_post(state, race_id, state.get("x_account", ""), "x_race", "投稿失敗", body=draft, note_url=state.get("note_urls", {}).get(venue, ""), message=str(exc), http_status=status, retryable=getattr(exc, "retryable", False))
                st.error(str(exc))
    selected = [r for r in races if state["x_targets"].get(_text(r.get("race_id")))]
    st.subheader("投稿前一覧プレビュー")
    st.dataframe([{"レース": f"{venue}{race_number(r)}", "状態": "投稿予定", "予約": state["schedules"].get(_text(r.get("race_id")), ""), "文字数": x_weighted_length(state["x_drafts"].get(_text(r.get("race_id")), ""))} for r in selected], hide_index=True, use_container_width=True)
    start_confirm = st.checkbox("一覧内容・投稿間隔・note URLを確認しました", key=f"batch-confirm-{venue}")
    if st.button("選択した未投稿レースの投稿開始", disabled=not start_confirm or not state.get("x_account") or mode != "手動投稿", use_container_width=True):
        st.info("1件目を投稿し、残りは設定間隔で予約します。予約時刻後に『予約分を実行』を押すとPublisherが送信します。")
        try:
            client = XApiClient(_credentials()); verify_account(client, state["expected_x_account"])
            for index, race in enumerate(selected):
                race_id = _text(race.get("race_id"))
                if index:
                    state["schedules"][race_id] = (datetime.now() + timedelta(minutes=interval * index)).isoformat(timespec="minutes")
                    continue
                try:
                    _post_one(state, race, client, state["x_account"], dry_run=False)
                except DuplicatePostError:
                    continue
                except Exception as exc:
                    record_post(state, race_id, state["x_account"], "x_race", "投稿失敗", body=state["x_drafts"][race_id], note_url=state["note_urls"].get(venue, ""), message=str(exc), http_status=getattr(exc, "http_status", None), retryable=getattr(exc, "retryable", False))
        except Exception as exc: st.error(str(exc))
    if st.button("予約分を実行", disabled=not state.get("x_account") or mode != "予約投稿", use_container_width=True):
        try:
            client = XApiClient(_credentials()); verify_account(client, state["expected_x_account"])
            now = datetime.now()
            for race in selected:
                race_id = _text(race.get("race_id")); raw = state["schedules"].get(race_id, "")
                try: due = datetime.fromisoformat(raw) if "T" in raw else datetime.combine(now.date(), datetime.strptime(raw, "%H:%M").time())
                except ValueError: continue
                if due <= now:
                    try: _post_one(state, race, client, state["x_account"], dry_run=False)
                    except DuplicatePostError: pass
                    except Exception as exc: record_post(state, race_id, state["x_account"], "x_race", "投稿失敗", body=state["x_drafts"][race_id], note_url=state["note_urls"].get(venue, ""), message=str(exc), http_status=getattr(exc, "http_status", None), retryable=getattr(exc, "retryable", False))
        except Exception as exc: st.error(str(exc))
    if mode == "予約投稿":
        if st.button("予約投稿を開始", disabled=not start_confirm or not state.get("x_account"), use_container_width=True):
            state["scheduler_active"] = True
        if state.get("scheduler_active"):
            st.success("予約投稿を監視中（30秒ごと）。Publisherを起動したままにしてください。")
            if st.button("予約投稿を停止", use_container_width=True): state["scheduler_active"] = False
        _scheduled_worker(state, venue, selected)


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
    state["publication_mode"] = st.radio("公開モード", ["全レース無料", "1R無料＋全レースnote"], index=0 if state.get("publication_mode") == "全レース無料" else 1, horizontal=True)
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
    if state["publication_mode"] == "1R無料＋全レースnote":
        free_options = {_text(r.get("race_id")): f"{venue}{race_number(r)}" for r in races}
        chosen = st.selectbox("無料公開レース", list(free_options), format_func=lambda rid: free_options[rid])
        state["free_race_ids"] = [chosen]
    _race_preview(selected)

    tab_note, tab_x, tab_history, tab_profile = st.tabs(["note原稿", "X投稿", "投稿履歴", "固定ポスト"])
    with tab_note:
        _note_tab(state, venue, races, _event_date(snapshot))
    with tab_x:
        _x_tab(state, venue, races)
    with tab_history:
        _history_tab(state, races)
    with tab_profile:
        st.text_area("Xプロフィール固定用文章（コピー可）", value=DEFAULT_PINNED_POST, height=260)

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
