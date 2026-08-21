from __future__ import annotations

import html
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import streamlit as st
import streamlit.components.v1 as components

from publisher.content import DEFAULT_NOTE_INTRO, DEFAULT_PINNED_POST, group_by_venue, note_article, note_title, race_number, short_commentary, x_post, x_weighted_length
from publisher.confidence import SHOW_PUBLIC_SS_DEFAULT, assess_confidence, assess_snapshot
from publisher.operations import OPERATION_MODES, publication_candidate, visible_x_races
from publisher.results import ResultDataError, load_result_json, merge_result_upload, result_map_from_snapshot, result_reply
from publisher.note_automation import NoteDraftAutomationError, NoteDraftConfig, save_note_draft
from publisher.note_payload import NotePayloadError, build_prediction_note_payload, build_reading_note_payload, hashtag_line
from publisher.reading import DuplicateReadingThemeError, READING_STATUSES, ReadingArticleError, generate_reading_article, generate_reading_candidates, reading_theme_options, update_reading_article
from publisher.snapshot import LoadedPrediction, load_prediction
from publisher.state import DuplicatePublicationError, dump_state, load_state, new_state, record_manual_publication


def _text(value: Any) -> str: return "" if value is None else str(value).strip()


def _event_date(snapshot: Mapping[str, Any]) -> str:
    dates = [_text(r.get("date")) for r in snapshot.get("races") or [] if _text(r.get("date"))]
    return dates[0] if dates else _text((snapshot.get("scope") or {}).get("dates", [""])[0])


def _mode_label(snapshot: Mapping[str, Any]) -> str:
    return " / ".join(sorted({_text(r.get("race_mode")).upper() for r in snapshot.get("races") or [] if _text(r.get("race_mode"))}))


def _copy_button(label: str, value: str, key: str) -> None:
    payload = json.dumps(value, ensure_ascii=False).replace("</", "<\\/")
    safe_label = html.escape(label)
    components.html(f"""<button id="b-{key}" style="width:100%;padding:.55rem;border:1px solid #bbb;border-radius:.45rem;background:white;cursor:pointer">{safe_label}</button>
<script>const b=document.getElementById('b-{key}');b.onclick=async()=>{{await navigator.clipboard.writeText({payload});b.textContent='✅ コピーしました';setTimeout(()=>b.textContent={json.dumps(label, ensure_ascii=False)},1600);}};</script>""", height=48)


def _split_tags(value: str) -> tuple[str, ...]:
    tags: list[str] = []
    for part in value.replace("、", ",").replace("#", "").split(","):
        clean = part.strip()
        if clean and clean not in tags:
            tags.append(clean)
    return tuple(tags)


def _tags_text(tags: list[str] | tuple[str, ...]) -> str:
    return ", ".join(tags)


def _is_streamlit_cloud_runtime() -> bool:
    truthy = {"1", "true", "yes", "on"}
    for name in ("STREAMLIT_CLOUD", "STREAMLIT_SHARING_MODE", "IS_RUNNING_IN_STREAMLIT_SHARING"):
        value = os.environ.get(name, "").strip().lower()
        if value in truthy:
            return True
    return Path.cwd().as_posix().startswith("/mount/src/")


def _post_set_text(payload: Any) -> str:
    tags = hashtag_line(payload.tags)
    parts = [
        "タイトル",
        payload.title,
        "",
        "本文",
        payload.body,
        "",
        "タグ",
        tags,
    ]
    if payload.heading_image_path:
        parts.extend(["", "見出し画像", str(payload.heading_image_path)])
    return "\n".join(parts).rstrip() + "\n"


def _post_set_key_suffix(payload: Any) -> str:
    raw = "\0".join([payload.title, payload.body, hashtag_line(payload.tags)])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _post_set_panel(payload: Any, key: str) -> None:
    st.subheader("投稿セット")
    suffix = _post_set_key_suffix(payload)
    if _is_streamlit_cloud_runtime():
        st.info("Streamlit Cloudではnoteのブラウザ自動操作を無効化しています。下の投稿セットをnoteへ移してください。")
    if payload.heading_image_path:
        st.image(str(payload.heading_image_path), caption="見出し画像", use_column_width=True)
        try:
            st.download_button(
                "見出し画像を保存",
                payload.heading_image_path.read_bytes(),
                file_name=payload.heading_image_path.name,
                mime="image/png",
                key=f"{key}-image-download",
                use_container_width=True,
            )
        except OSError:
            st.warning("見出し画像を読み込めませんでした。")
    _copy_button("タイトルをコピー", payload.title, f"{key}-title-copy")
    st.text_area("タイトル（コピー用）", value=payload.title, height=88, key=f"{key}-title-text-{suffix}")
    _copy_button("本文をコピー（末尾タグ込み）", payload.body, f"{key}-body-copy")
    st.text_area("本文（末尾タグ込み）", value=payload.body, height=460, key=f"{key}-body-text-{suffix}")
    tags = hashtag_line(payload.tags)
    _copy_button("タグをコピー", tags, f"{key}-tags-copy")
    st.text_area("タグ（コピー用）", value=tags, height=88, key=f"{key}-tags-text-{suffix}")
    st.download_button(
        "投稿セットをテキスト保存",
        _post_set_text(payload).encode("utf-8"),
        file_name=f"{payload.article_type}_note_post_set.txt",
        mime="text/plain",
        key=f"{key}-set-download",
        use_container_width=True,
    )


def _note_automation_config(state: dict[str, Any], key: str) -> NoteDraftConfig:
    settings = state.setdefault("note_automation", {})
    default = NoteDraftConfig()
    with st.expander("note下書き保存設定", expanded=False):
        profile = st.text_input("noteログインを使うブラウザ保存先", value=settings.get("user_data_dir") or default.user_data_dir, key=f"note-profile-{key}")
        browser_options = ["Chrome", "Edge", "Playwright標準"]
        current_label = settings.get("browser_label") or "Chrome"
        if current_label not in browser_options:
            current_label = "Chrome"
        browser_label = st.selectbox("使用ブラウザ", browser_options, index=browser_options.index(current_label), key=f"note-browser-{key}")
        timeout = st.number_input("待機秒数", min_value=5, max_value=60, value=int(settings.get("timeout_seconds") or 15), step=5, key=f"note-timeout-{key}")
    settings["user_data_dir"] = profile
    settings["browser_label"] = browser_label
    settings["timeout_seconds"] = int(timeout)
    channel = {"Chrome": "chrome", "Edge": "msedge", "Playwright標準": ""}[browser_label]
    return NoteDraftConfig(user_data_dir=profile, browser_channel=channel, timeout_ms=int(timeout) * 1000)


def _payload_summary(payload: Any) -> None:
    c1, c2, c3 = st.columns(3)
    c1.write("記事種別")
    c1.code(payload.article_type)
    c2.write("使用画像")
    c2.code(str(payload.heading_image_path) if payload.heading_image_path else "なし")
    c3.write("タグ")
    c3.code(_tags_text(payload.tags))


def _record_note_draft(state: dict[str, Any], payload: Any, result_url: str) -> None:
    record = payload.as_record()
    record.update({"draft_url": result_url, "status": "下書き済み", "saved_at": datetime.now().isoformat(timespec="seconds")})
    record.pop("body", None)
    state.setdefault("note_draft_records", []).append(record)


def _get_loaded(data: bytes) -> LoadedPrediction:
    key = f"loaded:{hash(data)}:{len(data)}"
    if st.session_state.get("loaded_key") != key:
        loaded = load_prediction(data)
        st.session_state.loaded_prediction = loaded; st.session_state.loaded_key = key
        st.session_state.publisher_state = new_state(loaded.source_info)
        st.session_state.publisher_state["result_data"] = result_map_from_snapshot(loaded.snapshot)
    return st.session_state.loaded_prediction


def _sync_confidence_state(state: dict[str, Any], snapshot: Mapping[str, Any]) -> None:
    # Current release always keeps SS private. The setting remains persisted for future release work.
    state["show_public_ss"] = SHOW_PUBLIC_SS_DEFAULT
    state["confidence_ranks"] = assess_snapshot(snapshot.get("races") or [], show_public_ss=state["show_public_ss"])


def _ensure_drafts(state: dict[str, Any], snapshot: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    _sync_confidence_state(state, snapshot)
    grouped = group_by_venue(snapshot, exclude_debut=bool(state.get("exclude_debut", True)))
    race_date = _event_date(snapshot)
    for venue, races in grouped.items():
        generated = note_article(venue, races, race_date, intro=state.get("note_intro") or DEFAULT_NOTE_INTRO, owner_comments=state.get("owner_comments") or {}, show_public_ss=bool(state.get("show_public_ss", False)))
        state.setdefault("note_generated", {})[venue] = generated
        state.setdefault("note_drafts", {}).setdefault(venue, generated)
        url = state.setdefault("note_urls", {}).get(venue, "")
        for race in races:
            race_id = _text(race.get("race_id")); generated_x = x_post(race, url, show_public_ss=bool(state.get("show_public_ss", False)))
            state.setdefault("x_generated", {})[race_id] = generated_x
            state.setdefault("x_drafts", {}).setdefault(race_id, generated_x)
            state.setdefault("race_status", {}).setdefault(race_id, "原稿生成済")
    return grouped


def _status_panel(state: Mapping[str, Any], venue: str, races: list[Mapping[str, Any]], race_date: str) -> None:
    posted = {str(row.get("race_id")) for row in state.get("publication_records") or []}
    results = state.get("result_data") or {}
    st.subheader(f"今日の公開状況｜{race_date} {venue}")
    c1, c2, c3 = st.columns(3)
    c1.metric("note", "✅ 原稿生成済み" if state.get("note_drafts", {}).get(venue) else "⬜ 未生成")
    c2.metric("note URL", "✅ 登録済み" if state.get("note_urls", {}).get(venue) else "⬜ 未登録")
    c3.metric("X公開記録", f"{len(posted & {_text(r.get('race_id')) for r in races})}/{len(races)}")
    st.dataframe([{"レース": f"{venue}{race_number(r)}", "注目度": (state.get("confidence_ranks", {}).get(_text(r.get("race_id"))) or {}).get("public_confidence_rank", ""), "X": "✅ 投稿済み" if _text(r.get("race_id")) in posted else "⬜ 未投稿", "結果": "✅" if (results.get(_text(r.get("race_id"))) or {}).get("results") else "⏳"} for r in races], hide_index=True, use_container_width=True)


def _note_section(state: dict[str, Any], venue: str, races: list[dict[str, Any]], race_date: str) -> None:
    st.header("④ note設定")
    title_default = note_title(venue, race_date)
    state.setdefault("note_titles", {}).setdefault(venue, title_default)
    title = st.text_input("noteタイトル（投稿前に編集可）", value=state["note_titles"].get(venue, title_default), key=f"note-title-{venue}")
    state["note_titles"][venue] = title
    intro = st.text_area("固定冒頭文（編集・state保存可）", value=state.get("note_intro") or DEFAULT_NOTE_INTRO, height=280, key=f"intro-{venue}")
    if intro != (state.get("note_intro") or DEFAULT_NOTE_INTRO):
        state["note_intro"] = intro
    with st.expander("【主のひとこと】をレース別に入力", expanded=False):
        for race in races:
            rid = _text(race.get("race_id")); state.setdefault("owner_comments", {})[rid] = st.text_input(f"{venue}{race_number(race)}", value=state.get("owner_comments", {}).get(rid, ""), key=f"owner-{rid}")
    if st.button("note原稿を再生成", key=f"regen-note-{venue}", use_container_width=True):
        generated = note_article(venue, races, race_date, intro=state.get("note_intro") or DEFAULT_NOTE_INTRO, owner_comments=state.get("owner_comments") or {}, show_public_ss=bool(state.get("show_public_ss", False)))
        state["note_generated"][venue] = generated; state["note_drafts"][venue] = generated
    draft = st.text_area("全レースnote完成原稿（編集可）", value=state["note_drafts"].get(venue, ""), height=650, key=f"note-{venue}")
    state["note_drafts"][venue] = draft
    _copy_button("📋 note全文をコピー", draft, f"note-{venue}")
    st.download_button("Markdown保存", draft.encode("utf-8"), file_name=f"{race_date}_{venue}_note.md", mime="text/markdown", use_container_width=True)
    try:
        preview_payload = build_prediction_note_payload(venue, races, race_date, body=draft, title=title)
        default_tags = _tags_text(preview_payload.tags)
        state.setdefault("note_tags", {}).setdefault(venue, default_tags)
        tags_value = st.text_input("noteタグ", value=state["note_tags"].get(venue, default_tags), key=f"note-tags-{venue}")
        state["note_tags"][venue] = tags_value
        payload = build_prediction_note_payload(venue, races, race_date, body=draft, title=title, tags=_split_tags(tags_value))
        st.subheader("投稿前確認")
        _payload_summary(payload)
        _post_set_panel(payload, f"prediction-{venue}")
        if not _is_streamlit_cloud_runtime():
            config = _note_automation_config(state, f"prediction-{venue}")
            if st.button("noteへ下書き保存", key=f"note-draft-{venue}", use_container_width=True):
                try:
                    result = save_note_draft(payload, config=config)
                    _record_note_draft(state, payload, result.url)
                    st.success("note下書き保存まで完了しました。公開はしていません。")
                except NoteDraftAutomationError as exc:
                    st.error(f"{exc}（停止位置: {exc.step or '不明'}）")
    except NotePayloadError as exc:
        st.error(str(exc))


def _reading_section(state: dict[str, Any]) -> None:
    st.header("KEIBA LAB読み物")
    articles = state.setdefault("reading_articles", [])
    if articles:
        st.dataframe(
            [
                {
                    "テーマ": row.get("theme", ""),
                    "タイトル": row.get("title", ""),
                    "ステータス": row.get("status", ""),
                    "作成日時": row.get("created_at", ""),
                }
                for row in articles
            ],
            hide_index=True,
            use_container_width=True,
        )
    theme = st.selectbox("記事テーマ", reading_theme_options(), key="reading-theme")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("読み物を1本生成", use_container_width=True):
            try:
                articles.append(generate_reading_article(theme, existing_articles=articles))
                st.success("読み物を1本生成しました。")
                st.rerun()
            except DuplicateReadingThemeError as exc:
                st.warning(str(exc))
            except ReadingArticleError as exc:
                st.error(str(exc))
    with c2:
        if st.button("候補を複数生成", use_container_width=True):
            try:
                candidates = generate_reading_candidates(3, existing_articles=articles)
                articles.extend(candidates)
                st.success(f"{len(candidates)}本の候補を生成しました。")
                st.rerun()
            except DuplicateReadingThemeError as exc:
                st.warning(str(exc))
            except ReadingArticleError as exc:
                st.error(str(exc))
    if not articles:
        st.caption("読み物記事はまだありません。")
        return
    labels = [f"{idx + 1}. {row.get('theme', '')}｜{row.get('status', '未投稿')}" for idx, row in enumerate(articles)]
    selected = st.selectbox("編集する読み物", range(len(articles)), format_func=lambda index: labels[index], key="reading-selected")
    article = articles[selected]
    title = st.text_input("タイトル", value=_text(article.get("title")), key=f"reading-title-{selected}")
    body = st.text_area("本文", value=_text(article.get("body")), height=520, key=f"reading-body-{selected}")
    tags_value = st.text_input("タグ", value=_tags_text(article.get("tags") or []), key=f"reading-tags-{selected}")
    status = st.selectbox("ステータス", READING_STATUSES, index=READING_STATUSES.index(article.get("status") if article.get("status") in READING_STATUSES else "未投稿"), key=f"reading-status-{selected}")
    if st.button("読み物を保存", key=f"reading-save-{selected}", use_container_width=True):
        try:
            articles[selected] = update_reading_article(article, title=title, body=body, tags=_split_tags(tags_value), status=status)
            st.success("読み物を保存しました。")
            st.rerun()
        except ReadingArticleError as exc:
            st.error(str(exc))
    try:
        preview = update_reading_article(article, title=title, body=body, tags=_split_tags(tags_value), status=status)
        payload = build_reading_note_payload(preview)
        st.subheader("投稿前確認")
        _payload_summary(payload)
        _post_set_panel(payload, f"reading-{selected}")
        if not _is_streamlit_cloud_runtime():
            config = _note_automation_config(state, f"reading-{selected}")
            if st.button("noteへ下書き保存", key=f"reading-draft-{selected}", use_container_width=True):
                try:
                    result = save_note_draft(payload, config=config)
                    preview["status"] = "下書き済み"
                    articles[selected] = preview
                    _record_note_draft(state, payload, result.url)
                    st.success("読み物をnote下書きに保存しました。公開はしていません。")
                    st.rerun()
                except NoteDraftAutomationError as exc:
                    st.error(f"{exc}（停止位置: {exc.step or '不明'}）")
    except (ReadingArticleError, NotePayloadError) as exc:
        st.error(str(exc))


def _x_section(state: dict[str, Any], venue: str, races: list[dict[str, Any]]) -> None:
    st.header("⑤ note公開後 / ⑥ X投稿")
    old_url = state.setdefault("note_urls", {}).get(venue, "")
    url = st.text_input("当日のnote URL", value=old_url, key=f"url-{venue}")
    if url != old_url:
        state["note_urls"][venue] = url
        for race in races:
            rid = _text(race.get("race_id")); previous_generated = state.get("x_generated", {}).get(rid, ""); current_draft = state.get("x_drafts", {}).get(rid, "")
            generated = x_post(race, url, show_public_ss=bool(state.get("show_public_ss", False))); state["x_generated"][rid] = generated
            if not current_draft or current_draft == previous_generated:
                state["x_drafts"][rid] = generated
            elif old_url and old_url in current_draft:
                state["x_drafts"][rid] = current_draft.replace(old_url, url)
    mode = state["operation_mode"]
    candidate = publication_candidate(races)
    if mode != OPERATION_MODES[0]:
        if candidate:
            st.info(f"🐴 本日の公開候補：{candidate['label']}\n\n{candidate['reason']}")
            if st.button("このレースをX無料公開にする", key=f"candidate-{venue}"):
                state["free_race_ids"] = [candidate["race_id"]]
        options = {_text(r.get("race_id")): f"{venue}{race_number(r)}" + (f"｜注目度{assess_confidence(r).public_confidence_rank}" if assess_confidence(r).public_confidence_rank else "") for r in races}
        current = (state.get("free_race_ids") or [next(iter(options))])[0]
        if current not in options: current = next(iter(options))
        chosen = st.selectbox("本日のX無料公開レース", list(options), index=list(options).index(current), format_func=lambda rid: options[rid])
        state["free_race_ids"] = [chosen]; state["publication_mode"] = "1R無料＋全レースnote"
    else:
        state["publication_mode"] = "全レース無料"
    visible = visible_x_races(mode, races, (state.get("free_race_ids") or [""])[0])
    if not url: st.warning("note公開後にURLを入力すると、URL込みの完成X投稿文へ更新されます。")
    for race in visible:
        rid = _text(race.get("race_id")); public_rank = (state.get("confidence_ranks", {}).get(rid) or {}).get("public_confidence_rank", ""); header = f"{venue}{race_number(race)}" + (f"｜注目度 {public_rank}" if public_rank else "")
        with st.expander(header, expanded=len(visible) == 1):
            draft = st.text_area(f"{header} X投稿文（編集可）", value=state["x_drafts"].get(rid, x_post(race, url, show_public_ss=bool(state.get("show_public_ss", False)))), height=270, key=f"x-{rid}")
            state["x_drafts"][rid] = draft; length = x_weighted_length(draft)
            st.caption(f"X換算 {length} / 280文字")
            if url and url not in draft: st.warning("手動修正版に当日のnote URLがありません。再生成するか、URLを本文へ追加してください。")
            if length > 280: st.error("280文字を超えています。再生成すると保存済み事実を維持したまま自動短縮します。")
            if st.button("保存データから再生成・自動短縮", key=f"regen-x-{rid}"):
                generated = x_post(race, url, show_public_ss=bool(state.get("show_public_ss", False))); state["x_generated"][rid] = generated; state["x_drafts"][rid] = generated; st.rerun()
            _copy_button("📋 X投稿文をコピー", draft, f"x-{rid}")
            is_free = rid in (state.get("free_race_ids") or [])
            if st.button("X投稿済みにする", key=f"published-{rid}", use_container_width=True):
                try:
                    record_manual_publication(state, race, draft, url, free_publication=is_free)
                    st.success("手動公開記録へ保存しました。")
                except DuplicatePublicationError as exc: st.warning(str(exc))


def _result_section(state: dict[str, Any], races: list[dict[str, Any]]) -> None:
    st.header("⑦ レース結果・検証")
    uploaded = st.file_uploader("正式結果JSONを読み込む（任意）", type=["json"], key="result-upload")
    if uploaded is not None and st.session_state.get("result-upload-name") != uploaded.name:
        try:
            payload = load_result_json(uploaded.getvalue()); merge_result_upload(state.setdefault("result_data", {}), payload)
            st.session_state["result-upload-name"] = uploaded.name; st.success("正式結果データを読み込みました。")
        except ResultDataError as exc: st.error(str(exc))
    for race in races:
        rid = _text(race.get("race_id")); header = f"{_text(race.get('venue'))}{race_number(race)}"
        with st.expander(header, expanded=False):
            comment = st.text_input("【主の結果ひとこと】", value=state.setdefault("result_comments", {}).get(rid, ""), key=f"result-comment-{rid}")
            state["result_comments"][rid] = comment
            reply = result_reply(race, state.get("result_data", {}).get(rid), comment)
            reply_key = f"reply-{rid}"
            if reply_key not in st.session_state: st.session_state[reply_key] = state.setdefault("result_reply_drafts", {}).get(rid, reply)
            if st.button("正式結果と主のひとことから更新", key=f"regen-reply-{rid}"): st.session_state[reply_key] = reply
            state.setdefault("result_reply_drafts", {})[rid] = st.text_area("結果リプ案（編集可）", height=160, key=reply_key)
            _copy_button("📋 結果リプをコピー", state["result_reply_drafts"][rid], f"reply-{rid}")


def main() -> None:
    st.set_page_config(page_title="KEIBA LAB Publisher Ver.3", page_icon="🐴", layout="wide")
    st.title("KEIBA LAB Publisher Ver.3")
    st.caption("保存済み予想を読み物へ変換し、手動公開と結果検証を支援します。X API・再予想は使用しません。")
    uploaded = st.file_uploader("① .keiba読み込み", type=["keiba"])
    if uploaded is None: st.info("Keiba AI Dashboardで保存した .keiba を選択してください。"); return
    try: loaded = _get_loaded(uploaded.getvalue())
    except Exception as exc: st.error(f".keibaを開けません: {exc}"); return
    snapshot = loaded.snapshot; state: dict[str, Any] = st.session_state.publisher_state
    state_file = st.file_uploader("Ver.1～3 Publisher stateを開く（任意）", type=["json"], key="state-upload")
    if state_file is not None and st.session_state.get("state-file") != state_file.name:
        try:
            state = load_state(state_file.getvalue(), source_info=loaded.source_info); st.session_state.publisher_state = state; st.session_state["state-file"] = state_file.name
            st.success("既存stateをVer.3として復元しました。")
        except Exception as exc: st.error(f"stateを開けません: {exc}")
    st.success(f"② {_event_date(snapshot)}｜{_mode_label(snapshot)}｜{len(snapshot.get('races') or [])}レース。Prediction Snapshotは読み取り専用です。")
    state["operation_mode"] = st.radio("③ 今日の運用モード", OPERATION_MODES, index=OPERATION_MODES.index(state.get("operation_mode", OPERATION_MODES[0])), horizontal=True)
    state["exclude_debut"] = st.toggle("新馬戦を除外", value=bool(state.get("exclude_debut", True)))
    st.checkbox("SSを公開表示", value=False, disabled=True, help="将来の公開解禁用設定です。現在は内部SSも注目度Sとして公開します。")
    st.caption("注目度ランクは現在検証中です。予想・印・買い推奨を変更するものではありません。")
    grouped = _ensure_drafts(state, snapshot)
    if not grouped: st.warning("掲載対象レースがありません。"); return
    venue = st.radio("会場", list(grouped), horizontal=True); races = grouped[venue]
    _status_panel(state, venue, races, _event_date(snapshot))
    tab_note, tab_reading, tab_x, tab_results, tab_records, tab_profile = st.tabs(["note原稿", "KEIBA LAB読み物", "X投稿", "結果・検証", "公開記録", "固定ポスト"])
    with tab_note: _note_section(state, venue, races, _event_date(snapshot))
    with tab_reading: _reading_section(state)
    with tab_x: _x_section(state, venue, races)
    with tab_results: _result_section(state, races)
    with tab_records:
        records = state.get("publication_records") or []
        if records:
            st.dataframe(records, hide_index=True, use_container_width=True)
        else:
            st.caption("手動公開記録はまだありません。")
        if state.get("legacy_x_api_history"): st.info("Ver.2のX API履歴は過去データとして分離保存されています。新しい公開記録には混在しません。")
    with tab_profile:
        st.text_area("Xプロフィール固定用文章", value=DEFAULT_PINNED_POST, height=250); _copy_button("📋 固定ポスト文をコピー", DEFAULT_PINNED_POST, "profile")
    loaded.assert_unchanged()
    st.download_button("Publisher作業状態を保存", dump_state(state), file_name=f"{_event_date(snapshot)}_publisher_v3.json", mime="application/json", use_container_width=True)
    with st.expander("読込監査情報"):
        st.json(loaded.source_info); st.write("Prediction Snapshot変更0件・予想再計算0回")


if __name__ == "__main__": main()
