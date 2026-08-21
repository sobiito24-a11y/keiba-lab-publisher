from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .note_payload import NoteDraftPayload, PROJECT_ROOT, normalize_note_tags


class NoteDraftAutomationError(RuntimeError):
    def __init__(self, message: str, *, step: str = "", retryable: bool = True):
        super().__init__(message)
        self.step = step
        self.retryable = retryable


class NoteLoginRequiredError(NoteDraftAutomationError):
    def __init__(self, message: str = "noteにログインしていないため、下書き保存を停止しました。"):
        super().__init__(message, step="login_check", retryable=True)


@dataclass(frozen=True)
class NoteDraftConfig:
    user_data_dir: str = field(
        default_factory=lambda: os.environ.get(
            "KEIBA_LAB_NOTE_PROFILE_DIR",
            str(PROJECT_ROOT / ".note_browser_profile"),
        )
    )
    create_url: str = "https://note.com/notes/new"
    browser_channel: str = field(default_factory=lambda: os.environ.get("KEIBA_LAB_NOTE_BROWSER_CHANNEL", "chrome"))
    headless: bool = False
    timeout_ms: int = 15000


@dataclass(frozen=True)
class NoteDraftResult:
    status: str
    url: str
    message: str
    steps: tuple[str, ...]


class NoteDraftDriver(Protocol):
    def open_editor(self) -> None: ...
    def ensure_logged_in(self) -> None: ...
    def set_heading_image(self, image_path: Path | None) -> None: ...
    def set_title(self, title: str) -> None: ...
    def set_body(self, body: str) -> None: ...
    def set_tags(self, tags: tuple[str, ...]) -> None: ...
    def save_draft(self) -> str: ...
    def close(self) -> None: ...


def save_note_draft(payload: NoteDraftPayload, *, config: NoteDraftConfig | None = None, driver: NoteDraftDriver | None = None) -> NoteDraftResult:
    payload.validate()
    steps: list[str] = []
    active_driver = driver or PlaywrightNoteDraftDriver(config or NoteDraftConfig())
    close_driver = driver is None
    step = "open_editor"
    try:
        active_driver.open_editor()
        steps.append(step)
        step = "login_check"
        active_driver.ensure_logged_in()
        steps.append(step)
        step = "heading_image"
        active_driver.set_heading_image(payload.heading_image_path)
        steps.append(step)
        step = "title"
        active_driver.set_title(payload.title)
        steps.append(step)
        step = "body"
        active_driver.set_body(payload.body)
        steps.append(step)
        step = "save_draft"
        url = active_driver.save_draft()
        steps.append(step)
        step = "tags"
        active_driver.set_tags(payload.tags)
        steps.append(step)
        step = "confirm_draft"
        url = active_driver.save_draft()
        steps.append(step)
        return NoteDraftResult(status="draft_saved", url=url, message="note下書きを保存しました。", steps=tuple(steps))
    except NoteDraftAutomationError:
        raise
    except Exception as exc:
        raise NoteDraftAutomationError("note画面の操作に失敗したため、安全に停止しました。", step=step, retryable=True) from exc
    finally:
        if close_driver:
            active_driver.close()


class PlaywrightNoteDraftDriver:
    def __init__(self, config: NoteDraftConfig):
        self.config = config
        self._playwright = None
        self._context = None
        self.page = None

    def open_editor(self) -> None:
        try:
            from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise NoteDraftAutomationError("Playwrightが未導入です。requirements.txtを更新してインストールしてください。", step="playwright_import", retryable=False) from exc
        self._timeout_error = PlaywrightTimeoutError
        Path(self.config.user_data_dir).mkdir(parents=True, exist_ok=True)
        self._playwright = sync_playwright().start()
        launch_kwargs = {
            "user_data_dir": self.config.user_data_dir,
            "headless": self.config.headless,
        }
        if self.config.browser_channel:
            launch_kwargs["channel"] = self.config.browser_channel
        try:
            self._context = self._playwright.chromium.launch_persistent_context(**launch_kwargs)
        except Exception as exc:
            raise NoteDraftAutomationError("ログイン済みブラウザプロファイルを開けませんでした。", step="browser_profile", retryable=True) from exc
        self.page = self._context.new_page()
        self.page.goto(self.config.create_url, wait_until="domcontentloaded", timeout=self.config.timeout_ms)
        self.page.wait_for_timeout(1200)

    def ensure_logged_in(self) -> None:
        page = self._page()
        url = page.url.lower()
        if "login" in url or "signin" in url:
            raise NoteLoginRequiredError()
        if self._editor_locator_count() == 0 and self._visible_text_count(re.compile("ログイン|会員登録")):
            raise NoteLoginRequiredError()

    def set_heading_image(self, image_path: Path | None) -> None:
        if image_path is None:
            return
        page = self._page()
        if not image_path.exists():
            raise NoteDraftAutomationError(f"見出し画像が見つかりません: {image_path}", step="heading_image", retryable=False)
        inputs = page.locator('input[type="file"]')
        try:
            if inputs.count() > 0:
                inputs.first.set_input_files(str(image_path), timeout=self.config.timeout_ms)
                page.wait_for_timeout(800)
                return
        except Exception:
            pass
        for locator in self._image_button_candidates():
            try:
                if locator.count() == 0 or not locator.first.is_visible(timeout=1000):
                    continue
                with page.expect_file_chooser(timeout=self.config.timeout_ms) as chooser_info:
                    locator.first.click(timeout=self.config.timeout_ms)
                chooser_info.value.set_files(str(image_path), timeout=self.config.timeout_ms)
                page.wait_for_timeout(800)
                return
            except Exception:
                continue
        for locator in self._image_menu_candidates():
            try:
                if locator.count() == 0 or not locator.first.is_visible(timeout=1000):
                    continue
                locator.first.click(timeout=self.config.timeout_ms)
                page.wait_for_timeout(500)
                upload = page.get_by_role("button", name=re.compile("画像をアップロード"))
                if upload.count() == 0 or not upload.first.is_visible(timeout=1000):
                    continue
                with page.expect_file_chooser(timeout=self.config.timeout_ms) as chooser_info:
                    upload.first.click(timeout=self.config.timeout_ms)
                chooser_info.value.set_files(str(image_path), timeout=self.config.timeout_ms)
                page.wait_for_timeout(1500)
                self._click_optional(re.compile("^保存$"))
                page.wait_for_timeout(800)
                return
            except Exception:
                continue
        raise NoteDraftAutomationError("見出し画像を設定するボタンが見つかりません。", step="heading_image", retryable=True)

    def set_title(self, title: str) -> None:
        self._replace_text(self._first_visible(self._title_candidates(), "タイトル入力欄が見つかりません。"), title)

    def set_body(self, body: str) -> None:
        self._replace_text(self._first_visible(self._body_candidates(), "本文入力欄が見つかりません。"), body)

    def set_tags(self, tags: tuple[str, ...]) -> None:
        normalized_tags = normalize_note_tags(tags)
        if not normalized_tags:
            return
        page = self._page()
        self._open_publish_settings_if_needed()
        field = None
        for locator in self._tag_candidates():
            try:
                if locator.count() and locator.first.is_visible(timeout=1000):
                    field = locator.first
                    break
            except Exception:
                continue
        if field is None:
            raise NoteDraftAutomationError("タグ入力欄が見つかりません。", step="tags", retryable=True)
        for tag in normalized_tags:
            field.click(timeout=self.config.timeout_ms)
            try:
                field.fill(tag, timeout=self.config.timeout_ms)
            except Exception:
                page.keyboard.insert_text(tag)
            page.keyboard.press("Enter")
            page.wait_for_timeout(150)

    def save_draft(self) -> str:
        page = self._page()
        if self._visible_text_count(re.compile("下書きを保存しました")):
            return page.url
        for locator in self._save_button_candidates():
            try:
                if locator.count() == 0 or not locator.first.is_visible(timeout=1000):
                    continue
                label = locator.first.inner_text(timeout=1000)
                if ("公開" in label or "投稿" in label) and "下書き" not in label:
                    continue
                locator.first.click(timeout=self.config.timeout_ms)
                page.wait_for_timeout(1800)
                return page.url
            except Exception:
                continue
        raise NoteDraftAutomationError("下書き保存ボタンが見つかりません。", step="save_draft", retryable=True)

    def close(self) -> None:
        try:
            if self._context is not None:
                self._context.close()
        finally:
            if self._playwright is not None:
                self._playwright.stop()

    def _page(self):
        if self.page is None:
            raise NoteDraftAutomationError("noteエディタを開けていません。", step="open_editor", retryable=True)
        return self.page

    def _replace_text(self, locator, value: str) -> None:
        page = self._page()
        locator.click(timeout=self.config.timeout_ms)
        try:
            locator.fill(value, timeout=self.config.timeout_ms)
        except Exception:
            page.keyboard.press("Control+A")
            page.keyboard.insert_text(value)
        page.wait_for_timeout(250)

    def _first_visible(self, locators, message: str):
        for locator in locators:
            try:
                if locator.count() > 0 and locator.first.is_visible(timeout=1000):
                    return locator.first
            except Exception:
                continue
        raise NoteDraftAutomationError(message, retryable=True)

    def _click_optional(self, name_pattern: re.Pattern[str]) -> None:
        page = self._page()
        candidates = [
            page.get_by_role("button", name=name_pattern),
            page.get_by_text(name_pattern),
        ]
        for locator in candidates:
            try:
                if locator.count() and locator.first.is_visible(timeout=700):
                    locator.first.click(timeout=1000)
                    page.wait_for_timeout(300)
                    return
            except Exception:
                continue

    def _open_publish_settings_if_needed(self) -> None:
        page = self._page()
        for locator in self._tag_candidates():
            try:
                if locator.count() and locator.first.is_visible(timeout=700):
                    return
            except Exception:
                continue
        publish_button = page.get_by_role("button", name=re.compile("公開に進む"))
        try:
            if publish_button.count() and publish_button.first.is_visible(timeout=1000):
                publish_button.first.click(timeout=self.config.timeout_ms)
                page.wait_for_timeout(1200)
        except Exception as exc:
            raise NoteDraftAutomationError("タグ設定画面を開けません。", step="tags", retryable=True) from exc

    def _visible_text_count(self, pattern: re.Pattern[str]) -> int:
        try:
            return self._page().get_by_text(pattern).count()
        except Exception:
            return 0

    def _editor_locator_count(self) -> int:
        try:
            return sum(locator.count() for locator in self._title_candidates() + self._body_candidates())
        except Exception:
            return 0

    def _title_candidates(self):
        page = self._page()
        title_re = re.compile("タイトル|記事タイトル")
        return [
            page.get_by_placeholder(title_re),
            page.get_by_label(title_re),
            page.locator('textarea[placeholder*="タイトル"]'),
            page.locator('[contenteditable="true"][data-placeholder*="タイトル"]'),
            page.locator('[aria-label*="タイトル"]'),
        ]

    def _body_candidates(self):
        page = self._page()
        body_re = re.compile("本文|自由に|書きはじめ")
        return [
            page.get_by_placeholder(body_re),
            page.get_by_label(body_re),
            page.locator('[contenteditable="true"][data-placeholder*="本文"]'),
            page.locator('[contenteditable="true"][aria-label*="本文"]'),
            page.locator('[contenteditable="true"]').nth(1),
            page.locator("textarea").nth(1),
        ]

    def _image_button_candidates(self):
        page = self._page()
        image_re = re.compile("見出し画像|画像|カバー|アイキャッチ")
        return [
            page.get_by_role("button", name=image_re),
            page.get_by_text(image_re),
            page.locator('[aria-label*="画像"]'),
        ]

    def _image_menu_candidates(self):
        page = self._page()
        return [
            page.locator("button.h-10.w-10.rounded-full"),
            page.locator('button[data-id="ButtonIcon"]').nth(1),
        ]

    def _tag_candidates(self):
        page = self._page()
        tag_re = re.compile("タグ|ハッシュタグ")
        return [
            page.get_by_placeholder(tag_re),
            page.get_by_label(tag_re),
            page.locator('input[placeholder*="タグ"]'),
            page.locator('[contenteditable="true"][data-placeholder*="タグ"]'),
        ]

    def _save_button_candidates(self):
        page = self._page()
        save_re = re.compile("下書き保存|保存")
        return [
            page.get_by_role("button", name=save_re),
            page.get_by_text(save_re),
        ]
