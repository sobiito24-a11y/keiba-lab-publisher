from __future__ import annotations


class PostingDisabledError(RuntimeError):
    pass


def post_to_x(*, race_id: str, account: str, body: str, enabled: bool = False) -> str:
    """Ver.1 boundary for a future official X API adapter. No network post is made."""

    if not enabled:
        raise PostingDisabledError("Publisher Ver.1ではX自動投稿は無効です。プレビューから手動投稿してください。")
    raise PostingDisabledError("X公式APIアダプターは未接続です。")

