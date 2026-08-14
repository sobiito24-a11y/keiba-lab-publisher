from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


class PostingError(RuntimeError):
    def __init__(self, message: str, *, http_status: int | None = None, retryable: bool = False):
        super().__init__(message)
        self.http_status = http_status
        self.retryable = retryable


@dataclass(frozen=True)
class XCredentials:
    api_key: str
    api_secret: str
    access_token: str
    access_token_secret: str

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "XCredentials":
        aliases = {"api_key": "X_API_KEY", "api_secret": "X_API_SECRET", "access_token": "X_ACCESS_TOKEN", "access_token_secret": "X_ACCESS_TOKEN_SECRET"}
        data = {field: str(values.get(env) or values.get(field) or "") for field, env in aliases.items()}
        if not all(data.values()):
            raise PostingError("X認証情報が未設定です。")
        return cls(**data)


class XApiClient:
    base_url = "https://api.x.com/2"

    def __init__(self, credentials: XCredentials, *, session: Any = None, timeout: float = 20.0):
        try:
            import requests
            from requests_oauthlib import OAuth1
            auth: Any = OAuth1(credentials.api_key, credentials.api_secret, credentials.access_token, credentials.access_token_secret)
        except ImportError as exc:
            if session is None:
                raise PostingError("requests / requests-oauthlib をインストールしてください。") from exc
            auth = None
            requests = None
        self.session = session or requests.Session()
        self.auth = auth
        self.timeout = timeout

    @staticmethod
    def _json(response: Any) -> dict[str, Any]:
        try:
            payload = response.json()
        except Exception:
            payload = {}
        if not 200 <= response.status_code < 300:
            detail = payload.get("detail") or payload.get("title") or getattr(response, "text", "") or "X API error"
            raise PostingError(str(detail), http_status=response.status_code, retryable=response.status_code in {429, 500, 502, 503, 504})
        return payload

    def get_me(self) -> dict[str, str]:
        response = self.session.get(f"{self.base_url}/users/me", auth=self.auth, timeout=self.timeout)
        data = self._json(response).get("data") or {}
        return {"id": str(data.get("id") or ""), "name": str(data.get("name") or ""), "username": str(data.get("username") or "")}

    def create_post(self, body: str) -> str:
        response = self.session.post(f"{self.base_url}/tweets", auth=self.auth, json={"text": body}, timeout=self.timeout)
        data = self._json(response).get("data") or {}
        post_id = str(data.get("id") or "")
        if not post_id:
            raise PostingError("X API応答にpost IDがありません。", http_status=response.status_code)
        return post_id


def verify_account(client: XApiClient, expected_username: str) -> dict[str, str]:
    me = client.get_me()
    if me["username"].lstrip("@").casefold() != expected_username.lstrip("@").casefold():
        raise PostingError(f"接続先 @{me['username']} は想定アカウント @{expected_username.lstrip('@')} と異なります。")
    return me


def post_to_x(client: XApiClient, *, body: str, dry_run: bool = True) -> str:
    return "dry-run" if dry_run else client.create_post(body)


def validate_post_prerequisites(*, note_url: str, account: str) -> None:
    if not note_url.strip():
        raise PostingError("会場のnote URLが未登録です。")
    if not account.strip():
        raise PostingError("X認証済みアカウントがありません。")
