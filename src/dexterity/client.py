from __future__ import annotations

import asyncio
import logging
import math
import random
import time
from typing import Any

import httpx

from .models import PlaceResponse, StartResponse, StatusResponse

logger = logging.getLogger(__name__)


def _parse_error(body: dict) -> str:
    detail = body.get("detail", body)
    if isinstance(detail, list):
        return "validation_error"
    if isinstance(detail, dict):
        return detail.get("error", "unknown")
    return "unknown"


class DexterityAPIError(Exception):
    def __init__(self, status_code: int, error_code: str, message: str = ""):
        self.status_code = status_code
        self.error_code = error_code
        super().__init__(f"HTTP {status_code}: {error_code} — {message}")


_RETRIABLE_EXCEPTIONS = (
    httpx.ConnectTimeout,
    httpx.ConnectError,
    httpx.PoolTimeout,
    httpx.RemoteProtocolError,
)

_RETRIABLE_STATUS = {500, 502, 503, 504}


class DexterityClient:
    BASE_URL = "https://dexterity.ai/challenge/api"

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str | None = None,
        max_connections: int = 200,
        max_keepalive: int = 50,
        timeout: float = 30.0,
        max_retries: int = 5,
        retry_base_delay: float = 0.5,
        retry_max_delay: float = 30.0,
        mode: str = "dev",
    ):
        self._api_key = api_key
        self._mode = mode
        self._max_retries = max_retries
        self._retry_base_delay = retry_base_delay
        self._retry_max_delay = retry_max_delay
        self._base_url = (base_url or self.BASE_URL).rstrip("/")

        limits = httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive,
        )
        self._client = httpx.AsyncClient(
            http2=True,
            limits=limits,
            timeout=httpx.Timeout(timeout),
            base_url=self._base_url,
        )

    async def aclose(self):
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.aclose()

    def _jitter_delay(self, attempt: int) -> float:
        cap = min(self._retry_max_delay, self._retry_base_delay * (2 ** attempt))
        return cap * random.uniform(0, 1)

    def _parse_response(self, response: httpx.Response) -> dict:
        try:
            body = response.json()
        except Exception:
            logger.error("Non-JSON response (status %s): %s", response.status_code, response.text[:500])
            raise DexterityAPIError(response.status_code, "non_json_response", response.text[:200])
        return body

    def _raise_for_error(self, response: httpx.Response, body: dict) -> None:
        if response.status_code < 400:
            return
        error_code = _parse_error(body)
        raise DexterityAPIError(response.status_code, error_code, str(body))

    async def _get_with_retry(self, path: str, **kwargs) -> dict:
        for attempt in range(self._max_retries + 1):
            try:
                resp = await self._client.get(path, **kwargs)
                body = self._parse_response(resp)
                if resp.status_code in _RETRIABLE_STATUS:
                    raise _RetriableError(resp.status_code)
                if resp.status_code == 429:
                    raise _RetriableError(429)
                self._raise_for_error(resp, body)
                return body
            except _RetriableError as e:
                if attempt >= self._max_retries:
                    raise DexterityAPIError(e.status_code, "max_retries_exceeded")
                await asyncio.sleep(self._jitter_delay(attempt))
            except _RETRIABLE_EXCEPTIONS as e:
                if attempt >= self._max_retries:
                    raise
                logger.warning("GET %s retriable error (attempt %d): %s", path, attempt, e)
                await asyncio.sleep(self._jitter_delay(attempt))

    async def _post_with_retry_safe(self, path: str, body: dict) -> dict:
        """Safe to retry: stop, status (GET). Does NOT include /place or /start."""
        for attempt in range(self._max_retries + 1):
            try:
                resp = await self._client.post(path, json=body)
                resp_body = self._parse_response(resp)
                if resp.status_code in _RETRIABLE_STATUS:
                    raise _RetriableError(resp.status_code)
                if resp.status_code == 429:
                    if self._mode == "compete":
                        raise DexterityAPIError(429, "rate_limited", "compete mode rate limit")
                    raise _RetriableError(429)
                self._raise_for_error(resp, resp_body)
                return resp_body
            except _RetriableError as e:
                if attempt >= self._max_retries:
                    raise DexterityAPIError(e.status_code, "max_retries_exceeded")
                await asyncio.sleep(self._jitter_delay(attempt))
            except _RETRIABLE_EXCEPTIONS as e:
                if attempt >= self._max_retries:
                    raise
                logger.warning("POST %s retriable error (attempt %d): %s", path, attempt, e)
                await asyncio.sleep(self._jitter_delay(attempt))

    async def _post_place_with_reconcile(self, game_id: str, path: str, body: dict) -> dict:
        """For /place: ambiguous outcome on ReadTimeout — reconcile via /status."""
        for attempt in range(self._max_retries + 1):
            try:
                resp = await self._client.post(path, json=body)
                resp_body = self._parse_response(resp)
                if resp.status_code in _RETRIABLE_STATUS:
                    raise _RetriableError(resp.status_code)
                if resp.status_code == 429:
                    if self._mode == "compete":
                        raise DexterityAPIError(429, "rate_limited")
                    raise _RetriableError(429)
                self._raise_for_error(resp, resp_body)
                return resp_body
            except httpx.ReadTimeout:
                logger.warning("ReadTimeout on %s (attempt %d); reconciling via /status", path, attempt)
                state = await self.status(game_id)
                reconciled = self._reconcile_place(state, body.get("box_id", ""))
                if reconciled is not None:
                    return reconciled
                # box not placed yet, safe to retry
                if attempt >= self._max_retries:
                    raise DexterityAPIError(0, "max_retries_exceeded", "ReadTimeout + reconcile failed")
                await asyncio.sleep(self._jitter_delay(attempt))
            except _RetriableError as e:
                if attempt >= self._max_retries:
                    raise DexterityAPIError(e.status_code, "max_retries_exceeded")
                await asyncio.sleep(self._jitter_delay(attempt))
            except _RETRIABLE_EXCEPTIONS as e:
                if attempt >= self._max_retries:
                    raise
                logger.warning("POST %s retriable error (attempt %d): %s", path, attempt, e)
                await asyncio.sleep(self._jitter_delay(attempt))

    def _reconcile_place(self, state: StatusResponse, box_id: str) -> dict | None:
        """Return a synthesized PlaceResponse dict if box was placed, else None."""
        placed_ids = {pb.id for pb in state.placed_boxes}
        if box_id in placed_ids:
            return {
                "status": "ok",
                "placed_boxes": [pb.model_dump() for pb in state.placed_boxes],
                "current_box": state.current_box.model_dump() if state.current_box else None,
                "boxes_remaining": state.boxes_remaining,
                "density": state.density,
                "game_status": state.game_status,
                "termination_reason": None,
            }
        return None

    async def _post_start_with_reconcile(self, path: str, body: dict) -> dict:
        """For /start: ReadTimeout reconciles via /my-games to avoid burning compete slots."""
        for attempt in range(self._max_retries + 1):
            try:
                resp = await self._client.post(path, json=body)
                resp_body = self._parse_response(resp)
                if resp.status_code in _RETRIABLE_STATUS:
                    raise _RetriableError(resp.status_code)
                if resp.status_code == 429:
                    if self._mode == "compete":
                        raise DexterityAPIError(429, "rate_limited")
                    raise _RetriableError(429)
                self._raise_for_error(resp, resp_body)
                return resp_body
            except httpx.ReadTimeout:
                logger.warning("ReadTimeout on /start (attempt %d); reconciling via /my-games", attempt)
                my_games = await self.my_games(mode=self._mode, status="active")
                games = my_games.get("games", [])
                if games:
                    # most recently started game
                    game = games[0]
                    game_id = game.get("game_id") or game.get("id")
                    if game_id:
                        state = await self.status(game_id)
                        return {
                            "game_id": state.game_id,
                            "truck": state.placed_boxes[0].dimensions if False else {},
                            "current_box": state.current_box.model_dump() if state.current_box else None,
                            "boxes_remaining": state.boxes_remaining,
                            "mode": state.mode,
                        }
                if attempt >= self._max_retries:
                    raise DexterityAPIError(0, "max_retries_exceeded", "ReadTimeout on /start")
                await asyncio.sleep(self._jitter_delay(attempt))
            except _RetriableError as e:
                if attempt >= self._max_retries:
                    raise DexterityAPIError(e.status_code, "max_retries_exceeded")
                await asyncio.sleep(self._jitter_delay(attempt))
            except _RETRIABLE_EXCEPTIONS as e:
                if attempt >= self._max_retries:
                    raise
                logger.warning("POST %s retriable error (attempt %d): %s", path, attempt, e)
                await asyncio.sleep(self._jitter_delay(attempt))

    async def start(self, mode: str = "dev") -> StartResponse:
        body = {"api_key": self._api_key, "mode": mode}
        resp_body = await self._post_start_with_reconcile("/start", body)
        return StartResponse.model_validate(resp_body)

    async def place(
        self,
        game_id: str,
        box_id: str,
        position: tuple[float, float, float],
        orientation_wxyz: tuple[float, float, float, float],
    ) -> PlaceResponse:
        if not all(math.isfinite(float(x)) for x in position):
            raise ValueError(f"Non-finite placement position for game_id={game_id} box_id={box_id}: {position}")
        if not all(math.isfinite(float(x)) for x in orientation_wxyz):
            raise ValueError(
                f"Non-finite placement orientation_wxyz for game_id={game_id} box_id={box_id}: {orientation_wxyz}"
            )
        body = {
            "game_id": game_id,
            "box_id": box_id,
            "position": list(position),
            "orientation_wxyz": list(orientation_wxyz),
        }
        resp_body = await self._post_place_with_reconcile(game_id, "/place", body)
        return PlaceResponse.model_validate(resp_body)

    async def status(self, game_id: str) -> StatusResponse:
        resp_body = await self._get_with_retry(f"/status/{game_id}")
        return StatusResponse.model_validate(resp_body)

    async def stop(self, game_id: str) -> None:
        body = {"api_key": self._api_key, "game_id": game_id}
        await self._post_with_retry_safe("/stop", body)

    async def my_games(self, mode: str | None = None, status: str | None = None) -> dict:
        params: dict[str, str] = {"api_key": self._api_key}
        if mode is not None:
            params["mode"] = mode
        if status is not None:
            params["status"] = status
        return await self._get_with_retry("/my-games", params=params)


class _RetriableError(Exception):
    def __init__(self, status_code: int):
        self.status_code = status_code
