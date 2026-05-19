"""Tests for async_wait_for_sandbox_ready in deerflow.community.aio_sandbox.backend."""

from __future__ import annotations

import asyncio
import time as _time_module

import httpx
import pytest

from deerflow.community.aio_sandbox import backend as backend_module
from deerflow.community.aio_sandbox.backend import async_wait_for_sandbox_ready


class _FakeResponse:
    def __init__(self, status_code: int = 200):
        self.status_code = status_code


class _FakeAsyncClient:
    """Minimal httpx.AsyncClient stub."""

    def __init__(self, responses: list):
        self._responses = iter(responses)
        self.get_calls: list[tuple] = []

    async def get(self, url: str, timeout: int) -> _FakeResponse:
        self.get_calls.append((url, timeout))
        resp = next(self._responses)
        if isinstance(resp, Exception):
            raise resp
        return resp

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        pass


@pytest.mark.anyio
async def test_async_wait_returns_true_on_first_200(monkeypatch):
    client = _FakeAsyncClient([_FakeResponse(200)])
    sleep_calls: list[float] = []

    async def fake_sleep(s: float) -> None:
        sleep_calls.append(s)

    monkeypatch.setattr(httpx, "AsyncClient", lambda: client)
    monkeypatch.setattr(backend_module.asyncio, "sleep", fake_sleep)

    result = await async_wait_for_sandbox_ready("http://sandbox:8080", timeout=30)

    assert result is True
    assert client.get_calls == [("http://sandbox:8080/v1/sandbox", 5)]
    assert sleep_calls == []


@pytest.mark.anyio
async def test_async_wait_retries_on_non_200_then_succeeds(monkeypatch):
    client = _FakeAsyncClient([_FakeResponse(503), _FakeResponse(503), _FakeResponse(200)])
    sleep_calls: list[float] = []

    async def fake_sleep(s: float) -> None:
        sleep_calls.append(s)

    monkeypatch.setattr(httpx, "AsyncClient", lambda: client)
    monkeypatch.setattr(backend_module.asyncio, "sleep", fake_sleep)

    result = await async_wait_for_sandbox_ready("http://sandbox:8080", timeout=30)

    assert result is True
    assert len(client.get_calls) == 3
    assert sleep_calls == [1, 1]


@pytest.mark.anyio
async def test_async_wait_retries_on_request_error_then_succeeds(monkeypatch):
    client = _FakeAsyncClient([httpx.ConnectError("refused"), _FakeResponse(200)])
    sleep_calls: list[float] = []

    async def fake_sleep(s: float) -> None:
        sleep_calls.append(s)

    monkeypatch.setattr(httpx, "AsyncClient", lambda: client)
    monkeypatch.setattr(backend_module.asyncio, "sleep", fake_sleep)

    result = await async_wait_for_sandbox_ready("http://sandbox:8080", timeout=30)

    assert result is True
    assert len(client.get_calls) == 2
    assert sleep_calls == [1]


@pytest.mark.anyio
async def test_async_wait_returns_false_on_timeout(monkeypatch):
    """Returns False when the timeout elapses before the sandbox is ready."""
    client = _FakeAsyncClient([_FakeResponse(503)] * 10)
    fake_time = [0.0]

    def advancing_time() -> float:
        t = fake_time[0]
        fake_time[0] += 2  # each call jumps 2 s so the 1-s timeout trips quickly
        return t

    async def fast_sleep(_: float) -> None:
        pass

    monkeypatch.setattr(httpx, "AsyncClient", lambda: client)
    monkeypatch.setattr(backend_module.time, "time", advancing_time)
    monkeypatch.setattr(backend_module.asyncio, "sleep", fast_sleep)

    result = await async_wait_for_sandbox_ready("http://sandbox:8080", timeout=1)

    assert result is False


@pytest.mark.anyio
async def test_async_wait_uses_httpx_not_requests(monkeypatch):
    """async_wait_for_sandbox_ready must not use requests.get."""
    import requests as _requests_module

    client = _FakeAsyncClient([_FakeResponse(200)])

    def forbidden_get(*_args, **_kwargs):
        raise AssertionError("async path must not call requests.get")

    monkeypatch.setattr(httpx, "AsyncClient", lambda: client)
    monkeypatch.setattr(_requests_module, "get", forbidden_get)
    monkeypatch.setattr(backend_module.asyncio, "sleep", lambda _: asyncio.sleep(0))

    result = await async_wait_for_sandbox_ready("http://sandbox:8080", timeout=30)

    assert result is True


@pytest.mark.anyio
async def test_async_wait_uses_asyncio_sleep_not_time_sleep(monkeypatch):
    """async_wait_for_sandbox_ready must not use time.sleep."""
    client = _FakeAsyncClient([_FakeResponse(503), _FakeResponse(200)])
    asyncio_sleep_calls: list[float] = []

    async def tracking_sleep(s: float) -> None:
        asyncio_sleep_calls.append(s)

    def forbidden_time_sleep(_: float) -> None:
        raise AssertionError("async path must not call time.sleep")

    monkeypatch.setattr(httpx, "AsyncClient", lambda: client)
    monkeypatch.setattr(backend_module.asyncio, "sleep", tracking_sleep)
    monkeypatch.setattr(_time_module, "sleep", forbidden_time_sleep)

    result = await async_wait_for_sandbox_ready("http://sandbox:8080", timeout=30)

    assert result is True
    assert asyncio_sleep_calls == [1]
