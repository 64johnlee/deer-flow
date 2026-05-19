"""Tests for make_store() three-tier priority: legacy checkpointer → unified database → InMemoryStore."""

from __future__ import annotations

import sys
import types
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(*, checkpointer=None, database_backend="memory", sqlite_path="/tmp/test.db", postgres_url=""):
    """Build a minimal AppConfig-like object for make_store tests."""
    db = MagicMock()
    db.backend = database_backend
    db.sqlite_path = sqlite_path
    db.postgres_url = postgres_url

    cfg = MagicMock()
    cfg.checkpointer = checkpointer
    cfg.database = db
    return cfg


def _store_cm(store_obj):
    """Return a @asynccontextmanager-compatible factory that yields store_obj."""

    @asynccontextmanager
    async def _inner(_config):
        yield store_obj

    return _inner


# ---------------------------------------------------------------------------
# Legacy checkpointer path (tier 1)
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_make_store_uses_legacy_checkpointer_when_present():
    """Legacy checkpointer section takes priority over database config."""
    legacy_cp = MagicMock()
    legacy_cp.type = "memory"
    cfg = _make_config(checkpointer=legacy_cp, database_backend="sqlite")

    from deerflow.runtime.store.async_provider import make_store

    with patch("deerflow.runtime.store.async_provider.get_app_config", return_value=cfg):
        async with make_store() as store:
            assert store is not None


@pytest.mark.anyio
async def test_make_store_legacy_checkpointer_skips_database_tier():
    """When legacy checkpointer is present, _async_store_from_database is never called."""
    legacy_cp = MagicMock()
    legacy_cp.type = "memory"
    cfg = _make_config(checkpointer=legacy_cp, database_backend="sqlite")

    from deerflow.runtime.store.async_provider import make_store

    fake_store = MagicMock()
    db_factory_calls: list = []

    @asynccontextmanager
    async def _tracking_db_factory(_config):
        db_factory_calls.append(_config)
        yield fake_store

    with (
        patch("deerflow.runtime.store.async_provider.get_app_config", return_value=cfg),
        patch("deerflow.runtime.store.async_provider._async_store_from_database", _tracking_db_factory),
    ):
        async with make_store():
            pass

    assert db_factory_calls == [], "_async_store_from_database should not be called when checkpointer is present"


# ---------------------------------------------------------------------------
# Unified database path (tier 2)
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_make_store_routes_to_database_factory_for_sqlite():
    """With no legacy checkpointer, sqlite database config → _async_store_from_database."""
    cfg = _make_config(checkpointer=None, database_backend="sqlite", sqlite_path="/tmp/db.db")

    from deerflow.runtime.store.async_provider import make_store

    fake_store = MagicMock()
    received_configs: list = []

    @asynccontextmanager
    async def _tracking_factory(db_config):
        received_configs.append(db_config)
        yield fake_store

    with (
        patch("deerflow.runtime.store.async_provider.get_app_config", return_value=cfg),
        patch("deerflow.runtime.store.async_provider._async_store_from_database", _tracking_factory),
    ):
        async with make_store() as store:
            assert store is fake_store

    assert len(received_configs) == 1
    assert received_configs[0] is cfg.database


@pytest.mark.anyio
async def test_make_store_routes_to_database_factory_for_postgres():
    """With no legacy checkpointer, postgres database config → _async_store_from_database."""
    cfg = _make_config(checkpointer=None, database_backend="postgres", postgres_url="postgresql://localhost/test")

    from deerflow.runtime.store.async_provider import make_store

    fake_store = MagicMock()

    @asynccontextmanager
    async def _fake_factory(db_config):
        yield fake_store

    with (
        patch("deerflow.runtime.store.async_provider.get_app_config", return_value=cfg),
        patch("deerflow.runtime.store.async_provider._async_store_from_database", _fake_factory),
    ):
        async with make_store() as store:
            assert store is fake_store


# ---------------------------------------------------------------------------
# In-memory fallback (tier 3)
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_make_store_falls_back_to_memory_when_unconfigured():
    """No checkpointer and database.backend=memory → InMemoryStore with warning."""
    cfg = _make_config(checkpointer=None, database_backend="memory")

    from langgraph.store.memory import InMemoryStore

    from deerflow.runtime.store.async_provider import make_store

    with (
        patch("deerflow.runtime.store.async_provider.get_app_config", return_value=cfg),
        patch("deerflow.runtime.store.async_provider.logger") as mock_logger,
    ):
        async with make_store() as store:
            assert isinstance(store, InMemoryStore)
        mock_logger.warning.assert_called_once()
        assert "InMemoryStore" in mock_logger.warning.call_args[0][0]


@pytest.mark.anyio
async def test_make_store_memory_backend_does_not_call_db_factory():
    """database.backend=memory → falls through to InMemoryStore, not _async_store_from_database."""
    cfg = _make_config(checkpointer=None, database_backend="memory")

    from langgraph.store.memory import InMemoryStore

    from deerflow.runtime.store.async_provider import make_store

    db_calls: list = []

    @asynccontextmanager
    async def _tracking_factory(db_config):
        db_calls.append(db_config)
        yield MagicMock()

    with (
        patch("deerflow.runtime.store.async_provider.get_app_config", return_value=cfg),
        patch("deerflow.runtime.store.async_provider._async_store_from_database", _tracking_factory),
    ):
        async with make_store() as store:
            assert isinstance(store, InMemoryStore)

    assert db_calls == []


# ---------------------------------------------------------------------------
# _async_store_from_database internals — sqlite_path and postgres_url
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_async_store_from_database_sqlite_uses_sqlite_path():
    """_async_store_from_database reads db_config.sqlite_path (not connection_string)."""
    db_cfg = MagicMock()
    db_cfg.backend = "sqlite"
    db_cfg.sqlite_path = "/tmp/unified.db"

    fake_store = MagicMock()
    fake_store.setup = AsyncMock()
    used_conn_strs: list[str] = []

    @asynccontextmanager
    async def capturing_cm(conn_str):
        used_conn_strs.append(conn_str)
        yield fake_store

    MockSqliteStore = MagicMock()
    MockSqliteStore.from_conn_string = lambda conn_str: capturing_cm(conn_str)
    fake_sqlite_module = types.SimpleNamespace(AsyncSqliteStore=MockSqliteStore)

    from deerflow.runtime.store.async_provider import _async_store_from_database

    with (
        patch.dict(sys.modules, {"langgraph.store.sqlite.aio": fake_sqlite_module}),
        patch("deerflow.runtime.store.async_provider.ensure_sqlite_parent_dir"),
    ):
        async with _async_store_from_database(db_cfg) as store:
            assert store is fake_store

    assert used_conn_strs == ["/tmp/unified.db"]
    fake_store.setup.assert_awaited_once()


@pytest.mark.anyio
async def test_async_store_from_database_postgres_raises_on_missing_url():
    """_async_store_from_database raises ValueError when postgres_url is empty."""
    db_cfg = MagicMock()
    db_cfg.backend = "postgres"
    db_cfg.postgres_url = ""

    MockPgStore = MagicMock()
    fake_pg_module = types.SimpleNamespace(AsyncPostgresStore=MockPgStore)

    from deerflow.runtime.store.async_provider import _async_store_from_database

    with patch.dict(sys.modules, {"langgraph.store.postgres.aio": fake_pg_module}):
        with pytest.raises(ValueError, match="postgres_url"):
            async with _async_store_from_database(db_cfg):
                pass


@pytest.mark.anyio
async def test_async_store_from_database_postgres_uses_postgres_url():
    """_async_store_from_database passes db_config.postgres_url to AsyncPostgresStore."""
    db_cfg = MagicMock()
    db_cfg.backend = "postgres"
    db_cfg.postgres_url = "postgresql://user:pass@host:5432/db"

    fake_store = MagicMock()
    fake_store.setup = AsyncMock()
    used_urls: list[str] = []

    @asynccontextmanager
    async def capturing_cm(conn_str):
        used_urls.append(conn_str)
        yield fake_store

    MockPgStore = MagicMock()
    MockPgStore.from_conn_string = lambda conn_str: capturing_cm(conn_str)
    fake_pg_module = types.SimpleNamespace(AsyncPostgresStore=MockPgStore)

    from deerflow.runtime.store.async_provider import _async_store_from_database

    with patch.dict(sys.modules, {"langgraph.store.postgres.aio": fake_pg_module}):
        async with _async_store_from_database(db_cfg) as store:
            assert store is fake_store

    assert used_urls == ["postgresql://user:pass@host:5432/db"]
    fake_store.setup.assert_awaited_once()
