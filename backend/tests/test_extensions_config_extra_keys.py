"""Tests that extra top-level keys in extensions_config.json (e.g. mcpInterceptors)
are preserved when any write path updates only mcpServers or skills.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.gateway.routers import mcp as mcp_router
from app.gateway.routers import skills as skills_router
from deerflow.client import DeerFlowClient
from deerflow.skills.types import Skill


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_config(path: Path, extra: dict | None = None) -> None:
    data: dict = {"mcpServers": {}, "skills": {}}
    if extra:
        data.update(extra)
    path.write_text(json.dumps(data))


def _read_config(path: Path) -> dict:
    return json.loads(path.read_text())


def _make_skill(name: str, *, enabled: bool = True) -> Skill:
    skill_dir = Path(f"/tmp/skill_{name}")
    return Skill(
        name=name,
        description=f"Description of {name}",
        license="MIT",
        skill_dir=skill_dir,
        skill_file=skill_dir / "SKILL.md",
        relative_path=Path(name),
        category="public",
        enabled=enabled,
    )


def _make_client() -> DeerFlowClient:
    with (
        patch("deerflow.client.get_app_config"),
        patch("deerflow.client.create_chat_model"),
    ):
        return DeerFlowClient()


# ---------------------------------------------------------------------------
# DeerFlowClient.update_mcp_config — preserves extra keys
# ---------------------------------------------------------------------------


class TestClientUpdateMcpConfigPreservesExtraKeys:
    def test_mcpInterceptors_preserved(self, tmp_path):
        config_file = tmp_path / "extensions_config.json"
        _write_config(config_file, extra={"mcpInterceptors": ["pkg:build"]})

        current_config = MagicMock()
        current_config.skills = {}
        reloaded_server = MagicMock()
        reloaded_server.model_dump.return_value = {"enabled": True, "type": "sse"}
        reloaded_config = MagicMock()
        reloaded_config.mcp_servers = {"srv": reloaded_server}

        client = _make_client()
        with (
            patch("deerflow.client.ExtensionsConfig.resolve_config_path", return_value=config_file),
            patch("deerflow.client.get_extensions_config", return_value=current_config),
            patch("deerflow.client.reload_extensions_config", return_value=reloaded_config),
        ):
            client.update_mcp_config({"srv": {"enabled": True}})

        result = _read_config(config_file)
        assert result.get("mcpInterceptors") == ["pkg:build"]

    def test_arbitrary_extra_key_preserved(self, tmp_path):
        config_file = tmp_path / "extensions_config.json"
        _write_config(config_file, extra={"myCustomKey": {"nested": True}})

        current_config = MagicMock()
        current_config.skills = {}
        reloaded_config = MagicMock()
        reloaded_config.mcp_servers = {}

        client = _make_client()
        with (
            patch("deerflow.client.ExtensionsConfig.resolve_config_path", return_value=config_file),
            patch("deerflow.client.get_extensions_config", return_value=current_config),
            patch("deerflow.client.reload_extensions_config", return_value=reloaded_config),
        ):
            client.update_mcp_config({})

        result = _read_config(config_file)
        assert result.get("myCustomKey") == {"nested": True}

    def test_mcpServers_and_skills_written_correctly(self, tmp_path):
        config_file = tmp_path / "extensions_config.json"
        _write_config(config_file, extra={"mcpInterceptors": []})

        skill_cfg = MagicMock()
        skill_cfg.enabled = True
        current_config = MagicMock()
        current_config.skills = {"my-skill": skill_cfg}
        reloaded_config = MagicMock()
        reloaded_config.mcp_servers = {}

        client = _make_client()
        with (
            patch("deerflow.client.ExtensionsConfig.resolve_config_path", return_value=config_file),
            patch("deerflow.client.get_extensions_config", return_value=current_config),
            patch("deerflow.client.reload_extensions_config", return_value=reloaded_config),
        ):
            client.update_mcp_config({"new-srv": {"enabled": True}})

        result = _read_config(config_file)
        assert result["mcpServers"] == {"new-srv": {"enabled": True}}
        assert result["skills"] == {"my-skill": {"enabled": True}}


# ---------------------------------------------------------------------------
# DeerFlowClient.update_skill — preserves extra keys
# ---------------------------------------------------------------------------


class TestClientUpdateSkillPreservesExtraKeys:
    def test_mcpInterceptors_preserved(self, tmp_path):
        config_file = tmp_path / "extensions_config.json"
        _write_config(config_file, extra={"mcpInterceptors": ["pkg:build"]})

        skill = _make_skill("demo", enabled=True)
        toggled = _make_skill("demo", enabled=False)
        ext_config = MagicMock()
        ext_config.mcp_servers = {}
        ext_config.skills = {}

        client = _make_client()
        with (
            patch("deerflow.skills.storage.local_skill_storage.LocalSkillStorage.load_skills", side_effect=[[skill], [toggled]]),
            patch("deerflow.client.ExtensionsConfig.resolve_config_path", return_value=config_file),
            patch("deerflow.client.get_extensions_config", return_value=ext_config),
            patch("deerflow.client.reload_extensions_config"),
        ):
            client.update_skill("demo", enabled=False)

        result = _read_config(config_file)
        assert result.get("mcpInterceptors") == ["pkg:build"]

    def test_skill_enabled_state_written_correctly(self, tmp_path):
        config_file = tmp_path / "extensions_config.json"
        _write_config(config_file, extra={"mcpInterceptors": []})

        skill = _make_skill("demo", enabled=True)
        toggled = _make_skill("demo", enabled=False)
        skill_state = MagicMock()
        skill_state.enabled = False
        ext_config = MagicMock()
        ext_config.mcp_servers = {}
        ext_config.skills = {"demo": skill_state}

        client = _make_client()
        with (
            patch("deerflow.skills.storage.local_skill_storage.LocalSkillStorage.load_skills", side_effect=[[skill], [toggled]]),
            patch("deerflow.client.ExtensionsConfig.resolve_config_path", return_value=config_file),
            patch("deerflow.client.get_extensions_config", return_value=ext_config),
            patch("deerflow.client.reload_extensions_config"),
        ):
            client.update_skill("demo", enabled=False)

        result = _read_config(config_file)
        assert result["skills"]["demo"]["enabled"] is False


# ---------------------------------------------------------------------------
# MCP router — PUT /api/mcp/config preserves extra keys
# ---------------------------------------------------------------------------


def _make_mcp_app() -> FastAPI:
    app = FastAPI()
    app.include_router(mcp_router.router)
    return app


class TestMcpRouterPreservesExtraKeys:
    def test_mcpInterceptors_preserved(self, tmp_path):
        config_file = tmp_path / "extensions_config.json"
        _write_config(config_file, extra={"mcpInterceptors": ["pkg:build"]})

        current_config = MagicMock()
        current_config.skills = {}
        reloaded_server = MagicMock()
        reloaded_server.model_dump.return_value = {"enabled": True, "type": "sse"}
        reloaded_config = MagicMock()
        reloaded_config.mcp_servers = {"srv": reloaded_server}

        app = _make_mcp_app()
        with (
            patch("app.gateway.routers.mcp.ExtensionsConfig.resolve_config_path", return_value=config_file),
            patch("app.gateway.routers.mcp.get_extensions_config", return_value=current_config),
            patch("app.gateway.routers.mcp.reload_extensions_config", return_value=reloaded_config),
        ):
            resp = TestClient(app).put("/api/mcp/config", json={"mcp_servers": {}})

        assert resp.status_code == 200
        result = _read_config(config_file)
        assert result.get("mcpInterceptors") == ["pkg:build"]

    def test_empty_existing_file_does_not_crash(self, tmp_path):
        config_file = tmp_path / "extensions_config.json"
        config_file.write_text("{}")

        current_config = MagicMock()
        current_config.skills = {}
        reloaded_config = MagicMock()
        reloaded_config.mcp_servers = {}

        app = _make_mcp_app()
        with (
            patch("app.gateway.routers.mcp.ExtensionsConfig.resolve_config_path", return_value=config_file),
            patch("app.gateway.routers.mcp.get_extensions_config", return_value=current_config),
            patch("app.gateway.routers.mcp.reload_extensions_config", return_value=reloaded_config),
        ):
            resp = TestClient(app).put("/api/mcp/config", json={"mcp_servers": {}})

        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Skills router — PUT /api/skills/{name} preserves extra keys
# ---------------------------------------------------------------------------


def _make_skills_app() -> FastAPI:
    app = FastAPI()
    app.state.config = SimpleNamespace()
    app.include_router(skills_router.router)
    return app


class TestSkillsRouterPreservesExtraKeys:
    def test_mcpInterceptors_preserved(self, tmp_path):
        config_file = tmp_path / "extensions_config.json"
        _write_config(config_file, extra={"mcpInterceptors": ["pkg:build"]})

        toggled = _make_skill("demo", enabled=False)
        extensions_config = MagicMock()
        extensions_config.mcp_servers = {}
        extensions_config.skills = {}

        app = _make_skills_app()
        with (
            patch("app.gateway.routers.skills.ExtensionsConfig.resolve_config_path", return_value=config_file),
            patch("app.gateway.routers.skills.get_extensions_config", return_value=extensions_config),
            patch("app.gateway.routers.skills.reload_extensions_config"),
            patch("app.gateway.routers.skills.refresh_skills_system_prompt_cache_async"),
            patch("app.gateway.routers.skills.get_or_new_skill_storage") as mock_factory,
        ):
            mock_storage = MagicMock()
            mock_storage.load_skills.return_value = [toggled]
            mock_factory.return_value = mock_storage

            resp = TestClient(app).put("/api/skills/demo", json={"enabled": False})

        assert resp.status_code == 200
        result = _read_config(config_file)
        assert result.get("mcpInterceptors") == ["pkg:build"]

    def test_arbitrary_extra_key_preserved(self, tmp_path):
        config_file = tmp_path / "extensions_config.json"
        _write_config(config_file, extra={"customKey": 42})

        toggled = _make_skill("demo", enabled=True)
        extensions_config = MagicMock()
        extensions_config.mcp_servers = {}
        extensions_config.skills = {}

        app = _make_skills_app()
        with (
            patch("app.gateway.routers.skills.ExtensionsConfig.resolve_config_path", return_value=config_file),
            patch("app.gateway.routers.skills.get_extensions_config", return_value=extensions_config),
            patch("app.gateway.routers.skills.reload_extensions_config"),
            patch("app.gateway.routers.skills.refresh_skills_system_prompt_cache_async"),
            patch("app.gateway.routers.skills.get_or_new_skill_storage") as mock_factory,
        ):
            mock_storage = MagicMock()
            mock_storage.load_skills.return_value = [toggled]
            mock_factory.return_value = mock_storage

            resp = TestClient(app).put("/api/skills/demo", json={"enabled": True})

        assert resp.status_code == 200
        result = _read_config(config_file)
        assert result.get("customKey") == 42
