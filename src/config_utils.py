from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


class ConfigNode(dict):
    """Small dict wrapper with attribute access for YAML configs."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def _wrap(value: Any) -> Any:
    if isinstance(value, Mapping):
        return ConfigNode({key: _wrap(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_wrap(item) for item in value]
    return value


def load_config(path: str | os.PathLike[str]) -> ConfigNode:
    with open(path, "r", encoding="utf-8") as file:
        cfg = _wrap(yaml.safe_load(file))
    cfg.config_path = str(path)
    return cfg


def repo_path(cfg: ConfigNode, value: str | os.PathLike[str]) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str(Path(cfg.paths.repo_root).resolve() / path)


def setup_pipeline_environment(cfg: ConfigNode) -> None:
    for key in ("output_dir", "log_dir", "temp_dir", "actionformer_output_dir"):
        Path(repo_path(cfg, cfg.paths[key])).mkdir(parents=True, exist_ok=True)
    for key in ("pseudo_json", "actionformer_annotation_json"):
        Path(repo_path(cfg, cfg.paths[key])).parent.mkdir(parents=True, exist_ok=True)

    os.environ["TMPDIR"] = repo_path(cfg, cfg.paths.temp_dir)
    os.environ["TMP"] = repo_path(cfg, cfg.paths.temp_dir)
    os.environ["TEMP"] = repo_path(cfg, cfg.paths.temp_dir)
    os.environ["XDG_CACHE_HOME"] = repo_path(cfg, cfg.paths.xdg_cache_home)
    os.environ["MPLCONFIGDIR"] = repo_path(cfg, cfg.paths.matplotlib_config_dir)
    os.environ["PYTHONPYCACHEPREFIX"] = repo_path(cfg, cfg.paths.python_pycache_prefix)
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
