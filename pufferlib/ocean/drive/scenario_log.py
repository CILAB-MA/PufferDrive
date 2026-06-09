"""Optional per-scenario metrics JSON logging (see drive.ini [eval] scenario_log_path)."""

from __future__ import annotations

import configparser
import json
import os
from typing import Any, Dict, List, Optional


def _normalize_scenario_log_path(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    raw = str(raw).strip().strip('"').strip("'")
    if not raw or raw.lower() == "none":
        return None
    return raw


def scenario_log_path_from_ini(ini_file: str = "pufferlib/config/ocean/drive.ini") -> Optional[str]:
    """Read [eval] scenario_log_path. Returns None when disabled (none/empty)."""
    if not os.path.isfile(ini_file):
        return None
    cp = configparser.ConfigParser()
    cp.read(ini_file)
    if not cp.has_option("eval", "scenario_log_path"):
        return None
    return _normalize_scenario_log_path(cp.get("eval", "scenario_log_path"))


def resolve_scenario_log_path(
    path: Optional[str] = None,
    ini_file: str = "pufferlib/config/ocean/drive.ini",
) -> Optional[str]:
    """CLI/kwargs override wins; otherwise read drive.ini [eval] scenario_log_path."""
    if path is not None:
        return _normalize_scenario_log_path(path)
    return scenario_log_path_from_ini(ini_file)


def _scenario_dict_key(row: Dict[str, Any]) -> Optional[str]:
    if "scenario_id" in row:
        return str(int(row["scenario_id"]))
    if "map_id" in row:
        return str(int(row["map_id"]))
    return None


def _scenario_metrics(row: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in row.items() if k not in ("scenario_id",)}


def _load_existing_by_scenario_id(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            existing = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(existing, dict):
        return {}

    payload: Dict[str, Any] = {}
    legacy = existing.get("scenario")
    if isinstance(legacy, list):
        for row in legacy:
            if isinstance(row, dict):
                key = _scenario_dict_key(row)
                if key:
                    payload[key] = _scenario_metrics(row)
    for key, value in existing.items():
        if key == "scenario" or not isinstance(value, dict):
            continue
        payload[str(key)] = value
    return payload


def append_scenario_logs(path: str, scenarios: List[Dict[str, Any]]) -> None:
    """Merge scenario metrics into JSON keyed by scenario_id (string)."""
    if not path or not scenarios:
        return
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    payload = _load_existing_by_scenario_id(path)
    for row in scenarios:
        if not isinstance(row, dict):
            continue
        key = _scenario_dict_key(row)
        if key:
            payload[key] = _scenario_metrics(row)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def split_aggregate_and_scenario(log: Dict[str, Any]) -> tuple[Dict[str, Any], Optional[List[Dict[str, Any]]]]:
    """Remove optional scenario list; return (aggregate_metrics, scenarios)."""
    log = dict(log)
    scenarios = log.pop("scenario", None)
    if scenarios is not None and not isinstance(scenarios, list):
        scenarios = None
    return log, scenarios
