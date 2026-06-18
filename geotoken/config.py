from __future__ import annotations

from pathlib import Path
from typing import Any
import ast
import json

try:
    import yaml
except ImportError:  # keep Step 1 runnable before dependencies are installed
    yaml = None


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a GeoToken YAML config."""
    config_path = Path(path).expanduser().resolve()
    text = config_path.read_text(encoding="utf-8")
    config = yaml.safe_load(text) if yaml else _load_simple_yaml(text)
    config = config or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    config.setdefault("config_path", str(config_path))
    return config


def print_config(config: dict[str, Any]) -> None:
    if yaml:
        print(yaml.safe_dump(config, sort_keys=False, allow_unicode=True).strip())
    else:
        print(json.dumps(config, indent=2, ensure_ascii=False))


def _load_simple_yaml(text: str) -> dict[str, Any]:
    lines = []
    for raw in text.splitlines():
        content = raw.split("#", 1)[0].rstrip()
        if not content:
            continue
        lines.append((len(content) - len(content.lstrip(" ")), content.lstrip()))
    value, index = _parse_block(lines, 0, 0)
    if index != len(lines):
        raise ValueError("Could not parse full config without PyYAML")
    return value


def _parse_block(lines: list[tuple[int, str]], index: int, indent: int):
    if index >= len(lines):
        return {}, index
    if lines[index][1].startswith("- "):
        items = []
        while index < len(lines) and lines[index][0] == indent and lines[index][1].startswith("- "):
            item = lines[index][1][2:].strip()
            index += 1
            if item:
                items.append(_parse_scalar(item))
            else:
                child, index = _parse_block(lines, index, lines[index][0])
                items.append(child)
        return items, index

    mapping = {}
    while index < len(lines) and lines[index][0] == indent:
        key, sep, value = lines[index][1].partition(":")
        if not sep:
            raise ValueError(f"Invalid config line: {lines[index][1]}")
        index += 1
        if value.strip():
            mapping[key.strip()] = _parse_scalar(value.strip())
        elif index < len(lines) and lines[index][0] > indent:
            child, index = _parse_block(lines, index, lines[index][0])
            mapping[key.strip()] = child
        else:
            mapping[key.strip()] = None
    return mapping, index


def _parse_scalar(value: str) -> Any:
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "None", "~"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        return ast.literal_eval(value)
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value.strip('"\'')
