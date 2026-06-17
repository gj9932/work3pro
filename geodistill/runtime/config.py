"""YAML config loading with ``extends:`` inheritance + deep merge.

Used by the baseline yamls under ``configs/geodistill/baselines/`` so each
baseline can be a small override on top of the main config:

    extends: ../geodistill_qwen25vl_nuscenes.yaml
    baseline:
      profile: r2ac_pe_no_relation
      row: 14
    loss_weights:
      delta: 0.0

Parents are resolved relative to the child file. ``extends`` may chain
arbitrarily deep but cycles raise. Mappings deep-merge (child overrides parent
key by key); lists / scalars replace.

We intentionally keep ``geotoken.config.load_config`` (the simple loader) for
backward compatibility — the new ``load_config_with_extends`` lives in this
package so existing P0 callers do not change behavior.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from geotoken.config import load_config


__all__ = ["load_config_with_extends", "deep_merge"]


def deep_merge(parent: dict, child: dict) -> dict:
    """Right-biased deep merge: ``child`` overrides ``parent`` per key."""
    out = dict(parent)
    for k, v in child.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config_with_extends(path: str | Path, _seen: set[str] | None = None) -> dict[str, Any]:
    """Load ``path`` and recursively resolve any ``extends:`` parent."""
    p = Path(path).expanduser().resolve()
    seen = set(_seen or ())
    if str(p) in seen:
        raise ValueError(f"cyclic extends through {p}")
    seen.add(str(p))

    cfg = load_config(p)
    extends = cfg.pop("extends", None)
    if extends is None:
        return cfg

    parent_path = (p.parent / extends).resolve()
    parent = load_config_with_extends(parent_path, _seen=seen)
    parent.pop("config_path", None)
    cfg.pop("config_path", None)
    merged = deep_merge(parent, cfg)
    merged["config_path"] = str(p)
    merged["_extends_chain"] = (parent.get("_extends_chain", []) or []) + [str(parent_path)]
    return merged
