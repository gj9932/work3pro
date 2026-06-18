"""JSONL logging + run metadata for P0 experiments.

Every experiment row is a flat-ish JSON object with a standard envelope so
``collect_tables.py`` can aggregate without knowing the experiment. Numbers that
require real nuScenes+Qwen and cannot be produced locally are written as the string
``"TBD"`` (never fabricated).
"""

from __future__ import annotations

import json
import platform
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

TBD = "TBD"


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "nogit"


def run_envelope(experiment: str, source: str, config: dict | None = None) -> dict:
    return {
        "experiment": experiment,
        "source": source,                 # "synthetic" | "nuscenes"
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_commit": git_commit(),
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "config": config or {},
    }


class JsonlWriter:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        self._fh.write(json.dumps(record, ensure_ascii=False, default=_default) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _default(o):
    try:
        import torch
        if isinstance(o, torch.Tensor):
            return o.tolist()
    except Exception:
        pass
    return str(o)


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
