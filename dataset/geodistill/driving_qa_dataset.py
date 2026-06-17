"""DriveLM / OmniDrive-style spatial QA dataset (paper §4.5 Table 1).

Loads a JSONL of QAs with 8 categories + three eval splits:
- ``test_template``       (same template, different scene)
- ``test_paraphrase``     (different surface form)
- ``test_composition``    (held-out object combos)

Each QA record has:
    {
      "question": str, "answer": str,
      "type": one_of(QA_CATEGORIES),
      "involved_object_ids": list[int],
      "source_relation": str | None,
      "reliability": float,
      "split_id": str
    }

Reliability filtering rejects sparse-LiDAR / ambiguous-occlusion / unstable-
dynamic-association / too-far / too-small QAs (paper §4.5).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:                                                                    # pragma: no cover
    torch = None

    class Dataset:  # type: ignore[no-redef]
        pass


from geodistill.eval.eval_spatial_qa import QA_CATEGORIES


__all__ = ["DrivingQAItem", "DrivingQADataset", "QA_CATEGORIES"]


class DrivingQAItem(dict):
    pass


class DrivingQADataset(Dataset):
    """Light JSONL-backed dataset; tokenization happens in the GPU runtime."""

    def __init__(
        self,
        jsonl_path: str | Path,
        categories: Sequence[str] = QA_CATEGORIES,
        min_reliability: float = 0.5,
        split_ids: Sequence[str] | None = None,
    ) -> None:
        self.path = Path(jsonl_path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self.categories = set(categories)
        self.min_reliability = float(min_reliability)
        self.split_ids = set(split_ids) if split_ids is not None else None
        self._records: list[DrivingQAItem] = []
        self._load()

    def _load(self) -> None:
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("type") not in self.categories:
                    continue
                if float(rec.get("reliability", 1.0)) < self.min_reliability:
                    continue
                if self.split_ids is not None and rec.get("split_id") not in self.split_ids:
                    continue
                self._records.append(DrivingQAItem(rec))

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> DrivingQAItem:
        return self._records[idx]

    def stats(self) -> dict:
        cnt: dict[str, int] = {c: 0 for c in self.categories}
        for r in self._records:
            cnt[r["type"]] += 1
        return {"total": len(self._records), "by_type": cnt}
