"""Single entry point for P0 CLIs to obtain a list of ``TokenScene``.

``--source synthetic`` (default, CPU-runnable) builds the synthetic fixture.
``--source nuscenes`` requires a GPU box with data+Qwen+transformers and routes
through ``geodistill.data.nuscenes_adapter`` (raises a clear error otherwise).

Keeping this in one place means the teacher / probe / depth-head CLIs are identical
across data sources — only this factory changes when you move to the GPU box.
"""

from __future__ import annotations

from geodistill.data.synthetic import make_synthetic_dataset, SyntheticConfig


def load_scenes(source: str, n_scenes: int, seed: int = 0, config_path: str | None = None,
                synthetic_overrides: dict | None = None):
    if source == "synthetic":
        cfg = SyntheticConfig(**(synthetic_overrides or {}))
        return make_synthetic_dataset(n_scenes, cfg, seed0=seed)
    if source == "nuscenes":
        if not config_path:
            raise ValueError("--source nuscenes requires --config pointing at a geodistill yaml")
        from geodistill.data.nuscenes_adapter import load_nuscenes_scenes
        return load_nuscenes_scenes(config_path, limit=n_scenes)
    raise ValueError(f"unknown source: {source!r} (expected 'synthetic' or 'nuscenes')")
