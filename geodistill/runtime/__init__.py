"""GeoDistill-VLM runtime: data factory, training loop, LLM glue.

This package owns the wiring that the CPU-only P0/P1 modules cannot import
(transformers, peft, nuScenes loaders). Every entry point is lazy: importing
``geodistill.runtime`` works on CPU; the actual functionality only fires when
the GPU dependencies are installed.
"""

from .config import load_config_with_extends, deep_merge
from .runner import train_loop, RunnerConfig, save_ckpt, load_ckpt
from .data_factory import iter_scenes, SceneSource

__all__ = [
    "load_config_with_extends", "deep_merge",
    "train_loop", "RunnerConfig", "save_ckpt", "load_ckpt",
    "iter_scenes", "SceneSource",
]
