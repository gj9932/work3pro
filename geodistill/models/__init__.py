"""LPGA and Qwen-frozen wrappers for GeoDistill-VLM."""

from .depth_heads import HEAD_REGISTRY, HeadTrainConfig
from .depth_head_eval import train_eval_head, MODEL_VARIANTS

# QwenVisualFrozen requires torch+transformers; import lazily so the P0 CPU
# harness (no transformers) can still use the depth heads.
try:
    from .qwen_visual_frozen import QwenVisualFrozen
    _HAVE_QWEN = True
except Exception:  # pragma: no cover
    QwenVisualFrozen = None
    _HAVE_QWEN = False

__all__ = ["HEAD_REGISTRY", "HeadTrainConfig", "train_eval_head", "MODEL_VARIANTS", "QwenVisualFrozen"]
