"""GeoDistill-VLM trainers (paper §3.13)."""

from .common import OptimSpec, build_optimizer, save_ckpt, load_ckpt, freeze_module, unfreeze_module, trainable_params, check_step0_identity
from .build_lpga import GeoDistillConfig, GeoDistillVLM, build_from_config
from .train_stage_a0 import StageA0Config, StageA0Step, stage_a0_step
from .train_stage_a1 import StageA1Config, StageA1Step, stage_a1_step
from .train_stage_b import StageBConfig, StageBStep, stage_b_step
from .train_stage_c import StageCConfig, stage_c_step, apply_corruption

__all__ = [
    "OptimSpec", "build_optimizer", "save_ckpt", "load_ckpt",
    "freeze_module", "unfreeze_module", "trainable_params", "check_step0_identity",
    "GeoDistillConfig", "GeoDistillVLM", "build_from_config",
    "StageA0Config", "StageA0Step", "stage_a0_step",
    "StageA1Config", "StageA1Step", "stage_a1_step",
    "StageBConfig", "StageBStep", "stage_b_step",
    "StageCConfig", "stage_c_step", "apply_corruption",
]
