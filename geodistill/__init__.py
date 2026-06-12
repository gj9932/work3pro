"""GeoDistill-VLM: distilling privileged LiDAR geometry into native Qwen2.5-VL tokens.

See ``plan/work3_1pro.md`` and ``paper/work3pro_cvpr2027_draft5_zh_qwen.md``.

Public surface kept minimal until milestones land their modules.
"""

from geotoken.config import load_config, print_config

__all__ = ["load_config", "print_config"]
