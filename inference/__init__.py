"""推理包。

低层零件在 :mod:`inference.pipeline` 与 :mod:`inference.guidance`；需要反复调用的
高层入口是 :class:`inference.restore.D2RRestorer` 与 :func:`inference.restore.restore_image`。
"""

from .pipeline import (
    load_model,
    load_stage2_generator,
    load_single_stage_generator,
    stage1_inference,
    refine_with_stage2,
    single_stage_inference,
    calculate_batch_metrics,
)
from .guidance import (
    extract_canny_edges,
    compute_adaptive_cfg,
    stage1_tta_inference,
    setup_dpm_scheduler,
)
from .restore import D2RRestorer, restore_image, MODES

__all__ = [
    # 高层入口
    "D2RRestorer",
    "restore_image",
    "MODES",
    # 低层零件
    "load_model",
    "load_stage2_generator",
    "load_single_stage_generator",
    "stage1_inference",
    "refine_with_stage2",
    "single_stage_inference",
    "calculate_batch_metrics",
    "extract_canny_edges",
    "compute_adaptive_cfg",
    "stage1_tta_inference",
    "setup_dpm_scheduler",
]
