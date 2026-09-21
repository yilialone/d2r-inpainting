#!/usr/bin/env python3
"""
推理辅助工具：Canny 边缘提取、自适应 CFG、TTA 集成、DPM-Solver++ 调度器。

Stage1 生成统一使用标准 StableDiffusionInpaintPipeline（见 pipeline.py），
不再使用自定义采样循环或潜在空间梯度引导。
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2

try:
    from diffusers import DPMSolverMultistepScheduler
except ImportError:
    DPMSolverMultistepScheduler = None


# ═══════════════════════════════════════════
# 边缘提取工具（可视化 / 调试用）
# ═══════════════════════════════════════════

def extract_canny_edges(image: Image.Image, low_threshold: int = 50,
                        high_threshold: int = 150) -> Image.Image:
    """从 PIL 图像自动提取 Canny 边缘图。

    使用自适应阈值：基于图像梯度中位数动态调整 Canny 阈值。
    结果可用于可视化参考，不参与扩散采样。
    """
    img_np = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
    median_grad = float(np.median(gradient_magnitude))

    low = max(1, int(0.5 * median_grad)) if low_threshold is None else low_threshold
    high = min(255, int(1.5 * median_grad)) if high_threshold is None else high_threshold

    edges = cv2.Canny(gray, low, high)
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)
    return Image.fromarray(edges, mode="L")


# ═══════════════════════════════════════════
# 自适应 CFG
# ═══════════════════════════════════════════

def compute_adaptive_cfg(mask: np.ndarray,
                         base_cfg: float = 5.0,
                         min_cfg: float = 3.0,
                         max_cfg: float = 8.0) -> float:
    """根据 mask 面积比例动态调整 CFG scale。

    破损面积大 → 低 CFG（避免过强约束导致伪影）
    破损面积小 → 高 CFG（增强结构引导）
    """
    mask_ratio = mask.sum() / mask.size
    if mask_ratio > 0.5:
        return min_cfg
    elif mask_ratio > 0.25:
        return max_cfg - (max_cfg - base_cfg) * (mask_ratio - 0.25) / 0.25
    else:
        return max_cfg


# ═══════════════════════════════════════════
# 多 CFG 集成推理 (TTA)
# ═══════════════════════════════════════════

def stage1_tta_inference(pipe, image, mask, prompt,
                         cfg_scales=[3.0, 5.0, 7.0],
                         num_steps=50):
    """多 CFG 集成推理：取不同 CFG 下输出的像素级中值。

    集成后用 mask 强制替换非破损区域，杜绝泄漏。
    """
    outputs = []
    for cfg in cfg_scales:
        result = pipe(
            image=image, mask_image=mask, prompt=prompt,
            guidance_scale=cfg, num_inference_steps=num_steps,
        ).images[0]
        outputs.append(np.array(result).astype(np.float32))

    stacked = np.stack(outputs, axis=0)  # (N, H, W, C)
    median_output = np.median(stacked, axis=0)

    original_np = np.array(image).astype(np.float32)
    mask_np = np.array(mask).astype(np.float32)
    if mask_np.ndim == 2:
        mask_3ch = np.stack([mask_np] * 3, axis=-1) / 255.0
    else:
        mask_3ch = mask_np / 255.0
    final = original_np * (1 - mask_3ch) + median_output * mask_3ch

    return Image.fromarray(final.clip(0, 255).astype(np.uint8))


# ═══════════════════════════════════════════
# DPM-Solver++ 调度器
# ═══════════════════════════════════════════

def setup_dpm_scheduler(pipe):
    """替换为 DPM-Solver++ 调度器（质量更好，步数需求更少）。

    num_inference_steps=25 即可达到 DDIM 50 步的质量。
    """
    if DPMSolverMultistepScheduler is None:
        raise ImportError("diffusers not installed.")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        algorithm_type="dpmsolver++",
        solver_order=2,
    )
    return pipe
