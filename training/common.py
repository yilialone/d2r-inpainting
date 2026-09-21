#!/usr/bin/env python3
"""共享训练组件：损失、验证指标、参数统计与预算记账。

为什么需要本模块
----------------
审稿意见 R2-2 要求新增一个"端到端训练的单阶段对照"，并把"两阶段解耦的收益"与
"见过青铜镜数据（域适配）的收益"分开。要让这个对照可解释，单阶段对照与 D2R 的
Stage-2 必须使用**同一份**损失与指标实现——否则两者的差异可能来自实现细节，而不是
架构本身。

因此本模块是下列量的唯一来源，`training/stage2.py` 与 `training/single_stage.py`
都从这里导入：

- Sobel 边缘纹理损失（掩膜内归一化 L1）
- hinge / BCE 对抗损失
- 掩膜 VGG 感知损失（可选，论文默认关闭）
- 掩膜 PSNR / SSIM 验证代理指标
- 可训练参数量统计
- 墙钟时间与 GPU·h 记账（Table 4 的 "Update count and GPU hours" 列）

注意：本模块不导入 diffusers / peft，因此单阶段对照的轻量测试不需要
Stable Diffusion 权重。
"""

import json
import math
import os
import platform
import time
from typing import Dict, Optional

import torch
import torch.nn.functional as F


# ═══════════════════════════════════════════════
# Stage-1 LoRA 目标模块
# ═══════════════════════════════════════════════
# 论文协议（默认）：LoRA 仅作用于注意力投影层。conv_in/conv_out 永远不能纳入
# （会破坏 UNet 的 9 通道 inpainting 输入处理），to_out.1 是 Dropout，PEFT 不支持。
LORA_TARGET_MODULES_ATTENTION = ["to_k", "to_q", "to_v", "to_out.0"]

# 显式 opt-in：额外把前馈层 proj 纳入 LoRA，提高域适配容量（可训练参数量随之增加，
# 论文 Table 4 口径需用 parameter_report 实测值，不能沿用正文估计值）。
# 该常量集中定义在此处，使 training/stage1.py 在默认路径下与论文正文保持一致。
LORA_TARGET_MODULES_ATTENTION_FF = LORA_TARGET_MODULES_ATTENTION + ["ff.net.0.proj"]


# ═══════════════════════════════════════════════
# 参数统计
# ═══════════════════════════════════════════════

def parameter_report(model) -> Dict[str, int]:
    """返回 {total, trainable, frozen}，单位为参数个数。

    论文 Table 4 的 "Trainable parameters" 列必须由实际实例化的模型导出，
    不能沿用正文里的估计值。
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": int(total),
        "trainable": int(trainable),
        "frozen": int(total - trainable),
    }


def parameter_report_multi(**models) -> Dict[str, Dict[str, int]]:
    """对多个模型分别统计并给出合计。"""
    report = {name: parameter_report(model) for name, model in models.items()}
    report["_total"] = {
        "total": int(sum(item["total"] for key, item in report.items() if key != "_total")),
        "trainable": int(sum(item["trainable"] for key, item in report.items() if key != "_total")),
        "frozen": int(sum(item["frozen"] for key, item in report.items() if key != "_total")),
    }
    return report


# ═══════════════════════════════════════════════
# 纹理（Sobel）损失
# ═══════════════════════════════════════════════

def sobel_edges(x: torch.Tensor) -> torch.Tensor:
    """逐通道 Sobel 梯度幅值，并按原稿定义把每张幅值图归一化到 [0, 1]。"""
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=x.dtype, device=x.device
    ).view(1, 1, 3, 3)

    edges = []
    for c in range(x.shape[1]):
        ch = x[:, c:c + 1, :, :]
        ex = F.conv2d(ch, sobel_x, padding=1)
        ey = F.conv2d(ch, sobel_y, padding=1)
        mag = torch.sqrt(ex ** 2 + ey ** 2 + 1e-8)
        # Original manuscript definition: normalize each Sobel magnitude
        # map to [0, 1] before the masked L1 texture comparison.
        denom = mag.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        mag = mag / denom
        edges.append(mag)
    return torch.cat(edges, dim=1)


def _mask_3ch(mask: torch.Tensor) -> torch.Tensor:
    if mask.shape[1] == 1:
        return mask.repeat(1, 3, 1, 1).clamp(0, 1)
    return mask[:, :3, :, :].clamp(0, 1)


def masked_texture_loss(generated: torch.Tensor, target: torch.Tensor,
                        mask: torch.Tensor) -> torch.Tensor:
    """掩膜内归一化 Sobel 纹理 L1（论文 λ_tex = 10.0 对应的那一项）。"""
    gen_edges = sobel_edges(generated)
    tgt_edges = sobel_edges(target)

    mask_3 = _mask_3ch(mask)
    diff = (gen_edges - tgt_edges).abs() * mask_3
    num_pixels = mask_3.sum() + 1e-6
    return diff.sum() / num_pixels


# ═══════════════════════════════════════════════
# 对抗损失
# ═══════════════════════════════════════════════

def gan_loss(disc_fake: torch.Tensor, disc_real: torch.Tensor,
             for_generator: bool, use_hinge: bool = True) -> torch.Tensor:
    """判别器输出 raw logits；默认 hinge，备选 BCEWithLogits（原稿实现一致）。"""
    if use_hinge:
        if for_generator:
            return -torch.mean(disc_fake)
        disc_loss_real = F.relu(1.0 - disc_real).mean()
        disc_loss_fake = F.relu(1.0 + disc_fake).mean()
        return disc_loss_real + disc_loss_fake
    if for_generator:
        return F.binary_cross_entropy_with_logits(disc_fake, torch.ones_like(disc_fake))
    loss_real = F.binary_cross_entropy_with_logits(disc_real, torch.ones_like(disc_real))
    loss_fake = F.binary_cross_entropy_with_logits(disc_fake, torch.zeros_like(disc_fake))
    return 0.5 * (loss_real + loss_fake)


# ═══════════════════════════════════════════════
# 感知损失（可选，论文默认关闭）
# ═══════════════════════════════════════════════

def make_vgg_normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """返回与 torchvision.transforms.Normalize 等价的归一化函数（设备/精度自适应）。"""
    def normalize(x: torch.Tensor) -> torch.Tensor:
        m = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        s = torch.tensor(std, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        return (x - m) / s
    return normalize


def build_vgg_feature_extractor(device):
    """VGG16 features[:9]（ImageNet 权重），冻结。返回 (vgg, normalize)。"""
    import torchvision.models as tv_models
    from torchvision.models import VGG16_Weights

    vgg16 = tv_models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    vgg = torch.nn.Sequential(*list(vgg16.features)[:9]).eval().to(device)
    for p in vgg.parameters():
        p.requires_grad = False
    return vgg, make_vgg_normalize()


def masked_perceptual_loss(vgg, normalize, generated: torch.Tensor,
                           target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """掩膜内 VGG 特征 L1（把 mask 上采样到 224×224 后加权）。"""
    if vgg is None:
        return torch.tensor(0.0, device=generated.device)
    gen = ((generated + 1.0) / 2.0).clamp(0, 1)
    tgt = ((target + 1.0) / 2.0).clamp(0, 1)
    gen = F.interpolate(gen, size=(224, 224), mode="bilinear", align_corners=False)
    tgt = F.interpolate(tgt, size=(224, 224), mode="bilinear", align_corners=False)
    gen = normalize(gen)
    tgt = normalize(tgt)
    gen_feat = vgg(gen)
    tgt_feat = vgg(tgt)
    mask_resized = F.interpolate(mask, size=(224, 224), mode="nearest")
    if mask_resized.shape[1] == 1:
        mask_resized = mask_resized.repeat(1, gen_feat.shape[1], 1, 1)
    return F.l1_loss(gen_feat * mask_resized, tgt_feat * mask_resized)


# ═══════════════════════════════════════════════
# 验证指标（训练期选模用；论文最终指标由 metrics/ 模块计算）
# ═══════════════════════════════════════════════

def masked_psnr_ssim(prediction: torch.Tensor, target: torch.Tensor,
                     mask: torch.Tensor) -> Dict[str, float]:
    """掩膜内 PSNR 与简化 SSIM（验证代理）。

    注意：这是训练期选模用的代理量。论文发表的标量指标由 `metrics/` 模块按
    空间掩膜加权方式计算，两者不可混用。
    """
    mask_3 = _mask_3ch(mask)
    active = mask_3.sum().clamp_min(1.0)
    mse = (((prediction - target) ** 2) * mask_3).sum() / active
    psnr = 10.0 * math.log10(4.0 / (float(mse.item()) + 1e-8))

    x = prediction[mask_3.bool()]
    y = target[mask_3.bool()]
    C1, C2 = (0.01 * 2) ** 2, (0.03 * 2) ** 2
    mu_x, mu_y = x.mean(), y.mean()
    sigma_x, sigma_y = x.var(unbiased=False), y.var(unbiased=False)
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean()
    ssim = float((((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / (
        (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2) + 1e-8
    )).item())
    return {"psnr": psnr, "ssim": ssim}


def composite_score(psnr: float, ssim: float, gen_loss: float,
                    weights=(0.4, 0.3, 0.3)) -> float:
    """Stage-2 早停/选模使用的复合分数；单阶段对照沿用同一公式与权重。"""
    w1, w2, w3 = weights
    return float(w1 * psnr / 40.0 + w2 * ssim + w3 * 1.0 / (gen_loss + 0.1))


# ═══════════════════════════════════════════════
# 预算记账（Table 4: update count and GPU hours）
# ═══════════════════════════════════════════════

class BudgetTracker:
    """记录墙钟时间、设备小时与峰值显存。

    ``gpu_hours`` 定义为 wall_seconds × world_size / 3600，即"所有参与进程的
    GPU 占用时间之和"，与 Table 4 的 "GPU hours" 列口径一致（单卡单进程时二者相同）。
    该口径会在写出的 JSON 中显式注明，避免与"用户墙钟时间"混淆。
    """

    DEFINITION = "device_hours = wall_clock_seconds * world_size / 3600"

    def __init__(self, world_size: int = 1):
        self.world_size = max(1, int(world_size))
        self._start = None
        self.wall_seconds = 0.0

    def start(self):
        self._start = time.time()
        if torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass

    def stop(self) -> float:
        if self._start is not None:
            self.wall_seconds = time.time() - self._start
        return self.wall_seconds

    def elapsed(self) -> float:
        if self._start is None:
            return self.wall_seconds
        return time.time() - self._start

    def gpu_hours(self) -> float:
        return self.wall_seconds * self.world_size / 3600.0

    def current_gpu_hours(self) -> float:
        """训练进行中的实时 GPU·h（用于按 GPU·h 匹配预算时逐步判断）。"""
        return self.elapsed() * self.world_size / 3600.0

    def peak_gpu_memory_gb(self) -> Optional[float]:
        if not torch.cuda.is_available():
            return None
        try:
            return float(torch.cuda.max_memory_allocated() / (1024 ** 3))
        except Exception:
            return None

    def summary(self) -> Dict[str, object]:
        return {
            "wall_clock_seconds": round(self.wall_seconds, 3),
            "wall_clock_hours": round(self.wall_seconds / 3600.0, 4),
            "world_size": self.world_size,
            "gpu_hours": round(self.gpu_hours(), 4),
            "gpu_hours_definition": self.DEFINITION,
            "peak_gpu_memory_gb": self.peak_gpu_memory_gb(),
        }


def write_budget_report(output_dir: str, record: Dict[str, object], filename: str = "budget_report.json"):
    """写出可审计的预算报告；非主进程不应调用本函数。"""
    os.makedirs(output_dir, exist_ok=True)
    record = dict(record)
    record.setdefault("environment", {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    })
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    return path
