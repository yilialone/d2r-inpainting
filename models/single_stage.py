#!/usr/bin/env python3
"""端到端单阶段对照模型（审稿意见 R2-2）。

定位
----
审稿人指出：原稿把"两阶段解耦的收益"与"仅仅见过青铜镜数据（域适配）的收益"
混在一起，要求补一个**真正端到端训练的单阶段模型**，并显式说明其架构、损失、
可训练参数与训练预算；把 Stage-1-only 换个名字不算，未说明预算的全量微调也不算。

本模块提供该对照的生成器：``SingleStageInpaintingGenerator``。

与 D2R 的关系（这一点必须写进论文）
-----------------------------------
本模型**故意复用** D2R 第二阶段（``SimpleUNetGeneratorWithTexture``）的骨干：
多滤波器纹理编码器（3 RGB + 3 Canny + 1 Sobel + 1 Laplacian + 4 Gabor = 12 通道）、
SE 通道注意力、高维层自注意力、纹理注意力门控与同一套编解码块。这样做是为了让比较
尽量干净——骨干容量相同，唯一差别是：

1. **输入**：4 通道 = 扣洞后的 masked RGB(3) + mask(1)。没有 Stage-1 扩散先验，
   也没有 Stage-1 输出这一路输入；模型必须在单次前向中从可见区域直接推断缺失内容。
2. **输出头**：直接预测受损区域内容（tanh，范围 [-1, 1]），而不是叠加在 Stage-1
   输出上的残差修正量。
3. **训练**：从随机初始化开始端到端训练，损伤掩膜外区域在合成时严格保留原图。

因此该对照回答的问题是："在同样的数据、同样的骨干容量与同样的优化预算下，
把修复拆成（扩散先验 + 残差细化）两阶段，是否比单个网络端到端训练更好？"
"""

import torch
import torch.nn as nn

from .generator import SimpleUNetGeneratorWithTexture


class SingleStageInpaintingGenerator(SimpleUNetGeneratorWithTexture):
    """单阶段端到端修复生成器：``(masked_image, mask) → 缺失内容``。

    Args:
        base_channels: 骨干宽度，默认 64（与 D2R Stage-2 生成器一致）。

    调用方式:
        >>> gen = SingleStageInpaintingGenerator()
        >>> masked = image * (1 - mask)          # 扣洞
        >>> prediction = gen(masked, mask)       # [-1, 1]
        >>> composite = masked * (1 - mask) + prediction * mask

    ``composite`` 在掩膜外严格等于原图（因为 ``masked`` 在掩膜外就等于原图），
    与 D2R 的合成约定完全一致，因此可用同一套指标与评测脚本比较。
    """

    INPUT_CHANNELS = 4  # masked RGB (3) + mask (1)

    def __init__(self, base_channels: int = 64):
        super().__init__(
            in_channels=self.INPUT_CHANNELS,
            residual_scale=1.0,
            base_channels=base_channels,
            use_full_reconstruction=True,   # 直接预测，而非残差
            use_enhanced_encoder=True,
        )
        # 父类在 use_full_reconstruction=True 时使用 Sigmoid（旧的死代码路径），
        # 但本管线统一使用 [-1, 1]；这里改为 Tanh，使输出与合成协议一致。
        self.final_activation = nn.Tanh()
        # 单阶段模型没有"残差缩放"这一概念；移除该参数，避免它出现在
        # 可训练参数统计和优化器里（requires_grad 恒为 False 的参数会被 AdamW 跳过，
        # 但保留会让 Table 4 的参数口径不清晰）。
        self.learnable_scale = None

    @property
    def architecture_summary(self) -> str:
        """供论文正文 / Table 4 直接引用的架构描述。"""
        return (
            "single-stage conditional U-Net ({} input channels: masked RGB + mask) with "
            "SE channel attention, self-attention at the two deepest levels, a 12-channel "
            "multi-filter texture encoder (3 RGB + 3 Canny + 1 Sobel + 1 Laplacian + 4 Gabor) "
            "and texture attention gating; direct tanh prediction of the missing content, "
            "no diffusion prior".format(self.INPUT_CHANNELS)
        )

    def forward(self, masked_image: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            masked_image: (B, 3, H, W)，取值 [-1, 1]，受损区域已置零/抹除。
            mask: (B, 1, H, W)，1 表示缺失区域。

        Returns:
            prediction: (B, 3, H, W)，取值 [-1, 1]，对受损区域内容的直接预测。
        """
        if mask is None:
            raise ValueError(
                "SingleStageInpaintingGenerator 需要显式 mask：forward(masked_image, mask)"
            )
        if mask.shape[1] != 1:
            raise ValueError(f"mask 必须是单通道 (B,1,H,W)，得到 {tuple(mask.shape)}")
        if masked_image.shape[1] != 3:
            raise ValueError(
                f"masked_image 必须是 3 通道 RGB，得到 {tuple(masked_image.shape)}"
            )
        # 纹理描述子从可见（扣洞后）图像上计算——这是单阶段模型唯一能看到的图像信息。
        x = torch.cat([masked_image, mask], dim=1)
        return self.decode(x, masked_image)


__all__ = ["SingleStageInpaintingGenerator"]
