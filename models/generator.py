#!/usr/bin/env python3
"""
增强版生成器模型：
- 多滤波器纹理编码器 (Canny + Sobel + Laplacian + Gabor)
- SE-Block 通道注意力
- 自注意力 (高维层)
- 增强编码器/解码器块
- 可学习残差缩放
- 可选完全重建模式
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np


# ===================== 纹理特征提取 =====================

def multiscale_canny(image: np.ndarray, scales=[(50, 150), (100, 200), (150, 250)]) -> list:
    """多尺度Canny边缘提取（带高斯模糊去噪）"""
    edges = []
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    for low, high in scales:
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edge = cv2.Canny(blurred, low, high)
        edges.append(edge)
    return edges


# ===================== 注意力模块 =====================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation 通道注意力"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        B, C, _, _ = x.shape
        y = self.squeeze(x).view(B, C)
        y = self.excitation(y).view(B, C, 1, 1)
        return x * y


class SelfAttention2D(nn.Module):
    """轻量级2D自注意力模块"""
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.key = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        Q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        K = self.key(x).view(B, -1, H * W)
        V = self.value(x).view(B, -1, H * W)
        attention = F.softmax(Q @ K / (C // 8) ** 0.5, dim=-1)
        out = V @ attention.permute(0, 2, 1)
        out = out.view(B, C, H, W)
        return self.gamma * out + x


# ===================== 纹理注意力门控 =====================

class TextureAttentionGate(nn.Module):
    """增强版纹理注意力门控——可学习门控网络"""
    def __init__(self, feat_channels, texture_channels):
        super().__init__()
        self.W_g = nn.Conv2d(feat_channels, feat_channels, kernel_size=1, bias=False)
        self.W_x = nn.Conv2d(texture_channels, feat_channels, kernel_size=1, bias=False)
        self.psi = nn.Sequential(
            nn.Conv2d(feat_channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, feat, texture):
        if texture.shape[2:] != feat.shape[2:]:
            texture = F.interpolate(texture, size=feat.shape[2:], mode='bilinear', align_corners=True)
        g = self.W_g(feat)
        x = self.W_x(texture)
        attention = self.psi(F.relu(g + x))
        return feat * attention + x * (1 - attention)


# ===================== 增强纹理编码器 =====================

class EnhancedTextureEncoder(nn.Module):
    """增强版纹理编码器：融合 Canny + Sobel + Laplacian + Gabor 特征"""

    def __init__(self, base_channels=64):
        super().__init__()
        # 输入: 3(Canny多尺度) + 1(Sobel幅值) + 1(Laplacian) + 4(Gabor方向) = 9 通道
        # 加上原始RGB的3通道 → 我们使用12通道输入
        self.input_conv = nn.Sequential(
            nn.Conv2d(12, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        self.enc1 = self._make_block(base_channels, 128)
        self.enc2 = self._make_block(128, 256)
        self.enc3 = self._make_block(256, 512)
        self.enc4 = self._make_block(512, 512)

        self.pool = nn.AvgPool2d(2, 2)

    def _make_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def extract_texture_features(image_batch: torch.Tensor) -> torch.Tensor:
        """从图像批次提取多滤波器纹理特征 → (B, 12, H, W)
        12通道组成: 3(RGB) + 3(Canny) + 1(Sobel) + 1(Laplacian) + 4(Gabor)
        """
        B, C, H, W = image_batch.shape
        device = image_batch.device
        # 转换到 [0, 255] numpy
        images_np = (image_batch.permute(0, 2, 3, 1).cpu().numpy() + 1) * 127.5
        images_np = np.clip(images_np, 0, 255).astype(np.uint8)

        features = []
        for b in range(B):
            img = images_np[b]
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            feat_list = []

            # 0. 原始RGB (3通道) — 保留色彩信息
            rgb = img.astype(np.float32).transpose(2, 0, 1)
            feat_list.append(torch.from_numpy(rgb).float())

            # 1. 多尺度 Canny (3通道)
            edges_multi = multiscale_canny(img)
            for edge in edges_multi:
                feat_list.append(torch.from_numpy(edge).float().unsqueeze(0))

            # 2. Sobel 梯度幅值 (1通道)
            sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            sobel_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
            sobel_mag = sobel_mag / (sobel_mag.max() + 1e-8) * 255
            feat_list.append(torch.from_numpy(sobel_mag).float().unsqueeze(0))

            # 3. Laplacian (1通道)
            laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
            laplacian = np.abs(laplacian)
            laplacian = laplacian / (laplacian.max() + 1e-8) * 255
            feat_list.append(torch.from_numpy(laplacian).float().unsqueeze(0))

            # 4. 多方向 Gabor (4通道: 0°, 45°, 90°, 135°)
            for theta in [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]:
                kernel = cv2.getGaborKernel(
                    ksize=(15, 15), sigma=4.0, theta=theta, lambd=10.0, gamma=0.5
                )
                gabor_resp = cv2.filter2D(gray, cv2.CV_32F, kernel)
                gabor_resp = np.abs(gabor_resp)
                gabor_resp = gabor_resp / (gabor_resp.max() + 1e-8) * 255
                feat_list.append(torch.from_numpy(gabor_resp).float().unsqueeze(0))

            feat_b = torch.cat(feat_list, dim=0)  # (12, H, W)
            features.append(feat_b)

        return torch.stack(features, dim=0).to(device) / 255.0

    def forward(self, image_batch: torch.Tensor) -> dict:
        """
        Returns:
            {'enc0': (B,64,H,W), 'enc1': (B,128,H/2,W/2),
             'enc2': (B,256,H/4,W/4), 'enc3': (B,512,H/8,W/8),
             'enc4': (B,512,H/16,W/16)}
        """
        texture_input = self.extract_texture_features(image_batch)
        enc0 = self.input_conv(texture_input)
        enc1 = self.enc1(self.pool(enc0))
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        return {'enc0': enc0, 'enc1': enc1, 'enc2': enc2, 'enc3': enc3, 'enc4': enc4}


# ===================== 增强编解码块 =====================

class EnhancedEncoderBlock(nn.Module):
    """带通道注意力的编码器块"""
    def __init__(self, in_ch, out_ch, use_attention=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.se = SEBlock(out_ch)
        self.attention = SelfAttention2D(out_ch) if use_attention else nn.Identity()
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = self.attention(out)
        return self.activation(out)


class EnhancedDecoderBlock(nn.Module):
    """带纹理门控 + 通道注意力的解码器块"""
    def __init__(self, in_ch, out_ch, skip_ch=None, texture_ch=None, use_attention=False):
        super().__init__()
        if skip_ch is None:
            skip_ch = out_ch
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(out_ch + skip_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.se = SEBlock(out_ch)
        self.attention = SelfAttention2D(out_ch) if use_attention else nn.Identity()
        self.texture_gate = TextureAttentionGate(out_ch, texture_ch) if texture_ch else None
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, skip, texture_feat=None):
        x = self.up(x)
        x = self.activation(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        if self.texture_gate is not None and texture_feat is not None:
            out = self.texture_gate(out, texture_feat)
        out = self.attention(out)
        return self.activation(out)


# ===================== 增强版生成器 =====================

class SimpleUNetGeneratorWithTexture(nn.Module):
    """增强版生成器（保持向后兼容的类名）

    输入 ``[I_S1, M]`` → 输出洞内残差修正量。

    特性:
    - 多滤波器纹理编码器 (Canny + Sobel + Laplacian + Gabor)
    - 通道注意力 (SE-Block)
    - 高维层自注意力
    - 可学习残差缩放
    - 可选完全重建模式
    """

    def __init__(self, in_channels=4, residual_scale=0.3, base_channels=64,
                 use_full_reconstruction=False, use_enhanced_encoder=True):
        super().__init__()

        # 4 通道 = [I_S1(3), M(1)]：Stage-1 粗修复 + 掩膜。
        # 洞外的已知像素本身就在 I_S1 里（Stage-1 在洞外逐字节保留原图），因此
        # 不再单列"原图"通道：早期版本把 [I_S1, filled, M] 拼成 7 通道，但
        # filled = orig×(1−M) + I_S1×M 在洞内恒等于 I_S1、在洞外是已知像素的拷贝，
        # 多出来的 3 个通道不携带任何新信息，只稀释首层卷积的输入。
        # 该形式与端到端单阶段对照（models/single_stage.py）的 [masked, M] 逐位对应。
        if in_channels != 4:
            raise ValueError(
                f"SimpleUNetGeneratorWithTexture 只支持 4 通道 [I_S1, M]，得到 in_channels={in_channels}"
            )

        self.residual_scale = residual_scale
        self.base_channels = base_channels
        self.use_full_reconstruction = use_full_reconstruction
        self.use_enhanced_encoder = use_enhanced_encoder

        # 纹理编码器（增强版：多滤波器融合）
        self.texture_encoder = EnhancedTextureEncoder(base_channels=base_channels)

        # 编码器
        if use_enhanced_encoder:
            self.enc1 = EnhancedEncoderBlock(in_channels, base_channels, use_attention=False)
            self.enc2 = EnhancedEncoderBlock(base_channels, base_channels * 2, use_attention=False)
            self.enc3 = EnhancedEncoderBlock(base_channels * 2, base_channels * 4, use_attention=False)
            self.enc4 = EnhancedEncoderBlock(base_channels * 4, base_channels * 8, use_attention=True)
        else:
            self.enc1 = self._simple_block(in_channels, base_channels, batch_norm=False)
            self.enc2 = self._simple_block(base_channels, base_channels * 2)
            self.enc3 = self._simple_block(base_channels * 2, base_channels * 4)
            self.enc4 = self._simple_block(base_channels * 4, base_channels * 8)

        self.pool = nn.AvgPool2d(2, 2)

        # 中间块
        if use_enhanced_encoder:
            self.mid_block = nn.Sequential(
                EnhancedEncoderBlock(base_channels * 8, base_channels * 8, use_attention=True),
                nn.Conv2d(base_channels * 8, base_channels * 8, kernel_size=3, padding=1),
            )
        else:
            self.mid_block = self._simple_block(base_channels * 8, base_channels * 8)

        # 解码器（带纹理门控）
        # 各层参数: dec_in_ch, dec_out_ch, skip_ch, texture_ch
        # enc: e1=64, e2=128, e3=256, e4=512
        # tex: enc0=64, enc1=128, enc2=256, enc3=512, enc4=512
        if use_enhanced_encoder:
            self.dec4 = EnhancedDecoderBlock(base_channels * 8, base_channels * 4,
                                             skip_ch=base_channels * 8, texture_ch=512, use_attention=True)
            self.dec3 = EnhancedDecoderBlock(base_channels * 4, base_channels * 2,
                                             skip_ch=base_channels * 4, texture_ch=256, use_attention=False)
            self.dec2 = EnhancedDecoderBlock(base_channels * 2, base_channels,
                                             skip_ch=base_channels * 2, texture_ch=128, use_attention=False)
            self.dec1 = EnhancedDecoderBlock(base_channels, base_channels,
                                             skip_ch=base_channels, texture_ch=64, use_attention=False)
        else:
            # in_ch, skip_ch, out_ch, texture_ch
            # texture_feats: enc3=512, enc2=256, enc1=128, enc0=64
            self.dec4 = self._simple_decoder_block(base_channels * 8, base_channels * 8, base_channels * 4, 512)
            self.dec3 = self._simple_decoder_block(base_channels * 4, base_channels * 4, base_channels * 2, 256)
            self.dec2 = self._simple_decoder_block(base_channels * 2, base_channels * 2, base_channels, 128)
            self.dec1 = self._simple_decoder_block(base_channels, base_channels, base_channels, 64)

        # 输出层
        self.final_conv = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels // 2, 3, kernel_size=1),
        )

        # 可学习残差缩放因子
        self.learnable_scale = nn.Parameter(torch.tensor(float(residual_scale)))

        # 最终激活
        self.final_activation = nn.Sigmoid() if use_full_reconstruction else nn.Tanh()

    def _simple_block(self, in_ch, out_ch, batch_norm=True):
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        ]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_ch))
        return nn.Sequential(*layers)

    def _simple_decoder_block(self, in_ch, skip_ch, out_ch, texture_ch):
        """简单解码块（用于非增强模式，带纹理门控）"""
        class SimpleDecBlock(nn.Module):
            def __init__(self, in_ch, skip_ch, out_ch, texture_ch):
                super().__init__()
                self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                self.conv = nn.Sequential(
                    nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(out_ch),
                )
                self.gate = TextureAttentionGate(out_ch, texture_ch)
                self.texture_ch = texture_ch

            def forward(self, x, skip, texture_feat=None):
                x = self.up(x)
                x = torch.cat([x, skip], dim=1)
                x = self.conv(x)
                if texture_feat is not None:
                    x = self.gate(x, texture_feat)
                return x
        return SimpleDecBlock(in_ch, skip_ch, out_ch, texture_ch)

    def forward(self, stage1_output, mask=None):
        """
        调用方式（唯一）: ``forward(stage1_output, mask)``

        Args:
            stage1_output: (B, 3, H, W)，Stage-1 粗修复结果（洞外为原图，洞内为扩散预测），[-1, 1]
            mask:          (B, 1, H, W)，1 表示受损区域

        Returns:
            correction: 残差修正量 (learnable_scale * tanh(raw))，范围受控

        说明：生成器输入与纹理编码器输入都是 ``[I_S1, M]``（4 通道）。
        洞内像素只来自 Stage-1 预测，洞外只来自已知像素，掩膜区域内的真值不会进入
        网络任意一路（见 training/stage2.py ``_refine`` 的扣洞契约）。
        """
        if mask is None:
            raise ValueError(
                "SimpleUNetGeneratorWithTexture 需要显式 mask：forward(stage1_output, mask)"
            )
        if stage1_output.shape[1] != 3:
            raise ValueError(f"stage1_output 必须是 3 通道，得到 {tuple(stage1_output.shape)}")
        if mask.shape[1] != 1:
            raise ValueError(f"mask 必须是单通道 (B,1,H,W)，得到 {tuple(mask.shape)}")

        x = torch.cat([stage1_output, mask], dim=1)  # (B, 4, H, W)
        return self.decode(x, stage1_output)

    def decode(self, x, texture_source):
        """编码-解码主干：``x`` 为送入 enc1 的通道拼装，``texture_source`` 为纹理编码器输入。

        该主干被两阶段 Stage-2 生成器与端到端单阶段对照
        （``models/single_stage.py``）共用，保证两者的骨干容量完全一致，
        差异只在输入语义（I_S1 vs 扣洞原图）与输出头（残差 vs 直接预测）。
        """
        # 纹理编码
        texture_feats = self.texture_encoder(texture_source)

        # 编码
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        mid = self.mid_block(self.pool(e4))

        # 解码（带纹理注入）
        d4 = self.dec4(mid, e4, texture_feats['enc3'])
        d3 = self.dec3(d4, e3, texture_feats['enc2'])
        d2 = self.dec2(d3, e2, texture_feats['enc1'])
        d1 = self.dec1(d2, e1, texture_feats['enc0'])

        raw = self.final_conv(d1)
        activated = self.final_activation(raw)

        if self.use_full_reconstruction:
            correction = activated
        else:
            scale = self.learnable_scale.abs()
            correction = activated * scale

        return correction


# ===================== 兼容性别名 =====================

# 保持原有导入路径
TextureEncoder = EnhancedTextureEncoder
