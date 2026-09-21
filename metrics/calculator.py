#!/usr/bin/env python3
"""Image-completion metrics with explicit full-image and masked-region semantics."""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import convolve1d
from scipy import linalg
from PIL import Image
from torchvision import transforms
from torchvision.models import inception_v3, Inception_V3_Weights

# LPIPS
import lpips


class MetricCalculator:
    """PSNR / SSIM / LPIPS / FID 指标计算器"""

    def __init__(self, device="cpu"):
        self.device = device
        self._lpips_fn = None
        self._inception = None
        self._inception_transform = None

    # ── LPIPS lazy-load ──────────────────────────────
    @property
    def lpips_fn(self):
        if self._lpips_fn is None:
            # Spatial output permits genuine mask-weighted feature comparison.
            self._lpips_fn = lpips.LPIPS(net='alex', spatial=True).to(self.device)
            self._lpips_fn.eval()
        return self._lpips_fn

    # ── InceptionV3 lazy-load (for FID) ──────────────
    @property
    def inception(self):
        if self._inception is None:
            self._inception = inception_v3(
                weights=Inception_V3_Weights.DEFAULT,
                transform_input=False,
            )
            self._inception.fc = torch.nn.Identity()  # 去掉分类头
            self._inception = self._inception.to(self.device)
            self._inception.eval()
        return self._inception

    @property
    def inception_transform(self):
        if self._inception_transform is None:
            self._inception_transform = Inception_V3_Weights.DEFAULT.transforms()
        return self._inception_transform

    # ═══════════════════════════════════════════════════
    # PSNR
    # ═══════════════════════════════════════════════════
    def calculate_psnr(self, original, inpainted):
        original_np = np.array(original).astype(np.float32) / 255.0
        inpainted_np = np.array(inpainted).astype(np.float32) / 255.0

        mse = np.mean((original_np - inpainted_np) ** 2)
        if mse == 0:
            return float('inf')

        psnr_value = 20 * np.log10(1.0 / np.sqrt(mse))
        return psnr_value

    # ═══════════════════════════════════════════════════
    # SSIM
    # ═══════════════════════════════════════════════════
    def calculate_ssim(self, original, inpainted, window_size=11, sigma=1.5):
        original_np = np.array(original).astype(np.float32) / 255.0
        inpainted_np = np.array(inpainted).astype(np.float32) / 255.0

        if len(original_np.shape) == 3 and original_np.shape[2] == 3:
            original_y = 0.299 * original_np[:, :, 0] + 0.587 * original_np[:, :, 1] + 0.114 * original_np[:, :, 2]
            inpainted_y = 0.299 * inpainted_np[:, :, 0] + 0.587 * inpainted_np[:, :, 1] + 0.114 * inpainted_np[:, :, 2]
        else:
            original_y = original_np
            inpainted_y = inpainted_np

        x = np.arange(window_size) - window_size // 2
        gaussian = np.exp(-x ** 2 / (2 * sigma ** 2))
        gaussian = gaussian / gaussian.sum()

        mu1 = convolve1d(convolve1d(original_y, gaussian, axis=0), gaussian, axis=1)
        mu2 = convolve1d(convolve1d(inpainted_y, gaussian, axis=0), gaussian, axis=1)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = convolve1d(convolve1d(original_y ** 2, gaussian, axis=0), gaussian, axis=1) - mu1_sq
        sigma2_sq = convolve1d(convolve1d(inpainted_y ** 2, gaussian, axis=0), gaussian, axis=1) - mu2_sq
        sigma12 = convolve1d(convolve1d(original_y * inpainted_y, gaussian, axis=0), gaussian, axis=1) - mu1_mu2

        C1 = (0.01 * 1) ** 2
        C2 = (0.03 * 1) ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        ssim_value = np.mean(ssim_map)

        return ssim_value

    def calculate_psnr_ssim(self, original, inpainted):
        psnr_value = self.calculate_psnr(original, inpainted)
        ssim_value = self.calculate_ssim(original, inpainted)
        return psnr_value, ssim_value

    # ═══════════════════════════════════════════════════
    # LPIPS (Learned Perceptual Image Patch Similarity)
    # ═══════════════════════════════════════════════════
    def calculate_lpips(self, original, inpainted):
        """计算两张图像的 LPIPS 感知距离。

        Args:
            original: PIL Image (RGB)
            inpainted: PIL Image (RGB)

        Returns:
            float: LPIPS 距离 (越小越相似)
        """
        # PIL → [-1, 1] tensor (lpips 期望的输入范围)
        transform_fn = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5]),
        ])
        orig_tensor = transform_fn(original).unsqueeze(0).to(self.device)
        inp_tensor = transform_fn(inpainted).unsqueeze(0).to(self.device)

        with torch.no_grad():
            distance = self.lpips_fn(orig_tensor, inp_tensor)
        return float(distance.mean().item())

    # ═══════════════════════════════════════════════════
    # Inception 特征提取（FID 相关）
    # ═══════════════════════════════════════════════════
    def _get_inception_features(self, images, batch_size=16):
        """从一批 PIL 图像提取 InceptionV3 pool3 特征 (2048-d)。

        Args:
            images: list of PIL Image (RGB)
            batch_size: 推理批次大小

        Returns:
            np.ndarray of shape (N, 2048)
        """
        features = []
        for i in range(0, len(images), batch_size):
            batch_imgs = images[i:i + batch_size]
            tensors = []
            for img in batch_imgs:
                t = self.inception_transform(img)
                tensors.append(t)
            batch_tensor = torch.stack(tensors, dim=0).to(self.device)

            with torch.no_grad():
                feat = self.inception(batch_tensor)
            features.append(feat.cpu().numpy())

        return np.concatenate(features, axis=0)

    # ═══════════════════════════════════════════════════
    # FID (Fréchet Inception Distance)
    # ═══════════════════════════════════════════════════
    def calculate_fid(self, original_images, inpainted_images):
        """计算两组图像之间的 FID。

        Args:
            original_images: list of PIL Image (RGB) — 原图集合
            inpainted_images: list of PIL Image (RGB) — 修复图集合

        Returns:
            float: FID 值 (越小越相似)
        """
        if len(original_images) < 2 or len(inpainted_images) < 2:
            raise ValueError("FID 至少需要每组 2 张图像，不能作为单图指标")
        orig_feats = self._get_inception_features(original_images)
        inp_feats = self._get_inception_features(inpainted_images)

        mu_orig = np.mean(orig_feats, axis=0)
        mu_inp = np.mean(inp_feats, axis=0)

        # 协方差 + 正则化防止奇异 (小样本 2048-d 下协方差严重欠秩)
        eps = 1e-6
        sigma_orig = np.cov(orig_feats, rowvar=False) + eps * np.eye(orig_feats.shape[1])
        sigma_inp = np.cov(inp_feats, rowvar=False) + eps * np.eye(inp_feats.shape[1])

        # 计算 Fréchet 距离
        diff = mu_orig - mu_inp
        # 数值稳定处理
        covmean, _ = linalg.sqrtm(sigma_orig.dot(sigma_inp), disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = diff.dot(diff) + np.trace(sigma_orig) + np.trace(sigma_inp) - 2 * np.trace(covmean)
        return max(float(fid), 0.0)  # 数值稳定：截断负值

    def calculate_kid(self, original_images, inpainted_images):
        """Unbiased KID estimate using the degree-3 polynomial Inception kernel."""
        x = self._get_inception_features(original_images)
        y = self._get_inception_features(inpainted_images)
        if len(x) < 2 or len(y) < 2:
            raise ValueError("KID 至少需要每组 2 张图像")
        dim = x.shape[1]
        k_xx = (x @ x.T / dim + 1.0) ** 3
        k_yy = (y @ y.T / dim + 1.0) ** 3
        k_xy = (x @ y.T / dim + 1.0) ** 3
        m, n = len(x), len(y)
        xx = (k_xx.sum() - np.trace(k_xx)) / (m * (m - 1))
        yy = (k_yy.sum() - np.trace(k_yy)) / (n * (n - 1))
        return float(xx + yy - 2.0 * k_xy.mean())

    # ═══════════════════════════════════════════════════
    # 综合计算（单张）
    # ═══════════════════════════════════════════════════
    def calculate_all_single(self, original, inpainted, mask=None):
        """计算单张图像对的所有四项指标。

        Args:
            original: PIL Image (RGB)
            inpainted: PIL Image (RGB)
            mask: PIL Image (L), 可选，白=修复区域. 提供时额外计算 mask-only 指标

        Returns:
            dict: 全图 + 可选 mask 区域指标
        """
        psnr = self.calculate_psnr(original, inpainted)
        ssim = self.calculate_ssim(original, inpainted)
        lpips_val = self.calculate_lpips(original, inpainted)

        # This is a diagnostic paired feature distance, not a per-image FID.
        orig_feat = self._get_inception_features([original])[0]
        inp_feat = self._get_inception_features([inpainted])[0]
        inception_feature_l2 = float(np.sum((orig_feat - inp_feat) ** 2))

        result = {
            "psnr": float(psnr),
            "ssim": float(ssim),
            "lpips": float(lpips_val),
            "inception_feature_l2": inception_feature_l2,
        }

        # ---- Mask 区域指标 ----
        if mask is not None:
            mask_np = (np.array(mask).astype(np.float32) / 255.0 >= 0.5).astype(np.float32)
            result["psnr_mask"] = self._calc_psnr_masked(original, inpainted, mask_np)
            result["ssim_mask"] = self._calc_ssim_masked(original, inpainted, mask_np)
            result["lpips_mask"] = self._calc_lpips_masked(original, inpainted, mask_np)

        return result

    # ═══════════════════════════════════════════════════
    # Mask 区域指标（内部方法）
    # ═══════════════════════════════════════════════════

    def _calc_psnr_masked(self, original, inpainted, mask_np):
        """仅 mask 区域的 PSNR"""
        orig_np = np.array(original).astype(np.float32) / 255.0
        inp_np = np.array(inpainted).astype(np.float32) / 255.0

        if mask_np.ndim == 2:
            mask_3ch = np.stack([mask_np] * 3, axis=-1)
        else:
            mask_3ch = mask_np

        se = ((orig_np - inp_np) ** 2) * mask_3ch
        mse = se.sum() / (mask_3ch.sum() + 1e-10)
        if mse == 0:
            return float("inf")
        return float(20 * np.log10(1.0 / np.sqrt(mse)))

    def _calc_ssim_masked(self, original, inpainted, mask_np):
        """仅 mask 区域的 SSIM（使用 masked 滑窗均值）"""
        from scipy.ndimage import uniform_filter
        orig_np = np.array(original).astype(np.float32) / 255.0
        inp_np = np.array(inpainted).astype(np.float32) / 255.0

        if len(orig_np.shape) == 3 and orig_np.shape[2] == 3:
            orig_y = 0.299 * orig_np[:, :, 0] + 0.587 * orig_np[:, :, 1] + 0.114 * orig_np[:, :, 2]
            inp_y = 0.299 * inp_np[:, :, 0] + 0.587 * inp_np[:, :, 1] + 0.114 * inp_np[:, :, 2]
        else:
            orig_y, inp_y = orig_np, inp_np

        if mask_np.ndim == 3:
            mask_y = mask_np[:, :, 0] if mask_np.shape[2] >= 1 else mask_np
        else:
            mask_y = mask_np

        ws = 11
        C1 = (0.01) ** 2
        C2 = (0.03) ** 2

        mu1 = uniform_filter(orig_y, ws)
        mu2 = uniform_filter(inp_y, ws)
        mu1_sq, mu2_sq = mu1 ** 2, mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = uniform_filter(orig_y ** 2, ws) - mu1_sq
        sigma2_sq = uniform_filter(inp_y ** 2, ws) - mu2_sq
        sigma12 = uniform_filter(orig_y * inp_y, ws) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-10)

        mask_binary = (mask_y >= 0.5).astype(np.float32)
        ssim_masked = (ssim_map * mask_binary).sum() / (mask_binary.sum() + 1e-10)
        return float(ssim_masked)

    def _calc_lpips_masked(self, original, inpainted, mask_np):
        """Mask-weight the spatial LPIPS distance map over the missing region."""
        transform_fn = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        orig_t = transform_fn(original).unsqueeze(0).to(self.device)
        inp_t = transform_fn(inpainted).unsqueeze(0).to(self.device)

        mask_t = torch.from_numpy(mask_np).float()
        if mask_t.ndim == 2:
            mask_t = mask_t.unsqueeze(0).unsqueeze(0)
        else:
            mask_t = mask_t.permute(2, 0, 1).unsqueeze(0)
        mask_t = mask_t.to(self.device)

        with torch.no_grad():
            distance_map = self.lpips_fn(orig_t, inp_t)
            mask_t = F.interpolate(mask_t[:, :1], size=distance_map.shape[-2:], mode="nearest")
            distance = (distance_map * mask_t).sum() / mask_t.sum().clamp_min(1.0)
        return float(distance.item())
