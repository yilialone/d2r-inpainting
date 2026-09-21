#!/usr/bin/env python3
"""
数据集加载与预处理
"""

import os
import csv
import random
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision.transforms import ColorJitter, InterpolationMode
import numpy as np
import re


def _seed_worker(worker_id):
    """Seed Python/NumPy RNGs in each DataLoader worker (Windows-safe)."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class InpaintingDataset(Dataset):
    """图像修复数据集

    Args:
        image_dir: 原图目录 (mirror{数字}.jpg)
        mask_dir: mask目录 (img{8位数字}.png)
        size: 输出尺寸
        augment: 是否数据增强
    """

    def __init__(self, image_dir, mask_dir, size=512, transform=None, augment=False,
                 seed=42, manifest_path=None, manifest_split=None,
                 photometric_jitter=0.0):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.size = size
        self.augment = augment
        self.seed = seed
        self.manifest_split = manifest_split
        # 光度增强（显式 opt-in，默认 0 = 关闭，论文协议不变）。
        # 只做亮度/对比度/饱和度抖动，不移动任何像素，因此 mask 无需同步，
        # 也不存在几何对齐风险——这与"只对图像做旋转"的写法有本质区别。
        self.photometric_jitter = float(photometric_jitter or 0.0)
        self._jitter = (
            ColorJitter(brightness=self.photometric_jitter,
                        contrast=self.photometric_jitter,
                        saturation=self.photometric_jitter / 2.0)
            if self.photometric_jitter > 0 else None
        )

        if manifest_path:
            self.samples = self._load_manifest(manifest_path)
        else:
            image_files = sorted(
                [f for f in os.listdir(image_dir) if f.startswith('mirror') and f.lower().endswith(('.jpg', '.jpeg', '.png'))],
                key=lambda x: int(re.search(r'mirror(\d+)', x).group(1))
            )
            mask_files = sorted(
                [f for f in os.listdir(mask_dir) if f.startswith('img') and f.lower().endswith(('.png', '.jpg', '.jpeg'))],
                key=lambda x: int(re.search(r'img(\d+)', x).group(1))
            )
            if len(image_files) != len(mask_files):
                raise ValueError(
                    f"原图数量({len(image_files)})和mask数量({len(mask_files)})不匹配"
                )
            self.samples = [
                {
                    "image_path": os.path.join(image_dir, image_name),
                    "mask_path": os.path.join(mask_dir, mask_name),
                    "image_name": image_name,
                    "mask_name": mask_name,
                    "sample_id": f"{os.path.splitext(image_name)[0]}__{os.path.splitext(mask_name)[0]}",
                }
                for image_name, mask_name in zip(image_files, mask_files)
            ]

        if not self.samples:
            raise ValueError("数据集中没有可用的图像-mask对")
        print(f"找到 {len(self.samples)} 对图像-mask对")

    @staticmethod
    def _resolve_manifest_path(value, manifest_dir):
        value = os.path.expandvars(os.path.expanduser(value.strip()))
        return value if os.path.isabs(value) else os.path.normpath(os.path.join(manifest_dir, value))

    def _load_manifest(self, manifest_path):
        manifest_path = os.path.abspath(manifest_path)
        manifest_dir = os.path.dirname(manifest_path)
        samples = []
        with open(manifest_path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            required = {"sample_id", "image_path", "mask_path"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"manifest 缺少字段: {sorted(missing)}")
            for row in reader:
                if self.manifest_split and row.get("split") and row["split"].strip() != self.manifest_split:
                    continue
                image_path = self._resolve_manifest_path(row["image_path"], manifest_dir)
                mask_path = self._resolve_manifest_path(row["mask_path"], manifest_dir)
                if not os.path.isfile(image_path) or not os.path.isfile(mask_path):
                    raise FileNotFoundError(f"manifest 文件不存在: {image_path} / {mask_path}")
                samples.append({
                    "image_path": image_path,
                    "mask_path": mask_path,
                    "image_name": os.path.basename(image_path),
                    "mask_name": os.path.basename(mask_path),
                    "sample_id": row["sample_id"].strip(),
                })
        sample_ids = [sample["sample_id"] for sample in samples]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("manifest 中 sample_id 不唯一")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_name = sample["image_name"]
        image_path = sample["image_path"]
        image = Image.open(image_path).convert("RGB")

        mask_name = sample["mask_name"]
        mask_path = sample["mask_path"]
        mask = Image.open(mask_path).convert("L")

        if image.size != (self.size, self.size):
            image = image.resize((self.size, self.size), Image.LANCZOS)
        if mask.size != (self.size, self.size):
            mask = mask.resize((self.size, self.size), Image.NEAREST)

        if self.augment:
            # 所有几何增强必须对 image 和 mask 使用完全相同的随机参数。
            if torch.rand(1) > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            angle = float(torch.empty(1).uniform_(-5.0, 5.0).item())
            image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR, fill=0)
            mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST, fill=0)
            if self._jitter is not None:
                # 光度增强只作用于图像：它不改变任何像素的位置，mask 无需同步。
                # 放在几何变换之后，使抖动幅度的语义仍是在 [0,1] 图像上定义的。
                image = self._jitter(image)

        # Stable Diffusion VAE、Stage-2 和推理路径统一使用 [-1, 1]。
        image_tensor = torch.from_numpy(np.array(image)).float().permute(2, 0, 1) / 127.5 - 1.0
        mask_tensor = torch.from_numpy(np.array(mask)).float().unsqueeze(0) / 255.0
        mask_tensor = (mask_tensor > 0.5).float()

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "prompt": "",
            "image_name": img_name,
            "mask_name": mask_name,
            "sample_id": sample["sample_id"],
            "image_path": image_path,
            "mask_path": mask_path,
        }


def create_dataloaders(train_image_dir, train_mask_dir, val_image_dir=None, val_mask_dir=None,
                       batch_size=2, num_workers=4, size=512, augment=True, seed=42,
                       train_manifest=None, val_manifest=None, photometric_jitter=0.0):
    """创建训练和验证数据加载器。

    ``num_workers`` 默认 4；设为 0 会在主进程内取数据，便于在限制多进程
    （无法创建命名管道）的环境中运行，例如受限沙箱或部分 Windows 调试场景。
    """
    train_dataset = InpaintingDataset(
        train_image_dir, train_mask_dir, size=size, augment=augment,
        seed=seed, manifest_path=train_manifest, manifest_split="train",
        photometric_jitter=photometric_jitter,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        worker_init_fn=_seed_worker, generator=generator,
    )

    val_loader = None
    if val_image_dir and val_mask_dir:
        val_dataset = InpaintingDataset(
            val_image_dir, val_mask_dir, size=size, augment=False,
            seed=seed, manifest_path=val_manifest, manifest_split="val",
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, drop_last=False,
            worker_init_fn=_seed_worker, generator=generator,
        )

    print(f"训练集: {len(train_dataset)} 样本")
    if val_loader:
        print(f"验证集: {len(val_dataset)} 样本")

    return train_loader, val_loader
