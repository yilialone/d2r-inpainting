#!/usr/bin/env python3
"""
第二阶段训练器：带纹理先验的GAN模型

整合了原 train_stage2.py 和 train_stage2_1.py 的所有功能：
- Stage1 输出缓存 (大幅加速)
- 正确的梯度累积
- Sobel边缘归一化
- 可选 Hinge Loss / VGG感知损失
- GAN平衡监控
- 复合评估指标 (PSNR + SSIM + GenLoss)
"""

import os
import glob
import re
import math
import json
import hashlib
from typing import Dict, Optional, Tuple
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from accelerate import Accelerator
from PIL import Image

from diffusers import StableDiffusionInpaintPipeline
from peft import PeftModel

from models import SimpleUNetGeneratorWithTexture, SimpleUNetDiscriminator
from .common import (
    BudgetTracker,
    build_vgg_feature_extractor,
    composite_score,
    gan_loss,
    masked_perceptual_loss,
    masked_psnr_ssim,
    masked_texture_loss,
    parameter_report,
    sobel_edges,
    write_budget_report,
)

try:
    import torchvision.models as models
    from torchvision.models import VGG16_Weights
    from torchvision import transforms as T
    from torchvision.transforms import functional as TF
    from torchvision.transforms import InterpolationMode
    _TORCHVISION_AVAILABLE = True
except Exception:
    _TORCHVISION_AVAILABLE = False

try:
    import torchmetrics
    _TORCHMETRICS_AVAILABLE = True
except Exception:
    _TORCHMETRICS_AVAILABLE = False

os.environ["WANDB_MODE"] = "disabled"


class Stage2GANTrainer:
    """
    第二阶段训练器（完整版）。

    整合特性：
    - Stage1 输出缓存机制，避免重复推理
    - 正确的梯度累积 (accelerator.accumulate)
    - Sobel 边缘归一化
    - 可选 Hinge Loss / VGG 感知损失
    - GAN 平衡监控 (判别器准确率)
    - 复合评估指标 (PSNR + SSIM + GenLoss)
    - 早停与检查点管理
    """

    # v4: 生成器输入由 7 通道 [I_S1, filled, M] 改为 4 通道 [I_S1, M]（信息等价、首层卷积形状改变）。
    # 协议 ID 变更使带旧 ID 的 training_state.pt 拒绝自动续训，避免误用 7 通道检查点。
    PROTOCOL_ID = "d2r-stage2-paper-v4-4ch-I_S1-M-leakage-free-sample-id-logits-hinge"

    def __init__(
        self,
        stage1_checkpoint_dir: str = "./stage1_results/checkpoint-best",
        output_dir: str = "./stage2_results",
        cache_dir: Optional[str] = None,
        model_name: str = "runwayml/stable-diffusion-inpainting",
        resolution: int = 512,
        train_batch_size: int = 1,
        gradient_accumulation_steps: int = 1,
        learning_rate: float = 1e-4,
        resume_from_checkpoint: str = "auto",
        save_best_only: bool = True,
        early_stopping_patience: int = 20,
        seed: int = 42,
        lambda_gan: float = 0.1,
        lambda_l1: float = 50.0,
        lambda_texture: float = 10.0,
        lambda_perceptual: float = 0.1,
        use_perceptual_loss: bool = False,
        use_hinge_loss: bool = True,
        residual_scale: float = 0.3,
        stage1_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        prompt: Optional[str] = None,
        mixed_precision: str = "fp16",
        use_cache: bool = True,
        force_regen_cache: bool = False,
        augment_training: bool = True,
        composite_score_weights: Tuple[float, float, float] = (0.4, 0.3, 0.3),
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            mixed_precision=mixed_precision,
        )
        if self.accelerator.num_processes > 1 and self.accelerator.is_main_process:
            print(
                f"启用分布式 Stage2: {self.accelerator.num_processes} 张 GPU；"
                f"每卡 batch_size={train_batch_size}，全局有效 batch_size="
                f"{train_batch_size * self.accelerator.num_processes * gradient_accumulation_steps}"
            )

        self.stage1_checkpoint_dir = stage1_checkpoint_dir
        self.output_dir = output_dir
        self.cache_dir = cache_dir or os.path.join(output_dir, "stage1_cache")
        self.model_name = model_name
        self.resolution = resolution
        self.train_batch_size = train_batch_size
        self.learning_rate = learning_rate
        self.resume_from_checkpoint = resume_from_checkpoint
        self.save_best_only = save_best_only
        self.early_stopping_patience = early_stopping_patience
        self.seed = seed

        self.lambda_gan = lambda_gan
        self.lambda_l1 = lambda_l1
        self.lambda_texture = lambda_texture
        self.lambda_perceptual = lambda_perceptual
        self.use_perceptual_loss = use_perceptual_loss
        self.use_hinge_loss = use_hinge_loss
        self.residual_scale = residual_scale
        self.composite_score_weights = composite_score_weights

        self.stage1_inference_steps = stage1_inference_steps
        self.guidance_scale = guidance_scale
        self.use_cache = use_cache
        self.force_regen_cache = force_regen_cache
        self.augment_training = augment_training

        self.prompt = prompt or (
            "An ancient bronze mirror engraved with the character '山' at the center, "
            "with natural patina and slight wear, photographed in soft, high-resolution realistic lighting."
        )

        checkpoint_signature = []
        if os.path.isdir(stage1_checkpoint_dir):
            for name in sorted(os.listdir(stage1_checkpoint_dir)):
                path = os.path.join(stage1_checkpoint_dir, name)
                if os.path.isfile(path):
                    stat = os.stat(path)
                    checkpoint_signature.append((name, stat.st_size, stat.st_mtime_ns))
        cache_identity = {
            "stage1_protocol_id": "d2r-stage1-paper-v3-noisy-mask-clean-condition-attention-lora",
            "stage1_checkpoint": os.path.abspath(stage1_checkpoint_dir),
            "stage1_checkpoint_signature": checkpoint_signature,
            "model_name": model_name,
            "resolution": resolution,
            "steps": stage1_inference_steps,
            "guidance_scale": guidance_scale,
            "prompt": self.prompt,
            "seed": seed,
        }
        self.cache_namespace = hashlib.sha256(
            json.dumps(cache_identity, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()[:16]
        self.cache_run_dir = os.path.join(self.cache_dir, self.cache_namespace)

        os.makedirs(self.output_dir, exist_ok=True)
        if self.use_cache:
            os.makedirs(self.cache_run_dir, exist_ok=True)

        # ---------- 加载第一阶段模型 ----------
        print("加载第一阶段模型...")
        self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
        )
        self.pipeline.to(self.accelerator.device)
        self.pipeline.set_progress_bar_config(disable=True)

        self.pipeline.vae.eval()
        self.pipeline.text_encoder.eval()
        self.pipeline.vae.requires_grad_(False)
        self.pipeline.text_encoder.requires_grad_(False)

        if os.path.exists(os.path.join(stage1_checkpoint_dir, "adapter_config.json")):
            base_unet = self.pipeline.unet
            self.unet = PeftModel.from_pretrained(
                base_unet, stage1_checkpoint_dir, torch_dtype=torch.float32
            )
            self.pipeline.unet = self.unet
            print("成功加载第一阶段 LoRA 适配器")
        else:
            print("警告: 未找到第一阶段 LoRA，使用原始 UNet")
            self.unet = self.pipeline.unet

        for param in self.unet.parameters():
            param.requires_grad = False
        self.unet.eval()

        self.noise_scheduler = self.pipeline.scheduler

        # ---------- 第二阶段网络 ----------
        # 生成器输入 4 通道 [I_S1, M]：与端到端单阶段对照的 [masked, M] 逐位对应，
        # 使匹配对照唯一剩下的差别是"有无扩散先验 / 拆不拆两阶段"。
        self.generator = SimpleUNetGeneratorWithTexture(
            in_channels=4,
            residual_scale=residual_scale,
            base_channels=64,
            use_full_reconstruction=False,
            use_enhanced_encoder=True,
        )
        self.discriminator = SimpleUNetDiscriminator()

        # 参数统计：必须在 accelerator 包装 / Stage-1 缓存生成后释放 pipeline 之前导出，
        # 供论文 Table 4 的 "Trainable parameters" 列以及预算报告使用。
        self.parameter_counts = {
            "stage1_unet_lora": parameter_report(self.unet),
            "generator": parameter_report(self.generator),
            "discriminator": parameter_report(self.discriminator),
        }
        print(f"[参数] 冻结的 Stage-1 UNet(含LoRA) trainable={self.parameter_counts['stage1_unet_lora']['trainable']}; "
              f"生成器 trainable={self.parameter_counts['generator']['trainable']}; "
              f"判别器 trainable={self.parameter_counts['discriminator']['trainable']}")

        self.optimizer_g = AdamW(self.generator.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
        self.optimizer_d = AdamW(self.discriminator.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))

        # 训练状态
        self.start_epoch = 0
        self.best_val_loss = float("inf")
        self.best_composite_score = -float("inf")
        self.epochs_no_improve = 0
        self.optimizer_g_state_dict = None
        self.optimizer_d_state_dict = None
        self._dataloaders_prepared = False

        # 预算记账（Table 4: update count and GPU hours）
        self.global_step = 0
        self.completed_epochs = 0
        self.budget = BudgetTracker(world_size=getattr(self.accelerator, "num_processes", 1))

        # 可选 VGG 感知损失
        self.vgg = None
        self._vgg_normalize = None
        if self.use_perceptual_loss:
            self._init_vgg()

        # 可选 torchmetrics
        self.psnr_metric = None
        self.ssim_metric = None
        if _TORCHMETRICS_AVAILABLE and self.accelerator.is_main_process:
            self.psnr_metric = torchmetrics.PeakSignalNoiseRatio().to(self.accelerator.device)
            self.ssim_metric = torchmetrics.StructuralSimilarityIndexMeasure(data_range=2.0).to(self.accelerator.device)

        # 恢复检查点
        if self.resume_from_checkpoint is not None:
            self._load_checkpoint()

        # 准备模型 (accelerator)
        self._prepare_models()

        print("第二阶段训练初始化完成")
        if self.start_epoch > 0:
            print(f"从 epoch {self.start_epoch} 继续训练")
        if self.use_hinge_loss:
            print("  使用 Hinge 对抗损失")
        if self.use_perceptual_loss:
            print(f"  使用 VGG 感知损失 (权重: {self.lambda_perceptual})")

    def _init_vgg(self):
        if not _TORCHVISION_AVAILABLE:
            raise RuntimeError("use_perceptual_loss=True 但 torchvision 不可用。")
        self.vgg, self._vgg_normalize = build_vgg_feature_extractor(self.accelerator.device)

    def _prepare_models(self):
        self.generator, self.discriminator, self.optimizer_g, self.optimizer_d = self.accelerator.prepare(
            self.generator, self.discriminator, self.optimizer_g, self.optimizer_d
        )

    def _prepare_dataloaders(self, train_loader, val_loader=None):
        """Shard DataLoaders across ranks exactly once."""
        if self._dataloaders_prepared:
            return train_loader, val_loader
        if val_loader is None:
            train_loader = self.accelerator.prepare(train_loader)
        else:
            train_loader, val_loader = self.accelerator.prepare(train_loader, val_loader)
        self._dataloaders_prepared = True
        return train_loader, val_loader

    def _distributed_mean(self, total: float, count: int) -> float:
        """Average a locally accumulated statistic over all examples/ranks."""
        # Keep lightweight unit-test doubles and single-process callers
        # compatible even when they do not expose ``Accelerator.reduce``.
        if not hasattr(self.accelerator, "reduce"):
            return float(total) / max(1, int(count))
        stats = torch.tensor(
            [float(total), float(count)],
            device=self.accelerator.device,
            dtype=torch.float64,
        )
        stats = self.accelerator.reduce(stats, reduction="sum")
        return float((stats[0] / stats[1].clamp_min(1.0)).item())

    # ---------- 检查点管理 ----------
    def _find_latest_checkpoint(self):
        pattern = os.path.join(self.output_dir, "checkpoint-*")
        checkpoints = glob.glob(pattern)
        if not checkpoints:
            return None

        def get_epoch(path: str):
            m = re.search(r"checkpoint-epoch-(\d+)", path)
            if m:
                return int(m.group(1))
            return -1

        epoch_checkpoints = [path for path in checkpoints if get_epoch(path) >= 0]
        if epoch_checkpoints:
            return max(epoch_checkpoints, key=get_epoch)
        best_path = os.path.join(self.output_dir, "checkpoint-best")
        return best_path if os.path.isdir(best_path) else None

    def _load_checkpoint(self):
        if self.resume_from_checkpoint == "auto":
            checkpoint_path = self._find_latest_checkpoint()
            if checkpoint_path is None:
                print("未找到检查点，从头开始训练")
                return
        else:
            checkpoint_path = self.resume_from_checkpoint
        print(f"从检查点恢复: {checkpoint_path}")

        state_path = os.path.join(checkpoint_path, "training_state.pt")
        if os.path.exists(state_path):
            state_probe = torch.load(state_path, map_location="cpu")
            if state_probe.get("protocol_id") != self.PROTOCOL_ID:
                raise RuntimeError(
                    "检查点使用旧的 Stage2 输入/损失/缓存协议，禁止自动续训；"
                    "请使用 --resume_from_checkpoint none 并指定新的输出目录"
                )

        gen_path = os.path.join(checkpoint_path, "generator.pth")
        dis_path = os.path.join(checkpoint_path, "discriminator.pth")
        try:
            if os.path.exists(gen_path):
                self.generator.load_state_dict(torch.load(gen_path, map_location="cpu"), strict=True)
                print("加载生成器权重")
            if os.path.exists(dis_path):
                self.discriminator.load_state_dict(torch.load(dis_path, map_location="cpu"), strict=True)
                print("加载判别器权重")
        except Exception as e:
            print(f"加载模型失败: {e}")

        if os.path.exists(state_path):
            try:
                state = torch.load(state_path, map_location="cpu")
                self.start_epoch = int(state.get("epoch", 0)) + 1
                self.best_val_loss = float(state.get("best_val_loss", float("inf")))
                self.best_composite_score = float(state.get("best_composite_score", -float("inf")))
                self.epochs_no_improve = int(state.get("epochs_no_improve", 0))
                self.optimizer_g_state_dict = state.get("optimizer_g_state_dict")
                self.optimizer_d_state_dict = state.get("optimizer_d_state_dict")
                self.global_step = int(state.get("global_step", 0))
                print(f"恢复状态: epoch={self.start_epoch}, best_val_loss={self.best_val_loss:.4f}")
            except Exception as e:
                print(f"加载训练状态失败: {e}")

    def _restore_optimizer_states(self):
        if self.optimizer_g_state_dict:
            try:
                self.optimizer_g.load_state_dict(self.optimizer_g_state_dict)
                print("恢复生成器优化器状态")
            except Exception as e:
                print(f"恢复生成器优化器状态失败: {e}")
        if self.optimizer_d_state_dict:
            try:
                self.optimizer_d.load_state_dict(self.optimizer_d_state_dict)
                print("恢复判别器优化器状态")
            except Exception as e:
                print(f"恢复判别器优化器状态失败: {e}")
        self.optimizer_g_state_dict = None
        self.optimizer_d_state_dict = None

    def save_checkpoint(self, epoch, train_loss, val_loss=None, is_best=False, extra_info=None):
        if not self.accelerator.is_main_process:
            return
        if is_best:
            ckpt_dir = os.path.join(self.output_dir, "checkpoint-best")
        else:
            ckpt_dir = os.path.join(self.output_dir, f"checkpoint-epoch-{epoch}")
        os.makedirs(ckpt_dir, exist_ok=True)

        unwrap_g = self.accelerator.unwrap_model(self.generator)
        unwrap_d = self.accelerator.unwrap_model(self.discriminator)
        torch.save(unwrap_g.state_dict(), os.path.join(ckpt_dir, "generator.pth"))
        torch.save(unwrap_d.state_dict(), os.path.join(ckpt_dir, "discriminator.pth"))

        state = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": self.best_val_loss,
            "best_composite_score": self.best_composite_score,
            "epochs_no_improve": self.epochs_no_improve,
            "optimizer_g_state_dict": self.optimizer_g.state_dict(),
            "optimizer_d_state_dict": self.optimizer_d.state_dict(),
            "seed": self.seed,
            "cache_namespace": self.cache_namespace,
            "gan_objective": "hinge_logits" if self.use_hinge_loss else "bce_with_logits",
            "protocol_id": self.PROTOCOL_ID,
            # 预算口径：Table 4 "Update count and GPU hours"
            "global_step": int(self.global_step),
            "epochs_completed": int(self.completed_epochs),
            "train_seconds": round(float(self.budget.elapsed()), 3),
        }
        torch.save(state, os.path.join(ckpt_dir, "training_state.pt"))

        if is_best:
            with open(os.path.join(ckpt_dir, "BEST_MODEL"), "w") as f:
                f.write(f"Best model at epoch {epoch} with val_loss {val_loss}\n")
            if extra_info is not None:
                info_file = os.path.join(ckpt_dir, "validation_metrics.txt")
                with open(info_file, "w") as f:
                    f.write(f"Best Model at Epoch {epoch}\n")
                    f.write(f"Composite Score: {extra_info.get('composite_score', 'N/A'):.4f}\n")
                    f.write(f"PSNR: {extra_info.get('psnr', 'N/A'):.2f} dB\n")
                    f.write(f"SSIM: {extra_info.get('ssim', 'N/A'):.3f}\n")
                    f.write(f"Generator Loss: {val_loss:.4f}\n")

        print(f"检查点已保存: {ckpt_dir}")

    # ---------- 图像/张量转换 ----------
    def _tensor_to_pil(self, x: torch.Tensor) -> Image.Image:
        x = x.detach().float().cpu().clamp(-1, 1)
        if x.shape[0] == 1:
            x = x.repeat(3, 1, 1)
        x01 = (x + 1.0) / 2.0
        x01 = x01.clamp(0, 1)
        if _TORCHVISION_AVAILABLE:
            return T.ToPILImage()(x01)
        arr = (x01.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        return Image.fromarray(arr)

    def _mask_to_pil(self, x: torch.Tensor) -> Image.Image:
        x = x.detach().float().cpu()
        if x.shape[0] > 1:
            x = x[:1]
        x = x.clamp(0, 1)
        x = x.repeat(3, 1, 1)
        if _TORCHVISION_AVAILABLE:
            return T.ToPILImage()(x)
        arr = (x.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        return Image.fromarray(arr)

    def _pil_to_tensor(self, img: Image.Image, device: torch.device) -> torch.Tensor:
        if _TORCHVISION_AVAILABLE:
            tensor = T.ToTensor()(img).unsqueeze(0)
        else:
            arr = np.array(img).astype(np.float32) / 255.0
            if arr.ndim == 2:
                arr = arr[:, :, None]
            tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(device=device, dtype=torch.float32)
        tensor = tensor * 2.0 - 1.0
        return tensor

    # ---------- Stage1 输出缓存 ----------
    def _get_cache_path(self, split: str, sample_id: str) -> str:
        """Return a stable cache path keyed by sample identity, never loader order."""
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(sample_id)).strip("._")[:80]
        digest = hashlib.sha256(str(sample_id).encode("utf-8")).hexdigest()[:16]
        return os.path.join(self.cache_run_dir, f"{split}_{safe_id}_{digest}.pt")

    @staticmethod
    def _batch_sample_ids(batch) -> list:
        sample_ids = batch.get("sample_id")
        if sample_ids is None:
            raise KeyError("数据批次缺少 sample_id；请使用 dataset.create_dataloaders 创建数据加载器")
        return [str(sample_id) for sample_id in sample_ids]

    def _generate_stage1_output_single(
        self, image_pil: Image.Image, mask_pil: Image.Image, sample_seed: int
    ) -> torch.Tensor:
        generator_device = self.accelerator.device.type if self.accelerator.device.type == "cuda" else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(sample_seed)
        with torch.no_grad():
            result = self.pipeline(
                prompt=self.prompt,
                image=image_pil,
                mask_image=mask_pil,
                num_inference_steps=self.stage1_inference_steps,
                guidance_scale=self.guidance_scale,
                generator=generator,
            )
        out_img = result.images[0]
        out_tensor = self._pil_to_tensor(out_img, self.accelerator.device)
        return out_tensor.cpu()

    def _prepare_stage1_cache(self, loader, split: str):
        # The training loader uses shuffle=True and may use drop_last=True. Build
        # a deterministic, non-dropping view here so every sample receives a
        # cache entry; otherwise a different shuffle can request the one sample
        # omitted from the cache (e.g. 433 samples -> 432 cached at batch 4).
        cache_loader = DataLoader(
            loader.dataset,
            batch_size=loader.batch_size or 1,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
        )
        if getattr(self.accelerator, "is_main_process", True):
            print(
                f"生成 stage1 缓存: {split} (共 {len(cache_loader)} 个 batch, "
                f"batch_size={cache_loader.batch_size}, "
                f"{self.accelerator.num_processes} 个进程分片)"
            )
        count = 0
        global_index = 0
        for batch in tqdm(
            cache_loader,
            desc=f"缓存 {split} (GPU {self.accelerator.process_index})",
            disable=not getattr(self.accelerator, "is_main_process", True),
        ):
            images = batch["image"]
            masks = batch["mask"]
            sample_ids = self._batch_sample_ids(batch)
            batch_size = images.shape[0]
            for i in range(batch_size):
                sample_index = global_index
                global_index += 1
                # Cache files are keyed by sample_id, so each rank can safely
                # generate a disjoint shard in parallel on a shared filesystem.
                if sample_index % self.accelerator.num_processes != self.accelerator.process_index:
                    continue
                img_tensor = images[i]
                mask_tensor = masks[i]
                masked_tensor = img_tensor * (1 - mask_tensor)
                img_pil = self._tensor_to_pil(masked_tensor)
                mask_pil = self._mask_to_pil(mask_tensor)
                sample_id = sample_ids[i]
                cache_path = self._get_cache_path(split, sample_id)
                if not os.path.exists(cache_path) or self.force_regen_cache:
                    seed_offset = int(hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:8], 16)
                    stage1_out = self._generate_stage1_output_single(
                        img_pil, mask_pil, (self.seed + seed_offset) % (2**31)
                    )
                    torch.save(stage1_out, cache_path)
                count += 1
        if hasattr(self.accelerator, "reduce"):
            count_tensor = torch.tensor(
                float(count), device=self.accelerator.device, dtype=torch.float64
            )
            total_count = int(self.accelerator.reduce(count_tensor, reduction="sum").item())
        else:
            total_count = count
        if getattr(self.accelerator, "is_main_process", True):
            print(f"缓存 {split} 完成，共 {total_count} 个样本；命名空间={self.cache_namespace}")

    def _load_stage1_batch(self, batch, split: str) -> torch.Tensor:
        tensors = []
        for sample_id in self._batch_sample_ids(batch):
            cache_path = self._get_cache_path(split, sample_id)
            if not os.path.exists(cache_path):
                raise FileNotFoundError(
                    f"缓存缺失: {cache_path}；样本={sample_id}。请设置 --force_regen_cache 重新生成"
                )
            tensor = torch.load(cache_path, map_location="cpu")
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
            tensors.append(tensor)
        return torch.cat(tensors, dim=0).to(self.accelerator.device, dtype=torch.float32)

    def _augment_triplet(self, images, masks, stage1_output):
        """Apply identical random geometry to target, mask, and cached Stage-1 output."""
        if not self.augment_training:
            return images, masks, stage1_output
        if not _TORCHVISION_AVAILABLE:
            raise RuntimeError("Stage2 数据增强需要 torchvision；也可使用 --no-augment 禁用")
        out_images, out_masks, out_stage1 = [], [], []
        for image, mask, stage1 in zip(images, masks, stage1_output):
            if torch.rand(()) < 0.5:
                image, mask, stage1 = TF.hflip(image), TF.hflip(mask), TF.hflip(stage1)
            angle = float(torch.empty(1).uniform_(-5.0, 5.0).item())
            image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR, fill=-1.0)
            mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)
            stage1 = TF.rotate(stage1, angle, interpolation=InterpolationMode.BILINEAR, fill=-1.0)
            out_images.append(image)
            out_masks.append((mask >= 0.5).to(mask.dtype))
            out_stage1.append(stage1)
        return torch.stack(out_images), torch.stack(out_masks), torch.stack(out_stage1)

    # ---------- 损失函数 ----------
    # 以下实现全部委托给 training/common.py，保证与端到端单阶段对照（R2-2）
    # 使用完全相同的损失与指标定义；本类的方法名与签名保持不变。
    def _sobel_edges(self, x: torch.Tensor) -> torch.Tensor:
        return sobel_edges(x)

    def compute_texture_loss(self, generated: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return masked_texture_loss(generated, target, mask)

    def compute_perceptual_loss(self, generated: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return masked_perceptual_loss(self.vgg, self._vgg_normalize, generated, target, mask)

    def _refine(self, stage1_out: torch.Tensor, orig_img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # 生成器输入 = [I_S1, M]（4 通道）。
        # 代码审计 🔴-1 + 通道精简：早期版本拼成 [I_S1, orig×(1−M)+I_S1×M, M] 共 7 通道，
        # 但中间那 3 个通道在洞内恒等于 I_S1、在洞外只是已知像素的拷贝，不携带新信息；
        # 且若原图不扣洞，洞内真值会经纹理编码器泄漏。现在 4 通道形式下，
        # 洞内像素只来自 Stage-1 预测、洞外只来自已知像素，网络任意一路都拿不到掩膜内真值。
        # 注意：此变更改变了输入分布与首层卷积形状，旧 7 通道检查点不再适用，必须重新训练。
        residual = self.generator(stage1_out, mask)
        # 仅 mask 区域 = stage1 + residual；非 mask 区域严格等于原图
        refined = orig_img * (1.0 - mask) + (stage1_out + residual) * mask
        return refined.clamp(-1, 1)

    def _compute_gan_loss(self, disc_fake: torch.Tensor, disc_real: torch.Tensor, for_generator: bool) -> torch.Tensor:
        return gan_loss(disc_fake, disc_real, for_generator, self.use_hinge_loss)

    def compute_generator_loss(
        self, stage1_out: torch.Tensor, orig_img: torch.Tensor, mask: torch.Tensor, target_img: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float], torch.Tensor]:
        refined = self._refine(stage1_out, orig_img, mask)
        fake = refined * mask
        disc_fake = self.discriminator(fake)
        disc_real = self.discriminator(target_img * mask)

        gen_adv = self._compute_gan_loss(disc_fake, disc_real, for_generator=True)
        l1_loss = F.l1_loss(refined * mask, target_img * mask)
        tex_loss = self.compute_texture_loss(refined, target_img, mask)

        perc_loss = torch.tensor(0.0, device=refined.device)
        if self.use_perceptual_loss:
            perc_loss = self.compute_perceptual_loss(refined, target_img, mask)

        total = (self.lambda_gan * gen_adv +
                 self.lambda_l1 * l1_loss +
                 self.lambda_texture * tex_loss +
                 self.lambda_perceptual * perc_loss)

        # GAN平衡监控
        disc_accuracy = ((disc_fake < 0.0).float().mean() + (disc_real > 0.0).float().mean()) / 2.0

        loss_dict = {
            "gen_adv": float(gen_adv.detach().item()),
            "l1": float(l1_loss.detach().item()),
            "texture": float(tex_loss.detach().item()),
            "perceptual": float(perc_loss.detach().item()),
            "total_gen": float(total.detach().item()),
            "disc_accuracy": float(disc_accuracy.detach().item()),
        }
        return total, loss_dict, refined

    def compute_discriminator_loss(
        self, stage1_out: torch.Tensor, orig_img: torch.Tensor, mask: torch.Tensor, target_img: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        with torch.no_grad():
            refined = self._refine(stage1_out, orig_img, mask)
        fake_input = (refined * mask).detach()
        real_input = (target_img * mask).detach()
        disc_fake = self.discriminator(fake_input)
        disc_real = self.discriminator(real_input)

        total = self._compute_gan_loss(disc_fake, disc_real, for_generator=False)
        loss_dict = {
            "disc_fake": float(F.binary_cross_entropy_with_logits(
                disc_fake, torch.zeros_like(disc_fake)).detach().item()),
            "disc_real": float(F.binary_cross_entropy_with_logits(
                disc_real, torch.ones_like(disc_real)).detach().item()),
            "total_disc": float(total.detach().item()),
        }
        return total, loss_dict

    # ---------- 验证指标 ----------
    def compute_metrics(self, refined: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
        # 与单阶段对照共用同一代理指标实现（training/common.py）。
        return masked_psnr_ssim(refined, target, mask)

    # ---------- 训练与验证 ----------
    def validate(self, val_loader):
        self.generator.eval()
        self.discriminator.eval()
        total_gen_loss = 0.0
        total_disc_loss = 0.0
        total_psnr = 0.0
        total_ssim = 0.0
        total_examples = 0

        with torch.no_grad():
            for batch in tqdm(
                val_loader,
                desc="验证",
                disable=not getattr(self.accelerator, "is_main_process", True),
            ):
                images = batch["image"].to(self.accelerator.device, dtype=torch.float32)
                masks = batch["mask"].to(self.accelerator.device, dtype=torch.float32)
                stage1_output = self._load_stage1_batch(batch, "val")

                gen_loss, _, refined = self.compute_generator_loss(
                    stage1_output, images, masks, images
                )
                disc_loss, _ = self.compute_discriminator_loss(
                    stage1_output, images, masks, images
                )
                metrics = self.compute_metrics(refined, images, masks)

                batch_size = images.shape[0]
                total_gen_loss += gen_loss.item() * batch_size
                total_disc_loss += disc_loss.item() * batch_size
                total_psnr += metrics["psnr"] * batch_size
                total_ssim += metrics["ssim"] * batch_size
                total_examples += batch_size

        avg_gen_loss = self._distributed_mean(total_gen_loss, total_examples)
        avg_disc_loss = self._distributed_mean(total_disc_loss, total_examples)
        avg_psnr = self._distributed_mean(total_psnr, total_examples)
        avg_ssim = self._distributed_mean(total_ssim, total_examples)

        composite = composite_score(avg_psnr, avg_ssim, avg_gen_loss,
                                    self.composite_score_weights)

        return {
            "gen_loss": avg_gen_loss,
            "disc_loss": avg_disc_loss,
            "psnr": avg_psnr,
            "ssim": avg_ssim,
            "composite_score": composite,
        }

    def train_epoch(self, train_loader, epoch: int):
        self.generator.train()
        self.discriminator.train()

        total_gen_loss = 0.0
        total_disc_loss = 0.0
        total_examples = 0
        progress_bar = tqdm(
            train_loader,
            desc=f"第二阶段 Epoch {epoch}",
            disable=not getattr(self.accelerator, "is_main_process", True),
        )

        for step, batch in enumerate(progress_bar):
            images = batch["image"].to(self.accelerator.device, dtype=torch.float32)
            masks = batch["mask"].to(self.accelerator.device, dtype=torch.float32)
            stage1_output = self._load_stage1_batch(batch, "train")
            images, masks, stage1_output = self._augment_triplet(images, masks, stage1_output)

            with self.accelerator.accumulate(self.generator, self.discriminator):
                self.optimizer_d.zero_grad(set_to_none=True)
                disc_loss, _ = self.compute_discriminator_loss(
                    stage1_output, images, masks, images
                )
                self.accelerator.backward(disc_loss)
                self.optimizer_d.step()

                for parameter in self.discriminator.parameters():
                    parameter.requires_grad_(False)
                self.optimizer_g.zero_grad(set_to_none=True)
                gen_loss, loss_dict, _ = self.compute_generator_loss(
                    stage1_output, images, masks, images
                )
                self.accelerator.backward(gen_loss)
                self.optimizer_g.step()
                for parameter in self.discriminator.parameters():
                    parameter.requires_grad_(True)

                if getattr(self.accelerator, "sync_gradients", True):
                    self.global_step = int(getattr(self, "global_step", 0)) + 1

            batch_size = images.shape[0]
            total_gen_loss += gen_loss.item() * batch_size
            total_disc_loss += disc_loss.item() * batch_size
            total_examples += batch_size

            progress_bar.set_postfix({
                "gen": f"{gen_loss.item():.4f}",
                "disc": f"{disc_loss.item():.4f}",
                "l1": f"{loss_dict['l1']:.4f}",
                "tex": f"{loss_dict['texture']:.4f}",
                "D-acc": f"{loss_dict.get('disc_accuracy', 0):.3f}",
            })

        return (
            self._distributed_mean(total_gen_loss, total_examples),
            self._distributed_mean(total_disc_loss, total_examples),
        )

    def train(self, train_loader, val_loader=None, num_epochs=50):
        print("开始第二阶段训练（带纹理先验）")
        train_loader, val_loader = self._prepare_dataloaders(train_loader, val_loader)
        if getattr(train_loader.dataset, "augment", False):
            raise ValueError(
                "Stage2 DataLoader 必须设置 augment=False；同步增强由 Stage2GANTrainer 执行"
            )

        # 生成/加载 Stage1 缓存
        if self.use_cache:
            # 缓存生成阶段只需要扩散管线；生成器与判别器此时若留在 GPU 上会与本就不宽裕的
            # 显存（本机 8GB）争抢，实测把单样本缓存时间从约 5 秒拖到约 160 秒。
            # 因此先把它们挪到 CPU，缓存结束后再随 accelerator.prepare 回到 GPU。
            device = self.accelerator.device
            if device.type == "cuda":
                print("缓存生成阶段：把生成器/判别器暂存到 CPU 以腾出显存")
                self.generator.to("cpu")
                self.discriminator.to("cpu")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            # Every rank generates a disjoint sample shard. This keeps cache
            # generation deterministic while using both GPUs instead of
            # leaving non-main ranks idle for the entire precomputation.
            self._prepare_stage1_cache(train_loader, "train")
            if val_loader is not None:
                self._prepare_stage1_cache(val_loader, "val")
            self.accelerator.wait_for_everyone()

            self.pipeline = None
            self.unet = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # 缓存期间挪到 CPU 的生成器/判别器此时回到训练设备
            if self.accelerator.device.type == "cuda":
                self.generator.to(self.accelerator.device)
                self.discriminator.to(self.accelerator.device)
            self.accelerator.wait_for_everyone()
        else:
            raise NotImplementedError("当前版本仅支持 use_cache=True 模式")

        self._restore_optimizer_states()
        self.budget.start()

        last_epoch = self.start_epoch - 1
        val_metrics = None
        train_gen_loss = float("nan")
        for epoch in range(self.start_epoch, num_epochs):
            last_epoch = epoch
            train_gen_loss, train_disc_loss = self.train_epoch(train_loader, epoch)
            self.completed_epochs = epoch + 1

            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                if self.accelerator.is_main_process:
                    print(
                        f"Epoch {epoch}: "
                        f"train_gen={train_gen_loss:.4f}, train_disc={train_disc_loss:.4f}, "
                        f"val_gen={val_metrics['gen_loss']:.4f}, val_disc={val_metrics['disc_loss']:.4f}, "
                        f"PSNR={val_metrics['psnr']:.2f}dB, SSIM={val_metrics['ssim']:.3f}, "
                        f"Composite={val_metrics['composite_score']:.4f}"
                    )

                if val_metrics["gen_loss"] < self.best_val_loss:
                    self.best_composite_score = val_metrics["composite_score"]
                    self.best_val_loss = val_metrics["gen_loss"]
                    self.epochs_no_improve = 0
                    self.save_checkpoint(epoch, train_gen_loss, val_metrics["gen_loss"], is_best=True,
                                         extra_info={"composite_score": val_metrics["composite_score"],
                                                     "psnr": val_metrics["psnr"], "ssim": val_metrics["ssim"]})
                    if self.accelerator.is_main_process:
                        print(f"新最佳模型! val_gen={val_metrics['gen_loss']:.4f}")
                else:
                    self.epochs_no_improve += 1
                    if self.accelerator.is_main_process:
                        print(f"分数未改善，连续 {self.epochs_no_improve} 轮")

                if train_disc_loss < 0.1 and self.accelerator.is_main_process:
                    print("警告: 判别器可能过强")
            else:
                if self.accelerator.is_main_process:
                    print(f"Epoch {epoch}: train_gen={train_gen_loss:.4f}, train_disc={train_disc_loss:.4f}")

            if self.accelerator.is_main_process and epoch % 10 == 0:
                self.save_checkpoint(epoch, train_gen_loss,
                                     None if val_metrics is None else val_metrics["gen_loss"])

            self.accelerator.wait_for_everyone()

            if self.early_stopping_patience and self.epochs_no_improve >= self.early_stopping_patience:
                if self.accelerator.is_main_process:
                    print(f"早停: {self.epochs_no_improve} 轮无改进")
                break

        if self.accelerator.is_main_process and last_epoch >= self.start_epoch:
            final_val = None if val_metrics is None else val_metrics["gen_loss"]
            self.save_checkpoint(last_epoch, train_gen_loss, final_val)
        self.accelerator.wait_for_everyone()
        self.budget.stop()
        self._write_budget_report()
        if self.accelerator.is_main_process:
            print("=== 第二阶段训练完成 ===")

    def _write_budget_report(self):
        """写出 Table 4 所需的预算记录（参数量 / 更新步数 / GPU·h）。"""
        if not self.accelerator.is_main_process:
            return None
        record = {
            "stage": "stage2_gan_refinement",
            "protocol_id": self.PROTOCOL_ID,
            "model_definition": (
                "SimpleUNetGeneratorWithTexture (residual refinement over the frozen Stage-1 output): "
                "7 input channels [Stage-1 output, masked original, mask], enhanced encoder with "
                "SE blocks and self-attention, 12-channel multi-filter texture encoder "
                "(3 RGB + 3 Canny + 1 Sobel + 1 Laplacian + 4 Gabor) and texture attention gating; "
                "learnable residual scaling initialised to 0.3"
            ),
            "losses": (
                "hinge GAN (lambda=%.3g) + masked L1 (lambda=%.3g) + masked normalised Sobel texture "
                "(lambda=%.3g)" % (self.lambda_gan, self.lambda_l1, self.lambda_texture)
            ),
            "parameters": self.parameter_counts,
            "update_count_definition": (
                "generator optimizer steps; one discriminator update and one generator update per step"
            ),
            "completed_updates": int(self.global_step),
            "completed_epochs": int(self.completed_epochs),
            "resumed_from_epoch": int(self.start_epoch),
            "optimizer": {
                "name": "AdamW",
                "betas": [0.5, 0.999],
                "learning_rate": self.learning_rate,
                "gradient_accumulation_steps": self.accelerator.gradient_accumulation_steps,
                "mixed_precision": str(self.accelerator.mixed_precision),
            },
            "data": {
                "resolution": self.resolution,
                "train_batch_size_per_process": self.train_batch_size,
                "augmentation": "synchronised hflip and +/-5 degree rotation on image, mask and Stage-1 cache"
                if self.augment_training else "disabled",
                "stage1_cache_namespace": self.cache_namespace,
            },
            "selection_rule": (
                "best checkpoint by minimum validation generator loss; composite score recorded alongside"
            ),
            "seed": self.seed,
            "budget": self.budget.summary(),
        }
        path = write_budget_report(self.output_dir, record)
        print(f"预算报告已写出: {path}")
        return path
