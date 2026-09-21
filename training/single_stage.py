#!/usr/bin/env python3
"""端到端单阶段对照训练器（审稿意见 R2-2）。

这个对照要回答的问题是：
    在**同样的语料、同样的骨干容量、同样的优化预算**下，把修复拆成
    "扩散先验 + 残差细化" 两阶段，是否真的比"单个网络端到端训练"更好？

因此本训练器与 D2R 的关系是刻意对齐的：

- **数据**：同一 train/val 划分、同一 512×512 分辨率、同一同步几何增强
  （hflip + ±5° 旋转），同一 batch size 与梯度累积（有效 batch size = 1）。
- **损失**：与 Stage-2 完全相同的三项 —— hinge 对抗、掩膜内 L1、掩膜内归一化
  Sobel 纹理；实现全部来自 `training/common.py`，不另写一份。
- **优化器**：AdamW，lr = 1e-4，betas = (0.5, 0.999)（与 Stage-2 相同）。
- **选模规则**：与 Stage-2 相同 —— 取验证集生成器损失最小的检查点；早停 patience 20。
- **预算**：默认按**更新步数**匹配参考 D2R 运行（Stage-1 + Stage-2 的实际优化步数之和），
  也可改用 GPU·h 匹配；两种情况都会把目标值与实际值写入 `budget_report.json`。

区别（必须在论文中写明）：单阶段模型没有 Stage-1 扩散先验，输入是
4 通道的"扣洞图像 + mask"，输出是对缺失内容的直接预测；它也**不读取 Stage-1 缓存**，
因此是完全独立的一次训练。

用法（详见 SINGLE_STAGE.md）：
    python train.py --mode single_stage --seed 2026 \\
        --single_stage_output_dir single_stage_results_seed2026 \\
        --single_stage_budget_ref_stage1 stage1_results_seed2026 \\
        --single_stage_budget_ref_stage2 stage2_results_seed2026
"""

import glob
import json
import os
import re
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import SimpleUNetDiscriminator, SingleStageInpaintingGenerator

from .common import (
    BudgetTracker,
    build_vgg_feature_extractor,
    composite_score,
    gan_loss,
    masked_perceptual_loss,
    masked_psnr_ssim,
    masked_texture_loss,
    parameter_report,
    write_budget_report,
)

try:
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


class SingleStageGANTrainer:
    """单阶段端到端对照：``(masked image, mask) → restored image``，一次训练、一次前向。"""

    PROTOCOL_ID = "d2r-single-stage-paper-v1-direct-4ch-masked-input"

    def __init__(
        self,
        output_dir: str = "./single_stage_results",
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
        mixed_precision: str = "fp16",
        augment_training: bool = True,
        base_channels: int = 64,
        composite_score_weights: Tuple[float, float, float] = (0.4, 0.3, 0.3),
        max_epochs: int = 100,
        target_steps: Optional[int] = None,
        target_gpu_hours: Optional[float] = None,
        budget_reference: Optional[dict] = None,
        max_train_minutes: Optional[float] = None,
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            mixed_precision=mixed_precision,
        )
        if self.accelerator.num_processes > 1 and self.accelerator.is_main_process:
            print(
                f"启用分布式单阶段训练: {self.accelerator.num_processes} 张 GPU；"
                f"每卡 batch_size={train_batch_size}，全局有效 batch_size="
                f"{train_batch_size * self.accelerator.num_processes * gradient_accumulation_steps}"
            )

        self.output_dir = output_dir
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
        self.augment_training = augment_training
        self.base_channels = base_channels
        self.composite_score_weights = composite_score_weights

        self.max_epochs = max_epochs
        self.target_steps = None if target_steps is None else int(target_steps)
        # 墙钟预算（分钟）：达到后停止但保留 best，用于在有限算力上保证有产物。
        self.max_train_minutes = float(max_train_minutes) if max_train_minutes else None
        self.target_gpu_hours = None if target_gpu_hours is None else float(target_gpu_hours)
        self.budget_reference = budget_reference

        # ---------- 模型 ----------
        # 注意：这里不加载 Stable Diffusion、不加载 Stage-1 LoRA、也不读取 Stage-1 缓存。
        # 单阶段对照是一次完全独立的训练。
        self.generator = SingleStageInpaintingGenerator(base_channels=base_channels)
        self.discriminator = SimpleUNetDiscriminator(in_channels=3)

        self.parameter_counts = {
            "generator": parameter_report(self.generator),
            "discriminator": parameter_report(self.discriminator),
        }
        print(
            f"[参数] 单阶段生成器 trainable={self.parameter_counts['generator']['trainable']}, "
            f"判别器 trainable={self.parameter_counts['discriminator']['trainable']} "
            f"(全程无冻结预训练权重)"
        )

        self.optimizer_g = AdamW(self.generator.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
        self.optimizer_d = AdamW(self.discriminator.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))

        # ---------- 训练状态 ----------
        self.start_epoch = 0
        self.global_step = 0
        self.completed_epochs = 0
        self.best_epoch = -1
        self.best_val_loss = float("inf")
        self.best_composite_score = -float("inf")
        self.epochs_no_improve = 0
        self.optimizer_g_state_dict = None
        self.optimizer_d_state_dict = None
        self._dataloaders_prepared = False
        self.stop_reason = None
        self.budget = BudgetTracker(world_size=getattr(self.accelerator, "num_processes", 1))

        # 可选 VGG 感知损失（论文默认关闭，保持与 Stage-2 相同的开关语义）
        self.vgg = None
        self._vgg_normalize = None
        if self.use_perceptual_loss:
            if not _TORCHVISION_AVAILABLE:
                raise RuntimeError("use_perceptual_loss=True 但 torchvision 不可用。")
            self.vgg, self._vgg_normalize = build_vgg_feature_extractor(self.accelerator.device)

        # 可选 torchmetrics（仅用于日志对照，选模不依赖它）
        self.psnr_metric = None
        self.ssim_metric = None
        if _TORCHMETRICS_AVAILABLE and self.accelerator.is_main_process:
            self.psnr_metric = torchmetrics.PeakSignalNoiseRatio().to(self.accelerator.device)
            self.ssim_metric = torchmetrics.StructuralSimilarityIndexMeasure(data_range=2.0).to(self.accelerator.device)

        os.makedirs(self.output_dir, exist_ok=True)
        if self.resume_from_checkpoint is not None:
            self._load_checkpoint()
        self._prepare_models()

        print("单阶段端到端对照训练初始化完成")
        if self.target_steps is not None:            print(f"  预算匹配：目标更新步数 = {self.target_steps}（D2R 两阶段实际步数之和）")
        if self.target_gpu_hours is not None:
            print(f"  预算匹配：目标 GPU·h = {self.target_gpu_hours}")
        if self.start_epoch > 0:
            print(f"从 epoch {self.start_epoch} 继续训练（global_step={self.global_step}）")

    # ═══════════════════════════════════════════
    # Accelerator / 分布式工具
    # ═══════════════════════════════════════════

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
        if not hasattr(self.accelerator, "reduce"):
            return float(total) / max(1, int(count))
        stats = torch.tensor(
            [float(total), float(count)], device=self.accelerator.device, dtype=torch.float64
        )
        stats = self.accelerator.reduce(stats, reduction="sum")
        return float((stats[0] / stats[1].clamp_min(1.0)).item())

    # ═══════════════════════════════════════════
    # 检查点
    # ═══════════════════════════════════════════

    def _find_latest_checkpoint(self):
        pattern = os.path.join(self.output_dir, "checkpoint-*")
        checkpoints = glob.glob(pattern)
        if not checkpoints:
            return None

        def get_epoch(path: str):
            m = re.search(r"checkpoint-epoch-(\d+)", path)
            return int(m.group(1)) if m else -1

        epoch_checkpoints = [p for p in checkpoints if get_epoch(p) >= 0]
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
                    "检查点使用旧的单阶段输入/损失协议，禁止自动续训；"
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
                self.global_step = int(state.get("global_step", 0))
                self.best_val_loss = float(state.get("best_val_loss", float("inf")))
                self.best_composite_score = float(state.get("best_composite_score", -float("inf")))
                self.best_epoch = int(state.get("best_epoch", -1))
                self.epochs_no_improve = int(state.get("epochs_no_improve", 0))
                self.optimizer_g_state_dict = state.get("optimizer_g_state_dict")
                self.optimizer_d_state_dict = state.get("optimizer_d_state_dict")
                print(f"恢复状态: epoch={self.start_epoch}, global_step={self.global_step}, "
                      f"best_val_loss={self.best_val_loss:.4f}")
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
            "global_step": int(self.global_step),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": self.best_val_loss,
            "best_composite_score": self.best_composite_score,
            "best_epoch": self.best_epoch,
            "epochs_no_improve": self.epochs_no_improve,
            "optimizer_g_state_dict": self.optimizer_g.state_dict(),
            "optimizer_d_state_dict": self.optimizer_d.state_dict(),
            "seed": self.seed,
            "gan_objective": "hinge_logits" if self.use_hinge_loss else "bce_with_logits",
            "protocol_id": self.PROTOCOL_ID,
            "target_steps": self.target_steps,
            "target_gpu_hours": self.target_gpu_hours,
            "train_seconds": round(float(self.budget.elapsed()), 3),
        }
        torch.save(state, os.path.join(ckpt_dir, "training_state.pt"))

        if is_best:
            with open(os.path.join(ckpt_dir, "BEST_MODEL"), "w", encoding="utf-8") as f:
                f.write(f"Best model at epoch {epoch} with val_loss {val_loss}\n")
            if extra_info is not None:
                info_file = os.path.join(ckpt_dir, "validation_metrics.txt")
                with open(info_file, "w", encoding="utf-8") as f:
                    f.write(f"Best Model at Epoch {epoch}\n")
                    f.write(f"Global step: {int(self.global_step)}\n")
                    f.write(f"Composite Score: {extra_info.get('composite_score', float('nan')):.4f}\n")
                    f.write(f"PSNR: {extra_info.get('psnr', float('nan')):.2f} dB\n")
                    f.write(f"SSIM: {extra_info.get('ssim', float('nan')):.3f}\n")
                    f.write(f"Generator Loss: {val_loss:.4f}\n")

        print(f"检查点已保存: {ckpt_dir}")

    # ═══════════════════════════════════════════
    # 前向与损失
    # ═══════════════════════════════════════════

    def _augment_pair(self, images: torch.Tensor, masks: torch.Tensor):
        """对 image/mask 施加完全相同的随机几何变换（与 Stage-1/Stage-2 协议一致）。"""
        if not self.augment_training:
            return images, masks
        if not _TORCHVISION_AVAILABLE:
            raise RuntimeError("单阶段数据增强需要 torchvision；也可使用 --no-augment 禁用")
        out_images, out_masks = [], []
        for image, mask in zip(images, masks):
            if torch.rand(()) < 0.5:
                image, mask = TF.hflip(image), TF.hflip(mask)
            angle = float(torch.empty(1).uniform_(-5.0, 5.0).item())
            image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR, fill=-1.0)
            mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)
            out_images.append(image)
            out_masks.append((mask >= 0.5).to(mask.dtype))
        return torch.stack(out_images), torch.stack(out_masks)

    def _predict(self, masked_image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """单阶段一次前向：预测缺失内容，并按与 D2R 相同的约定合成整图。

        Returns:
            composite: 掩膜外严格等于原图（因为 masked_image 在掩膜外就是原图）
            prediction: 模型对受损区域的直接预测
        """
        prediction = self.generator(masked_image, mask)
        composite = masked_image * (1.0 - mask) + prediction * mask
        return composite, prediction

    def compute_generator_loss(
        self, masked_image: torch.Tensor, mask: torch.Tensor, target_img: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float], torch.Tensor]:
        composite, prediction = self._predict(masked_image, mask)

        fake = composite * mask
        real = target_img * mask
        disc_fake = self.discriminator(fake)
        disc_real = self.discriminator(real)

        gen_adv = gan_loss(disc_fake, disc_real, for_generator=True, use_hinge=self.use_hinge_loss)
        l1_loss = F.l1_loss(composite * mask, target_img * mask)
        tex_loss = masked_texture_loss(composite, target_img, mask)

        perc_loss = torch.tensor(0.0, device=composite.device)
        if self.use_perceptual_loss:
            perc_loss = masked_perceptual_loss(
                self.vgg, self._vgg_normalize, composite, target_img, mask
            )

        total = (self.lambda_gan * gen_adv +
                 self.lambda_l1 * l1_loss +
                 self.lambda_texture * tex_loss +
                 self.lambda_perceptual * perc_loss)

        disc_accuracy = ((disc_fake < 0.0).float().mean() + (disc_real > 0.0).float().mean()) / 2.0

        loss_dict = {
            "gen_adv": float(gen_adv.detach().item()),
            "l1": float(l1_loss.detach().item()),
            "texture": float(tex_loss.detach().item()),
            "perceptual": float(perc_loss.detach().item()),
            "total_gen": float(total.detach().item()),
            "disc_accuracy": float(disc_accuracy.detach().item()),
        }
        return total, loss_dict, composite

    def compute_discriminator_loss(
        self, masked_image: torch.Tensor, mask: torch.Tensor, target_img: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        with torch.no_grad():
            composite, _ = self._predict(masked_image, mask)
        fake_input = (composite * mask).detach()
        real_input = (target_img * mask).detach()
        disc_fake = self.discriminator(fake_input)
        disc_real = self.discriminator(real_input)

        total = gan_loss(disc_fake, disc_real, for_generator=False, use_hinge=self.use_hinge_loss)
        loss_dict = {
            "disc_fake": float(F.binary_cross_entropy_with_logits(
                disc_fake, torch.zeros_like(disc_fake)).detach().item()),
            "disc_real": float(F.binary_cross_entropy_with_logits(
                disc_real, torch.ones_like(disc_real)).detach().item()),
            "total_disc": float(total.detach().item()),
        }
        return total, loss_dict

    def compute_metrics(self, composite: torch.Tensor, target: torch.Tensor,
                        mask: torch.Tensor) -> Dict[str, float]:
        # 与 D2R Stage-2 共用同一代理指标实现（training/common.py）。
        return masked_psnr_ssim(composite, target, mask)

    # ═══════════════════════════════════════════
    # 验证与训练
    # ═══════════════════════════════════════════

    def validate(self, val_loader) -> Dict[str, float]:
        self.generator.eval()
        self.discriminator.eval()
        total_gen_loss = 0.0
        total_disc_loss = 0.0
        total_psnr = 0.0
        total_ssim = 0.0
        total_examples = 0

        with torch.no_grad():
            for batch in tqdm(
                val_loader, desc="验证", disable=not getattr(self.accelerator, "is_main_process", True)
            ):
                images = batch["image"].to(self.accelerator.device, dtype=torch.float32)
                masks = batch["mask"].to(self.accelerator.device, dtype=torch.float32)
                masked_image = images * (1.0 - masks)

                gen_loss, _, composite = self.compute_generator_loss(masked_image, masks, images)
                disc_loss, _ = self.compute_discriminator_loss(masked_image, masks, images)
                metrics = self.compute_metrics(composite, images, masks)

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

        return {
            "gen_loss": avg_gen_loss,
            "disc_loss": avg_disc_loss,
            "psnr": avg_psnr,
            "ssim": avg_ssim,
            "composite_score": composite_score(avg_psnr, avg_ssim, avg_gen_loss,
                                               self.composite_score_weights),
            "examples": total_examples,
        }

    def train_epoch(self, train_loader, epoch: int) -> Tuple[float, float, float]:
        """一个 epoch；返回 (gen_loss, disc_loss, 本 epoch 实际使用的墙钟秒数)。"""
        self.generator.train()
        self.discriminator.train()

        total_gen_loss = 0.0
        total_disc_loss = 0.0
        total_examples = 0
        epoch_start = self.budget.elapsed()
        progress_bar = tqdm(
            train_loader, desc=f"单阶段 Epoch {epoch}",
            disable=not getattr(self.accelerator, "is_main_process", True),
        )

        for batch in progress_bar:
            images = batch["image"].to(self.accelerator.device, dtype=torch.float32)
            masks = batch["mask"].to(self.accelerator.device, dtype=torch.float32)
            images, masks = self._augment_pair(images, masks)
            masked_image = images * (1.0 - masks)

            with self.accelerator.accumulate(self.generator, self.discriminator):
                # ---- 判别器更新 ----
                self.optimizer_d.zero_grad(set_to_none=True)
                disc_loss, _ = self.compute_discriminator_loss(masked_image, masks, images)
                self.accelerator.backward(disc_loss)
                self.optimizer_d.step()

                # ---- 生成器更新 ----
                for parameter in self.discriminator.parameters():
                    parameter.requires_grad_(False)
                self.optimizer_g.zero_grad(set_to_none=True)
                gen_loss, loss_dict, _ = self.compute_generator_loss(masked_image, masks, images)
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
                "step": int(getattr(self, "global_step", 0)),
            })

            if self._budget_exhausted():
                break

        elapsed = self.budget.elapsed() - epoch_start
        return (
            self._distributed_mean(total_gen_loss, total_examples),
            self._distributed_mean(total_disc_loss, total_examples),
            elapsed,
        )

    def _budget_exhausted(self) -> bool:
        target_steps = getattr(self, "target_steps", None)
        target_gpu_hours = getattr(self, "target_gpu_hours", None)
        max_minutes = getattr(self, "max_train_minutes", None)
        if target_steps is not None and int(getattr(self, "global_step", 0)) >= target_steps:
            self.stop_reason = self.stop_reason or "target_steps_reached"
            return True
        if target_gpu_hours is not None and self.budget.current_gpu_hours() >= target_gpu_hours:
            self.stop_reason = self.stop_reason or "target_gpu_hours_reached"
            return True
        if max_minutes is not None and float(self.budget.elapsed()) / 60.0 >= float(max_minutes):
            self.stop_reason = self.stop_reason or "wall_clock_budget_reached"
            return True
        return False

    def train(self, train_loader, val_loader=None, num_epochs: Optional[int] = None):
        print("开始单阶段端到端对照训练")
        num_epochs = self.max_epochs if num_epochs is None else num_epochs
        train_loader, val_loader = self._prepare_dataloaders(train_loader, val_loader)
        if getattr(train_loader.dataset, "augment", False):
            raise ValueError(
                "单阶段 DataLoader 必须设置 augment=False；同步增强由 SingleStageGANTrainer 执行"
            )

        self._restore_optimizer_states()
        self.budget.start()

        last_epoch = self.start_epoch - 1
        val_metrics = None
        train_gen_loss = float("nan")
        for epoch in range(self.start_epoch, num_epochs):
            last_epoch = epoch
            train_gen_loss, train_disc_loss, epoch_seconds = self.train_epoch(train_loader, epoch)
            self.completed_epochs = epoch + 1

            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                if self.accelerator.is_main_process:
                    print(
                        f"Epoch {epoch}: train_gen={train_gen_loss:.4f}, train_disc={train_disc_loss:.4f}, "
                        f"val_gen={val_metrics['gen_loss']:.4f}, val_disc={val_metrics['disc_loss']:.4f}, "
                        f"PSNR={val_metrics['psnr']:.2f}dB, SSIM={val_metrics['ssim']:.3f}, "
                        f"Composite={val_metrics['composite_score']:.4f}, "
                        f"step={self.global_step}, epoch_time={epoch_seconds:.1f}s"
                    )

                # 选模规则与 Stage-2 完全一致：验证生成器损失最小者为最佳。
                if val_metrics["gen_loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["gen_loss"]
                    self.best_composite_score = val_metrics["composite_score"]
                    self.best_epoch = epoch
                    self.epochs_no_improve = 0
                    self.save_checkpoint(
                        epoch, train_gen_loss, val_metrics["gen_loss"], is_best=True,
                        extra_info={"composite_score": val_metrics["composite_score"],
                                    "psnr": val_metrics["psnr"], "ssim": val_metrics["ssim"]})
                    if self.accelerator.is_main_process:
                        print(f"新最佳模型! val_gen={val_metrics['gen_loss']:.4f}")
                else:
                    self.epochs_no_improve += 1
                    if self.accelerator.is_main_process:
                        print(f"分数未改善，连续 {self.epochs_no_improve} 轮")
            else:
                if self.accelerator.is_main_process:
                    print(f"Epoch {epoch}: train_gen={train_gen_loss:.4f}, train_disc={train_disc_loss:.4f}, "
                          f"step={self.global_step}")

            self._append_epoch_record(epoch, train_gen_loss, train_disc_loss, epoch_seconds, val_metrics)

            if self.accelerator.is_main_process and epoch % 10 == 0:
                self.save_checkpoint(epoch, train_gen_loss,
                                     None if val_metrics is None else val_metrics["gen_loss"])

            self.accelerator.wait_for_everyone()

            if self._budget_exhausted():
                if self.accelerator.is_main_process:
                    print(f"预算已用尽（{self.stop_reason}）：已完成 {self.global_step} 步 / "
                          f"{self.budget.current_gpu_hours():.4f} GPU·h")
                break

            if self.early_stopping_patience and self.epochs_no_improve >= self.early_stopping_patience:
                self.stop_reason = "early_stopping"
                if self.accelerator.is_main_process:
                    print(f"早停: {self.epochs_no_improve} 轮无改进")
                break
        else:
            self.stop_reason = self.stop_reason or "max_epochs_reached"

        if self.accelerator.is_main_process and last_epoch >= self.start_epoch:
            final_val = None if val_metrics is None else val_metrics["gen_loss"]
            self.save_checkpoint(last_epoch, train_gen_loss, final_val)
        self.accelerator.wait_for_everyone()
        self.budget.stop()
        self._write_budget_report()
        if self.accelerator.is_main_process:
            print(f"=== 单阶段对照训练完成（{self.global_step} 步，{self.budget.wall_seconds / 3600:.2f} 墙钟小时）===")

    # ═══════════════════════════════════════════
    # 记录
    # ═══════════════════════════════════════════

    def _append_epoch_record(self, epoch, train_gen_loss, train_disc_loss, epoch_seconds, val_metrics):
        if not self.accelerator.is_main_process:
            return
        record = {
            "epoch": epoch,
            "global_step": int(self.global_step),
            "train_gen_loss": float(train_gen_loss),
            "train_disc_loss": float(train_disc_loss),
            "epoch_seconds": float(epoch_seconds),
            "elapsed_seconds": float(self.budget.elapsed()),
        }
        if val_metrics is not None:
            record.update({f"val_{k}": float(v) for k, v in val_metrics.items()})
        with open(os.path.join(self.output_dir, "epoch_metrics.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _write_budget_report(self):
        """写出 Table 4 所需的预算记录（模型定义 / 参数量 / 更新步数 / GPU·h）。"""
        if not self.accelerator.is_main_process:
            return None
        generator = self.accelerator.unwrap_model(self.generator)
        record = {
            "stage": "single_stage_end_to_end_control",
            "protocol_id": self.PROTOCOL_ID,
            "model_definition": generator.architecture_summary,
            "pretraining": "none (trained from random initialisation on the study corpus)",
            "losses": (
                "hinge GAN (lambda=%.3g) + masked L1 (lambda=%.3g) + masked normalised Sobel texture "
                "(lambda=%.3g)%s"
                % (self.lambda_gan, self.lambda_l1, self.lambda_texture,
                   " + masked VGG perceptual (lambda=%.3g)" % self.lambda_perceptual
                   if self.use_perceptual_loss else "")
            ),
            "discriminator": (
                "SimpleUNetDiscriminator, per-pixel logits restricted to the damaged region (same as Stage-2)"
            ),
            "parameters": self.parameter_counts,
            "update_count_definition": (
                "generator optimizer steps; one discriminator update and one generator update per step"
            ),
            "completed_updates": int(self.global_step),
            "completed_epochs": int(self.completed_epochs),
            "resumed_from_epoch": int(self.start_epoch),
            "best_epoch": int(self.best_epoch),
            "best_val_generator_loss": None if self.best_val_loss == float("inf") else float(self.best_val_loss),
            "stop_reason": self.stop_reason,
            "budget_match": {
                "mode": ("steps" if self.target_steps is not None
                         else ("gpu_hours" if self.target_gpu_hours is not None else "none")),
                "target_steps": self.target_steps,
                "target_gpu_hours": self.target_gpu_hours,
                "reference": self.budget_reference,
            },
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
                "augmentation": "synchronised hflip and +/-5 degree rotation on image and mask"
                if self.augment_training else "disabled",
                "stage1_cache_used": False,
            },
            "selection_rule": (
                "best checkpoint by minimum validation generator loss, identical to the D2R Stage-2 rule"
            ),
            "seed": self.seed,
            "budget": self.budget.summary(),
        }
        path = write_budget_report(self.output_dir, record)
        print(f"预算报告已写出: {path}")
        return path
