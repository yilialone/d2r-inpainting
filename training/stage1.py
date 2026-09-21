#!/usr/bin/env python3
"""
第一阶段训练器：LoRA微调Stable Diffusion Inpainting模型
"""

import os
import glob
import json
import re
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from accelerate import Accelerator
from tqdm import tqdm
import numpy as np
from diffusers import StableDiffusionInpaintPipeline, DDPMScheduler
from peft import LoraConfig, get_peft_model, PeftModel
from torchvision.transforms import functional as TF

from .common import (
    BudgetTracker,
    LORA_TARGET_MODULES_ATTENTION,
    LORA_TARGET_MODULES_ATTENTION_FF,
    parameter_report,
    write_budget_report,
)

os.environ["WANDB_MODE"] = "disabled"


class Stage1DiffusionTrainer:
    """第一阶段：使用LoRA微调扩散模型"""

    PROTOCOL_ID = "d2r-stage1-paper-v3-noisy-mask-clean-condition-attention-lora"

    def __init__(
        self,
        model_name="runwayml/stable-diffusion-inpainting",
        output_dir="./stage1_results",
        resolution=512,
        train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=5e-4,
        lora_rank=32,
        lora_alpha=64,
        lora_dropout=0.05,
        lora_target_ff=False,
        use_8bit_adam=True,
        resume_from_checkpoint="auto",
        save_best_only=True,
        early_stopping_patience=20,
        seed=42,
        lr_warmup_steps=500,
        max_train_steps=10000,
        max_grad_norm=1.0,
        selection_metric="psnr",
        selection_num_images=7,
        selection_steps=6,
        selection_guidance=7.5,
        max_train_minutes=None,
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Stage-1 选模口径。
        # "loss"  = 最小化掩膜内噪声预测 MSE（原实现）。实测该指标在本数据集上
        #           分辨力不足：未微调基座 0.1522、epoch-0 0.1517、epoch-20 0.2196，
        #           前两者在 2 个标准误内不可分，用它选模等于在噪声里挑模型。
        # "psnr"  = 最大化验证集掩膜内修复 PSNR（少量 DDIM 步），与论文最终评价口径一致。
        self.selection_metric = str(selection_metric).lower()
        if self.selection_metric not in {"loss", "psnr"}:
            raise ValueError(f"selection_metric 只支持 'loss' 或 'psnr'，得到 {selection_metric}")
        self.selection_num_images = int(selection_num_images)
        self.selection_steps = int(selection_steps)
        self.selection_guidance = float(selection_guidance)
        # 墙钟预算（分钟）：达到后停止训练但保留当前 best。用于在有限算力上
        # 保证"至少有产物"，避免训练无限期跑下去。
        self.max_train_minutes = float(max_train_minutes) if max_train_minutes else None

        self.accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            mixed_precision="fp16",
        )
        # fp16 混合精度下梯度裁剪是标准做法；此前完全没有裁剪，
        # 单个异常 batch 可以把 6.4M 个 LoRA 参数一次性推很远。
        # 注意：Accelerator 自身不接受 max_grad_norm（那是 TrainingArguments 的参数），
        # 这里自行保存并在 train_epoch 中调用 accelerator.clip_grad_norm_。
        self.max_grad_norm = float(max_grad_norm) if max_grad_norm else None
        if self.accelerator.num_processes > 1 and self.accelerator.is_main_process:
            print(
                f"启用分布式 Stage1: {self.accelerator.num_processes} 张 GPU；"
                f"每卡 batch_size={train_batch_size}，全局有效 batch_size="
                f"{train_batch_size * self.accelerator.num_processes * gradient_accumulation_steps}"
            )

        self.output_dir = output_dir
        self.resolution = resolution
        self.train_batch_size = train_batch_size
        self.learning_rate = learning_rate
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        # LoRA 目标模块：默认 = 论文协议的注意力投影层；前馈层为显式 opt-in。
        self.lora_target_ff = bool(lora_target_ff)
        self.lora_target_modules = list(
            LORA_TARGET_MODULES_ATTENTION_FF if self.lora_target_ff
            else LORA_TARGET_MODULES_ATTENTION
        )
        # 协议号随 LoRA 目标集合变化：默认路径下与 PROTOCOL_ID 完全一致，因此既有论文
        # 检查点的续训行为一字不变；开启 ff 后协议号改变，旧检查点会被明确拒绝，
        # 避免用"不同可训练参数集合"的权重静默续训。
        self.protocol_id = self.PROTOCOL_ID + ("-ff" if self.lora_target_ff else "")
        self.lr_warmup_steps = lr_warmup_steps
        self.max_train_steps = max_train_steps
        self.resume_from_checkpoint = resume_from_checkpoint
        self.save_best_only = save_best_only
        self.early_stopping_patience = early_stopping_patience
        self.seed = seed

        os.makedirs(self.output_dir, exist_ok=True)

        print("加载预训练模型...")
        self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
        )
        self.pipeline.to(self.accelerator.device)

        # 启用注意力优化（大幅加速训练）
        # 优先级：xformers > SDPA(PyTorch 2.0原生) > attention slicing
        _attn_ok = False
        try:
            self.pipeline.unet.enable_xformers_memory_efficient_attention()
            print("[注意力] xformers 内存高效注意力")
            _attn_ok = True
        except Exception:
            pass

        if not _attn_ok:
            try:
                # PyTorch >= 2.0 默认使用 SDPA，无需额外配置
                if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
                    print("[注意力] PyTorch 原生 SDPA (默认)")
                    _attn_ok = True
            except Exception:
                pass

        if not _attn_ok:
            try:
                self.pipeline.unet.enable_attention_slicing()
                print("[注意力] 注意力切片 (降级)")
                _attn_ok = True
            except Exception:
                print("[注意力] 警告: 未启用任何优化，训练速度可能较慢")

        self.pipeline.vae.eval()
        self.pipeline.text_encoder.eval()
        self.pipeline.vae.requires_grad_(False)
        self.pipeline.text_encoder.requires_grad_(False)

        # 预计算文本嵌入（prompt 全程不变，避免每步重复编码）
        self._cached_encoder_hidden_states = None
        self._prompt = (
            "An ancient bronze mirror engraved with the character '山' at the center, "
            "with natural patina and slight wear, photographed in soft, high-resolution realistic lighting."
        )

        # target_modules 由 training/common.py 的常量给出：默认 = 原论文所述的注意力
        # 投影层；前馈层 proj 仅在显式开启 lora_target_ff 时加入。
        # 注意：不包含 conv_in/conv_out（会破坏 UNet 的 9 通道 inpainting 输入处理）；
        # 不包含 to_out.1（它是 Dropout 层，PEFT 不支持）。
        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            target_modules=self.lora_target_modules,
            lora_dropout=self.lora_dropout,
            bias="none",
        )
        base_unet = self.pipeline.unet
        self.unet = get_peft_model(base_unet, lora_config)

        self.noise_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")

        self.optimizer = None
        self.optimizer_state_dict = None
        self.start_epoch = 0
        self.best_val_loss = float("inf")
        self.epochs_no_improve = 0
        self._dataloaders_prepared = False
        # 预算记账（Table 4: update count and GPU hours）
        self.completed_epochs = 0
        self.budget = BudgetTracker(world_size=getattr(self.accelerator, "num_processes", 1))

        if self.resume_from_checkpoint is not None:
            self._load_checkpoint()

        self._setup_optimizer()
        self._prepare_model()

        print(f"第一阶段训练初始化完成")
        if self.start_epoch > 0:
            print(f"从 epoch {self.start_epoch} 继续训练")

    def _setup_optimizer(self):
        for name, param in self.unet.named_parameters():
            if "lora" in name.lower():
                param.requires_grad = True

        stage1_params = [p for p in self.unet.parameters() if p.requires_grad]
        if len(stage1_params) == 0:
            for n, p in self.unet.named_parameters():
                if "lora" in n.lower():
                    p.requires_grad = True
            stage1_params = [p for p in self.unet.parameters() if p.requires_grad]

        self.optimizer = AdamW(stage1_params, lr=self.learning_rate, weight_decay=0.01)
        # 实际可训练参数（LoRA）数量：论文 Table 4 的 "Trainable parameters" 列必须由此导出，
        # 不能沿用正文中的估计值。
        self.parameter_counts = {"stage1_unet_lora": parameter_report(self.unet)}
        print(f"[参数] Stage-1 LoRA trainable={self.parameter_counts['stage1_unet_lora']['trainable']}, "
              f"total={self.parameter_counts['stage1_unet_lora']['total']}")
        self.lr_scheduler = CosineAnnealingLR(self.optimizer, T_max=self.max_train_steps)
        self.warmup_scheduler = None
        if self.lr_warmup_steps and self.lr_warmup_steps > 0:
            def lr_lambda(current_step):
                if current_step < self.lr_warmup_steps:
                    return float(current_step) / float(max(1, self.lr_warmup_steps))
                return 1.0
            self.warmup_scheduler = LambdaLR(self.optimizer, lr_lambda)

    def _prepare_model(self):
        # 只准备模型和优化器，scheduler 不需要 accelerator 包裹
        self.unet, self.optimizer = self.accelerator.prepare(self.unet, self.optimizer)
        # 手动计数器，避免依赖 AcceleratedScheduler.last_epoch；
        # 若 _load_checkpoint 已恢复 global_step，则保留恢复值。
        self._global_step = int(getattr(self, "_global_step", 0))

    def _prepare_dataloaders(self, train_loader, val_loader=None):
        """Shard DataLoaders across processes exactly once.

        ``Accelerator.prepare`` installs a distributed sampler when launched
        with ``accelerate launch``/``torchrun``.  Without this call every rank
        would iterate the complete dataset and perform duplicated updates.
        """
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
        # Small test doubles may not implement ``reduce``; retain the normal
        # single-process behavior for them.
        if not hasattr(self.accelerator, "reduce"):
            return float(total) / max(1, int(count))
        stats = torch.tensor(
            [float(total), float(count)],
            device=self.accelerator.device,
            dtype=torch.float64,
        )
        stats = self.accelerator.reduce(stats, reduction="sum")
        return float((stats[0] / stats[1].clamp_min(1.0)).item())

    def _find_latest_checkpoint(self):
        pattern = os.path.join(self.output_dir, "checkpoint-*")
        checkpoints = glob.glob(pattern)
        if not checkpoints:
            return None

        def get_epoch_from_checkpoint(path):
            match = re.search(r"checkpoint-epoch-(\d+)", path)
            if match:
                return int(match.group(1))
            return -1

        epoch_checkpoints = [path for path in checkpoints if get_epoch_from_checkpoint(path) >= 0]
        if epoch_checkpoints:
            return max(epoch_checkpoints, key=get_epoch_from_checkpoint)
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
            if state_probe.get("protocol_id") != self.protocol_id:
                raise RuntimeError(
                    "检查点使用旧的 Stage1 输入/LoRA 协议，禁止自动续训；"
                    "请使用 --resume_from_checkpoint none 并指定新的输出目录"
                )

        try:
            try:
                base_unet = self.unet.get_base_model()
            except Exception:
                base_unet = self.unet

            if os.path.exists(os.path.join(checkpoint_path, "adapter_config.json")):
                pefted = PeftModel.from_pretrained(base_unet, checkpoint_path, torch_dtype=torch.float32)
                self.unet = pefted
                print("成功加载LoRA适配器")
        except Exception as e:
            print(f"加载模型失败: {e}")

        if os.path.exists(state_path):
            try:
                state = torch.load(state_path, map_location="cpu")
                self.start_epoch = state.get("epoch", 0) + 1
                self.best_val_loss = state.get("best_val_loss", float("inf"))
                self.epochs_no_improve = state.get("epochs_no_improve", 0)
                self.optimizer_state_dict = state.get("optimizer_state_dict", None)
                self._global_step = int(state.get("global_step", 0))
                print(f"恢复训练状态: epoch={self.start_epoch}, best_val_loss={self.best_val_loss:.4f}")
            except Exception as e:
                print(f"加载训练状态失败: {e}")

    def save_checkpoint(self, epoch, train_loss, val_loss=None, is_best=False):
        if not self.accelerator.is_main_process:
            return
        if is_best:
            checkpoint_path = os.path.join(self.output_dir, "checkpoint-best")
        else:
            checkpoint_path = os.path.join(self.output_dir, f"checkpoint-epoch-{epoch}")
        os.makedirs(checkpoint_path, exist_ok=True)

        try:
            unwrapped_unet = self.accelerator.unwrap_model(self.unet)
            if hasattr(unwrapped_unet, "save_pretrained"):
                unwrapped_unet.save_pretrained(checkpoint_path)
            else:
                torch.save(unwrapped_unet.state_dict(), os.path.join(checkpoint_path, "unet_state.pth"))
        except Exception as e:
            print("保存模型失败:", e)

        state = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": self.best_val_loss,
            "epochs_no_improve": self.epochs_no_improve,
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
            "seed": self.seed,
            "protocol_id": self.protocol_id,
            "lora_target_modules": list(self.lora_target_modules),
            # 预算口径：Table 4 "Update count and GPU hours"
            "global_step": int(getattr(self, "_global_step", 0)),
            "train_seconds": round(float(self.budget.elapsed()), 3),
        }
        try:
            torch.save(state, os.path.join(checkpoint_path, "training_state.pt"))
        except Exception as e:
            print("保存训练状态失败:", e)

        if is_best:
            try:
                with open(os.path.join(checkpoint_path, "BEST_MODEL"), "w") as f:
                    f.write(f"Best model at epoch {epoch} with val_loss {val_loss}")
            except OSError as exc:
                print(f"[警告] 未能写入 BEST_MODEL 标记: {exc}")

        print("检查点已保存:", checkpoint_path)

    def _restore_optimizer_states(self):
        if hasattr(self, 'optimizer_state_dict') and self.optimizer_state_dict is not None:
            try:
                self.optimizer.load_state_dict(self.optimizer_state_dict)
                print("已恢复优化器状态")
            except Exception as e:
                print(f"恢复优化器状态失败: {e}")
        if hasattr(self, 'optimizer_state_dict'):
            del self.optimizer_state_dict

    def _get_text_embeddings(self, batch_size: int) -> torch.Tensor:
        """获取缓存的文本嵌入（仅首次调用时编码，后续直接复用）"""
        if self._cached_encoder_hidden_states is None:
            with torch.no_grad():
                text_inputs = self.pipeline.tokenizer(
                    [self._prompt], return_tensors="pt", padding=True,
                    truncation=True, max_length=self.pipeline.tokenizer.model_max_length,
                )
                text_input_ids = text_inputs.input_ids.to(self.accelerator.device)
                self._cached_encoder_hidden_states = self.pipeline.text_encoder(text_input_ids)[0]
        return self._cached_encoder_hidden_states.repeat(batch_size, 1, 1)

    def compute_loss(self, latent_model_input, noise, timesteps, mask_latent, target_latents):
        batch_size = latent_model_input.shape[0]
        encoder_hidden_states = self._get_text_embeddings(batch_size)

        outputs = self.unet(
            latent_model_input, timesteps,
            encoder_hidden_states=encoder_hidden_states,
            cross_attention_kwargs=None, return_dict=False,
        )
        noise_pred = outputs[0]

        mask_resized = F.interpolate(mask_latent, size=noise_pred.shape[-2:], mode="nearest")
        mask_resized = mask_resized.repeat(1, noise_pred.shape[1], 1, 1)

        loss = F.mse_loss(noise_pred, noise, reduction="none")
        loss = (loss * mask_resized).sum() / (mask_resized.sum() + 1e-6)
        return loss

    def validate(self, val_loader, num_draws: int = 1, return_stats: bool = False):
        """验证集扩散损失。

        ``num_draws`` 次独立抽样（每次不同 generator 种子），返回多次采样的均值。
        单次抽样等价于"每张验证图只用一个噪声 + 一个时间步"，其方差足以掩盖学习率
        之间或相邻 epoch 之间的真实差异，因此调参与选模都应使用多次采样；调用方也可
        取 ``return_stats=True`` 拿到标准误，用来判断两个配置是否可分。
        """
        self.unet.eval()
        device_type = self.accelerator.device.type
        per_draw_means = []

        with torch.no_grad():
            for draw in range(max(1, int(num_draws))):
                draw_generator = torch.Generator(device=device_type).manual_seed(self.seed + draw)
                total_val_loss = 0.0
                total_examples = 0
                for batch in tqdm(
                    val_loader,
                    desc=f"验证(draw {draw + 1}/{num_draws})" if num_draws > 1 else "验证",
                    disable=not getattr(self.accelerator, "is_main_process", True),
                    leave=False,
                ):
                    images = batch["image"].to(self.accelerator.device)
                    masks = batch["mask"].to(self.accelerator.device)

                    vae_dtype = next(self.pipeline.vae.parameters()).dtype
                    images = images.to(dtype=vae_dtype)
                    masks = masks.to(dtype=vae_dtype)

                    masked_images = images * (1 - masks)

                    # VAE 在 eval 模式下使用 mode()，保证 latent 确定，仅噪声与时间步随 draw 变化。
                    latents = self.pipeline.vae.encode(images).latent_dist.mode()
                    latents = latents * self.pipeline.vae.config.scaling_factor

                    masked_image_latents = self.pipeline.vae.encode(masked_images).latent_dist.mode()
                    masked_image_latents = masked_image_latents * self.pipeline.vae.config.scaling_factor

                    noise = torch.randn(
                        latents.shape, generator=draw_generator,
                        device=latents.device, dtype=latents.dtype,
                    )
                    timesteps = torch.randint(
                        0, self.noise_scheduler.config.num_train_timesteps,
                        (latents.shape[0],), device=latents.device,
                        generator=draw_generator,
                    ).long()

                    noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

                    mask_latents = F.interpolate(masks, size=latents.shape[-2:], mode="nearest")
                    # SD-inpainting conditions on the clean masked-image latent; only
                    # the target latent is noised at timestep t (4 + 1 + 4 channels:
                    # noisy latent, mask, clean masked-image latent).
                    latent_model_input = torch.cat([noisy_latents, mask_latents, masked_image_latents], dim=1)

                    loss = self.compute_loss(latent_model_input, noise, timesteps, mask_latents, latents)
                    batch_size = images.shape[0]
                    total_val_loss += loss.item() * batch_size
                    total_examples += batch_size

                per_draw_means.append(self._distributed_mean(total_val_loss, total_examples))

        self.unet.train()

        mean_loss = float(np.mean(per_draw_means))
        if not return_stats:
            return mean_loss
        std = float(np.std(per_draw_means, ddof=1)) if len(per_draw_means) > 1 else 0.0
        stderr = std / (len(per_draw_means) ** 0.5) if len(per_draw_means) > 1 else 0.0
        return {
            "mean": mean_loss,
            "std": std,
            "stderr": stderr,
            "num_draws": len(per_draw_means),
            "per_draw": per_draw_means,
        }

    # ------------------------------------------------------------------ #
    # 选模用的修复质量指标（与论文最终评价口径一致）
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def restoration_score(self, val_loader, num_images=None, steps=None,
                          guidance=None, use_adapter=True):
        """在验证集子集上做真实修复，返回掩膜内 PSNR。

        这是 Stage-1 的选模口径：噪声预测 MSE 在本数据集上与"修复得好不好"
        几乎无关（见 `selection_metric` 的说明），因此改用与论文报告一致的
        掩膜内 PSNR。

        实现上直接复用 diffusers 的标准 inpainting 管线（`self.pipeline`），
        而不是手写 DDIM 循环——这样选模口径与最终评测口径不会出现细微不一致：
          * 管线自带 CFG、masked-image latent、洞外保留等全部 inpainting 逻辑；
          * 只需把 `self.unet`（含 LoRA）替换进管线，评估后再还原。

        Args:
            use_adapter: False 时临时禁用 LoRA，用同一接口评估**未微调基座**，
                使"微调是否真的有用"变成可直接比较的数字。

        Returns:
            dict: ``psnr``（掩膜内均值，dB）、``n``、``steps``、``guidance``、
                  ``use_adapter``、``per_image``
        """
        num_images = int(num_images or self.selection_num_images)
        steps = int(steps or self.selection_steps)
        guidance = float(self.selection_guidance if guidance is None else guidance)

        # 收集验证集前 N 张（子集很小，不必另建 DataLoader）
        images, masks = [], []
        for batch in val_loader:
            for i in range(batch["image"].shape[0]):
                images.append(batch["image"][i])
                masks.append(batch["mask"][i])
                if len(images) >= num_images:
                    break
            if len(images) >= num_images:
                break
        if not images:
            return {"psnr": float("nan"), "n": 0, "per_image": []}

        from contextlib import nullcontext

        was_training = self.unet.training
        self.unet.eval()

        # 让管线使用当前的（含 LoRA / 禁用 LoRA 的）UNet
        original_unet = self.pipeline.unet
        self.pipeline.unet = self.unet
        adapter_ctx = nullcontext() if use_adapter else self.unet.disable_adapter()

        per_image = []
        try:
            with adapter_ctx:
                for idx, (img, msk) in enumerate(zip(images, masks)):
                    # [-1,1] 张量 → PIL（管线接口）
                    img_pil = TF.to_pil_image(((img + 1.0) / 2.0).clamp(0, 1))
                    msk_pil = TF.to_pil_image(msk.squeeze(0).clamp(0, 1))
                    out = self.pipeline(
                        prompt=self._prompt,
                        image=img_pil,
                        mask_image=msk_pil,
                        num_inference_steps=steps,
                        guidance_scale=guidance,
                        generator=torch.Generator(device="cpu").manual_seed(
                            self.seed + idx
                        ),
                    ).images[0]

                    rec = TF.to_tensor(out).to(self.accelerator.device) * 2.0 - 1.0
                    x0 = img.to(self.accelerator.device)
                    m = msk.to(self.accelerator.device)
                    comp = x0 * (1 - m) + rec * m
                    mse = ((comp - x0) ** 2 * m).sum() / (m.sum() * comp.shape[0] + 1e-8)
                    psnr = float(10.0 * torch.log10(4.0 / mse.clamp_min(1e-12)))
                    per_image.append({
                        "index": idx,
                        "psnr_masked": psnr,
                        "mask_fraction": float(m.mean()),
                    })
        finally:
            self.pipeline.unet = original_unet
            if was_training:
                self.unet.train()

        vals = [p["psnr_masked"] for p in per_image]
        return {
            "psnr": float(np.mean(vals)),
            "psnr_std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "n": len(vals),
            "steps": steps,
            "guidance": guidance,
            "use_adapter": bool(use_adapter),
            "per_image": per_image,
        }

    def train_epoch(self, train_loader, epoch):
        self.unet.train()
        total_loss = 0.0
        total_examples = 0
        progress_bar = tqdm(
            train_loader,
            desc=f"第一阶段 Epoch {epoch}",
            disable=not getattr(self.accelerator, "is_main_process", True),
        )
        self.optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(progress_bar):
            with self.accelerator.accumulate(self.unet):
                images = batch["image"].to(self.accelerator.device)
                masks = batch["mask"].to(self.accelerator.device)

                vae_dtype = next(self.pipeline.vae.parameters()).dtype
                images = images.to(dtype=vae_dtype)
                masks = masks.to(dtype=vae_dtype)

                masked_images = images * (1 - masks)

                with torch.no_grad():
                    latents = self.pipeline.vae.encode(images).latent_dist.sample()
                    latents = latents * self.pipeline.vae.config.scaling_factor

                    masked_image_latents = self.pipeline.vae.encode(masked_images).latent_dist.sample()
                    masked_image_latents = masked_image_latents * self.pipeline.vae.config.scaling_factor

                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, self.noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],), device=latents.device
                ).long()

                noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
                mask_latents = F.interpolate(masks, size=latents.shape[-2:], mode="nearest")
                latent_model_input = torch.cat([noisy_latents, mask_latents, masked_image_latents], dim=1)

                loss = self.compute_loss(latent_model_input, noise, timesteps, mask_latents, latents)

                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients and self.max_grad_norm:
                    self.accelerator.clip_grad_norm_(self.unet.parameters(), self.max_grad_norm)
                self.optimizer.step()
                if self.accelerator.sync_gradients:
                    if self.warmup_scheduler is not None and self._global_step < self.lr_warmup_steps:
                        self.warmup_scheduler.step()
                    elif self.lr_scheduler is not None:
                        self.lr_scheduler.step()
                    self._global_step += 1
                self.optimizer.zero_grad(set_to_none=True)

                batch_size = images.shape[0]
                total_loss += loss.item() * batch_size
                total_examples += batch_size
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        return self._distributed_mean(total_loss, total_examples)

    def current_lr(self) -> float:
        """当前优化器学习率（用于诊断 scheduler 是否真的在按预期衰减）。"""
        try:
            return float(self.optimizer.optimizer.param_groups[0]["lr"])
        except Exception:
            pass
        try:
            return float(self.optimizer.param_groups[0]["lr"])
        except Exception:
            return float("nan")

    def _append_training_log(self, record: dict):
        """逐 epoch 追加训练记录，供调参与论文预算核对使用。"""
        if not self.accelerator.is_main_process:
            return None
        path = os.path.join(self.output_dir, "training_log.jsonl")
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as exc:
            print(f"写入训练日志失败: {exc}")
            return None
        return path

    def train(self, train_loader, val_loader=None, num_epochs=50, val_draws=1):
        print("开始第一阶段训练")
        train_loader, val_loader = self._prepare_dataloaders(train_loader, val_loader)
        self._restore_optimizer_states()
        self.budget.start()

        use_psnr = self.selection_metric == "psnr" and val_loader is not None
        base_psnr = None
        if use_psnr and self.start_epoch == 0:
            # 先量出"完全未微调基座"的修复质量。没有这个基线，
            # "LoRA 有没有用"就无法回答——实测噪声 MSE 看不出差别，
            # 而基座在修复 PSNR 上未必比微调后的检查点差。
            print("测量未微调基座（禁用 LoRA）的验证集修复 PSNR ...")
            base = self.restoration_score(val_loader, use_adapter=False)
            base_psnr = base["psnr"]
            self.baseline_selection = base
            print(f"[基线] 未微调基座 掩膜内 PSNR = {base_psnr:.3f} dB "
                  f"({base['n']} 张, {base['steps']} 步 DDIM, CFG {base['guidance']})")
            # 只有超过基线的检查点才值得保存为 best
            self.best_val_loss = -float("inf") if use_psnr else self.best_val_loss

        last_epoch = self.start_epoch - 1
        train_loss = float("nan")
        val_loss = None
        for epoch in range(self.start_epoch, num_epochs):
            last_epoch = epoch
            train_loss = self.train_epoch(train_loader, epoch)
            self.completed_epochs = epoch + 1

            val_loss = None
            val_stats = None
            sel_score = None
            if val_loader is not None:
                val_stats = self.validate(val_loader, num_draws=val_draws, return_stats=True)
                val_loss = float(val_stats["mean"])
                line = (f"第一阶段 Epoch {epoch}: train_loss={train_loss:.4f}, "
                        f"val_loss={val_loss:.4f} ± {val_stats['stderr']:.4f} "
                        f"({val_stats['num_draws']} draws), lr={self.current_lr():.3e}")
                if use_psnr:
                    sel = self.restoration_score(val_loader)
                    sel_score = sel["psnr"]
                    line += f", val_PSNR={sel_score:.3f} dB (masked)"
                print(line)

                if use_psnr:
                    # 必须严格优于"当前最好的 val_PSNR"，且不劣于未微调基座。
                    # 只看第一项会保存一个比基座还差的 best（实测发生过）。
                    improved = (sel_score > self.best_val_loss and
                                (base_psnr is None or sel_score >= base_psnr))
                    metric_desc = f"val_PSNR={sel_score:.3f} dB"
                else:
                    improved = val_loss < self.best_val_loss
                    metric_desc = f"val_loss={val_loss:.4f}"

                if improved:
                    self.best_val_loss = sel_score if use_psnr else val_loss
                    self.epochs_no_improve = 0
                    self.save_checkpoint(epoch, train_loss, val_loss, is_best=True)
                    gain = ""
                    if use_psnr and base_psnr is not None:
                        gain = f"（相对未微调基座 {base_psnr:.3f} dB：{sel_score - base_psnr:+.3f} dB）"
                    print(f"新的最佳模型! {metric_desc}{gain}")
                else:
                    self.epochs_no_improve += 1
                    note = ""
                    if use_psnr and base_psnr is not None and sel_score < base_psnr:
                        note = (f"（该轮 {sel_score:.3f} dB 低于未微调基座 {base_psnr:.3f} dB，"
                                f"不保存为 best）")
                    print(f"选模指标未改善，已连续 {self.epochs_no_improve} 轮{note}")
            else:
                print(f"第一阶段 Epoch {epoch}: train_loss={train_loss:.4f}")

            if epoch % 10 == 0:
                self.save_checkpoint(epoch, train_loss, val_loss)

            self._append_training_log({
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": val_loss,
                "val_stderr": float(val_stats["stderr"]) if val_stats else None,
                "val_draws": int(val_stats["num_draws"]) if val_stats else None,
                "val_per_draw": val_stats["per_draw"] if val_stats else None,
                "selection_metric": self.selection_metric,
                "selection_score": sel_score,
                "baseline_psnr_no_adapter": base_psnr,
                "lr": self.current_lr(),
                "global_step": int(getattr(self, "_global_step", 0)),
                "best_val_loss": float(self.best_val_loss),
                "epochs_no_improve": int(self.epochs_no_improve),
                "train_seconds": round(float(self.budget.elapsed()), 3),
            })

            if (self.early_stopping_patience is not None and
                    self.epochs_no_improve >= self.early_stopping_patience):
                print(f"早停: {self.epochs_no_improve} 轮无改进")
                break

            if self.max_train_minutes is not None:
                elapsed_min = float(self.budget.elapsed()) / 60.0
                if elapsed_min >= self.max_train_minutes:
                    print(f"达到时间预算 {self.max_train_minutes:.0f} 分钟"
                          f"（已用 {elapsed_min:.1f} 分钟），停止训练并保留当前 best")
                    break

        if last_epoch >= self.start_epoch:
            self.save_checkpoint(last_epoch, train_loss, val_loss)
        self.accelerator.wait_for_everyone()
        self.budget.stop()
        self._write_budget_report()
        print("=== 第一阶段训练完成 ===")

    def _write_budget_report(self):
        """写出 Table 4 所需的预算记录（参数量 / 更新步数 / GPU·h）。"""
        if not self.accelerator.is_main_process:
            return None
        record = {            "stage": "stage1_lora_diffusion",
            "protocol_id": self.protocol_id,
            "model_definition": (
                "Stable Diffusion Inpainting UNet adapted with LoRA (r=%d, alpha=%d, dropout=%.3g) "
                "on LoRA target modules [%s]; VAE and text encoder frozen; "
                "masked-region noise-prediction MSE"
                % (self.lora_rank, self.lora_alpha, self.lora_dropout,
                   ", ".join(self.lora_target_modules))
            ),
            "losses": "masked noise-prediction MSE at a uniformly sampled timestep",
            "parameters": getattr(self, "parameter_counts", None),
            "update_count_definition": "optimizer steps on the LoRA parameters",
            "completed_updates": int(getattr(self, "_global_step", 0)),
            "completed_epochs": int(self.completed_epochs),
            "resumed_from_epoch": int(self.start_epoch),
            "optimizer": {
                "name": "AdamW",
                "weight_decay": 0.01,
                "learning_rate": self.learning_rate,
                "lr_warmup_steps": self.lr_warmup_steps,
                "max_train_steps": self.max_train_steps,
                "gradient_accumulation_steps": self.accelerator.gradient_accumulation_steps,
                "mixed_precision": str(self.accelerator.mixed_precision),
                "max_grad_norm": self.max_grad_norm,
            },
            "data": {
                "resolution": self.resolution,
                "train_batch_size_per_process": self.train_batch_size,
            },
            "selection_rule": (
                "best checkpoint by maximum masked-region restoration PSNR "
                f"({self.selection_num_images} val images, {self.selection_steps} DDIM steps, "
                f"CFG {self.selection_guidance})"
                if self.selection_metric == "psnr" else
                "best checkpoint by minimum validation diffusion loss"
            ),
            "selection_metric": self.selection_metric,
            "seed": self.seed,
            "budget": self.budget.summary(),
        }
        try:
            path = write_budget_report(self.output_dir, record)
        except OSError as exc:
            print(f"[警告] 未能写出预算报告: {exc}")
            return None
        print(f"预算报告已写出: {path}")
        return path
