#!/usr/bin/env python3
"""
训练入口：协调两阶段训练，以及审稿意见 R2-2 要求的端到端单阶段对照

用法:
  # 完整两阶段训练
  python train.py --mode full

  # 仅第一阶段
  python train.py --mode stage1

  # 仅第二阶段（需先完成第一阶段）
  python train.py --mode stage2

  # 端到端单阶段对照（R2-2）：独立训练，预算默认匹配参考 D2R 运行的更新步数
  python train.py --mode single_stage --seed 2026 \\
      --single_stage_output_dir single_stage_results_seed2026 \\
      --single_stage_budget_ref_stage1 stage1_results_seed2026 \\
      --single_stage_budget_ref_stage2 stage2_results_seed2026

  # 一条命令跑完 D2R 两阶段 + 单阶段对照（对照预算自动取自刚跑完的两阶段）
  python train.py --mode all

  # 自定义参数
  python train.py --mode full --resolution 512 --train_batch_size 1 --stage1_epochs 100 --stage2_epochs 100
"""

import os
import glob
import argparse
import json
import math
import platform
import re
import subprocess
from datetime import datetime, timezone
import torch

from dataset.dataset import create_dataloaders
from training import Stage1DiffusionTrainer, Stage2GANTrainer, SingleStageGANTrainer


def save_run_config(args, output_dir, extra=None):
    """Persist the exact invocation and environment before any training starts."""
    # All ranks execute the entry point under torchrun/accelerate launch, but
    # only rank 0 should write shared metadata files.
    if int(os.environ.get("RANK", "0")) != 0:
        return
    os.makedirs(output_dir, exist_ok=True)
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__),
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        git_commit = None
    record = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": vars(args),
        "effective_batch_size_per_gpu": args.train_batch_size * args.gradient_accumulation_steps,
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "effective_global_batch_size": (
            args.train_batch_size * args.gradient_accumulation_steps
            * int(os.environ.get("WORLD_SIZE", "1"))
        ),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "git_commit": git_commit,
        },
    }
    if extra:
        record.update(extra)
    path = os.path.join(output_dir, "run_config.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
    except OSError as exc:
        # 审计记录写不出去不应该让训练直接失败（只读挂载、权限受限的共享盘等）。
        print(f"[警告] 未能写入运行配置 {path}: {exc}")
        print("[警告] 本次运行将缺少 run_config.json，复现信息请以控制台输出为准。")


def parse_resume_checkpoint(value):
    """Allow ``none`` on the CLI to explicitly disable stale-checkpoint resume."""
    if value is None or str(value).strip().lower() in {"none", "null", "off", "false"}:
        return None
    return value


def run_stage1(args):
    """运行第一阶段训练"""
    print("=" * 60)
    print("=== 第一阶段：LoRA 微调扩散模型 ===")
    print("=" * 60)

    train_loader, val_loader = create_dataloaders(
        train_image_dir=args.train_image_dir,
        train_mask_dir=args.train_mask_dir,
        val_image_dir=args.val_image_dir,
        val_mask_dir=args.val_mask_dir,
        batch_size=args.train_batch_size,
        size=args.resolution,
        augment=args.augment,
        seed=args.seed,
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        num_workers=args.num_workers,
        photometric_jitter=args.augment_photometric,
    )
    # ``len(train_loader)`` is the single-process length. Under DDP each rank
    # receives approximately 1/world_size of those batches.
    world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
    optimizer_steps_per_epoch = math.ceil(
        math.ceil(len(train_loader) / world_size) / args.gradient_accumulation_steps
    )
    config = {
        "model_name": args.model_name,
        "output_dir": args.stage1_output_dir,
        "resolution": args.resolution,
        "train_batch_size": args.train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": 0.05,
        "lora_target_ff": args.stage1_lora_ff,
        "lr_warmup_steps": args.lr_warmup_steps,
        "max_train_steps": args.stage1_epochs * optimizer_steps_per_epoch,
        "resume_from_checkpoint": args.resume_from_checkpoint,
        "early_stopping_patience": args.early_stopping_patience,
        "seed": args.seed,
        "max_grad_norm": args.stage1_grad_clip,
        "selection_metric": args.stage1_selection_metric,
        "selection_num_images": args.stage1_selection_images,
        "selection_steps": args.stage1_selection_steps,
        "selection_guidance": args.stage1_selection_guidance,
        "max_train_minutes": args.stage1_max_minutes,
    }

    save_run_config(args, args.stage1_output_dir, {"trainer_config": config})
    trainer = Stage1DiffusionTrainer(**config)

    if val_loader is not None:
        trainer.train(train_loader, val_loader, num_epochs=args.stage1_epochs,
                      val_draws=args.stage1_val_draws)
    else:
        trainer.train(train_loader, num_epochs=args.stage1_epochs)

    print("=== 第一阶段训练完成 ===")


def run_stage2(args):
    """运行第二阶段训练"""
    print("=" * 60)
    print("=== 第二阶段：GAN 纹理细化 ===")
    print("=" * 60)

    config = {
        "stage1_checkpoint_dir": args.stage1_checkpoint_dir,
        "output_dir": args.stage2_output_dir,
        "cache_dir": args.cache_dir,
        "model_name": args.model_name,
        "resolution": args.resolution,
        "train_batch_size": args.train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.stage2_lr,
        "resume_from_checkpoint": args.resume_from_checkpoint,
        "early_stopping_patience": args.early_stopping_patience,
        "seed": args.seed,
        "lambda_gan": args.lambda_gan,
        "lambda_l1": args.lambda_l1,
        "lambda_texture": args.lambda_texture,
        "lambda_perceptual": args.lambda_perceptual,
        "use_perceptual_loss": args.use_perceptual_loss,
        "use_hinge_loss": args.use_hinge_loss,
        "residual_scale": args.residual_scale,
        "stage1_inference_steps": args.stage1_inference_steps,
        "guidance_scale": args.guidance_scale,
        "use_cache": not args.no_cache,
        "force_regen_cache": args.force_regen_cache,
        "augment_training": args.augment,
        "composite_score_weights": tuple(args.composite_weights),
    }

    train_loader, val_loader = create_dataloaders(
        train_image_dir=args.train_image_dir,
        train_mask_dir=args.train_mask_dir,
        val_image_dir=args.val_image_dir,
        val_mask_dir=args.val_mask_dir,
        batch_size=args.train_batch_size,
        size=args.resolution,
        # Stage-1 outputs are cached before each epoch. Stage-2 therefore performs
        # synchronized augmentation inside the trainer on target/mask/cache triplets.
        augment=False,
        seed=args.seed,
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        num_workers=args.num_workers,
    )

    save_run_config(args, args.stage2_output_dir, {"trainer_config": config})
    trainer = Stage2GANTrainer(**config)

    if val_loader is not None:
        trainer.train(train_loader, val_loader, num_epochs=args.stage2_epochs)
    else:
        trainer.train(train_loader, num_epochs=args.stage2_epochs)

    print("=== 第二阶段训练完成 ===")


def run_full_training(args):
    """运行完整的两阶段训练"""
    run_stage1(args)
    _reset_accelerator_state()
    args.stage1_checkpoint_dir = os.path.join(args.stage1_output_dir, "checkpoint-best")
    run_stage2(args)
    print("=== 完整训练完成 ===")


def _reset_accelerator_state():
    """重置 AcceleratorState 单例，避免同一进程内新建 Accelerator 时报冲突。"""
    from accelerate.state import PartialState
    PartialState._shared_state = {}
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════
# R2-2 端到端单阶段对照：预算匹配与训练
# ═══════════════════════════════════════════════════════════

def steps_per_epoch_from_loader(loader, world_size, gradient_accumulation_steps):
    """与 Stage-1/Stage-2 相同的步数换算：单进程 batch 数 / 进程数 / 梯度累积。"""
    return math.ceil(
        math.ceil(len(loader) / max(1, world_size)) / max(1, gradient_accumulation_steps)
    )


def _load_reference_run_config(stage_dir):
    path = os.path.join(stage_dir, "run_config.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def reference_stage_budget(stage_dir, steps_per_epoch):
    """从参考 D2R 运行的产物中导出**实际**优化步数与 GPU·h。

    优先级：training_state.pt 的 global_step → checkpoint-epoch-N 的最大 N × 每 epoch 步数
    → run_config.json 的 trainer_config.max_train_steps（计划值，非实际值）。

    返回的 dict 会原样写入单阶段对照的 budget_report.json，便于审稿人核对
    "单阶段对照的预算确实与 D2R 匹配"。
    """
    info = {
        "dir": os.path.abspath(stage_dir),
        "steps": None,
        "epochs_completed": None,
        "seconds": None,
        "world_size": None,
        "gpu_hours": None,
        "source": None,
        "protocol_id": None,
    }
    if not os.path.isdir(stage_dir):
        info["source"] = "directory not found"
        return info

    run_config = _load_reference_run_config(stage_dir)
    if run_config:
        info["world_size"] = int(run_config.get("world_size", 1))

    best_state = None
    for ckpt_dir in glob.glob(os.path.join(stage_dir, "checkpoint-*")):
        state_path = os.path.join(ckpt_dir, "training_state.pt")
        if not os.path.isfile(state_path):
            continue
        try:
            state = torch.load(state_path, map_location="cpu")
        except Exception:
            continue
        epoch = int(state.get("epoch", -1))
        if best_state is None or epoch > best_state[0]:
            best_state = (epoch, state, state_path)

    if best_state is not None:
        epoch, state, state_path = best_state
        info["epochs_completed"] = epoch + 1
        info["protocol_id"] = state.get("protocol_id")
        global_step = int(state.get("global_step", 0) or 0)
        if global_step > 0:
            info["steps"] = global_step
            info["source"] = f"{os.path.relpath(state_path, stage_dir)}:global_step"
        elif steps_per_epoch:
            info["steps"] = (epoch + 1) * steps_per_epoch
            info["source"] = (
                f"checkpoint-epoch-{epoch} x {steps_per_epoch} steps/epoch "
                "(checkpoint predates global_step recording)"
            )
        seconds = state.get("train_seconds")
        if seconds:
            info["seconds"] = float(seconds)

    if info["steps"] is None and run_config:
        planned = (run_config.get("trainer_config") or {}).get("max_train_steps")
        if planned:
            info["steps"] = int(planned)
            info["source"] = "run_config.json:trainer_config.max_train_steps (planned, not actual)"

    if info["seconds"] is not None and info["world_size"]:
        info["gpu_hours"] = round(info["seconds"] * info["world_size"] / 3600.0, 4)
    return info


def resolve_single_stage_budget(args, steps_per_epoch, stage1_dir=None, stage2_dir=None):
    """决定单阶段对照的预算目标，并返回可写入报告的参考信息。"""
    stage1_dir = stage1_dir or args.single_stage_budget_ref_stage1
    stage2_dir = stage2_dir or args.single_stage_budget_ref_stage2

    ref1 = reference_stage_budget(stage1_dir, steps_per_epoch)
    ref2 = reference_stage_budget(stage2_dir, steps_per_epoch)
    total_steps = None
    if ref1["steps"] and ref2["steps"]:
        total_steps = int(ref1["steps"] + ref2["steps"])
    total_gpu_hours = None
    if ref1["gpu_hours"] and ref2["gpu_hours"]:
        total_gpu_hours = round(ref1["gpu_hours"] + ref2["gpu_hours"], 4)

    reference = {
        "stage1": ref1,
        "stage2": ref2,
        "total_reference_steps": total_steps,
        "total_reference_gpu_hours": total_gpu_hours,
        "steps_per_epoch_used_for_derivation": steps_per_epoch,
    }

    mode = args.single_stage_budget_match
    target_steps = None
    target_gpu_hours = None
    notes = []

    if mode == "steps":
        target_steps = args.single_stage_total_steps or total_steps
        if target_steps is None:
            notes.append(
                "未能从参考运行导出步数；已回退为按 --single_stage_epochs 训练，"
                "该运行不能声称预算匹配。"
            )
    elif mode == "gpu_hours":
        target_gpu_hours = args.single_stage_target_gpu_hours or total_gpu_hours
        if target_gpu_hours is None:
            notes.append(
                "未能从参考运行导出 GPU·h（旧检查点未记录 train_seconds）；"
                "已回退为按 --single_stage_epochs 训练。"
            )
    else:
        notes.append("已显式禁用预算匹配（--single_stage_budget_match none）。")

    for note in notes:
        print(f"[预算匹配警告] {note}")

    reference["mode"] = mode if (target_steps or target_gpu_hours) else "none"
    reference["notes"] = notes
    return target_steps, target_gpu_hours, reference


def run_single_stage(args):
    """训练端到端单阶段对照（审稿意见 R2-2）。"""
    print("=" * 60)
    print("=== 端到端单阶段对照（R2-2） ===")
    print("=" * 60)

    train_loader, val_loader = create_dataloaders(
        train_image_dir=args.train_image_dir,
        train_mask_dir=args.train_mask_dir,
        val_image_dir=args.val_image_dir,
        val_mask_dir=args.val_mask_dir,
        batch_size=args.train_batch_size,
        size=args.resolution,
        # 增强在训练器内部同步施加（image/mask 同参数），与 Stage-2 一致。
        augment=False,
        seed=args.seed,
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        num_workers=args.num_workers,
    )

    world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
    steps_per_epoch = steps_per_epoch_from_loader(
        train_loader, world_size, args.gradient_accumulation_steps
    )
    target_steps, target_gpu_hours, reference = resolve_single_stage_budget(
        args, steps_per_epoch, stage1_dir=args._single_stage_ref_stage1,
        stage2_dir=args._single_stage_ref_stage2,
    )

    config = {
        "output_dir": args.single_stage_output_dir,
        "resolution": args.resolution,
        "train_batch_size": args.train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.single_stage_lr,
        "resume_from_checkpoint": args.resume_from_checkpoint,
        "early_stopping_patience": args.early_stopping_patience,
        "seed": args.seed,
        "lambda_gan": args.lambda_gan,
        "lambda_l1": args.lambda_l1,
        "lambda_texture": args.lambda_texture,
        "lambda_perceptual": args.lambda_perceptual,
        "use_perceptual_loss": args.use_perceptual_loss,
        "use_hinge_loss": args.use_hinge_loss,
        "augment_training": args.augment,
        "base_channels": args.single_stage_base_channels,
        "composite_score_weights": tuple(args.composite_weights),
        "max_epochs": args.single_stage_epochs,
        "target_steps": target_steps,
        "target_gpu_hours": target_gpu_hours,
        "budget_reference": reference,
        "max_train_minutes": args.single_stage_max_minutes,
    }

    if target_steps is not None:
        print(f"预算匹配：目标更新步数 = {target_steps}（参考运行导出）")
    if target_gpu_hours is not None:
        print(f"预算匹配：目标 GPU·h = {target_gpu_hours}")

    save_run_config(args, args.single_stage_output_dir,
                    {"trainer_config": config, "budget_reference": reference})
    trainer = SingleStageGANTrainer(**config)

    if val_loader is not None:
        trainer.train(train_loader, val_loader, num_epochs=args.single_stage_epochs)
    else:
        trainer.train(train_loader, num_epochs=args.single_stage_epochs)

    print("=== 单阶段对照训练完成 ===")


def run_all(args):
    """D2R 两阶段 + 端到端单阶段对照，一条命令跑完。"""
    run_full_training(args)
    _reset_accelerator_state()
    # 单阶段对照的预算取自刚刚完成的两阶段运行。
    args._single_stage_ref_stage1 = args.stage1_output_dir
    args._single_stage_ref_stage2 = args.stage2_output_dir
    run_single_stage(args)
    print("=== 全部训练完成（D2R 两阶段 + 单阶段对照）===")


def build_parser():
    """构建命令行参数解析器（默认值均与论文一致，供测试与训练共用）"""
    parser = argparse.ArgumentParser(description="两阶段图像修复训练")

    # 训练模式
    parser.add_argument("--mode", type=str, default="full",
                        choices=["stage1", "stage2", "single_stage", "full", "all"],
                        help="训练模式；single_stage = R2-2 端到端单阶段对照，all = 两阶段 + 单阶段对照")

    # 数据路径
    parser.add_argument("--train_image_dir", type=str, default="./datasets/train/img")
    parser.add_argument("--train_mask_dir", type=str, default="./datasets/train/mask")
    parser.add_argument("--val_image_dir", type=str, default="./datasets/val/img")
    parser.add_argument("--val_mask_dir", type=str, default="./datasets/val/mask")
    parser.add_argument("--train_manifest", type=str, default=None,
                        help="可选训练集 CSV；必须含 sample_id,image_path,mask_path")
    parser.add_argument("--val_manifest", type=str, default=None,
                        help="可选验证集 CSV；必须含 sample_id,image_path,mask_path")

    # 模型参数
    parser.add_argument("--model_name", type=str,
                        default="runwayml/stable-diffusion-inpainting",
                        help="Stable Diffusion Inpainting 基座：Hugging Face Hub id 或本地 "
                             "snapshot 目录；也可用环境变量 D2R_SD_MODEL 指定")
    parser.add_argument("--stage1_output_dir", type=str, default="./stage1_results")
    parser.add_argument("--stage2_output_dir", type=str, default="./stage2_results")
    parser.add_argument("--stage1_checkpoint_dir", type=str, default="./stage1_results/checkpoint-best")
    parser.add_argument("--cache_dir", type=str, default="./stage2_results/stage1_cache")

    # 训练超参数（数值以论文为准）
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=1,
                        help="批大小 (论文表2: 1)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="默认1，使有效 batch size 与论文表2一致")
    parser.add_argument("--learning_rate", type=float, default=5e-4,
                        help="Stage1 LoRA学习率 (论文表2: 5e-4)")
    parser.add_argument("--stage2_lr", type=float, default=1e-4,
                        help="Stage2 GAN 学习率 (论文表2: 1e-4)")
    parser.add_argument("--stage1_epochs", type=int, default=100,
                        help="Stage1训练epoch数 (论文: 最多100)")
    parser.add_argument("--stage2_epochs", type=int, default=100)
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--stage1_val_draws", type=int, default=1,
                        help="Stage-1 每次验证的独立噪声/时间步抽样次数；>1 可显著降低"
                             "验证损失方差，调参与选模建议 5")
    parser.add_argument("--stage1_grad_clip", type=float, default=1.0,
                        help="Stage-1 梯度裁剪范数上限；0 关闭。fp16 混合精度下建议保留")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader worker 数；设为 0 则在主进程内取数据，"
                             "适用于不能创建命名管道的环境（部分 Windows/受限沙箱）")
    # ── Stage-1 选模口径 ──
    parser.add_argument("--stage1_selection_metric", type=str, default="psnr",
                        choices=["psnr", "loss"],
                        help="Stage-1 选模指标：psnr = 验证集掩膜内修复 PSNR（与论文口径一致，"
                             "推荐）；loss = 掩膜内噪声预测 MSE（实测分辨力不足）")
    parser.add_argument("--stage1_selection_images", type=int, default=7,
                        help="选模用的验证图张数")
    parser.add_argument("--stage1_selection_steps", type=int, default=6,
                        help="选模推理的 DDIM 步数（只需相对可比，不必等于最终推理步数）")
    parser.add_argument("--stage1_selection_guidance", type=float, default=7.5,
                        help="选模推理的 CFG 强度")
    parser.add_argument("--stage1_max_minutes", type=float, default=None,
                        help="Stage-1 墙钟预算（分钟）；达到后停止但保留当前 best")

    # LoRA 参数（论文表2: r=32, alpha=64, dropout=0.05）
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)

    # ── Stage-1 可选容量/数据增强（显式 opt-in，默认关闭 = 论文协议不变） ──
    parser.add_argument("--stage1_lora_ff", action="store_true", default=False,
                        help="Stage-1 的 LoRA 额外覆盖前馈层 proj（提高域适配容量）。"
                             "默认关闭，保持论文所述的\"仅注意力投影层\"；开启后协议号变为 "
                             "...-ff，旧的 Stage-1 检查点会被拒绝续训")
    parser.add_argument("--augment_photometric", type=float, default=0.0,
                        help="Stage-1 训练的光度增强强度（ColorJitter 的 brightness/contrast 系数，"
                             "saturation 取其一半）。只作用于图像、不移动像素，因此 mask 无需同步。"
                             "默认 0 = 关闭，论文协议不变")

    # GAN 损失权重（论文: lambda_GAN=0.1, lambda_L1=50.0, lambda_texture=10.0）
    parser.add_argument("--lambda_gan", type=float, default=0.1)
    parser.add_argument("--lambda_l1", type=float, default=50.0)
    parser.add_argument("--lambda_texture", type=float, default=10.0)
    parser.add_argument("--lambda_perceptual", type=float, default=0.1)
    # Stage2 高级选项
    parser.add_argument("--use_perceptual_loss", action="store_true", default=False)
    parser.add_argument("--use_hinge_loss", action=argparse.BooleanOptionalAction, default=True,
                        help="使用 logits 上的 hinge loss；--no-use_hinge_loss 改用 BCEWithLogits")
    parser.add_argument("--residual_scale", type=float, default=0.3,
                        help="生成器残差缩放初值 (论文: 可学习因子初始化为0.3)")
    parser.add_argument("--no_cache", action="store_true", default=False,
                        help="禁用 Stage1 输出缓存")
    parser.add_argument("--force_regen_cache", action="store_true", default=False)
    parser.add_argument("--stage1_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                        help="Stage1 推理时的文本引导强度")
    parser.add_argument("--composite_weights", type=float, nargs=3, default=[0.4, 0.3, 0.3],
                        help="复合评分权重 (PSNR SSIM GenLoss倒数)")

    # ── R2-2 端到端单阶段对照 ──
    parser.add_argument("--single_stage_output_dir", type=str, default="./single_stage_results",
                        help="单阶段对照输出目录（含 checkpoint-best/generator.pth）")
    parser.add_argument("--single_stage_lr", type=float, default=1e-4,
                        help="单阶段对照学习率（与 Stage-2 相同：1e-4）")
    parser.add_argument("--single_stage_epochs", type=int, default=100,
                        help="单阶段对照的最大 epoch 数（硬上限；预算匹配通常会先触发停止）")
    parser.add_argument("--single_stage_base_channels", type=int, default=64,
                        help="单阶段生成器骨干宽度（与 Stage-2 生成器一致：64）")
    parser.add_argument("--single_stage_max_minutes", type=float, default=None,
                        help="单阶段对照的墙钟预算（分钟）；达到后停止但保留 best")
    parser.add_argument("--single_stage_budget_match", type=str, default="steps",
                        choices=["steps", "gpu_hours", "none"],
                        help="预算匹配口径：更新步数 / GPU·h / 不匹配（默认 steps）")
    parser.add_argument("--single_stage_total_steps", type=int, default=None,
                        help="显式指定目标更新步数；缺省时由参考 D2R 运行导出")
    parser.add_argument("--single_stage_target_gpu_hours", type=float, default=None,
                        help="显式指定目标 GPU·h；缺省时由参考 D2R 运行导出（需其记录过 train_seconds）")
    parser.add_argument("--single_stage_budget_ref_stage1", type=str, default="./stage1_results",
                        help="预算匹配用的参考 Stage-1 输出目录")
    parser.add_argument("--single_stage_budget_ref_stage2", type=str, default="./stage2_results",
                        help="预算匹配用的参考 Stage-2 输出目录")

    # 其他
    parser.add_argument("--resume_from_checkpoint", type=parse_resume_checkpoint, default="auto",
                        help="auto 自动恢复；输入 none 从头开始，避免沿用旧协议 checkpoint")
    parser.add_argument("--early_stopping_patience", type=int, default=20,
                        help="早停耐心值 (论文: 20)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow_cpu", action="store_true", default=False,
                        help="仅调试时允许无 CUDA 运行；正式论文训练默认拒绝 CPU 回退")

    return parser


def main():
    args = build_parser().parse_args()

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError(
            "CUDA 不可用，已阻止在 CPU 上静默训练。请先修复 NVIDIA 驱动/CUDA/PyTorch 环境；"
            "仅调试时可显式添加 --allow_cpu。"
        )

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if torch.cuda.is_available() and torch.cuda.device_count() >= 2 and world_size == 1:
        print(
            "提示：检测到至少两张 GPU，但当前是单进程。若要使用两张卡，"
            "请用 `accelerate launch --multi_gpu --num_processes 2 train.py ...` "
            "或 `torchrun --nproc_per_node=2 train.py ...` 启动。"
        )

    os.makedirs(args.stage1_output_dir, exist_ok=True)
    os.makedirs(args.stage2_output_dir, exist_ok=True)

    # 单阶段对照的预算参考目录：默认取 CLI 值；--mode all 会用刚训练完的两阶段目录覆盖。
    args._single_stage_ref_stage1 = args.single_stage_budget_ref_stage1
    args._single_stage_ref_stage2 = args.single_stage_budget_ref_stage2

    if args.mode == "stage1":
        run_stage1(args)
    elif args.mode == "stage2":
        run_stage2(args)
    elif args.mode == "single_stage":
        os.makedirs(args.single_stage_output_dir, exist_ok=True)
        run_single_stage(args)
    elif args.mode == "all":
        os.makedirs(args.single_stage_output_dir, exist_ok=True)
        run_all(args)
    else:
        run_full_training(args)


if __name__ == "__main__":
    main()
