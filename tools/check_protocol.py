#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_protocol.py — 断言论文协议默认值未被改动，且 opt-in 开关处于关闭状态。

这是发布版的"协议自检"：只做静态与默认值检查，不加载 Stable Diffusion 权重、
不读取数据集，因此几秒内可跑完。改动任何默认值、通道顺序或 LoRA 目标集合之前，
请先跑这个脚本。

用法（在仓库根目录下）:
    python tools/check_protocol.py
"""
import inspect
import os
import sys

# 本文件位于 tools/ 下，仓库根目录是它的上一级
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import train  # noqa: E402
from training import stage1  # noqa: E402
from training.common import (  # noqa: E402
    LORA_TARGET_MODULES_ATTENTION,
    LORA_TARGET_MODULES_ATTENTION_FF,
)
from dataset.dataset import InpaintingDataset, create_dataloaders  # noqa: E402

FAIL = 0


def check(name, cond, detail=""):
    global FAIL
    status = "PASS" if cond else "FAIL"
    if not cond:
        FAIL += 1
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))


print("=== 1. LoRA 目标集合 ===")
print(f"  默认（论文协议）: {LORA_TARGET_MODULES_ATTENTION}")
print(f"  opt-in（含 ff）  : {LORA_TARGET_MODULES_ATTENTION_FF}")
check("默认集合与改造前逐项一致",
      LORA_TARGET_MODULES_ATTENTION == ["to_k", "to_q", "to_v", "to_out.0"],
      str(LORA_TARGET_MODULES_ATTENTION))
check("opt-in 集合 = 默认 + ff.net.0.proj",
      LORA_TARGET_MODULES_ATTENTION_FF == LORA_TARGET_MODULES_ATTENTION + ["ff.net.0.proj"])
check("两集合均不含 conv_in/conv_out（9 通道 inpainting 输入不可破坏）",
      not any("conv_in" in m or "conv_out" in m for m in LORA_TARGET_MODULES_ATTENTION_FF))

print("\n=== 2. Stage-1 训练器默认值 ===")
sig = inspect.signature(stage1.Stage1DiffusionTrainer.__init__)
expected = {
    "learning_rate": 5e-4,
    "lora_rank": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.05,
    "lr_warmup_steps": 500,
    "early_stopping_patience": 20,
    "selection_metric": "psnr",
    "selection_num_images": 7,
    "selection_steps": 6,
    "selection_guidance": 7.5,
    "max_grad_norm": 1.0,
    "gradient_accumulation_steps": 1,
}
for key, want in expected.items():
    got = sig.parameters[key].default
    check(f"{key} = {want}", got == want, f"got {got}")
check("lora_target_ff 默认 False（新增，默认不启用）",
      sig.parameters["lora_target_ff"].default is False,
      f"got {sig.parameters['lora_target_ff'].default}")
check("PROTOCOL_ID 未被修改",
      stage1.Stage1DiffusionTrainer.PROTOCOL_ID
      == "d2r-stage1-paper-v3-noisy-mask-clean-condition-attention-lora",
      stage1.Stage1DiffusionTrainer.PROTOCOL_ID)

print("\n=== 3. 数据增强默认值 ===")
check("InpaintingDataset.photometric_jitter 默认 0.0",
      inspect.signature(InpaintingDataset.__init__).parameters["photometric_jitter"].default == 0.0)
check("create_dataloaders.photometric_jitter 默认 0.0",
      inspect.signature(create_dataloaders).parameters["photometric_jitter"].default == 0.0)
src_ds = open(os.path.join(ROOT, "dataset/dataset.py"), encoding="utf-8").read()
check("几何增强仍对 image/mask 使用同一 angle",
      "TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR, fill=0)" in src_ds
      and "TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST, fill=0)" in src_ds)
check("图像归一化仍是 [-1, 1]（未被改成 [0,1]）",
      "/ 127.5 - 1.0" in src_ds)

print("\n=== 4. train.py CLI 默认值 ===")
a = train.build_parser().parse_args([])
for key, want in [
    ("train_batch_size", 1),
    ("gradient_accumulation_steps", 1),
    ("learning_rate", 5e-4),
    ("stage1_epochs", 100),
    ("early_stopping_patience", 20),
    ("seed", 42),
    ("augment", True),
    ("stage1_lora_ff", False),
    ("augment_photometric", 0.0),
    ("stage1_selection_metric", "psnr"),
    ("single_stage_budget_match", "steps"),
]:
    got = getattr(a, key)
    check(f"--{key} = {want}", got == want, f"got {got}")

print("\n=== 5. 论文版关键实现未被触碰 ===")
src_s1 = open(os.path.join(ROOT, "training/stage1.py"), encoding="utf-8").read()
src_s2 = open(os.path.join(ROOT, "training/stage2.py"), encoding="utf-8").read()
src_train = open(os.path.join(ROOT, "train.py"), encoding="utf-8").read()
check("stage1.py 中不存在带引号的 ff 目标名（原测试断言保持通过）",
      '"ff.net.0.proj"' not in src_s1)
check("SD Inpainting 通道顺序仍为 noisy/mask/masked-latent",
      "torch.cat([noisy_latents, mask_latents, masked_image_latents]" in src_s1)
check("Stage-2 输入协议未变（仍为 4 通道 I_S1+M）",
      "residual = self.generator(stage1_out, mask)" in src_s2
      and "d2r-stage2-paper-v4-4ch-I_S1-M" in src_s2)
check("train.py 仍使用 dataset.dataset 导入",
      "from dataset.dataset import create_dataloaders" in src_train)
check("train.py 的 Stage-2 配置仍使用 stage2_lr",
      '"learning_rate": args.stage2_lr' in src_train)

print(f"\n======== 结果: {'全部通过' if FAIL == 0 else str(FAIL) + ' 项失败'} ========")
sys.exit(1 if FAIL else 0)
