#!/usr/bin/env python3
"""
推理管线：模型加载、Stage1 扩散生成、Stage2 GAN 细化

Stage1 统一使用标准 StableDiffusionInpaintPipeline，
保证非 mask 区域严格不变、mask 区域内生成合理。
"""

import os
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
from peft import PeftModel

from models import SimpleUNetGeneratorWithTexture, SingleStageInpaintingGenerator
from metrics import MetricCalculator
from utils import (
    create_red_mask_overlay,
    create_comparison_image,
    ensure_output_directory,
    find_image_mask_pairs,
)


# ═══════════════════════════════════════════
# 模型加载
# ═══════════════════════════════════════════

def load_stage2_generator(checkpoint_path, device="cuda"):
    """加载第二阶段生成器权重"""
    if os.path.isdir(checkpoint_path):
        checkpoint_file = os.path.join(checkpoint_path, "generator.pth")
    else:
        checkpoint_file = checkpoint_path
    if not os.path.isfile(checkpoint_file):
        raise FileNotFoundError(f"第二阶段生成器文件不存在: {checkpoint_file}")

    print(f"加载第二阶段生成器: {checkpoint_file}")
    generator = SimpleUNetGeneratorWithTexture(in_channels=4)
    state_dict = torch.load(checkpoint_file, map_location="cpu")
    # strict=True：旧 7 通道检查点的 enc1.conv1.weight 形状为 (64,7,3,3)，
    # 与新 4 通道不匹配会在此报错，避免静默加载出错配权重。
    generator.load_state_dict(state_dict, strict=True)
    generator = generator.to(device)
    generator.eval()
    print("第二阶段生成器加载成功")
    return generator


def load_single_stage_generator(checkpoint_path, device="cuda", base_channels=64):
    """加载端到端单阶段对照（R2-2）的生成器权重。

    接受三种路径写法，便于直接用训练输出目录评估：
      - 运行目录（内部自动找 ``checkpoint-best/generator.pth``）
      - 检查点目录（内部含 ``generator.pth``）
      - ``generator.pth`` 文件本身

    单阶段对照只需要这一个检查点：没有 LoRA、没有 Stage-1 输出、没有判别器参与推理。
    """
    if os.path.isdir(checkpoint_path):
        candidates = [
            os.path.join(checkpoint_path, "generator.pth"),
            os.path.join(checkpoint_path, "checkpoint-best", "generator.pth"),
        ]
        checkpoint_file = next((p for p in candidates if os.path.isfile(p)), candidates[0])
    else:
        checkpoint_file = checkpoint_path
    if not os.path.isfile(checkpoint_file):
        raise FileNotFoundError(
            f"单阶段对照生成器文件不存在: {checkpoint_file}；"
            "请指向训练输出目录、checkpoint-best 目录或 generator.pth 文件"
        )

    print(f"加载单阶段对照生成器: {checkpoint_file}")
    generator = SingleStageInpaintingGenerator(base_channels=base_channels)
    state_dict = torch.load(checkpoint_file, map_location="cpu")
    generator.load_state_dict(state_dict, strict=True)
    generator = generator.to(device)
    generator.eval()
    print("单阶段对照生成器加载成功")
    return generator


def _to_tensor(pil_or_tensor, device):
    """PIL Image 或已是 tensor 的输入 → [-1, 1] 的 (1, C, H, W) tensor。"""
    if isinstance(pil_or_tensor, Image.Image):
        # 注意必须 .to(device)：否则 PIL 分支返回 CPU tensor，与已搬到 GPU 的
        # mask 相乘时会报 "Expected all tensors to be on the same device"。
        return (torch.from_numpy(np.array(pil_or_tensor)).float()
                .permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0).to(device)
    return pil_or_tensor.to(device)


def _restore_outside_mask(restored_np: np.ndarray, original_image, mask, mask_tensor: torch.Tensor) -> np.ndarray:
    """把掩膜外的像素替换回原图的 uint8 值。

    uint8 → float32 → uint8 的往返会在部分像素上产生 1/255 的截断误差（例如 128 → 127）。
    论文与代码都声明"非 mask 区域严格不变"，因此在 uint8 空间再做一次合成，
    使掩膜外像素逐字节等于输入。掩膜区域内的指标（PSNR_M/SSIM_M/LPIPS_M）不受影响。
    """
    if not isinstance(original_image, Image.Image):
        return restored_np  # tensor 输入：调用方自行保证合成语义

    orig_np = np.asarray(original_image.convert("RGB"), dtype=np.uint8)
    if orig_np.shape != restored_np.shape:
        return restored_np

    if isinstance(mask, Image.Image):
        mask_bool = np.asarray(mask.convert("L")) >= 128
    else:
        mask_bool = (mask_tensor.reshape(mask_tensor.shape[-2:]).detach().cpu().numpy() > 0.5)
    if mask_bool.shape != orig_np.shape[:2]:
        return restored_np
    return np.where(mask_bool[..., None], restored_np, orig_np).astype(np.uint8)


def load_model(base_model_path, lora_path=None, stage1_checkpoint=None,
               stage2_checkpoint=None, device="cuda", use_cpu_offload=False,
               dtype=None):
    """加载完整模型管线。

    Args:
        dtype: 计算精度。``None``（默认）沿用历史行为：CUDA 上用 fp16，否则 fp32。
            评测脚本（``evaluate.py``）全程使用 fp32，因此若要复现论文数值请显式传
            ``torch.float32``——fp16 会带来细微差异。

    返回: (pipe, generator)
      - pipe:   StableDiffusionInpaintPipeline（可能已注入 LoRA）
      - generator: Stage2 GAN 生成器，未加载时返回 None
    """
    if dtype is None:
        dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        safety_checker=None,
        requires_safety_checker=False,
    )

    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass

    if use_cpu_offload and device == "cuda":
        try:
            pipe.enable_model_cpu_offload()
        except Exception:
            pipe = pipe.to(device)
    else:
        pipe = pipe.to(device)

    # 加载 LoRA（使用 diffusers 官方 API，保证设备/dtype 正确匹配）
    lora_dir = lora_path
    if lora_dir is None and stage1_checkpoint is not None:
        if os.path.exists(os.path.join(stage1_checkpoint, "adapter_config.json")):
            lora_dir = stage1_checkpoint

    if lora_dir is not None:
        print(f"加载 LoRA: {lora_dir}")
        try:
            # diffusers 官方 API：自动处理设备/dtype/融合
            pipe.load_lora_weights(lora_dir)
            pipe.fuse_lora()  # 熔合 LoRA → 基座模型，推理更快更稳定
            print("LoRA 加载并熔合成功")
        except Exception as e1:
            # 注意：本仓库 Stage-1 保存的是 PEFT 格式适配器（key 带 "base_model.model."
            # 前缀），diffusers 的 load_lora_weights 对它一律失败，正常情况下真正生效的
            # 是下面的"前缀修复"分支。
            # 该异常文本会列出全部未匹配的 target modules，长度可达数千字符，直接打印
            # 会淹没日志，因此截断——只保留判断所需的信息。
            msg = str(e1)
            if len(msg) > 300:
                msg = f"{msg[:300]} …（原信息共 {len(msg)} 字符，已截断）"
            print(f"load_lora_weights 失败: {msg}")
            print("（PEFT 格式适配器属预期情况，改用前缀修复路径重新加载）")
            # 尝试修复 state_dict 前缀 (base_model.model. → 去掉)
            try:
                import safetensors.torch
                import tempfile
                adapter_file = os.path.join(lora_dir, "adapter_model.safetensors")
                state_dict = safetensors.torch.load_file(adapter_file)
                # 去掉 base_model.model. 前缀
                fixed_sd = {}
                for k, v in state_dict.items():
                    fixed_sd[k.replace("base_model.model.", "")] = v
                # 写入临时文件，保留原始 adapter_config.json
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp_adapter = os.path.join(tmpdir, "adapter_model.safetensors")
                    safetensors.torch.save_file(fixed_sd, tmp_adapter)
                    # 复制 adapter_config.json
                    import shutil
                    shutil.copy2(
                        os.path.join(lora_dir, "adapter_config.json"),
                        os.path.join(tmpdir, "adapter_config.json"),
                    )
                    pipe.load_lora_weights(tmpdir)
                    pipe.fuse_lora()
                print("LoRA 加载并熔合成功（已自动修复 state_dict 前缀）")
            except Exception as e2:
                print(f"前缀修复后仍失败: {e2}")
                try:
                    # 最后回退：PEFT 直接加载，不熔合
                    from peft import PeftModel
                    pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_dir)
                    print("LoRA 加载成功（PEFT 回退，不熔合）")
                except Exception as e3:
                    print(f"LoRA 加载失败: {e3}")
                    print("将使用基础模型继续推理")

    # 加载 Stage2 生成器
    generator = None
    if stage2_checkpoint is not None:
        generator = load_stage2_generator(stage2_checkpoint, device)

    return pipe, generator


# ═══════════════════════════════════════════
# Stage1 扩散生成
# ═══════════════════════════════════════════

def stage1_inference(pipe, image, mask, prompt, negative_prompt="",
                     num_steps=20, cfg_scale=6.0, seed=None):
    """Stage1 扩散修复 —— 标准 StableDiffusionInpaintPipeline 调用。

    标准管线已针对 inpainting 任务微调，保证：
    - 非 mask 区域像素严格不变
    - mask 区域内根据上下文 + 文本提示生成合理内容

    Args:
        pipe: StableDiffusionInpaintPipeline
        image: PIL Image (RGB)
        mask:  PIL Image (L)
        prompt: 文本提示
        negative_prompt: 负面提示
        num_steps: 推理步数
        cfg_scale: 文本引导强度
        seed: 随机种子

    Returns:
        inpainted: PIL Image (RGB)
    """
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipe.device).manual_seed(seed)

    result = pipe(
        prompt=prompt,
        image=image,
        mask_image=mask,
        negative_prompt=negative_prompt if negative_prompt else None,
        num_inference_steps=num_steps,
        guidance_scale=cfg_scale,
        generator=generator,
    )
    generated = result.images[0].convert("RGB")
    # Enforce the inpainting contract exactly at pixel level.
    image_np = np.asarray(image.convert("RGB"), dtype=np.uint8)
    generated_np = np.asarray(generated, dtype=np.uint8)
    mask_np = np.asarray(mask.convert("L"), dtype=np.uint8) >= 128
    composite = np.where(mask_np[..., None], generated_np, image_np)
    return Image.fromarray(composite.astype(np.uint8))


# ═══════════════════════════════════════════
# Stage2 GAN 细化
# ═══════════════════════════════════════════

def refine_with_stage2(generator, stage1_output, original_image, mask,
                       device="cuda", strength=1.0):
    """使用第二阶段生成器细化 mask 区域。

    非 mask 区域严格等于原图，仅 mask 区域 = Stage1 + GAN残差(×strength)。

    Args:
        generator: SimpleUNetGeneratorWithTexture
        stage1_output: PIL Image 或 tensor [B,3,H,W] in [-1,1]
        original_image: PIL Image 或 tensor
        mask: PIL Image 或 tensor
        strength: GAN 残差强度，推荐 1.0（生成器内置了可学习缩放）

    Returns:
        refined: PIL Image (RGB)
    """
    # PIL → Tensor
    def to_tensor(pil_or_tensor):
        return _to_tensor(pil_or_tensor, device)

    stage1_tensor = to_tensor(stage1_output).to(device)
    original_tensor = to_tensor(original_image).to(device)

    if isinstance(mask, Image.Image):
        mask_tensor = torch.from_numpy(
            np.array(mask)).float().unsqueeze(0).unsqueeze(0) / 255.0
    else:
        mask_tensor = mask
    mask_tensor = mask_tensor.to(device)

    with torch.no_grad():
        # 生成器输入 = [I_S1, M]（4 通道），与训练一致（training/stage2.py _refine）。
        # 洞外像素来自 I_S1（Stage-1 在洞外保留原图），洞内来自 Stage-1 预测，
        # 掩膜区域内真值不进入网络任意一路（含纹理编码器）。
        residual = generator(stage1_tensor, mask_tensor)

        if generator.use_full_reconstruction:
            refined_tensor = original_tensor * (1.0 - mask_tensor) + residual * mask_tensor
        else:
            refined_tensor = original_tensor * (1.0 - mask_tensor) + \
                             (stage1_tensor + residual * strength) * mask_tensor

    refined_tensor = torch.clamp(refined_tensor, -1, 1)
    refined_np = ((refined_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1) * 127.5).astype(np.uint8)
    refined_np = _restore_outside_mask(refined_np, original_image, mask, mask_tensor)
    return Image.fromarray(refined_np)


# ═══════════════════════════════════════════
# 端到端单阶段对照推理（R2-2）
# ═══════════════════════════════════════════

def single_stage_inference(generator, original_image, mask, device="cuda"):
    """端到端单阶段对照的一次前向推理。

    与 D2R 的差别：不调用 Stage-1 扩散管线（无 LoRA、无文本提示、无采样步数），
    直接把"扣洞图像 + mask"送进单阶段生成器，输出即受损区域的预测内容。

    非 mask 区域严格等于原图（扣洞图像在掩膜外就是原图）。

    Args:
        generator: SingleStageInpaintingGenerator
        original_image: PIL Image (RGB) 或 tensor [B,3,H,W]
        mask: PIL Image (L) 或 tensor [B,1,H,W]

    Returns:
        restored: PIL Image (RGB)
    """
    original_tensor = _to_tensor(original_image, device)

    if isinstance(mask, Image.Image):
        mask_tensor = torch.from_numpy(
            np.array(mask)).float().unsqueeze(0).unsqueeze(0) / 255.0
    else:
        mask_tensor = mask
    mask_tensor = mask_tensor.to(device)
    mask_tensor = (mask_tensor > 0.5).float()

    masked_tensor = original_tensor * (1.0 - mask_tensor)

    with torch.no_grad():
        prediction = generator(masked_tensor, mask_tensor)
        restored_tensor = masked_tensor * (1.0 - mask_tensor) + prediction * mask_tensor

    restored_tensor = torch.clamp(restored_tensor, -1, 1)
    restored_np = ((restored_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1) * 127.5).astype(np.uint8)
    # 掩膜外逐字节还原。注意 mask_tensor 必须传 CPU 上的 tensor：
    # 若传 GPU tensor，np.asarray(tensor) 会报 "can't convert cuda:0 device type tensor"。
    restored_np = _restore_outside_mask(restored_np, original_image, mask, mask_tensor.cpu())
    return Image.fromarray(restored_np)


# ═══════════════════════════════════════════
# 批量指标
# ═══════════════════════════════════════════

def calculate_batch_metrics(original_images, inpainted_images, eval_size=(512, 512),
                          image_nums=None, mask_names=None, mask_images=None,
                          bootstrap_samples=2000, bootstrap_seed=42):
    """计算批量图像的全套指标：PSNR / SSIM / LPIPS / FID（全图 + mask 区域）。

    Args:
        original_images: list of PIL Image
        inpainted_images: list of PIL Image
        eval_size: 评估尺寸 (w, h)
        image_nums: 图片编号列表
        mask_names: mask 文件名列表
        mask_images: list of PIL Image (L), 可选，提供时额外计算 mask-only 指标

    Returns:
        dict: 全图平均 + mask 区域平均 + per_image 列表
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mc = MetricCalculator(device=device)

    has_masks = mask_images is not None and len(mask_images) > 0
    per_image = []

    for i, (orig, inp) in enumerate(zip(original_images, inpainted_images)):
        orig_r = orig.resize(eval_size, Image.LANCZOS)
        inp_r = inp.resize(eval_size, Image.LANCZOS)
        mask_r = mask_images[i].resize(eval_size, Image.NEAREST) if has_masks else None

        all_metrics = mc.calculate_all_single(orig_r, inp_r, mask=mask_r)

        entry = {
            "image_num": image_nums[i] if image_nums is not None else i,
            "mask_name": mask_names[i] if mask_names is not None else f"img_{i}",
            "psnr": all_metrics["psnr"],
            "ssim": all_metrics["ssim"],
            "lpips": all_metrics["lpips"],
            "inception_feature_l2": all_metrics["inception_feature_l2"],
        }
        if has_masks:
            entry["psnr_mask"] = all_metrics.get("psnr_mask", 0)
            entry["ssim_mask"] = all_metrics.get("ssim_mask", 0)
            entry["lpips_mask"] = all_metrics.get("lpips_mask", 0)

        per_image.append(entry)

        label = f"image#{entry['image_num']}"
        mask_str = ""
        if has_masks:
            mask_str = (f" | Mask: PSNR={all_metrics.get('psnr_mask',0):.2f}, "
                        f"SSIM={all_metrics.get('ssim_mask',0):.4f}, "
                        f"LPIPS={all_metrics.get('lpips_mask',0):.4f}")
        print(f"  {label}: PSNR={all_metrics['psnr']:.4f}, "
              f"SSIM={all_metrics['ssim']:.4f}, "
              f"LPIPS={all_metrics['lpips']:.4f}, "
              f"Inception-L2={all_metrics['inception_feature_l2']:.4f}{mask_str}")

    # 数据集级 FID
    orig_resized = [img.resize(eval_size, Image.LANCZOS) for img in original_images]
    inp_resized = [img.resize(eval_size, Image.LANCZOS) for img in inpainted_images]
    if has_masks:
        # Distributional metrics are evaluated on composites that contain the
        # prediction only inside M and the identical reference outside M.
        composite_resized = []
        for orig, inp, mask in zip(orig_resized, inp_resized, mask_images):
            mask_np = np.asarray(mask.resize(eval_size, Image.NEAREST)) >= 128
            orig_np, inp_np = np.asarray(orig), np.asarray(inp)
            composite_resized.append(Image.fromarray(
                np.where(mask_np[..., None], inp_np, orig_np).astype(np.uint8)
            ))
        inp_resized = composite_resized
    fid = mc.calculate_fid(orig_resized, inp_resized) if len(orig_resized) >= 2 else None
    kid = mc.calculate_kid(orig_resized, inp_resized) if len(orig_resized) >= 2 else None

    def bootstrap_ci(key):
        values = np.asarray([row[key] for row in per_image], dtype=np.float64)
        finite = values[np.isfinite(values)]
        if len(finite) < 2 or bootstrap_samples <= 0:
            return None
        rng = np.random.default_rng(bootstrap_seed)
        means = np.empty(bootstrap_samples, dtype=np.float64)
        for j in range(bootstrap_samples):
            means[j] = rng.choice(finite, size=len(finite), replace=True).mean()
        low, high = np.percentile(means, [2.5, 97.5])
        return [float(low), float(high)]

    result = {
        "avg_psnr": float(np.mean([m["psnr"] for m in per_image])),
        "avg_ssim": float(np.mean([m["ssim"] for m in per_image])),
        "avg_lpips": float(np.mean([m["lpips"] for m in per_image])),
        "fid": None if fid is None else float(fid),
        "kid": None if kid is None else float(kid),
        "num": len(original_images),
        "per_image": per_image,
        "ci95": {key: bootstrap_ci(key) for key in ("psnr", "ssim", "lpips")},
    }
    if has_masks:
        result["avg_psnr_mask"] = float(np.mean([m["psnr_mask"] for m in per_image]))
        result["avg_ssim_mask"] = float(np.mean([m["ssim_mask"] for m in per_image]))
        result["avg_lpips_mask"] = float(np.mean([m["lpips_mask"] for m in per_image]))
        result["ci95"].update({
            key: bootstrap_ci(key) for key in ("psnr_mask", "ssim_mask", "lpips_mask")
        })

    return result
