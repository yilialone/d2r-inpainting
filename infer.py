#!/usr/bin/env python3
"""
统一图像修复推理入口。

Stage1 统一使用标准 StableDiffusionInpaintPipeline（保证非 mask 区域严格不变）。
Stage2 可选 GAN 纹理细化（仅修改 mask 区域）。
Canny 边缘提取仅用于可视化参考。

用法:
  # 基础模型
  python infer.py --image ./test.jpg --mask ./mask.png --out ./results

  # LoRA + Stage2
  python infer.py --image ./test.jpg --mask ./mask.png --out ./results \\
      --stage1_checkpoint ./stage1_results/checkpoint-best \\
      --stage2_checkpoint ./stage2_results/checkpoint-best/generator.pth

  # R2-2 端到端单阶段对照（不需要 SD 基座、不需要 Stage-1/Stage-2）
  python infer.py --image_dir ./datasets/val/img --mask_dir ./datasets/val/mask \\
      --single_stage_checkpoint ./single_stage_results_seed2026 \\
      --out ./evaluation_single_stage --compute_metrics

  # 批量处理 + 指标计算
  python infer.py --image_dir ./datasets/val/img --mask_dir ./datasets/val/mask \\
      --out ./results --compute_metrics
"""

import os
import json
import argparse
import struct
import hashlib
import csv
import torch
import numpy as np
from PIL import Image, ImageFile
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
ImageFile.LOAD_TRUNCATED_IMAGES = True  # 容错加载截断/损坏的图像


# ═══════════════════════════════════════════
# 安全图像加载
# ═══════════════════════════════════════════

def _safe_load_image(path: str, mode: str = "RGB") -> Image.Image:
    """安全加载图像，自动跳过空文件/截断文件。

    先检查文件大小（跳过 0 字节空文件），再用 PIL 加载。
    如果文件为空或损坏，抛出 OSError，由调用方跳过。
    """
    if os.path.getsize(path) == 0:
        raise OSError(f"文件为空 (0 bytes): {os.path.basename(path)}")
    img = Image.open(path)
    img.load()  # 显式触发解码，在转换前捕获截断错误
    return img.convert(mode)

from utils import (
    ensure_output_directory,
    find_image_mask_pairs,
    create_red_mask_overlay,
    create_comparison_image,
)
from inference import (
    load_model,
    load_single_stage_generator,
    stage1_inference,
    refine_with_stage2,
    single_stage_inference,
    calculate_batch_metrics,
    extract_canny_edges,
    compute_adaptive_cfg,
)
from metrics import MetricCalculator


# ═══════════════════════════════════════════
# Excel 输出工具
# ═══════════════════════════════════════════

def _write_metrics_excel(per_image: list, xlsx_path: str):
    """将逐张指标写入格式化的 Excel 文件。

    Args:
        per_image: [{image_num, mask_name, psnr, ssim, lpips, inception_feature_l2}, ...]
        xlsx_path: 输出 .xlsx 路径
    """
    wb = Workbook()
    ws = wb.active
    ws.title = "Per-Image Metrics"

    # 样式
    header_font = Font(name="Arial", size=11, bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
    header_align = Alignment(horizontal="center", vertical="center")
    cell_align = Alignment(horizontal="center", vertical="center")
    thin_border = Border(
        left=Side(style="thin"),
        right=Side(style="thin"),
        top=Side(style="thin"),
        bottom=Side(style="thin"),
    )

    # 检测是否有 mask 区域指标
    has_mask = any("psnr_mask" in entry for entry in per_image)

    # 表头
    headers = ["图片编号", "Mask 编号",
               "PSNR(dB)全图", "SSIM全图", "LPIPS全图", "Inception特征L2"]
    if has_mask:
        headers += ["PSNR(dB)Mask", "SSIM Mask", "LPIPS Mask"]
    for col_idx, header in enumerate(headers, 1):
        cell = ws.cell(row=1, column=col_idx, value=header)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = header_align
        cell.border = thin_border

    # 数据行
    full_metric_keys = ["psnr", "ssim", "lpips", "inception_feature_l2"]
    mask_metric_keys = ["psnr_mask", "ssim_mask", "lpips_mask"]

    for row_idx, entry in enumerate(per_image, 2):
        mask_name = entry.get("mask_name", "")

        values = [
            entry["image_num"],
            mask_name,
            round(entry["psnr"], 4),
            round(entry["ssim"], 4),
            round(entry["lpips"], 4),
            round(entry["inception_feature_l2"], 4),
        ]
        if has_mask:
            values += [
                round(entry.get("psnr_mask", 0), 4),
                round(entry.get("ssim_mask", 0), 4),
                round(entry.get("lpips_mask", 0), 4),
            ]
        for col_idx, value in enumerate(values, 1):
            cell = ws.cell(row=row_idx, column=col_idx, value=value)
            cell.alignment = cell_align
            cell.border = thin_border
            if col_idx >= 3:
                cell.number_format = "0.0000"

    # 列宽
    col_widths = [14, 30, 14, 14, 14, 20]
    if has_mask:
        col_widths += [14, 14, 14]
    for col_idx, width in enumerate(col_widths, 1):
        ws.column_dimensions[ws.cell(row=1, column=col_idx).column_letter].width = width

    # 冻结首行
    ws.freeze_panes = "A2"

    # 平均值行
    last_row = len(per_image) + 2
    avg_font = Font(name="Arial", size=11, bold=True)
    avg_fill = PatternFill(start_color="D9E2F3", end_color="D9E2F3", fill_type="solid")

    ws.cell(row=last_row, column=1, value="平均值").font = avg_font
    ws.cell(row=last_row, column=1).fill = avg_fill
    ws.cell(row=last_row, column=1).alignment = cell_align
    ws.cell(row=last_row, column=1).border = thin_border

    ws.cell(row=last_row, column=2, value="—").font = avg_font
    ws.cell(row=last_row, column=2).fill = avg_fill
    ws.cell(row=last_row, column=2).alignment = cell_align
    ws.cell(row=last_row, column=2).border = thin_border

    for col_idx, key in enumerate(full_metric_keys, start=3):
        col_values = [entry[key] for entry in per_image]
        avg_val = round(sum(col_values) / len(col_values), 4)
        cell = ws.cell(row=last_row, column=col_idx, value=avg_val)
        cell.font = avg_font
        cell.fill = avg_fill
        cell.alignment = cell_align
        cell.border = thin_border
        cell.number_format = "0.0000"

    if has_mask:
        for col_idx, key in enumerate(mask_metric_keys, start=3 + len(full_metric_keys)):
            col_values = [entry.get(key, 0) for entry in per_image]
            avg_val = round(sum(col_values) / len(col_values), 4)
            cell = ws.cell(row=last_row, column=col_idx, value=avg_val)
            cell.font = avg_font
            cell.fill = avg_fill
            cell.alignment = cell_align
            cell.border = thin_border
            cell.number_format = "0.0000"

    wb.save(xlsx_path)


def _write_metrics_csv(per_image: list, csv_path: str):
    """Write machine-readable per-image values without rounding."""
    if not per_image:
        return
    fieldnames = list(per_image[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_image)


# ═══════════════════════════════════════════
# 单张图像推理
# ═══════════════════════════════════════════

def run_inference(pipe, image_path, mask_path, output_dir, args,
                  generator=None, device="cuda", single_stage_generator=None):
    """推理单张图像。

    Stage1: 标准 SD Inpainting Pipeline
    Stage2: GAN 纹理细化（可选）
    Single-stage: 端到端单阶段对照（R2-2），一次前向、无扩散（可选，与 Stage2 互斥）
    """
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    use_single_stage = single_stage_generator is not None
    use_stage2 = (not use_single_stage) and generator is not None
    use_lora = args.lora_path is not None or args.stage1_checkpoint is not None
    if use_single_stage:
        model_label = "single_stage"
    elif use_stage2:
        model_label = "stage2"
    elif use_lora:
        model_label = "lora"
    else:
        model_label = "base"

    try:
        # 安全加载图像（捕获截断/空文件等损坏情况）
        try:
            original_image = _safe_load_image(image_path, "RGB")
            mask_image = _safe_load_image(mask_path, "L")
        except (OSError, IOError, ValueError, struct.error) as e:
            print(f" -> SKIP (图像损坏: {e})")
            return False, None, None, None

        target_size = (args.width, args.height)
        if original_image.size != target_size:
            original_image = original_image.resize(target_size, Image.LANCZOS)
            mask_image = mask_image.resize(target_size, Image.NEAREST)

        print(f"\n处理: {base_name} ({model_label})", end="")

        # ---- Canny 边缘提取（仅可视化参考，不影响生成） ----
        structure_map = None
        if args.structure_guidance:
            print(" [Canny]", end="")
            structure_map = extract_canny_edges(original_image)

        # ---- 端到端单阶段对照（R2-2）：一次前向，不经过 Stage-1 ----
        if use_single_stage:
            print(" [single-stage]", end="")
            inpainted_result = single_stage_inference(
                single_stage_generator, original_image, mask_image, device=device,
            )
            stage1_image = None
        else:
            # ---- Stage1: 标准 SD Inpainting Pipeline ----
            # Final evaluation defaults must match Stage-2 cache generation.
            mask_np = np.array(mask_image).astype(float) / 255.0
            cfg_scale = compute_adaptive_cfg(mask_np) if args.adaptive_cfg else args.guidance_scale
            print(f" [CFG={cfg_scale:.1f}]", end="")

            # 每张图片用不同种子：seed + image_number，保证可复现
            seed_offset = int(hashlib.sha256(base_name.encode("utf-8")).hexdigest()[:8], 16)
            img_seed = (args.seed + seed_offset) % (2**31)
            stage1_image = stage1_inference(
                pipe,
                image=original_image,
                mask=mask_image,
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                num_steps=args.steps,
                cfg_scale=cfg_scale,
                seed=img_seed,
            )

            # ---- Stage2: GAN 纹理细化（仅修改 mask 区域） ----
            if use_stage2:
                print(f" [GAN x{args.stage2_strength:.1f}]", end="")
                inpainted_result = refine_with_stage2(
                    generator, stage1_image, original_image, mask_image,
                    device=device, strength=args.stage2_strength,
                )
            else:
                inpainted_result = stage1_image

        # ---- 保存结果 ----
        single_dir = os.path.join(output_dir, "single_images")
        comparison_dir = os.path.join(output_dir, "comparison_images")
        os.makedirs(single_dir, exist_ok=True)
        os.makedirs(comparison_dir, exist_ok=True)

        original_image.save(os.path.join(single_dir, f"{base_name}_original.png"))
        mask_overlay = create_red_mask_overlay(original_image, mask_image)
        mask_overlay.save(os.path.join(single_dir, f"{base_name}_mask_overlay.png"))
        inpainted_result.save(os.path.join(single_dir,
                             f"{base_name}_inpainted_{model_label}.png"))

        if use_stage2:
            stage1_image.save(os.path.join(single_dir, f"{base_name}_stage1.png"))
        if structure_map is not None:
            structure_map.save(os.path.join(single_dir, f"{base_name}_edges.png"))

        comparison = create_comparison_image(original_image, mask_overlay, inpainted_result)
        comparison.save(os.path.join(comparison_dir,
                        f"{base_name}_comparison_{model_label}.png"))

        print(f" -> OK")
        return True, original_image, inpainted_result, mask_image

    except Exception as e:
        print(f" -> FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None, None


# ═══════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="图像修复推理")

    # 模型
    parser.add_argument("--base_model",
                        default=os.environ.get("D2R_SD_MODEL",
                                               "runwayml/stable-diffusion-inpainting"),
                        help="基座 SD-Inpainting 模型：Hugging Face Hub id 或本地 snapshot 目录；"
                             "也可用环境变量 D2R_SD_MODEL 指定")
    parser.add_argument("--lora_path", help="LoRA 权重目录")
    parser.add_argument("--stage1_checkpoint", help="第一阶段检查点目录")
    parser.add_argument("--stage2_checkpoint", help="第二阶段生成器路径 (.pth)")
    parser.add_argument("--single_stage_checkpoint",
                        help="R2-2 端到端单阶段对照的生成器路径（目录或 generator.pth）；"
                             "给出时只用该模型推理，不需要 Stage-1 与 Stage-2 检查点")
    parser.add_argument("--single_stage_base_channels", type=int, default=64,
                        help="单阶段生成器骨干宽度（须与训练时一致，默认 64）")
    parser.add_argument("--use_cpu_offload", action="store_true")

    # 输入
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", help="单张图像路径")
    input_group.add_argument("--image_dir", help="批量图像目录")
    parser.add_argument("--mask", help="单张掩码路径")
    parser.add_argument("--mask_dir", help="批量掩码目录")
    parser.add_argument("--manifest", help="批量模式 CSV 配对清单（优先于目录自动配对）")

    # 输出
    parser.add_argument("--out", required=True, help="输出目录")

    # 推理参数（空 prompt 匹配 CLIP 编码兼容性）
    parser.add_argument("--prompt", default="")
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument("--steps", type=int, default=30,
                        help="默认与 Stage-2 缓存生成一致")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                        help="默认与 Stage-2 缓存生成一致")
    parser.add_argument("--adaptive_cfg", action="store_true", default=False,
                        help="探索性选项；正式比较不应与固定 CFG 结果混用")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)

    # Stage2
    parser.add_argument("--stage2_strength", type=float, default=1.0,
                        help="GAN 细化强度 (推荐 1.0，生成器内置可学习缩放)")

    # Canny 边缘（仅可视化）
    parser.add_argument("--structure_guidance", action="store_true", default=False,
                        help="提取 Canny 边缘图用于可视化（不影响生成）")

    # 指标
    parser.add_argument("--compute_metrics", action="store_true")
    parser.add_argument("--eval_width", type=int, default=512)
    parser.add_argument("--eval_height", type=int, default=512)

    args = parser.parse_args()

    # 参数验证
    if args.image and not args.mask:
        parser.error("单张模式需要 --mask")
    if args.image_dir and not args.mask_dir:
        parser.error("批量模式需要 --mask_dir")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    torch.manual_seed(args.seed)

    # 加载模型
    # 单阶段对照（R2-2）只需要它自己的生成器：不需要 Stable Diffusion 基座、
    # 不需要 LoRA、不需要 Stage-2。这样评估单阶段对照时不必下载/加载 SD 权重。
    single_only = (
        args.single_stage_checkpoint is not None
        and args.stage1_checkpoint is None
        and args.lora_path is None
        and args.stage2_checkpoint is None
    )
    if single_only:
        print("单阶段对照模式：跳过 Stable Diffusion 管线加载")
        pipe, generator = None, None
    else:
        pipe, generator = load_model(
            base_model_path=args.base_model,
            lora_path=args.lora_path,
            stage1_checkpoint=args.stage1_checkpoint,
            stage2_checkpoint=args.stage2_checkpoint,
            device=device,
            use_cpu_offload=args.use_cpu_offload,
        )

    single_stage_generator = None
    if args.single_stage_checkpoint is not None:
        single_stage_generator = load_single_stage_generator(
            args.single_stage_checkpoint, device=device,
            base_channels=args.single_stage_base_channels,
        )

    if pipe is None and single_stage_generator is None:
        print("模型加载失败，退出")
        return

    output_dir = ensure_output_directory(args.out)
    print(f"输出目录: {output_dir}")
    with open(os.path.join(output_dir, "inference_config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "arguments": vars(args),
            "device": device,
            "torch": torch.__version__,
            "seed_policy": "base seed plus SHA-256-derived image-name offset",
        }, f, indent=2, ensure_ascii=False)

    # ═════════ 批量模式 ═════════
    if args.image_dir:
        pairs = find_image_mask_pairs(args.image_dir, args.mask_dir, args.manifest)
        if not pairs:
            print("未找到有效图像-掩码对")
            return

        success_count = 0
        original_images, inpainted_images = [], []
        image_nums, mask_names, mask_images = [], [], []

        for i, (img_path, mask_path, img_num) in enumerate(pairs, 1):
            print(f"[{i}/{len(pairs)}]", end="")
            success, orig, inp, msk = run_inference(
                pipe, img_path, mask_path, output_dir, args,
                generator=generator, device=device,
                single_stage_generator=single_stage_generator,
            )
            if success:
                success_count += 1
                if args.compute_metrics:
                    original_images.append(orig)
                    inpainted_images.append(inp)
                    mask_images.append(msk)
                    image_nums.append(img_num)
                    mask_names.append(os.path.basename(mask_path))

        if args.compute_metrics and success_count > 0:
            print("\n计算全量指标 (PSNR/SSIM/LPIPS/FID，全图 + Mask 区域)...")
            metrics = calculate_batch_metrics(
                original_images, inpainted_images,
                (args.eval_width, args.eval_height),
                image_nums=image_nums,
                mask_names=mask_names,
                mask_images=mask_images,
            )

            # ── 保存平均指标 JSON ──
            avg_metrics = {
                "protocol": {
                    "manuscript_primary_scope": "mask_region",
                    "evaluation_size": [args.eval_width, args.eval_height],
                    "image_range": "RGB uint8 [0,255] converted to [0,1] for PSNR/SSIM",
                    "mask_definition": "mask >= 0.5 is the evaluated missing region",
                    "lpips_mask_scope": "spatial LPIPS map weighted over active mask pixels",
                    "fid_kid_scope": "prediction inside mask and identical reference outside mask",
                    "distribution_features": "torchvision Inception-v3 DEFAULT weights, 2048-d pool features",
                    "bootstrap_samples": 2000,
                    "bootstrap_seed": 42,
                },
                "full_image": {
                    "avg_psnr": metrics["avg_psnr"],
                    "avg_ssim": metrics["avg_ssim"],
                    "avg_lpips": metrics["avg_lpips"],
                    "fid": metrics["fid"],
                    "kid": metrics["kid"],
                },
                "num_images": metrics["num"],
                "ci95": metrics["ci95"],
            }
            if "avg_psnr_mask" in metrics:
                avg_metrics["mask_region"] = {
                    "avg_psnr": metrics["avg_psnr_mask"],
                    "avg_ssim": metrics["avg_ssim_mask"],
                    "avg_lpips": metrics["avg_lpips_mask"],
                }
            json_path = os.path.join(output_dir, "batch_metrics.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(avg_metrics, f, indent=2, ensure_ascii=False)
            print(f"\n══ 全图平均指标 ══")
            print(f"  PSNR:  {metrics['avg_psnr']:.4f} dB")
            print(f"  SSIM:  {metrics['avg_ssim']:.4f}")
            print(f"  LPIPS: {metrics['avg_lpips']:.4f}")
            if metrics["fid"] is None:
                print("  FID/KID: N/A（至少需要 2 张图像）")
            else:
                print(f"  FID:   {metrics['fid']:.4f}")
                print(f"  KID:   {metrics['kid']:.6f}")
            if "avg_psnr_mask" in metrics:
                print(f"══ Mask 区域平均指标 ══")
                print(f"  PSNR:  {metrics['avg_psnr_mask']:.4f} dB")
                print(f"  SSIM:  {metrics['avg_ssim_mask']:.4f}")
                print(f"  LPIPS: {metrics['avg_lpips_mask']:.4f}")
            print(f"JSON 已保存到: {json_path}")

            # ── 保存逐张指标 Excel ──
            xlsx_path = os.path.join(output_dir, "per_image_metrics.xlsx")
            _write_metrics_excel(metrics["per_image"], xlsx_path)
            print(f"Excel 已保存到: {xlsx_path}")
            csv_path = os.path.join(output_dir, "per_image_metrics.csv")
            _write_metrics_csv(metrics["per_image"], csv_path)
            print(f"CSV 已保存到: {csv_path}")

        print(f"\n批量处理完成: 成功 {success_count}/{len(pairs)}")

    # ═════════ 单张模式 ═════════
    else:
        success, orig, inp, msk = run_inference(
            pipe, args.image, args.mask, output_dir, args,
            generator=generator, device=device,
            single_stage_generator=single_stage_generator,
        )
        if success and args.compute_metrics:
            mc = MetricCalculator()
            all_m = mc.calculate_all_single(orig, inp, mask=msk)
            metrics = {
                "psnr": all_m["psnr"],
                "ssim": all_m["ssim"],
                "lpips": all_m["lpips"],
                "inception_feature_l2": all_m["inception_feature_l2"],
                "psnr_mask": all_m["psnr_mask"],
                "ssim_mask": all_m["ssim_mask"],
                "lpips_mask": all_m["lpips_mask"],
            }
            with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            print(f"PSNR={all_m['psnr']:.4f}, SSIM={all_m['ssim']:.4f}, "
                  f"LPIPS={all_m['lpips']:.4f}, "
                  f"Inception-L2={all_m['inception_feature_l2']:.4f}")

        print("单张处理完成")


if __name__ == "__main__":
    main()
