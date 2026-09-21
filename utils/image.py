#!/usr/bin/env python3
"""
通用图像处理工具函数
"""

import os
import glob
import re
import csv
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def resize_to_target(image, target_size, resample=Image.LANCZOS):
    """将图像调整到目标尺寸"""
    if image.size == target_size:
        return image
    return image.resize(target_size, resample)


def create_red_mask_overlay(image, mask):
    """创建原图+半透明红色mask叠加图"""
    image_rgba = image.convert("RGBA")
    red_overlay = Image.new("RGBA", image.size, (255, 0, 0, 128))
    mask_binary = mask.convert("L")
    mask_array = mask_binary.point(lambda x: 255 if x > 128 else 0)
    red_mask = Image.new("RGBA", image.size, (0, 0, 0, 0))
    red_mask.paste(red_overlay, (0, 0), mask_array)
    result = Image.alpha_composite(image_rgba, red_mask)
    return result.convert("RGB")


def create_comparison_image(original, mask_overlay, inpainted):
    """创建三图横向对比图：原图 | 原图+Mask | 修复图"""
    width, height = original.size
    margin = 20
    caption_height = 40
    canvas_width = width * 3 + margin * 4
    canvas_height = height + caption_height + margin * 2
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")

    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except Exception:
        try:
            font = ImageFont.truetype("Arial.ttf", 24)
        except Exception:
            font = ImageFont.load_default()

    draw = ImageDraw.Draw(canvas)

    canvas.paste(original, (margin, margin))
    draw.text((margin + width // 2 - 30, height + margin), "原图", fill="black", font=font)

    canvas.paste(mask_overlay, (width + margin * 2, margin))
    draw.text((width + margin * 2 + width // 2 - 60, height + margin), "原图+Mask", fill="black", font=font)

    canvas.paste(inpainted, (width * 2 + margin * 3, margin))
    draw.text((width * 2 + margin * 3 + width // 2 - 30, height + margin), "修复图", fill="black", font=font)

    return canvas


def ensure_output_directory(output_path):
    """确保输出路径有效，返回目录路径"""
    if not output_path.strip():
        output_path = "./output"
    if os.path.splitext(output_path)[1]:
        output_path = os.path.dirname(output_path)
    os.makedirs(output_path, exist_ok=True)
    return output_path


def find_image_mask_pairs(image_dir, mask_dir, manifest_path=None):
    """在目录中查找图像和掩码文件对

    图像命名: mirror{数字}.jpg/.jpeg/.png
    掩码命名: img{数字}.png/.jpg/.jpeg

    优先通过数字精确匹配；若无匹配，按各自数字排序后按位置配对
    （与 dataset/dataset.py 行为一致）。
    """
    if manifest_path:
        manifest_path = os.path.abspath(manifest_path)
        manifest_dir = os.path.dirname(manifest_path)
        pairs = []
        with open(manifest_path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            required = {"sample_id", "image_path", "mask_path"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"manifest 缺少字段: {sorted(missing)}")
            for row in reader:
                image_path = row["image_path"].strip()
                mask_path = row["mask_path"].strip()
                if not os.path.isabs(image_path):
                    image_path = os.path.normpath(os.path.join(manifest_dir, image_path))
                if not os.path.isabs(mask_path):
                    mask_path = os.path.normpath(os.path.join(manifest_dir, mask_path))
                if not os.path.isfile(image_path) or not os.path.isfile(mask_path):
                    raise FileNotFoundError(f"manifest 文件不存在: {image_path} / {mask_path}")
                pairs.append((image_path, mask_path, row["sample_id"].strip()))
        print(f"从 manifest 读取 {len(pairs)} 对样本")
        return pairs

    pairs = []

    image_files = []
    for ext in ['.jpg', '.jpeg', '.png']:
        image_files.extend(glob.glob(os.path.join(image_dir, f"mirror*{ext}")))
    print(f"在图像目录中找到 {len(image_files)} 个mirror文件")

    mask_files = []
    for ext in ['.png', '.jpg', '.jpeg']:
        mask_files.extend(glob.glob(os.path.join(mask_dir, f"img*{ext}")))
    print(f"在掩码目录中找到 {len(mask_files)} 个img文件")

    def extract_number(f, pattern):
        match = re.search(pattern, os.path.basename(f))
        return int(match.group(1)) if match else None

    if len(image_files) != len(mask_files):
        raise ValueError(
            f"图像数量({len(image_files)})与 mask 数量({len(mask_files)})不一致；"
            "拒绝静默截断。请提供 --manifest 明确配对关系"
        )
    sorted_images = sorted(image_files, key=lambda f: extract_number(f, r'mirror(\d+)') or 0)
    sorted_masks = sorted(mask_files, key=lambda f: extract_number(f, r'img(\d+)') or 0)
    print("未提供 manifest：按数字排序后逐项配对；正式实验应使用 manifest")
    for i, (image_path, mask_path) in enumerate(zip(sorted_images, sorted_masks)):
        img_num = extract_number(image_path, r'mirror(\d+)')
        sample_id = f"mirror{img_num}" if img_num is not None else f"sample_{i:06d}"
        pairs.append((image_path, mask_path, sample_id))

    print(f"最终配对数量: {len(pairs)}")
    return pairs
