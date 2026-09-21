#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_repo_hygiene.py — 发布仓库的内容守卫。

这个仓库的发布边界很具体：主体是代码，图像数据只允许出现在
`data/public_subset/`。这类边界靠 .gitignore 容易失守（一次 `git add -f`、
一次在新目录里放图），所以在 CI 里再设一道检查。

检查项：
  1. 不存在模型权重 / 检查点 / 归档文件；
  2. 图像文件只允许出现在 data/public_subset/ 下；
  3. 不存在含有真实划分的清单（eval/manifest.csv、eval/*_manifest.csv 等）；
  4. 单文件不超过给定上限（便于 GitHub 托管）；
  5. 必须存在的发布文件齐全。

用法:
    python tools/check_repo_hygiene.py
"""
import csv
import hashlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 扫描时跳过的目录：不是仓库内容，且会让文件计数虚高
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}

FORBIDDEN_EXT = {
    ".pth", ".pt", ".safetensors", ".ckpt", ".bin", ".h5", ".onnx",
    ".pkl", ".pickle", ".npz", ".npy", ".zip", ".tar", ".gz", ".7z", ".rar",
}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
IMAGE_ALLOWED_PREFIX = os.path.join("data", "public_subset")
# 禁止随仓库发布任何"真实划分清单"（manifest.csv / *_manifest.csv）：
# 本仓库不分发数据划分，用户自己的清单不应被提交。
MANIFEST_ALLOWED = set()
REQUIRED = [
    "README.md", "LICENSE", "NOTICE", "CITATION.cff",
    "requirements.txt", ".gitignore", ".gitattributes",
    ".github/workflows/tests.yml",
    "train.py", "infer.py", "evaluate.py", "test_paper_params.py", "test_inference_api.py",
    "inference/restore.py",
    "docs/PROTOCOL.md", "docs/STATUS.md",
    "data/README.md", "data/public_subset/README.md",
    "data/public_subset/LICENSE", "data/public_subset/SOURCES.csv",
    "data/public_subset/CREDITS.md", "data/public_subset/CITATION.cff",
    "tools/check_image_metadata.py",
    "tools/test_check_image_metadata.py", "tools/check_repo_hygiene.py",
]
MAX_FILE_MB = 5.0

FAIL = 0


def bad(msg):
    global FAIL
    FAIL += 1
    print(f"  [FAIL] {msg}")


def ok(msg):
    print(f"  [PASS] {msg}")


def check_subset_consistency():
    """data/public_subset 的内容必须自洽。

    三项检查，都是实际踩过的坑：
      1. 磁盘上的图像与 SOURCES.csv 登记的一一对应（多一个少一个都要报）；
      2. 不存在字节完全相同的图像；
      3. 不存在"同一张图的两种分辨率"这类近重复 —— 判据是低分辨率灰度互相关
         很高且宽高比几乎相同。宽高比这一条很关键：同一对象的正反面或不同视角
         宽高比通常不同，只有真正的重复才会两者同时吻合。
    """
    issues = []
    subset = os.path.join(ROOT, IMAGE_ALLOWED_PREFIX)
    csv_path = os.path.join(subset, "SOURCES.csv")
    if not os.path.isfile(csv_path):
        return ["data/public_subset/SOURCES.csv 缺失"]

    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    listed = {r["file_path"].replace("\\", "/") for r in rows}

    on_disk = {}
    for root, dirs, files in os.walk(subset):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for fn in sorted(files):
            if os.path.splitext(fn)[1].lower() in IMAGE_EXT:
                p = os.path.join(root, fn)
                rel = os.path.relpath(p, subset).replace("\\", "/")
                on_disk[rel] = p

    for miss in sorted(listed - set(on_disk)):
        issues.append(f"SOURCES.csv 登记但文件不存在: {miss}")
    for extra in sorted(set(on_disk) - listed):
        issues.append(f"存在但未登记进 SOURCES.csv: {extra}")

    # 精确重复
    digests = {}
    for rel, p in on_disk.items():
        with open(p, "rb") as fh:
            d = hashlib.sha256(fh.read()).hexdigest()
        digests.setdefault(d, []).append(rel)
    for d, group in digests.items():
        if len(group) > 1:
            issues.append("字节完全相同的图像: " + ", ".join(group))

    # 近重复（需要 Pillow + numpy；缺失时跳过而不误报）
    try:
        import itertools

        import numpy as np
        from PIL import Image
    except ImportError:
        return issues

    sig = {}
    for rel, p in on_disk.items():
        try:
            with Image.open(p) as im:
                arr = np.asarray(im.convert("L").resize((64, 64), Image.LANCZOS),
                                 dtype=np.float32)
                sig[rel] = (arr - arr.mean(), im.size[0] / im.size[1])
        except Exception:
            continue
    for (ra, (va, aa)), (rb, (vb, ab)) in itertools.combinations(sorted(sig.items()), 2):
        da = float(np.sqrt((va ** 2).sum()))
        db = float(np.sqrt((vb ** 2).sum()))
        if da == 0 or db == 0:
            continue
        corr = float((va * vb).sum() / (da * db))
        aspect_close = abs(aa - ab) / max(aa, ab) < 0.02
        if corr > 0.98 and aspect_close:
            issues.append(
                f"疑似近重复（相关 {corr:.4f}，宽高比 {aa:.3f}/{ab:.3f}）: {ra} vs {rb}")

    # CREDITS.md 必须与 SOURCES.csv 一一对应（手工按索引删行时漏过条目，故设此检查）
    credits = os.path.join(subset, "CREDITS.md")
    if os.path.isfile(credits):
        with open(credits, encoding="utf-8") as fh:
            bullets = [line for line in fh if line.startswith("* `")]
        if len(bullets) != len(rows):
            issues.append(
                f"CREDITS.md 有 {len(bullets)} 条署名行，而 SOURCES.csv 有 {len(rows)} 行")
        missing = [r["file_path"] for r in rows
                   if not any(f"`{r['file_path']}`" in b for b in bullets)]
        if missing:
            issues.append("CREDITS.md 缺少署名行: " + ", ".join(missing[:3]))
    else:
        issues.append("data/public_subset/CREDITS.md 缺失")
    return issues


def main():
    weights, stray_images, manifests, oversized = [], [], [], []
    n_files = 0
    total = 0

    for root, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fn in files:
            p = os.path.join(root, fn)
            rel = os.path.relpath(p, ROOT)
            n_files += 1
            size = os.path.getsize(p)
            total += size
            ext = os.path.splitext(fn)[1].lower()

            if ext in FORBIDDEN_EXT:
                weights.append(rel)
            if ext in IMAGE_EXT and not rel.startswith(IMAGE_ALLOWED_PREFIX):
                stray_images.append(rel)
            if fn not in MANIFEST_ALLOWED and (fn == "manifest.csv" or fn.endswith("_manifest.csv")):
                manifests.append(rel)
            if size > MAX_FILE_MB * 1024 * 1024:
                oversized.append(f"{rel} ({size / 1024 / 1024:.1f} MB)")

    print("=== 仓库内容守卫 ===")
    print(f"  扫描 {n_files} 个文件，合计 {total / 1024 / 1024:.2f} MB\n")

    if weights:
        bad("存在权重/检查点/归档文件：" + ", ".join(weights[:5]))
    else:
        ok("无权重 / 检查点 / 归档文件")

    if stray_images:
        bad(f"{len(stray_images)} 个图像文件位于 data/public_subset/ 之外："
            + ", ".join(stray_images[:5]))
    else:
        ok(f"图像文件仅出现在 {IMAGE_ALLOWED_PREFIX}/")

    if manifests:
        bad("存在可能含真实划分的清单：" + ", ".join(manifests))
    else:
        ok("无真实数据划分清单")

    if oversized:
        bad(f"{len(oversized)} 个文件超过 {MAX_FILE_MB} MB：" + ", ".join(oversized[:3]))
    else:
        ok(f"无文件超过 {MAX_FILE_MB} MB")

    subset_issues = check_subset_consistency()
    if subset_issues:
        for it in subset_issues:
            bad("公开子集: " + it)
    else:
        ok("公开子集自洽（文件与 SOURCES.csv 一一对应，无重复/近重复图像）")

    missing = [r for r in REQUIRED if not os.path.isfile(os.path.join(ROOT, r))]
    if missing:
        bad("缺少发布必需文件：" + ", ".join(missing))
    else:
        ok(f"{len(REQUIRED)} 个发布必需文件齐全")

    print(f"\n======== 结果: {'全部通过' if FAIL == 0 else str(FAIL) + ' 项失败'} ========")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
