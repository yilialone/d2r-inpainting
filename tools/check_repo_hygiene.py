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
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FORBIDDEN_EXT = {
    ".pth", ".pt", ".safetensors", ".ckpt", ".bin", ".h5", ".onnx",
    ".pkl", ".pickle", ".npz", ".npy", ".zip", ".tar", ".gz", ".7z", ".rar",
}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
IMAGE_ALLOWED_PREFIX = os.path.join("data", "public_subset")
# 禁止随仓库发布任何"真实划分清单"：manifest.csv 或 *_manifest.csv。
# 只有纯占位的 eval/manifest_template.csv 例外。
MANIFEST_ALLOWED = {"manifest_template.csv"}
REQUIRED = [
    "README.md", "LICENSE", "NOTICE", "CITATION.cff",
    "requirements.txt", ".gitignore", ".gitattributes",
    ".github/workflows/tests.yml",
    "train.py", "infer.py", "evaluate.py", "test_paper_params.py",
    "docs/PROTOCOL.md", "docs/RESULTS.md", "docs/ARCHITECTURE.md", "docs/STATUS.md",
    "data/README.md", "data/public_subset/README.md",
    "data/public_subset/LICENSE", "data/public_subset/SOURCES.csv",
    "data/public_subset/CREDITS.md", "data/public_subset/CITATION.cff",
    "tools/check_protocol.py", "tools/check_image_metadata.py",
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


def main():
    weights, stray_images, manifests, oversized = [], [], [], []
    n_files = 0
    total = 0

    for root, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in (".git", "__pycache__")]
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
        ok("无真实划分清单（仅保留 manifest_template.csv）")

    if oversized:
        bad(f"{len(oversized)} 个文件超过 {MAX_FILE_MB} MB：" + ", ".join(oversized[:3]))
    else:
        ok(f"无文件超过 {MAX_FILE_MB} MB")

    missing = [r for r in REQUIRED if not os.path.isfile(os.path.join(ROOT, r))]
    if missing:
        bad("缺少发布必需文件：" + ", ".join(missing))
    else:
        ok(f"{len(REQUIRED)} 个发布必需文件齐全")

    print(f"\n======== 结果: {'全部通过' if FAIL == 0 else str(FAIL) + ' 项失败'} ========")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
