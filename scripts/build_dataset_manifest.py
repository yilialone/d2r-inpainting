#!/usr/bin/env python3
"""Build an auditable image/mask manifest without inferring archaeological metadata.

The script records file identities and leaves provenance-related fields blank for
manual verification. It never invents ownership, rights, object identity, or shan
character counts.
"""

import argparse
import csv
import hashlib
import os
import re


FIELDS = [
    "split", "sample_id", "image_path", "mask_path", "image_sha256", "mask_sha256",
    "object_id", "source_category", "rights_basis", "shan_count", "notes",
]


def numbered_files(directory, prefix):
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$", re.IGNORECASE)
    rows = []
    for name in os.listdir(directory):
        stem, ext = os.path.splitext(name)
        match = pattern.match(stem)
        if match and ext.lower() in {".jpg", ".jpeg", ".png"}:
            rows.append((int(match.group(1)), os.path.abspath(os.path.join(directory, name))))
    return sorted(rows)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="生成待人工核验的数据集 manifest")
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--mask_dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    images = numbered_files(args.image_dir, "mirror")
    masks = numbered_files(args.mask_dir, "img")
    if len(images) != len(masks):
        raise ValueError(
            f"图像数量({len(images)})与 mask 数量({len(masks)})不一致；拒绝静默截断"
        )

    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for (_, image_path), (_, mask_path) in zip(images, masks):
            image_stem = os.path.splitext(os.path.basename(image_path))[0]
            mask_stem = os.path.splitext(os.path.basename(mask_path))[0]
            writer.writerow({
                "split": args.split,
                "sample_id": f"{args.split}_{image_stem}__{mask_stem}",
                "image_path": os.path.relpath(image_path, output_dir),
                "mask_path": os.path.relpath(mask_path, output_dir),
                "image_sha256": sha256(image_path),
                "mask_sha256": sha256(mask_path),
                "object_id": "",
                "source_category": "",
                "rights_basis": "",
                "shan_count": "",
                "notes": "VERIFY_PAIRING_AND_METADATA",
            })
    print(f"已写入 {len(images)} 行: {os.path.abspath(args.output)}")
    print("注意：object_id/source_category/rights_basis/shan_count 必须人工核验后填写。")


if __name__ == "__main__":
    main()
