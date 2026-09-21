#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_image_metadata.py — 图片元数据体检与 GPS 剥离。

为什么需要这个工具
------------------
图像元数据是一条很容易被忽略的泄露通道。对考古类数据集尤其如此：手机/相机写入的
GPS 标签可以精确到米级，公开出去等于公开遗址坐标。相机型号与拍摄日期则可能是有意
保留的（用于记录器物被记录的时间），所以不能一刀切地删掉全部 EXIF。

本工具因此把两件事分开：
  * 体检（默认）：逐图报告是否存在 GPS、以及相机型号/拍摄日期等可识别字段；
  * 剥离（--strip-gps）：**仅**移除 GPS 信息。

剥离是无损的
------------
实现方式是直接在 JPEG 的 APP1/TIFF 结构里做字节级修补，**不重新编码**，因此像素数据
逐位不变（重新用 PIL 保存会引入有损压缩）。具体做三件事：
  1. 把 GPS IFD 的条目数清零；
  2. 零化每个 GPS 条目的 12 字节条目本身；
  3. 零化条目所指向的外部数值区（坐标就是存在那里的有理数），避免"只断指针、
     数据仍留在文件里"的假剥离；
  4. 把 IFD0 里的 GPSInfo 标签号改成私有标签 0xC825，解析器不再将其识别为 GPS。

用法
----
    python tools/check_image_metadata.py --dir data/public_subset
    python tools/check_image_metadata.py --dir data/public_subset --strip-gps
    python tools/check_image_metadata.py --dir some/dir --json report.json

退出码：0 = 无 GPS 残留；1 = 存在 GPS 且未剥离（或剥离后仍可检出）。
"""
import argparse
import json
import os
import struct
import sys

from PIL import ExifTags, Image

GPS_IFD_TAG = 0x8825
EXIF_IFD_TAG = 0x8769
PRIVATE_GPS_TAG = 0xC825

# TIFF 字段类型 → 单元素字节数
TYPE_SIZE = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8}

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


# --------------------------------------------------------------------------- #
# JPEG / TIFF 字节级定位
# --------------------------------------------------------------------------- #
def find_app1_tiff(data):
    """返回 TIFF 头在 data 中的偏移；找不到返回 None。"""
    if data[:2] != b"\xff\xd8":
        return None
    i = 2
    n = len(data)
    while i < n - 1:
        if data[i] != 0xFF:
            return None
        marker = data[i + 1]
        if marker == 0xD8 or marker == 0x01 or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        if marker == 0xDA:  # 进入压缩数据，EXIF 不会在后面
            return None
        if i + 4 > n:
            return None
        seglen = struct.unpack(">H", data[i + 2:i + 4])[0]
        if marker == 0xE1 and data[i + 4:i + 10] == b"Exif\x00\x00":
            return i + 10
        i += 2 + seglen
    return None


def _byte_order(data, tiff):
    e = data[tiff:tiff + 2]
    if e == b"II":
        return "<"
    if e == b"MM":
        return ">"
    return None


def strip_gps(data):
    """无损移除 JPEG 中的 GPS 信息。返回 (新字节, 是否改动, 说明)。"""
    tiff = find_app1_tiff(data)
    if tiff is None:
        return data, False, "无 EXIF APP1"
    fmt = _byte_order(data, tiff)
    if fmt is None:
        return data, False, "TIFF 字节序无法识别"

    (ifd0_off,) = struct.unpack(fmt + "I", data[tiff + 4:tiff + 8])
    p = tiff + ifd0_off
    (count,) = struct.unpack(fmt + "H", data[p:p + 2])

    buf = bytearray(data)
    removed = []
    for k in range(count):
        e = p + 2 + 12 * k
        tag = struct.unpack(fmt + "H", buf[e:e + 2])[0]
        if tag != GPS_IFD_TAG:
            continue
        (gps_off,) = struct.unpack(fmt + "I", buf[e + 8:e + 12])
        g = tiff + gps_off
        (gn,) = struct.unpack(fmt + "H", buf[g:g + 2])

        # 先把每个条目指向的外部数值区零化（坐标的有理数就存在那里）
        for j in range(gn):
            s = g + 2 + 12 * j
            gtag = struct.unpack(fmt + "H", buf[s:s + 2])[0]
            gtype = struct.unpack(fmt + "H", buf[s + 2:s + 4])[0]
            gcnt = struct.unpack(fmt + "I", buf[s + 4:s + 8])[0]
            removed.append(ExifTags.GPSTAGS.get(gtag, f"tag{gtag}"))
            total = TYPE_SIZE.get(gtype, 1) * gcnt
            if total > 4:
                (voff,) = struct.unpack(fmt + "I", buf[s + 8:s + 12])
                a = tiff + voff
                b = min(a + total, len(buf))
                if 0 <= a < len(buf):
                    buf[a:b] = b"\x00" * (b - a)
            buf[s:s + 12] = b"\x00" * 12

        buf[g:g + 2] = b"\x00\x00"                       # GPS IFD 条目数清零
        buf[e:e + 2] = struct.pack(fmt + "H", PRIVATE_GPS_TAG)  # 断开 GPS 识别

    if not removed:
        return data, False, "无 GPS IFD"
    return bytes(buf), True, "已移除: " + ", ".join(sorted(set(removed)))


# --------------------------------------------------------------------------- #
# 读取与报告
# --------------------------------------------------------------------------- #
def _to_deg(v):
    try:
        d, m, s = (float(x) for x in v)
        return d + m / 60.0 + s / 3600.0
    except Exception:
        return None


def inspect(path):
    """返回该图的元数据摘要。"""
    out = {"file": path, "has_exif": False, "gps": None, "camera": None,
           "date": None, "software": None, "raw_len": os.path.getsize(path)}
    try:
        with Image.open(path) as im:
            exif = im.getexif()
    except Exception as exc:
        out["error"] = repr(exc)
        return out
    if not exif:
        return out
    out["has_exif"] = True
    ifd0 = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
    try:
        sub = {ExifTags.TAGS.get(k, k): v for k, v in exif.get_ifd(EXIF_IFD_TAG).items()}
    except Exception:
        sub = {}
    try:
        gps = {ExifTags.GPSTAGS.get(k, k): v for k, v in exif.get_ifd(GPS_IFD_TAG).items()}
    except Exception:
        gps = {}

    if gps:
        lat = _to_deg(gps.get("GPSLatitude")) if "GPSLatitude" in gps else None
        lon = _to_deg(gps.get("GPSLongitude")) if "GPSLongitude" in gps else None
        if gps.get("GPSLatitudeRef") in ("S", b"S") and lat is not None:
            lat = -lat
        if gps.get("GPSLongitudeRef") in ("W", b"W") and lon is not None:
            lon = -lon
        out["gps"] = {"tags": sorted(str(k) for k in gps.keys()),
                      "latitude": lat, "longitude": lon,
                      "date": str(gps.get("GPSDateStamp", "")) or None,
                      "method": str(gps.get("GPSProcessingMethod", "")) or None}
    make = sub.get("Make") or ifd0.get("Make")
    model = sub.get("Model") or ifd0.get("Model")
    cam = " ".join(str(x).strip() for x in (make, model) if x)
    out["camera"] = cam or None
    dt = sub.get("DateTimeOriginal") or ifd0.get("DateTime")
    out["date"] = str(dt) if dt else None
    sw = ifd0.get("Software")
    out["software"] = str(sw).strip() if sw else None
    return out


def main():
    ap = argparse.ArgumentParser(description="图片元数据体检 / GPS 剥离")
    ap.add_argument("--dir", required=True, help="要扫描的目录")
    ap.add_argument("--strip-gps", action="store_true",
                    help="无损移除 GPS（不重新编码，像素逐位不变）")
    ap.add_argument("--json", default=None, help="把报告写到该路径")
    args = ap.parse_args()

    if not os.path.isdir(args.dir):
        raise SystemExit(f"目录不存在: {args.dir}")

    files = []
    for root, dirs, names in os.walk(args.dir):
        dirs.sort()
        for fn in sorted(names):
            if fn.lower().endswith(IMAGE_EXT):
                files.append(os.path.join(root, fn))
    if not files:
        raise SystemExit(f"{args.dir} 下未找到图像")

    print(f"扫描 {len(files)} 个图像文件：{args.dir}\n")
    print(f"{'文件':<50}{'EXIF':<7}{'GPS':<7}{'相机':<24}{'日期'}")
    print("-" * 118)

    report, n_gps, n_stripped, n_fail = [], 0, 0, 0
    for path in files:
        rel = os.path.relpath(path, args.dir)
        info = inspect(path)
        had_gps = info["gps"] is not None
        if had_gps:
            n_gps += 1

        status = "有" if had_gps else ("空" if info["has_exif"] else "无")
        if args.strip_gps and had_gps:
            with open(path, "rb") as fh:
                data = fh.read()
            new, changed, note = strip_gps(data)
            if changed:
                with open(path, "wb") as fh:
                    fh.write(new)
                after = inspect(path)
                if after["gps"] is None:
                    n_stripped += 1
                    status = "已剥离"
                else:
                    n_fail += 1
                    status = "剥离失败!"
                info["strip_note"] = note
            else:
                info["strip_note"] = note

        cam = (info["camera"] or "—")[:22]
        print(f"{rel:<50}{('有' if info['has_exif'] else '无'):<7}{status:<7}{cam:<24}{info['date'] or '—'}")
        if had_gps and info["gps"].get("latitude") is not None:
            print(f"    └─ GPS 坐标: {info['gps']['latitude']:.6f}, "
                  f"{info['gps']['longitude']:.6f}"
                  + (f"   日期={info['gps']['date']}" if info["gps"].get("date") else "")
                  + (f"   定位方式={info['gps']['method']}" if info["gps"].get("method") else ""))
        report.append(info)

    print()
    print(f"含 GPS 的文件            : {n_gps}")
    if args.strip_gps:
        print(f"成功剥离                 : {n_stripped}")
        if n_fail:
            print(f"剥离后仍可检出           : {n_fail}  ⚠")
    remaining = n_gps - n_stripped if args.strip_gps else n_gps
    print()
    if remaining > 0:
        print("⚠ 仍存在 GPS 信息。" + ("" if args.strip_gps else " 加 --strip-gps 可无损移除。"))
    else:
        print("✅ 无 GPS 信息残留。")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
        print(f"报告已写出: {args.json}")

    return 1 if remaining > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
