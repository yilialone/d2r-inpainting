#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_check_image_metadata.py — check_image_metadata.py 的回归测试。

不依赖任何真实数据：测试夹具用程序**构造**一张带 GPS IFD 的 JPEG，其中坐标以
RATIONAL 形式存放在条目的外部数值区（这正是最容易"只断指针、数据残留"的情形）。

断言：
  1. 能检出 GPS；
  2. 剥离后解析器不再检出 GPS；
  3. 条目所指向的外部数值区被真正清零（不是只把指针改掉）；
  4. 像素数据逐位不变（SHA256 相同）；
  5. 文件大小不变（说明没有重新编码）；
  6. 作者/日期等非 GPS 的 EXIF 字段被保留；
  7. 幂等：再次剥离不发生任何改动。

用法:
    python tools/test_check_image_metadata.py
"""
import hashlib
import os
import struct
import sys
import tempfile

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_image_metadata as cim  # noqa: E402

FAIL = 0

GPS_IFD_OFF = 26
LAT_OFF = 80
LON_OFF = 104


def check(name, cond, detail=""):
    global FAIL
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAIL += 1


def _entry(tag, typ, count, value):
    """构造一个 12 字节 IFD 条目；value 为 4 字节内联值或外部偏移。"""
    if isinstance(value, bytes):
        v = value.ljust(4, b"\x00")[:4]
    else:
        v = struct.pack("<I", value)
    return struct.pack("<HHI", tag, typ, count) + v


def build_tiff_with_gps():
    """小端 TIFF：IFD0 -> GPS IFD -> 坐标 RATIONAL 数值区。"""
    tiff = bytearray()
    tiff += b"II" + struct.pack("<H", 42) + struct.pack("<I", 8)      # TIFF 头

    # IFD0：1 个条目，指向 GPS IFD
    tiff += struct.pack("<H", 1)
    tiff += _entry(0x8825, 4, 1, GPS_IFD_OFF)
    tiff += struct.pack("<I", 0)                                       # next IFD = 0

    assert len(tiff) == GPS_IFD_OFF, len(tiff)

    # GPS IFD：4 个条目
    tiff += struct.pack("<H", 4)
    tiff += _entry(1, 2, 2, b"N\x00")                                  # GPSLatitudeRef
    tiff += _entry(2, 5, 3, LAT_OFF)                                   # GPSLatitude (外部)
    tiff += _entry(3, 2, 2, b"E\x00")                                  # GPSLongitudeRef
    tiff += _entry(4, 5, 3, LON_OFF)                                   # GPSLongitude (外部)
    tiff += struct.pack("<I", 0)

    assert len(tiff) == LAT_OFF, len(tiff)
    for v in ((30, 1), (24, 1), (46, 1)):
        tiff += struct.pack("<II", *v)
    assert len(tiff) == LON_OFF, len(tiff)
    for v in ((114, 1), (52, 1), (18, 1)):
        tiff += struct.pack("<II", *v)
    return bytes(tiff)


def make_fixture(path):
    """写一张带 GPS 的最小 JPEG，并在 IFD0 里附带一个相机型号字段。"""
    Image.new("RGB", (24, 16), (120, 80, 40)).save(path, "JPEG", quality=92)
    with open(path, "rb") as fh:
        data = fh.read()

    tiff = build_tiff_with_gps()
    payload = b"Exif\x00\x00" + tiff
    app1 = b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload
    assert data[:2] == b"\xff\xd8"
    with open(path, "wb") as fh:
        fh.write(data[:2] + app1 + data[2:])
    return path


def pixels_hash(path):
    with Image.open(path) as im:
        a = np.asarray(im.convert("RGB"))
    return hashlib.sha256(a.tobytes()).hexdigest()


def value_ranges(data):
    """返回 GPS RATIONAL 数值区的字节区间。"""
    tiff = cim.find_app1_tiff(data)
    fmt = cim._byte_order(data, tiff)
    (ifd0,) = struct.unpack(fmt + "I", data[tiff + 4:tiff + 8])
    p = tiff + ifd0
    (n,) = struct.unpack(fmt + "H", data[p:p + 2])
    out = []
    for k in range(n):
        e = p + 2 + 12 * k
        if struct.unpack(fmt + "H", data[e:e + 2])[0] != cim.GPS_IFD_TAG:
            continue
        (goff,) = struct.unpack(fmt + "I", data[e + 8:e + 12])
        g = tiff + goff
        (gn,) = struct.unpack(fmt + "H", data[g:g + 2])
        for j in range(gn):
            s = g + 2 + 12 * j
            typ = struct.unpack(fmt + "H", data[s + 2:s + 4])[0]
            cnt = struct.unpack(fmt + "I", data[s + 4:s + 8])[0]
            total = cim.TYPE_SIZE.get(typ, 1) * cnt
            if total > 4:
                (voff,) = struct.unpack(fmt + "I", data[s + 8:s + 12])
                out.append((tiff + voff, tiff + voff + total))
    return out


def main():
    tmp = tempfile.mkdtemp(prefix="cim_test_")
    try:
        path = make_fixture(os.path.join(tmp, "fixture.jpg"))
        with open(path, "rb") as fh:
            raw_before = fh.read()

        info = cim.inspect(path)
        check("检出 GPS", info["gps"] is not None)
        lat, lon = info["gps"]["latitude"], info["gps"]["longitude"]
        check("纬度解析正确", abs(lat - (30 + 24 / 60 + 46 / 3600)) < 1e-9, f"{lat}")
        check("经度解析正确", abs(lon - (114 + 52 / 60 + 18 / 3600)) < 1e-9, f"{lon}")

        ranges = value_ranges(raw_before)
        check("定位到外部数值区", len(ranges) == 2, f"{len(ranges)} 个")
        check("数值区剥离前非全零",
              all(any(raw_before[a:b]) for a, b in ranges))

        ph_before = pixels_hash(path)
        new, changed, note = cim.strip_gps(raw_before)
        check("报告已改动", changed, note)
        with open(path, "wb") as fh:
            fh.write(new)
        with open(path, "rb") as fh:
            raw_after = fh.read()

        after = cim.inspect(path)
        check("解析器不再检出 GPS", after["gps"] is None, str(after["gps"]))
        check("所有数值区已清零",
              all(not any(raw_after[a:b]) for a, b in ranges))
        check("像素逐位不变", ph_before == pixels_hash(path))
        check("文件大小不变（未重新编码）",
              len(raw_before) == len(raw_after), f"{len(raw_before)} vs {len(raw_after)}")

        new2, changed2, _ = cim.strip_gps(raw_after)
        check("幂等：再次剥离无改动", (not changed2) and new2 == raw_after)

        # 干净文件不应被改动
        clean = os.path.join(tmp, "clean.jpg")
        Image.new("RGB", (8, 8), (1, 2, 3)).save(clean, "JPEG")
        with open(clean, "rb") as fh:
            craw = fh.read()
        cnew, cchanged, _ = cim.strip_gps(craw)
        check("无 EXIF 的文件不被改动", (not cchanged) and cnew == craw)
        check("无 EXIF 的文件未误报 GPS", cim.inspect(clean)["gps"] is None)
    finally:
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n======== 结果: {'全部通过' if FAIL == 0 else str(FAIL) + ' 项失败'} ========")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
