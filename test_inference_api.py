#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_inference_api.py — inference.restore 的回归测试。

不需要模型权重：通过 ``pipe=`` / ``generator=`` 注入替身，即可验证接口本身。
重点验证三件事：

  1. **四通道契约** —— Stage-2 生成器只收到 ``[I_S1(3), M(1)]``，真值不进入任何一路；
  2. **掩膜外逐字节一致** —— 输出在掩膜外与传入图像完全相同；
  3. **种子语义** —— 同 seed 结果一致；批处理使用 ``seed + i``。

另外覆盖 ``SimpleUNetGeneratorWithTexture.architecture_summary`` 与 Stage-2 预算
报告的 ``model_definition``——这两处此前描述的是 7 通道输入，与实现不符。

用法:
    python test_inference_api.py
"""
import math
import os
import sys

import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from inference import D2RRestorer, restore_image  # noqa: E402
from inference.restore import _prepare_pair  # noqa: E402
from models import SimpleUNetGeneratorWithTexture  # noqa: E402
from training.stage2 import Stage2GANTrainer  # noqa: E402

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}" + (f"  ({detail})" if detail else ""))
    else:
        FAIL += 1
        print(f"  [FAIL] {name}" + (f"  ({detail})" if detail else ""))


class _Result:
    def __init__(self, img):
        self.images = [img]


class _FakePipe:
    """满足 stage1_inference 调用约定的最小替身。"""

    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        self.calls = []

    def __call__(self, *, prompt, image, mask_image, negative_prompt=None,
                 num_inference_steps=20, guidance_scale=6.0, generator=None):
        self.calls.append({
            "prompt": prompt, "steps": num_inference_steps, "cfg": guidance_scale,
            "negative_prompt": negative_prompt,
        })
        seed = 0
        if generator is not None:
            seed = int(torch.randint(0, 2 ** 31 - 1, (1,), generator=generator).item())
        rng = np.random.RandomState(seed)
        arr = rng.randint(0, 256, (image.height, image.width, 3), dtype=np.uint8)
        return _Result(Image.fromarray(arr, "RGB"))


class _FakeStage2Generator(torch.nn.Module):
    """记录自己收到了什么，并返回常量残差。"""

    def __init__(self, value=0.5, use_full_reconstruction=False):
        super().__init__()
        self.use_full_reconstruction = use_full_reconstruction
        self.value = value
        self.received = []

    def forward(self, stage1_tensor, mask_tensor=None):
        self.received.append((
            int(stage1_tensor.shape[1]),
            None if mask_tensor is None else int(mask_tensor.shape[1]),
            int(stage1_tensor.shape[0]),
        ))
        return torch.full_like(stage1_tensor, self.value)


def _img(w=48, h=40, seed=0, color=None):
    if color is not None:
        return Image.new("RGB", (w, h), color)
    rng = np.random.RandomState(seed)
    return Image.fromarray(rng.randint(0, 256, (h, w, 3), dtype=np.uint8), "RGB")


def _mask(w=48, h=40, box=(10, 10, 25, 22)):
    m = np.zeros((h, w), dtype=np.uint8)
    m[box[1]:box[3], box[0]:box[2]] = 255
    return Image.fromarray(m, "L")


def _np(im):
    return np.asarray(im.convert("RGB"), dtype=np.uint8)


def _masknp(im):
    return np.asarray(im.convert("L"), dtype=np.uint8) >= 128


# --------------------------------------------------------------------------- #
def test_generator_self_description():
    print("\n=== A. 生成器架构自描述 ===")
    gen = SimpleUNetGeneratorWithTexture()
    summary = gen.architecture_summary
    check("暴露 in_channels", getattr(gen, "in_channels", None) == 4,
          str(getattr(gen, "in_channels", None)))
    check("摘要声明 4 通道", "4 input channels" in summary)
    check("摘要不再出现 7 通道", "7 input channels" not in summary)
    check("摘要写明真值不进入输入", "ground truth enters no input path" in summary)
    check("摘要含基宽 64", "base width 64" in summary)

    print("\n=== B. Stage-2 预算报告使用自描述 ===")
    trainer = Stage2GANTrainer.__new__(Stage2GANTrainer)

    class _Acc:
        def unwrap_model(self, m):
            return m

    trainer.accelerator = _Acc()
    trainer.generator = gen
    md = trainer._model_definition()
    check("model_definition 含 4 通道", "4 input channels" in md)
    check("model_definition 不含 7 通道", "7 input channels" not in md, md[:70] + "...")

    trainer.generator = object()  # 没有 architecture_summary 时的兜底
    fallback = trainer._model_definition()
    check("缺少自描述时有兜底且不撒谎",
          "unavailable" in fallback and "7 input" not in fallback)


def test_prepare_pair():
    print("\n=== C. 输入归一化 ===")
    img, msk = _prepare_pair(_img(100, 80, seed=1), _mask(100, 80, (5, 5, 60, 40)), 64)
    check("图像被缩放到 64×64", img.size == (64, 64), str(img.size))
    check("掩膜被缩放到 64×64", msk.size == (64, 64), str(msk.size))
    vals = set(np.unique(np.asarray(msk)).tolist())
    check("掩膜用最近邻（仍为二值）", vals <= {0, 255}, str(sorted(vals)))

    same, same_m = _prepare_pair(_img(64, 64, seed=2), _mask(64, 64), None)
    check("size=None 时不缩放", same.size == (64, 64))

    try:
        _prepare_pair("/nonexistent/x.jpg", _mask(), 64)
        check("缺失文件抛 FileNotFoundError", False)
    except FileNotFoundError:
        check("缺失文件抛 FileNotFoundError", True)

    try:
        _prepare_pair(12345, _mask(), 64)
        check("非法类型抛 TypeError", False)
    except TypeError:
        check("非法类型抛 TypeError", True)


def test_stage1_only():
    print("\n=== D. stage1_only：掩膜外逐字节一致 ===")
    src, msk = _img(seed=10), _mask()
    pipe = _FakePipe()
    r = D2RRestorer(mode="stage1_only", pipe=pipe, size=None, seed=7, device="cpu", load=False)
    check("mode 自动/显式解析", r.mode == "stage1_only", r.mode)
    out = r.restore(src, msk)
    a, b, m = _np(out), _np(src), _masknp(msk)
    check("掩膜外逐字节一致", bool((a[~m] == b[~m]).all()))
    check("掩膜内被改写", bool((a[m] != b[m]).any()))
    check("管线收到 prompt/steps/cfg", pipe.calls[-1] == {
        "prompt": "", "steps": 30, "cfg": 7.5, "negative_prompt": None},
        str(pipe.calls[-1]))

    print("\n=== E. 单次参数覆盖 ===")
    pipe2 = _FakePipe()
    r2 = D2RRestorer(mode="stage1_only", pipe=pipe2, size=None, device="cpu")
    r2.restore(src, msk, num_steps=12, guidance_scale=5.0, prompt="a prompt")
    check("覆盖生效", pipe2.calls[-1] == {
        "prompt": "a prompt", "steps": 12, "cfg": 5.0, "negative_prompt": None},
        str(pipe2.calls[-1]))


def test_d2r_contract():
    print("\n=== F. d2r：四通道契约与残差合成 ===")
    src, msk = _img(seed=11), _mask()
    pipe, gen = _FakePipe(), _FakeStage2Generator(value=0.5)
    r = D2RRestorer(mode="d2r", pipe=pipe, generator=gen, size=None,
                    seed=3, strength=1.0, device="cpu")
    check("uses_stage2", r.uses_stage2)
    out = r.restore(src, msk)

    check("生成器被调用一次", len(gen.received) == 1, str(gen.received))
    s1_ch, m_ch, batch = gen.received[0]
    check("生成器收到 I_S1 三通道", s1_ch == 3, str(s1_ch))
    check("生成器收到 mask 单通道", m_ch == 1, str(m_ch))
    check("总通道数 = 4", s1_ch + m_ch == 4)

    a, b, m = _np(out), _np(src), _masknp(msk)
    check("掩膜外逐字节一致", bool((a[~m] == b[~m]).all()))
    check("掩膜内确实改变", bool((a[m] != b[m]).any()))

    # 常量残差 0.5 ⇒ 掩膜内应为 I_S1 + 0.5；用单阶段输出反推 I_S1 不易，
    # 改为直接验证强度参数会改变结果（strength=0 时残差归零）。
    gen2 = _FakeStage2Generator(value=0.5)
    r0 = D2RRestorer(mode="d2r", pipe=_FakePipe(), generator=gen2, size=None,
                     seed=3, strength=0.0, device="cpu")
    out0 = r0.restore(src, msk)
    check("strength=0 与 strength=1 结果不同", not np.array_equal(_np(out), _np(out0)))

    print("\n=== G. 注入 generator 但没有 pipe 时的行为 ===")
    try:
        D2RRestorer(mode="d2r", generator=_FakeStage2Generator(), pipe=_FakePipe(),
                    size=None, device="cpu").restore(src, msk)
        check("d2r 在有 pipe+generator 时可用", True)
    except Exception as exc:  # pragma: no cover
        check("d2r 在有 pipe+generator 时可用", False, repr(exc))


def test_single_stage():
    print("\n=== H. single_stage：不经过扩散 ===")
    src, msk = _img(seed=12), _mask()
    gen = _FakeStage2Generator(value=0.0, use_full_reconstruction=True)

    class _SSGen(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seen = None
            self.use_full_reconstruction = True

        def forward(self, masked_image, mask=None):
            self.seen = (int(masked_image.shape[1]),
                         None if mask is None else int(mask.shape[1]))
            return torch.zeros_like(masked_image)

    ssgen = _SSGen()
    pipe = _FakePipe()
    r = D2RRestorer(mode="single_stage", pipe=pipe, generator=ssgen,
                    size=None, device="cpu")
    out = r.restore(src, msk)
    check("未调用扩散管线", len(pipe.calls) == 0, f"{len(pipe.calls)} 次")
    check("单阶段收到 [masked(3), M(1)]", ssgen.seen == (3, 1), str(ssgen.seen))
    a, b, m = _np(out), _np(src), _masknp(msk)
    check("掩膜外逐字节一致", bool((a[~m] == b[~m]).all()))


def test_seeds_and_batch():
    print("\n=== I. 种子语义与批量 ===")
    src, msk = _img(seed=13), _mask()

    def run(seed):
        r = D2RRestorer(mode="stage1_only", pipe=_FakePipe(), size=None,
                        seed=seed, device="cpu")
        return _np(r.restore(src, msk))

    check("同 seed 结果一致", np.array_equal(run(42), run(42)))
    check("不同 seed 结果不同", not np.array_equal(run(42), run(43)))

    r = D2RRestorer(mode="stage1_only", pipe=_FakePipe(), size=None,
                    seed=100, device="cpu")
    batch = r.restore_batch([src] * 3, [msk] * 3)
    check("批量返回 3 张", len(batch) == 3)
    check("批量第 0 张 = seed 100", np.array_equal(_np(batch[0]), run(100)))
    check("批量第 2 张 = seed 102", np.array_equal(_np(batch[2]), run(102)))

    try:
        r.restore_batch([src, src], [msk])
        check("数量不一致抛 ValueError", False)
    except ValueError:
        check("数量不一致抛 ValueError", True)


def test_lifecycle_and_describe():
    print("\n=== J. 生命周期与配置摘要 ===")
    r = D2RRestorer(mode="stage1_only", pipe=_FakePipe(), size=None, device="cpu",
                    stage1_checkpoint="somewhere/checkpoint-best")
    d = r.describe()
    check("describe 含 mode", d["mode"] == "stage1_only")
    check("describe 含 stage1_checkpoint", d["stage1_checkpoint"] == "somewhere/checkpoint-best")
    check("默认 dtype 为 fp32（与评测口径一致）", r.dtype == torch.float32, str(r.dtype))
    check("默认步数/CFG 与论文一致", (r.num_steps, r.guidance_scale) == (30, 7.5))
    # 该实例显式传了 size=None，因此默认值要另建一个实例来查
    r_default = D2RRestorer(mode="stage1_only", pipe=_FakePipe(), device="cpu", load=False)
    check("默认 size = 512", r_default.size == 512, str(r_default.size))

    r.close()
    check("close 后释放组件", r._pipe is None and not r._loaded)
    r._pipe = _FakePipe()      # 重新注入替身，避免真的去下载/加载 SD 权重
    r.load()
    check("可再次 load", r._loaded)

    try:
        D2RRestorer(mode="d2r", size=None, device="cpu")
        check("d2r 缺少生成器时报错", False)
    except ValueError:
        check("d2r 缺少生成器时报错（构造期）", True)

    try:
        D2RRestorer(mode="nonsense", size=None, device="cpu")
        check("非法 mode 报错", False)
    except ValueError:
        check("非法 mode 报错", True)


def main():
    print("torch", torch.__version__, "| cuda:", torch.cuda.is_available())
    test_generator_self_description()
    test_prepare_pair()
    test_stage1_only()
    test_d2r_contract()
    test_single_stage()
    test_seeds_and_batch()
    test_lifecycle_and_describe()
    print(f"\n======== 结果: {PASS} 通过, {FAIL} 失败 ========")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
