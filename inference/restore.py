#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""高层推理接口：加载一次模型，反复调用修复。

为什么单独一个模块
------------------
`inference/pipeline.py` 提供的是"零件"（加载管线、跑一次 Stage-1、跑一次 Stage-2
细化），`infer.py` 提供的是命令行流程。两者都不便在没有命令行、需要反复调用的场合
使用：前者每次都要自己拼装参数，后者每张图都要重新加载模型。

本模块补上这一层，给两个入口：

* :class:`D2RRestorer` —— 加载一次、可重复调用的对象，支持批量、上下文管理器，
  也支持注入已加载好的 ``pipe`` / ``generator``（便于复用与测试）；
* :func:`restore_image` —— 一次性函数，加载 → 修复一张 → 释放。

最小用法
--------
>>> from inference import D2RRestorer
>>> with D2RRestorer(stage1_checkpoint="stage1_results/checkpoint-best",
...                  stage2_checkpoint="stage2_results/checkpoint-best") as r:
...     out = r.restore("photo.jpg", "mask.png")
...     out.save("restored.png")

只跑第一阶段（不加载 Stage-2 生成器）：

>>> with D2RRestorer(stage1_checkpoint="stage1_results/checkpoint-best") as r:
...     out = r.restore(photo_pil, mask_pil)          # 也接受 PIL.Image

四类模式
--------
====== ============================================= ==========================
mode   何时自动选中                                  需要什么
====== ============================================= ==========================
d2r    ``stage2_checkpoint`` 已给出                   SD 基座 + LoRA + Stage-2 生成器
stage1_only 只给出 ``stage1_checkpoint``              SD 基座 + LoRA
base   两者都没给                                     SD 基座（未微调，下界参考）
single_stage ``single_stage_checkpoint`` 已给出        仅单阶段生成器，不加载 SD
====== ============================================= ==========================

被本模块保证的协议不变量（详见 ``docs/PROTOCOL.md``）
----------------------------------------------------
1. 掩膜外像素逐字节等于**传入的图像**（在 uint8 空间二次合成，不做 float 往返）；
2. Stage-2 生成器只接收 ``[I_S1, M]`` 四通道，真值不进入任何一路；
3. 给定 ``seed`` 时结果可复现；批处理使用 ``seed + i``，与 ``evaluate.py`` 一致。
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple, Union

import torch
from PIL import Image

from .pipeline import (
    load_model,
    load_single_stage_generator,
    refine_with_stage2,
    single_stage_inference,
    stage1_inference,
)

__all__ = ["D2RRestorer", "restore_image", "MODES", "DEFAULT_PROMPT"]

MODES = ("base", "stage1_only", "d2r", "single_stage")

#: 论文评测协议使用空 prompt（实测训练所用的那段提示词反而略差）。
DEFAULT_PROMPT = ""

#: 训练与评测统一在 512×512 上进行。
DEFAULT_SIZE = 512

_UNSET = object()

ImageLike = Union[str, os.PathLike, Image.Image]


# --------------------------------------------------------------------------- #
# 输入归一化
# --------------------------------------------------------------------------- #
def _as_pil(x: ImageLike, mode: str) -> Image.Image:
    """把路径或 PIL.Image 统一成已加载的 PIL.Image。"""
    if isinstance(x, Image.Image):
        return x.convert(mode)
    if isinstance(x, (str, os.PathLike)):
        path = os.fspath(x)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"找不到图像文件: {path}")
        with Image.open(path) as im:
            # 必须 copy()：with 退出后延迟加载的像素就取不到了
            return im.convert(mode).copy()
    raise TypeError(f"期望文件路径或 PIL.Image，得到 {type(x).__name__}")


def _prepare_pair(image: ImageLike, mask: ImageLike,
                  size: Optional[int]) -> Tuple[Image.Image, Image.Image]:
    img = _as_pil(image, "RGB")
    msk = _as_pil(mask, "L")
    if size:
        target = (int(size), int(size))
        if img.size != target:
            img = img.resize(target, Image.LANCZOS)
        if msk.size != target:
            # 掩膜必须最近邻，否则会插值出中间灰度、改变损伤区域边界
            msk = msk.resize(target, Image.NEAREST)
    return img, msk


def _outside_mask_equal(out: Image.Image, src: Image.Image, mask: Image.Image) -> bool:
    """用于自检与测试：掩膜外是否逐字节一致。"""
    import numpy as np

    a = np.asarray(out.convert("RGB"), dtype=np.uint8)
    b = np.asarray(src.convert("RGB"), dtype=np.uint8)
    m = np.asarray(mask.convert("L"), dtype=np.uint8) >= 128
    return bool((a[~m] == b[~m]).all())


# --------------------------------------------------------------------------- #
# 主类
# --------------------------------------------------------------------------- #
class D2RRestorer:
    """加载一次、可反复调用的修复器。

    Args:
        stage1_checkpoint: Stage-1 输出目录或 ``checkpoint-best``（内含
            ``adapter_config.json`` 的 PEFT 适配器）。给了它就是 LoRA 微调后的模型。
        stage2_checkpoint: Stage-2 检查点目录或 ``generator.pth``。给了它就走完整
            两阶段（``mode="d2r"``）。
        single_stage_checkpoint: 端到端单阶段对照的生成器；给了它则**不加载** Stable
            Diffusion，也不需要 Stage-1/Stage-2。
        base_model: 基座模型。默认取环境变量 ``D2R_SD_MODEL``，否则用 Hugging Face Hub
            上的 ``runwayml/stable-diffusion-inpainting``。
        device: ``"cuda"`` / ``"cpu"`` 等；``None`` 时自动选择。
        prompt / negative_prompt: 文本条件。默认空串，与论文评测协议一致。
        num_steps / guidance_scale: DDIM 步数与 CFG。默认 30 / 7.5，与论文一致。
        seed: 默认随机种子。``None`` 表示不固定（不可复现）。
        strength: Stage-2 残差强度，1.0 表示使用生成器自带的缩放。
        size: 统一缩放到的边长；``None`` 表示不缩放（调用方需自行保证 512×512）。
        dtype: 计算精度。默认 ``torch.float32``，与 ``evaluate.py`` 的评测口径一致；
            传 ``torch.float16`` 可换速度，但数值会有细微差异。
        pipe / generator: 可注入已加载好的组件，跳过加载（便于复用与单元测试）。
        load: 是否在构造时立即加载模型。
    """

    def __init__(
        self,
        *,
        stage1_checkpoint: Optional[str] = None,
        stage2_checkpoint: Optional[str] = None,
        single_stage_checkpoint: Optional[str] = None,
        base_model: Optional[str] = None,
        device: Optional[str] = None,
        prompt: str = DEFAULT_PROMPT,
        negative_prompt: str = "",
        num_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: Optional[int] = 42,
        strength: float = 1.0,
        size: Optional[int] = DEFAULT_SIZE,
        dtype: Optional[torch.dtype] = torch.float32,
        use_cpu_offload: bool = False,
        single_stage_base_channels: int = 64,
        mode: Optional[str] = None,
        pipe=None,
        generator=None,
        load: bool = True,
    ):
        self.base_model = (base_model
                           or os.environ.get("D2R_SD_MODEL")
                           or "runwayml/stable-diffusion-inpainting")
        self.stage1_checkpoint = stage1_checkpoint
        self.stage2_checkpoint = stage2_checkpoint
        self.single_stage_checkpoint = single_stage_checkpoint

        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(dev)
        self.device_str = str(self.device)

        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.num_steps = int(num_steps)
        self.guidance_scale = float(guidance_scale)
        self.seed = seed
        self.strength = float(strength)
        self.size = size
        self.dtype = dtype
        self.use_cpu_offload = bool(use_cpu_offload)
        self.single_stage_base_channels = int(single_stage_base_channels)

        self._pipe = pipe
        self._generator = generator
        self._loaded = False
        self.mode = mode or self._infer_mode()
        if self.mode not in MODES:
            raise ValueError(f"mode 必须是 {MODES} 之一，得到 {self.mode!r}")
        self._validate_prerequisites()

        if load:
            self.load()

    # ---------------------------------------------------------------- #
    # 模式与前置条件
    # ---------------------------------------------------------------- #
    def _infer_mode(self) -> str:
        if self.single_stage_checkpoint:
            return "single_stage"
        if self.stage2_checkpoint or self._generator is not None:
            return "d2r"
        if self.stage1_checkpoint:
            return "stage1_only"
        return "base"

    def _validate_prerequisites(self) -> None:
        if self.mode == "single_stage":
            if not (self.single_stage_checkpoint or self._generator is not None):
                raise ValueError("mode='single_stage' 需要 single_stage_checkpoint 或注入 generator")
            return
        if self.mode == "d2r" and not (self.stage2_checkpoint or self._generator is not None):
            raise ValueError("mode='d2r' 需要 stage2_checkpoint 或注入 generator")

    @property
    def uses_stage2(self) -> bool:
        return self.mode == "d2r"

    def __repr__(self) -> str:
        return (f"D2RRestorer(mode={self.mode!r}, device={self.device_str!r}, "
                f"loaded={self._loaded}, steps={self.num_steps}, cfg={self.guidance_scale})")

    # ---------------------------------------------------------------- #
    # 加载 / 释放
    # ---------------------------------------------------------------- #
    def load(self) -> "D2RRestorer":
        """加载所需模型。重复调用是安全的（幂等）。"""
        if self._loaded:
            return self

        if self.mode == "single_stage":
            if self._generator is None:
                self._generator = load_single_stage_generator(
                    self.single_stage_checkpoint, self.device_str,
                    base_channels=self.single_stage_base_channels)
        else:
            if self._pipe is None:
                need_stage2 = self.stage2_checkpoint if self.uses_stage2 else None
                pipe, gen = load_model(
                    self.base_model,
                    stage1_checkpoint=self.stage1_checkpoint,
                    stage2_checkpoint=need_stage2,
                    device=self.device_str,
                    use_cpu_offload=self.use_cpu_offload,
                    dtype=self.dtype,
                )
                self._pipe = pipe
                if gen is not None:
                    self._generator = gen
            if self.uses_stage2 and self._generator is None:
                raise RuntimeError(
                    "mode='d2r' 需要 Stage-2 生成器：请给出 stage2_checkpoint，"
                    "或把已加载的生成器通过 generator= 注入。"
                )

        self._loaded = True
        return self

    def close(self) -> None:
        """释放模型并清空显存缓存。之后可再次调用 :meth:`load`。"""
        self._pipe = None
        self._generator = None
        self._loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self) -> "D2RRestorer":
        return self.load()

    def __exit__(self, *exc_info) -> bool:
        self.close()
        return False

    # ---------------------------------------------------------------- #
    # 推理
    # ---------------------------------------------------------------- #
    def restore(
        self,
        image: ImageLike,
        mask: ImageLike,
        *,
        seed=_UNSET,
        num_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        strength: Optional[float] = None,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        size=_UNSET,
    ) -> Image.Image:
        """修复单张图像，返回 PIL.Image。

        Args:
            image: 待修复图像；文件路径或 ``PIL.Image``。
            mask: 损伤掩膜（L 模式）；白色（>127）表示待修复区域。
            seed: 覆盖默认种子；显式传 ``None`` 表示不固定种子。
            num_steps / guidance_scale / strength / prompt / negative_prompt / size:
                单次覆盖构造时的取值。

        Returns:
            修复后的 ``PIL.Image``（RGB），尺寸等于 ``size``（或输入尺寸，当
            ``size=None``）。掩膜外像素与传入图像逐字节一致。
        """
        self.load()

        use_size = self.size if size is _UNSET else size
        img, msk = _prepare_pair(image, mask, use_size)

        use_seed = self.seed if seed is _UNSET else seed
        steps = self.num_steps if num_steps is None else int(num_steps)
        cfg = self.guidance_scale if guidance_scale is None else float(guidance_scale)
        stg = self.strength if strength is None else float(strength)
        pr = self.prompt if prompt is None else prompt
        npr = self.negative_prompt if negative_prompt is None else negative_prompt

        # 单阶段对照：不经过扩散，也不需要文本条件
        if self.mode == "single_stage":
            return single_stage_inference(self._generator, img, msk, device=self.device_str)

        stage1_out = stage1_inference(
            self._pipe, img, msk,
            prompt=pr, negative_prompt=npr,
            num_steps=steps, cfg_scale=cfg, seed=use_seed,
        )

        if not self.uses_stage2:
            # base / stage1_only：Stage-1 的输出即可交付
            return stage1_out

        # 注意 original_image 只用于掩膜外的合成；掩膜内的真值不会进入生成器
        return refine_with_stage2(
            self._generator, stage1_out, img, msk,
            device=self.device_str, strength=stg,
        )

    def restore_batch(
        self,
        images: Sequence[ImageLike],
        masks: Sequence[ImageLike],
        *,
        seed: Optional[int] = None,
        **kwargs,
    ) -> List[Image.Image]:
        """批量修复。

        与 ``evaluate.py`` 相同的种子约定：第 ``i`` 张用 ``seed + i``，因此在给定
        基础种子时整个批次可复现。``seed=None`` 时使用构造时的 ``self.seed``。
        """
        if len(images) != len(masks):
            raise ValueError(
                f"images 与 masks 数量不一致：{len(images)} vs {len(masks)}")
        base = self.seed if seed is None else seed
        return [
            self.restore(im, mk, seed=(None if base is None else base + i), **kwargs)
            for i, (im, mk) in enumerate(zip(images, masks))
        ]

    def describe(self) -> dict:
        """返回本次推理的配置摘要，便于写进实验记录。"""
        return {
            "mode": self.mode,
            "device": self.device_str,
            "dtype": str(self.dtype),
            "base_model": self.base_model,
            "stage1_checkpoint": self.stage1_checkpoint,
            "stage2_checkpoint": self.stage2_checkpoint,
            "single_stage_checkpoint": self.single_stage_checkpoint,
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "num_steps": self.num_steps,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "strength": self.strength,
            "size": self.size,
        }


# --------------------------------------------------------------------------- #
# 一次性便捷函数
# --------------------------------------------------------------------------- #
def restore_image(
    image: ImageLike,
    mask: ImageLike,
    *,
    stage1_checkpoint: Optional[str] = None,
    stage2_checkpoint: Optional[str] = None,
    single_stage_checkpoint: Optional[str] = None,
    base_model: Optional[str] = None,
    device: Optional[str] = None,
    prompt: str = DEFAULT_PROMPT,
    negative_prompt: str = "",
    num_steps: int = 30,
    guidance_scale: float = 7.5,
    seed: Optional[int] = 42,
    strength: float = 1.0,
    size: Optional[int] = DEFAULT_SIZE,
    dtype: Optional[torch.dtype] = torch.float32,
) -> Image.Image:
    """加载 → 修复一张 → 释放。

    适合脚本里偶尔用一次。要连续处理多张，请用 :class:`D2RRestorer`，否则每张图都会
    重新加载模型。
    """
    with D2RRestorer(
        stage1_checkpoint=stage1_checkpoint,
        stage2_checkpoint=stage2_checkpoint,
        single_stage_checkpoint=single_stage_checkpoint,
        base_model=base_model,
        device=device,
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_steps=num_steps,
        guidance_scale=guidance_scale,
        seed=seed,
        strength=strength,
        size=size,
        dtype=dtype,
    ) as restorer:
        return restorer.restore(image, mask)
