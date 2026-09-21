#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对照评测：在固定 51 对测试集上比较 D2R / 仅第一阶段 / 单阶段对照 / 未微调基座。

为什么要单独写这个脚本，而不是用现成的 `指标评测.py`：
  * `指标评测.py` 比较的是"两个文件夹里的图像"，而本研究要比较的是
    **同一批测试对上的多个方法**，且必须先按各自协议生成输出；
  * 审稿人 R2-2 要求三个条件使用**完全相同的测试对、掩膜与指标**，
    因此配对只在这里解析一次，之后所有方法共用。

口径与 `metrics/calculator.py` 一致：
  * 掩膜内 PSNR/SSIM（论文主口径）与全图 PSNR/SSIM 同时报告；
  * LPIPS（AlexNet）；集合级 FID/KID 在**洞外合成**后的图上计算，
    使各方法只在掩膜内有差异。

用法:
    python evaluate.py --manifest eval/manifest.csv --out_dir eval_outputs \
        --stage1_checkpoint stage1_results/checkpoint-best \
        --stage2_checkpoint stage2_results/checkpoint-best
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
os.environ.setdefault("WANDB_MODE", "disabled")
# 说明：不要在这里强制 HF_HUB_OFFLINE。公开仓库的用户第一次运行需要能联网下载
# 基座权重；离线环境请自行设置 HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1。

# 基座模型：默认使用 Hugging Face Hub 官方 id。
# 离线 / 内网环境用环境变量 D2R_SD_MODEL 指向本地 snapshot 目录，例如
#   D2R_SD_MODEL=/path/to/models--runwayml--stable-diffusion-inpainting/snapshots/<hash>
DEFAULT_MODEL = os.environ.get("D2R_SD_MODEL", "runwayml/stable-diffusion-inpainting")

# 评测清单：本仓库**不分发**数据集，也不分发作者使用的划分清单。
# 请用 scripts/build_dataset_manifest.py 基于你自己的数据生成，或用 --manifest 指定。
# 格式示例见 eval/manifest_template.csv（纯占位内容，不含任何真实数据）。
DEFAULT_MANIFEST = os.path.join(ROOT, "eval", "manifest.csv")
# 默认输出目录（预测图、对比图与指标报告）
EVAL_OUT = os.path.join(ROOT, "eval_outputs")


def load_pairs(manifest_path):
    """读取 (image_path, mask_path, sample_id)；路径相对 manifest 所在目录解析。"""
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(
            f"未找到评测清单: {manifest_path}\n"
            "本仓库不分发数据集与划分清单。请先自行生成，例如：\n"
            "  python scripts/build_dataset_manifest.py --split test \\\n"
            "      --image_dir <你的图像目录> --mask_dir <你的掩膜目录> \\\n"
            "      --output eval/manifest.csv\n"
            "然后用 --manifest 指向它。格式示例见 eval/manifest_template.csv。"
        )
    mdir = os.path.dirname(os.path.abspath(manifest_path))
    pairs = []
    with open(manifest_path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            ip = os.path.normpath(os.path.join(mdir, row["image_path"]))
            mp = os.path.normpath(os.path.join(mdir, row["mask_path"]))
            if not (os.path.isfile(ip) and os.path.isfile(mp)):
                raise FileNotFoundError(f"manifest 文件不存在: {ip} / {mp}")
            pairs.append({"sample_id": row["sample_id"], "image": ip, "mask": mp})
    return pairs


def to_pil_pair(image_path, mask_path, size=512):
    img = Image.open(image_path).convert("RGB").resize((size, size), Image.LANCZOS)
    msk = Image.open(mask_path).convert("L").resize((size, size), Image.NEAREST)
    return img, msk


def cache_method(outs, pairs, name, out_dir):
    """把某方法的预测落盘，便于中断后续跑时复用（推理是确定性的：seed 固定）。"""
    d = os.path.join(out_dir, "preds", name)
    os.makedirs(d, exist_ok=True)
    for p in pairs:
        outs[p["sample_id"]].save(os.path.join(d, f"{p['sample_id']}.png"))


def load_cached_method(pairs, name, out_dir):
    """若该方法已有完整缓存的预测，直接读回，跳过重新推理。"""
    d = os.path.join(out_dir, "preds", name)
    need = [os.path.join(d, f"{p['sample_id']}.png") for p in pairs]
    if not all(os.path.isfile(f) for f in need):
        return None
    print(f"  复用已缓存的 {name} 预测（{len(need)} 张）")
    return {p["sample_id"]: Image.open(f).convert("RGB")
            for p, f in zip(pairs, need)}


def composite_outside(gt_pil, pred_pil, mask_pil):
    """洞外取 GT、洞内取预测——集合级指标必须这样合成，各方法才只在洞内有差异。"""
    gt = np.asarray(gt_pil.convert("RGB"), dtype=np.uint8)
    pr = np.asarray(pred_pil.convert("RGB"), dtype=np.uint8)
    m = np.asarray(mask_pil.convert("L"), dtype=np.uint8) > 127
    return Image.fromarray(np.where(m[..., None], pr, gt).astype(np.uint8))


def summarize(values):
    a = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if a.size == 0:
        return None
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "sem": float(a.std(ddof=1) / np.sqrt(a.size)) if a.size > 1 else 0.0,
        "min": float(a.min()),
        "max": float(a.max()),
    }


# --------------------------------------------------------------------------- #
# 各方法的推理
# --------------------------------------------------------------------------- #
def run_stage1_only(pipe, pairs, prompt, steps, guidance, seed_base):
    """仅第一阶段：LoRA 适配的扩散补全，不做 GAN 细化。"""
    outs = {}
    for i, p in enumerate(pairs):
        img, msk = to_pil_pair(p["image"], p["mask"])
        torch.manual_seed(seed_base + i)
        gen = torch.Generator(device="cpu").manual_seed(seed_base + i)
        res = pipe(prompt=prompt, image=img, mask_image=msk,
                   num_inference_steps=steps, guidance_scale=guidance,
                   generator=gen).images[0].convert("RGB")
        outs[p["sample_id"]] = res
    return outs


def main():
    ap = argparse.ArgumentParser(description="D2R / 仅第一阶段 / 单阶段对照 的统一评测")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--out_dir", default=EVAL_OUT)
    ap.add_argument("--base_model", default=DEFAULT_MODEL)
    ap.add_argument("--stage1_checkpoint", default=None)
    ap.add_argument("--stage2_checkpoint", default=None)
    ap.add_argument("--single_stage_checkpoint", default=None)
    ap.add_argument("--prompt", default="")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help=">0 时只跑前 N 对（调试用）")
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    pairs = load_pairs(args.manifest)
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"测试对: {len(pairs)}（manifest={args.manifest}）")

    gts, msks, comps = {}, {}, {}
    for p in pairs:
        g, m = to_pil_pair(p["image"], p["mask"])
        gts[p["sample_id"]] = g
        msks[p["sample_id"]] = m

    results = {}

    # 每个方法跑完立刻落盘；若已存在完整缓存则直接复用（推理确定性，seed 固定），
    # 这样中断后重跑不必从头再来。
    def _get_or_run(name, run_fn):
        cached = load_cached_method(pairs, name, args.out_dir)
        if cached is not None:
            return cached, 0.0
        t0 = time.time()
        outs = run_fn()
        el = time.time() - t0
        cache_method(outs, pairs, name, args.out_dir)
        print(f"  完成，用时 {el:.0f}s（已落盘到 preds/{name}）")
        return outs, el

    # ---------------- 未微调基座（下界参考，非论文条件） ----------------
    # 注意：--base_model 既可以是本地目录，也可以是 Hugging Face Hub id，
    # 因此这里不能再用 os.path.isdir 判断，否则默认的 hub id 会被静默跳过。
    if args.base_model:
        from diffusers import StableDiffusionInpaintPipeline
        print("\n[未微调基座] 加载 SD-Inpainting ...")

        def _run_base():
            pipe = StableDiffusionInpaintPipeline.from_pretrained(
                args.base_model, torch_dtype=torch.float32,
                safety_checker=None, requires_safety_checker=False,
            ).to("cuda" if torch.cuda.is_available() else "cpu")
            try:
                return run_stage1_only(pipe, pairs, args.prompt, args.steps,
                                       args.guidance, args.seed)
            finally:
                del pipe
                torch.cuda.empty_cache()

        results["base_sd_inpainting"], _ = _get_or_run("base_sd_inpainting", _run_base)

    # ---------------- 仅第一阶段（LoRA） ----------------
    if args.stage1_checkpoint and os.path.isdir(args.stage1_checkpoint):
        from diffusers import StableDiffusionInpaintPipeline
        from peft import PeftModel
        print(f"\n[仅第一阶段] 加载 LoRA: {args.stage1_checkpoint}")

        def _run_s1():
            pipe = StableDiffusionInpaintPipeline.from_pretrained(
                args.base_model, torch_dtype=torch.float32,
                safety_checker=None, requires_safety_checker=False,
            ).to("cuda" if torch.cuda.is_available() else "cpu")
            try:
                pipe.unet = PeftModel.from_pretrained(
                    pipe.unet, args.stage1_checkpoint, torch_dtype=torch.float32).to(pipe.device)
                return run_stage1_only(pipe, pairs, args.prompt, args.steps,
                                       args.guidance, args.seed)
            finally:
                del pipe
                torch.cuda.empty_cache()

        results["stage1_only"], _ = _get_or_run("stage1_only", _run_s1)

    # ---------------- D2R = 第一阶段 + 第二阶段细化 ----------------
    if (args.stage1_checkpoint and args.stage2_checkpoint
            and os.path.isdir(args.stage2_checkpoint)):
        from diffusers import StableDiffusionInpaintPipeline
        from peft import PeftModel
        from inference.pipeline import load_stage2_generator, refine_with_stage2
        print(f"\n[D2R] LoRA + Stage-2: {args.stage2_checkpoint}")

        def _run_d2r():
            pipe = StableDiffusionInpaintPipeline.from_pretrained(
                args.base_model, torch_dtype=torch.float32,
                safety_checker=None, requires_safety_checker=False,
            ).to("cuda" if torch.cuda.is_available() else "cpu")
            generator = None
            try:
                pipe.unet = PeftModel.from_pretrained(
                    pipe.unet, args.stage1_checkpoint, torch_dtype=torch.float32).to(pipe.device)
                generator = load_stage2_generator(args.stage2_checkpoint, pipe.device)
                s1_outs = run_stage1_only(pipe, pairs, args.prompt, args.steps,
                                          args.guidance, args.seed)
                out = {}
                for p in pairs:
                    img, msk = to_pil_pair(p["image"], p["mask"])
                    out[p["sample_id"]] = refine_with_stage2(
                        generator, s1_outs[p["sample_id"]], img, msk, device=str(pipe.device))
                return out
            finally:
                del pipe, generator
                torch.cuda.empty_cache()

        results["d2r"], _ = _get_or_run("d2r", _run_d2r)

    # ---------------- 单阶段对照（R2-2） ----------------
    if args.single_stage_checkpoint:
        from inference.pipeline import load_single_stage_generator, single_stage_inference
        print(f"\n[单阶段对照] {args.single_stage_checkpoint}")

        def _run_ss():
            device = "cuda" if torch.cuda.is_available() else "cpu"
            generator = load_single_stage_generator(args.single_stage_checkpoint, device)
            try:
                out = {}
                for p in pairs:
                    img, msk = to_pil_pair(p["image"], p["mask"])
                    out[p["sample_id"]] = single_stage_inference(
                        generator, img, msk, device=device)
                return out
            finally:
                del generator
                torch.cuda.empty_cache()

        results["single_stage"], _ = _get_or_run("single_stage", _run_ss)

    if not results:
        raise SystemExit("没有可评测的方法：请至少给出一个存在的检查点路径。")

    # ---------------- 指标 ----------------
    from metrics import MetricCalculator
    mc = MetricCalculator(device="cuda" if torch.cuda.is_available() else "cpu")

    per_method = {}
    out_dir = os.path.join(args.out_dir, "preds")
    os.makedirs(out_dir, exist_ok=True)
    for name, outs in results.items():
        rows = []
        comp_list, gt_list = [], []
        # 预测已经由 cache_method 落在 preds/<name>/，这里不再重复保存
        for p in pairs:
            sid = p["sample_id"]
            gt, msk, pred = gts[sid], msks[sid], outs[sid]
            d = mc.calculate_all_single(gt, pred, mask=msk)
            comp = composite_outside(gt, pred, msk)
            comps[(name, sid)] = comp
            comp_list.append(comp)
            gt_list.append(gt)
            rows.append({"sample_id": sid, **{k: v for k, v in d.items() if v is not None}})

        keys = [k for k in rows[0].keys() if k != "sample_id"]
        entry = {k: summarize([r.get(k) for r in rows]) for k in keys}
        if len(comp_list) >= 2:
            entry["FID_composited"] = float(mc.calculate_fid(gt_list, comp_list))
            entry["KID_composited"] = float(mc.calculate_kid(gt_list, comp_list))
        per_method[name] = entry
        print(f"\n=== {name} ===")
        for k in ("psnr", "ssim", "lpips", "psnr_mask", "ssim_mask", "lpips_mask"):
            s = entry.get(k)
            if s:
                print(f"  {k:<12} {s['mean']:8.4f} ± {s['sem']:.4f} (n={s['n']})")
        for k in ("FID_composited", "KID_composited"):
            if k in entry:
                print(f"  {k:<12} {entry[k]:8.4f}")

    report = {
        "manifest": args.manifest,
        "n_pairs": len(pairs),
        "config": {"steps": args.steps, "guidance": args.guidance, "seed": args.seed,
                   "prompt": args.prompt},
        "checkpoints": {
            "stage1_checkpoint": args.stage1_checkpoint,
            "stage2_checkpoint": args.stage2_checkpoint,
            "single_stage_checkpoint": args.single_stage_checkpoint,
        },
        "metrics": per_method,
    }
    rp = args.report or os.path.join(args.out_dir, "report.json")
    with open(rp, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print(f"\n报告: {rp}")


if __name__ == "__main__":
    main()
