#!/usr/bin/env python3
"""生成论文 Table 4（Adaptation and single-stage controls）可直接粘贴的记录。

审稿意见 R2-2 要求两个对照都必须显式给出：
  架构与训练定义、可训练参数、更新步数与 GPU 时间。

这些量全部来自训练时写出的产物：
  <run_dir>/budget_report.json      训练器自动写出的权威记录（新运行）
  <run_dir>/run_config.json         启动参数
  <run_dir>/checkpoint-*/training_state.pt   旧运行的回退来源（无计时）

用法:
  python scripts/control_budget_report.py \\
      --stage1_dir stage1_results_seed2026 \\
      --stage2_dir stage2_results_seed2026 \\
      --single_stage_dir single_stage_results_seed2026 \\
      --d2r_metrics evaluation_d2r/batch_metrics.json \\
      --stage1_only_metrics evaluation_stage1/batch_metrics.json \\
      --single_stage_metrics evaluation_single/batch_metrics.json \\
      --out TABLE4_matched_controls.md

注意：脚本**不会**编造任何数字。缺失的量会写成 `[[...]]` 占位符并列出原因，
避免把"未记录"误填成"已测量"。
"""

import argparse
import glob
import json
import os
import sys


# ═══════════════════════════════════════════
# 读取工具
# ═══════════════════════════════════════════

def _read_json(path):
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _latest_checkpoint_state(run_dir):
    """返回 (epoch, state_dict, source_path)，取 epoch 最大的 checkpoint。"""
    best = None
    for ckpt in glob.glob(os.path.join(run_dir, "checkpoint-*")):
        state_path = os.path.join(ckpt, "training_state.pt")
        if not os.path.isfile(state_path):
            continue
        try:
            import torch
            state = torch.load(state_path, map_location="cpu")
        except Exception:
            continue
        epoch = int(state.get("epoch", -1))
        if best is None or epoch > best[0]:
            best = (epoch, state, state_path)
    return best


def _steps_per_epoch_from_run_config(run_config):
    """从 Stage-1 的 run_config 反推每 epoch 的优化步数（max_train_steps / epochs）。"""
    if not run_config:
        return None
    trainer_config = run_config.get("trainer_config") or {}
    max_steps = trainer_config.get("max_train_steps")
    args = run_config.get("arguments") or {}
    epochs = args.get("stage1_epochs")
    if max_steps and epochs:
        return int(round(int(max_steps) / int(epochs)))
    return None


def collect_run(run_dir, steps_per_epoch=None, role=""):
    """汇总一次运行的预算记录。缺失项显式标记，不做推断式补全。"""
    record = {
        "role": role,
        "dir": os.path.abspath(run_dir) if run_dir else None,
        "exists": bool(run_dir) and os.path.isdir(run_dir),
        "model_definition": None,
        "pretraining": None,
        "losses": None,
        "trainable_parameters": None,
        "total_parameters": None,
        "updates": None,
        "gpu_hours": None,
        "wall_clock_hours": None,
        "world_size": None,
        "epochs": None,
        "seed": None,
        "sources": [],
        "missing": [],
    }
    if not record["exists"]:
        record["missing"].append("输出目录不存在")
        return record

    budget = _read_json(os.path.join(run_dir, "budget_report.json"))
    run_config = _read_json(os.path.join(run_dir, "run_config.json"))
    if run_config:
        record["sources"].append("run_config.json")
        record["seed"] = (run_config.get("arguments") or {}).get("seed")
        record["world_size"] = run_config.get("world_size")

    if budget:
        record["sources"].append("budget_report.json")
        record["model_definition"] = budget.get("model_definition")
        record["pretraining"] = budget.get("pretraining")
        record["losses"] = budget.get("losses")
        params = budget.get("parameters") or {}
        # 单阶段：{generator, discriminator}；两阶段：{stage1_unet_lora, generator, discriminator}
        trainable = sum(v.get("trainable", 0) for k, v in params.items() if isinstance(v, dict))
        total = sum(v.get("total", 0) for k, v in params.items() if isinstance(v, dict))
        if trainable:
            record["trainable_parameters"] = int(trainable)
            record["total_parameters"] = int(total)
        record["updates"] = budget.get("completed_updates")
        record["epochs"] = budget.get("completed_epochs")
        snapshot = budget.get("budget") or {}
        record["gpu_hours"] = snapshot.get("gpu_hours")
        record["wall_clock_hours"] = snapshot.get("wall_clock_hours")
    else:
        record["missing"].append("budget_report.json（该运行早于计时/计数功能）")

    latest = _latest_checkpoint_state(run_dir)
    if latest is not None:
        epoch, state, state_path = latest
        record["sources"].append(os.path.relpath(state_path, run_dir))
        if record["updates"] is None:
            global_step = int(state.get("global_step", 0) or 0)
            if global_step > 0:
                record["updates"] = global_step
            elif steps_per_epoch:
                record["updates"] = (epoch + 1) * steps_per_epoch
                record["missing"].append(
                    "更新步数由 checkpoint epoch × steps/epoch 估算（旧检查点未记录 global_step）"
                )
        if record["epochs"] is None:
            record["epochs"] = epoch + 1
        if record["gpu_hours"] is None:
            seconds = state.get("train_seconds")
            world_size = record["world_size"] or 1
            if seconds:
                record["wall_clock_hours"] = round(float(seconds) / 3600.0, 4)
                record["gpu_hours"] = round(float(seconds) * world_size / 3600.0, 4)
            else:
                record["missing"].append("GPU·h（该检查点未记录 train_seconds，需重跑或人工说明）")

    if record["model_definition"] is None:
        record["missing"].append("模型定义字符串（仅 budget_report.json 提供）")
    return record


def read_metrics(path):
    """从 infer.py 的 batch_metrics.json 读取掩膜区域三项指标。"""
    data = _read_json(path)
    if not data:
        return {"psnr": None, "ssim": None, "lpips": None, "path": path}
    mask_region = data.get("mask_region") or {}
    return {
        "psnr": mask_region.get("avg_psnr"),
        "ssim": mask_region.get("avg_ssim"),
        "lpips": mask_region.get("avg_lpips"),
        "num_images": data.get("num_images"),
        "path": path,
    }


def fmt(value, digits=4, placeholder="[[ MISSING ]]"):
    if value is None:
        return placeholder
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def fmt_params(value):
    return "[[ MISSING ]]" if value is None else f"{int(value):,}"


# ═══════════════════════════════════════════
# 报告
# ═══════════════════════════════════════════

def build_markdown(stage1, stage2, single, metrics):
    d2r_updates = None
    if stage1["updates"] is not None and stage2["updates"] is not None:
        d2r_updates = int(stage1["updates"] + stage2["updates"])
    d2r_params = None
    if stage1["trainable_parameters"] is not None and stage2["trainable_parameters"] is not None:
        d2r_params = int(stage1["trainable_parameters"] + stage2["trainable_parameters"])
    d2r_gpu_hours = None
    if stage1["gpu_hours"] is not None and stage2["gpu_hours"] is not None:
        d2r_gpu_hours = round(stage1["gpu_hours"] + stage2["gpu_hours"], 4)

    d2r_def = "Stage 1 ({}) then Stage 2 ({})".format(
        stage1["model_definition"] or "[[ STAGE1_MODEL_DEFINITION ]]",
        stage2["model_definition"] or "[[ STAGE2_MODEL_DEFINITION ]]",
    )

    lines = []
    lines.append("# Table 4 data — adaptation and single-stage controls\n")
    lines.append("由 `scripts/control_budget_report.py` 自动生成；所有数值直接来自训练产物，"
                 "缺失项保留 `[[ MISSING ]]` 占位符，不做推断或编造。\n")

    lines.append("## Table 4 (budget rows)\n")
    lines.append("> \"Trainable parameters\" 一列按**该配置下参与训练的全部网络**求和"
                 "（D2R = Stage-1 LoRA + Stage-2 生成器 + 判别器；单阶段 = 生成器 + 判别器），"
                 "分项数值见 `budget_report.json` 的 `parameters` 字段，可按需拆分叙述。\n")
    lines.append("| Method | Actual model and training definition | Trainable parameters | Update count | GPU hours |")
    lines.append("|---|---|---|---|---|")
    lines.append(
        "| Stage-1-only SD-Inpainting-LoRA | Same Stage-1 checkpoint, prompt and inference settings as the "
        "corresponding D2R run (no additional training) | {} (already counted in D2R Stage 1) | "
        "0 additional updates | 0 additional |".format(fmt_params(stage1["trainable_parameters"]))
    )
    lines.append(
        "| Single-stage end-to-end control | {} | {} | {} | {} |".format(
            single["model_definition"] or "[[ SINGLE_STAGE_MODEL_AND_TRAINING ]]",
            fmt_params(single["trainable_parameters"]),
            fmt(single["updates"], 0),
            fmt(single["gpu_hours"], 2),
        )
    )
    lines.append(
        "| D2R | {} | {} | {} | {} |".format(
            d2r_def, fmt_params(d2r_params), fmt(d2r_updates, 0), fmt(d2r_gpu_hours, 2),
        )
    )
    lines.append("")

    lines.append("## Table 4 (image-quality rows)\n")
    lines.append("| Method | PSNR_M ↑ | SSIM_M ↑ | LPIPS_M ↓ | Metrics source |")
    lines.append("|---|---|---|---|---|")
    for label, key in (("Stage-1-only SD-Inpainting-LoRA", "stage1_only"),
                       ("Single-stage end-to-end control", "single_stage"),
                       ("D2R", "d2r")):
        m = metrics.get(key) or {}
        lines.append("| {} | {} | {} | {} | {} |".format(
            label, fmt(m.get("psnr"), 4), fmt(m.get("ssim"), 4), fmt(m.get("lpips"), 4),
            m.get("path") or "[[ provide --*_metrics ]]",
        ))
    lines.append("")

    lines.append("## 预算匹配核对\n")
    lines.append(f"- D2R 两阶段实际更新步数合计：{fmt(d2r_updates, 0)}"
                 f"（Stage-1 {fmt(stage1['updates'], 0)} + Stage-2 {fmt(stage2['updates'], 0)}）")
    lines.append(f"- 单阶段对照实际更新步数：{fmt(single['updates'], 0)}")
    if d2r_updates and single["updates"]:
        delta = int(single["updates"]) - int(d2r_updates)
        verdict = "匹配" if abs(delta) <= max(1, int(0.02 * d2r_updates)) else "不匹配"
        lines.append(f"- 差额：{delta:+d} 步（判定：{verdict}）")
    lines.append(f"- D2R GPU·h 合计：{fmt(d2r_gpu_hours, 2)}；单阶段 GPU·h：{fmt(single['gpu_hours'], 2)}")
    lines.append("")

    all_missing = []
    for record in (stage1, stage2, single):
        for item in record["missing"]:
            all_missing.append(f"- {record['role']}: {item}")
    lines.append("## 缺失项（必须补齐后才能定稿，不要留占位符进投稿版）\n")
    lines.append("\n".join(all_missing) if all_missing else "- 无")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="生成 Table 4 的对照记录")
    parser.add_argument("--stage1_dir", default="./stage1_results")
    parser.add_argument("--stage2_dir", default="./stage2_results")
    parser.add_argument("--single_stage_dir", default="./single_stage_results")
    parser.add_argument("--steps_per_epoch", type=int, default=None,
                        help="旧检查点缺少 global_step 时用于估算；缺省时从 Stage-1 的 run_config 反推")
    parser.add_argument("--d2r_metrics", default=None, help="D2R 的 batch_metrics.json")
    parser.add_argument("--stage1_only_metrics", default=None, help="Stage-1-only 的 batch_metrics.json")
    parser.add_argument("--single_stage_metrics", default=None, help="单阶段对照的 batch_metrics.json")
    parser.add_argument("--out", default="TABLE4_matched_controls.md")
    parser.add_argument("--json_out", default=None, help="可选：同时写出机器可读 JSON")
    args = parser.parse_args()

    steps_per_epoch = args.steps_per_epoch
    if steps_per_epoch is None:
        steps_per_epoch = _steps_per_epoch_from_run_config(
            _read_json(os.path.join(args.stage1_dir, "run_config.json"))
        )

    stage1 = collect_run(args.stage1_dir, steps_per_epoch, role="D2R Stage 1")
    stage2 = collect_run(args.stage2_dir, steps_per_epoch, role="D2R Stage 2")
    single = collect_run(args.single_stage_dir, steps_per_epoch, role="Single-stage control")

    metrics = {
        "d2r": read_metrics(args.d2r_metrics),
        "stage1_only": read_metrics(args.stage1_only_metrics),
        "single_stage": read_metrics(args.single_stage_metrics),
    }

    markdown = build_markdown(stage1, stage2, single, metrics)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(markdown)
    print(markdown)
    print(f"\n已写出: {os.path.abspath(args.out)}")

    if args.json_out:
        payload = {
            "stage1": stage1, "stage2": stage2, "single_stage": single, "metrics": metrics,
            "steps_per_epoch_used": steps_per_epoch,
        }
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"已写出: {os.path.abspath(args.json_out)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
