#!/usr/bin/env python3
"""Check whether the server is ready for the D2R training environment.

This script performs diagnostics only: it does not load the dataset, train a
model, write checkpoints, or generate images.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


# 基座模型：默认 Hugging Face Hub id；离线环境用 D2R_SD_MODEL 指向本地 snapshot 目录
DEFAULT_MODEL = os.environ.get("D2R_SD_MODEL", "runwayml/stable-diffusion-inpainting")


def report(label, ok, detail=""):
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {label}: {detail}")
    return ok


def check_nvidia_smi():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15,
        )
        output = result.stdout.strip() or result.stderr.strip()
        return report("nvidia-smi", result.returncode == 0, output)
    except FileNotFoundError:
        return report("nvidia-smi", False, "命令不存在；请安装/加载 NVIDIA 驱动")
    except Exception as exc:
        return report("nvidia-smi", False, repr(exc))


def check_torch_cuda():
    try:
        import torch
    except Exception as exc:
        report("PyTorch import", False, repr(exc))
        return False, None

    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}; compiled CUDA: {torch.version.cuda}")
    try:
        available = bool(torch.cuda.is_available())
        count = int(torch.cuda.device_count())
    except Exception as exc:
        report("PyTorch CUDA query", False, repr(exc))
        return False, torch

    ok = report("PyTorch CUDA available", available and count > 0,
                f"is_available={available}, device_count={count}")
    if not ok:
        return False, torch

    for index in range(count):
        try:
            name = torch.cuda.get_device_name(index)
            capability = torch.cuda.get_device_capability(index)
            props = torch.cuda.get_device_properties(index)
            total_gb = props.total_memory / (1024 ** 3)
            print(f"GPU {index}: {name}; compute capability={capability}; VRAM={total_gb:.2f} GB")
        except Exception as exc:
            report(f"GPU {index} properties", False, repr(exc))
            return False, torch

    try:
        device = torch.device("cuda:0")
        x = torch.randn((2048, 2048), device=device)
        y = torch.randn((2048, 2048), device=device)
        z = (x @ y).mean()
        z.backward() if z.requires_grad else None
        torch.cuda.synchronize()
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        report("CUDA tensor smoke test", True,
               f"matmul={float(z):.6f}, free={free_bytes / 1024**3:.2f}/{total_bytes / 1024**3:.2f} GB")
    except Exception as exc:
        report("CUDA tensor smoke test", False,
               f"{repr(exc)}；常见原因是驱动与 PyTorch CUDA 版本不兼容")
        return False, torch
    finally:
        try:
            del x, y, z
            torch.cuda.empty_cache()
        except Exception:
            pass
    return True, torch


def check_model_layout(model_path):
    root = Path(model_path)
    if not report("model directory", root.is_dir(), str(root)):
        return False

    required_dirs = ["unet", "vae", "scheduler", "tokenizer", "text_encoder"]
    ok = True
    for name in required_dirs:
        ok = report(f"model/{name}", (root / name).is_dir(), "存在" if (root / name).is_dir() else "缺失") and ok

    for component in ("unet", "vae"):
        directory = root / component
        config_ok = (directory / "config.json").is_file()
        weight_files = list(directory.glob("*.safetensors")) + list(directory.glob("*.bin"))
        ok = report(f"model/{component} config", config_ok, "config.json") and ok
        ok = report(f"model/{component} weights", bool(weight_files),
                    ", ".join(p.name for p in weight_files) if weight_files else "未找到 .safetensors/.bin") and ok
    return ok


def check_diffusers_load(model_path, torch_module):
    try:
        from diffusers import StableDiffusionInpaintPipeline
        print("开始只读加载 Diffusers pipeline（不会移动到 GPU）...")
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_path,
            torch_dtype=torch_module.float32,
            safety_checker=None,
            requires_safety_checker=False,
            local_files_only=True,
        )
        channels_ok = int(pipe.unet.config.in_channels) == 9
        report("Diffusers UNet in_channels=9", channels_ok,
               f"got {pipe.unet.config.in_channels}")
        report("Diffusers pipeline load", channels_ok,
               f"VAE scaling={pipe.vae.config.scaling_factor}")
        del pipe
        return channels_ok
    except Exception as exc:
        report("Diffusers pipeline load", False, repr(exc))
        return False


def main():
    parser = argparse.ArgumentParser(description="D2R 服务器 CUDA/模型环境自检")
    parser.add_argument("--model_path", default=DEFAULT_MODEL)
    parser.add_argument("--load_pipeline", action="store_true",
                        help="额外只读加载一次 Diffusers pipeline，耗时较长但不训练")
    args = parser.parse_args()

    print("=" * 72)
    print("D2R server readiness check")
    print("=" * 72)
    smi_ok = check_nvidia_smi()
    cuda_ok, torch_module = check_torch_cuda()
    model_ok = check_model_layout(args.model_path)
    pipeline_ok = True
    if args.load_pipeline and torch_module is not None:
        pipeline_ok = check_diffusers_load(args.model_path, torch_module)

    print("=" * 72)
    ready = smi_ok and cuda_ok and model_ok and pipeline_ok
    if ready:
        print("READY: 可以开始正式训练。")
        return 0
    print("NOT READY: 请先修复上面的 FAIL 项；不要直接开始正式训练。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
