# D2R — Two-Stage Diffusion + GAN Refinement for Heritage Image Inpainting

**Stage 1** LoRA fine-tuning of Stable Diffusion Inpainting ·
**Stage 2** texture-aware GAN residual refinement ·
**Control** matched end-to-end single-stage U-Net

Reference implementation for the accompanying manuscript.

> **Status: unfinished, under maintenance.** Known defects and what that commits to:
> [`docs/STATUS.md`](docs/STATUS.md).

## Install

```bash
# 1. PyTorch for your CUDA (the CUDA wheel is ~2.5 GB; use .../whl/cpu for CPU-only)
pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu126

# 2. everything else
pip install -r requirements.txt
```

On Linux, OpenCV also needs `sudo apt-get install -y libgl1 libglib2.0-0`.

Stable Diffusion weights are **not** bundled: the code downloads
`runwayml/stable-diffusion-inpainting` from Hugging Face Hub. On an offline machine, point
the `D2R_SD_MODEL` environment variable at a local snapshot directory instead.

## Data

Neither the corpus nor any damage masks are distributed. The loaders expect
`datasets/{train,val}/{img,mask}/`, with image and mask files paired by sorted filename
order; `--train_image_dir` and friends override the defaults. Masks are single-channel
images in which white (value > 127) marks the region to restore. For reproducible splits,
pass CSV manifests (`sample_id,image_path,mask_path`) via `--train_manifest` /
`--val_manifest`.

`data/public_subset/` holds 9 museum open-access photographs (CC0 1.0 Universal) with
per-image provenance, as a visual reference for the corpus. It contains **no masks**, so
it cannot be used as a benchmark.

## Train

```bash
python train.py --mode full --resume_from_checkpoint none
```

Modes: `stage1`, `stage2`, `single_stage`, `full`, `all`. Defaults follow the manuscript
(512×512, batch 1, Stage-1 lr 5e-4, Stage-2 lr 1e-4). Every run writes `run_config.json`
and `budget_report.json`.

## Inference

```bash
python infer.py --image_dir datasets/val/img --mask_dir datasets/val/mask \
    --stage1_checkpoint stage1_results/checkpoint-best \
    --stage2_checkpoint stage2_results/checkpoint-best --out evaluation/run1
```

Or from Python — `inference.restore` loads the models once and can be called repeatedly:

```python
from inference import D2RRestorer

with D2RRestorer(stage1_checkpoint="stage1_results/checkpoint-best",
                 stage2_checkpoint="stage2_results/checkpoint-best") as restorer:
    restored = restorer.restore("photo.jpg", "mask.png")   # paths or PIL images
```

`evaluate.py` compares several methods on one fixed manifest using a single shared metric
implementation.

## Checks

```bash
python test_paper_params.py --quick   # protocol invariants; no weights or data needed
python test_inference_api.py          # inference API, with injected stubs
python tools/check_repo_hygiene.py    # no weights or corpus images committed
```

Read [`docs/PROTOCOL.md`](docs/PROTOCOL.md) before changing the conditioning — it records
the input and leakage invariants the pipeline depends on and why they are not cosmetic.

## License

Code under [Apache-2.0](LICENSE); third-party components in [NOTICE](NOTICE). The image
subset is a separate work under [CC0-1.0](data/public_subset/LICENSE).
