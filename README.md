<div align="center">

# D2R — Two-Stage Diffusion + GAN Refinement for Heritage Image Inpainting

**Stage 1:** LoRA fine-tuning of Stable Diffusion Inpainting ·
**Stage 2:** texture-aware GAN residual refinement ·
**Control:** matched end-to-end single-stage U-Net

[![tests](https://github.com/yilialone/d2r-inpainting/actions/workflows/tests.yml/badge.svg)](https://github.com/yilialone/d2r-inpainting/actions/workflows/tests.yml)
[![license: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![data: CC0-1.0](https://img.shields.io/badge/data-CC0--1.0-lightgrey.svg)](data/public_subset/LICENSE)
[![python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](requirements.txt)

</div>

---

## What this repository is

A leakage-free reference implementation of a two-stage image restoration pipeline
("D2R") for damaged bronze-mirror photographs, plus the matched single-stage control
requested during peer review. It is the code accompanying the revised manuscript.

The repository is **code plus one small, licence-clear image subset**. Model weights,
checkpoints, the study corpus and generated results are deliberately excluded — see
[Data availability](#data-availability) and [`.gitignore`](.gitignore).

> ### Status: unfinished, under maintenance
>
> This is research code accompanying a manuscript under revision — **not a finished
> library**. It is still being maintained, and the remaining work is listed openly in
> [`docs/STATUS.md`](docs/STATUS.md): known defects (including a Stage-2 budget report
> that describes an architecture the code no longer implements), rough edges, and what
> "under maintenance" does and does not commit to. There is **no API stability
> guarantee** and no support commitment.
>
> Two further caveats. The numbers in [`docs/RESULTS.md`](docs/RESULTS.md) come from a
> **single seed (2026)** on one internal 51-pair test set, so nothing here should be
> cited as a published result. And [`data/public_subset/`](data/public_subset/) is a
> **living collection**: further images will be added as additional permissions are
> obtained, so the file set is not frozen.
>
> **New here?** Read the [protocol invariants](#why-the-input-protocol-is-the-interesting-part)
> below, then [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for how the modules fit
> together.

---

## Why the input protocol is the interesting part

Most public inpainting pipelines silently leak the ground truth into the network. This
implementation is built to make that impossible, and the invariants are enforced by
tests rather than by comments:

| Invariant | Where it is enforced |
|---|---|
| Conditioning follows the official SD-Inpainting channel order `[noisy_latent, mask, masked_image_latent]`, and the masked-image latent stays **clean** (never noised) | `training/stage1.py`, asserted in `test_paper_params.py` |
| Stage-2 generator sees **only** `[I_S1, M]` (4 channels) — ground-truth pixels never enter any path, including the texture encoder | `training/stage2.py::_refine`, `inference/pipeline.py::refine_with_stage2` |
| Stage-1 trains with the masked-image latent clean, so the target noise is not algebraically recoverable from the inputs | `training/stage1.py` |
| Pixels outside the mask are preserved **byte-for-byte** (re-composited in uint8) | `inference/pipeline.py` |
| Checkpoints carry a `PROTOCOL_ID`; resuming across an incompatible protocol raises instead of silently continuing | `training/stage1.py`, `training/stage2.py` |

See [`docs/PROTOCOL.md`](docs/PROTOCOL.md) for the full specification.

---

## Repository layout

```
.
├── train.py                     # two-stage / single-stage training entry point
├── infer.py                     # single-image & batch inference CLI
├── evaluate.py                  # multi-method benchmark on a fixed manifest
├── test_paper_params.py         # protocol & default-value test suite (sections A–F)
├── test_inference_api.py        # high-level inference API, with injected stubs
├── requirements.txt
├── LICENSE  NOTICE              # Apache-2.0 (code) + third-party attributions
├── CITATION.cff                 # "Cite this repository" metadata
├── dataset/dataset.py           # synchronized geometric + optional photometric augmentation
├── training/
│   ├── common.py                # shared losses, metrics, parameter & budget accounting
│   ├── stage1.py                # LoRA diffusion fine-tuning
│   ├── stage2.py                # texture-aware GAN refinement
│   └── single_stage.py          # matched end-to-end control
├── models/                      # generator (12-ch texture descriptor + TAG), discriminator
├── inference/                   # pipelines and guidance utilities
│   ├── pipeline.py              # load / stage1_inference / refine_with_stage2
│   └── restore.py               # D2RRestorer + restore_image (high-level API)
├── metrics/                     # PSNR / SSIM / LPIPS / FID / KID with explicit mask semantics
├── utils/
├── scripts/                     # manifest builder, server check, budget report, uncertainty
├── data/
│   ├── README.md                # what is released, and what is not
│   └── public_subset/           # 14 museum CC0 images + provenance (see Data availability)
├── tools/
│   ├── check_protocol.py        # asserts the paper defaults are intact
│   ├── check_image_metadata.py  # EXIF/GPS audit; lossless GPS stripping
│   ├── test_check_image_metadata.py
│   └── check_repo_hygiene.py    # no weights / corpus images / split manifests
├── eval/manifest_template.csv   # evaluation-manifest format example (placeholders only)
├── .github/workflows/tests.yml  # CI: hygiene + protocol suite (CPU only)
└── docs/
    ├── STATUS.md                # what is unfinished, and the maintenance policy
    ├── ARCHITECTURE.md          # how the modules and stages fit together
    ├── PROTOCOL.md              # input/leakage invariants
    └── RESULTS.md               # measured numbers + exact reproduction commands
```

---

## Data availability

This repository ships **one small, licence-clear image subset** and no other data:

| What | Status |
|---|---|
| `data/public_subset/` | **Included** — 14 museum open-access JPEGs, ~3.2 MB, all **CC0 1.0 Universal** |
| Study corpus (full) | **Not included** — not redistributed; contact the corresponding author |
| Damage masks | **Not included** — derived annotations, not redistributed |
| Evaluation split (51 pairs) | **Not included** — `evaluate.py` expects you to build your own manifest |
| Model weights / checkpoints | **Not included** — obtained from Hugging Face Hub or trained locally |

### The public image subset

`data/public_subset/` contains 14 photographs from museum open-access programmes (The
Cleveland Museum of Art, Harvard Art Museums, and the National Museum of Asian Art,
Smithsonian Institution), each with a documented CC0 rights basis. See
[`data/public_subset/README.md`](data/public_subset/README.md) for the per-folder
inclusion table and [`data/README.md`](data/README.md) for how the subset relates to the
code.

Three things to note about it:

* **It is not a benchmark set.** No masks are supplied, so nothing in it can be fed to
  `evaluate.py` as-is. It documents the visual domain and the provenance of the corpus.
* **It carries no EXIF.** No GPS coordinates, no camera identifiers, no capture dates.
  Image metadata is an easily overlooked disclosure channel; precise coordinates of
  archaeological sites are sensitive. [`tools/check_image_metadata.py`](tools/check_image_metadata.py)
  audits a directory for GPS and other EXIF tags and can strip GPS losslessly.
* **Ten author field photographs are not currently included**, because their rights
  status is not yet settled; no licence is asserted over them. They can be added once it
  is documented, at which point `data/public_subset/README.md`, `LICENSE`, `CREDITS.md`,
  `SOURCES.csv` and `CITATION.cff` must be updated together and consistently.

The data licence is **separate from the software licence** — `data/public_subset/LICENSE`
covers the images only.

`eval/manifest_template.csv` is a format reference containing placeholder rows and no real
data. `.gitignore` blocks `*.jpg`, `*.png`, `datasets/` and `eval/*_manifest.csv` so that
corpus images and split listings cannot be committed by accident; `data/public_subset/`
is the single deliberate exception.

## Installation

```bash
git clone https://github.com/yilialone/d2r-inpainting.git
cd d2r-inpainting

# Install the PyTorch build matching your CUDA driver first, then the rest:
python -m pip install -r requirements.txt
```

`requirements.txt` pins the versions used for the reported runs
(torch 2.9.0 / diffusers 0.30.2 / peft 0.17.1 / transformers 4.57.6).

The Stable Diffusion inpainting weights are **not** bundled. By default the code
resolves the Hugging Face Hub id `runwayml/stable-diffusion-inpainting`. For offline or
air-gapped machines, point `D2R_SD_MODEL` at a local snapshot directory:

```bash
export D2R_SD_MODEL=/path/to/models--runwayml--stable-diffusion-inpainting/snapshots/<hash>
```

The same environment variable is honoured by `train.py`, `infer.py`, `evaluate.py`,
`test_paper_params.py` and `scripts/check_server.py`.

---

## Data layout

Images and masks are not distributed with this repository. The loaders expect:

```
datasets/
├── train/{img,mask}/     # e.g. img/0001.jpg  mask/0001.png
└── val/{img,mask}/
```

Pairs are matched by sorted order. For reproducible splits, build a manifest instead —
the required runtime columns are `sample_id,image_path,mask_path`
(see `eval/manifest_template.csv` for the format):

```bash
python scripts/build_dataset_manifest.py --split train \
    --image_dir datasets/train/img --mask_dir datasets/train/mask \
    --output manifests/train.csv
```

You must generate your own evaluation manifest and pass it with `--manifest`; the
manifest shipped here is a placeholder-only template. Manifest paths are resolved
relative to the manifest file, so mirror the layout above or edit the CSV.

---

## Training

```bash
# Full two-stage run (defaults follow the manuscript: 512px, batch 1, accum 1,
# stage-1 lr 5e-4, stage-2 lr 1e-4, up to 100 epochs, patience 20)
python train.py --mode full --resume_from_checkpoint none \
    --stage1_output_dir stage1_results --stage2_output_dir stage2_results

# Individual stages
python train.py --mode stage1 --resume_from_checkpoint none
python train.py --mode stage2 --stage1_checkpoint_dir stage1_results/checkpoint-best

# Matched end-to-end single-stage control (reviewer point R2-2)
python train.py --mode all --seed 2026
```

Every trainer writes `run_config.json` (exact invocation + environment) and
`budget_report.json` (trainable parameters, completed optimizer updates, wall-clock and
device hours, peak memory). `budget_report.json` is the **only** authoritative source
for the manuscript's parameter-count and compute columns — do not quote estimates from
the text.

Multi-GPU is supported through `accelerate launch --multi_gpu` or `torchrun`;
`--train_batch_size` is per process. Note that this changes the effective global batch
size, so report the hardware and global batch size alongside any multi-GPU result.

### Optional Stage-1 capacity / augmentation flags

These are **opt-in**; with the defaults the trainer reproduces the manuscript protocol
exactly. `tools/check_protocol.py` asserts that.

| Flag | Default | Effect |
|---|---|---|
| `--stage1_lora_ff` | off | Also applies LoRA to the FF projection (`ff.net.0.proj`). Measured trainable Stage-1 LoRA: **9,971,712** parameters with the flag on. The `PROTOCOL_ID` gains an `-ff` suffix, so older checkpoints are rejected on resume. |
| `--augment_photometric <float>` | `0.0` | Adds brightness/contrast/saturation jitter (`saturation = value/2`) to the **image only**. Photometric transforms move no pixels, so the mask needs no synchronization — unlike a rotation applied to the image alone. |

Stage-1 learning rate and effective batch size are plain CLI arguments
(`--learning_rate`, `--train_batch_size`, `--gradient_accumulation_steps`). See
[`docs/RESULTS.md`](docs/RESULTS.md) for the measured effect of raising them.

---

## Evaluation

`evaluate.py` compares several methods on one fixed manifest, always with the same
pairs, masks and metric implementation:

```bash
python evaluate.py --manifest eval/manifest.csv --out_dir eval_outputs \
    --stage1_checkpoint stage1_results/checkpoint-best \
    --stage2_checkpoint stage2_results/checkpoint-best \
    --steps 30 --guidance 7.5 --seed 42
```

Predictions are cached per method under `<out_dir>/preds/<method>/`, so an interrupted
run resumes without re-sampling. Mask-normalized PSNR, mask-weighted SSIM and spatially
mask-weighted LPIPS are the primary scalars; full-image diagnostics are reported
alongside; FID/KID are computed on composites that take predictions inside the mask and
the identical reference outside it.

`infer.py` is the single-image / folder entry point and supports the single-stage
control without loading Stable Diffusion at all.

---

## Programmatic inference

`infer.py` is a command-line tool. To call the pipeline from a script or notebook, use
`inference.restore` — it loads the models **once** and can then be called repeatedly,
whereas the lower-level helpers in `inference/pipeline.py` require you to assemble every
step yourself.

```python
from inference import D2RRestorer

with D2RRestorer(
    stage1_checkpoint="stage1_results/checkpoint-best",   # PEFT LoRA adapter
    stage2_checkpoint="stage2_results/checkpoint-best",   # generator.pth
    device="cuda",
    num_steps=30, guidance_scale=7.5, seed=42,            # the protocol of record
) as restorer:
    restored = restorer.restore("photo.jpg", "mask.png")  # paths or PIL.Image
    restored.save("restored.png")

    batch = restorer.restore_batch(images, masks)         # seeds are base_seed + i
```

Only one stage, or the control:

```python
D2RRestorer(stage1_checkpoint="stage1_results/checkpoint-best")            # stage 1 only
D2RRestorer(single_stage_checkpoint="single_stage_results/checkpoint-best")  # no SD loaded
D2RRestorer()                                                             # un-finetuned base
```

Notes on behaviour:

* **Inputs are resized to `size` (default 512)** — image with Lanczos, mask with nearest
  neighbour. Pass `size=None` to supply 512×512 yourself.
* **Outside the mask the output equals the input byte-for-byte**, re-composited in uint8.
  Where the input was resized, the guarantee applies to the resized input.
* **The mask is white-on-black** (values > 127 mark the region to restore).
* `dtype` defaults to `torch.float32` to match the evaluation protocol in
  [`docs/RESULTS.md`](docs/RESULTS.md); pass `torch.float16` for speed at a small
  numerical cost.
* `restore_image(...)` is the one-shot equivalent for when you only need a single call —
  it loads and releases the models each time.

See [`inference/restore.py`](inference/restore.py) for the full parameter list and
`test_inference_api.py` for executable examples.

---

## Tests

```bash
python -m compileall -q dataset models training metrics inference utils train.py infer.py evaluate.py test_paper_params.py test_inference_api.py scripts tools
python test_paper_params.py --quick     # protocol + defaults, no data or weights needed
python test_inference_api.py            # inference API contract, with injected stubs
python tools/check_protocol.py          # opt-in defaults are intact
python tools/test_check_image_metadata.py   # GPS stripping is lossless (synthetic fixture)
python tools/check_image_metadata.py --dir data/public_subset   # audit the released images
python tools/check_repo_hygiene.py      # no weights, corpus images or split manifests
```

`test_paper_params.py` covers parameter defaults, model forward shapes, loss and
mask-fidelity properties, a real one-step G/D update, the single-stage control and the
budget-report parser. `test_inference_api.py` covers the high-level inference entry
point: mode resolution, input normalisation, the 4-channel Stage-2 contract, outside-mask
byte-exactness, seed semantics and batching — all with injected stubs, so no weights are
needed. Section E of `test_paper_params.py` skips itself when `D2R_SD_MODEL` and the test
images are unavailable.

Every command above runs on CPU without model weights or datasets, so this is what
[`.github/workflows/tests.yml`](.github/workflows/tests.yml) runs on each push and pull
request — split into a light "hygiene" job and a full "protocol" job.

---

## Hardware notes (measured)

Reported runs were produced on a single **RTX 4060 Laptop, 8 GB**.

* Stage-1 at batch size 1 uses ~7.5 GB and runs at ~0.64 s/micro-step.
  **Batch size 2 does not fit**: it reaches 7.8/8.2 GB and thrashes to ~13.8 s/step,
  an ~18× slowdown. Use gradient accumulation for a larger effective batch.
* Stage-2 peak memory is only ~2.5 GB; its cost is dominated by Stage-1 cache
  generation (482 images × 30 DDIM steps, ~13 s each at fp32).
* Set `--num_workers 0` on Windows or in restricted sandboxes where named pipes are
  unavailable.

---

## Known limitations

* **Single seed.** All numbers in `docs/RESULTS.md` come from seed 2026 on one internal
  51-pair test set. Multi-seed runs are required before any claim of significance.
* **Discriminator saturation.** In the reported Stage-2 run the hinge discriminator
  loss stayed pinned at its 2.0 floor, i.e. the discriminator did not learn. This
  weakens any argument that depends on adversarial texture refinement and deserves
  separate investigation.
* **PSNR is flat.** The optional Stage-1 flags improve structural and perceptual
  metrics (masked SSIM/LPIPS, FID) but not PSNR. See `docs/RESULTS.md`.
* **Dataset provenance.** The corpus is not redistributed here; object identity and
  usage rights must be verified by the data owner before publication.

---

## Citation

If you use this software, please cite it as below. GitHub can generate the reference from
[`CITATION.cff`](CITATION.cff) ("Cite this repository" in the sidebar).

```bibtex
@software{d2r_bronze_mirror,
  title     = {D2R: Two-Stage Diffusion and GAN Refinement for Heritage Image Inpainting},
  author    = {Guan, Jun and Jia, Qian and Zhang, Jianmin and Li, Yang},
  year      = {2026},
  version   = {1.0.0},
  license   = {Apache-2.0},
  url       = {https://github.com/yilialone/d2r-inpainting}
}
```

The released images are a separate work with their own citation — see
[`data/public_subset/CITATION.cff`](data/public_subset/CITATION.cff). Both accompany:

> "Two-Stage Structural Reconstruction and Texture Refinement for Digital Restoration of
> Ancient Chinese Mountain-Pattern Bronze Mirror Photographs", *npj Heritage Science*,
> manuscript ID `a062e16c-271f-4765-88b0-c2ef3985d848` (under revision).

## License

**Code: [Apache License 2.0](LICENSE).** Copyright 2026 Guan Jun, Jia Qian, Zhang
Jianmin, Li Yang. See [`NOTICE`](NOTICE) for third-party components.

**Images: [CC0 1.0 Universal](data/public_subset/LICENSE)** — the two are separate works
under separate licences, and the Apache licence does not extend to the image subset.

Two dependency caveats worth knowing before you redistribute anything derived from this
code:

* `runwayml/stable-diffusion-inpainting` is distributed under the **CreativeML Open
  RAIL-M** licence, which carries use restrictions that Apache-2.0 does not. The weights
  are not bundled here, but checkpoints you train are derived from them.
* LPIPS and the other third-party packages have their own terms; see [`NOTICE`](NOTICE).
