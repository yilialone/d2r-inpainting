<div align="center">

# D2R — Two-Stage Diffusion + GAN Refinement for Heritage Image Inpainting

**Stage 1:** LoRA fine-tuning of Stable Diffusion Inpainting ·
**Stage 2:** texture-aware GAN residual refinement ·
**Control:** matched end-to-end single-stage U-Net

</div>

---

## What this repository is

A leakage-free reference implementation of a two-stage image restoration pipeline
("D2R") for damaged bronze-mirror photographs, plus the matched single-stage control
requested during peer review. It is the code accompanying the revised manuscript.

The repository contains **code only**. Model weights, checkpoints, datasets and
generated results are deliberately excluded — see [`Data layout`](#data-layout) and
[`.gitignore`](.gitignore).

> **Status.** This is a maintained research codebase, not a released artifact. The
> numbers in [`docs/RESULTS.md`](docs/RESULTS.md) come from a **single seed (2026)**
> on one internal 51-pair test set and are reported for engineering comparison only.
> They are not the final manuscript table.

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
├── requirements.txt
├── dataset/dataset.py           # synchronized geometric + optional photometric augmentation
├── training/
│   ├── common.py                # shared losses, metrics, parameter & budget accounting
│   ├── stage1.py                # LoRA diffusion fine-tuning
│   ├── stage2.py                # texture-aware GAN refinement
│   └── single_stage.py          # matched end-to-end control
├── models/                      # generator (12-ch texture descriptor + TAG), discriminator
├── inference/                   # pipelines and guidance utilities
├── metrics/                     # PSNR / SSIM / LPIPS / FID / KID with explicit mask semantics
├── utils/
├── scripts/                     # manifest builder, server check, budget report, uncertainty
├── tools/check_protocol.py      # asserts the paper defaults are intact
├── eval/manifest_template.csv   # evaluation-manifest format example (placeholders only)
└── docs/
    ├── PROTOCOL.md              # input/leakage invariants
    └── RESULTS.md               # measured numbers + exact reproduction commands
```

---

## Data availability

**No data is distributed with this repository** — no images, no masks, no manifests of
the authors' splits, and no model weights. `.gitignore` additionally blocks `*.png`,
`*.jpg`, `datasets/` and `eval/*_manifest.csv` so a split listing cannot be committed by
accident.

Use `eval/manifest_template.csv` only as a format reference; it contains placeholder rows
and no real data.

The corpus used in the manuscript is not redistributed here. Contact the corresponding
author for access, and confirm object identity and usage rights before publishing any
result derived from it.

## Installation

```bash
git clone <your-repo-url> d2r-inpainting
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

## Tests

```bash
python -m compileall -q dataset models training metrics inference utils train.py infer.py scripts
python test_paper_params.py --quick     # protocol + defaults, no data or weights needed
python tools/check_protocol.py          # opt-in defaults are intact
```

`test_paper_params.py` covers parameter defaults, model forward shapes, loss and
mask-fidelity properties, a real one-step G/D update, the single-stage control and the
budget-report parser. Section E (end-to-end inference) skips itself when
`D2R_SD_MODEL` and the test images are unavailable.

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

```bibtex
@article{d2r-bronze-mirror,
  title   = {Two-stage diffusion and GAN refinement for the restoration of
             damaged bronze mirror photographs},
  journal = {npj Heritage Science},
  note    = {Manuscript under revision},
  year    = {2025}
}
```

## License

**Not yet specified.** Add a `LICENSE` file before publishing — without one, the code is
"all rights reserved" by default, which prevents reuse. For research code of this kind,
Apache-2.0 or MIT are the usual choices; note that the code depends on
`runwayml/stable-diffusion-inpainting` (CreativeML Open RAIL-M) and on LPIPS, whose
terms differ from the license you pick here.
