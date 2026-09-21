# Architecture

How the pieces fit together. For the tensor-level input contract see
[`PROTOCOL.md`](PROTOCOL.md); for measured numbers see [`RESULTS.md`](RESULTS.md).

---

## 1. Data flow

```
                    ┌──────────────────────────────────────────┐
   image + mask ───►│ Stage 1 — LoRA fine-tuned SD-Inpainting  │───►  I_S1
                    │ training/stage1.py                       │      (coarse,
                    └──────────────────────────────────────────┘       plausible fill)
                                     │
                                     │  I_S1 is precomputed once per sample and
                                     │  cached on disk (keyed by sample_id)
                                     ▼
                    ┌──────────────────────────────────────────┐
   image + mask ───►│ Stage 2 — texture-aware GAN refinement   │───►  residual
   + I_S1 (cache)   │ training/stage2.py                       │
                    └──────────────────────────────────────────┘
                                     │
                                     ▼
              refined = orig ⊙ (1 − M) + (I_S1 + residual) ⊙ M

   Control path (no diffusion, no cache):

   masked image + mask ──► Single-stage U-Net ──► prediction ──► composite
                           training/single_stage.py
```

Every path composites in **uint8 space**, so pixels outside the mask are preserved
byte-for-byte.

---

## 2. Module map

| Module | Responsibility |
|---|---|
| `train.py` | Entry point. Modes: `stage1`, `stage2`, `single_stage`, `full`, `all`. Owns the CLI and writes `run_config.json`. |
| `infer.py` | Inference CLI for single images or a directory; can run the two-stage pipeline or the single-stage control (the latter never loads Stable Diffusion). |
| `evaluate.py` | Multi-method benchmark over one fixed manifest. Caches predictions per method under `<out_dir>/preds/<method>/`. |
| `dataset/dataset.py` | `InpaintingDataset` + `create_dataloaders`. Owns normalization and augmentation (geometric transforms stay synchronized between image and mask). |
| `training/common.py` | **Single source of truth** for the losses, validation proxies, composite score, parameter counting and budget accounting. Stage-2 and the single-stage control import from here so their comparison differs only in architecture. |
| `training/stage1.py` | LoRA fine-tuning of the SD-Inpainting UNet. |
| `training/stage2.py` | GAN refinement on top of the frozen Stage-1 output; owns cache generation. |
| `training/single_stage.py` | Matched end-to-end control trained from random initialisation. |
| `models/generator.py` | `SimpleUNetGeneratorWithTexture`: 4-channel input, 12-channel texture descriptor (3 RGB + 3 Canny + 1 Sobel + 1 Laplacian + 4 Gabor), SE blocks, self-attention, texture attention gating, learnable residual scale (init 0.3). |
| `models/discriminator.py` | `SimpleUNetDiscriminator`: per-pixel raw logits, no output sigmoid. |
| `models/single_stage.py` | `SingleStageInpaintingGenerator`: tanh head, backbone layer-for-layer isomorphic to the Stage-2 generator apart from the first conv and the head. |
| `inference/pipeline.py` | Loading and running the pipelines; `stage1_inference`, `refine_with_stage2`, `single_stage_inference`, batch metrics. |
| `inference/restore.py` | **High-level, reusable entry point.** `D2RRestorer` loads the models once and can be called repeatedly (batching, context manager, injectable components); `restore_image` is the one-shot helper. This is the layer to use from a script or notebook. |
| `inference/guidance.py` | Canny extraction, adaptive CFG, multi-CFG TTA, DPM-Solver++ helper. |
| `metrics/calculator.py` | PSNR / SSIM / LPIPS / FID / KID with explicit full-image and mask-region semantics. |
| `utils/image.py` | Resizing, mask overlays, comparison sheets, pair discovery. |
| `scripts/` | Offline utilities: manifest builder, server readiness check, Table-4 budget report, paired-uncertainty analysis. |
| `tools/` | Protocol, metadata and repository-hygiene checks. |

---

## 3. Stage 1 — LoRA diffusion fine-tuning

* Loads `StableDiffusionInpaintPipeline` in fp32; **freezes** the VAE and text encoder.
* Attaches LoRA to the UNet attention projections. `--stage1_lora_ff` additionally covers
  `ff.net.0.proj`; the default keeps the manuscript's attention-only configuration.
* Caches the text embeddings once, because the prompt is constant for the whole run.
* Per step: VAE-encode the image and the masked image → sample noise and a timestep →
  concatenate the 9-channel conditioner → predict the noise → **masked** MSE.
  The masked-image latent is kept clean; see `PROTOCOL.md` §3 for why that is not
  cosmetic.
* Validation reports masked noise MSE averaged over `--stage1_val_draws` independent
  draws, which is what makes two configurations comparable at all.
* Model selection uses `restoration_score()`: run an actual restoration through the
  diffusers pipeline on a small validation subset and measure masked PSNR. An
  un-finetuned baseline is measured first, and a checkpoint is only kept if it beats
  the previous best *and* is not worse than that baseline. The validation loss alone has
  poor discriminative power on this corpus.
* Writes a PEFT adapter plus `training_state.pt`, which records `protocol_id`,
  `lora_target_modules`, `global_step` and `train_seconds` so that a run is
  self-describing.

## 4. Stage 2 — GAN refinement

* **Cache.** Running Stage 1 for every training step would dominate the cost, so `I_S1`
  is precomputed once per sample for both splits and stored under
  `<stage2_out>/stage1_cache/<namespace>/`, where the namespace is a hash of the
  Stage-1 inference configuration. Entries are keyed by `sample_id`, so they survive
  loader reordering; changing the Stage-1 checkpoint or the sampling settings produces a
  different namespace and the cache is regenerated rather than silently mixed.
* **Generator input is `[I_S1, M]`** (4 channels). The ground truth enters no path,
  including the texture encoder — the invariant that earlier versions violated.
* **Losses** (`training/common.py`): hinge GAN (λ 0.1) + masked L1 (λ 50) + masked
  normalized Sobel texture (λ 10); optional masked VGG perceptual loss, off by default.
* **Selection** is by minimum validation generator loss; the composite score is recorded
  alongside it.
* The generator is smaller and cheaper than Stage 1 (peak ~2.5 GB versus ~7.5 GB).

## 5. Single-stage control

The reviewer-requested control answers "how much comes from the two-stage design, and
how much merely from seeing the domain?". It therefore shares everything it can with
Stage 2 — backbone, losses, optimiser, selection rule — and differs only in having no
diffusion prior: 4-channel `[masked RGB, M]` input, direct tanh prediction, random
initialisation, no Stage-1 cache. Training budget is matched to a reference two-stage run
by optimizer steps (or GPU-hours) and the derivation is written into `budget_report.json`.

## 6. Where the invariants live

| Invariant | Code | Test |
|---|---|---|
| SD-Inpainting channel order; clean masked-image latent | `training/stage1.py` | `test_paper_params.py` §A |
| 4-channel `[I_S1, M]`; no ground truth in any generator path | `training/stage2.py::_refine`, `inference/pipeline.py::refine_with_stage2` | `test_paper_params.py` §A, §C |
| Outside-mask pixels byte-identical | `inference/pipeline.py` | `test_paper_params.py` §F |
| Protocol IDs reject incompatible checkpoints | both trainers | `test_paper_params.py` §A |
| Paper defaults unchanged; opt-in flags off | `train.py`, `training/stage1.py` | `tools/check_protocol.py` |
| Released images carry no GPS | — | `tools/check_image_metadata.py`, `tools/test_check_image_metadata.py` |
| No weights or corpus images in the repository | — | `tools/check_repo_hygiene.py` |

## 7. Extension points

* **New augmentation** — add it in `dataset/dataset.py` and decide explicitly whether the
  mask must follow. Photometric transforms need no synchronization; geometric ones do.
* **New conditioning** — change the channel layout in `stage1.py` together with the
  inference pipeline, bump `PROTOCOL_ID`, and start a new output directory.
* **New generator or discriminator** — implement it in `models/` and keep both trainers
  importing their losses from `training/common.py`; otherwise a comparison between two
  runs stops being attributable to the architecture.
* **New metric** — add it to `metrics/calculator.py` and state whether it is a
  full-image or a mask-region quantity. Mixing the two is the most common way to make a
  results table meaningless.
