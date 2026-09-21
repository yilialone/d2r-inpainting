# Measured results

> **Scope.** Everything below comes from **one seed (2026)** on an internal 51-pair
> test set, on a single RTX 4060 Laptop (8 GB). These are engineering measurements used
> to sanity-check the pipeline, **not** the final manuscript table. Do not cite them
> without multi-seed runs.
>
> This repository is unfinished and under maintenance; the known defects and rough edges
> behind several of these numbers are listed in [`STATUS.md`](STATUS.md) — in particular
> the non-converging discriminator (§8) and the Stage-2 budget-report string that
> misdescribes its own architecture.

---

## 1. Benchmark setup

| Item | Value |
|---|---|
| Test set | 51 image/mask pairs from the authors' internal evaluation split |
| Manifest | a local manifest in the format of `eval/manifest_template.csv` (not distributed) |
| Sampler | DDIM, 30 steps |
| Guidance | CFG 7.5 |
| Prompt | empty string (the protocol of record) |
| Seed | 42 |
| Resolution | 512 × 512 |
| Mask area | mean 9.9 %, median 9.4 %, range 2.1 %–19.9 % — **all pairs are small holes** |

Because every mask is under 25 % of the frame, the adaptive-CFG helper
(`inference/guidance.py::compute_adaptive_cfg`) returns its constant upper bound of 8.0
for all 51 pairs. **On this dataset it is not adaptive at all** — it is equivalent to
setting CFG to 8.0.

## 2. Headline comparison

Two systems are compared. "Paper protocol" = manuscript defaults
(`lr 5e-4` for Stage-1 as documented, batch 1, attention-only LoRA, no photometric
augmentation). "Opt-in" = the optional flags described in the README.

| Method | Masked PSNR | Masked SSIM | Masked LPIPS | Full PSNR | Full SSIM | Full LPIPS | FID |
|---|---|---|---|---|---|---|---|
| Base SD-Inpainting (no fine-tuning) | 16.9129 | 0.2238 | 0.3212 | 21.2368 | 0.6082 | 0.1254 | 4.3491 |
| Stage-1 only — paper protocol | 17.0424 | 0.2284 | 0.3186 | 21.2684 | 0.6082 | 0.1254 | 4.4180 |
| Stage-1 only — opt-in | 17.0506 | **0.2341** | **0.3046** | 21.2803 | 0.6084 | 0.1242 | **4.1553** |
| D2R two-stage — paper protocol | **17.7522** | 0.2584 | 0.3579 | **28.0281** | 0.9246 | 0.0446 | 6.2700 |
| D2R two-stage — opt-in | 17.7375 | **0.2684** | **0.3212** | 28.0135 | **0.9256** | **0.0404** | **5.0388** |

Per-pair paired differences, opt-in minus paper protocol (n = 51):

| Metric | Mean Δ | SEM | opt-in better | paper better |
|---|---|---|---|---|
| Masked PSNR | −0.0147 | 0.0519 | 22 | 29 |
| Masked SSIM | **+0.0100** | 0.0020 | **36** | 15 |
| Masked LPIPS | −0.0366 (lower is better) | — | — | — |
| Full-image LPIPS | −0.0042 (lower is better) | — | — | — |
| FID | −1.2313 (lower is better) | — | — | — |

**Reading.** The optional Stage-1 settings produce a consistent, statistically
comfortable improvement in structural and perceptual fidelity (masked SSIM and LPIPS,
FID) while leaving PSNR unchanged — the paired PSNR difference is 0.015 dB against a
standard error of 0.052 dB, i.e. indistinguishable from zero. Report masked SSIM / LPIPS
/ FID if you report this change at all; reporting PSNR alone hides the effect.

## 3. Stage-1 development log

Selection metric: masked-region restoration PSNR on 10 validation images, 10 DDIM steps,
CFG 7.5, using the training prompt. The un-finetuned baseline, measured before training,
is **18.995 dB** and was reproduced identically in every run.

| Run | Configuration | Best epoch | Best selection PSNR | vs baseline |
|---|---|---|---|---|
| Paper protocol | lr 1e-4, batch 1, accum 1, attention-only LoRA | 0 | 19.071 | +0.075 |
| Effective batch 8, high LR | lr 5e-4, batch 1, accum 8, +FF LoRA, +jitter 0.1 | 0 | 18.966 | **−0.029** |
| Effective batch 8, moderate LR | lr 2e-4, batch 1, accum 8, +FF LoRA, +jitter 0.1 | 5 | **19.237** | **+0.242** |

The `lr 5e-4` run degraded monotonically (epoch 1: 18.512 dB) and was aborted: with
`T_max = 25 epochs` the cosine schedule would not have decayed into an effective range
before early stopping, so it could never have produced a `checkpoint-best`. **Scaling the
learning rate linearly with the effective batch size (1e-4 × 8 = 8e-4) is too aggressive
here; 2e-4 works.** The successful run early-stopped at epoch 13 with best epoch 5.

Per-epoch selection PSNR of the lr 2e-4 run — note the run is essentially flat with a
gentle upward drift, and the spread between adjacent epochs (~0.2 dB) is the same order
as the differences being compared:

```
ep  0    1     2     3     4     5     6     7     8     9    10    11    12    13
   19.119 18.960 19.125 19.037 19.135 19.237 19.079 18.691 19.136 19.010 19.053 19.065 18.993 19.089
```

Stage-1 LoRA trainable parameters: **9,971,712** with `--stage1_lora_ff` (attention
projections + `ff.net.0.proj`, r=32, α=64, dropout=0.05), read from the checkpoint's
`training_state.pt`.

## 4. Inference-side sweep (negative result)

Before the full benchmark, 5 sampling configurations were compared on 12 test pairs
using the Stage-1 checkpoint only. Inference is deterministic (fixed per-sample
generator), so differences come only from the sampling parameters:

| Variant | Prompt | CFG | Steps | Masked PSNR | Masked SSIM |
|---|---|---|---|---|---|
| A — protocol of record | empty | 7.5 | 30 | **17.7202** ± 1.2034 | **0.2186** |
| B | training prompt | 7.5 | 30 | 17.4321 | 0.2058 |
| C | training prompt | 8.0 | 30 | 17.4026 | 0.2043 |
| D | training prompt | 5.0 | 30 | 17.5853 | 0.2127 |
| E | training prompt | 7.5 | 20 | 17.5811 | 0.2106 |

The standard error on 12 images is ~1.2 dB, so none of these differences is significant.
Two conclusions were nevertheless taken:

1. The empty prompt used at evaluation is **not** a mistake — using the training prompt
   is slightly worse, not better.
2. Adaptive CFG has no effect on this dataset (§1), so it cannot explain anything either.

## 5. Cost measurements

| Phase | Measured |
|---|---|
| Stage-1, 433 micro-steps/epoch @ batch 1 | ~0.64 s/step, ~4.7 min/epoch |
| Stage-1, 216 micro-steps/epoch @ batch 2 | ~13.8 s/step — 7.8/8.2 GB, **thrashes, unusable** |
| Stage-1 peak memory @ batch 1 | ~7.5 GB |
| Stage-2 Stage-1 cache generation | 482 images × 30 DDIM steps ≈ 104 min (fp32, ~13 s/image) |
| Stage-2 training | ~3.4 min/epoch, peak memory ~2.5 GB |
| Full 51-pair evaluation | ~11 s/image for the 30-step Stage-1 pass |

Stage-2 trainable parameters: generator 26,177,299; discriminator 7,085,505.

## 6. Reproduction commands

The opt-in run reported above:

```bash
# Stage 1
python train.py --mode stage1 \
    --stage1_output_dir stage1_v2_seed2026 \
    --train_batch_size 1 --gradient_accumulation_steps 8 \
    --learning_rate 2e-4 --lr_warmup_steps 60 \
    --stage1_epochs 25 --early_stopping_patience 8 \
    --stage1_selection_metric psnr --stage1_selection_images 10 --stage1_selection_steps 10 \
    --stage1_val_draws 4 --stage1_grad_clip 1.0 \
    --stage1_lora_ff --augment_photometric 0.1 \
    --seed 2026 --resume_from_checkpoint none

# Stage 2 (reuses the Stage-1 best checkpoint; regenerates the cache)
python train.py --mode stage2 \
    --stage1_checkpoint_dir stage1_v2_seed2026/checkpoint-best \
    --stage2_output_dir stage2_v2_seed2026 \
    --cache_dir stage2_v2_seed2026/stage1_cache \
    --stage2_lr 1e-4 --stage2_epochs 20 --early_stopping_patience 6 \
    --seed 2026 --resume_from_checkpoint none

# Benchmark (paper protocol)
python evaluate.py --manifest eval/manifest.csv --out_dir eval_optin \
    --stage1_checkpoint stage1_v2_seed2026/checkpoint-best \
    --stage2_checkpoint stage2_v2_seed2026/checkpoint-best \
    --steps 30 --guidance 7.5 --seed 42
```

To reproduce the "paper protocol" column, drop `--stage1_lora_ff`,
`--augment_photometric`, `--gradient_accumulation_steps 8` and use
`--learning_rate 1e-4 --stage1_epochs 25 --early_stopping_patience 8`, pointing the
output directories elsewhere. Run `evaluate.py` with the same `--steps/--guidance/--seed`
into a *separate* `--out_dir`; both runs write their own `report.json`, and the
per-method predictions under `preds/` are the inputs to a paired comparison.

> `--num_workers 0` was used throughout because the runs happened on Windows.

## 7. Verification performed

| Check | Result |
|---|---|
| `test_paper_params.py --quick` | 94 passed / 0 failed |
| `tools/check_protocol.py` | all defaults intact |
| Metric-harness parity: recompute the cached predictions and compare with the stored report | PSNR identical, SSIM within 5e-8 |
| Two independent runs of the evaluation harness | bit-identical metrics (max deviation 0.0) |
| Stage-1 un-finetuned baseline PSNR | 18.995 dB, identical across independent runs |
| Benchmark manifest integrity | all 51 `mask_path` entries resolve (checked locally; the split listing itself is not distributed) |

## 8. Open items

* **Single seed only.** No significance claim is possible from one run per configuration.
* **The hinge discriminator saturated.** In the Stage-2 run reported here the
  discriminator loss stayed pinned at its 2.0 floor (`train_disc=2.0000` for every
  epoch), i.e. the discriminator produced near-constant logits and did not learn. Any
  claim that depends on adversarial texture refinement should be revisited before it is
  published.
* **PSNR did not move.** See §2.
* **Full-image metrics are composite-dependent.** Compare them only between methods
  evaluated with the same code version.
