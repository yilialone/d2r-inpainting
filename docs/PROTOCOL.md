# D2R input & leakage protocol

This document specifies the exact tensor contracts the pipeline relies on, and why each
one matters. Everything here is enforced in code and covered by assertions in
`test_paper_params.py`.

If you change any item marked **frozen**, you must bump the relevant `PROTOCOL_ID` and
retrain from scratch — old checkpoints will be rejected on resume by design.

---

## 1. Data normalization (frozen)

Images are converted to `float32` in **[-1, 1]** (`/127.5 - 1.0`), matching what the
Stable Diffusion VAE expects. Masks are `float32` in `{0, 1}` (thresholded at 0.5).

Feeding `[0, 1]` images to `vae.encode` while the inference path feeds `[-1, 1]` is a
silent train/inference mismatch. Do not "fix" this locally; change it in
`dataset/dataset.py` for every stage at once.

## 2. Augmentation (frozen for reported results)

* **Geometric** — horizontal flip (p=0.5) and rotation (±5°) are applied to the image
  *and* the mask with **identical random parameters**. The mask uses nearest-neighbour
  interpolation and fills with 0.
* **Photometric** — optional brightness/contrast/saturation jitter, applied to the
  image only. This is safe because a photometric transform moves no pixels, so there is
  nothing to synchronize. It is **off by default**
  (`--augment_photometric 0.0`).

> Applying a geometric transform to the image alone desynchronizes the mask and silently
> corrupts the supervision signal. This is the single easiest way to break the pipeline:
> if you add an augmentation, decide explicitly whether the mask must follow it.

## 3. Stage-1 conditioning — 9 channels (frozen)

```python
masked_images        = images * (1 - masks)
latents              = vae.encode(images).latent_dist.sample()          * scaling_factor
masked_image_latents = vae.encode(masked_images).latent_dist.sample()   * scaling_factor

noisy_latents = scheduler.add_noise(latents, noise, t)

latent_model_input = torch.cat([noisy_latents, mask_latents, masked_image_latents], dim=1)
```

Two properties are non-negotiable:

**(a) Channel order.** The UNet's `conv_in` expects
`[noisy_latent (4) | mask (1) | masked_image_latent (4)]`. `conv_in` is **frozen** —
LoRA adapters are attached to attention projections (and optionally FF projections), so
a permuted channel order cannot be compensated by fine-tuning. It also contradicts the
diffusers inpainting pipeline used at inference, which builds the correct order itself.

**(b) The masked-image latent is clean — never noised.** This is the subtler point, and
it is a genuine leakage channel. If the masked-image latent were noised with *the same*
`noise` tensor, then

```
noisy_latents - noisy_masked_latents
    = sqrt(alpha_bar_t) * (latents - masked_image_latents)
    = sqrt(alpha_bar_t) * (encode(x0) - encode(x0 * (1 - M)))
```

i.e. the difference of two input channel groups would be a **noise-free term proportional
to the ground-truth content inside the hole**. Combined with the timestep embedding, the
target noise becomes algebraically recoverable, the masked MSE can be driven towards
zero without learning inpainting at all — and, because the shortcut is unavailable at
inference (where the masked-image latent is clean), the checkpoint gets *worse* at the
actual task while its training loss looks excellent.

The same reasoning applies to the stage-2 design below.

## 4. Stage-2 conditioning — 4 channels (frozen)

```python
residual = generator(stage1_out, mask)                       # [I_S1 (3) | M (1)]
refined  = orig * (1 - mask) + (stage1_out + residual) * mask
```

`orig` is used **only** for compositing outside the mask, where it is identical to the
input image's undamaged region. Ground-truth pixels inside the mask enter no path of the
network — including the 12-channel texture descriptor and the texture attention gate.

An earlier design fed the generator `[I_S1, orig, M]` (7 channels). That version is
removed: it let the generator read the answer inside the hole during training, and the
same call site also received the ground-truth image at inference.

`test_paper_params.py` asserts both the 4-channel contract and the absence of the old
concatenation in `stage2.py`, `pipeline.py` and `generator.py`.

## 5. Outside-mask fidelity (frozen)

Both Stage-2 refinement and single-stage inference re-composite in **uint8** space:

```python
out = original_uint8.copy(); out[mask] = prediction_uint8[mask]
```

A naive `uint8 → float32 → uint8` round trip introduces ±1/255 truncation errors outside
the mask (e.g. 128 → 127), contradicting any claim that non-mask pixels are untouched.
Mask-region metrics are unaffected either way; **full-image diagnostics must be
recomputed with a single code version before being tabulated.**

## 6. Protocol IDs and checkpoint compatibility (frozen)

| Stage | `PROTOCOL_ID` |
|---|---|
| Stage-1 | `d2r-stage1-paper-v3-noisy-mask-clean-condition-attention-lora` (adds `-ff` when `--stage1_lora_ff` is used) |
| Stage-2 | `d2r-stage2-paper-v4-4ch-I_S1-M-leakage-free-sample-id-logits-hinge` |

`_load_checkpoint` compares the stored id with the active one and **raises** on mismatch
instead of silently resuming. `load_state_dict(strict=True)` additionally rejects
shape-incompatible generators.

Changing the channel order, the LoRA target set, or the conditioning contents therefore
requires: bump the id → new output directory → `--resume_from_checkpoint none` → retrain
→ re-evaluate every number.

## 7. Model-selection rules

* **Stage-1** — best checkpoint by **maximum masked-region restoration PSNR** on a small
  validation subset (default 7 images / 6 DDIM steps / CFG 7.5), measured *before*
  training starts as an un-finetuned baseline. A checkpoint is only saved as best if it
  is strictly better than the previous best **and** not worse than the baseline.
  Masked noise-prediction MSE is available via `--stage1_selection_metric loss` but has
  poor discriminative power on this corpus and is not recommended.
* **Stage-2** — best checkpoint by minimum validation generator loss, with the composite
  score recorded alongside.
* **Single-stage control** — same rule as Stage-2, sharing the same implementation in
  `training/common.py` so the comparison differs only in architecture.

## 8. Metric conventions

* Primary scalars are **mask-normalized**: masked PSNR, mask-weighted SSIM, spatially
  mask-weighted LPIPS.
* Full-image PSNR/SSIM/LPIPS are reported as diagnostics. Because the outside-mask
  region is byte-identical to the reference, full-image PSNR is dominated by that region
  and is **not** comparable across methods that composite differently.
* Dataset-level FID/KID use composites: predictions inside the mask, the identical
  reference outside it.
* The per-image Inception feature distance is a diagnostic named
  `inception_feature_l2`; it is not FID.

## 9. Quick self-check

```bash
python test_paper_params.py --quick
```

It must pass before you trust any run. It verifies the defaults, the channel order,
the 4-channel Stage-2 contract, the absence of ground-truth leakage, and the
outside-mask byte-exactness property.
