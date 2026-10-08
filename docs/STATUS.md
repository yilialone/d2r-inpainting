# Project status and maintenance

## 1. What this is

The code release for a research prototype, published alongside a manuscript under
revision. It contains the implementation of the two-stage restoration method described in
that manuscript. It is **not** a finished library, and it is not maintained to library
standards.

**This is a code release only.** No training logs, no loss curves, no per-run metric
records and no checkpoints are distributed here. The manuscript is the reference for the
method and for any reported result.

## 2. Development status: unfinished

The following are known and accepted at this version. They are listed so that users can
judge what they are relying on, not as a promise of when each will be addressed.

### Correctness and provenance

* **Budget reports written before this version misdescribe the Stage-2 model.**
  `training/stage2.py` used to hard-code `model_definition` as `"7 input channels
  [Stage-1 output, masked original, mask]"`, while the same file's `PROTOCOL_ID` was
  already `d2r-stage2-paper-v4-4ch-I_S1-M-...` and `_refine()` passed 4 channels
  (`[I_S1, M]`). `scripts/control_budget_report.py` copies that string straight into the
  Table-4 output. The field is now produced by
  `SimpleUNetGeneratorWithTexture.architecture_summary`, which derives the channel count
  from the instance, so it can no longer drift. **Reports generated before the fix are
  not retroactively corrected — regenerate them, or edit the string by hand, before
  quoting one in a manuscript.**
* **Training results are not stored in this repository.** This is a code release only, as
  stated in section 1: no training logs, no loss curves, no per-run metric records and no
  checkpoints are distributed here.
* **The loaders and the evaluation split are configuration, not data.** The corpus and
  the article's damage masks are not distributed (see `data/README.md`), so the
  artefacts and masks that produced any published number are not available here. Nothing
  in this repository fixes a particular 51-pair evaluation set; `evaluate.py` expects
  you to supply your own manifest with the columns `sample_id,image_path,mask_path`.

### Rough edges

* `inference/guidance.py::compute_adaptive_cfg()` returns its constant upper bound for
  any mask below 25 % of the frame — which is every mask in the evaluation set. On this
  corpus it is therefore equivalent to a fixed `CFG = 8.0`, not an adaptive scheme.
* Stage-2 cost is dominated by Stage-1 cache generation, and the cache is regenerated
  whenever the Stage-1 checkpoint or the sampling settings change (482 images × 30 DDIM
  steps ≈ 104 min on the reference hardware).
* `evaluate.py` reuses a cached prediction set only when every sample in the current
  manifest is present, so a `--limit` run leaves a partial cache that a later full run
  ignores rather than resumes.
* **No script regenerates the paper's qualitative comparison figures.** They were
  produced ad hoc during revision and are not part of this repository.
* Only one file under `scripts/` has a test suite; the rest is exercised only through
  `test_paper_params.py` section F.
* `--num_workers 0` is required on Windows and in sandboxes that forbid named pipes. The
  default of `4` will fail in those environments.
* There is no container image and no pinned lockfile, so a fresh install can drift from
  the pinned versions in `requirements.txt` if a transitive dependency changes.
* **The diffusers-native LoRA path never succeeds for these adapters.**
  `pipe.load_lora_weights()` rejects the PEFT-format keys that Stage 1 saves (the
  `base_model.model.` prefix), so `inference/pipeline.py` always falls back to rewriting
  the adapter into a temporary directory and loading that. It works, but the first
  attempt is wasted and used to dump a multi-thousand-character error into the log; the
  message is now truncated. Detecting the prefix up front would be the real fix.
* **`inference.restore` is new in this version.** `D2RRestorer` / `restore_image` are
  covered by `test_inference_api.py` using injected stubs, and were smoke-tested once
  end to end against real checkpoints, but the API should still be treated as
  provisional.

## 3. API stability

**None.** Module layout, function signatures and command-line flags may change between
versions without a deprecation period. If you need stable behaviour, pin a tag or a
commit rather than tracking `main`.

## 4. What "under maintenance" means

* Maintenance is **best-effort, with no support commitment and no response-time
  guarantee.** Issues and pull requests are welcome and will be handled as time allows.
* The protocol invariants in [`PROTOCOL.md`](PROTOCOL.md) are the part we treat as
  load-bearing: changes to the input contract must bump the relevant `PROTOCOL_ID` and
  must keep `test_paper_params.py` passing. Everything else is open to change.
* The CI in [`.github/workflows/tests.yml`](../.github/workflows/tests.yml) defines the
  minimum bar a change must clear. It runs on CPU without weights or data, so passing it
  does **not** establish scientific validity — only that the protocol contract and the
  release hygiene still hold.

## 5. The image subset is a living collection

`data/public_subset/` is deliberately incomplete. It contains only material whose rights
status is unambiguous, and **further images will be added as additional permissions are
obtained** from the holding institutions and from the recording staff. In practice this
means:

* the set of files will change between versions — do not assume it is frozen;
* the `version` field of `data/public_subset/CITATION.cff` is incremented when it does;
* if you need byte-stable inputs, pin a commit or record the exact file list you used;
* any addition must keep `README.md`, `LICENSE`, `CREDITS.md`, `SOURCES.csv` and
  `CITATION.cff` in that directory mutually consistent, and must pass
  `python tools/check_image_metadata.py --dir data/public_subset`.

No date is promised for any addition: the constraint is the rights position, not
editorial effort.
