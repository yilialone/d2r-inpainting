# Project status and maintenance

## 1. What this is

A research prototype released alongside a manuscript under revision. It reproduces the
experiments described in that manuscript. It is **not** a finished library, and it is
not maintained to library standards.

## 2. Development status: unfinished

The following are known and accepted at this version. They are listed so that users can
judge what they are relying on, not as a promise of when each will be addressed.

### Correctness and provenance

* **The Stage-2 budget report misdescribes its own model.** `training/stage2.py` writes
  `model_definition` as `"7 input channels [Stage-1 output, masked original, mask]"`,
  while the same file's `PROTOCOL_ID` is
  `d2r-stage2-paper-v4-4ch-I_S1-M-...` and `_refine()` passes 4 channels
  (`[I_S1, M]`). The Stage-1 equivalent was made dynamic; the Stage-2 string was not.
  This matters because `scripts/control_budget_report.py` copies the string straight
  into the Table-4 output, so a report generated without editing it will describe an
  architecture the code does not implement. **Check this string before quoting any
  `budget_report.json` in a manuscript.**
* **The discriminator did not converge** in the reported Stage-2 run: the hinge loss
  stayed pinned at its 2.0 floor for every epoch, i.e. it produced near-constant logits.
  Any claim that depends on adversarial texture refinement needs re-examination.
* **Every reported number is single-seed** (2026) on one internal 51-pair split. No
  significance claim is possible from one run per configuration.

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
* Only `scripts/paired_uncertainty.py` has a test; the rest of `scripts/` is untested.
* `--num_workers 0` is required on Windows and in sandboxes that forbid named pipes. The
  default of `4` will fail in those environments.
* There is no container image and no pinned lockfile, so a fresh install can drift from
  the pinned versions in `requirements.txt` if a transitive dependency changes.

## 3. API stability

**None.** Module layout, function signatures and command-line flags may change between
versions without a deprecation period. If you need stable behaviour, pin a tag or a
commit rather than tracking `main`.

## 4. What "under maintenance" means

* Maintenance is **best-effort, with no support commitment and no response-time
  guarantee.** Issues and pull requests are welcome and will be handled as time allows.
* The protocol invariants in [`PROTOCOL.md`](PROTOCOL.md) are the part we treat as
  load-bearing: changes to the input contract must bump the relevant `PROTOCOL_ID` and
  must keep `test_paper_params.py` and `tools/check_protocol.py` passing. Everything
  else is open to change.
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
