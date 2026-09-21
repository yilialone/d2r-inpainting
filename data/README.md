# `data/` — released image data

This directory holds the **only** image data distributed with this repository.

```
data/
└── public_subset/         13 museum open-access JPEGs, CC0 1.0 Universal
    ├── 02_museum_Cleveland_CC0/
    ├── 03_museum_Harvard_no_permission_required/
    ├── 04_museum_NMAA_Smithsonian_CC0/
    ├── SOURCES.csv        per-image provenance (14 rows)
    ├── CREDITS.md         credit lines
    ├── LICENSE            CC0 1.0 Universal (images only)
    └── README.md
```

The authoritative description — what is included, what is not yet included and why, and
the rights basis for each folder — is [`public_subset/README.md`](public_subset/README.md).

## This set will grow

`data/public_subset/` is a **living collection**, not a frozen release. Only material
whose rights status is unambiguous is included, and **further images will be added as
additional permissions are obtained** from the holding institutions and from the
recording staff.

Consequences for anyone reusing it:

* the file list will change between versions — do not assume it is stable;
* the `version` field in `public_subset/CITATION.cff` is incremented when it changes;
* if you need reproducible inputs, pin a commit and record the exact file list you used;
* no date is promised for any particular addition — the constraint is the rights
  position, not editorial effort.

Every addition must keep `public_subset/README.md`, `LICENSE`, `CREDITS.md`,
`SOURCES.csv` and `CITATION.cff` mutually consistent, and must pass the metadata audit
below. See [`../docs/STATUS.md`](../docs/STATUS.md) §5 for the full policy.

## What this data is, and is not

It **is** a provenance record and a visual reference for the corpus: it lets a reader see
what kind of object the pipeline was built for, and it documents where each image came
from.

It **is not** a runnable benchmark set:

* **No masks are supplied.** The pipeline is an inpainting system; every entry point
  (`train.py`, `infer.py`, `evaluate.py`) needs an image *and* a mask. Without masks
  nothing here can be fed to the code as-is.
* The article's damage masks are derived annotations and are not redistributed.
* The 51-pair evaluation split is likewise not distributed — `evaluate.py` expects you
  to build your own manifest with the columns `sample_id,image_path,mask_path` (see
  `scripts/build_dataset_manifest.py`).

To exercise the code you therefore need your own image/mask pairs. To use these
photographs as *inputs* you would have to annotate masks yourself; that is a legitimate
way to try the method on museum imagery, but note that any metric computed against these
images would not be comparable with the article's numbers.

## Adding more images

If images are added to `data/public_subset/`, keep the following consistent in one
commit — a mismatch between them is what makes a data-availability statement
unverifiable:

1. `README.md` — the inclusion table and the image count.
2. `LICENSE` — the rights basis and licence for each folder. **Never assert a Creative
   Commons licence over material whose copyright holder has not granted it.**
3. `SOURCES.csv` — one row per image, including `licence` and `credit_line`.
4. `CREDITS.md` — the credit line a reuser must reproduce.
5. `CITATION.cff` — the `license` field, if the set of licences changes.

Then audit the metadata before committing:

```bash
python tools/check_image_metadata.py --dir data/public_subset
```

That reports GPS and other EXIF tags per file. **GPS tags must be removed** — precise
coordinates of archaeological sites are sensitive:

```bash
python tools/check_image_metadata.py --dir data/public_subset --strip-gps
```

The stripper edits the JPEG's APP1/TIFF structure at byte level, so it is **lossless** —
it does not re-encode, and pixel data is unchanged. It also zeroes the byte ranges the
GPS entries point at, rather than merely disconnecting the pointer. This is covered by a
regression test that builds a GPS-bearing JPEG as a fixture:

```bash
python tools/test_check_image_metadata.py
```

Camera model and capture date may be retained where they document the recording of an
object, but only deliberately.
