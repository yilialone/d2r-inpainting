# Public image subset — Ancient Chinese mountain-pattern (*shan*) bronze mirrors

Image subset released with the study "Two-Stage Structural Reconstruction and Texture
Refinement for Digital Restoration of Ancient Chinese Mountain-Pattern Bronze Mirror
Photographs" (manuscript ID `a062e16c-271f-4765-88b0-c2ef3985d848`).

* **13 images** covering **11 objects**, ~2.9 MB
* JPEG
* Per-image provenance and credit lines: [`SOURCES.csv`](SOURCES.csv), [`CREDITS.md`](CREDITS.md)
* **Licence: CC0 1.0 Universal** for every file in this directory (see [`LICENSE`](LICENSE))

## What is included, and why

Only images whose rights status is unambiguous *and* whose licence permits
redistribution without a permission request are released here. All 13 files come from
museum open-access programmes:

| Folder | Images | Objects | Source | Rights basis |
|---|---|---|---|---|
| `02_museum_Cleveland_CC0/` | 3 | 2 | The Cleveland Museum of Art | **CC0 1.0 Universal** (Open Access API: `share_license_status = CC0`) |
| `03_museum_Harvard_no_permission_required/` | 4 | 3 | Harvard Art Museums | API reports `imagepermissionlevel = 0`, i.e. no permission required; objects are Warring States period, public domain |
| `04_museum_NMAA_Smithsonian_CC0/` | 6 | 6 | National Museum of Asian Art, Smithsonian Institution | **CC0** under the museum's Image Services policy |

The file count exceeds the object count because two objects are represented by two
photographs from different viewpoints — `1995.281` (Cleveland, `02-02`/`02-03`) and
`1943.52.145` (Harvard, `03-02`/`03-03`). They are distinct photographs, not duplicates.
If you need one image per object, keep either one of each pair.

CC0 material carries no attribution requirement; credit lines are nevertheless given in
`CREDITS.md` so that provenance is preserved.

### These are images, not a benchmark set

**No masks are supplied with this subset.** It documents the visual domain and the
provenance of the corpus; it is *not* a runnable evaluation set. `evaluate.py` in the
repository root requires image/mask pairs and cannot be run on this directory. The
damage masks used in the article are derived annotations and are not redistributed.

## This set will grow

This is a **living collection**, not a frozen release. Only material whose rights status
is unambiguous is included here, and **further images will be added as additional
permissions are obtained** from the holding institutions and from the recording staff.

Two practical consequences:

* The file list will change between versions. The `version` field in
  [`CITATION.cff`](CITATION.cff) is incremented when it does. If you need reproducible
  inputs, pin a commit and record the exact file list you used rather than tracking
  `main`.
* No date is promised for any particular addition. The constraint is the rights
  position, not editorial effort — an image appears here only once its licence is
  settled and documented, never before.

Each addition must keep `README.md`, `LICENSE`, `CREDITS.md`, `SOURCES.csv` and
`CITATION.cff` in this directory mutually consistent, and must pass

```bash
python tools/check_image_metadata.py --dir data/public_subset
```

See [`../../docs/STATUS.md`](../../docs/STATUS.md) §5 for the full policy.

## What is deliberately excluded

### 1. Author field photographs (not currently included)

Ten photographs taken at excavation sites and in stores are **not included at this
time**, because their rights status is not yet settled. They will be added once it is,
at which point `README.md`, `LICENSE`, `CREDITS.md`, `SOURCES.csv` and `CITATION.cff`
in this directory must be updated together and consistently.

### 2. Photographs taken by the authors inside museum galleries

The copyright in a photograph belongs to the person who pressed the shutter, but museum
conditions of entry normally restrict such photographs to personal, non-commercial use
and do not cover redistribution. The photographs concerned are used in the article only.

### 3. Photographs taken from museum websites or from excavation reports and catalogues

These remain subject to the copyright and access policies of the holding institutions
and publishers. Several Chinese museums, including the Palace Museum, the National
Museum of China and the Shanghai Museum, require a written application before their
images may be reproduced, and the Palace Museum additionally prohibits the creation of a
similar database of its content. These images are used in the article for scholarly
citation, comparison and criticism, with the holding institution identified, and are not
redistributed.

The remaining photographs of the full study corpus are likewise not redistributed, for
the same reasons.

## Metadata

The images in this directory carry **no EXIF metadata** as distributed: no GPS
coordinates, no camera identifiers, no capture dates. This is deliberate — image
metadata is an easily overlooked disclosure channel, and precise coordinates of
archaeological sites are sensitive. `tools/check_image_metadata.py` in the repository
root audits a directory for GPS and other EXIF tags and can strip them. The stripper
edits the JPEG's APP1/TIFF structure at byte level, so it is lossless (no re-encoding,
pixels unchanged) and it zeroes the byte ranges the GPS entries point at rather than
merely disconnecting the pointer; `tools/test_check_image_metadata.py` covers this with
a synthetic fixture.

If author field photographs are added later, their camera model and capture date are
intended to be retained (they document when each object was recorded), while GPS tags
must be removed. Use:

```bash
python tools/check_image_metadata.py --dir data/public_subset --strip-gps
```

## Directory layout

```
.
├── 02_museum_Cleveland_CC0/                  3 images / 2 objects
├── 03_museum_Harvard_no_permission_required/ 4 images / 3 objects
├── 04_museum_NMAA_Smithsonian_CC0/           6 images / 6 objects
├── SOURCES.csv      per-image source table (13 rows)
├── CREDITS.md       credit lines to reproduce (13 lines)
├── LICENSE          CC0 1.0 Universal
├── CITATION.cff
└── README.md
```

## Citation

Please cite the article when using this subset. See [`CITATION.cff`](CITATION.cff).
