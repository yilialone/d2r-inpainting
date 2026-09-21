#!/usr/bin/env python3
"""Fixed-checkpoint, paired object-level percentile bootstrap. Standard library only."""
import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

DIRECTIONS = {"psnr": 1, "ssim": 1, "lpips": -1,
              "psnr_mask": 1, "ssim_mask": 1, "lpips_mask": -1}


def read_rows(path, required):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError(f"{path}: missing/duplicate column names")
        missing = set(required) - set(reader.fieldnames)
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: empty file")
    if any(None in r for r in rows):
        raise ValueError(f"{path}: malformed CSV row")
    return rows


def index_rows(path, key_columns, metrics=()):
    result = {}
    for number, row in enumerate(read_rows(path, [*key_columns, *metrics]), 2):
        key = tuple((row.get(k) or "").strip() for k in key_columns)
        if not all(key):
            raise ValueError(f"{path}:{number}: empty pairing identifier")
        if key in result:
            raise ValueError(f"{path}:{number}: duplicate pair {key}; do not pool runs")
        for metric in metrics:
            try:
                value = float(row[metric])
            except (TypeError, ValueError):
                raise ValueError(f"{path}:{number}: invalid {metric}") from None
            if not math.isfinite(value):
                raise ValueError(f"{path}:{number}: nonfinite {metric}; no silent exclusion")
            row[metric] = value
        result[key] = row
    return result


def require_same_keys(reference, other, label):
    missing, extra = set(reference) - set(other), set(other) - set(reference)
    if missing or extra:
        raise ValueError(f"{label}: pair mismatch; missing={len(missing)}, extra={len(extra)}; "
                         f"examples missing={sorted(missing)[:3]}, extra={sorted(extra)[:3]}")


def quantile(sorted_values, probability):
    position = (len(sorted_values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def bootstrap_mean(values, repeats, confidence, seed):
    if len(values) < 2:
        raise ValueError("At least two independent objects are needed")
    rng = random.Random(seed)
    n = len(values)
    draws = sorted(statistics.fmean(values[rng.randrange(n)] for _ in range(n))
                   for _ in range(repeats))
    alpha = (1 - confidence) / 2
    return quantile(draws, alpha), quantile(draws, 1 - alpha)


def object_pairs(target, baseline, groups, metrics):
    collected = defaultdict(list)
    for key in sorted(target):
        collected[groups[key]].append(key)
    rows = []
    for obj, keys in sorted(collected.items()):
        for metric in metrics:
            t = statistics.fmean(target[k][metric] for k in keys)
            b = statistics.fmean(baseline[k][metric] for k in keys)
            rows.append({"object_id": obj, "metric": metric, "n_image_mask_pairs": len(keys),
                         "target_mean": t, "baseline_mean": b, "delta_target_minus_baseline": t-b})
    return rows


def write_csv(path, rows):
    with Path(path).open("x", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, help="Target per_image_metrics.csv")
    parser.add_argument("--target-name", default="D2R")
    parser.add_argument("--baseline", action="append", default=[], metavar="NAME=CSV")
    parser.add_argument("--key-columns", nargs="+", default=["image_num", "mask_name"])
    parser.add_argument("--metrics", nargs="+", choices=sorted(DIRECTIONS),
                        default=["psnr_mask", "ssim_mask", "lpips_mask"])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--object-map", help="CSV with pairing identifiers and object_id")
    group.add_argument("--assume-independent-images", action="store_true",
                       help="Explicit assertion: each image ID is a different object; masks stay grouped")
    parser.add_argument("--image-column", default="image_num")
    parser.add_argument("--make-object-template", metavar="NEW_CSV",
                        help="Create blank object_id mapping; no statistical analysis")
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--confidence", type=float, default=.95)
    parser.add_argument("--resampling-seed", type=int, default=20260907,
                        help="Only bootstrap RNG; NOT a training seed")
    parser.add_argument("--output-dir", help="New directory; existing directory is never overwritten")
    parser.add_argument("--protocol-note", default="Protocol/checkpoint equivalence must be verified by the author.")
    args = parser.parse_args(argv)
    if len(set(args.key_columns)) != len(args.key_columns) or len(set(args.metrics)) != len(args.metrics):
        raise ValueError("Duplicate key columns or metrics")
    if args.make_object_template:
        if "object_id" in args.key_columns:
            raise ValueError("object_id must not also be a pairing key in the template")
        target = index_rows(args.target, args.key_columns)
        rows = [dict(zip(args.key_columns, key), object_id="") for key in sorted(target)]
        write_csv(args.make_object_template, rows)
        print("Template created. Fill true object IDs; blank IDs are not inferred.")
        return
    if args.resamples < 1000 or not 0 < args.confidence < 1:
        raise ValueError("Use resamples >= 1000 and 0 < confidence < 1")
    if not args.baseline or not args.output_dir:
        raise ValueError("Analysis requires --baseline NAME=CSV and --output-dir")
    if not args.object_map and not args.assume_independent_images:
        raise ValueError("Provide --object-map, or explicitly assert --assume-independent-images")
    target = index_rows(args.target, args.key_columns, args.metrics)
    inputs = {"target": {"name": args.target_name, "path": str(Path(args.target).resolve()),
                         "sha256": digest(args.target)}}
    groups = {}
    if args.object_map:
        mapping = index_rows(args.object_map, args.key_columns)
        require_same_keys(target, mapping, "object map")
        for key, row in mapping.items():
            obj = (row.get("object_id") or "").strip()
            if not obj:
                raise ValueError(f"Missing object_id for {key}")
            groups[key] = obj
        inputs["object_map"] = {"path": str(Path(args.object_map).resolve()), "sha256": digest(args.object_map)}
    else:
        if args.image_column not in args.key_columns:
            raise ValueError("--image-column must be included in --key-columns")
        idx = args.key_columns.index(args.image_column)
        groups = {key: key[idx] for key in target}
    n_objects = len(set(groups.values()))
    if n_objects < 2:
        raise ValueError("Fewer than two objects: no meaningful object bootstrap")
    warnings = []
    if n_objects < 20:
        warnings.append("Fewer than 20 objects: small-sample percentile intervals may be unreliable; 20 is only a warning heuristic.")
    if args.assume_independent_images:
        warnings.append("Author asserted one distinct object per image ID; this was NOT verified by the script.")
    summary, objects, pairs, seen = [], [], [], set()
    for spec in args.baseline:
        if "=" not in spec:
            raise ValueError("Use --baseline NAME=CSV")
        name, path = spec.split("=", 1)
        if not name or not path or name in seen or name == args.target_name or name in {"target", "object_map"}:
            raise ValueError("Baseline names must be nonempty, unique and different from target")
        seen.add(name)
        baseline = index_rows(path, args.key_columns, args.metrics)
        require_same_keys(target, baseline, name)
        inputs[name] = {"path": str(Path(path).resolve()), "sha256": digest(path)}
        obj_rows = object_pairs(target, baseline, groups, args.metrics)
        objects.extend(dict(baseline=name, **r) for r in obj_rows)
        for key in sorted(target):
            for metric in args.metrics:
                pairs.append(dict(zip(args.key_columns, key), object_id=groups[key], baseline=name,
                                  metric=metric, target_value=target[key][metric],
                                  baseline_value=baseline[key][metric],
                                  delta_target_minus_baseline=target[key][metric]-baseline[key][metric]))
        for metric in args.metrics:
            selected = [r for r in obj_rows if r["metric"] == metric]
            values = [r["delta_target_minus_baseline"] for r in selected]
            low, high = bootstrap_mean(values, args.resamples, args.confidence, args.resampling_seed)
            direction = DIRECTIONS[metric]
            improvement_low, improvement_high = sorted((direction*low, direction*high))
            if low == high:
                warnings.append(f"{name}/{metric}: degenerate interval; identical observed differences do not establish population certainty.")
            summary.append({"target": args.target_name, "baseline": name, "metric": metric,
                            "better": "higher" if direction == 1 else "lower",
                            "n_image_mask_pairs": len(target), "n_objects": n_objects,
                            "target_object_mean": statistics.fmean(r["target_mean"] for r in selected),
                            "baseline_object_mean": statistics.fmean(r["baseline_mean"] for r in selected),
                            "delta_target_minus_baseline": statistics.fmean(values),
                            "ci_low": low, "ci_high": high,
                            "improvement_positive_is_better": direction*statistics.fmean(values),
                            "improvement_ci_low": improvement_low, "improvement_ci_high": improvement_high,
                            "confidence": args.confidence, "ci_method": "paired_object_percentile_bootstrap",
                            "scope": "fixed_checkpoints_fixed_split_not_training_variability"})
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    write_csv(out/"paired_summary.csv", summary)
    write_csv(out/"object_differences.csv", objects)
    write_csv(out/"paired_image_metrics.csv", pairs)
    metadata = {"arguments": vars(args), "inputs": inputs, "warnings": warnings,
                "python": sys.version, "script_sha256": digest(__file__),
                "estimand": "Equal-object mean of within-object mean paired image-mask differences",
                "scope": "Conditional on fixed checkpoints and test split; no training-seed uncertainty",
                "multiplicity": "Pointwise intervals only; no p values or simultaneous coverage",
                "note": "CSV IDs cannot establish identical masks, preprocessing or checkpoint validity."}
    (out/"analysis_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Fixed-checkpoint paired uncertainty", "",
             "Equal-object weighting; pointwise percentile intervals. No training-seed variation or p values.", "",
             "| Baseline | Metric | Objects | Target minus baseline | CI low | CI high |",
             "|---|---|---:|---:|---:|---:|"]
    for r in summary:
        safe_name = r['baseline'].replace('|', '/').replace('\n', ' ')
        lines.append(f"| {safe_name} | {r['metric']} | {n_objects} | {r['delta_target_minus_baseline']:.6g} | {r['ci_low']:.6g} | {r['ci_high']:.6g} |")
    lines.extend(["", "PSNR/SSIM: positive difference favours target. LPIPS: negative difference favours target.",
                  "An interval spanning zero does not establish equivalence. An interval excluding zero is not a multiplicity-adjusted significance claim.",
                  "", "## Warnings", ""] + ["- "+w for w in warnings])
    (out/"report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(f"Done: {len(target)} paired cases, {n_objects} objects. Results: {out.resolve()}")
    for warning in warnings:
        print("WARNING:", warning)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
