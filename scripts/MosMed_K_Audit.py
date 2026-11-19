import argparse
import os
import glob
import json
from collections import Counter

import numpy as np
import pandas as pd

from PIL import Image
from scipy import ndimage

TARGET_SIZE = (224, 224)

RESIZE_INTERPOLATION = "nearest"

CONNECTIVITY = 2
CONNECTIVITY_NAME = "8-connectivity"

MOSMED_MIN_AREA = 15

K_REPORT_LIST = [2, 3, 4, 6, 7, 8, 10, 16]

IMAGE_COLUMN = "Image"

# ---------------------------------------------------------------------------
# Paths (command-line arguments; defaults assume the repository data layout
# described in README.md and that the script is run from the repository root).
# ---------------------------------------------------------------------------
_ap = argparse.ArgumentParser(
    description="MosMedData+ train-based region-count (K) audit.")
_ap.add_argument("--frames_root", default="data/MosMedData/frames",
                 help="Folder with the MosMedData+ CT frames.")
_ap.add_argument("--mask_root", default="data/MosMedData/masks",
                 help="Folder with the MosMedData+ lesion masks.")
_ap.add_argument("--train_csv", default="splits/mosmed/Train_resplit_by_group.csv",
                 help="Final TRAIN split manifest.")
_ap.add_argument("--val_csv", default="splits/mosmed/Val_resplit_by_group.csv",
                 help="Final VALIDATION split manifest.")
_ap.add_argument("--output_dir", default="outputs/mosmed_region_k_audit",
                 help="Folder where the audit tables and metadata are written.")
_cli, _ = _ap.parse_known_args()

FRAMES_ROOT = _cli.frames_root
MASK_ROOT = _cli.mask_root
TRAIN_CSV_PATH = _cli.train_csv
VAL_CSV_PATH = _cli.val_csv
OUTPUT_DIR = _cli.output_dir

_RESAMPLING = getattr(Image, "Resampling", Image)
RESAMPLE_MODES = {
    "nearest": _RESAMPLING.NEAREST,
    "bilinear": _RESAMPLING.BILINEAR,
    "bicubic": _RESAMPLING.BICUBIC,
    "box": _RESAMPLING.BOX,
}
if RESIZE_INTERPOLATION not in RESAMPLE_MODES:
    raise ValueError(f"Unknown RESIZE_INTERPOLATION: {RESIZE_INTERPOLATION}")


def normalize_image_name(name):
    if pd.isna(name):
        return None
    name = str(name).strip().replace("\\", "/")
    return os.path.basename(name)


def index_folder(root, label):
    patterns = ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp"]
    paths = []
    for pattern in patterns:
        paths.extend(glob.glob(os.path.join(root, "**", pattern), recursive=True))
    paths = sorted(set(paths))

    by_name, by_stem, duplicates = {}, {}, 0
    for path in paths:
        name = os.path.basename(path)
        if name in by_name:
            duplicates += 1
            continue
        by_name[name] = path
        by_stem.setdefault(os.path.splitext(name)[0], []).append(path)

    print(f"{label}: {len(paths)} files | {len(by_name)} unique names "
          f"| {duplicates} duplicate names")
    return by_name, by_stem


def find_path(image_name, by_name, by_stem):
    path = by_name.get(image_name)
    if path is not None:
        return path
    candidates = by_stem.get(os.path.splitext(image_name)[0], [])
    return candidates[0] if len(candidates) == 1 else None


def load_mask(mask_path):
    with Image.open(mask_path) as im:
        native = np.array(im.convert("L"))

    native_hw = native.shape
    binary_native = ((native > 0).astype(np.uint8)) * 255

    resized = Image.fromarray(binary_native).resize(
        TARGET_SIZE, resample=RESAMPLE_MODES[RESIZE_INTERPOLATION]
    )
    binary = np.array(resized) >= 128
    return binary, native_hw


def extract_regions(binary_mask, min_area, connectivity):
    structure = ndimage.generate_binary_structure(2, connectivity)
    labeled, n_raw = ndimage.label(binary_mask, structure=structure)
    if n_raw == 0:
        return 0, []
    sizes = np.bincount(labeled.ravel())[1:]
    valid = np.sort(sizes[sizes >= min_area])
    return int(n_raw), [int(v) for v in valid]


def coverage_for_k(region_counts, k):
    region_counts = np.asarray(region_counts)
    if len(region_counts) == 0:
        return np.nan
    return float(np.mean(region_counts <= k))


def find_k_for_coverage(region_counts, target):
    region_counts = np.asarray(region_counts)
    max_k = int(region_counts.max())
    for k in range(1, max_k + 1):
        cov = coverage_for_k(region_counts, k)
        if cov >= target:
            return k, cov
    return max_k, coverage_for_k(region_counts, max_k)


def k_capacity_report(result_df, k_values, split_name):
    n_images = len(result_df)
    counts = result_df["region_count"].to_numpy()
    total_fg = float(result_df["total_foreground_pixels"].sum())
    region_fg_per_image = result_df["valid_foreground_pixels"].to_numpy(dtype=float)
    region_fg = float(region_fg_per_image.sum())
    total_regions = int(counts.sum())

    areas_desc = [
        np.sort(np.asarray(a, dtype=np.int64))[::-1]
        for a in result_df["valid_component_areas"]
    ]
    has_region = region_fg_per_image > 0

    rows = []
    for k in k_values:
        upper = np.array([a[:k].sum() for a in areas_desc], dtype=float)
        lower = np.array([a[::-1][:k].sum() for a in areas_desc], dtype=float)

        overflow_images = int(np.sum(counts > k))
        beyond = int(np.sum(np.maximum(counts - k, 0)))

        rows.append({
            "split": split_name,
            "K": int(k),
            "n_images": n_images,
            "sample_coverage_pct": 100.0 * coverage_for_k(counts, k),
            "images_with_overflow": overflow_images,
            "images_with_overflow_pct": 100.0 * overflow_images / n_images,
            "regions_total": total_regions,
            "regions_beyond_capacity": beyond,
            "regions_beyond_capacity_pct": (
                100.0 * beyond / total_regions if total_regions else np.nan
            ),

            "fg_covered_upper_pct_of_total_fg": 100.0 * upper.sum() / total_fg,
            "fg_covered_lower_pct_of_total_fg": 100.0 * lower.sum() / total_fg,

            "fg_in_unmatched_regions_min_pct_of_total_fg":
                100.0 * (region_fg - upper.sum()) / total_fg,
            "fg_in_unmatched_regions_max_pct_of_total_fg":
                100.0 * (region_fg - lower.sum()) / total_fg,

            "fg_covered_upper_pct_of_region_fg": 100.0 * upper.sum() / region_fg,
            "fg_covered_lower_pct_of_region_fg": 100.0 * lower.sum() / region_fg,

            "fg_covered_upper_per_image_mean_pct": float(
                100.0 * np.mean(upper[has_region] / region_fg_per_image[has_region])
            ),

            "fg_outside_valid_regions_pct_of_total_fg":
                100.0 * (total_fg - region_fg) / total_fg,
        })

    return pd.DataFrame(rows)

print("=" * 80)
print("INDEXING")
print("=" * 80)

frame_by_name, frame_by_stem = index_folder(FRAMES_ROOT, "Frames")
mask_by_name, mask_by_stem = index_folder(MASK_ROOT, "Masks")


def load_split_csv(csv_path, split_name):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if IMAGE_COLUMN not in df.columns:
        raise ValueError(
            f"Column '{IMAGE_COLUMN}' not found in {csv_path}. "
            f"Available columns: {list(df.columns)}"
        )

    df = df.copy()
    df["image_name"] = df[IMAGE_COLUMN].apply(normalize_image_name)
    df = df[df["image_name"].notna()].drop_duplicates("image_name").reset_index(drop=True)
    df["split"] = split_name
    print(f"{split_name}: {len(df)} unique images in CSV")
    return df


def match_split_images(split_df):
    records = []
    for row in split_df.itertuples(index=False):
        frame_path = find_path(row.image_name, frame_by_name, frame_by_stem)
        mask_path = find_path(row.image_name, mask_by_name, mask_by_stem)
        records.append({
            "split": row.split,
            "image_name": row.image_name,
            "frame_path": frame_path,
            "mask_path": mask_path,
            "frame_found": frame_path is not None,
            "mask_found": mask_path is not None,
        })
    return pd.DataFrame(records)

print()
train_csv = load_split_csv(TRAIN_CSV_PATH, "train")
val_csv = load_split_csv(VAL_CSV_PATH, "validation")

train_matches = match_split_images(train_csv)
val_matches = match_split_images(val_csv)

for name, m in (("TRAIN", train_matches), ("VALIDATION", val_matches)):
    print(f"{name}: frames found {int(m.frame_found.sum())}/{len(m)} | "
          f"masks found {int(m.mask_found.sum())}/{len(m)}")

overlap = set(train_csv["image_name"]) & set(val_csv["image_name"])
print(f"Train/Validation image-name overlap: {len(overlap)}")
if overlap:
    print("WARNING: overlapping image names:", list(overlap)[:20])


def audit_split(matched_df, split_name):
    valid_df = matched_df[matched_df["mask_found"]]

    print()
    print("=" * 80)
    print(f"MOSMEDDATA+ | {split_name.upper()}")
    print("=" * 80)
    print(f"Images in CSV           : {len(matched_df)}")
    print(f"Images with a mask      : {len(valid_df)}")
    print(f"Analysis resolution     : {TARGET_SIZE[0]} x {TARGET_SIZE[1]} "
          f"(interpolation: {RESIZE_INTERPOLATION})")
    print(f"Connectivity            : {CONNECTIVITY_NAME}")
    print(f"Minimum component area  : {MOSMED_MIN_AREA} px (at analysis resolution)")

    records, native_shapes = [], Counter()
    for row in valid_df.itertuples(index=False):
        try:
            binary, native_hw = load_mask(row.mask_path)
            n_raw, areas = extract_regions(binary, MOSMED_MIN_AREA, CONNECTIVITY)
        except Exception as exc:
            print(f"ERROR processing {row.image_name}: {exc}")
            continue

        native_shapes[native_hw] += 1
        records.append({
            "split": split_name,
            "image_name": row.image_name,
            "native_height": native_hw[0],
            "native_width": native_hw[1],
            "raw_component_count": n_raw,
            "region_count": len(areas),
            "removed_small_components": n_raw - len(areas),
            "total_foreground_pixels": int(binary.sum()),
            "valid_foreground_pixels": int(sum(areas)),
            "valid_component_areas": areas,
        })

    result_df = pd.DataFrame(records)
    if result_df.empty:
        print("No valid masks were processed.")
        return None

    print(f"Native mask shapes (H, W): {dict(native_shapes)}")

    counts = result_df["region_count"].to_numpy()
    n = len(counts)

    stats = {
        "samples": n,
        "mean": float(np.mean(counts)),
        "median": float(np.median(counts)),
        "std": float(np.std(counts, ddof=1)) if n > 1 else 0.0,
        "min": int(counts.min()),
        "Q1": float(np.percentile(counts, 25)),
        "Q3": float(np.percentile(counts, 75)),
        "P90": float(np.percentile(counts, 90)),
        "P95": float(np.percentile(counts, 95)),
        "P99": float(np.percentile(counts, 99)),
        "max": int(counts.max()),
        "zero_regions": int(np.sum(counts == 0)),
        "single_region": int(np.sum(counts == 1)),
        "multi_region": int(np.sum(counts >= 2)),
    }
    for target, tag in ((0.90, "K90"), (0.95, "K95"), (0.99, "K99")):
        k, cov = find_k_for_coverage(counts, target)
        stats[tag] = int(k)
        stats[f"Coverage_{tag}"] = float(cov)

    print()
    print("--- Region count statistics (after filtering) ---")
    for key in ("samples", "mean", "median", "std", "min", "Q1", "Q3",
                "P90", "P95", "P99", "max"):
        print(f"{key:<10}: {stats[key]}")
    print(f"0 region  : {stats['zero_regions']} ({100*stats['zero_regions']/n:.3f}%)")
    print(f"1 region  : {stats['single_region']} ({100*stats['single_region']/n:.3f}%)")
    print(f">=2 region: {stats['multi_region']} ({100*stats['multi_region']/n:.3f}%)")
    print(f"K90={stats['K90']} ({100*stats['Coverage_K90']:.3f}%) | "
          f"K95={stats['K95']} ({100*stats['Coverage_K95']:.3f}%) | "
          f"K99={stats['K99']} ({100*stats['Coverage_K99']:.3f}%)")

    distribution_df = (
        result_df["region_count"].value_counts().sort_index()
        .rename_axis("region_count").reset_index(name="frequency")
    )
    distribution_df["percentage"] = 100.0 * distribution_df["frequency"] / n

    coverage_df = pd.DataFrame({
        "K": range(1, stats["max"] + 1),
    })
    coverage_df["covered_samples"] = [int(np.sum(counts <= k)) for k in coverage_df["K"]]
    coverage_df["total_samples"] = n
    coverage_df["coverage_percent"] = 100.0 * coverage_df["covered_samples"] / n

    report_ks = sorted({int(k) for k in K_REPORT_LIST + [stats["K90"], stats["K95"], stats["K99"]] if k >= 1})
    k_report_df = k_capacity_report(result_df, report_ks, split_name)

    print()
    print("--- K capacity report ---")
    print("FG upper = K largest regions matched (Top-K by area); "
          "FG lower = K smallest regions matched.")
    print("The Hungarian baseline lies between the two bounds "
          "(measured during training).")
    show = k_report_df[[
        "K", "sample_coverage_pct", "images_with_overflow_pct",
        "fg_covered_lower_pct_of_total_fg", "fg_covered_upper_pct_of_total_fg",
    ]].rename(columns={
        "sample_coverage_pct": "samples_covered_%",
        "images_with_overflow_pct": "overflow_images_%",
        "fg_covered_lower_pct_of_total_fg": "FG_covered_lower_%",
        "fg_covered_upper_pct_of_total_fg": "FG_covered_upper_%",
    })
    print(show.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
    print(f"Foreground removed by the area filter itself (all K): "
          f"{k_report_df['fg_outside_valid_regions_pct_of_total_fg'].iloc[0]:.3f}% of total foreground")

    return {
        "df": result_df,
        "stats": stats,
        "distribution": distribution_df,
        "coverage": coverage_df,
        "k_report": k_report_df,
        "native_shapes": native_shapes,
    }

train_audit = audit_split(train_matches, "train")
val_audit = audit_split(val_matches, "validation")

print()
print("#" * 90)
print("# MOSMEDDATA+ | TRAIN-BASED K SELECTION")
print("#" * 90)

if train_audit is not None:
    s = train_audit["stats"]
    print("The values below use the TRAIN split only.")
    print(f"P90={s['P90']:.3f} | P95={s['P95']:.3f} | P99={s['P99']:.3f}")
    print(f"K90={s['K90']} | K95={s['K95']} | K99={s['K99']}")
    print(f"Starting point for the Sensitivity Analysis: K = K95 = {s['K95']} "
          f"(TRAIN coverage {100*s['Coverage_K95']:.3f}%). This is a starting "
          f"point, not the final value.")

    if val_audit is not None:
        k95 = s["K95"]
        val_cov = coverage_for_k(val_audit["df"]["region_count"].to_numpy(), k95)
        print(f"Validation check: the TRAIN-selected K95={k95} covers "
              f"{100*val_cov:.3f}% of Validation images "
              f"(Validation is NOT used to choose K).")
else:
    print("TRAIN audit failed or no valid masks were found.")

dominant_shape = None
if train_audit is not None and train_audit["native_shapes"]:
    dominant_shape = train_audit["native_shapes"].most_common(1)[0][0]
    scale = (dominant_shape[0] * dominant_shape[1]) / float(TARGET_SIZE[0] * TARGET_SIZE[1])
    print()
    print(f"A {MOSMED_MIN_AREA}-px component at {TARGET_SIZE[0]}x{TARGET_SIZE[1]} corresponds to "
          f"about {MOSMED_MIN_AREA * scale:.1f} px at the dominant native shape "
          f"{dominant_shape[0]}x{dominant_shape[1]}.")

os.makedirs(OUTPUT_DIR, exist_ok=True)

summary_rows = []
for audit in (train_audit, val_audit):
    if audit is None:
        continue
    s = audit["stats"]
    summary_rows.append({
        "Dataset": "MosMedData+",
        "Split": audit["df"]["split"].iloc[0],
        "N": s["samples"], "Mean": s["mean"], "Median": s["median"],
        "Std": s["std"], "Min": s["min"], "Q1": s["Q1"], "Q3": s["Q3"],
        "P90": s["P90"], "P95": s["P95"], "P99": s["P99"], "Max": s["max"],
        "0_Region": s["zero_regions"], "1_Region": s["single_region"],
        ">=2_Regions": s["multi_region"],
        "K90": s["K90"], "Coverage_K90_%": 100 * s["Coverage_K90"],
        "K95": s["K95"], "Coverage_K95_%": 100 * s["Coverage_K95"],
        "K99": s["K99"], "Coverage_K99_%": 100 * s["Coverage_K99"],
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(OUTPUT_DIR, "MosMed_K_Audit_Summary.csv"), index=False)

train_matches.to_csv(os.path.join(OUTPUT_DIR, "MosMed_Train_Matching.csv"), index=False)
val_matches.to_csv(os.path.join(OUTPUT_DIR, "MosMed_Validation_Matching.csv"), index=False)

for tag, audit in (("Train", train_audit), ("Validation", val_audit)):
    if audit is None:
        continue
    per_image = audit["df"].copy()
    per_image["valid_component_areas"] = per_image["valid_component_areas"].apply(json.dumps)
    per_image.to_csv(os.path.join(OUTPUT_DIR, f"MosMed_{tag}_Per_Image.csv"), index=False)
    audit["distribution"].to_csv(
        os.path.join(OUTPUT_DIR, f"MosMed_{tag}_Region_Count_Distribution.csv"), index=False)
    audit["coverage"].to_csv(
        os.path.join(OUTPUT_DIR, f"MosMed_{tag}_Coverage_K.csv"), index=False)
    audit["k_report"].to_csv(
        os.path.join(OUTPUT_DIR, f"MosMed_{tag}_K_Capacity_Report.csv"), index=False)

metadata = {
    "analysis_resolution_wh": list(TARGET_SIZE),
    "resize_interpolation": RESIZE_INTERPOLATION,
    "binarization": "native: pixel > 0; after resize: >= 128",
    "connectivity": CONNECTIVITY_NAME,
    "scipy_connectivity_parameter": CONNECTIVITY,
    "component_area_definition": "number of foreground pixels of the component",
    "min_component_area_px_at_analysis_resolution": MOSMED_MIN_AREA,
    "k_report_list": K_REPORT_LIST,
    "native_mask_shapes_train": (
        {f"{h}x{w}": c for (h, w), c in train_audit["native_shapes"].items()}
        if train_audit else {}
    ),
    "foreground_coverage_definition": {
        "upper": "K largest regions of every image matched (Top-K by area)",
        "lower": "K smallest regions of every image matched",
        "note": "the Hungarian baseline lies between the two bounds",
    },
    "test_data_used": False,
}
with open(os.path.join(OUTPUT_DIR, "MosMed_K_Audit_Metadata.json"), "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print()
print("#" * 90)
print("# RESULT FILES")
print("#" * 90)
print(f"Output directory: {OUTPUT_DIR}")
for fname in sorted(os.listdir(OUTPUT_DIR)):
    print(" -", fname)

print()
print("OK  Test data was NOT used.")
print("OK  Every mask was analysed at "
      f"{TARGET_SIZE[0]}x{TARGET_SIZE[1]} with a {MOSMED_MIN_AREA}-px area threshold.")
print("OK  Foreground coverage is reported as an upper and a lower bound; "
      "the Hungarian baseline lies between them.")
print("OK  Per-image component areas are saved, so all percentages can be recomputed.")
