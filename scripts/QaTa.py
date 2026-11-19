import argparse
import os
import re
import ast
import json
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from skimage.measure import label, regionprops

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Paths (command-line arguments; defaults assume the repository data layout
# described in README.md and that the script is run from the repository root).
# ---------------------------------------------------------------------------
_ap = argparse.ArgumentParser(
    description="QaTa-COV19 data audit (component statistics, threshold selection, text/mask agreement).")
_ap.add_argument("--dataset_root", default="data/QaTa-COV19/QaTa-COV19-v2",
                 help="QaTa-COV19-v2 root folder (must contain 'Train Set/Ground-truths').")
_ap.add_argument("--text_file", default="data/QaTa-COV19/text/Train_text_for_Covid19(Sheet1).csv",
                 help="QaTa-COV19 text annotation CSV.")
_ap.add_argument("--train_id_file", default="splits/qata/Train_ID(sheet1).csv",
                 help="Final TRAIN split manifest (image IDs).")
_ap.add_argument("--output_dir", default="outputs/QaTa_Data_Audit_Final_224",
                 help="Folder where the audit tables, figures and metadata are written.")
_cli, _ = _ap.parse_known_args()

DATASET_ROOT = _cli.dataset_root

MASK_DIR = os.path.join(
    DATASET_ROOT,
    "Train Set",
    "Ground-truths"
)

TEXT_FILE = _cli.text_file

TRAIN_ID_FILE = _cli.train_id_file

OUTPUT_DIR = _cli.output_dir

TEXT_IMAGE_COLUMN_OVERRIDE = None
TEXT_COLUMN_OVERRIDE = None

RESIZE_TO = 224
RESIZE_INTERPOLATION = cv2.INTER_NEAREST

SENS_INCLUDE_NATIVE = True

CONNECTIVITY = 2
CONNECTIVITY_NAME = "8-connectivity"

THRESHOLDS = sorted(set([
    1, 2, 3, 4, 5, 6, 8, 10,
    15, 20, 25, 30, 35, 40, 45, 50,
    53, 55, 60, 65, 70, 75, 80, 90, 100,
    125, 150, 200, 250, 300, 400, 500
]))

STABILITY_TOLERANCES = [0.005, 0.01, 0.02]
PRIMARY_STABILITY_TOLERANCE = 0.01
MIN_PLATEAU_WIDTH_PX = 50

RANDOM_VISUAL_SAMPLES_PER_SPLIT = 10
REPRESENTATIVE_SAMPLES_PER_CATEGORY = 8
MIN_TOTAL_REPRESENTATIVE_SAMPLES_PER_SPLIT = 30
MAX_TOTAL_REPRESENTATIVE_SAMPLES_PER_SPLIT = 50

RANDOM_SEED = 42

SMALL_COMPONENT_DEFINITION = 10
MANY_COMPONENTS_DEFINITION = 5

WRITE_EXCEL = True
WRITE_CSV = True

RESOLUTION_NAME = (
    "native" if RESIZE_TO is None else f"{RESIZE_TO}x{RESIZE_TO}"
)

OUTPUT_DIR = Path(OUTPUT_DIR)
VISUAL_DIR = OUTPUT_DIR / "visual_inspection"
REPRESENTATIVE_DIR = OUTPUT_DIR / "representative_samples"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VISUAL_DIR.mkdir(parents=True, exist_ok=True)
REPRESENTATIVE_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = (
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"
)


def normalize_image_name(x):
    if pd.isna(x):
        return ""
    return os.path.basename(str(x).strip())


def image_variants(x):
    x = normalize_image_name(x)
    if not x:
        return set()

    variants = {x}
    root, ext = os.path.splitext(x)

    if ext:
        variants.add(root)
    else:
        for e in IMAGE_EXTENSIONS:
            variants.add(x + e)

    return variants


def find_column(columns, candidates, exclude=None):
    normalized = {
        str(c).strip().lower(): c
        for c in columns
    }

    for candidate in candidates:
        key = str(candidate).strip().lower()
        if key in normalized:
            col = normalized[key]
            if col != exclude:
                return col

    return None


def safe_area_list(x):
    if x is None:
        return []

    if isinstance(x, (list, tuple, np.ndarray)):
        return [int(v) for v in x]

    if isinstance(x, str):
        try:
            value = ast.literal_eval(x)
            if isinstance(value, (list, tuple)):
                return [int(v) for v in value]
        except Exception:
            return []

    if pd.isna(x):
        return []

    return []


def percentile_or_nan(values, q):
    values = np.asarray(values)
    if len(values) == 0:
        return np.nan
    return float(np.percentile(values, q))


def json_list(values):
    return json.dumps(
        [int(v) for v in values],
        ensure_ascii=False
    )

NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}


def normalize_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower().replace("-", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def word_number_to_int(words):
    words = [w.lower() for w in words]

    if len(words) == 1:
        return NUMBER_WORDS.get(words[0])

    if len(words) == 2:
        a = NUMBER_WORDS.get(words[0])
        b = NUMBER_WORDS.get(words[1])
        if a is not None and b is not None and a >= 20 and b < 10:
            return a + b

    return None


def extract_region_number(text):
    t = normalize_text(text)

    if not t:
        return {
            "text_regions": np.nan,
            "extraction_status": "empty_text",
            "matched_phrase": "",
        }

    zero_patterns = [
        r"\bno\s+infected\s+areas?\b",
        r"\bno\s+infection\s+areas?\b",
        r"\bno\s+affected\s+areas?\b",
        r"\bno\s+infected\s+regions?\b",
        r"\bno\s+infection\s+regions?\b",
        r"\bno\s+affected\s+regions?\b",
        r"\bno\s+lesions?\b",
        r"\bno\s+regions?\s+of\s+infection\b",
        r"\bno\s+evidence\s+of\s+infection\b",
        r"\bno\s+sign(?:s)?\s+of\s+infection\b",
        r"\bwithout\s+(?:any\s+)?infection\b",
        r"\bwithout\s+(?:any\s+)?infected\s+(?:areas?|regions?|lesions?)\b",
    ]

    for pattern in zero_patterns:
        m = re.search(pattern, t)
        if m:
            return {
                "text_regions": 0,
                "extraction_status": "zero_phrase",
                "matched_phrase": m.group(0),
            }

    numeric_patterns = [
        r"\b(\d+)\s+infected\s+areas?\b",
        r"\b(\d+)\s+affected\s+areas?\b",
        r"\b(\d+)\s+infection\s+areas?\b",
        r"\b(\d+)\s+lesions?\b",
        r"\b(\d+)\s+regions?\b",
        r"\b(\d+)\s+areas?\b",
    ]

    for pattern in numeric_patterns:
        m = re.search(pattern, t)
        if m:
            return {
                "text_regions": int(m.group(1)),
                "extraction_status": "digit",
                "matched_phrase": m.group(0),
            }

    tens = [
        "twenty", "thirty", "forty", "fifty",
        "sixty", "seventy", "eighty", "ninety"
    ]
    units = [
        "one", "two", "three", "four", "five",
        "six", "seven", "eight", "nine"
    ]

    for ten in tens:
        for unit in units:
            phrase = f"{ten} {unit}"
            pattern = (
                rf"\b{re.escape(phrase)}\s+"
                rf"(?:infected\s+)?"
                rf"(?:areas?|regions?|lesions?)\b"
            )
            m = re.search(pattern, t)
            if m:
                return {
                    "text_regions": word_number_to_int(phrase.split()),
                    "extraction_status": "number_word",
                    "matched_phrase": m.group(0),
                }

    for word in sorted(NUMBER_WORDS, key=len, reverse=True):
        pattern = (
            rf"\b{re.escape(word)}\s+"
            rf"(?:infected\s+)?"
            rf"(?:areas?|regions?|lesions?)\b"
        )
        m = re.search(pattern, t)
        if m:
            return {
                "text_regions": NUMBER_WORDS[word],
                "extraction_status": "number_word",
                "matched_phrase": m.group(0),
            }

    single_patterns = [
        r"\bsingle\s+infected\s+(?:area|region|lesion)\b",
        r"\ba\s+single\s+infected\s+(?:area|region|lesion)\b",
        r"\bone\s+infected\s+(?:area|region|lesion)\b",
        r"\ban?\s+infected\s+(?:area|region|lesion)\b",
        r"\bone\s+(?:area|region|lesion)\b",
        r"\ba\s+single\s+(?:area|region|lesion)\b",
    ]

    for pattern in single_patterns:
        m = re.search(pattern, t)
        if m:
            return {
                "text_regions": 1,
                "extraction_status": "single_area_phrase",
                "matched_phrase": m.group(0),
            }

    for qualifier in ["multiple", "several", "many"]:
        for noun in ["area", "region", "lesion"]:
            for suffix in ("s", ""):
                for infix in ("infected ", ""):
                    phrase = f"{qualifier} {infix}{noun}{suffix}"
                    if phrase in t:
                        return {
                            "text_regions": np.nan,
                            "extraction_status": "qualitative_unknown",
                            "matched_phrase": phrase,
                        }

    return {
        "text_regions": np.nan,
        "extraction_status": "not_found",
        "matched_phrase": "",
    }

print("=" * 90)
print("QaTa-COV19 DATA AUDIT")
print("=" * 90)
print(f"Connectivity: {CONNECTIVITY_NAME}")
print("Definition: skimage connectivity=2 for 2-D masks")
print(f"Working resolution: {RESOLUTION_NAME}")
print()

MASK_DIR = Path(MASK_DIR)
TEXT_FILE = Path(TEXT_FILE)
TRAIN_ID_FILE = Path(TRAIN_ID_FILE)

if not MASK_DIR.exists():
    raise FileNotFoundError(f"Mask directory not found: {MASK_DIR}")

mask_files = sorted(
    [
        p.name
        for p in MASK_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
)

if not mask_files:
    raise RuntimeError("No mask/image files were found.")

print(f"All mask files found: {len(mask_files):,}")

if not TRAIN_ID_FILE.exists():
    raise FileNotFoundError(f"Train-ID file not found: {TRAIN_ID_FILE}")

train_id_raw = pd.read_csv(TRAIN_ID_FILE)

image_candidates = [
    "Image", "image", "filename", "Filename",
    "image_name", "ImageName", "ID", "Id", "id"
]

train_id_image_col = (
    find_column(train_id_raw.columns, image_candidates)
    or train_id_raw.columns[0]
)

train_ids = set(
    train_id_raw[train_id_image_col]
    .dropna()
    .map(normalize_image_name)
)
train_ids.discard("")

mask_lookup = {}
for filename in mask_files:
    for variant in image_variants(filename):
        mask_lookup[variant] = filename

train_images = []
unmatched_train_ids = []

for train_id in sorted(train_ids):
    found = None
    for variant in image_variants(train_id):
        if variant in mask_lookup:
            found = mask_lookup[variant]
            break

    if found is None:
        unmatched_train_ids.append(train_id)
    else:
        train_images.append(found)

train_images = sorted(set(train_images))
train_image_set = set(train_images)

validation_images = sorted(
    set(mask_files) - train_image_set
)
validation_image_set = set(validation_images)

print()
print("TRAIN / VALIDATION SPLIT")
print("-" * 90)
print(f"Train masks:       {len(train_images):,}")
print(f"Validation masks:  {len(validation_images):,}")
print(f"Unmatched train IDs: {len(unmatched_train_ids):,}")
print("Test data: NOT USED")

split_df = pd.DataFrame({
    "image": train_images + validation_images,
    "split": (
        ["train"] * len(train_images)
        + ["validation"] * len(validation_images)
    ),
})

split_df.to_csv(
    OUTPUT_DIR / "qata_train_validation_split_audit.csv",
    index=False
)

if not TEXT_FILE.exists():
    raise FileNotFoundError(f"Text CSV not found: {TEXT_FILE}")

raw_text = pd.read_csv(TEXT_FILE)

text_image_candidates = [
    "Image", "image", "filename", "Filename",
    "image_name", "ImageName", "Imagetext"
]

text_column_candidates = [
    "text", "Text", "caption", "Caption",
    "description", "Description", "Imagetext"
]

text_image_col = (
    TEXT_IMAGE_COLUMN_OVERRIDE
    or find_column(raw_text.columns, text_image_candidates)
    or raw_text.columns[0]
)

text_col = (
    TEXT_COLUMN_OVERRIDE
    or find_column(
        raw_text.columns,
        text_column_candidates,
        exclude=text_image_col
    )
)

if text_col is None:
    if len(raw_text.columns) > 1:
        text_col = raw_text.columns[1]
    else:
        raise ValueError("Could not identify text column.")

text_records = []
unmatched_text_images = []

for _, row in raw_text.iterrows():
    original_image = normalize_image_name(row[text_image_col])

    found_mask = None
    for variant in image_variants(original_image):
        if variant in mask_lookup:
            found_mask = mask_lookup[variant]
            break

    if found_mask is None:
        unmatched_text_images.append(original_image)
        continue

    extraction = extract_region_number(row[text_col])

    if found_mask in train_image_set:
        split = "train"
    elif found_mask in validation_image_set:
        split = "validation"
    else:
        split = "unknown"

    text_records.append({
        "image": found_mask,
        "split": split,
        "text": row[text_col],
        **extraction,
    })

text_df = pd.DataFrame(text_records)

if len(text_df):
    text_df = (
        text_df
        .drop_duplicates("image", keep="first")
        .copy()
    )

train_text_df = text_df[text_df["split"] == "train"].copy()
val_text_df = text_df[text_df["split"] == "validation"].copy()

text_df.to_csv(
    OUTPUT_DIR / "all_train_validation_text_region_extraction.csv",
    index=False
)
train_text_df.to_csv(
    OUTPUT_DIR / "train_text_region_extraction_audit.csv",
    index=False
)
val_text_df.to_csv(
    OUTPUT_DIR / "validation_text_region_extraction_audit.csv",
    index=False
)

print()
print("TEXT AUDIT")
print("-" * 90)
print(f"Text rows matched to masks: {len(text_df):,}")
print(f"Train text rows:            {len(train_text_df):,}")
print(f"Validation text rows:       {len(val_text_df):,}")
print(f"Unmatched text rows:        {len(unmatched_text_images):,}")

if len(text_df):
    print("\nExtraction status:")
    print(text_df["extraction_status"].value_counts(dropna=False).to_string())


def read_mask(mask_path, resize_to=RESIZE_TO):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise ValueError(f"Could not read mask: {mask_path}")

    original_shape = (int(mask.shape[0]), int(mask.shape[1]))

    if resize_to is not None:
        mask = cv2.resize(
            mask,
            (resize_to, resize_to),
            interpolation=RESIZE_INTERPOLATION
        )

    return mask, original_shape


def extract_connected_components(mask_path):
    mask, original_shape = read_mask(mask_path, RESIZE_TO)

    binary = (mask > 0).astype(np.uint8)

    labeled = label(
        binary,
        connectivity=CONNECTIVITY
    )

    props = regionprops(labeled)

    components = []

    for prop in props:
        components.append({
            "component_label": int(prop.label),
            "area_px": int(prop.area),
            "bbox_min_row": int(prop.bbox[0]),
            "bbox_min_col": int(prop.bbox[1]),
            "bbox_max_row": int(prop.bbox[2]),
            "bbox_max_col": int(prop.bbox[3]),
            "centroid_row": float(prop.centroid[0]),
            "centroid_col": float(prop.centroid[1]),
        })

    areas = np.array(
        [c["area_px"] for c in components],
        dtype=np.int64
    )

    return {
        "mask": mask,
        "binary": binary,
        "labeled": labeled,
        "components": components,
        "areas": areas,
        "foreground_pixels": int(binary.sum()),
        "height": int(binary.shape[0]),
        "width": int(binary.shape[1]),
        "orig_height": original_shape[0],
        "orig_width": original_shape[1],
    }

sample_records = []
component_records = []

text_lookup = (
    text_df.set_index("image").to_dict("index")
    if len(text_df)
    else {}
)

all_images_by_split = [
    ("train", train_images),
    ("validation", validation_images),
]

errors = []

for split, images in all_images_by_split:
    print()
    print("=" * 90)
    print(
        f"EXTRACTING RAW CONNECTED COMPONENTS: {split.upper()} "
        f"[{RESOLUTION_NAME}]"
    )
    print("=" * 90)

    for image_name in tqdm(images, desc=f"{split}"):
        try:
            result = extract_connected_components(
                MASK_DIR / image_name
            )

            areas = result["areas"]

            text_info = text_lookup.get(image_name, {})

            sample_records.append({
                "image": image_name,
                "split": split,
                "orig_height": result["orig_height"],
                "orig_width": result["orig_width"],
                "height": result["height"],
                "width": result["width"],
                "foreground_pixels": result["foreground_pixels"],
                "raw_component_count": int(len(areas)),
                "raw_component_area_list": json_list(areas),
                "raw_min_component_area_px": (
                    int(areas.min()) if len(areas) else 0
                ),
                "raw_max_component_area_px": (
                    int(areas.max()) if len(areas) else 0
                ),
                "raw_mean_component_area_px": (
                    float(areas.mean()) if len(areas) else np.nan
                ),
                "raw_median_component_area_px": (
                    float(np.median(areas)) if len(areas) else np.nan
                ),
                "text": text_info.get("text", np.nan),
                "text_regions": text_info.get(
                    "text_regions", np.nan
                ),
                "extraction_status": text_info.get(
                    "extraction_status", "no_text"
                ),
                "matched_phrase": text_info.get(
                    "matched_phrase", ""
                ),
            })

            for component in result["components"]:
                component_records.append({
                    "image": image_name,
                    "split": split,
                    **component,
                })

        except Exception as exc:
            errors.append({
                "image": image_name,
                "split": split,
                "error": repr(exc),
            })

sample_df = pd.DataFrame(sample_records)
component_df = pd.DataFrame(component_records)
errors_df = pd.DataFrame(errors)

if len(errors_df):
    errors_df.to_csv(
        OUTPUT_DIR / "audit_errors.csv",
        index=False
    )
    print(f"\nWARNING: {len(errors_df)} masks failed.")

print()
print(f"Successfully analyzed masks: {len(sample_df):,}")
print(f"Raw components extracted:    {len(component_df):,}")

if len(sample_df):
    orig_sizes = (
        sample_df[["orig_height", "orig_width"]]
        .value_counts()
        .head(5)
    )
    print(
        f"Working resolution: {RESOLUTION_NAME}. "
        f"Most common ORIGINAL mask sizes (h, w):"
    )
    print(orig_sizes.to_string())

component_df.to_csv(
    OUTPUT_DIR / "raw_component_results.csv",
    index=False
)

sample_df.to_csv(
    OUTPUT_DIR / "raw_sample_results.csv",
    index=False
)


def compute_area_distribution(component_table, split_name):
    x = component_table[
        component_table["split"] == split_name
    ]["area_px"].to_numpy(dtype=np.int64)

    if len(x) == 0:
        return pd.DataFrame()

    rows = [{
        "split": split_name,
        "component_count": len(x),
        "min_px": int(x.min()),
        "max_px": int(x.max()),
        "mean_px": float(x.mean()),
        "median_px": float(np.median(x)),
        "P01_px": percentile_or_nan(x, 1),
        "P05_px": percentile_or_nan(x, 5),
        "P10_px": percentile_or_nan(x, 10),
        "P25_px": percentile_or_nan(x, 25),
        "P50_px": percentile_or_nan(x, 50),
        "P75_px": percentile_or_nan(x, 75),
        "P90_px": percentile_or_nan(x, 90),
        "P95_px": percentile_or_nan(x, 95),
        "P99_px": percentile_or_nan(x, 99),
    }]

    return pd.DataFrame(rows)

area_summary = pd.concat([
    compute_area_distribution(component_df, "train"),
    compute_area_distribution(component_df, "validation"),
], ignore_index=True)

area_summary.to_csv(
    OUTPUT_DIR / "component_area_percentiles_train_validation.csv",
    index=False
)


def compute_area_bins(component_table, split_name):
    x = component_table[
        component_table["split"] == split_name
    ]["area_px"].to_numpy(dtype=np.int64)

    bins = [
        ("1", 1, 1),
        ("2-5", 2, 5),
        ("6-10", 6, 10),
        ("11-25", 11, 25),
        ("26-50", 26, 50),
        ("51-100", 51, 100),
        ("101-250", 101, 250),
        ("251-500", 251, 500),
        ("501-1000", 501, 1000),
        ("1001-2500", 1001, 2500),
        ("2501-5000", 2501, 5000),
        ("5001-10000", 5001, 10000),
        ("10001+", 10001, np.inf),
    ]

    rows = []
    total = len(x)

    for label_name, lo, hi in bins:
        count = int(np.sum((x >= lo) & (x <= hi)))
        rows.append({
            "split": split_name,
            "area_range_px": label_name,
            "count": count,
            "percentage_of_components": (
                100.0 * count / total if total else np.nan
            ),
        })

    return pd.DataFrame(rows)

area_bins = pd.concat([
    compute_area_bins(component_df, "train"),
    compute_area_bins(component_df, "validation"),
], ignore_index=True)

area_bins.to_csv(
    OUTPUT_DIR / "component_area_bins_train_validation.csv",
    index=False
)


def threshold_counts_for_sample(area_list, threshold):
    areas = np.asarray(safe_area_list(area_list), dtype=np.int64)
    return int(np.sum(areas >= threshold))


def threshold_mask_metrics(sample_table, threshold):
    counts = np.array([
        threshold_counts_for_sample(
            row["raw_component_area_list"],
            threshold
        )
        for _, row in sample_table.iterrows()
    ], dtype=np.int64)

    raw_component_total = 0
    kept_component_total = 0
    raw_foreground_total = 0
    kept_foreground_total = 0

    for _, row in sample_table.iterrows():
        areas = np.asarray(
            safe_area_list(row["raw_component_area_list"]),
            dtype=np.int64
        )

        raw_component_total += len(areas)
        kept = areas[areas >= threshold]
        kept_component_total += len(kept)

        raw_foreground_total += int(row["foreground_pixels"])
        kept_foreground_total += int(kept.sum())

    removed_component_total = (
        raw_component_total - kept_component_total
    )

    return {
        "samples": len(sample_table),
        "threshold_px": int(threshold),
        "raw_component_total": int(raw_component_total),
        "kept_component_total": int(kept_component_total),
        "removed_component_total": int(removed_component_total),
        "component_removal_pct": (
            100.0 * removed_component_total / raw_component_total
            if raw_component_total else np.nan
        ),
        "raw_foreground_pixels": int(raw_foreground_total),
        "kept_foreground_pixels": int(kept_foreground_total),
        "removed_foreground_pixels": int(
            raw_foreground_total - kept_foreground_total
        ),
        "foreground_retention_pct": (
            100.0 * kept_foreground_total / raw_foreground_total
            if raw_foreground_total else np.nan
        ),
        "foreground_removed_pct": (
            100.0
            * (raw_foreground_total - kept_foreground_total)
            / raw_foreground_total
            if raw_foreground_total else np.nan
        ),
        "mean_regions": (
            float(counts.mean()) if len(counts) else np.nan
        ),
        "median_regions": (
            float(np.median(counts)) if len(counts) else np.nan
        ),
        "P90_regions": percentile_or_nan(counts, 90),
        "P95_regions": percentile_or_nan(counts, 95),
        "P99_regions": percentile_or_nan(counts, 99),
        "max_regions": (
            int(counts.max()) if len(counts) else 0
        ),
        "no_region_count": int(np.sum(counts == 0)),
        "single_region_count": int(np.sum(counts == 1)),
        "multi_region_count": int(np.sum(counts >= 2)),
        "no_region_pct": (
            100.0 * np.mean(counts == 0)
            if len(counts) else np.nan
        ),
        "single_region_pct": (
            100.0 * np.mean(counts == 1)
            if len(counts) else np.nan
        ),
        "multi_region_pct": (
            100.0 * np.mean(counts >= 2)
            if len(counts) else np.nan
        ),
    }

threshold_rows = []

for threshold in THRESHOLDS:
    for split_name in ["train", "validation"]:
        split_table = sample_df[
            sample_df["split"] == split_name
        ].copy()

        metrics = threshold_mask_metrics(
            split_table,
            threshold
        )
        metrics["split"] = split_name
        threshold_rows.append(metrics)

threshold_mask_df = pd.DataFrame(threshold_rows)

threshold_mask_df = threshold_mask_df[
    [
        "split",
        "threshold_px",
        "samples",
        "raw_component_total",
        "kept_component_total",
        "removed_component_total",
        "component_removal_pct",
        "raw_foreground_pixels",
        "kept_foreground_pixels",
        "removed_foreground_pixels",
        "foreground_retention_pct",
        "foreground_removed_pct",
        "mean_regions",
        "median_regions",
        "P90_regions",
        "P95_regions",
        "P99_regions",
        "max_regions",
        "no_region_count",
        "single_region_count",
        "multi_region_count",
        "no_region_pct",
        "single_region_pct",
        "multi_region_pct",
    ]
]

threshold_mask_df.to_csv(
    OUTPUT_DIR / "threshold_analysis_MASK_ONLY_train_validation.csv",
    index=False
)

train_mask_thresholds = (
    threshold_mask_df[threshold_mask_df["split"] == "train"]
    .sort_values("threshold_px")
    .reset_index(drop=True)
    .copy()
)

CANDIDATE_THRESHOLDS_SORTED = [
    int(t) for t in train_mask_thresholds["threshold_px"]
]


def relative_step_changes(values):
    values = np.asarray(values, dtype=float)
    prev = values[:-1]
    curr = values[1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(
            prev != 0,
            np.abs(curr - prev) / np.abs(prev),
            np.where(curr == 0, 0.0, np.inf),
        )
    return np.concatenate([[np.nan], rel])

train_mask_thresholds["rel_change_mean_regions"] = (
    relative_step_changes(train_mask_thresholds["mean_regions"])
)
train_mask_thresholds["rel_change_multi_region_pct"] = (
    relative_step_changes(train_mask_thresholds["multi_region_pct"])
)

train_mask_thresholds.to_csv(
    OUTPUT_DIR / "threshold_stability_step_changes_TRAIN.csv",
    index=False
)


def plateau_upper_bound_px(thresholds, rel_changes, tolerance):
    upper = thresholds[0]
    for t, rel in zip(thresholds[1:], rel_changes[1:]):
        if np.isnan(rel) or rel > tolerance:
            break
        upper = t
    return upper

sensitivity_rows = []
for tol in STABILITY_TOLERANCES:
    ub_mean = plateau_upper_bound_px(
        CANDIDATE_THRESHOLDS_SORTED,
        train_mask_thresholds["rel_change_mean_regions"].to_numpy(),
        tol,
    )
    ub_multi = plateau_upper_bound_px(
        CANDIDATE_THRESHOLDS_SORTED,
        train_mask_thresholds["rel_change_multi_region_pct"].to_numpy(),
        tol,
    )
    combined_upper_bound = min(ub_mean, ub_multi)
    plateau_width = combined_upper_bound - CANDIDATE_THRESHOLDS_SORTED[0]

    sensitivity_rows.append({
        "tolerance_fraction": tol,
        "plateau_upper_bound_mean_regions_px": ub_mean,
        "plateau_upper_bound_multi_region_pct_px": ub_multi,
        "plateau_upper_bound_combined_px": combined_upper_bound,
        "plateau_width_px": plateau_width,
        "meets_min_plateau_width": bool(
            plateau_width >= MIN_PLATEAU_WIDTH_PX
        ),
    })

threshold_stability_sensitivity = pd.DataFrame(sensitivity_rows)
threshold_stability_sensitivity.to_csv(
    OUTPUT_DIR / "threshold_stability_sensitivity_MASK_ONLY.csv",
    index=False
)

primary_row = threshold_stability_sensitivity[
    threshold_stability_sensitivity["tolerance_fraction"]
    == PRIMARY_STABILITY_TOLERANCE
].iloc[0]

plateau_upper_bound_px_primary = int(
    primary_row["plateau_upper_bound_combined_px"]
)
plateau_width_px_primary = int(primary_row["plateau_width_px"])


def chord_distance_curve(thresholds, values):
    x = np.asarray(thresholds, dtype=float)
    y = np.asarray(values, dtype=float)

    x_range = x.max() - x.min()
    y_range = y.max() - y.min()

    xn = (x - x.min()) / x_range if x_range else np.zeros_like(x)
    yn = (y - y.min()) / y_range if y_range else np.zeros_like(y)

    x0, y0 = xn[0], yn[0]
    x1, y1 = xn[-1], yn[-1]

    numerator = np.abs(
        (y1 - y0) * xn - (x1 - x0) * yn + x1 * y0 - y1 * x0
    )
    denominator = np.sqrt((y1 - y0) ** 2 + (x1 - x0) ** 2)

    if denominator == 0:
        return np.zeros_like(xn)

    return numerator / denominator

elbow_curve_df = train_mask_thresholds[
    ["threshold_px", "mean_regions", "multi_region_pct"]
].copy()

elbow_curve_df["chord_distance_mean_regions"] = chord_distance_curve(
    elbow_curve_df["threshold_px"], elbow_curve_df["mean_regions"]
)
elbow_curve_df["chord_distance_multi_region_pct"] = chord_distance_curve(
    elbow_curve_df["threshold_px"], elbow_curve_df["multi_region_pct"]
)

elbow_threshold_mean = int(
    elbow_curve_df.loc[
        elbow_curve_df["chord_distance_mean_regions"].idxmax(),
        "threshold_px",
    ]
)
elbow_threshold_multi = int(
    elbow_curve_df.loc[
        elbow_curve_df["chord_distance_multi_region_pct"].idxmax(),
        "threshold_px",
    ]
)

elbow_curve_df["log10_threshold_px"] = np.log10(
    elbow_curve_df["threshold_px"]
)
elbow_curve_df["chord_distance_mean_regions_logaxis"] = (
    chord_distance_curve(
        elbow_curve_df["log10_threshold_px"],
        elbow_curve_df["mean_regions"],
    )
)
elbow_curve_df["chord_distance_multi_region_pct_logaxis"] = (
    chord_distance_curve(
        elbow_curve_df["log10_threshold_px"],
        elbow_curve_df["multi_region_pct"],
    )
)

elbow_threshold_mean_logaxis = int(
    elbow_curve_df.loc[
        elbow_curve_df["chord_distance_mean_regions_logaxis"].idxmax(),
        "threshold_px",
    ]
)
elbow_threshold_multi_logaxis = int(
    elbow_curve_df.loc[
        elbow_curve_df["chord_distance_multi_region_pct_logaxis"].idxmax(),
        "threshold_px",
    ]
)

DISAGREEMENT_RATIO_FLAG_THRESHOLD = 3.0


def nearest_candidate_threshold(value, candidates):
    candidates = sorted(candidates)
    return min(
        candidates,
        key=lambda c: (abs(c - value), c)
    )

log_axis_knee_estimates_px = [
    elbow_threshold_mean_logaxis,
    elbow_threshold_multi_logaxis,
]

log_axis_knee_median_raw = float(np.median(log_axis_knee_estimates_px))
elbow_threshold_combined = nearest_candidate_threshold(
    log_axis_knee_median_raw,
    CANDIDATE_THRESHOLDS_SORTED
)

log_axis_disagreement_ratio = (
    max(log_axis_knee_estimates_px) / min(log_axis_knee_estimates_px)
    if min(log_axis_knee_estimates_px) > 0
    else float("inf")
)
knee_estimates_high_disagreement = bool(
    log_axis_disagreement_ratio > DISAGREEMENT_RATIO_FLAG_THRESHOLD
)

knee_estimates_px = [
    elbow_threshold_mean,
    elbow_threshold_multi,
    elbow_threshold_mean_logaxis,
    elbow_threshold_multi_logaxis,
]
knee_estimates_median_raw = float(np.median(knee_estimates_px))
all_four_disagreement_ratio = (
    max(knee_estimates_px) / min(knee_estimates_px)
    if min(knee_estimates_px) > 0
    else float("inf")
)
linear_vs_log_axis_gap_mean_regions_px = (
    elbow_threshold_mean - elbow_threshold_mean_logaxis
)
linear_vs_log_axis_gap_multi_region_pct_px = (
    elbow_threshold_multi - elbow_threshold_multi_logaxis
)

elbow_curve_df.to_csv(
    OUTPUT_DIR / "threshold_knee_elbow_curve_TRAIN.csv",
    index=False
)


def compute_near_elbow_robustness(mean_distance_col, multi_distance_col):
    rows = []
    for margin in (0.90, 0.95, 0.99):
        max_d_mean = elbow_curve_df[mean_distance_col].max()
        max_d_multi = elbow_curve_df[multi_distance_col].max()

        near_mean = elbow_curve_df.loc[
            elbow_curve_df[mean_distance_col] >= margin * max_d_mean,
            "threshold_px",
        ]
        near_multi = elbow_curve_df.loc[
            elbow_curve_df[multi_distance_col] >= margin * max_d_multi,
            "threshold_px",
        ]

        rows.append({
            "margin_of_max_distance": margin,
            "near_elbow_range_mean_regions_px":
                f"[{int(near_mean.min())}, {int(near_mean.max())}]",
            "near_elbow_range_multi_region_pct_px":
                f"[{int(near_multi.min())}, {int(near_multi.max())}]",
        })
    return pd.DataFrame(rows)

threshold_elbow_robustness = compute_near_elbow_robustness(
    "chord_distance_mean_regions_logaxis",
    "chord_distance_multi_region_pct_logaxis",
)
threshold_elbow_robustness.to_csv(
    OUTPUT_DIR / "threshold_knee_elbow_robustness_MASK_ONLY.csv",
    index=False
)

threshold_elbow_robustness_linear_diagnostic = compute_near_elbow_robustness(
    "chord_distance_mean_regions",
    "chord_distance_multi_region_pct",
)
threshold_elbow_robustness_linear_diagnostic.to_csv(
    OUTPUT_DIR
    / "threshold_knee_elbow_robustness_LINEAR_AXIS_diagnostic_only.csv",
    index=False
)

PREVIOUS_TEXT_BASED_THRESHOLD_PX = 1

previous_threshold_check_rows = []
for _, row in threshold_elbow_robustness.iterrows():
    lo_mean, hi_mean = (
        row["near_elbow_range_mean_regions_px"]
        .strip("[]")
        .split(",")
    )
    lo_multi, hi_multi = (
        row["near_elbow_range_multi_region_pct_px"]
        .strip("[]")
        .split(",")
    )
    lo_mean, hi_mean = int(lo_mean), int(hi_mean)
    lo_multi, hi_multi = int(lo_multi), int(hi_multi)

    inside_mean = lo_mean <= PREVIOUS_TEXT_BASED_THRESHOLD_PX <= hi_mean
    inside_multi = lo_multi <= PREVIOUS_TEXT_BASED_THRESHOLD_PX <= hi_multi

    previous_threshold_check_rows.append({
        "margin_of_max_distance": row["margin_of_max_distance"],
        "previous_threshold_px": PREVIOUS_TEXT_BASED_THRESHOLD_PX,
        "inside_near_elbow_range_mean_regions": bool(inside_mean),
        "inside_near_elbow_range_multi_region_pct": bool(inside_multi),
        "inside_both": bool(inside_mean and inside_multi),
    })

previous_threshold_check = pd.DataFrame(previous_threshold_check_rows)
previous_threshold_check.to_csv(
    OUTPUT_DIR
    / f"previous_threshold_{PREVIOUS_TEXT_BASED_THRESHOLD_PX}px_sanity_check_MASK_ONLY.csv",
    index=False
)

if bool(primary_row["meets_min_plateau_width"]):

    selected_threshold = int(CANDIDATE_THRESHOLDS_SORTED[0])
    selection_status = "TIER1_stable_plateau_smallest_threshold_selected"
    selection_tier = "tier1_stability_plateau"
else:
    selected_threshold = int(elbow_threshold_combined)
    selection_status = (
        "TIER2_knee_elbow_fallback_selected__log_axis_combiner"
        + (
            "__HIGH_DISAGREEMENT_VISUAL_REVIEW_RECOMMENDED"
            if knee_estimates_high_disagreement
            else ""
        )
    )
    selection_tier = "tier2_knee_elbow"

selection_metadata = {
    "working_resolution": RESOLUTION_NAME,
    "resize_to": RESIZE_TO,
    "resize_interpolation": (
        "INTER_NEAREST" if RESIZE_INTERPOLATION == cv2.INTER_NEAREST
        else str(RESIZE_INTERPOLATION)
    ),
    "connectivity": CONNECTIVITY_NAME,
    "skimage_connectivity_parameter": CONNECTIVITY,
    "foreground_definition": "mask_pixel_value > 0",
    "component_area_definition": (
        "regionprops(labeled_component).area; "
        "number of foreground pixels in the component "
        "(in the working-resolution mask)"
    ),
    "threshold_selection_basis": (
        "MASK ONLY: Tier 1 stability/plateau rule, falling back to "
        "Tier 2 knee/elbow (max distance-to-chord) detection when "
        "no stable plateau exists from the smallest threshold; "
        "text excluded from both tiers"
    ),
    "train_only_selection": True,
    "candidate_thresholds_px": CANDIDATE_THRESHOLDS_SORTED,
    "selection_tier": selection_tier,
    "tier1": {
        "stability_tolerances_tested_fraction": STABILITY_TOLERANCES,
        "primary_stability_tolerance_fraction":
            PRIMARY_STABILITY_TOLERANCE,
        "min_required_plateau_width_px": MIN_PLATEAU_WIDTH_PX,
        "plateau_upper_bound_px_at_primary_tolerance":
            plateau_upper_bound_px_primary,
        "plateau_width_px_at_primary_tolerance":
            plateau_width_px_primary,
        "plateau_found": bool(primary_row["meets_min_plateau_width"]),
    },
    "tier2": {
        "method": (
            "maximum perpendicular distance to the chord joining "
            "the first and last candidate threshold, on min-max "
            "normalized axes (Kneedle-style knee/elbow detection). "
            "Computed independently for mean_regions and "
            "multi_region_pct, and independently on a LINEAR "
            "threshold_px axis and a log10(threshold_px) axis -- "
            "4 estimates total. SELECTION uses only the two "
            "LOG-AXIS estimates (median of the two, snapped to the "
            "nearest tested candidate), because the candidate grid "
            "is close to log-spaced, not evenly spaced. The two "
            "LINEAR-axis estimates are reported here for "
            "transparency but are NOT part of the selection. "
            "Tier 2 determines the selected threshold only when "
            "Tier 1 finds no plateau (see selection_tier)."
        ),
        "individual_knee_estimates_px": {
            "linear_axis_mean_regions": elbow_threshold_mean,
            "linear_axis_multi_region_pct": elbow_threshold_multi,
            "log_axis_mean_regions": elbow_threshold_mean_logaxis,
            "log_axis_multi_region_pct": elbow_threshold_multi_logaxis,
        },
        "selection_uses": "log_axis_mean_regions, log_axis_multi_region_pct",
        "log_axis_median_raw": log_axis_knee_median_raw,
        "elbow_threshold_combined_px_snapped_to_candidate":
            elbow_threshold_combined,
        "log_axis_disagreement_ratio_max_over_min":
            log_axis_disagreement_ratio,
        "disagreement_ratio_flag_threshold":
            DISAGREEMENT_RATIO_FLAG_THRESHOLD,
        "high_disagreement_flagged": knee_estimates_high_disagreement,
        "diagnostic_only_NOT_used_for_selection": {
            "median_of_all_four_estimates_raw":
                knee_estimates_median_raw,
            "all_four_disagreement_ratio_max_over_min":
                all_four_disagreement_ratio,
            "linear_vs_log_axis_gap_mean_regions_px":
                linear_vs_log_axis_gap_mean_regions_px,
            "linear_vs_log_axis_gap_multi_region_pct_px":
                linear_vs_log_axis_gap_multi_region_pct_px,
        },
        "near_elbow_robustness_check":
            "see threshold_knee_elbow_robustness_MASK_ONLY.csv "
            "(log axis, used) and "
            "threshold_knee_elbow_robustness_LINEAR_AXIS_diagnostic_only.csv "
            "(linear axis, diagnostic only)",
    },
    "previous_text_based_threshold_sanity_check": {
        "note": (
            "The threshold previously selected on TRAIN by maximizing "
            "exact agreement with text-derived region counts (a "
            "criterion no longer used above, and originally computed "
            "at native resolution) is checked against the "
            "mask-only near-elbow region (log axis) purely as a "
            "sanity check, not as part of the selection rule."
        ),
        "previous_threshold_px": PREVIOUS_TEXT_BASED_THRESHOLD_PX,
        "results_by_margin":
            previous_threshold_check.to_dict(orient="records"),
    },
    "selected_threshold_px": selected_threshold,
    "selection_status": selection_status,
    "validation_policy": (
        "selected TRAIN threshold is frozen and applied unchanged "
        "to validation"
    ),
}

with open(
    OUTPUT_DIR / "threshold_selection_metadata.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        selection_metadata,
        f,
        indent=2,
        ensure_ascii=False
    )

print()
print("=" * 90)
print(f"MASK-ONLY THRESHOLD SELECTION  [resolution: {RESOLUTION_NAME}]")
print("=" * 90)
print(f"Selected TRAIN threshold:  {selected_threshold} px")
print(f"Selection status:          {selection_status}")
print(f"Selection tier:            {selection_tier}")
print()
print(
    f"TIER 1 (stability/plateau) @ tol={PRIMARY_STABILITY_TOLERANCE:.1%}: "
    f"plateau upper bound = {plateau_upper_bound_px_primary} px "
    f"(width {plateau_width_px_primary} px, "
    f"found={bool(primary_row['meets_min_plateau_width'])})"
)
print("Sensitivity of the plateau to the tolerance value itself:")
print(threshold_stability_sensitivity.to_string(index=False))
print()
print(
    f"TIER 2 (knee/elbow; "
    f"{'USED for the selection' if selection_tier == 'tier2_knee_elbow' else 'computed for transparency only, Tier 1 decided'}): "
    f"linear mean_regions={elbow_threshold_mean}px, "
    f"linear multi_region_pct={elbow_threshold_multi}px, "
    f"log-axis mean_regions={elbow_threshold_mean_logaxis}px, "
    f"log-axis multi_region_pct={elbow_threshold_multi_logaxis}px"
)
print(
    f"  -> combined (median of the 2 log-axis estimates, snapped "
    f"to nearest tested threshold) = {elbow_threshold_combined} px "
    f"(raw median = {log_axis_knee_median_raw:.1f} px)"
)
print(
    f"  -> log-axis disagreement ratio = "
    f"{log_axis_disagreement_ratio:.2f}x "
    f"(flag threshold = {DISAGREEMENT_RATIO_FLAG_THRESHOLD:.1f}x, "
    f"HIGH DISAGREEMENT = {knee_estimates_high_disagreement})"
)
print("Near-elbow robustness (log axis, the axis used for selection):")
print(threshold_elbow_robustness.to_string(index=False))
print()
print(
    f"Sanity check -- is the previous text-based threshold "
    f"({PREVIOUS_TEXT_BASED_THRESHOLD_PX} px, native resolution) inside "
    f"the mask-only near-elbow region (log axis)?"
)
print(previous_threshold_check.to_string(index=False))
print()
print(
    "Text-region agreement was NOT used to select this threshold."
)


def add_frozen_threshold_columns(sample_table, threshold):
    out = sample_table.copy()

    out["threshold_used_px"] = int(threshold)

    out["component_count_after_filtering"] = out[
        "raw_component_area_list"
    ].apply(
        lambda x: threshold_counts_for_sample(x, threshold)
    )

    out["component_count_removed"] = (
        out["raw_component_count"]
        - out["component_count_after_filtering"]
    )

    def kept_area_sum(x):
        areas = np.asarray(
            safe_area_list(x),
            dtype=np.int64
        )
        return int(areas[areas >= threshold].sum())

    out["foreground_pixels_after_filtering"] = out[
        "raw_component_area_list"
    ].apply(kept_area_sum)

    out["foreground_pixels_removed"] = (
        out["foreground_pixels"]
        - out["foreground_pixels_after_filtering"]
    )

    out["foreground_retention_pct"] = np.where(
        out["foreground_pixels"] > 0,
        100.0
        * out["foreground_pixels_after_filtering"]
        / out["foreground_pixels"],
        np.nan
    )

    out["region_category"] = np.select(
        [
            out["component_count_after_filtering"] == 0,
            out["component_count_after_filtering"] == 1,
            out["component_count_after_filtering"] >= 2,
        ],
        [
            "no-region",
            "single-region",
            "multi-region",
        ],
        default="unknown"
    )

    out["is_two_region"] = (
        out["component_count_after_filtering"] == 2
    )

    out["has_small_component_below_threshold"] = (
        out["raw_component_area_list"].apply(
            lambda x: any(
                a < threshold
                for a in safe_area_list(x)
            )
        )
    )

    return out

frozen_df = add_frozen_threshold_columns(
    sample_df,
    selected_threshold
)

frozen_df.to_csv(
    OUTPUT_DIR / "04_Data_Audit_Raw_Results.csv",
    index=False
)

component_raw = component_df.copy()

component_raw["threshold_used_px"] = selected_threshold
component_raw["kept_after_threshold"] = (
    component_raw["area_px"] >= selected_threshold
)
component_raw["removed_as_small_component"] = (
    component_raw["area_px"] < selected_threshold
)

component_raw.to_csv(
    OUTPUT_DIR / "04_Data_Audit_Raw_Component_Results.csv",
    index=False
)

threshold_per_sample = sample_df[
    [
        "image",
        "split",
        "raw_component_count",
        "foreground_pixels",
        "text_regions",
        "extraction_status",
    ]
].copy()

for threshold in THRESHOLDS:
    threshold_per_sample[
        f"component_count_threshold_{threshold}px"
    ] = [
        threshold_counts_for_sample(
            areas,
            threshold
        )
        for areas in sample_df["raw_component_area_list"]
    ]

threshold_per_sample.to_csv(
    OUTPUT_DIR / "per_sample_all_threshold_results.csv",
    index=False
)


def summary_at_threshold(table, split_name, threshold):
    x = table[
        table["split"] == split_name
    ]["component_count_after_filtering"].to_numpy()

    n = len(x)

    if n == 0:
        return {
            "split": split_name,
            "threshold_px": threshold,
            "samples": 0,
            "mean_regions": np.nan,
            "median_regions": np.nan,
            "P90_regions": np.nan,
            "P95_regions": np.nan,
            "P99_regions": np.nan,
            "max_regions": np.nan,
            "no_region_count": 0,
            "single_region_count": 0,
            "multi_region_count": 0,
            "no_region_pct": np.nan,
            "single_region_pct": np.nan,
            "multi_region_pct": np.nan,
        }

    return {
        "split": split_name,
        "threshold_px": threshold,
        "samples": n,
        "mean_regions": float(np.mean(x)),
        "median_regions": float(np.median(x)),
        "P90_regions": percentile_or_nan(x, 90),
        "P95_regions": percentile_or_nan(x, 95),
        "P99_regions": percentile_or_nan(x, 99),
        "max_regions": int(np.max(x)),
        "no_region_count": int(np.sum(x == 0)),
        "single_region_count": int(np.sum(x == 1)),
        "multi_region_count": int(np.sum(x >= 2)),
        "no_region_pct": 100.0 * np.mean(x == 0),
        "single_region_pct": 100.0 * np.mean(x == 1),
        "multi_region_pct": 100.0 * np.mean(x >= 2),
    }

frozen_summary_df = pd.DataFrame([
    summary_at_threshold(
        frozen_df, "train", selected_threshold
    ),
    summary_at_threshold(
        frozen_df, "validation", selected_threshold
    ),
])

frozen_summary_df.to_csv(
    OUTPUT_DIR / "frozen_threshold_train_validation_summary.csv",
    index=False
)


def compare_to_text(table, split_name):
    x = table[
        (table["split"] == split_name)
        & table["text_regions"].notna()
    ].copy()

    if len(x) == 0:
        return pd.DataFrame([{
            "split": split_name,
            "threshold_px": selected_threshold,
            "text_available_samples": 0,
            "exact_agreement_count": 0,
            "exact_agreement_pct": np.nan,
            "MAE": np.nan,
            "mean_signed_error": np.nan,
            "overestimation_rate": np.nan,
            "underestimation_rate": np.nan,
        }])

    pred = x["component_count_after_filtering"].to_numpy()
    target = x["text_regions"].astype(int).to_numpy()
    diff = pred - target

    return pd.DataFrame([{
        "split": split_name,
        "threshold_px": selected_threshold,
        "text_available_samples": len(x),
        "exact_agreement_count": int(np.sum(diff == 0)),
        "exact_agreement_pct": 100.0 * np.mean(diff == 0),
        "MAE": float(np.mean(np.abs(diff))),
        "mean_signed_error": float(np.mean(diff)),
        "overestimation_rate": 100.0 * np.mean(diff > 0),
        "underestimation_rate": 100.0 * np.mean(diff < 0),
    }])

text_comparison_summary = pd.concat([
    compare_to_text(frozen_df, "train"),
    compare_to_text(frozen_df, "validation"),
], ignore_index=True)

text_comparison_summary.to_csv(
    OUTPUT_DIR / "text_mask_comparison_frozen_threshold_summary.csv",
    index=False
)

comparison_rows = []

for split_name in ["train", "validation"]:
    x = frozen_df[
        (frozen_df["split"] == split_name)
        & frozen_df["text_regions"].notna()
    ].copy()

    if len(x):
        x["difference"] = (
            x["component_count_after_filtering"]
            - x["text_regions"].astype(int)
        )
        x["absolute_difference"] = x["difference"].abs()
        x["exact_match"] = x["difference"] == 0
        comparison_rows.append(x)

if comparison_rows:
    comparison_df = pd.concat(
        comparison_rows,
        ignore_index=True
    )
else:
    comparison_df = pd.DataFrame()

comparison_df.to_csv(
    OUTPUT_DIR / "text_mask_comparison_frozen_threshold.csv",
    index=False
)

text_threshold_rows = []

for threshold in THRESHOLDS:
    for split_name in ["train", "validation"]:
        x = frozen_df[
            frozen_df["split"] == split_name
        ].copy()

        predicted = np.array([
            threshold_counts_for_sample(
                areas,
                threshold
            )
            for areas in x["raw_component_area_list"]
        ])

        valid = x["text_regions"].notna().to_numpy()

        if valid.any():
            target = (
                x.loc[valid, "text_regions"]
                .astype(int)
                .to_numpy()
            )
            pred = predicted[valid]

            diff = pred - target

            exact_pct = 100.0 * np.mean(diff == 0)
            mae = float(np.mean(np.abs(diff)))
        else:
            exact_pct = np.nan
            mae = np.nan

        text_threshold_rows.append({
            "split": split_name,
            "threshold_px": threshold,
            "text_available_samples": int(valid.sum()),
            "exact_agreement_pct": exact_pct,
            "MAE": mae,
        })

text_threshold_df = pd.DataFrame(text_threshold_rows)

text_threshold_df.to_csv(
    OUTPUT_DIR / "threshold_text_agreement_descriptive_only.csv",
    index=False
)

rng = np.random.default_rng(RANDOM_SEED)


def add_category_flags(table):
    out = table.copy()

    out["category_no_region"] = (
        out["component_count_after_filtering"] == 0
    )
    out["category_single_region"] = (
        out["component_count_after_filtering"] == 1
    )
    out["category_two_region"] = (
        out["component_count_after_filtering"] == 2
    )
    out["category_multi_region"] = (
        out["component_count_after_filtering"] >= 3
    )
    out["category_small_component"] = (
        out["raw_component_area_list"].apply(
            lambda x: any(
                a < SMALL_COMPONENT_DEFINITION
                for a in safe_area_list(x)
            )
        )
    )
    out["category_many_components"] = (
        out["raw_component_count"] >= MANY_COMPONENTS_DEFINITION
    )

    return out

frozen_df = add_category_flags(frozen_df)

categories = [
    ("no_region", "category_no_region"),
    ("single_region", "category_single_region"),
    ("two_region", "category_two_region"),
    ("multi_region", "category_multi_region"),
    ("small_component", "category_small_component"),
    ("many_components", "category_many_components"),
]


def choose_representatives(split_table):
    chosen = []
    chosen_names = set()

    for category_name, flag_col in categories:
        candidates = split_table[
            split_table[flag_col]
            & (~split_table["image"].isin(chosen_names))
        ]

        if len(candidates):
            take_n = min(
                REPRESENTATIVE_SAMPLES_PER_CATEGORY,
                len(candidates)
            )
            sampled = candidates.sample(
                take_n,
                random_state=RANDOM_SEED
            )

            for _, row in sampled.iterrows():
                chosen.append(
                    (category_name, row["image"])
                )
                chosen_names.add(row["image"])

    remaining = split_table[
        ~split_table["image"].isin(chosen_names)
    ]

    needed = max(
        0,
        MIN_TOTAL_REPRESENTATIVE_SAMPLES_PER_SPLIT
        - len(chosen)
    )

    if needed > 0 and len(remaining):
        take_n = min(needed, len(remaining))
        extra = remaining.sample(
            take_n,
            random_state=RANDOM_SEED + 1
        )

        for _, row in extra.iterrows():
            chosen.append(
                ("additional", row["image"])
            )
            chosen_names.add(row["image"])

    chosen = chosen[:MAX_TOTAL_REPRESENTATIVE_SAMPLES_PER_SPLIT]

    return chosen

representative_selection = []

for split_name in ["train", "validation"]:
    split_table = frozen_df[
        frozen_df["split"] == split_name
    ].copy()

    chosen = choose_representatives(split_table)

    for category_name, image_name in chosen:
        representative_selection.append({
            "split": split_name,
            "category": category_name,
            "image": image_name,
        })

representative_selection_df = pd.DataFrame(
    representative_selection
)

representative_selection_df.to_csv(
    OUTPUT_DIR / "representative_sample_selection.csv",
    index=False
)


def component_color_image(labeled):
    return labeled


def create_filtered_label_image(labeled, threshold):
    filtered_labels = np.zeros_like(labeled, dtype=np.int32)

    for prop in regionprops(labeled):
        if prop.area >= threshold:
            filtered_labels[labeled == prop.label] = prop.label

    return filtered_labels


def create_small_component_label_image(labeled, threshold):
    small_labels = np.zeros_like(labeled, dtype=np.int32)

    for prop in regionprops(labeled):
        if prop.area < threshold:
            small_labels[labeled == prop.label] = prop.label

    return small_labels


def plot_audit_sample(
    image_name,
    split,
    threshold,
    category=None,
    save_path=None,
    show=False,
):
    result = extract_connected_components(
        MASK_DIR / image_name
    )

    mask = result["mask"]
    binary = result["binary"]
    labeled = result["labeled"]
    areas = result["areas"]

    filtered_labels = create_filtered_label_image(
        labeled, threshold
    )
    small_labels = create_small_component_label_image(
        labeled, threshold
    )

    row = frozen_df[
        (frozen_df["image"] == image_name)
        & (frozen_df["split"] == split)
    ]

    text_regions = np.nan
    text = ""

    if len(row):
        text_regions = row.iloc[0]["text_regions"]
        text = row.iloc[0]["text"]

    raw_count = len(areas)
    kept_count = int(np.sum(areas >= threshold))
    small_count = int(np.sum(areas < threshold))

    fig, ax = plt.subplots(
        1, 5,
        figsize=(22, 5)
    )

    ax[0].imshow(mask, cmap="gray")
    ax[0].set_title(f"Ground Truth Mask ({RESOLUTION_NAME})")
    ax[0].axis("off")

    ax[1].imshow(binary, cmap="gray")
    ax[1].set_title(
        f"Binary Mask\nForeground={result['foreground_pixels']:,} px"
    )
    ax[1].axis("off")

    ax[2].imshow(
        component_color_image(labeled),
        cmap="nipy_spectral"
    )
    ax[2].set_title(
        f"Connected Components\nRaw K={raw_count}"
    )
    ax[2].axis("off")

    ax[3].imshow(
        component_color_image(small_labels),
        cmap="nipy_spectral"
    )
    ax[3].set_title(
        f"Removed Components\nArea < {threshold}px: {small_count}"
    )
    ax[3].axis("off")

    ax[4].imshow(
        component_color_image(filtered_labels),
        cmap="nipy_spectral"
    )
    ax[4].set_title(
        f"After Filtering\nK={kept_count}"
    )
    ax[4].axis("off")

    if pd.isna(text_regions):
        text_region_string = "NA"
    else:
        text_region_string = str(int(text_regions))

    text_string = str(text)
    text_string = (
        text_string[:180] + "..."
        if len(text_string) > 180
        else text_string
    )

    title = (
        f"{split.upper()} | {image_name}"
        f"{' | ' + category if category else ''}\n"
        f"Frozen mask-only threshold={threshold}px "
        f"[{RESOLUTION_NAME}] | "
        f"Raw components={raw_count} | "
        f"Kept={kept_count} | "
        f"Text count={text_region_string}\n"
        f"Text: {text_string}"
    )

    fig.suptitle(title, fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.86])

    if save_path is not None:
        plt.savefig(
            save_path,
            dpi=200,
            bbox_inches="tight"
        )

    if show:
        plt.show()

    plt.close(fig)

for _, row in representative_selection_df.iterrows():
    split_name = row["split"]
    category = row["category"]
    image_name = row["image"]

    safe_name = re.sub(
        r"[^A-Za-z0-9_.-]",
        "_",
        image_name
    )

    split_dir = REPRESENTATIVE_DIR / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    save_path = (
        split_dir
        / f"{category}__{safe_name}.png"
    )

    plot_audit_sample(
        image_name=image_name,
        split=split_name,
        threshold=selected_threshold,
        category=category,
        save_path=save_path,
        show=False,
    )

for split_name in ["train", "validation"]:
    candidates = frozen_df[
        frozen_df["split"] == split_name
    ]

    if len(candidates) == 0:
        continue

    n = min(
        RANDOM_VISUAL_SAMPLES_PER_SPLIT,
        len(candidates)
    )

    sampled = candidates.sample(
        n,
        random_state=RANDOM_SEED
    )

    for _, row in sampled.iterrows():
        safe_name = re.sub(
            r"[^A-Za-z0-9_.-]",
            "_",
            row["image"]
        )

        save_path = (
            VISUAL_DIR
            / f"random_{split_name}__{safe_name}.png"
        )

        plot_audit_sample(
            image_name=row["image"],
            split=split_name,
            threshold=selected_threshold,
            save_path=save_path,
            show=False,
        )

qualitative_rows = []

for _, row in representative_selection_df.iterrows():
    image_name = row["image"]
    split_name = row["split"]
    category = row["category"]

    data_row = frozen_df[
        (frozen_df["image"] == image_name)
        & (frozen_df["split"] == split_name)
    ]

    if len(data_row) == 0:
        continue

    r = data_row.iloc[0]
    areas = safe_area_list(r["raw_component_area_list"])

    qualitative_rows.append({
        "split": split_name,
        "category": category,
        "image": image_name,
        "raw_component_count": r["raw_component_count"],
        "component_areas_px": json_list(areas),
        "threshold_px": selected_threshold,
        "components_after_filtering":
            r["component_count_after_filtering"],
        "small_components_removed":
            r["component_count_removed"],
        "text_regions": r["text_regions"],
        "visual_file": str(
            (
                REPRESENTATIVE_DIR
                / split_name
                / (
                    f"{category}__"
                    + re.sub(
                        r"[^A-Za-z0-9_.-]",
                        "_",
                        image_name
                    )
                    + ".png"
                )
            )
        ),
        "auditor_note": (
            "Inspect whether connected components correspond "
            "to visually meaningful separated foreground regions "
            "or to fragmentation/noise."
        ),
    })

qualitative_df = pd.DataFrame(qualitative_rows)

qualitative_df.to_csv(
    OUTPUT_DIR / "qualitative_representative_audit.csv",
    index=False
)

small_component_rows = []

for split_name in ["train", "validation"]:
    c = component_df[
        component_df["split"] == split_name
    ].copy()

    total = len(c)

    for cutoff in [1, 2, 5, 10, 20, 50, 100]:
        count = int(np.sum(c["area_px"] < cutoff))

        small_component_rows.append({
            "split": split_name,
            "cutoff_px": cutoff,
            "components_below_cutoff": count,
            "percentage_of_components":
                100.0 * count / total if total else np.nan,
        })

small_component_stats = pd.DataFrame(
    small_component_rows
)

small_component_stats.to_csv(
    OUTPUT_DIR / "small_component_statistics.csv",
    index=False
)

method_text = f"""
QaTa-COV19 Data Audit -- working resolution {RESOLUTION_NAME}
=====================

Dataset:
QaTa-COV19-v2, Train Set Ground-truths.

Split:
Train IDs are read from:
{TRAIN_ID_FILE}

Validation is defined as all mask files in the same Train Set Ground-truths
directory that are not present in the Train-ID list.

Test data:
Not loaded, not analyzed, and not used.

Working resolution:
Each ground-truth mask is read as grayscale and resized to
{RESOLUTION_NAME} (cv2.resize, interpolation = INTER_NEAREST) BEFORE
binarisation and connected-component labelling, so that all areas and
thresholds are expressed in the pixel space used for model training.
RESIZE_TO = {RESIZE_TO} (None = native resolution).

Foreground definition:
A mask pixel is foreground when its grayscale value is > 0 (after the
resize above).

Connected components:
skimage.measure.label(binary_mask, connectivity={CONNECTIVITY})

For a 2-D mask:
connectivity=1 corresponds to 4-connectivity.
connectivity=2 corresponds to 8-connectivity.

This audit uses:
{CONNECTIVITY_NAME}
4-connectivity is evaluated in the connectivity-sensitivity analysis
(section 29).

Component area:
skimage.measure.regionprops(...).area.
For a binary mask this is the number of foreground pixels belonging
to that connected component (in the working-resolution mask).

Region definition:
A Region is operationally defined as a foreground connected component
of the Ground Truth Mask that survives the selected area threshold.

Important:
A computational connected component is not automatically equivalent
to a clinically independent lesion.

Threshold analysis:
Candidate thresholds are:
{THRESHOLDS}
This grid is deliberately dense at small values and sparse at large
ones -- close to log-spaced, not evenly spaced. This matters for
Tier 2 below.

The primary threshold is selected from TRAIN using only mask-derived
statistics. Text-region agreement is explicitly excluded from the
selection rule.

Mask-only selection (two-tier: stability/plateau, then knee/elbow):
Rather than hand-picking a target retention percentage and a minimum
removal percentage (two arbitrary numbers whose choice would directly
and non-transparently determine the outcome), the threshold is chosen
by an empirical, two-tier procedure on mask-derived statistics only.

TIER 1 -- stability / plateau (used when the dataset needs no filtering):
1. For every pair of consecutive candidate thresholds (TRAIN only),
   the relative change in mean_regions and in multi_region_pct is
   computed.
2. The "plateau" is the contiguous run of candidate thresholds,
   starting at the smallest candidate, over which every step's
   relative change stays below a pre-registered tolerance
   ({PRIMARY_STABILITY_TOLERANCE:.1%}).
3. If that plateau spans at least {MIN_PLATEAU_WIDTH_PX} px, the
   smallest candidate threshold is selected (no unnecessary
   filtering), and the plateau width is reported as evidence that the
   decision is not sensitive to exactly which threshold within that
   range is used.
Robustness: this test is repeated at three pre-registered tolerances
(0.5%, 1%, 2%); see threshold_stability_sensitivity_MASK_ONLY.csv.

TIER 2 -- knee / elbow fallback (used only when Tier 1 finds no
plateau at the smallest threshold):
1. The mean_regions and multi_region_pct curves (TRAIN only) are each
   min-max normalized. The "knee" of each is the candidate threshold
   with the maximum perpendicular distance to the straight line
   joining the first and last candidate (Kneedle-style detection,
   Satopaa et al., IEEE ICDCSW 2011).
2. This is computed on BOTH a LINEAR threshold_px axis and a
   log10(threshold_px) axis, for both metrics -- 4 estimates total.
3. SELECTION USES ONLY THE TWO LOG-AXIS ESTIMATES (median of the two,
   snapped to the nearest tested candidate). Reason: the candidate
   grid above is close to log-spaced, not evenly spaced, and Kneedle-
   style knee detection assumes roughly even spacing along whichever
   axis it is given. Feeding it the grid on a LINEAR axis lets a
   handful of large, widely-separated candidates (125, 150, 200, ...,
   500) dominate the chord's geometry, pulling the detected knee
   toward large thresholds for reasons that are an artifact of the
   grid's spacing rather than evidence about the mask data. The two
   LINEAR-axis estimates are still computed and exported in full
   (threshold_knee_elbow_curve_TRAIN.csv,
   threshold_selection_metadata.json) as a transparency diagnostic,
   but are NOT mixed into the selected number.
4. A disagreement diagnostic (ratio of the larger to the smaller of
   the two LOG-axis estimates) is reported.
5. The near-elbow robustness range and the "previous threshold"
   sanity check below are ALSO judged on the log axis (the axis
   actually used for selection), not the linear axis; a linear-axis
   version of the robustness range is exported separately as a
   diagnostic only.

Selection tier actually used in this run:
{selection_tier}

Plateau upper bound at the primary ({PRIMARY_STABILITY_TOLERANCE:.1%})
tolerance: {plateau_upper_bound_px_primary} px
(plateau width: {plateau_width_px_primary} px)

Knee/elbow thresholds (Tier 2, computed regardless of which tier was
finally used, for transparency):
  linear-axis mean_regions elbow      = {elbow_threshold_mean} px  [diagnostic only]
  linear-axis multi_region_pct elbow  = {elbow_threshold_multi} px  [diagnostic only]
  log-axis mean_regions elbow         = {elbow_threshold_mean_logaxis} px  [used if Tier 2 triggers]
  log-axis multi_region_pct elbow     = {elbow_threshold_multi_logaxis} px  [used if Tier 2 triggers]
  log-axis median (raw)               = {log_axis_knee_median_raw:.1f} px
  combined (snapped to candidate)     = {elbow_threshold_combined} px
  log-axis disagreement ratio         = {log_axis_disagreement_ratio:.2f}x
  high disagreement flagged           = {knee_estimates_high_disagreement}

Selected TRAIN threshold ({RESOLUTION_NAME}):
{selected_threshold} px
Selection status:
{selection_status}

Sanity check against the previously used (text-based) threshold:
The threshold previously selected on TRAIN by maximizing exact
agreement with text-derived region counts ({PREVIOUS_TEXT_BASED_THRESHOLD_PX} px,
obtained at native resolution)
is NOT used anywhere in the selection rule above. It is checked here
only as a sanity check (on the log axis) against the mask-only
near-elbow region. Because it was obtained at a different resolution,
the comparison is only indicative.

Validation:
The selected TRAIN threshold is frozen and applied to validation
without modification.

Text annotations:
Text region counts are extracted only for auxiliary consistency analysis.
They are NOT used to define connected components and are NOT used to
select the primary mask-only threshold.

Raw outputs:
1) raw_sample_results.csv:
   one row per image/sample, including raw component count and the complete
   list of component areas.

2) raw_component_results.csv:
   one row per connected component, including image, split, component label,
   area and bounding-box/centroid information.

3) 04_Data_Audit_Raw_Results.csv:
   sample-level audit output containing raw count, threshold, post-filter
   count, component removal, foreground retention and text information.

4) per_sample_all_threshold_results.csv:
   sample-level component counts for every candidate threshold.

Qualitative audit:
Representative samples are selected using mask-derived categories:
no-region, single-region, two-region, multi-region, small-component,
and many-components. Text counts are not used for sample selection.
"""

with open(
    OUTPUT_DIR / "AUDIT_METHODS.txt",
    "w",
    encoding="utf-8"
) as f:
    f.write(method_text.strip())

requirements_text = """numpy>=1.23
pandas>=1.5
opencv-python>=4.7
scikit-image>=0.20
matplotlib>=3.6
tqdm>=4.64
openpyxl>=3.1
"""

with open(
    OUTPUT_DIR / "requirements.txt",
    "w",
    encoding="utf-8"
) as f:
    f.write(requirements_text)

if WRITE_EXCEL:
    excel_path = OUTPUT_DIR / "04_Data_Audit_Raw_Results.xlsx"

    with pd.ExcelWriter(
        excel_path,
        engine="openpyxl"
    ) as writer:

        frozen_df.to_excel(
            writer,
            sheet_name="Sample_Audit",
            index=False
        )

        component_raw.to_excel(
            writer,
            sheet_name="Component_Raw",
            index=False
        )

        threshold_per_sample.to_excel(
            writer,
            sheet_name="All_Thresholds",
            index=False
        )

        threshold_mask_df.to_excel(
            writer,
            sheet_name="Mask_Threshold_Sweep",
            index=False
        )

        train_mask_thresholds.to_excel(
            writer,
            sheet_name="Stability_Step_Changes",
            index=False
        )

        threshold_stability_sensitivity.to_excel(
            writer,
            sheet_name="Stability_Sensitivity",
            index=False
        )

        elbow_curve_df.to_excel(
            writer,
            sheet_name="Knee_Elbow_Curve",
            index=False
        )

        threshold_elbow_robustness.to_excel(
            writer,
            sheet_name="Knee_Elbow_Robustness_LogAxis",
            index=False
        )

        threshold_elbow_robustness_linear_diagnostic.to_excel(
            writer,
            sheet_name="Knee_Elbow_Robustness_LinearDiag",
            index=False
        )

        previous_threshold_check.to_excel(
            writer,
            sheet_name="Previous_TextBased_Sanity_Check",
            index=False
        )

        frozen_summary_df.to_excel(
            writer,
            sheet_name="Frozen_Summary",
            index=False
        )

        text_comparison_summary.to_excel(
            writer,
            sheet_name="Text_Comparison",
            index=False
        )

        area_summary.to_excel(
            writer,
            sheet_name="Area_Percentiles",
            index=False
        )

        area_bins.to_excel(
            writer,
            sheet_name="Area_Bins",
            index=False
        )

        qualitative_df.to_excel(
            writer,
            sheet_name="Qualitative_Audit",
            index=False
        )

RESIZE_LIST = [RESIZE_TO]
if SENS_INCLUDE_NATIVE and RESIZE_TO is not None:
    RESIZE_LIST.append(None)

CONN_VARIANTS = {"4-connectivity": 1, "8-connectivity": 2}
EXTRA_THRESHOLDS = [53, 60, 90]


def get_mask_path(image_name):

    return MASK_DIR / image_name


def areas_and_fg(mask_path, connectivity, resize_to):
    mask, _ = read_mask(mask_path, resize_to)
    binary = (mask > 0).astype(np.uint8)
    labeled = label(binary, connectivity=connectivity)
    areas = np.array(
        [p.area for p in regionprops(labeled)], dtype=np.int64
    )
    return areas, int(binary.sum())


def build_conn_table(connectivity, resize_to, desc):
    rows = []
    for _, r in tqdm(
        sample_df[["image", "split"]].iterrows(),
        total=len(sample_df), desc=desc
    ):
        areas, fg = areas_and_fg(
            get_mask_path(r["image"]), connectivity, resize_to
        )
        rows.append({
            "image": r["image"],
            "split": r["split"],
            "foreground_pixels": fg,
            "raw_component_count": int(len(areas)),
            "raw_component_area_list": json_list(areas),
        })
    return pd.DataFrame(rows)


def rerun_selection(sweep_df):
    t = (
        sweep_df[sweep_df["split"] == "train"]
        .sort_values("threshold_px").reset_index(drop=True)
    )
    cands = [int(x) for x in t["threshold_px"]]

    rc_mean = relative_step_changes(t["mean_regions"])
    rc_multi = relative_step_changes(t["multi_region_pct"])
    ub = min(
        plateau_upper_bound_px(
            cands, rc_mean, PRIMARY_STABILITY_TOLERANCE),
        plateau_upper_bound_px(
            cands, rc_multi, PRIMARY_STABILITY_TOLERANCE),
    )
    width = ub - cands[0]
    plateau_found = bool(width >= MIN_PLATEAU_WIDTH_PX)

    log_t = np.log10(t["threshold_px"].to_numpy(dtype=float))
    k_mean = int(t["threshold_px"].iloc[int(np.argmax(
        chord_distance_curve(log_t, t["mean_regions"])))])
    k_multi = int(t["threshold_px"].iloc[int(np.argmax(
        chord_distance_curve(log_t, t["multi_region_pct"])))])
    combined = nearest_candidate_threshold(
        float(np.median([k_mean, k_multi])), cands
    )

    return {
        "plateau_found": plateau_found,
        "plateau_width_px": int(width),
        "log_axis_knee_mean_regions_px": k_mean,
        "log_axis_knee_multi_region_pct_px": k_multi,
        "knee_combined_px": int(combined),
        "selected_threshold_px": int(
            cands[0] if plateau_found else combined
        ),
        "selection_tier": (
            "tier1_stability_plateau" if plateau_found
            else "tier2_knee_elbow"
        ),
    }


def run_connectivity_sensitivity(resize_to):
    tag = "native" if resize_to is None else f"{resize_to}x{resize_to}"
    is_main_resolution = (resize_to == RESIZE_TO)

    print("\n" + "=" * 90)
    print(
        f"CONNECTIVITY SENSITIVITY -- resolution: {tag}"
        f"{'  (MAIN)' if is_main_resolution else '  (comparison)'}"
    )
    print("=" * 90)

    tables, sweep_parts, sel_rows = {}, [], []

    for conn_name, conn_val in CONN_VARIANTS.items():
        if conn_val == CONNECTIVITY and is_main_resolution:
            tbl = sample_df[[
                "image", "split", "foreground_pixels",
                "raw_component_count", "raw_component_area_list"
            ]].copy()
        else:
            tbl = build_conn_table(
                conn_val, resize_to, f"{conn_name} [{tag}]"
            )
        tables[conn_name] = tbl

        rows = []
        for th in THRESHOLDS:
            for sp in ["train", "validation"]:
                m = threshold_mask_metrics(
                    tbl[tbl["split"] == sp].copy(), th
                )
                m["split"] = sp
                m["connectivity"] = conn_name
                m["resolution"] = tag
                rows.append(m)
        sweep = pd.DataFrame(rows)
        sweep_parts.append(sweep)

        sel = rerun_selection(sweep)
        sel["connectivity"] = conn_name
        sel["resolution"] = tag
        sel_rows.append(sel)

    sweep_df = pd.concat(sweep_parts, ignore_index=True)
    sel_df = pd.DataFrame(sel_rows)

    thr8 = int(sel_df[
        sel_df["connectivity"] == CONNECTIVITY_NAME
    ]["selected_threshold_px"].iloc[0])

    report_thresholds = sorted(set(
        [int(selected_threshold), thr8] + EXTRA_THRESHOLDS
    ))

    summary_rows, agree_rows = [], []
    for thr in report_thresholds:
        fz = {
            cn: add_frozen_threshold_columns(tables[cn], thr)
            for cn in CONN_VARIANTS
        }
        for cn in CONN_VARIANTS:
            for sp in ["train", "validation"]:
                s = summary_at_threshold(fz[cn], sp, thr)
                s["connectivity"] = cn
                s["resolution"] = tag
                summary_rows.append(s)

        a = fz["4-connectivity"][[
            "image", "split", "raw_component_count",
            "component_count_after_filtering"
        ]].rename(columns={
            "raw_component_count": "raw_k4",
            "component_count_after_filtering": "k4"})
        b = fz["8-connectivity"][[
            "image", "raw_component_count",
            "component_count_after_filtering"
        ]].rename(columns={
            "raw_component_count": "raw_k8",
            "component_count_after_filtering": "k8"})
        cmp_df = a.merge(b, on="image")

        for sp in ["train", "validation"]:
            x = cmp_df[cmp_df["split"] == sp]
            agree_rows.append({
                "resolution": tag,
                "split": sp,
                "threshold_px": thr,
                "samples": len(x),
                "pct_same_count_after_filter":
                    100.0 * np.mean(x["k4"] == x["k8"]),
                "pct_same_count_raw_before_filter":
                    100.0 * np.mean(x["raw_k4"] == x["raw_k8"]),
                "pct_multi_flag_changed":
                    100.0 * np.mean((x["k4"] >= 2) != (x["k8"] >= 2)),
                "mean_regions_4conn": float(x["k4"].mean()),
                "mean_regions_8conn": float(x["k8"].mean()),
                "mean_diff_4_minus_8":
                    float((x["k4"] - x["k8"]).mean()),
            })

    summary_df = pd.DataFrame(summary_rows)
    agree_df = pd.DataFrame(agree_rows)

    sweep_df.to_csv(
        OUTPUT_DIR / f"connsens_{tag}_threshold_sweep.csv", index=False)
    sel_df.to_csv(
        OUTPUT_DIR / f"connsens_{tag}_threshold_selection.csv",
        index=False)
    summary_df.to_csv(
        OUTPUT_DIR / f"connsens_{tag}_summary_by_threshold.csv",
        index=False)
    agree_df.to_csv(
        OUTPUT_DIR / f"connsens_{tag}_4_vs_8_agreement.csv",
        index=False)

    print("\nThreshold re-selected with the SAME rule per connectivity:")
    print(sel_df.to_string(index=False))
    print(f"\nThresholds reported: {report_thresholds}")
    print("\nSummary (mean_regions / multi_region_pct) per threshold:")
    print(summary_df[[
        "resolution", "connectivity", "split", "threshold_px",
        "mean_regions", "multi_region_pct", "no_region_pct"
    ]].to_string(index=False))
    print("\n4- vs 8-connectivity agreement:")
    print(agree_df.to_string(index=False))

    if is_main_resolution:
        print(
            "\nSanity check (8-conn re-selection == main run): "
            f"{thr8 == int(selected_threshold)}"
        )
    return sel_df, summary_df, agree_df

connsens_results = {}
for _r in RESIZE_LIST:
    connsens_results[_r] = run_connectivity_sensitivity(_r)

print()
print("=" * 90)
print("FINAL QaTa-COV19 DATA AUDIT SUMMARY")
print("=" * 90)

print(f"Working resolution:           {RESOLUTION_NAME}")
print(f"Connectivity:                 {CONNECTIVITY_NAME}")
print(f"skimage connectivity:         {CONNECTIVITY}")
print(f"Train samples analyzed:       {len(train_images):,}")
print(f"Validation samples analyzed:  {len(validation_images):,}")
print(f"Raw components:               {len(component_df):,}")
print(f"Selected TRAIN threshold:     {selected_threshold} px")
print(f"Threshold selection basis:    MASK ONLY ({selection_tier})")
print("Validation threshold policy:  FROZEN TRAIN THRESHOLD")
print("Text used for threshold:      NO")

print()
print("Frozen threshold summary:")
print(
    frozen_summary_df.to_string(
        index=False
    )
)

print()
print("Text comparison (auxiliary only):")
print(
    text_comparison_summary.to_string(
        index=False
    )
)

print()
print("Main output files:")
for path in sorted(OUTPUT_DIR.rglob("*")):
    if path.is_file():
        print(" -", path)

print()
print("=" * 90)
print("AUDIT FINISHED")
print("=" * 90)
