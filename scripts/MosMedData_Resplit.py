import os
import argparse
import re
import glob
import json
import hashlib
import platform
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths (command-line arguments; defaults assume the repository data layout
# described in README.md and that the script is run from the repository root).
# ---------------------------------------------------------------------------
_ap = argparse.ArgumentParser(
    description="MosMedData+ group-aware, source-stratified re-split.")
_ap.add_argument("--mask_root", default="data/MosMedData/masks",
                 help="Folder with the MosMedData+ lesion masks (searched recursively).")
_ap.add_argument("--train_csv", default="data/MosMedData/text/Train_text_MosMedData 1(in).csv",
                 help="Original MosMedData+ TRAIN text CSV.")
_ap.add_argument("--val_csv", default="data/MosMedData/text/Val_text_MosMedData 1(in).csv",
                 help="Original MosMedData+ VALIDATION text CSV.")
_ap.add_argument("--test_csv", default="data/MosMedData/text/Test_text_MosMedData(in).csv",
                 help="Original MosMedData+ TEST text CSV. Pass 'none' to infer the test "
                      "set as the remainder of --mask_root.")
_ap.add_argument("--output_dir", default="outputs/MosMedData_Resplit_SourceStratified",
                 help="Folder where the new split CSVs, reports and manifest are written.")
_cli, _ = _ap.parse_known_args()

MASK_ROOT = _cli.mask_root
TRAIN_CSV_PATH = _cli.train_csv
VAL_CSV_PATH = _cli.val_csv
TEST_CSV_PATH = None if str(_cli.test_csv).strip().lower() in ("", "none") else _cli.test_csv
TEST_IMAGE_COLUMN_OVERRIDE = None
IMAGE_COLUMN_OVERRIDE = None

OUTPUT_DIR = _cli.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

TEST_FRACTION = 0.10
VAL_FRACTION = 0.10
RANDOM_STATE = 42

SPLIT_STRATEGY = "group_aware_source_stratified"

DEFAULT_STUDY_ID_REGEX = re.compile(r"^(?P<study_id>.+?)_\d+$")

SOURCE_PATTERNS = [
    ("Jun_coronacases", re.compile(r"^Jun_coronacases", re.IGNORECASE)),
    ("Jun_radiopaedia", re.compile(r"^Jun_radiopaedia", re.IGNORECASE)),
    ("Morozov",         re.compile(r"^Morozov", re.IGNORECASE)),
    ("bjorke",          re.compile(r"^bjorke", re.IGNORECASE)),
]

N_RESTARTS = 60
N_ITER_PER_RESTART = 4000
STRATIFICATION_WEIGHT = 1.0
GLOBAL_WEIGHT = 2.0
GLOBAL_TOLERANCE = 0.03
GLOBAL_TOLERANCE_PENALTY = 50.0
MISSING_SOURCE_PENALTY = 0.05
MIN_GROUPS_FOR_FULL_COVERAGE = 3

TEXT_COLUMN_CANDIDATES = ["text", "Text", "report", "Report", "caption", "Caption",
                          "label", "Label", "description", "Description"]

SPLITS = ["train", "validation", "test"]
TARGET_FRACTIONS = np.array([
    1.0 - VAL_FRACTION - TEST_FRACTION,
    VAL_FRACTION,
    TEST_FRACTION,
])

TERMINOLOGY_NOTE = (
    "The group ID is a FILENAME-DERIVED study/group ID (final '_<number>' "
    "slice suffix removed by regex). It is not confirmed by official dataset "
    "metadata and must not be described as a confirmed patient-level ID."
)


def extract_group_id(filename):
    stem = os.path.splitext(os.path.basename(str(filename)))[0]
    m = DEFAULT_STUDY_ID_REGEX.match(stem)
    if m and m.group("study_id"):
        return m.group("study_id")
    return stem


def extract_source(filename):
    stem = os.path.splitext(os.path.basename(str(filename)))[0]
    for label, pattern in SOURCE_PATTERNS:
        if pattern.search(stem):
            return label, False
    token = stem.split("_")[0] if stem else "unknown"
    return f"other:{token}", True


def normalize_image_name(x):
    if pd.isna(x):
        return ""
    x = str(x).strip().replace("\\", "/")
    return os.path.basename(x)


def find_column(columns, candidates, exclude=None):
    normalized = {str(c).strip().lower(): c for c in columns}
    for candidate in candidates:
        key = str(candidate).strip().lower()
        if key in normalized and normalized[key] != exclude:
            return normalized[key]
    return None


def recursive_index(root, extensions):
    paths = []
    for ext in extensions:
        paths.extend(glob.glob(os.path.join(root, "**", ext), recursive=True))
    return sorted(set(paths))


def is_missing_text(series):
    return series.isna() | (series.astype(str).str.strip() == "")


def sha256_of_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_full_csv(csv_path, split_name, image_col_override=None):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{split_name} CSV not found:\n{csv_path}")
    raw = pd.read_csv(csv_path)
    image_candidates = ["Image", "image", "filename", "Filename",
                        "image_name", "ImageName", "Imagetext"]
    image_col = (
        image_col_override
        or find_column(raw.columns, image_candidates)
        or raw.columns[0]
    )

    df = raw.copy()
    df["image_key"] = df[image_col].apply(normalize_image_name)
    df = df[df["image_key"] != ""].copy()

    before = len(df)
    df = df.drop_duplicates(subset="image_key", keep="first").reset_index(drop=True)
    if len(df) < before:
        print(f"  WARNING {split_name}: dropped {before - len(df)} duplicate rows")

    if image_col != "Image":
        df = df.rename(columns={image_col: "Image"})

    other_cols = [c for c in df.columns if c not in ("Image", "image_key")]
    print(f"{split_name}: {len(raw):,} rows -> {len(df):,} unique image rows "
          f"(image column = '{image_col}'); other columns kept: {other_cols}")
    return df


def report_text_completeness(df, name, text_col):
    total = len(df)
    if text_col is None or text_col not in df.columns:
        print(f"  {name:<28s}: {total:,} rows -- no text column, skipped")
        return total, None
    n_missing = int(is_missing_text(df[text_col]).sum())
    pct = (n_missing / total * 100) if total else 0.0
    print(f"  {name:<28s}: {total:,} rows, missing/empty text = "
          f"{n_missing:,} ({pct:.2f}%)")
    return total, n_missing

print("=" * 90)
print("STEP 1 -- Gathering the full pool from the CURRENT (original) split")
print("=" * 90)

train_df = load_full_csv(TRAIN_CSV_PATH, "TRAIN (original)", IMAGE_COLUMN_OVERRIDE)
val_df = load_full_csv(VAL_CSV_PATH, "VALIDATION (original)", IMAGE_COLUMN_OVERRIDE)
train_keys = set(train_df["image_key"])
val_keys = set(val_df["image_key"])

if TEST_CSV_PATH is not None:
    test_df = load_full_csv(TEST_CSV_PATH, "TEST (original)", TEST_IMAGE_COLUMN_OVERRIDE)
else:
    extensions = ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp"]
    all_mask_images = sorted(set(os.path.basename(p)
                                 for p in recursive_index(MASK_ROOT, extensions)))
    test_only_keys = sorted(set(all_mask_images) - train_keys - val_keys)
    test_df = pd.DataFrame({"Image": test_only_keys, "image_key": test_only_keys})
    print(f"TEST (original, inferred as remainder of MASK_ROOT): {len(test_df):,} images")
    print("  WARNING: no TEST_CSV_PATH -- these images have NO text.")

text_col = (find_column(train_df.columns, TEXT_COLUMN_CANDIDATES)
            or find_column(val_df.columns, TEXT_COLUMN_CANDIDATES)
            or find_column(test_df.columns, TEXT_COLUMN_CANDIDATES))
print(f"\nDetected text column: {text_col!r}")

print("\n" + "-" * 90)
print("BEFORE SPLIT -- empty/missing text check (source CSVs)")
print("-" * 90)
report_text_completeness(train_df, "TRAIN (original)", text_col)
report_text_completeness(val_df, "VALIDATION (original)", text_col)
report_text_completeness(test_df, "TEST (original)", text_col)

tagged = []
for name, d in (("train", train_df), ("validation", val_df), ("test", test_df)):
    d = d.copy()
    d["original_split"] = name
    tagged.append(d)

pool_raw = pd.concat(tagged, ignore_index=True, sort=False)
before = len(pool_raw)
pool_df = pool_raw.drop_duplicates(subset="image_key", keep="first").reset_index(drop=True)
if len(pool_df) < before:
    print(f"\nWARNING: dropped {before - len(pool_df)} images listed in more than one "
          f"source file (priority train > validation > test).")

pool_df["group_id"] = pool_df["image_key"].apply(extract_group_id)
src = pool_df["image_key"].apply(extract_source)
pool_df["source"] = [s for s, _ in src]
pool_df["source_is_fallback"] = [f for _, f in src]

if text_col is not None and text_col in pool_df.columns:
    pool_df["_missing_text"] = is_missing_text(pool_df[text_col])
else:
    pool_df["_missing_text"] = False

src_per_group = pool_df.groupby("group_id")["source"].nunique()
assert (src_per_group == 1).all(), (
    "Some filename-derived group IDs map to more than one source: "
    f"{list(src_per_group[src_per_group > 1].index)[:10]}"
)

n_fallback = int(pool_df["source_is_fallback"].sum())
if n_fallback:
    print(f"\nWARNING: {n_fallback:,} images did not match any SOURCE_PATTERNS "
          f"entry (fallback label 'other:<first-token>'). "
          f"Labels: {sorted(pool_df.loc[pool_df.source_is_fallback, 'source'].unique())}")

print(f"\nTotal pooled images : {len(pool_df):,}")
print(f"Filename-derived group IDs : {pool_df['group_id'].nunique():,}")
print(f"Sources found       : {sorted(pool_df['source'].unique())}")

print("\nREGEX SANITY CHECK -- example filename -> filename-derived group ID per source:")
for s in sorted(pool_df["source"].unique()):
    sub = pool_df.loc[pool_df["source"] == s, ["image_key", "group_id"]]
    for _, r in sub.head(3).iterrows():
        print(f"  [{s}] {r['image_key']}  ->  {r['group_id']}")
    print(f"  [{s}] distinct group IDs: {sub['group_id'].nunique():,} "
          f"for {len(sub):,} images")

orig_membership = pool_df.groupby("group_id")["original_split"].nunique()
leaking_orig_groups = sorted(orig_membership[orig_membership > 1].index)
n_leak_orig = len(leaking_orig_groups)
n_leak_orig_images = int(pool_df["group_id"].isin(leaking_orig_groups).sum())
print(f"\nORIGINAL split leakage: {n_leak_orig:,} of {len(orig_membership):,} "
      f"filename-derived groups appear in more than one split "
      f"({n_leak_orig_images:,} images belong to such groups).")

ORIGINAL_OUTPUT_COLUMNS = [c for c in pool_df.columns
                           if c not in ("image_key", "group_id", "source",
                                        "source_is_fallback", "original_split",
                                        "_missing_text", "split")]
if "Image" in ORIGINAL_OUTPUT_COLUMNS:
    ORIGINAL_OUTPUT_COLUMNS = ["Image"] + [c for c in ORIGINAL_OUTPUT_COLUMNS if c != "Image"]
print(f"Columns kept in every per-split CSV: {ORIGINAL_OUTPUT_COLUMNS}")

group_df = (
    pool_df.groupby("group_id")
    .agg(source=("source", "first"),
         n_images=("image_key", "count"),
         has_missing_text=("_missing_text", "any"))
    .reset_index()
    .sort_values("group_id")
    .reset_index(drop=True)
)

forced_test_groups = set(group_df.loc[group_df.has_missing_text, "group_id"])
print(f"\nGroups forced to TEST (contain >= 1 missing-text image): "
      f"{len(forced_test_groups):,}")

feas_rows = []
for s, g in group_df.groupby("source"):
    largest = int(g.n_images.max())
    total = int(g.n_images.sum())
    feas_rows.append({
        "source": s,
        "n_groups": len(g),
        "n_images": total,
        "largest_group_images": largest,
        "largest_group_share_of_source_pct": 100.0 * largest / total,
        "can_appear_in_all_3_splits": len(g) >= MIN_GROUPS_FOR_FULL_COVERAGE,
        "note": ("single group: cannot be spread over several splits without "
                 "breaking group integrity" if len(g) == 1 else
                 ("fewer than 3 groups: cannot cover all 3 splits" if len(g) < 3 else "")),
    })
feasibility_df = pd.DataFrame(feas_rows).sort_values("n_images", ascending=False)
feasibility_df.to_csv(os.path.join(OUTPUT_DIR, "source_stratification_feasibility.csv"),
                      index=False)
print("\nSOURCE STRATIFICATION FEASIBILITY (indivisible groups):")
print(feasibility_df.to_string(index=False))

groups = group_df["group_id"].tolist()
src_labels = sorted(group_df["source"].unique())
src_index = group_df["source"].map({s: i for i, s in enumerate(src_labels)}).to_numpy()
sizes = group_df["n_images"].to_numpy(dtype=float)
n_groups_per_source = np.bincount(src_index, minlength=len(src_labels))
forced_mask = group_df["group_id"].isin(forced_test_groups).to_numpy()
movable = np.where(~forced_mask)[0]
K = len(SPLITS)
TEST_IDX = SPLITS.index("test")


def baseline_group_only_assignment():
    total = sizes.sum()
    targets = TARGET_FRACTIONS * total
    current = np.zeros(K)
    assign = np.full(len(groups), -1, dtype=int)

    for i in np.where(forced_mask)[0]:
        assign[i] = TEST_IDX
        current[TEST_IDX] += sizes[i]

    free = group_df.loc[~forced_mask, ["group_id", "n_images"]].copy()
    free = free.sample(frac=1.0, random_state=RANDOM_STATE)
    free = free.sort_values("n_images", ascending=False, kind="stable")
    pos = {g: i for i, g in enumerate(groups)}
    for _, row in free.iterrows():
        i = pos[row["group_id"]]
        deficits = targets - current
        k = int(np.argmax(deficits))
        assign[i] = k
        current[k] += sizes[i]
    return assign


def objective(assign):
    S = len(src_labels)
    img = np.zeros((S, K))
    grp = np.zeros((S, K))
    np.add.at(img, (src_index, assign), sizes)
    np.add.at(grp, (src_index, assign), 1.0)

    src_tot = img.sum(axis=1)
    p = img / src_tot[:, None]
    w = src_tot / src_tot.sum()
    strat = (w[:, None] * (p - TARGET_FRACTIONS[None, :]) ** 2).sum()

    P = img.sum(axis=0) / img.sum()
    glob_term = ((P - TARGET_FRACTIONS) ** 2).sum()
    excess = np.maximum(0.0, np.abs(P - TARGET_FRACTIONS) - GLOBAL_TOLERANCE)
    glob_pen = (excess ** 2).sum() * GLOBAL_TOLERANCE_PENALTY

    missing = ((grp == 0) & (n_groups_per_source[:, None] >= MIN_GROUPS_FOR_FULL_COVERAGE)).sum()

    return (STRATIFICATION_WEIGHT * strat + GLOBAL_WEIGHT * glob_term
            + glob_pen + MISSING_SOURCE_PENALTY * missing)


def optimise_stratified(init_assign):
    rng = np.random.RandomState(RANDOM_STATE)
    best = init_assign.copy()
    best_obj = objective(best)

    for r in range(N_RESTARTS):
        cur = best.copy()
        if r > 0:
            n_perturb = max(1, int(0.15 * len(movable)))
            for i in rng.choice(movable, size=n_perturb, replace=False):
                cur[i] = rng.randint(K)
        cur_obj = objective(cur)

        for _ in range(N_ITER_PER_RESTART):
            if rng.rand() < 0.5:
                i = int(rng.choice(movable))
                k = int(rng.randint(K))
                if k == cur[i]:
                    continue
                old = cur[i]
                cur[i] = k
                o = objective(cur)
                if o < cur_obj - 1e-12:
                    cur_obj = o
                else:
                    cur[i] = old
            else:
                if len(movable) < 2:
                    continue
                i, j = rng.choice(movable, size=2, replace=False)
                if cur[i] == cur[j]:
                    continue
                cur[i], cur[j] = cur[j], cur[i]
                o = objective(cur)
                if o < cur_obj - 1e-12:
                    cur_obj = o
                else:
                    cur[i], cur[j] = cur[j], cur[i]

        if cur_obj < best_obj - 1e-12:
            best, best_obj = cur.copy(), cur_obj

    return best, best_obj

print("\n" + "=" * 90)
print("STEP 2 -- Group-aware split")
print("=" * 90)

baseline_assign = baseline_group_only_assignment()
baseline_obj = objective(baseline_assign)
print(f"Baseline (group-only LPT) objective : {baseline_obj:.5f}")

stratified_assign, stratified_obj = optimise_stratified(baseline_assign)
print(f"Source-stratified objective         : {stratified_obj:.5f} "
      f"(lower = closer to per-source target fractions)")

final_assign = stratified_assign if SPLIT_STRATEGY == "group_aware_source_stratified" \
    else baseline_assign
print(f"Strategy used for the FINAL split   : {SPLIT_STRATEGY}")

group_df["split_group_only_baseline"] = [SPLITS[k] for k in baseline_assign]
group_df["split_final"] = [SPLITS[k] for k in final_assign]

pool_df = pool_df.merge(
    group_df[["group_id", "split_group_only_baseline", "split_final"]],
    on="group_id", how="left"
)
pool_df["split"] = pool_df["split_final"]


def distribution_long(df, split_col, scheme):
    rows = []
    split_img_tot = df.groupby(split_col).size()
    split_grp_tot = df.groupby(split_col)["group_id"].nunique()
    src_img_tot = df.groupby("source").size()
    for s in src_labels:
        for sp in SPLITS:
            sub = df[(df.source == s) & (df[split_col] == sp)]
            n_img = len(sub)
            n_grp = sub["group_id"].nunique()
            rows.append({
                "scheme": scheme,
                "source": s,
                "split": sp,
                "n_groups": n_grp,
                "pct_of_split_groups": (100.0 * n_grp / split_grp_tot.get(sp, np.nan)
                                        if split_grp_tot.get(sp, 0) else np.nan),
                "n_images": n_img,
                "pct_of_split_images": (100.0 * n_img / split_img_tot.get(sp, np.nan)
                                        if split_img_tot.get(sp, 0) else np.nan),
                "pct_of_source_images": 100.0 * n_img / src_img_tot[s],
            })
    return pd.DataFrame(rows)


def composition_distance(df, split_col, scheme):
    comp = {}
    for sp in SPLITS:
        sub = df[df[split_col] == sp]
        shares = sub["source"].value_counts(normalize=True)
        comp[sp] = np.array([shares.get(s, 0.0) for s in src_labels])
    rows = []
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        rows.append({
            "scheme": scheme,
            "pair": f"{a} vs {b}",
            "total_variation_distance_source_composition":
                float(0.5 * np.abs(comp[a] - comp[b]).sum()),
        })
    return rows

dist_original = distribution_long(pool_df, "original_split", "original_split")
dist_baseline = distribution_long(pool_df, "split_group_only_baseline",
                                  "group_only_baseline")
dist_final = distribution_long(pool_df, "split_final", f"FINAL:{SPLIT_STRATEGY}")

distribution_all = pd.concat([dist_original, dist_baseline, dist_final],
                             ignore_index=True)
distribution_all.to_csv(
    os.path.join(OUTPUT_DIR, "source_distribution_by_split_all_schemes.csv"), index=False)
dist_final.to_csv(
    os.path.join(OUTPUT_DIR, "source_distribution_by_split_FINAL.csv"), index=False)

tvd_df = pd.DataFrame(
    composition_distance(pool_df, "original_split", "original_split")
    + composition_distance(pool_df, "split_group_only_baseline", "group_only_baseline")
    + composition_distance(pool_df, "split_final", f"FINAL:{SPLIT_STRATEGY}")
)
tvd_df.to_csv(os.path.join(OUTPUT_DIR, "split_source_composition_distance.csv"),
              index=False)


def wide_view(dist, value_col):
    return dist.pivot(index="source", columns="split", values=value_col)[SPLITS]


def print_scheme(dist, title, note=None):
    print("\n" + "-" * 90)
    print(title)
    print("-" * 90)
    imgs = wide_view(dist, "n_images")
    grps = wide_view(dist, "n_groups")
    comp = wide_view(dist, "pct_of_split_images").round(1)
    strat = wide_view(dist, "pct_of_source_images").round(1)
    print("Images per source/split:")
    print(imgs.assign(total=imgs.sum(axis=1)).to_string())
    print("\nGroups per source/split:")
    print(grps.assign(total=grps.sum(axis=1)).to_string())
    if note:
        print(f"  NOTE: {note}")
    print("\nComposition (% of each split's images coming from each source):")
    print(comp.to_string())
    print("\nSpread (% of each source's images located in each split; "
          f"target = {[round(100*t) for t in TARGET_FRACTIONS]}):")
    print(strat.to_string())

print_scheme(
    dist_original, "ORIGINAL SPLIT (not group-aware)",
    note=(f"{n_leak_orig} groups appear in several splits, so the 'total' of the "
          f"group counts exceeds the true number of distinct groups "
          f"({group_df.shape[0]}). This is the original group leakage."))
print_scheme(dist_baseline, "GROUP-ONLY BASELINE (no source awareness)")
print_scheme(dist_final, f"FINAL SPLIT ({SPLIT_STRATEGY})")

print("\n" + "-" * 90)
print("SOURCE-COMPOSITION DISTANCE BETWEEN SPLITS "
      "(total variation, 0 = identical mix, 1 = disjoint)")
print("-" * 90)
print(tvd_df.pivot(index="pair", columns="scheme",
                   values="total_variation_distance_source_composition")
      .round(3).to_string())

new_splits = {sp: pool_df[pool_df["split"] == sp].sort_values("image_key")
              .reset_index(drop=True) for sp in SPLITS}

group_sets = {sp: set(new_splits[sp]["group_id"]) for sp in SPLITS}
image_sets = {sp: set(new_splits[sp]["image_key"]) for sp in SPLITS}

n_group_leak_final = 0
n_image_leak_final = 0
for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
    n_group_leak_final += len(group_sets[a] & group_sets[b])
    n_image_leak_final += len(image_sets[a] & image_sets[b])
assert n_group_leak_final == 0, "Group overlap between splits"
assert n_image_leak_final == 0, "Image overlap between splits"
assert sum(len(v) for v in new_splits.values()) == len(pool_df), "Images lost/duplicated"
print("\nSafety checks passed: zero group overlap and zero image overlap between splits; "
      "every pooled image is assigned exactly once.")

if text_col is not None:
    for sp in ("train", "validation"):
        assert new_splits[sp]["_missing_text"].sum() == 0, \
            f"{sp} contains missing-text rows"
    print("Safety check passed: zero missing/empty text in Train and Validation.")

size_warnings = []
for sp, t in zip(SPLITS, TARGET_FRACTIONS):
    frac = len(new_splits[sp]) / len(pool_df)
    if abs(frac - t) > GLOBAL_TOLERANCE:
        msg = f"{sp} share {frac:.3f} deviates from target {t:.2f} by more than {GLOBAL_TOLERANCE}"
        size_warnings.append(msg)
        print(f"WARNING: {msg}")
if not size_warnings:
    print(f"Split-size check passed: every split is within +/-{GLOBAL_TOLERANCE:.2f} "
          f"of its target fraction.")

out_files = {
    "train": "Train_resplit_by_group.csv",
    "validation": "Val_resplit_by_group.csv",
    "test": "Test_resplit_by_group.csv",
}
for sp, fname in out_files.items():
    new_splits[sp][ORIGINAL_OUTPUT_COLUMNS].to_csv(
        os.path.join(OUTPUT_DIR, fname), index=False)

full_cols = ["image_key", "group_id", "source", "split"] + ORIGINAL_OUTPUT_COLUMNS
full_table = pd.concat(
    [new_splits[sp].assign(split=sp) for sp in SPLITS], ignore_index=True
)[full_cols].rename(columns={"group_id": "filename_derived_group_id"})
full_table.to_csv(os.path.join(OUTPUT_DIR, "resplit_full_table.csv"), index=False)

group_assignment = group_df.rename(columns={"group_id": "filename_derived_group_id"})
group_assignment.to_csv(os.path.join(OUTPUT_DIR, "assignment_by_group.csv"), index=False)

pd.DataFrame({"filename_derived_group_id": leaking_orig_groups}).to_csv(
    os.path.join(OUTPUT_DIR, "original_split_leaking_groups.csv"), index=False)

print(f"\nFinal files written to: {OUTPUT_DIR}")

print("\n" + "-" * 90)
print("AFTER SPLIT -- empty/missing text check")
print("-" * 90)
missing_after = {}
for sp in SPLITS:
    _, m = report_text_completeness(new_splits[sp], f"NEW {sp.upper()}", text_col)
    missing_after[sp] = m

total_images = len(pool_df)
for sp in SPLITS:
    n_img = len(new_splits[sp])
    n_grp = new_splits[sp]["group_id"].nunique()
    print(f"NEW {sp.upper():<10s}: {n_img:,} images ({n_img/total_images:.1%}), "
          f"{n_grp:,} filename-derived groups")

split_sizes = {sp: {"images": int(len(new_splits[sp])),
                    "groups": int(new_splits[sp]["group_id"].nunique())}
               for sp in SPLITS}

not_covered = [
    s for s in src_labels
    if (wide_view(dist_final, "n_images").loc[s] == 0).any()
]

verify_path = os.path.join(OUTPUT_DIR, "resplit_verification_report.txt")
with open(verify_path, "w", encoding="utf-8") as f:
    f.write("MosMedData+ | Group-aware + Source-stratified Resplit -- Verification\n")
    f.write("=" * 78 + "\n\n")
    f.write("TERMINOLOGY\n" + TERMINOLOGY_NOTE + "\n\n")
    f.write(f"Strategy             : {SPLIT_STRATEGY}\n")
    f.write(f"Random state         : {RANDOM_STATE}\n")
    f.write(f"Target fractions     : train={TARGET_FRACTIONS[0]:.2f}, "
            f"validation={TARGET_FRACTIONS[1]:.2f}, test={TARGET_FRACTIONS[2]:.2f}\n")
    f.write(f"Total images         : {total_images:,}\n")
    f.write(f"Filename-derived groups: {group_df.shape[0]:,}\n")
    for sp in SPLITS:
        f.write(f"  {sp:<11s}: {split_sizes[sp]['images']:,} images "
                f"({split_sizes[sp]['images']/total_images:.1%}), "
                f"{split_sizes[sp]['groups']:,} groups\n")
    f.write("  Note: split sizes are approximately (not exactly) equal to the targets "
            "because groups are indivisible.\n")
    if size_warnings:
        f.write("  Size warnings: " + "; ".join(size_warnings) + "\n")
    f.write(f"\nColumns preserved in each split file: {ORIGINAL_OUTPUT_COLUMNS}\n")
    f.write(f"Text column: {text_col!r}\n")
    f.write(f"Groups forced to Test (missing text): {len(forced_test_groups):,}\n")
    f.write(f"Missing/empty text after split: {missing_after}\n\n")
    f.write("LEAKAGE OF THE ORIGINAL SPLIT\n")
    f.write(f"  Groups appearing in more than one original split : "
            f"{n_leak_orig:,} of {len(orig_membership):,} "
            f"({n_leak_orig_images:,} images)\n\n")
    f.write("GROUP / IMAGE LEAKAGE CHECK (final split, computed)\n")
    f.write(f"  Groups appearing in more than one split : {n_group_leak_final}\n")
    f.write(f"  Images appearing in more than one split : {n_image_leak_final}\n\n")
    f.write("SOURCE FEASIBILITY (indivisible groups)\n")
    f.write(feasibility_df.to_string(index=False) + "\n\n")
    f.write("SOURCE SPREAD OF THE FINAL SPLIT (% of each source's images per split)\n")
    f.write(wide_view(dist_final, "pct_of_source_images").round(1).to_string() + "\n\n")
    f.write("SOURCE COMPOSITION OF THE FINAL SPLIT (% of each split's images per source)\n")
    f.write(wide_view(dist_final, "pct_of_split_images").round(1).to_string() + "\n\n")
    f.write("SOURCE-COMPOSITION DISTANCE BETWEEN SPLITS (total variation)\n")
    f.write(tvd_df.pivot(index="pair", columns="scheme",
                         values="total_variation_distance_source_composition")
            .round(3).to_string() + "\n\n")
    f.write("SOURCES NOT PRESENT IN ALL THREE SPLITS (explicit limitation)\n")
    f.write((", ".join(not_covered) if not_covered else "none") + "\n")
    if not_covered:
        f.write("Sources listed above have too few indivisible filename-derived "
                "groups (see feasibility table) and therefore stay in a single "
                "split (Train). This is a structural limit of group-aware "
                "splitting, not an oversight; moving such a group into "
                "Validation/Test would remove most of that source from Train.\n")
    f.write("\nLIMITATION\n"
            "Source stratification controls the SOURCE MIX of each split only; "
            "it minimises, but does not eliminate, source-related domain shift. "
            "It does not remove differences inside a source (scanner, protocol, "
            "annotation style), and the filename-derived group ID is not "
            "confirmed by official metadata. A single-group source may itself "
            "contain several scans/patients that cannot be separated using "
            "filenames alone.\n")

manifest = {
    "description": "Fixed, reproducible MosMedData+ group-aware / source-stratified split",
    "terminology_note": TERMINOLOGY_NOTE,
    "strategy": SPLIT_STRATEGY,
    "random_state": RANDOM_STATE,
    "target_fractions": dict(zip(SPLITS, TARGET_FRACTIONS.tolist())),
    "group_id_regex": DEFAULT_STUDY_ID_REGEX.pattern,
    "source_patterns": [(label, p.pattern) for label, p in SOURCE_PATTERNS],
    "original_split_leaking_groups": n_leak_orig,
    "original_split_images_in_leaking_groups": n_leak_orig_images,
    "final_split_leaking_groups": n_group_leak_final,
    "sources_not_in_all_splits": not_covered,
    "optimiser": {
        "n_restarts": N_RESTARTS, "n_iter_per_restart": N_ITER_PER_RESTART,
        "stratification_weight": STRATIFICATION_WEIGHT,
        "global_weight": GLOBAL_WEIGHT, "global_tolerance": GLOBAL_TOLERANCE,
        "global_tolerance_penalty": GLOBAL_TOLERANCE_PENALTY,
        "missing_source_penalty": MISSING_SOURCE_PENALTY,
        "min_groups_for_full_coverage": MIN_GROUPS_FOR_FULL_COVERAGE,
        "baseline_objective": float(baseline_obj),
        "final_objective_of_stratified_search": float(stratified_obj),
    },
    "input_files": {"train": TRAIN_CSV_PATH, "validation": VAL_CSV_PATH,
                    "test": TEST_CSV_PATH},
    "split_sizes": split_sizes,
    "versions": {"python": platform.python_version(),
                 "pandas": pd.__version__, "numpy": np.__version__},
    "output_sha256": {},
}
for fname in list(out_files.values()) + ["resplit_full_table.csv",
                                         "assignment_by_group.csv",
                                         "original_split_leaking_groups.csv"]:
    manifest["output_sha256"][fname] = sha256_of_file(os.path.join(OUTPUT_DIR, fname))

with open(os.path.join(OUTPUT_DIR, "split_manifest.json"), "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)

print("\n" + open(verify_path, encoding="utf-8").read())
print(f"Verification report : {verify_path}")
print(f"Manifest (hashes)   : {os.path.join(OUTPUT_DIR, 'split_manifest.json')}")
print("\nDONE.")
