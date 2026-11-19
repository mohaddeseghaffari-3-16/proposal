
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import re
import json
import math
import time
import random
import warnings
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from einops import rearrange, repeat
from transformers import AutoModel, AutoTokenizer
from scipy.optimize import linear_sum_assignment
from scipy.stats import fisher_exact
from skimage.measure import label
from tqdm.auto import tqdm
from monai.networks.blocks.unetr_block import UnetrUpBlock
from monai.networks.blocks.upsample import SubpixelUpsample
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.losses import DiceCELoss
from functools import partial
from PIL import Image
tqdm = partial(tqdm, disable=True)

from monai.data.utils import worker_init_fn as monai_worker_init_fn
from monai.transforms import (Compose, NormalizeIntensityd, RandZoomd, Rotate90d,
                              Resized, ToTensord, LoadImaged, EnsureChannelFirstd)

import logging
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)


# Experiment hyper-parameters and default paths live in configs/mosmed_rq_v11.yaml
# (loaded below via --config).


class CfgNode(dict):
    def __init__(self, init_dict=None, key_list=None, new_allowed=False):
        init_dict = {} if init_dict is None else init_dict
        key_list = [] if key_list is None else key_list
        for k, v in init_dict.items():
            if type(v) is dict:
                init_dict[k] = CfgNode(v, key_list=key_list + [k])
        super(CfgNode, self).__init__(init_dict)

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value


def load_cfg_from_yaml_text(text):
    cfg = {}
    cfg_from_file = yaml.safe_load(text)
    for key in cfg_from_file:
        for k, v in cfg_from_file[key].items():
            cfg[k] = v
    return CfgNode(cfg)


def _parse_cli():
    ap = argparse.ArgumentParser(description="MosMed_RQ_TextAssocciation.py -- region-query pilot.")
    ap.add_argument("--config", default="configs/mosmed_rq_ta.yaml",
                    help="Experiment YAML (hyper-parameters + default paths).")
    ap.add_argument("--frames_dir", default=None, help="Override DATA.frames_dir from the config file.")
    ap.add_argument("--masks_dir", default=None, help="Override DATA.masks_dir from the config file.")
    ap.add_argument("--train_csv", default=None, help="Override DATA.train_csv from the config file.")
    ap.add_argument("--valid_csv", default=None, help="Override DATA.valid_csv from the config file.")
    ap.add_argument("--output_dir", default=None, help="Override PILOT.output_dir from the config file.")
    ap.add_argument("--bert_type", default=None, help="Override MODEL.bert_type from the config file.")
    ap.add_argument("--vision_type", default=None, help="Override MODEL.vision_type from the config file.")
    cli, _unknown = ap.parse_known_args()
    return cli


_CLI = _parse_cli()
if not os.path.isfile(_CLI.config):
    raise FileNotFoundError(
        f"Config file not found: {_CLI.config}. Run from the repository root or pass --config.")
with open(_CLI.config, "r", encoding="utf-8") as _f:
    CONFIG_YAML = _f.read()

args = load_cfg_from_yaml_text(CONFIG_YAML)
for _key in ("frames_dir", "masks_dir", "train_csv", "valid_csv", "output_dir", "bert_type", "vision_type",):
    _val = getattr(_CLI, _key)
    if _val is not None:
        args[_key] = _val

if not os.path.isdir(args.vision_type):
    print(f"[info] '{args.vision_type}' not found -> using 'facebook/convnext-tiny-224'")
    args.vision_type = "facebook/convnext-tiny-224"

BASE_OUTPUT_DIR = Path(f"{args.output_dir}_seed{args.seed}")
OUTPUT_DIR = BASE_OUTPUT_DIR
_run_index = 2
while OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
    OUTPUT_DIR = BASE_OUTPUT_DIR.with_name(f"{BASE_OUTPUT_DIR.name}_run{_run_index}")
    _run_index += 1
VIS_DIR = OUTPUT_DIR / "visualizations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VIS_DIR.mkdir(parents=True, exist_ok=True)

RA_HIT_KEYS = ("hits_cross", "n_cross", "hits_batch", "n_batch", "hits_hard", "n_hard", "hits_tok", "n_tok",
               "hits_null", "n_null", "hits_unm", "n_unm")
RA_COUNT_KEYS = ("positive_pairs", "neg_cross", "neg_batch", "neg_hard", "tok_anchors", "null_anchors",
                 "anchors_skipped")


def set_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def seed_worker(worker_id):
    monai_worker_init_fn(worker_id)
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def make_loader_generator(seed):
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _recursive_image_files(root):
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root}")
    return sorted(p for p in root.rglob("*")
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def _stem_key(value):
    if value is None or pd.isna(value):
        return ""
    name = os.path.basename(str(value).strip().replace("\\", "/"))
    if not name:
        return ""
    stem = Path(name).stem.lower().strip()
    for prefix in ("mask_", "mask-", "gt_", "gt-", "ground_truth_", "ground-truth_"):
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
    return stem


def _find_split_column(df):
    candidates = [
        "image", "image_name", "imagename", "filename", "file_name",
        "frame", "frame_name", "frame_filename", "image_path", "frame_path",
        "mask", "mask_name", "mask_filename", "mask_path", "id", "study_image"
    ]
    normalized = {str(c).strip().lower(): c for c in df.columns}
    for c in candidates:
        if c in normalized:
            return normalized[c]
    for col in df.columns:
        vals = df[col].dropna().astype(str).head(20)
        if len(vals) and sum(Path(v.replace("\\", "/")).suffix.lower() in IMAGE_EXTENSIONS
                             for v in vals) >= max(1, len(vals) // 3):
            return col
    return None


def _read_split_keys(csv_path, split_name):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{split_name} CSV not found: {csv_path}\n"
            "Check the path in the config file (configs/*.yaml) or pass it on the command line (see --help).")
    df = pd.read_csv(csv_path)
    if df.empty:
        raise RuntimeError(f"{split_name} CSV is empty: {csv_path}")
    col = _find_split_column(df)
    if col is None:
        raise ValueError(
            f"Could not identify an image/filename column in {csv_path}. "
            f"Available columns: {list(df.columns)}")
    keys = {_stem_key(v) for v in df[col].dropna().tolist()}
    keys.discard("")
    print(f"{split_name} split: {len(df):,} CSV rows; {len(keys):,} unique filename keys "
          f"(column='{col}')")
    return keys


def build_mosmed_split():
    frame_files = _recursive_image_files(args.frames_dir)
    mask_files = _recursive_image_files(args.masks_dir)
    if not frame_files:
        raise RuntimeError(f"No frame images found recursively under {args.frames_dir}")
    if not mask_files:
        raise RuntimeError(f"No masks found recursively under {args.masks_dir}")

    def make_lookup(files, label_name):
        lookup = {}
        duplicates = {}
        for path in files:
            key = _stem_key(path.name)
            if key in lookup:
                duplicates.setdefault(key, [lookup[key]]).append(path)
            else:
                lookup[key] = path
        if duplicates:
            examples = list(duplicates.items())[:5]
            raise RuntimeError(
                f"Duplicate normalized stems in {label_name}; cannot pair safely. Examples: {examples}")
        return lookup

    frame_lookup = make_lookup(frame_files, "frames")
    mask_lookup = make_lookup(mask_files, "masks")

    paired_by_key = {}
    missing_frames = []
    for key, mask_path in mask_lookup.items():
        frame_path = frame_lookup.get(key)
        if frame_path is None:
            candidates = [p for k, p in frame_lookup.items()
                          if k.endswith(key) or key.endswith(k)]
            if len(candidates) == 1:
                frame_path = candidates[0]
        if frame_path is None:
            missing_frames.append(mask_path.name)
            continue
        paired_by_key[key] = (str(frame_path), str(mask_path), mask_path.name)

    train_keys = _read_split_keys(args.train_csv, "Train")
    valid_keys = _read_split_keys(args.valid_csv, "Validation")

    def resolve_split(keys, split_name):
        records, unresolved = [], []
        for key in sorted(keys):
            record = paired_by_key.get(key)
            if record is None:
                candidates = [v for k, v in paired_by_key.items()
                              if k.endswith(key) or key.endswith(k)]
                if len(candidates) == 1:
                    record = candidates[0]
            if record is None:
                unresolved.append(key)
            else:
                records.append(record)
        if unresolved:
            print(f"[warning] {split_name}: {len(unresolved)} CSV keys did not match "
                  f"frame/mask files; examples: {unresolved[:8]}")
        unique = {r[2]: r for r in records}
        records = [unique[n] for n in sorted(unique)]
        if not records:
            raise RuntimeError(
                f"No {split_name} samples could be matched. "
                "Inspect CSV filename column and actual frame/mask names.")
        print(f"{split_name} paired samples: {len(records):,}")
        return records

    train_records = resolve_split(train_keys, "Train")
    valid_records = resolve_split(valid_keys, "Validation")

    train_ids = {r[2] for r in train_records}
    valid_ids = {r[2] for r in valid_records}
    overlap = train_ids & valid_ids
    if overlap:
        raise RuntimeError(
            f"Train/validation image overlap detected ({len(overlap)} samples), "
            f"e.g. {sorted(overlap)[:10]}. Check the group-level resplit CSVs.")

    print(f"Indexed frames: {len(frame_files):,} | masks: {len(mask_files):,} | "
          f"paired: {len(paired_by_key):,} | masks without frame: {len(missing_frames):,}")
    print(f"Final split -> train: {len(train_records):,} | validation: {len(valid_records):,} "
          "| test excluded")
    return train_records, valid_records


def _base_loader(keys):
    tfs = [LoadImaged(keys, reader="PILReader", reverse_indexing=bool(args.pil_reverse_indexing)),
           EnsureChannelFirstd(keys)]
    k = int(args.rot90_k) % 4
    if k:
        tfs.append(Rotate90d(keys, k=k, spatial_axes=(0, 1)))
    return tfs


_MASK_COUNT_TF = {}


def _mask_count_transform(image_size):
    key = (bool(args.pil_reverse_indexing), int(args.rot90_k) % 4, tuple(image_size))
    if key not in _MASK_COUNT_TF:
        _MASK_COUNT_TF[key] = Compose(_base_loader(["gt"]) + [
            Resized(["gt"], spatial_size=list(image_size), mode="nearest"),
            ToTensord(["gt"]),
        ])
    return _MASK_COUNT_TF[key]


def count_gt_components_from_file(mask_path, image_size, min_area, connectivity):
    gt = _mask_count_transform(image_size)({"gt": str(mask_path)})["gt"]
    gt = torch.as_tensor(np.asarray(gt))
    if gt.ndim == 2:
        gt = gt.unsqueeze(0)
    elif gt.ndim == 3 and gt.shape[0] > 1:
        gt = gt.max(dim=0, keepdim=True).values
    binary = (gt[0].cpu().numpy() > 0).astype(np.uint8)
    lab = label(binary, connectivity=connectivity)
    counts = np.bincount(lab.ravel())
    return int(sum(counts[i] >= min_area for i in range(1, int(lab.max()) + 1)))


def select_balanced_train_records(records, cfg, per_count=50):
    buckets = {n: [] for n in (3, 4, 5, 6)}
    for rec in records:
        n_comp = count_gt_components_from_file(rec[1], cfg.image_size,
                                               cfg.min_component_area, cfg.connectivity)
        if n_comp in buckets:
            buckets[n_comp].append((rec, n_comp))
    rng = random.Random(cfg.data_seed)
    chosen = []
    counts = {}
    for n in (3, 4, 5, 6):
        rng.shuffle(buckets[n])
        if len(buckets[n]) < per_count:
            raise RuntimeError(
                f"Train split has only {len(buckets[n])} eligible images with N={n} "
                f"(component area >= {cfg.min_component_area}); need {per_count}. "
                "No paths or thresholds were changed."
            )
        selected = buckets[n][:per_count]
        chosen.extend(selected)
        counts.update({rec[2]: n for rec, n in selected})
        print(f"Train N={n}: available={len(buckets[n])}, selected={len(selected)}")
    rng.shuffle(chosen)
    print(f"Balanced train selection: {len(chosen)} images; distribution="
          f"{ {n: sum(v == n for v in counts.values()) for n in (3,4,5,6)} }")
    return [rec for rec, _ in chosen], counts


def select_multi_region_records(records, limit, cfg, split_name="validation"):
    candidates = []
    for rec in records:
        n_comp = count_gt_components_from_file(rec[1], cfg.image_size,
                                               cfg.min_component_area, cfg.connectivity)
        if n_comp >= 2:
            candidates.append((rec, n_comp))
    rng = random.Random(cfg.data_seed + 1)
    rng.shuffle(candidates)
    chosen = candidates[:min(int(limit), len(candidates))]
    print(f"Multi-region {split_name}: found {len(candidates)} images with >=2 GT components; "
          f"selected {len(chosen)} / requested {limit}.")
    if len(chosen) < int(limit):
        print(f"[warning] Requested {limit} validation images but only {len(chosen)} are available.")
    return [rec for rec, _ in chosen], {rec[2]: n for rec, n in chosen}


TEXT_LOOKUP = {}
TEXT_CSVS = [args.train_csv, args.valid_csv]


def _norm_image_key(x):
    if pd.isna(x): return ""
    name = os.path.basename(str(x).replace("\\", "/").strip())
    return os.path.splitext(name)[0].lower()


def load_sample_text_lookup():
    lookup = {}
    for csv_path in TEXT_CSVS:
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(
                f"Text annotation CSV not found: {csv_path}. Check train_csv / valid_csv in the config file or pass --train_csv / --valid_csv.")
        df = pd.read_csv(csv_path)
        cols = {str(c).strip().lower(): c for c in df.columns}
        image_col = next((cols[k] for k in ["image", "filename", "image_name", "imagename", "imagetext"] if k in cols), df.columns[0])
        text_col = next((cols[k] for k in ["text", "report", "caption", "description", "label"] if k in cols and cols[k] != image_col), None)
        if text_col is None:
            raise ValueError(f"Could not detect a text column in {csv_path}. Columns: {list(df.columns)}")
        for _, row in df.iterrows():
            key = _norm_image_key(row[image_col])
            val = "" if pd.isna(row[text_col]) else str(row[text_col]).strip()
            if key and val:
                if key in lookup and lookup[key] != val:
                    continue
                lookup[key] = val
    print(f"Loaded sample-level text for {len(lookup):,} image keys.")
    return lookup


TEXT_CLEAN_PUNCTUATION = True


def preprocess_text_for_encoder(text):
    text = str(text or "").strip()
    if TEXT_CLEAN_PUNCTUATION:
        text = re.sub(r"[.,!?;:\"()\[\]{}]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalized_caption_key(text):
    return preprocess_text_for_encoder(text).casefold()


def build_mismatched_caption_map(names, texts, split_label="evaluation"):
    if len(names) != len(texts):
        raise ValueError(f"{split_label}: names/texts length mismatch ({len(names)} vs {len(texts)}).")
    keys = [normalized_caption_key(t) for t in texts]
    result = {}
    source = {}
    n = len(names)
    for i, name in enumerate(names):
        chosen = None
        chosen_source = None
        for step in range(1, n):
            j = (i + step) % n
            if j != i and keys[j] and keys[j] != keys[i]:
                chosen = str(texts[j])
                chosen_source = str(names[j])
                break
        if chosen is None:
            raise RuntimeError(
                f"Cannot construct a true Mismatched caption for {name!r} in {split_label}: "
                "the split contains no other non-empty, textually different annotation. "
                "This is a data limitation, not a model error. Check the caption CSV/split."
            )
        if normalized_caption_key(chosen) == keys[i]:
            raise AssertionError(f"Internal mismatch-map error for {name!r}.")
        result[str(name)] = chosen
        source[str(name)] = chosen_source
    return result, source


PERTURB_CONDITIONS = ("count_changed", "count_removed", "location_removed",
                      "one_side_removed", "left_side_removed")


def validate_text_perturbations(records, split_label="dataset"):
    names = [str(r.get("name", r.get("image_name", r.get("image", "")))) for r in records]
    texts = [str(r.get("text", "")) for r in records]
    if not records:
        raise RuntimeError(f"{split_label}: no records available for perturbation preflight.")
    if any(not t.strip() for t in texts):
        bad = [names[i] for i, t in enumerate(texts) if not t.strip()]
        raise RuntimeError(f"{split_label}: empty captions found, e.g. {bad[:5]}")
    mismatch, _ = build_mismatched_caption_map(names, texts, split_label)
    noops = []
    for name, text in zip(names, texts):
        for condition in PERTURB_CONDITIONS:
            changed = perturb_text(text, condition)
            if normalized_caption_key(changed) == normalized_caption_key(text):
                noops.append((name, condition))
    if noops:
        raise RuntimeError(f"{split_label}: perturbation generator produced no-op transformations: {noops[:10]}")
    return {"n_records": len(records), "n_unique_normalized_captions": len(set(normalized_caption_key(t) for t in texts)),
            "mismatched_caption_pairs": len(mismatch), "no_op_perturbations": 0}


_NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
              "eight": 8, "nine": 9, "ten": 10,
              "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10}
_COUNT_RE = re.compile(
    r"\b(?P<count>one|two|three|four|five|six|seven|eight|nine|ten|\d+)\b"
    r"(?P<tail>\s+(?:infected\s+)?(?:areas?|regions?|lesions?|opacities?|infections?))",
    flags=re.I)
_PHRASE_RE = re.compile(r"\b(?:(upper|middle|lower)\s+)?(left|right)\s+lung\b", flags=re.I)
_VERTICALS = ("upper", "middle", "lower")


def caption_count(text):
    m = _COUNT_RE.search(str(text or ""))
    return _NUM_WORDS.get(m.group("count").lower(), 0) if m else 0


def caption_sides(text):
    return {m.group(2).lower() for m in _PHRASE_RE.finditer(preprocess_text_for_encoder(text))}


def _clean_after_drop(t):
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r",\s*and\b", ",", t, flags=re.I)
    t = re.sub(r"\b(?:and|all|in)\s*(?=[.,;]|$)", "", t, flags=re.I)
    t = re.sub(r"\s+([.,;])", r"\1", t)
    t = re.sub(r",\s*\.", ".", t)
    t = re.sub(r",\s*,+", ",", t)
    return re.sub(r"\s+", " ", t).strip(" ,;")


def _drop_matches(text, matches):
    t = text
    for p in sorted(matches, key=lambda m: m.start(), reverse=True):
        t = t[:p.start()] + t[p.end():]
    return _clean_after_drop(t)


def _flip_side_spans(text, matches):
    t = text
    for p in sorted(matches, key=lambda m: m.start(2), reverse=True):
        w = p.group(2)
        n = "right" if w.lower() == "left" else "left"
        n = n.capitalize() if w[0].isupper() else n
        t = t[:p.start(2)] + n + t[p.end(2):]
    return t


def make_location_hard_negatives(text, rng, max_out=4):
    original = str(text or "").strip()
    key0 = normalized_caption_key(original)
    phrases = list(_PHRASE_RE.finditer(original))
    cands = []
    for side_word, lacks in (("left", 1), ("right", 2)):
        ms = [p for p in phrases if p.group(2).lower() == side_word]
        if not ms:
            continue
        cands.append((_flip_side_spans(original, ms), lacks))
        cands.append((_drop_matches(original, ms), lacks))
    final, seen = [], {key0}
    for t, lacks in cands:
        k = normalized_caption_key(t)
        if not k or k in seen:
            continue
        if ("left", "right")[lacks - 1] in caption_sides(t):
            continue
        seen.add(k)
        final.append((t, lacks))
    rng.shuffle(final)
    return final[:int(max_out)]


def perturb_text(text, condition):
    original = str(text or "").strip()
    if condition == "correct":
        return original

    number_words = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "1": 1, "2": 2, "3": 3, "4": 4, "5": 5,
        "6": 6, "7": 7, "8": 8, "9": 9, "10": 10,
    }
    number_names = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
                    6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}

    count_pattern = re.compile(
        r"\b(?P<count>one|two|three|four|five|six|seven|eight|nine|ten|\d+)\b"
        r"(?P<tail>\s+(?:infected\s+)?(?:areas?|regions?|lesions?|opacities?|infections?))",
        flags=re.I,
    )

    if condition in ("count_removed", "count_changed"):
        match = count_pattern.search(original)
        if condition == "count_removed":
            if match:
                changed = original[:match.start()] + original[match.end():]
                changed = re.sub(r",\s*,+", ",", changed)
                changed = re.sub(r"\s+", " ", changed)
                changed = re.sub(r"\s+([,.;:])", r"\1", changed).strip(" ,.;:")
                if changed == original:
                    changed = re.sub(r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)\b", "", original, count=1, flags=re.I)
            else:
                changed = re.sub(r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)\b", "", original, count=1, flags=re.I)
            changed = re.sub(r",\s*,+", ",", changed)
            changed = re.sub(r"\s+", " ", changed)
            changed = re.sub(r"\s+([,.;:])", r"\1", changed).strip(" ,.;:")
            return changed or "pulmonary infection; count information removed"

        if match:
            old_token = match.group("count")
            old_num = number_words.get(old_token.lower())
            if old_num is None:
                old_num = number_words.get(old_token)
            if old_num is None:
                new_num = 2
            else:
                new_num = 1 if old_num >= 10 else old_num + 1
            new_token = number_names.get(new_num, str(new_num))
            changed = original[:match.start("count")] + new_token + original[match.end("count"):]
            if changed == original:
                changed = original + " The stated count is two regions."
            return changed
        return original.rstrip(" .;,") + ". The stated count is two infected areas."

    if condition == "location_removed":
        changed = original
        location_patterns = [
            r"\b(?:in\s+)?(?:the\s+)?(?:middle|upper|lower|central|peripheral)\s+(?:left|right)\s+lung\b",
            r"\b(?:in\s+)?(?:the\s+)?(?:left|right)\s+(?:upper|lower|middle)\s+lung\b",
            r"\b(?:in\s+)?(?:the\s+)?(?:left|right)\s+lung\b",
            r"\b(?:in\s+)?(?:the\s+)?(?:upper|lower|middle|central|peripheral)\s+lung\b",
            r"\b(?:bilateral|unilateral|left-sided|right-sided)\b",
            r"\b(?:left|right|upper|lower|middle|central|peripheral)\b",
        ]
        for pat in location_patterns:
            changed = re.sub(pat, " ", changed, flags=re.I)
        changed = re.sub(r"\b(?:and|or)\b(?=\s*[,.;])", " ", changed, flags=re.I)
        changed = re.sub(r",\s*,+", ", ", changed)
        changed = re.sub(r"\s+", " ", changed)
        changed = re.sub(r"\s+([,.;:])", r"\1", changed)
        changed = changed.strip(" ,.;:")
        if not changed or changed.lower() == original.lower():
            changed = re.sub(r"\b(?:left|right|bilateral|upper|lower|middle|central|peripheral)\b", "", original, flags=re.I)
            changed = re.sub(r"\s+", " ", changed).strip(" ,.;:")
        return changed or "pulmonary infection; location information removed"

    if condition in ("one_side_removed", "left_side_removed"):
        want = "right" if condition == "one_side_removed" else "left"
        ph = list(_PHRASE_RE.finditer(original))
        if not ph:
            return original.rstrip(" .;,") + " only"
        target = next((p for p in ph if p.group(2).lower() == want), ph[0])
        changed = _drop_matches(original, [target])
        return changed if changed.strip() else "pulmonary infection"

    raise ValueError(f"Unknown text perturbation condition: {condition}")


def build_lexicon_tables(tokenizer, vocab_size):
    type_t = torch.zeros(vocab_size, dtype=torch.long)
    num_t = torch.zeros(vocab_size, dtype=torch.long)
    side_t = torch.zeros(vocab_size, dtype=torch.long)
    vert_t = torch.zeros(vocab_size, dtype=torch.long)
    unk = tokenizer.unk_token_id
    found, missing = {}, []

    def put(word, t, num=0, side=0, vert=0):
        tid = tokenizer.convert_tokens_to_ids(word)
        if tid is None or tid == unk or int(tid) >= vocab_size:
            missing.append(word)
            return
        type_t[int(tid)] = t
        if num: num_t[int(tid)] = num
        if side: side_t[int(tid)] = side
        if vert: vert_t[int(tid)] = vert
        found[word] = int(tid)

    for w, n in _NUM_WORDS.items():
        put(w, 1, num=n)
    for i, w in enumerate(_VERTICALS, 1):
        put(w, 2, vert=i)
    put("left", 3, side=1)
    put("right", 3, side=2)
    put("lung", 4)
    print(f"Lexicon: {len(found)} single-token words found; missing/multi-piece: {missing}")
    for must in ("left", "right", "lung"):
        if must not in found:
            raise RuntimeError(f"Tokenizer has no single token for '{must}'; phrase parsing cannot work.")
    return type_t, num_t, side_t, vert_t


class MosMedDataset(Dataset):
    def __init__(self, records, mode="train", image_size=(224, 224)):
        super().__init__()
        self.records = list(records)
        self.mode = mode
        self.image_size = list(image_size)
        self.transforms = self._build_transform()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        image_path, mask_path, name = self.records[idx]
        data = {"image": image_path, "gt": mask_path}
        data = self.transforms(data)
        image, gt = data["image"], data["gt"]

        if gt.ndim == 2:
            gt = gt.unsqueeze(0)
        elif gt.ndim == 3 and gt.shape[0] > 1:
            gt = gt.max(dim=0, keepdim=True).values
        if gt.ndim != 3 or gt.shape[0] != 1:
            raise ValueError(f"Expected a single-channel mask after conversion; got {tuple(gt.shape)} for {mask_path}")
        gt = (gt > 0).to(torch.uint8)
        gt_np = gt[0].cpu().numpy().astype(np.uint8)

        lab = label(gt_np, connectivity=args.connectivity)
        counts = np.bincount(lab.ravel())
        comp_list = [
            (lab == i) for i in range(1, int(lab.max()) + 1)
            if counts[i] >= args.min_component_area
        ]
        if comp_list:
            comps = torch.from_numpy(np.stack(comp_list).astype(np.uint8))
        else:
            comps = torch.zeros((0, gt_np.shape[0], gt_np.shape[1]), dtype=torch.uint8)

        text = TEXT_LOOKUP.get(_norm_image_key(name), "")
        if not text:
            raise ValueError(f"Missing sample-level text annotation for {name}. Check image keys in text CSVs.")
        return image, gt.int(), comps, name, text

    def _build_transform(self):
        common = _base_loader(["image", "gt"])
        if self.mode == "train":
            common += [
                RandZoomd(["image", "gt"], min_zoom=0.95, max_zoom=1.2,
                          mode=["bicubic", "nearest"], prob=0.1),
            ]
        common += [
            Resized(["image"], spatial_size=self.image_size, mode="bicubic"),
            Resized(["gt"], spatial_size=self.image_size, mode="nearest"),
            NormalizeIntensityd(["image"], channel_wise=True),
            ToTensord(["image", "gt"]),
        ]
        return Compose(common)


def collate_fn(batch):
    images = torch.stack([b[0] for b in batch])
    gts = torch.stack([b[1] for b in batch])
    comps = [b[2] for b in batch]
    names = [b[3] for b in batch]
    texts = [b[4] for b in batch]
    return images, gts, comps, names, texts


_SIDE_WORD_RE = re.compile(r"\b(left|right)\b", flags=re.I)
_BILATERAL_RE = re.compile(r"\b(bilateral|both)\b", flags=re.I)


def caption_side_category(text):
    t = preprocess_text_for_encoder(text)
    words = {m.group(1).lower() for m in _SIDE_WORD_RE.finditer(t)}
    if _BILATERAL_RE.search(t) or words == {"left", "right"}:
        return "both"
    if len(words) == 1 and caption_sides(text) == words:
        return next(iter(words))
    return None


def categorize_records(records):
    cats = {"left": [], "right": [], "both": [], "other": []}
    for rec in records:
        c = caption_side_category(TEXT_LOOKUP.get(_norm_image_key(rec[2]), ""))
        cats[c if c else "other"].append(rec)
    return cats


def single_side_pool(records, max_n, seed):
    cats = categorize_records(records)
    rng = random.Random(int(seed))
    out = []
    for sd in ("left", "right"):
        lst = list(cats[sd])
        rng.shuffle(lst)
        out.extend((r, sd) for r in lst[: max(1, int(max_n) // 2)])
    return out, {k: len(v) for k, v in cats.items()}


def both_side_pool(records, max_n, seed):
    lst = list(categorize_records(records)["both"])
    random.Random(int(seed) + 1).shuffle(lst)
    return lst[: int(max_n)]


def _mask_geometry(mask_path, cfg):
    gt = _mask_count_transform(cfg.image_size)({"gt": str(mask_path)})["gt"]
    gt = torch.as_tensor(np.asarray(gt))
    if gt.ndim == 3:
        gt = gt.max(dim=0).values
    binary = (gt.cpu().numpy() > 0).astype(np.uint8)
    lab = label(binary, connectivity=cfg.connectivity)
    cnt = np.bincount(lab.ravel())
    keep = np.zeros(binary.shape, dtype=bool)
    comps = []
    for i in range(1, int(lab.max()) + 1):
        if cnt[i] >= cfg.min_component_area:
            m = lab == i
            keep |= m
            ys, xs = np.nonzero(m)
            comps.append((float(xs.mean()), float(ys.mean())))
    if not comps:
        return None
    ys, xs = np.nonzero(keep)
    return {"comps": comps, "union": (float(xs.mean()), float(ys.mean())),
            "W": binary.shape[1], "H": binary.shape[0]}


def _fisher_p(acc_left, acc_right):
    if len(acc_left) == 0 or len(acc_right) == 0:
        return 1.0
    a = int(sum(acc_left)); b = len(acc_left) - a
    c = int(sum(acc_right)); d = len(acc_right) - c
    return float(fisher_exact([[a, b], [c, d]])[1])


def orientation_stats(pool_single, pool_both, cfg):
    margin = float(cfg.att_center_margin)
    acc = {"left": {"x": [], "y": []}, "right": {"x": [], "y": []}}
    strad = {"single": {"x": [], "y": []}, "both": {"x": [], "y": []}}

    def straddles(vals, mid):
        return float(any(v > mid + margin for v in vals) and any(v < mid - margin for v in vals))

    for rec, sd in pool_single:
        g = _mask_geometry(rec[1], cfg)
        if g is None:
            continue
        cx, cy = g["union"]
        W, H = g["W"], g["H"]
        if abs(cx - W / 2.0) >= margin:
            acc[sd]["x"].append(float(cx > W / 2.0))
        if abs(cy - H / 2.0) >= margin:
            acc[sd]["y"].append(float(cy > H / 2.0))
        strad["single"]["x"].append(straddles([c[0] for c in g["comps"]], W / 2.0))
        strad["single"]["y"].append(straddles([c[1] for c in g["comps"]], H / 2.0))
    for rec in pool_both:
        g = _mask_geometry(rec[1], cfg)
        if g is None:
            continue
        strad["both"]["x"].append(straddles([c[0] for c in g["comps"]], g["W"] / 2.0))
        strad["both"]["y"].append(straddles([c[1] for c in g["comps"]], g["H"] / 2.0))

    def mean(v):
        return float(np.mean(v)) if len(v) else float("nan")

    out = {"n_both": len(strad["both"]["x"]), "n_single": len(strad["single"]["x"])}
    for ax_ in ("x", "y"):
        out[f"n_left_{ax_}"], out[f"n_right_{ax_}"] = len(acc["left"][ax_]), len(acc["right"][ax_])
        out[f"p_left_{ax_}"], out[f"p_right_{ax_}"] = mean(acc["left"][ax_]), mean(acc["right"][ax_])
        sep = abs(out[f"p_left_{ax_}"] - out[f"p_right_{ax_}"])
        out[f"sep_{ax_}"] = 0.0 if not np.isfinite(sep) else float(sep)
        out[f"fisher_{ax_}"] = _fisher_p(acc["left"][ax_], acc["right"][ax_])
        out[f"strad_both_{ax_}"] = mean(strad["both"][ax_])
        out[f"strad_single_{ax_}"] = mean(strad["single"][ax_])
        sb = out[f"strad_both_{ax_}"]
        ss = out[f"strad_single_{ax_}"] if (len(strad["single"][ax_]) >= 5 and np.isfinite(out[f"strad_single_{ax_}"])) else 0.0
        out[f"score_{ax_}"] = float(sb - ss) if np.isfinite(sb) else float("nan")
    return out


def _decide_axis(st, min_sep):
    def ok(a):
        return (st[f"fisher_{a}"] < 0.01 and st[f"sep_{a}"] >= min_sep
                and st[f"n_left_{a}"] >= 3 and st[f"n_right_{a}"] >= 3)
    ok_x, ok_y = ok("x"), ok("y")
    if ok_x and not ok_y:
        return "x", "single-side separation"
    if ok_y and not ok_x:
        return "y", "single-side separation"
    if ok_x and ok_y:
        return ("x" if st["sep_x"] >= st["sep_y"] else "y"), "single-side separation (both axes significant; larger wins)"
    sx, sy = st["score_x"], st["score_y"]
    if st["n_both"] >= 30 and np.isfinite(sx) and np.isfinite(sy) and abs(sx - sy) >= 0.15:
        return ("x" if sx > sy else "y"), "straddle score of bilateral captions"
    return None, "undecided"


def resolve_orientation(records_pool, out_dir):
    print("=" * 90)
    print("ORIENTATION AUDIT (is the patient's left/right axis the horizontal axis of the model input?)")
    print("=" * 90)
    pool_s, avail = single_side_pool(records_pool, int(args.orient_audit_max), int(args.data_seed))
    pool_b = both_side_pool(records_pool, int(args.orient_both_audit_max), int(args.data_seed))
    print(f"caption categories over the whole train+val pool: {avail} | audited: single={len(pool_s)} both={len(pool_b)}")
    min_sep = float(args.orient_min_sep)
    history, st, axis, why = [], None, None, ""
    for attempt in range(2):
        st = orientation_stats(pool_s, pool_b, args)
        st["rot90_k"] = int(args.rot90_k) % 4
        st["pil_reverse_indexing"] = bool(args.pil_reverse_indexing)
        axis, why = _decide_axis(st, min_sep)
        st["axis_decision"], st["axis_reason"] = axis, why
        history.append(dict(st))
        print(f"[attempt {attempt}] rot90_k={st['rot90_k']} reverse_indexing={st['pil_reverse_indexing']}")
        for ax_ in ("x", "y"):
            print(f"   axis {ax_}: n(L/R)={st[f'n_left_{ax_}']}/{st[f'n_right_{ax_}']} "
                  f"P(high side|left-only)={st[f'p_left_{ax_}']:.2f} P(high side|right-only)={st[f'p_right_{ax_}']:.2f} "
                  f"sep={st[f'sep_{ax_}']:.2f} fisher_p={st[f'fisher_{ax_}']:.4f} | "
                  f"straddle both-captions={st[f'strad_both_{ax_}']:.2f} single-captions={st[f'strad_single_{ax_}']:.2f} "
                  f"score={st[f'score_{ax_}']:.2f}")
        print(f"   -> L/R axis decision: {axis} ({why}) | n_both={st['n_both']} n_single={st['n_single']}")
        if axis == "y" and attempt == 0:
            args.rot90_k = (int(args.rot90_k) + 3) % 4
            print(f"   -> the patient's left/right axis is VERTICAL in the tensor (picture rotated by 90 deg). "
                  f"Setting rot90_k={int(args.rot90_k)} and re-checking.")
            continue
        break

    axis_ok = (axis == "x")
    conv = str(args.patient_left_on_image_side).lower()
    conv_source = "config"
    data_conv = None
    conv_conclusive = bool(axis_ok and st["fisher_x"] < 0.01 and st["sep_x"] >= min_sep
                           and st["n_left_x"] >= 3 and st["n_right_x"] >= 3)
    if conv_conclusive:
        data_conv = "right" if st["p_left_x"] > st["p_right_x"] else "left"
        print(f"   -> data says: patient-LEFT lung appears on the image-{data_conv.upper()} side "
              f"(sep {st['sep_x']:.2f}, Fisher p={st['fisher_x']:.4f}).")
        if conv == "auto":
            conv, conv_source = data_conv, "data"
        elif conv != data_conv:
            print(f"   [WARNING] config patient_left_on_image_side='{conv}' CONTRADICTS the data ('{data_conv}'). "
                  "Keeping the config value; set it to 'auto' to follow the data.")
    elif conv == "auto":
        conv, conv_source = "right", "ASSUMED (too few single-side captions to measure it)"
        print("   [WARNING] the side convention could NOT be measured; ASSUMING patient-left on image-RIGHT "
              "(radiological). If this is wrong, left/right phrases are swapped consistently. Verify with "
              "visualizations/orientation_check.png or set patient_left_on_image_side by hand.")
    try:
        render_orientation_check(records_pool, args, VIS_DIR / "orientation_check.png")
    except Exception as e:
        print("[warning] could not render orientation_check.png:", e)
    with open(Path(out_dir) / "orientation_audit.json", "w") as f:
        json.dump({"caption_categories": avail, "attempts": history, "final_rot90_k": int(args.rot90_k) % 4,
                   "pil_reverse_indexing": bool(args.pil_reverse_indexing), "axis_ok": bool(axis_ok),
                   "patient_left_on_image_side": conv, "convention_source": conv_source}, f, indent=2, default=float)
    if not axis_ok:
        msg = ("Orientation audit could not establish the patient's left/right axis (decision=%s). Open "
               "visualizations/orientation_check.png and the numbers printed above, then set auto_orientation=false and "
               "rot90_k (0..3), pil_reverse_indexing and patient_left_on_image_side ('left'/'right') by hand. "
               "If the lungs appear stacked TOP/BOTTOM (spine on the image right), use rot90_k=3." % axis)
        if bool(args.strict_orientation):
            raise RuntimeError(msg)
        print("[WARNING]", msg)
    args.patient_left_on_image_side = conv
    print(f"FINAL: rot90_k={int(args.rot90_k) % 4} | pil_reverse_indexing={bool(args.pil_reverse_indexing)} | "
          f"patient_left_on_image_side={conv} ({conv_source}) | axis_ok={axis_ok}")
    print("=" * 90)


def render_orientation_check(records, cfg, save_path, n_per_side=3):
    chosen = []
    pool, _ = single_side_pool(records, 2 * n_per_side * 40, int(cfg.data_seed))
    per = {"left": [r for r, sd in pool if sd == "left"][:n_per_side], "right": [r for r, sd in pool if sd == "right"][:n_per_side]}
    for sd in ("left", "right"):
        chosen += [(r, f"caption: {sd.upper()} lung only") for r in per[sd]]
    if len(chosen) < 2 * n_per_side:
        chosen += [(r, "caption: mixed") for r in list(records)[: 2 * n_per_side - len(chosen)]]
    left_on_right = str(cfg.patient_left_on_image_side).lower() == "right"
    ds = MosMedDataset([r for r, _ in chosen], mode="valid", image_size=cfg.image_size)
    cmap = plt.get_cmap("tab10")
    ncol = 3
    nrow = int(math.ceil(len(chosen) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 5.8 * nrow), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for a, (_, tag) in enumerate(chosen):
        image, gt, comps, name, text = ds[a]
        img = image[0].numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        ax = axes[a // ncol, a % ncol]
        ax.imshow(img, cmap="gray")
        H, W = img.shape
        ax.axvline(W / 2.0, color="yellow", linestyle="--", linewidth=1.2)
        ax.axhline(H / 2.0, color="cyan", linestyle=":", linewidth=0.8)
        names = []
        for j in range(comps.shape[0]):
            m = comps[j].numpy().astype(bool)
            if m.any():
                ax.contour(m.astype(float), levels=[0.5], colors=[cmap(j % 10)], linewidths=1.4)
            s_ = image_side_of_component(comps[j], cfg)
            if s_ is None:
                names.append(f"G{j+1}:mid")
            else:
                ps = (0 if s_ == 1 else 1) if left_on_right else s_
                names.append(f"G{j+1}:img-{'R' if s_ == 1 else 'L'}->pt-{('LEFT', 'RIGHT')[ps]}")
        ax.set_title(f"{tag}\n{' '.join(names)}", fontsize=8)
        ax.text(0.01, -0.02, text[:110], transform=ax.transAxes, fontsize=7, va="top", wrap=True)
    fig.suptitle(f"Orientation check | rot90_k={int(cfg.rot90_k) % 4} reverse_indexing={bool(cfg.pil_reverse_indexing)} "
                 f"patient_left_on_image_side={cfg.patient_left_on_image_side} | yellow dashed = L/R split used by the code", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


class VisionModel(nn.Module):
    def __init__(self, vision_type, project_dim):
        super(VisionModel, self).__init__()
        self.model = AutoModel.from_pretrained(vision_type, output_hidden_states=True)
        self.project_head = nn.Linear(768, project_dim)
        self.spatial_dim = 768

    def forward(self, x):
        output = self.model(x, output_hidden_states=True)
        return {"feature": output['hidden_states']}


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout=0, max_len: int = 5000) -> None:
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


def build_2d_sincos(d, h, w, temperature=100.0):
    assert d % 4 == 0, "d_m must be divisible by 4 for the 2-D positional code"
    d4 = d // 4
    omega = 1.0 / (temperature ** (torch.arange(d4, dtype=torch.float32) / d4))
    ys = torch.arange(h, dtype=torch.float32)[:, None] * omega[None, :]
    xs = torch.arange(w, dtype=torch.float32)[:, None] * omega[None, :]
    pe_y = torch.cat([ys.sin(), ys.cos()], dim=1)
    pe_x = torch.cat([xs.sin(), xs.cos()], dim=1)
    pe = torch.cat([pe_y[:, None, :].expand(h, w, d // 2),
                    pe_x[None, :, :].expand(h, w, d // 2)], dim=2)
    return pe.permute(2, 0, 1).contiguous()


class RegionQueryHead(nn.Module):

    def __init__(self, num_queries, in_dim=768, mask_in_dim=96,
                 d_q=768, d=768, d_v=768, d_m=256, mask_pos_enc=True):
        super().__init__()
        self.mask_pos_enc = bool(mask_pos_enc)
        self.K = num_queries
        self.d = d
        self.d_m = d_m
        self.Q = nn.Parameter(torch.randn(num_queries, d_q))
        self.W_Q = nn.Linear(d_q, d, bias=False)
        self.W_K = nn.Linear(in_dim, d, bias=False)
        self.W_V = nn.Linear(in_dim, d_v, bias=False)
        self.mask_proj = nn.Conv2d(mask_in_dim, d_m, kernel_size=1)
        self.W_M = nn.Linear(d_v, d_m, bias=False)
        if self.mask_pos_enc:
            self.register_buffer('pos2d', build_2d_sincos(d_m, 56, 56), persistent=False)
            self.pos_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, F_tokens, E56, side=7):
        B = F_tokens.shape[0]
        q = self.W_Q(self.Q)
        k = self.W_K(F_tokens)
        S = torch.einsum('kd,bnd->bkn', q, k) / math.sqrt(self.d)
        A = S.softmax(dim=-1)
        v = self.W_V(F_tokens)
        R = torch.einsum('bkn,bnc->bkc', A, v)

        E = self.mask_proj(E56)
        if self.mask_pos_enc:
            E = E + self.pos_scale * self.pos2d[None]
        m = self.W_M(R)
        Z = torch.einsum('bchw,bkc->bkhw', E, m) / math.sqrt(self.d_m)
        logits = F.interpolate(Z, scale_factor=4, mode='bilinear', align_corners=False)
        return R, A.view(B, self.K, side, side), logits, S


def region_geometry(mask_probs, obj_pivot=20.0):
    B, K, H, W = mask_probs.shape
    col = mask_probs.sum(2)
    row = mask_probs.sum(3)
    area = col.sum(-1)
    a = area.clamp_min(1e-6)
    xs = torch.linspace(0, 1, W, device=mask_probs.device, dtype=mask_probs.dtype)
    ys = torch.linspace(0, 1, H, device=mask_probs.device, dtype=mask_probs.dtype)
    cx = (col * xs).sum(-1) / a
    cy = (row * ys).sum(-1) / a
    sx = (((xs - cx[..., None]) ** 2 * col).sum(-1) / a).clamp_min(0).sqrt()
    sy = (((ys - cy[..., None]) ** 2 * row).sum(-1) / a).clamp_min(0).sqrt()
    act = area / (area + obj_pivot)
    frac = area / float(H * W)
    geom = torch.stack([(cx - 0.5) * 4, (cy - 0.5) * 4, sx * 4, sy * 4, frac.sqrt() * 4], dim=-1)
    return geom * act[..., None], act


def act_split(act_i, rows):
    m = set(int(r) for r in rows)
    am = [float(act_i[k]) for k in sorted(m)]
    au = [float(act_i[k]) for k in range(int(act_i.shape[0])) if k not in m]
    return am, au


class TextTokenAdapter(nn.Module):

    def __init__(self, text_dim, dim, use_typing=True):
        super().__init__()
        self.proj = nn.Linear(text_dim, dim) if text_dim != dim else nn.Identity()
        self.adapter = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 256), nn.GELU(), nn.Linear(256, dim))
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)
        self.use_typing = bool(use_typing)
        if self.use_typing:
            self.side_emb = nn.Embedding(3, dim, padding_idx=0)
            self.vert_emb = nn.Embedding(4, dim, padding_idx=0)
            nn.init.normal_(self.side_emb.weight, std=0.5)
            nn.init.normal_(self.vert_emb.weight, std=0.5)
            with torch.no_grad():
                self.side_emb.weight[0].zero_()
                self.vert_emb.weight[0].zero_()
        self.norm = nn.LayerNorm(dim)

    def forward(self, h, side_tok, vert_tok):
        x = self.proj(h)
        x = x + self.adapter(x)
        if self.use_typing:
            x = x + self.side_emb(side_tok) + self.vert_emb(vert_tok)
        return self.norm(x)


class RegionRoutedFiLM(nn.Module):

    def __init__(self, channels, region_dim=768, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(region_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, 2 * channels))
        nn.init.normal_(self.net[-1].weight, std=1e-2)
        nn.init.zeros_(self.net[-1].bias)
        self.gate = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, regions, mask_probs, side):
        m = F.adaptive_avg_pool2d(mask_probs, side).flatten(2)
        m = m / m.sum(dim=1, keepdim=True).clamp_min(1.0)
        gamma, beta = self.net(regions).chunk(2, dim=-1)
        gamma = torch.tanh(gamma)
        g = torch.einsum('bkn,bkc->bnc', m, gamma)
        b = torch.einsum('bkn,bkc->bnc', m, beta)
        return x * (1.0 + self.gate * g) + self.gate * b


class RegionGuidedDecoderLayer(nn.Module):
    def __init__(self, in_channels, region_dim=768):
        super().__init__()
        self.self_attn_norm = nn.LayerNorm(in_channels)
        self.cross_attn_norm = nn.LayerNorm(in_channels)
        self.self_attn = nn.MultiheadAttention(embed_dim=in_channels, num_heads=1, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim=in_channels, num_heads=4, batch_first=True)
        self.region_project = nn.Sequential(nn.Linear(region_dim, in_channels), nn.LeakyReLU())
        self.vis_pos = PositionalEncoding(in_channels)
        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)
        self.scale = nn.Parameter(torch.tensor(1.0), requires_grad=True)

    def forward(self, x, regions):
        reg = self.region_project(regions)

        vis2 = self.norm1(x)
        q = k = self.vis_pos(vis2)
        vis2 = self.self_attn(q, k, value=vis2)[0]
        vis2 = self.self_attn_norm(vis2)
        vis = x + vis2

        vis2 = self.norm2(vis)
        vis2, weights = self.cross_attn(query=self.vis_pos(vis2), key=reg, value=reg,
                                        need_weights=True, average_attn_weights=True)
        vis2 = self.cross_attn_norm(vis2)
        vis = vis + self.scale * vis2
        return vis, weights


class RegionGuidedDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, spatial_size, use_film=True, film_hidden=256):
        super().__init__()
        self.film = RegionRoutedFiLM(in_channels, 768, film_hidden) if use_film else None
        self.guide_layer = RegionGuidedDecoderLayer(in_channels)
        self.spatial_size = spatial_size
        self.decoder = UnetrUpBlock(2, in_channels, out_channels, 3, 2, norm_name='BATCH')

    def forward(self, vis, skip_vis, regions, mask_probs=None):
        if self.film is not None and mask_probs is not None:
            vis = self.film(vis, regions, mask_probs, self.spatial_size)
        vis, weights = self.guide_layer(vis, regions)
        vis = rearrange(vis, 'B (H W) C -> B C H W', H=self.spatial_size, W=self.spatial_size)
        skip_vis = rearrange(skip_vis, 'B (H W) C -> B C H W',
                             H=self.spatial_size * 2, W=self.spatial_size * 2)
        output = self.decoder(vis, skip_vis)
        output = rearrange(output, 'B C H W -> B (H W) C')
        return output, weights


class RegionQueryPilot(nn.Module):
    def __init__(self, vision_type, project_dim, cfg):
        super().__init__()
        self.encoder = VisionModel(vision_type, project_dim)
        text_path = cfg.bert_type
        if not os.path.isdir(text_path):
            text_path = "microsoft/BiomedVLP-CXR-BERT-specialized"
            print(f"[info] local text encoder not found; loading {text_path}")
        self.text_encoder = AutoModel.from_pretrained(text_path)
        self.text_encoder.requires_grad_(False)
        self.text_encoder.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(text_path)
        text_dim = int(getattr(self.text_encoder.config, "hidden_size", 768))

        vocab = max(int(self.text_encoder.config.vocab_size), len(self.tokenizer))
        type_t, num_t, side_t, vert_t = build_lexicon_tables(self.tokenizer, vocab)
        self._type_np, self._num_np = type_t.numpy(), num_t.numpy()
        self._side_np, self._vert_np = side_t.numpy(), vert_t.numpy()

        self.text_project = TextTokenAdapter(text_dim, 768, use_typing=bool(getattr(cfg, "use_text_typing", True)))

        self.assoc_v = nn.Linear(768, 768, bias=False)
        self.null_key = nn.Parameter(torch.randn(768) * 0.02)
        self.null_val = nn.Parameter(torch.randn(768) * 0.1)
        self.attn_log_scale = nn.Parameter(torch.tensor(math.log(float(getattr(cfg, "attn_init_scale", 14.0)))))

        self.norm_ctx = nn.LayerNorm(768)
        self.use_geometry = bool(getattr(cfg, "use_region_geometry", True))
        self.geom_mlp = (nn.Sequential(nn.Linear(5, 128), nn.GELU(), nn.Linear(128, 768))
                         if self.use_geometry else None)
        self.use_activity = bool(getattr(cfg, "use_activity_embed", True))
        if self.use_activity:
            self.act_on = nn.Parameter(torch.randn(768))
            self.act_off = nn.Parameter(torch.randn(768))
        self.detach_regions_in_assoc = bool(getattr(cfg, "detach_regions_in_assoc", True))

        self.fuse = nn.Linear(1536, 768)
        self.norm_T = nn.LayerNorm(768)

        self.align_r = nn.Linear(768, 256, bias=False)
        self.align_t = nn.Linear(768, 256, bias=False)
        self.spatial_dim = [7, 14, 28, 56]
        feature_dim = [768, 384, 192, 96]
        self.vis_pos = PositionalEncoding(768)
        self.head = RegionQueryHead(cfg.num_queries, in_dim=768, mask_in_dim=96,
                                    d_q=cfg.d_q, d=cfg.d, d_v=cfg.d_v, d_m=cfg.d_m,
                                    mask_pos_enc=bool(cfg.mask_pos_enc))
        use_film = bool(getattr(cfg, "use_region_film", True))
        film_hidden = int(getattr(cfg, "film_hidden", 256))
        self.film_detach_masks = bool(getattr(cfg, "film_detach_masks", True))
        self.decoder16 = RegionGuidedDecoder(feature_dim[0], feature_dim[1], self.spatial_dim[0], use_film, film_hidden)
        self.decoder8 = RegionGuidedDecoder(feature_dim[1], feature_dim[2], self.spatial_dim[1], use_film, film_hidden)
        self.decoder4 = RegionGuidedDecoder(feature_dim[2], feature_dim[3], self.spatial_dim[2], use_film, film_hidden)
        self.decoder1 = SubpixelUpsample(2, feature_dim[3], 24, 4)
        self.out = UnetOutBlock(2, in_channels=24, out_channels=1)

    def train(self, mode: bool = True):
        super().train(mode)
        self.text_encoder.eval()
        return self

    def attn_scale(self):
        return torch.exp(self.attn_log_scale.clamp(max=math.log(20.0)))

    def prepare_texts(self, texts):
        return [preprocess_text_for_encoder(t) for t in texts]

    def tokenize(self, texts, device):
        enc = self.tokenizer(self.prepare_texts(texts), padding=True, truncation=True,
                             max_length=int(args.max_text_length), return_tensors='pt')
        return enc['input_ids'].to(device), enc['attention_mask'].to(device)

    def _content_token_mask(self, input_ids, attention_mask):
        valid = attention_mask.bool().clone()
        cls_id = getattr(self.tokenizer, "cls_token_id", None)
        sep_id = getattr(self.tokenizer, "sep_token_id", None)
        if cls_id is not None:
            valid &= input_ids.ne(int(cls_id))
        if sep_id is not None:
            valid &= input_ids.ne(int(sep_id))
        return valid

    def parse_lexicon(self, input_ids, attention_mask):
        ids = input_ids.detach().cpu().numpy()
        content = self._content_token_mask(input_ids, attention_mask)
        valid = content.detach().cpu().numpy()
        B, L = ids.shape
        types, sides = self._type_np[ids], self._side_np[ids]
        verts, nums = self._vert_np[ids], self._num_np[ids]
        pside = np.zeros((B, L), dtype=np.int64)
        pvert = np.zeros((B, L), dtype=np.int64)
        cval = np.zeros(B, dtype=np.int64)
        for b in range(B):
            for p in range(L):
                if not valid[b, p]:
                    continue
                if types[b, p] == 1 and cval[b] == 0:
                    cval[b] = int(nums[b, p])
                s = int(sides[b, p])
                if s > 0:
                    v = 0
                    pside[b, p] = s
                    if p > 0 and valid[b, p - 1] and types[b, p - 1] == 2:
                        v = int(verts[b, p - 1])
                        pside[b, p - 1] = s
                        pvert[b, p - 1] = v
                    pvert[b, p] = v
                    if p + 1 < L and valid[b, p + 1] and types[b, p + 1] == 4:
                        pside[b, p + 1] = s
                        pvert[b, p + 1] = v
        dev = input_ids.device
        pside_t = torch.from_numpy(pside).to(dev)
        return {"content_mask": content, "phrase_side": pside_t,
                "phrase_vert": torch.from_numpy(pvert).to(dev),
                "phrase_mask": pside_t > 0,
                "count_val": torch.from_numpy(np.clip(cval, 0, 10)).to(dev)}

    def encode_text(self, input_ids, attention_mask, lex):
        with torch.no_grad():
            out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        return self.text_project(out.last_hidden_state, lex["phrase_side"], lex["phrase_vert"])

    def encode_hard_negatives(self, items, device):
        if not items:
            return None
        ids, mask = self.tokenize([t for t, _, _ in items], device)
        lex = self.parse_lexicon(ids, mask)
        tokens = self.encode_text(ids, mask, lex)
        return {"tokens": tokens, "key_mask": lex["phrase_mask"],
                "owner": [o for _, _, o in items], "lacks": [l for _, l, _ in items]}

    def forward(self, image, input_ids, attention_mask):
        if image.shape[1] == 1:
            image = repeat(image, 'b 1 h w -> b c h w', c=3)
        image_features = self.encoder(image)['feature']
        if len(image_features[0].shape) == 4:
            image_features = image_features[1:]
        e56 = image_features[0]
        tok = [rearrange(item, 'b c h w -> b (h w) c') for item in image_features]
        os32 = tok[3]
        regions, A, mask_logits, region_attn_logits = self.head(self.vis_pos(os32), e56, side=7)
        B, K = regions.shape[0], regions.shape[1]

        mask_probs_all = mask_logits.sigmoid()
        geom, act = region_geometry(mask_probs_all.detach())
        geom_emb = self.geom_mlp(geom) if self.geom_mlp is not None else 0.0
        if self.use_activity:
            a_ = act[..., None].to(regions.dtype)
            act_emb = a_ * self.act_on + (1.0 - a_) * self.act_off
        else:
            act_emb = 0.0
        ctx_extra = geom_emb + act_emb
        Rc_fuse = self.norm_ctx(regions) + ctx_extra
        Rc_assoc = (self.norm_ctx(regions.detach()) + ctx_extra) if self.detach_regions_in_assoc else Rc_fuse

        lex = self.parse_lexicon(input_ids, attention_mask)
        text_tokens = self.encode_text(input_ids, attention_mask, lex)

        r_n = F.normalize(self.align_r(Rc_assoc), dim=-1, eps=1e-6)
        k_n = F.normalize(self.align_t(text_tokens), dim=-1, eps=1e-6)
        null_n = F.normalize(self.align_t(self.null_key), dim=-1, eps=1e-6)
        v = self.assoc_v(text_tokens)
        scale = self.attn_scale()
        scores = torch.einsum('bkd,bld->bkl', r_n, k_n) * scale
        key_mask = lex["content_mask"]
        scores = scores.masked_fill(~key_mask[:, None, :], -1e4)
        null_score = (r_n * null_n).sum(-1, keepdim=True) * scale
        alpha_full = torch.softmax(torch.cat([null_score, scores], dim=-1), dim=-1)
        alpha_null, alpha = alpha_full[..., 0], alpha_full[..., 1:]
        t_assoc = alpha_null[..., None] * self.null_val + torch.einsum('bkl,bld->bkd', alpha, v)

        Tn = self.norm_T(t_assoc)
        W = self.fuse.weight
        contrib_R = F.linear(Rc_fuse, W[:, :768])
        contrib_T = F.linear(Tn, W[:, 768:])
        fused = contrib_R + contrib_T + self.fuse.bias

        mask_probs = mask_probs_all.detach() if self.film_detach_masks else mask_probs_all
        os16, attn16 = self.decoder16(os32, tok[2], fused, mask_probs)
        os8, attn8 = self.decoder8(os16, tok[1], fused, mask_probs)
        os4, attn4 = self.decoder4(os8, tok[0], fused, mask_probs)
        os4 = rearrange(os4, 'B (H W) C -> B C H W', H=self.spatial_dim[-1], W=self.spatial_dim[-1])
        os1 = self.decoder1(os4)
        seg_logits = self.out(os1)

        with torch.no_grad():
            fuse_T_over_R = contrib_T.norm(dim=-1).mean(1) / contrib_R.norm(dim=-1).mean(1).clamp_min(1e-8)
            a_safe = alpha_full.clamp_min(1e-12)
            nkeys = (key_mask.sum(1) + 1).clamp_min(2).to(a_safe.dtype)
            ent_q = -(a_safe * a_safe.log()).sum(-1) / nkeys.log()[:, None]
            w = act / act.sum(1, keepdim=True).clamp_min(1e-6)
            text_entropy_norm = (ent_q * w).sum(1)
            ta = F.normalize(t_assoc.float(), dim=-1)
            sim = torch.einsum('bkd,bld->bkl', ta, ta)
            ww = act[:, :, None] * act[:, None, :]
            ww = ww.masked_fill(torch.eye(K, dtype=torch.bool, device=ww.device)[None], 0.0)
            qcos = (sim * ww).sum((1, 2)) / ww.sum((1, 2)).clamp_min(1e-6)
            assoc_cos = F.cosine_similarity(self.align_r(Rc_assoc), self.align_t(t_assoc), dim=-1)

        return {'regions': regions, 'region_ctx': Rc_assoc, 'attention': A,
                'region_attention_logits': region_attn_logits,
                'mask_logits': mask_logits, 'seg_logits': seg_logits,
                'decoder_attn': [attn16, attn8, attn4],
                't_assoc': t_assoc, 'alpha': alpha, 'alpha_null': alpha_null,
                'text_attention_logits': scores, 'text_tokens': text_tokens, 'fused_regions': fused,
                'phrase_side': lex["phrase_side"], 'phrase_mask': lex["phrase_mask"],
                'content_mask': lex["content_mask"], 'association_mask': key_mask,
                'count_val': lex["count_val"],
                'assoc_cos': assoc_cos, 'act': act,
                'decoder_features': [os16, os8, os4],
                'diag_fuse_T_over_R': fuse_T_over_R,
                'diag_text_entropy_norm': text_entropy_norm,
                'diag_text_entropy_per_query': ent_q,
                'diag_tassoc_query_cos': qcos}

    def _attend(self, r, keys_n, vals, key_mask, scale):
        null_n = F.normalize(self.align_t(self.null_key), dim=-1, eps=1e-6)
        scores = torch.einsum('nld,d->nl', keys_n, r) * scale
        scores = scores.masked_fill(~key_mask, -1e4)
        null_s = (r * null_n).sum() * scale
        full = torch.cat([null_s.reshape(1, 1).expand(scores.shape[0], 1), scores], dim=1)
        a = torch.softmax(full, dim=1)
        return a[:, :1] * self.null_val[None, :] + torch.einsum('nl,nld->nd', a[:, 1:], vals)

    def alignment_loss(self, outputs, rows_batch, cols_batch, comps_list, cfg, hard_neg=None):
        eps, tau = 1e-6, float(cfg.ra_temperature)
        beta = float(getattr(cfg, "ra_token_weight", 1.0))
        gamma = float(getattr(cfg, "ra_null_weight", 1.0))
        w_unm = float(getattr(cfg, "ra_unm_null_weight", 2.0))
        region_ctx, t_assoc = outputs["region_ctx"], outputs["t_assoc"]
        text_tokens, phrase_side = outputs["text_tokens"], outputs["phrase_side"]
        key_mask, content_mask = outputs["phrase_mask"], outputs["content_mask"]
        B, K = region_ctx.shape[0], region_ctx.shape[1]
        scale = self.attn_scale()
        r_all = F.normalize(self.align_r(region_ctx), dim=-1, eps=eps)
        t_all = F.normalize(self.align_t(t_assoc), dim=-1, eps=eps)
        k_all = F.normalize(self.align_t(text_tokens), dim=-1, eps=eps)
        v_all = self.assoc_v(text_tokens)
        null_n = F.normalize(self.align_t(self.null_key), dim=-1, eps=eps)
        has_side = torch.stack([(phrase_side == 1).any(1), (phrase_side == 2).any(1)], dim=1)
        if hard_neg is not None:
            hk = F.normalize(self.align_t(hard_neg["tokens"]), dim=-1, eps=eps)
            hv = self.assoc_v(hard_neg["tokens"])
            h_mask = hard_neg["key_mask"]
            h_owner = np.asarray(hard_neg["owner"])
            h_lacks = np.asarray(hard_neg["lacks"])

        def null_loss(r, keys_n_i, mask_i):
            c = torch.matmul(keys_n_i, r)[mask_i]
            if c.numel() == 0:
                return None, None
            c_null = torch.sum(r * null_n).reshape(1)
            lg = torch.cat([c_null, c]) / tau
            loss = -F.log_softmax(lg, dim=0)[0]
            with torch.no_grad():
                hit = bool((c_null[0] > c.max()).item())
            return loss, hit

        st = {k: 0 for k in RA_HIT_KEYS + RA_COUNT_KEYS}
        seq_losses, tok_losses = [], []
        null_unm_losses, null_other_losses = [], []
        for i, (rows, cols) in enumerate(zip(rows_batch, cols_batch)):
            sides = matched_sides(rows, cols, comps_list[i], phrase_side[i], cfg)

            matched_q = set(int(q) for q in rows)
            unm_set = set(q for q in range(K) if q not in matched_q)
            null_q = sorted(unm_set)
            for q, j in zip(rows, cols):
                if int(q) in sides:
                    continue
                if gt_side_of_component(comps_list[i][j], cfg) is not None:
                    null_q.append(int(q))
                elif bool(key_mask[i].any()):
                    r_amb = r_all[i, int(q)]
                    c_amb = torch.matmul(k_all[i], r_amb)[key_mask[i]]
                    c_nl = torch.sum(r_amb * null_n).reshape(1)
                    den_amb = torch.cat([c_amb, c_nl]) / tau
                    tok_losses.append(-(torch.logsumexp(c_amb / tau, dim=0) - torch.logsumexp(den_amb, dim=0)))
                    st["tok_anchors"] += 1
            for q in null_q:
                l, hit = null_loss(r_all[i, q], k_all[i], key_mask[i])
                if l is None:
                    continue
                kind = "unm" if q in unm_set else "null"
                (null_unm_losses if kind == "unm" else null_other_losses).append(l)
                st["null_anchors"] += 1
                st[f"n_{kind}"] += 1
                st[f"hits_{kind}"] += int(hit)

            for qi, s in sides.items():
                r = r_all[i, qi]
                hi = ([m for m in range(len(h_owner)) if h_owner[m] == i and h_lacks[m] == s + 1]
                      if hard_neg is not None else [])

                own = (phrase_side[i] == s + 1)
                oth = (phrase_side[i] == 2 - s)
                c = torch.matmul(k_all[i], r)
                c_own, c_oth = c[own], c[oth]
                c_null = torch.sum(r * null_n).reshape(1)
                den = torch.cat([c_own, c_oth, c_null]) / tau
                tok_losses.append(-(torch.logsumexp(c_own / tau, dim=0) - torch.logsumexp(den, dim=0)))
                st["tok_anchors"] += 1
                if bool(oth.any()):
                    with torch.no_grad():
                        st["n_tok"] += 1
                        st["hits_tok"] += int(bool((c_own.max() > c_oth.max()).item()))

                for m in hi:
                    l, hit = null_loss(r, hk[m], h_mask[m])
                    if l is None:
                        continue
                    null_other_losses.append(l)
                    st["null_anchors"] += 1
                    st["n_null"] += 1
                    st["hits_null"] += int(hit)

                pos = torch.sum(r * t_all[i, qi])
                negs, kinds = [], {}
                other = [q2 for q2, s2 in sides.items() if s2 != s]
                if other:
                    sA = torch.matmul(t_all[i, other], r)
                    negs.append(sA); kinds["cross"] = sA; st["neg_cross"] += len(other)
                nb = [n for n in range(B) if n != i and not bool(has_side[n, s]) and bool(content_mask[n].any())]
                if nb:
                    T = self._attend(r, k_all[nb], v_all[nb], key_mask[nb], scale)
                    sB = torch.matmul(F.normalize(self.align_t(T), dim=-1, eps=eps), r)
                    negs.append(sB); kinds["batch"] = sB; st["neg_batch"] += len(nb)
                if hi:
                    T = self._attend(r, hk[hi], hv[hi], h_mask[hi], scale)
                    sH = torch.matmul(F.normalize(self.align_t(T), dim=-1, eps=eps), r)
                    negs.append(sH); kinds["hard"] = sH; st["neg_hard"] += len(hi)
                if not negs:
                    st["anchors_skipped"] += 1
                    continue
                st["positive_pairs"] += 1
                logits = torch.cat([pos.reshape(1)] + negs) / tau
                seq_losses.append(-F.log_softmax(logits, dim=0)[0])
                with torch.no_grad():
                    for name, sv in kinds.items():
                        st[f"n_{name}"] += 1
                        st[f"hits_{name}"] += int(bool((pos > sv.max()).item()))
        terms = []
        if tok_losses:
            terms.append(beta * torch.stack(tok_losses).mean())
        null_terms = []
        if null_unm_losses:
            null_terms.append(w_unm * torch.stack(null_unm_losses).mean())
        if null_other_losses:
            null_terms.append(torch.stack(null_other_losses).mean())
        if null_terms:
            terms.append(gamma * sum(null_terms))
        def _mean_f(lst):
            return float(torch.stack(lst).mean().detach()) if lst else 0.0
        st["L_seq"] = _mean_f(seq_losses)
        st["L_tok"] = beta * _mean_f(tok_losses)
        st["L_null_unm"] = gamma * w_unm * _mean_f(null_unm_losses)
        st["L_null_other"] = gamma * _mean_f(null_other_losses)
        if not terms:
            return region_ctx.sum() * 0.0, st
        return sum(terms), st


@torch.no_grad()
def hungarian_match(x, g, lam_bce, lam_dice, eps):
    K, P = x.shape
    N = g.shape[0]
    if N == 0:
        return [], [], None
    M = x.sigmoid()
    bce = (F.softplus(x).sum(1, keepdim=True) - x @ g.t()) / P
    inter = M @ g.t()
    dice = 1.0 - (2.0 * inter + eps) / (M.sum(1, keepdim=True) + g.sum(1)[None, :] + eps)
    cost = lam_bce * bce + lam_dice * dice
    cost_np = cost.detach().cpu().numpy()
    rows, cols = linear_sum_assignment(cost_np)
    return [int(r) for r in rows], [int(c) for c in cols], cost_np


def spatial_diversity_loss(logits, matched_rows, eps):
    if len(matched_rows) < 2:
        return logits.sum() * 0.0
    probs = logits[matched_rows].sigmoid().flatten(1)
    pair_losses = []
    for i in range(probs.shape[0]):
        for j in range(i + 1, probs.shape[0]):
            a, b = probs[i], probs[j]
            soft_dice = (2.0 * (a * b).sum() + eps) / (a.sum() + b.sum() + eps)
            pair_losses.append(soft_dice)
    return torch.stack(pair_losses).mean()


def region_match_loss(logits, comps, A, cfg):
    K = logits.shape[0]
    x = logits.reshape(K, -1)
    P = x.shape[1]
    N = comps.shape[0]
    g = comps.reshape(N, -1).float() if N > 0 else x.new_zeros((0, P))
    eps = float(cfg.eps)

    rows, cols, cost_np = hungarian_match(
        x.detach(), g, cfg.lambda_bce, cfg.lambda_dice, eps
    )
    zero = x.sum() * 0.0

    assigned_gt = set(cols)
    unassigned_gt = [j for j in range(N) if j not in assigned_gt]
    if unassigned_gt:
        ignore_union = g[unassigned_gt].sum(dim=0).clamp(0.0, 1.0)
        valid_pixels = 1.0 - ignore_union
    else:
        valid_pixels = x.new_ones(P)

    matched_terms = []
    if rows:
        valid_count = valid_pixels.sum().clamp_min(1.0)
        for k, j in zip(rows, cols):
            target = g[j] * valid_pixels
            logits_k = x[k]
            bce_map = F.binary_cross_entropy_with_logits(
                logits_k, g[j], reduction="none"
            )
            bce = (bce_map * valid_pixels).sum() / valid_count

            prob = logits_k.sigmoid() * valid_pixels
            inter = (prob * target).sum()
            dice_loss = 1.0 - (2.0 * inter + eps) / (
                prob.sum() + target.sum() + eps
            )
            matched_terms.append(
                float(cfg.lambda_bce) * bce + float(cfg.lambda_dice) * dice_loss
            )
        L_matched = torch.stack(matched_terms).mean()
    else:
        L_matched = zero

    matched_q = set(rows)
    unmatched_q = [k for k in range(K) if k not in matched_q]
    if unmatched_q:
        unmatched_logits = x[unmatched_q]
        L_unmatched_bce = F.binary_cross_entropy_with_logits(
            unmatched_logits, torch.zeros_like(unmatched_logits), reduction="mean"
        )
        if N > 0 and float(g.sum().detach()) > 0:
            gt_union = g.max(dim=0).values
            unmatched_prob = unmatched_logits.sigmoid()
            L_unmatched_fg = (unmatched_prob * gt_union[None, :]).sum() / (
                gt_union.sum().clamp_min(1.0) * len(unmatched_q)
            )
        else:
            L_unmatched_fg = zero
        L_unmatched = L_unmatched_bce + float(cfg.lambda_unmatched_fg) * L_unmatched_fg
    else:
        L_unmatched_bce = zero
        L_unmatched_fg = zero
        L_unmatched = zero

    L_foreign = zero
    if rows and len(rows) >= 2:
        foreign_terms = []
        for q_idx, gt_idx in zip(rows, cols):
            other_gt_indices = [j for j in cols if j != gt_idx]
            if not other_gt_indices:
                continue
            other_union = g[other_gt_indices].sum(dim=0).clamp(0.0, 1.0)
            other_area = other_union.sum()
            if float(other_area.detach()) > 0:
                foreign_terms.append((x[q_idx].sigmoid() * other_union).sum() / other_area)
        if foreign_terms:
            L_foreign = torch.stack(foreign_terms).mean()

    L_unmatched_gt_region = zero
    L_match = L_matched + float(cfg.lambda_u) * L_unmatched
    L_div = spatial_diversity_loss(logits, rows, eps)
    L_total = L_match

    info = {
        "L_total": float(L_total.detach()),
        "L_match": float(L_match.detach()),
        "L_matched": float(L_matched.detach()),
        "L_unmatched": float(L_unmatched.detach()),
        "L_unmatched_bce": float(L_unmatched_bce.detach()),
        "L_unmatched_fg": float(L_unmatched_fg.detach()),
        "L_unmatched_area": 0.0,
        "L_unmatched_gt_region": float(L_unmatched_gt_region.detach()),
        "L_foreign_region": float(L_foreign.detach()),
        "effective_lambda_foreign": 0.0,
        "L_div": float(L_div.detach()),
        "effective_lambda_div": 0.0,
        "rows": rows,
        "cols": cols,
        "cost": cost_np,
        "N": N,
        "num_unassigned_gt": len(unassigned_gt),
    }
    return L_total, info


def image_side_of_component(comp, cfg):
    c = (comp if torch.is_tensor(comp) else torch.as_tensor(comp)).detach().cpu().float()
    colsum = c.sum(0)
    if float(colsum.sum()) <= 0:
        return None
    W = c.shape[-1]
    cx = float((colsum * torch.arange(W, dtype=torch.float32)).sum() / colsum.sum())
    if abs(cx - W / 2.0) < float(getattr(cfg, "att_center_margin", 8)):
        return None
    return 1 if cx > W / 2.0 else 0


def gt_side_of_component(comp, cfg):
    s = image_side_of_component(comp, cfg)
    if s is None:
        return None
    left_on_right = str(getattr(cfg, "patient_left_on_image_side", "right")).lower() == "right"
    return (0 if s == 1 else 1) if left_on_right else s


def matched_sides(rows, cols, comps_i, phrase_side_i, cfg):
    out = {}
    for q, j in zip(rows, cols):
        s = gt_side_of_component(comps_i[j], cfg)
        if s is None:
            continue
        if not bool((phrase_side_i == s + 1).any()):
            continue
        out[int(q)] = s
    return out


def attention_loss(outputs, rows_batch, cols_batch, comps_list, cfg):
    alpha, phrase_side = outputs["alpha"], outputs["phrase_side"]
    terms = []
    for i, (rows, cols) in enumerate(zip(rows_batch, cols_batch)):
        sides = matched_sides(rows, cols, comps_list[i], phrase_side[i], cfg)
        for q, s in sides.items():
            m = (phrase_side[i] == s + 1).to(alpha.dtype)
            terms.append(-torch.log((alpha[i, q] * m).sum().clamp_min(1e-4)))
    zero = alpha.sum() * 0.0
    return (torch.stack(terms).mean() if terms else zero), len(terms)


def compute_text_losses(model, outputs, rows_batch, cols_batch, comps_list, texts, cfg, hn_rng, device):
    hard = None
    if bool(getattr(cfg, "ra_hard_negatives", True)):
        items = []
        for i, t in enumerate(texts):
            for ht, lacks in make_location_hard_negatives(t, hn_rng, max_out=int(cfg.hn_per_image)):
                items.append((ht, lacks, i))
        hard = model.encode_hard_negatives(items, device)
    l_ra, ra_stats = model.alignment_loss(outputs, rows_batch, cols_batch, comps_list, cfg, hard_neg=hard)
    l_att, n_att = attention_loss(outputs, rows_batch, cols_batch, comps_list, cfg)
    return {"l_ra": l_ra, "l_att": l_att, "ra_stats": ra_stats, "n_att": n_att}


@torch.no_grad()
def sample_diagnostics(logits, comps, A, rows, cols, cfg):
    K = logits.shape[0]
    M = logits.sigmoid().reshape(K, -1)
    Bm = (M > cfg.mask_binarize_thr).float()
    area = Bm.sum(1)
    out = {"n_active_queries": int((area > 0).sum().item())}

    ious = []
    inter = Bm @ Bm.t()
    union = area[:, None] + area[None, :] - inter
    for a in range(K):
        for b in range(a + 1, K):
            if area[a] > 0 and area[b] > 0:
                ious.append(float(inter[a, b] / union[a, b].clamp(min=1.0)))
    out["max_pair_iou"] = max(ious) if ious else 0.0
    out["mean_pair_iou"] = float(np.mean(ious)) if ious else 0.0
    out["queries_overlap"] = bool(out["max_pair_iou"] > cfg.collapse_iou_thr)

    An = F.normalize(A.reshape(K, -1), dim=1)
    sim = An @ An.t()
    off = sim[~torch.eye(K, dtype=torch.bool, device=sim.device)]
    out["attn_cos_mean"] = float(off.mean()) if K > 1 else 0.0

    N = comps.shape[0]
    dices = []
    if N > 0:
        g = comps.reshape(N, -1).float()
        for k, j in zip(rows, cols):
            d = 2.0 * (Bm[k] * g[j]).sum() / (Bm[k].sum() + g[j].sum()).clamp(min=1.0)
            dices.append(float(d))
    out["matched_dice"] = float(np.mean(dices)) if dices else float("nan")
    out["gt_hits"] = int(sum(d >= 0.5 for d in dices))
    out["gt_total"] = int(N)

    unmatched_active = [k for k in range(K) if k not in set(rows) and float(area[k]) > 0]
    out["n_unm_active"] = len(unmatched_active)
    out["gt_multi_covered"] = 0
    out["n_unm_dup"] = 0
    out["n_queries_cover_multiple_gt"] = 0
    if N > 0:
        inter_kg = Bm @ g.t()
        cover = inter_kg / g.sum(1).clamp(min=1.0)[None, :]
        contain = inter_kg / area.clamp(min=1.0)[:, None]
        covered = (cover >= cfg.cover_thr) & (area > 0)[:, None]
        out["gt_multi_covered"] = int((covered.sum(0) >= 2).sum().item())
        out["n_unm_dup"] = int(sum(bool((contain[k] >= cfg.dup_thr).any()) for k in unmatched_active))
        out["n_queries_cover_multiple_gt"] = int(((cover >= cfg.cover_thr).sum(1) >= 2).sum().item())
    return out


@torch.no_grad()
def attention_grounding_diagnostics(alpha_i, alpha_null_i, phrase_side_i, assoc_cos_i, rows, cols, comps, cfg):
    cm, wm, nm, base, amax = [], [], [], [], []
    o_null, o_wrong, a_sup, a_orph = [], [], [], []
    tally = np.zeros((2, 2))
    n_both = 0
    n_phr = int((phrase_side_i > 0).sum())
    has_both = bool((phrase_side_i == 1).any()) and bool((phrase_side_i == 2).any())
    mL = (phrase_side_i == 1).to(alpha_i.dtype)
    mR = (phrase_side_i == 2).to(alpha_i.dtype)
    for k, j in zip(rows, cols):
        a = alpha_i[k]
        tok_mass, null_mass = float(a.sum()), float(alpha_null_i[k])
        amax.append(int(a.argmax()) if tok_mass > null_mass else -1)
        s = gt_side_of_component(comps[j], cfg)
        img_s = image_side_of_component(comps[j], cfg)
        if s is None or img_s is None:
            continue
        m_c = (phrase_side_i == s + 1)
        if bool(m_c.any()):
            cm.append(float((a * m_c.to(a.dtype)).sum()))
            wm.append(float((a * (phrase_side_i == (2 - s)).to(a.dtype)).sum()))
            nm.append(null_mass)
            base.append(float(m_c.sum()) / (n_phr + 1))
            a_sup.append(float(assoc_cos_i[k]))
        else:
            o_null.append(null_mass)
            o_wrong.append(float((a * (phrase_side_i == (2 - s)).to(a.dtype)).sum()))
            a_orph.append(float(assoc_cos_i[k]))
        if has_both:
            massL, massR = float((a * mL).sum()), float((a * mR).sum())
            n_both += 1
            if massL + massR > 1e-6 and massL != massR:
                tally[img_s, 0 if massL > massR else 1] += 1
    matched_set = set(int(r) for r in rows)
    um_null = [float(alpha_null_i[k]) for k in range(alpha_i.shape[0]) if k not in matched_set]
    um_loc = [float(alpha_i[k].sum()) for k in range(alpha_i.shape[0]) if k not in matched_set]
    nan = float("nan")
    return {"att_correct_mass": float(np.mean(cm)) if cm else nan,
            "att_wrong_mass": float(np.mean(wm)) if wm else nan,
            "att_null_mass": float(np.mean(nm)) if nm else nan,
            "att_uniform_baseline": float(np.mean(base)) if base else nan,
            "att_argmax_unique_frac": (len(set(amax)) / len(amax)) if amax else nan,
            "att_argmax_null_frac": (sum(1 for x in amax if x == -1) / len(amax)) if amax else nan,
            "att_orphan_null_mass": float(np.mean(o_null)) if o_null else nan,
            "att_orphan_wrong_mass": float(np.mean(o_wrong)) if o_wrong else nan,
            "att_unmatched_null_mass": float(np.mean(um_null)) if um_null else nan,
            "att_unmatched_loc_mass": float(np.mean(um_loc)) if um_loc else nan,
            "assoc_cos_supported": float(np.mean(a_sup)) if a_sup else nan,
            "assoc_cos_orphan": float(np.mean(a_orph)) if a_orph else nan,
            "n_orphan": len(o_null),
            "tally_i0_t0": tally[0, 0], "tally_i0_t1": tally[0, 1],
            "tally_i1_t0": tally[1, 0], "tally_i1_t1": tally[1, 1], "tally_n": n_both}


def render_sample(rec, cfg, save_path):
    img = rec["image"]
    comps = rec["comps"].numpy().astype(bool)
    logits = rec["logits"].detach().cpu()
    A = rec["A"].detach().cpu()
    K, H, W = logits.shape
    N = comps.shape[0]

    x = logits.reshape(K, -1)
    g = torch.from_numpy(comps.reshape(N, -1)).float() if N > 0 else x.new_zeros((0, H * W))
    rows, cols, cost = hungarian_match(
        x, g, cfg.lambda_bce, cfg.lambda_dice, cfg.eps
    )
    match = {int(k): int(j) for k, j in zip(rows, cols)}
    matched_queries = set(match.keys())

    M_raw = logits.sigmoid().numpy()
    B_raw = M_raw > cfg.mask_binarize_thr

    cmap = plt.get_cmap("tab10")
    ncols = max(K, 4)
    fig, axes = plt.subplots(3, ncols, figsize=(4.2 * ncols, 12.5))
    for ax in axes.ravel():
        ax.axis("off")

    axes[0, 0].imshow(img, cmap="gray")
    axes[0, 0].set_title(f"Image (N = {N} GT components)")

    ov = np.stack([img, img, img], -1).astype(np.float32)
    ov = (ov - ov.min()) / (ov.max() - ov.min() + 1e-6)
    ov_gt = ov.copy()
    for j in range(N):
        color = np.asarray(cmap(j % 10)[:3])
        ov_gt[comps[j]] = 0.45 * ov_gt[comps[j]] + 0.55 * color
    axes[0, 1].imshow(ov_gt)
    axes[0, 1].set_title("GT components G1..GN")
    for j in range(N):
        ys, xs = np.nonzero(comps[j])
        if len(xs):
            axes[0, 1].text(xs.mean(), ys.mean(), f"G{j + 1}", color="white",
                            fontsize=9, ha="center", va="center", fontweight="bold")

    axes[0, 2].imshow(ov)
    for k in range(K):
        color = cmap(match[k] % 10) if k in match else (1.0, 0.25, 0.25, 1.0)
        if B_raw[k].any():
            axes[0, 2].contour(B_raw[k].astype(float), levels=[0.5],
                               colors=[color], linewidths=1.8,
                               linestyles="solid" if k in match else "dashed")
    axes[0, 2].set_title("ALL raw query contours\nsolid=matched, dashed=unmatched")

    txt = "Cost C_kj (rows Q_k, cols G_j)\n[ ] = Hungarian assignment\n\n"
    if N > 0:
        txt += "      " + "".join(f"  G{j + 1:<5d}" for j in range(N)) + "\n"
        for k in range(K):
            line = f"Q{k + 1:<4d} "
            for j in range(N):
                cell = f"{cost[k, j]:.3f}"
                line += f" [{cell}]" if match.get(k) == j else f"  {cell} "
            txt += line + "\n"
        unmatched_j = [j + 1 for j in range(N) if j not in set(cols)]
        unmatched_q = [k + 1 for k in range(K) if k not in matched_queries]
        txt += f"\nunmatched queries: {['Q%d' % q for q in unmatched_q] or '-'}"
        txt += f"\nunmatched GT: {['G%d' % q for q in unmatched_j] or '-'}"
    else:
        txt += "N = 0 -> all queries unmatched; raw masks are still shown"
    axes[0, 3].text(0.0, 1.0, txt, family="monospace", fontsize=8.5, va="top")
    axes[0, 3].set_title("Hungarian matching")

    for k in range(K):
        axes[1, k].imshow(M_raw[k], cmap="viridis", vmin=0, vmax=1)
        if k in match:
            j = match[k]
            if comps[j].any():
                axes[1, k].contour(comps[j].astype(float), levels=[0.5],
                                   colors=[cmap(j % 10)], linewidths=1.6)
            inter = int((B_raw[k] & comps[j]).sum())
            d = 2.0 * inter / max(1, int(B_raw[k].sum()) + int(comps[j].sum()))
            iou = inter / max(1, int((B_raw[k] | comps[j]).sum()))
            axes[1, k].set_title(f"Q{k + 1} -> G{j + 1} | Dice={d:.2f}, IoU={iou:.2f}")
        else:
            axes[1, k].set_title(f"Q{k + 1}: UNMATCHED (RAW)\narea={int(B_raw[k].sum())} px")

    A_up = F.interpolate(A[None], size=(H, W), mode="bilinear", align_corners=False)[0].numpy()
    for k in range(K):
        axes[2, k].imshow(img, cmap="gray")
        axes[2, k].imshow(A_up[k], cmap="jet", alpha=0.5)
        suffix = "matched" if k in matched_queries else "unmatched"
        axes[2, k].set_title(f"A{k + 1} | {suffix}")

    fig.suptitle(f"{rec['name']} | K={K} | N={N} | all raw masks displayed", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def render_query_mask_text_attention(rec, cfg, save_path):
    img = np.asarray(rec["image"])
    logits = rec["logits"].detach().cpu()
    alpha = rec["alpha"].detach().cpu().numpy()
    alpha_null = rec["alpha_null"].detach().cpu().numpy()
    phrase_side = rec["phrase_side"].detach().cpu().numpy()
    phrase_mask = rec["phrase_mask"].detach().cpu().numpy().astype(bool)
    content_mask = rec["content_mask"].detach().cpu().numpy().astype(bool)
    tokens = list(rec["tokens"])
    comps = rec["comps"].numpy().astype(bool)
    K, H, W = logits.shape
    left_on_right = str(cfg.patient_left_on_image_side).lower() == "right"
    thr = float(cfg.mask_binarize_thr)

    def patient_name(img_side):
        if img_side is None:
            return "ambiguous(midline)"
        ps = (0 if img_side == 1 else 1) if left_on_right else img_side
        return ("LEFT", "RIGHT")[ps]

    x = logits.reshape(K, -1)
    N = comps.shape[0]
    g = torch.from_numpy(comps.reshape(N, -1)).float() if N else x.new_zeros((0, H * W))
    rows, cols, _ = hungarian_match(x, g, cfg.lambda_bce, cfg.lambda_dice, cfg.eps)
    match = {int(k): int(j) for k, j in zip(rows, cols)}

    display_img = img.astype(np.float32)
    display_img = (display_img - display_img.min()) / (display_img.max() - display_img.min() + 1e-8)

    fig, axes = plt.subplots(K, 2, figsize=(14.5, max(4.0 * K, 7.0)), squeeze=False)
    for k in range(K):
        prob = logits[k].sigmoid().numpy()
        binm = prob > thr
        axm = axes[k, 0]
        axm.imshow(display_img, cmap="gray", vmin=0, vmax=1)
        axm.imshow(prob, cmap="jet", vmin=0, vmax=1, alpha=0.45)
        axm.axvline(W / 2.0, color="yellow", linestyle="--", linewidth=1.0)
        axm.axhline(H / 2.0, color="cyan", linestyle=":", linewidth=0.7)
        if binm.any():
            ys, xs = np.nonzero(binm)
            axm.plot([xs.mean()], [ys.mean()], marker="+", color="cyan", markersize=14, mew=2.5)
            q_img_side = image_side_of_component(torch.from_numpy(binm.astype(np.float32)), cfg)
            q_desc = (f"own mask: centroid x={xs.mean():.0f}/{W} -> img-{'R' if q_img_side == 1 else 'L' if q_img_side == 0 else '?'}"
                      f" -> patient {patient_name(q_img_side)}")
        else:
            q_desc = "own mask: EMPTY (no pixel > thr)"
        gt_side = None
        if k in match and comps[match[k]].any():
            axm.contour(comps[match[k]].astype(float), levels=[0.5], colors=["white"],
                        linewidths=1.4, linestyles="dashed")
            gt_img_side = image_side_of_component(torch.from_numpy(comps[match[k]].astype(np.float32)), cfg)
            gt_side = gt_side_of_component(comps[match[k]], cfg)
            head = f"Q{k+1} matched G{match[k]+1} | GT: img-{'R' if gt_img_side == 1 else 'L' if gt_img_side == 0 else '?'} -> patient {patient_name(gt_img_side)}"
        else:
            head = f"Q{k+1} UNMATCHED (expects NULL)"
        axm.set_title(head + "\n" + q_desc, fontsize=9)
        axm.axis("off")

        a = alpha[k]
        valid_idx = [
            j for j in range(min(len(tokens), len(a), len(content_mask)))
            if content_mask[j]
        ]
        labels, values, sides = [], [], []
        for j in valid_idx:
            labels.append(f"{str(tokens[j]).replace('##', '▸')}@{j}")
            values.append(float(a[j]))
            sides.append(int(phrase_side[j]) if phrase_mask[j] else 0)
        labels.append("[NULL]")
        values.append(float(alpha_null[k]))
        sides.append(0)

        left_mass = sum(v for v, s in zip(values, sides) if s == 1)
        right_mass = sum(v for v, s in zip(values, sides) if s == 2)
        null_mass = float(alpha_null[k])
        has_L, has_R = bool((phrase_side == 1).any()), bool((phrase_side == 2).any())

        if k in match and gt_side is not None:
            exp_phrase = gt_side + 1
            caption_has = has_L if exp_phrase == 1 else has_R
            if caption_has:
                exp_name = ("LEFT", "RIGHT")[exp_phrase - 1] + " phrase"
                exp_mass = left_mass if exp_phrase == 1 else right_mass
                oth_mass = right_mass if exp_phrase == 1 else left_mass
                ok = exp_mass > max(oth_mass, null_mass)
            else:
                exp_name = f"NULL (caption has no {('LEFT', 'RIGHT')[exp_phrase - 1]} phrase)"
                exp_mass, oth_mass = null_mass, left_mass + right_mass
                ok = null_mass > (left_mass + right_mass)
        elif k in match:
            exp_name = "n/a (GT near midline)"
            exp_mass, oth_mass, ok = float("nan"), float("nan"), None
        else:
            exp_name = "NULL (unmatched)"
            exp_mass, oth_mass = null_mass, left_mass + right_mass
            ok = null_mass > (left_mass + right_mass)

        ax = axes[k, 1]
        n_tok_bars = len(labels)
        labels_plot = labels + ["SUM LEFT tokens", "SUM RIGHT tokens"]
        values_plot = values + [left_mass, right_mass]
        sides_plot = sides + [1, 2]
        y = np.arange(len(labels_plot))
        colors = ["#377eb8" if s_ == 1 else "#e41a1c" if s_ == 2 else "#777777" for s_ in sides_plot]
        alphas = [0.9] * n_tok_bars + [0.5, 0.5]
        bars = ax.barh(y, values_plot, color=colors)
        for b_i, (bar, s_) in enumerate(zip(bars, sides_plot)):
            bar.set_alpha(alphas[b_i])
            if k in match and gt_side is not None and s_ == gt_side + 1:
                bar.set_hatch("//")
                bar.set_edgecolor("black")
            if b_i >= n_tok_bars:
                bar.set_edgecolor("black")
                bar.set_linewidth(1.2)
        for yy, vv in zip(y, values_plot):
            ax.text(vv + 0.01, yy, f"{vv:.3f}", va="center", fontsize=7)
        ax.axhline(n_tok_bars - 0.5, color="black", linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(labels_plot, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.0)
        ax.set_xlabel("Attention weight, absolute scale 0-1 (blue = LEFT phrase, red = RIGHT phrase, "
                      "grey = other content token / NULL; hatched = expected; bottom two bars = sums over spatial phrase tokens)")
        ax.grid(axis="x", alpha=0.2)
        verdict = "" if ok is None else ("  ✓" if ok else "  ✗")
        ax.set_title(f"Q{k+1} text attention | expects: {exp_name}{verdict}\n"
                     f"SUM over spatial phrase tokens: LEFT={left_mass:.3f}  RIGHT={right_mass:.3f}  NULL={null_mass:.3f}", fontsize=9)

    fig.suptitle(f"{rec['name']} | frame = model input (rot90_k={int(cfg.rot90_k) % 4}, "
                 f"patient-left on image-{cfg.patient_left_on_image_side}) | per-Query soft masks + token attention",
                 fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def evaluate(model, loader, device, cfg, collect_vis=False, vis_limit_per_category=None):
    model.eval()
    sums = {"L_total": 0.0, "L_match": 0.0, "L_matched": 0.0,
            "L_unmatched": 0.0, "L_unmatched_bce": 0.0, "L_unmatched_fg": 0.0,
            "L_unmatched_gt_region": 0.0, "L_foreign_region": 0.0,
            "L_div": 0.0, "L_seg": 0.0, "L_RA": 0.0, "L_att": 0.0,
            "L_RA_seq": 0.0, "L_RA_tok": 0.0, "L_RA_null_unm": 0.0, "L_RA_null_other": 0.0}
    n_img = 0
    per_image, per_match = [], []
    match_freq = np.zeros(cfg.num_queries)
    vis_store = {c: [] for c in range(5)}
    hn_rng = random.Random(123)
    ra = {k: 0 for k in RA_HIT_KEYS}
    act_m_all, act_u_all = [], []

    for images, gts, comps_list, names, texts in tqdm(loader, desc="valid", leave=False):
        images = images.to(device)
        bs = int(images.shape[0])
        input_ids, attention_mask = model.tokenize(texts, device)
        outputs = model(images, input_ids, attention_mask)
        A, logits = outputs["attention"], outputs["mask_logits"]
        seg_loss = DiceCELoss(sigmoid=True)(outputs["seg_logits"], gts.to(device).float())
        rows_b, cols_b = [], []
        for i in range(bs):
            comps = comps_list[i].to(device)
            _, info = region_match_loss(logits[i], comps, A[i], cfg)
            rows_b.append(info["rows"]); cols_b.append(info["cols"])
            am_, au_ = act_split(outputs["act"][i].detach().cpu(), info["rows"])
            act_m_all += am_; act_u_all += au_
            diag = sample_diagnostics(logits[i], comps, A[i], info["rows"], info["cols"], cfg)
            att = attention_grounding_diagnostics(
                outputs["alpha"][i], outputs["alpha_null"][i], outputs["phrase_side"][i],
                outputs["assoc_cos"][i], info["rows"], info["cols"], comps_list[i], cfg)
            tcount = int(outputs["count_val"][i].item())
            cnt = {"text_count_value": tcount,
                   "text_count_eq_N": (float(tcount == info["N"]) if tcount > 0 else float("nan"))}
            pred_bin = logits[i].sigmoid() > cfg.mask_binarize_thr
            for q_idx, gt_idx in zip(info["rows"], info["cols"]):
                pred = pred_bin[q_idx].reshape(-1).float()
                target = comps[gt_idx].reshape(-1).float()
                inter = float((pred * target).sum().item())
                union = float(((pred + target) > 0).sum().item())
                dice = 2.0 * inter / max(1.0, float(pred.sum().item() + target.sum().item()))
                iou = inter / max(1.0, union)
                per_match.append({"image": names[i], "query_id": int(q_idx + 1),
                                  "gt_component_id": int(gt_idx + 1), "N_gt_components": int(info["N"]),
                                  "matching_cost": float(info["cost"][q_idx, gt_idx]) if info["cost"] is not None else float("nan"),
                                  "binary_dice": dice, "binary_iou": iou,
                                  "success_dice_ge_0_5": bool(dice >= 0.5),
                                  "success_iou_ge_0_5": bool(iou >= 0.5)})
            for k in info["rows"]:
                match_freq[k] += 1
            for key in ("L_match", "L_matched", "L_unmatched", "L_unmatched_bce", "L_unmatched_fg", "L_unmatched_gt_region", "L_foreign_region", "L_div"):
                sums[key] += info[key]
            sums["L_seg"] += float(seg_loss.detach())
            sums["L_total"] += info["L_total"] + float(cfg.lambda_seg) * float(seg_loss.detach())
            n_img += 1
            per_image.append({"name": names[i], "N": info["N"], **diag, **att, **cnt,
                              "L_total": info["L_total"], "L_match": info["L_match"], "L_div": info["L_div"]})
            if collect_vis:
                c = min(info["N"], 4)
                vis_limit = (cfg.vis_per_category if vis_limit_per_category is None
                             else int(vis_limit_per_category))
                if len(vis_store[c]) < vis_limit:
                    vis_store[c].append({
                        "name": names[i], "image": images[i, 0].cpu().numpy(),
                        "comps": comps_list[i].cpu(), "logits": logits[i].cpu(),
                        "A": A[i].cpu(),
                        "alpha": outputs["alpha"][i].cpu(),
                        "alpha_null": outputs["alpha_null"][i].cpu(),
                        "phrase_side": outputs["phrase_side"][i].cpu(),
                        "phrase_mask": outputs["phrase_mask"][i].cpu(),
                        "content_mask": outputs["content_mask"][i].cpu(),
                        "tokens": model.tokenizer.convert_ids_to_tokens(input_ids[i].detach().cpu().tolist())
                    })

        tl = compute_text_losses(model, outputs, rows_b, cols_b, comps_list, texts, cfg, hn_rng, device)
        for name, key in (("L_RA", "l_ra"), ("L_att", "l_att")):
            sums[name] += float(tl[key]) * bs
        for name, key in (("L_RA_seq", "L_seq"), ("L_RA_tok", "L_tok"),
                          ("L_RA_null_unm", "L_null_unm"), ("L_RA_null_other", "L_null_other")):
            sums[name] += float(tl["ra_stats"][key]) * bs
        sums["L_total"] += (float(cfg.lambda_ra) * float(tl["l_ra"]) + float(cfg.lambda_att) * float(tl["l_att"])) * bs
        for k in ra:
            ra[k] += tl["ra_stats"][k]

    df = pd.DataFrame(per_image)
    res = {k: v / max(1, n_img) for k, v in sums.items()}
    res["matched_dice"] = float(df["matched_dice"].mean())
    res["gt_recall@0.5"] = float(df["gt_hits"].sum() / max(1, df["gt_total"].sum()))
    res["frac_images_queries_overlap"] = float(df["queries_overlap"].mean())
    multi = df[df["N"] >= 2]
    res["frac_overlap_images_N>=2"] = float(multi["queries_overlap"].mean()) if len(multi) else float("nan")
    res["mean_max_pair_iou"] = float(df["max_pair_iou"].mean())
    res["attn_cos_mean"] = float(df["attn_cos_mean"].mean())
    res["mean_active_queries"] = float(df["n_active_queries"].mean())
    res["frac_gt_regions_covered_by_>=2_queries"] = float(df["gt_multi_covered"].sum() / max(1, df["gt_total"].sum()))
    res["frac_images_with_gt_covered_by_>=2_queries"] = float((df["gt_multi_covered"] > 0).mean())
    res["mean_unmatched_active_queries"] = float(df["n_unm_active"].mean())
    res["frac_unmatched_active_that_duplicate_a_GT"] = float(df["n_unm_dup"].sum() / max(1, df["n_unm_active"].sum()))
    res["mean_queries_cover_multiple_gt"] = float(df["n_queries_cover_multiple_gt"].mean())
    res["frac_images_with_query_covering_multiple_gt"] = float((df["n_queries_cover_multiple_gt"] > 0).mean())
    res["act_matched"] = float(np.mean(act_m_all)) if act_m_all else float("nan")
    res["act_unmatched"] = float(np.mean(act_u_all)) if act_u_all else float("nan")
    for col in ("att_correct_mass", "att_wrong_mass", "att_null_mass", "att_uniform_baseline",
                "att_argmax_unique_frac", "att_argmax_null_frac", "att_orphan_null_mass",
                "att_orphan_wrong_mass", "att_unmatched_null_mass", "att_unmatched_loc_mass",
                "assoc_cos_supported", "assoc_cos_orphan"):
        res[col] = float(df[col].mean())
    tn = max(1.0, float(df["tally_n"].sum()))
    t00, t01, t10, t11 = (float(df[c].sum()) for c in ("tally_i0_t0", "tally_i0_t1", "tally_i1_t0", "tally_i1_t1"))
    res["side_map_acc_conv_right"] = (t10 + t01) / tn
    res["side_map_acc_conv_left"] = (t00 + t11) / tn
    res["att_pred_right_phrase_frac"] = (t01 + t11) / max(1.0, t00 + t01 + t10 + t11)
    res["ra_acc_cross"] = ra["hits_cross"] / max(1, ra["n_cross"])
    res["ra_acc_batch"] = ra["hits_batch"] / max(1, ra["n_batch"])
    res["ra_acc_hard"] = ra["hits_hard"] / max(1, ra["n_hard"])
    res["ra_acc_tok"] = ra["hits_tok"] / max(1, ra["n_tok"])
    res["ra_acc_null"] = ra["hits_null"] / max(1, ra["n_null"])
    res["ra_acc_unm"] = ra["hits_unm"] / max(1, ra["n_unm"])
    res["text_count_eq_N"] = float(df["text_count_eq_N"].mean())
    res["query_match_freq"] = (match_freq / max(1, match_freq.sum())).round(3).tolist()
    return res, df, vis_store, pd.DataFrame(per_match)


def _paired_stats(df, conditions, out_path, metric="seg_dice", n_boot=2000):
    base = df[df["condition"] == "correct"].set_index("image")[metric]
    rng = np.random.default_rng(0)
    rows = []
    for cond in conditions:
        if cond == "correct":
            continue
        cur = df[df["condition"] == cond].set_index("image")[metric]
        d = (cur - base.reindex(cur.index)).dropna().values
        if len(d) == 0:
            continue
        boots = np.array([rng.choice(d, size=len(d), replace=True).mean() for _ in range(n_boot)])
        rows.append({"condition": cond, "metric": metric, "n_images": int(len(d)), "mean_delta": float(d.mean()),
                     "ci95_low": float(np.percentile(boots, 2.5)), "ci95_high": float(np.percentile(boots, 97.5)),
                     "n_images_worse": int((d < -1e-6).sum()), "n_images_better": int((d > 1e-6).sum()),
                     "ci_excludes_zero": bool(np.percentile(boots, 2.5) > 0 or np.percentile(boots, 97.5) < 0)})
    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    return out


def evaluate_text_perturbations(model, loader, device, cfg, out_dir):
    model.eval()
    conditions = ["correct", "count_changed", "count_removed", "location_removed",
                  "one_side_removed", "left_side_removed", "mismatched"]
    rows, alpha_rows, token_rows, attn_summary_rows = [], [], [], []
    all_texts, all_names = [], []
    for batch in loader:
        all_texts.extend([str(t) for t in batch[4]])
        all_names.extend([str(n) for n in batch[3]])
    if not all_names:
        raise RuntimeError("Text perturbation evaluation received an empty loader.")
    mismatch_text, mismatch_source = build_mismatched_caption_map(
        all_names, all_texts, split_label=out_dir.name)
    reference = {}

    def masked_mean(x, mask):
        w = mask.to(dtype=x.dtype).unsqueeze(-1)
        return (x * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)

    def stats(a, b, eps=1e-12):
        a = a.float().reshape(-1); b = b.float().reshape(-1)
        mae = torch.mean(torch.abs(a - b))
        l2diff = torch.linalg.vector_norm(a - b)
        l2base = torch.linalg.vector_norm(b).clamp_min(eps)
        rel_l2 = l2diff / l2base
        cos = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=1, eps=eps)[0]
        return float(mae.cpu()), float(rel_l2.cpu()), float(cos.cpu())

    def valid_logit_mae(cur, ref):
        L = min(cur.shape[-1], ref.shape[-1])
        c, r = cur[..., :L], ref[..., :L]
        ok = (c > -1e3) & (r > -1e3)
        return float((c - r).abs()[ok].mean()) if bool(ok.any()) else float("nan")

    def count_cc(binary_np):
        lab_p = label(binary_np.astype(np.uint8), connectivity=cfg.connectivity)
        cnts = np.bincount(lab_p.ravel())
        return int(sum(cnts[i] >= cfg.min_component_area for i in range(1, int(lab_p.max()) + 1)))

    with torch.no_grad():
        for condition in conditions:
            for images, gts, comps_list, names, texts in loader:
                names = [str(n) for n in names]
                texts = [str(t) for t in texts]
                images = images.to(device); gts = gts.to(device).float()
                if condition == "correct":
                    cond_texts = list(texts)
                elif condition == "mismatched":
                    cond_texts = [mismatch_text[n] for n in names]
                else:
                    cond_texts = [perturb_text(t, condition) for t in texts]
                if condition == "mismatched":
                    for n, orig, mt in zip(names, texts, cond_texts):
                        if normalized_caption_key(mt) == normalized_caption_key(orig):
                            raise RuntimeError(
                                f"Mismatched annotation equals original after the exact encoder cleanup for {n!r}; "
                                f"selected source={mismatch_source.get(n)!r}. This should have been caught by preflight.")

                ids, attn = model.tokenize(cond_texts, device)
                out = model(images, ids, attn)
                probs = out["seg_logits"].sigmoid(); preds = probs > 0.5
                inter = (preds.float() * gts).sum(dim=(1, 2, 3))
                denom = preds.float().sum(dim=(1, 2, 3)) + gts.sum(dim=(1, 2, 3))
                dice = (2.0 * inter + 1e-6) / (denom + 1e-6)
                union = preds.float().sum(dim=(1, 2, 3)) + gts.sum(dim=(1, 2, 3)) - inter
                iou = (inter + 1e-6) / (union + 1e-6)
                seg_loss_per_image = [DiceCELoss(sigmoid=True)(out["seg_logits"][j:j + 1], gts[j:j + 1])
                                      for j in range(len(texts))]
                pooled_text = masked_mean(out["text_tokens"], attn)
                alpha = out["alpha"]
                text_logits = out["text_attention_logits"]
                ent_q = out["diag_text_entropy_per_query"]
                region_logits = out["region_attention_logits"]
                spatial_alpha = out["attention"].flatten(2)
                spatial_ent = -(spatial_alpha.clamp_min(1e-12) * spatial_alpha.clamp_min(1e-12).log()).sum(-1)
                spatial_norm_ent = spatial_ent / math.log(spatial_alpha.shape[-1])

                for j, name in enumerate(names):
                    comps = comps_list[j].to(device)
                    _, mi = region_match_loss(out["mask_logits"][j], comps, out["attention"][j], cfg)
                    assoc_j = out["assoc_cos"][j]
                    att = attention_grounding_diagnostics(alpha[j], out["alpha_null"][j], out["phrase_side"][j],
                                                          assoc_j, mi["rows"], mi["cols"], comps_list[j], cfg)
                    assoc_matched = (float(np.mean([float(assoc_j[k]) for k in mi["rows"]]))
                                     if mi["rows"] else float("nan"))
                    prob_np = probs[j].cpu().numpy(); logit = out["seg_logits"][j].detach()
                    valid_ids = ids[j][attn[j].bool()].cpu().tolist()
                    valid_tokens = model.tokenizer.convert_ids_to_tokens(valid_ids)
                    encoder_text = preprocess_text_for_encoder(cond_texts[j])
                    token_rows.append({"condition": condition, "image": name, "text_original": texts[j], "text_used_before_cleanup": cond_texts[j],
                                       "text_passed_to_encoder": encoder_text,
                                       "input_ids": " ".join(map(str, valid_ids)), "tokens": " ".join(valid_tokens),
                                       "valid_token_count": len(valid_ids),
                                       "input_ids_same_as_correct": valid_ids == reference.get(name, {}).get("valid_ids", valid_ids) if condition != "correct" else True,
                                       "mismatch_source_image": (mismatch_source.get(name, "") if condition == "mismatched" else "")})
                    if condition == "correct":
                        reference[name] = {"prob": prob_np.copy(), "logit": logit.cpu().numpy().copy(), "pred": (prob_np > 0.5),
                                           "valid_ids": valid_ids, "pooled_text": pooled_text[j].cpu().clone(), "text_tokens": out["text_tokens"][j].cpu().clone(),
                                           "t_assoc": out["t_assoc"][j].cpu().clone(), "fused": out["fused_regions"][j].cpu().clone(),
                                           "decoder16": out["decoder_features"][0][j].cpu().clone(), "decoder8": out["decoder_features"][1][j].cpu().clone(),
                                           "decoder4": out["decoder_features"][2][j].cpu().clone(), "alpha": alpha[j].cpu().clone(),
                                           "text_logits": text_logits[j].cpu().clone(), "region_logits": region_logits[j].cpu().clone(),
                                           "spatial_alpha": spatial_alpha[j].cpu().clone()}
                    ref = reference.get(name)
                    have_ref = (condition != "correct") and (ref is not None)
                    pooled_mae, pooled_rel, pooled_cos = stats(pooled_text[j].cpu(), ref["pooled_text"]) if have_ref else (0., 0., 1.)
                    text_mae, text_rel, text_cos = pooled_mae, pooled_rel, pooled_cos
                    ta_mae, ta_rel, ta_cos = stats(out["t_assoc"][j].cpu(), ref["t_assoc"]) if have_ref else (0., 0., 1.)
                    fu_mae, fu_rel, fu_cos = stats(out["fused_regions"][j].cpu(), ref["fused"]) if have_ref else (0., 0., 1.)
                    d16 = stats(out["decoder_features"][0][j].cpu(), ref["decoder16"]) if have_ref else (0., 0., 1.)
                    d8 = stats(out["decoder_features"][1][j].cpu(), ref["decoder8"]) if have_ref else (0., 0., 1.)
                    d4 = stats(out["decoder_features"][2][j].cpu(), ref["decoder4"]) if have_ref else (0., 0., 1.)
                    prob_mae = float(np.mean(np.abs(prob_np - ref["prob"]))) if have_ref else 0.
                    logit_ref = torch.as_tensor(ref["logit"], device=device).float() if have_ref else None
                    prob_ref = torch.as_tensor(ref["prob"], device=device).float() if have_ref else None
                    logit_cur = logit.float()
                    prob_cur = probs[j].float()
                    logit_mae = float(torch.mean(torch.abs(logit_cur - logit_ref)).cpu()) if have_ref else 0.
                    if have_ref:
                        logit_delta_l2 = float(torch.linalg.vector_norm(logit_cur - logit_ref).cpu())
                        logit_delta_rel_l2 = float((torch.linalg.vector_norm(logit_cur - logit_ref) / torch.linalg.vector_norm(logit_ref).clamp_min(1e-12)).cpu())
                        prob_delta_l2 = float(torch.linalg.vector_norm(prob_cur - prob_ref).cpu())
                        prob_delta_rel_l2 = float((torch.linalg.vector_norm(prob_cur - prob_ref) / torch.linalg.vector_norm(prob_ref).clamp_min(1e-12)).cpu())
                        t_assoc_delta_l2 = float(torch.linalg.vector_norm(out["t_assoc"][j].float().cpu() - ref["t_assoc"]).item())
                        fused_delta_l2 = float(torch.linalg.vector_norm(out["fused_regions"][j].float().cpu() - ref["fused"]).item())
                    else:
                        logit_delta_l2 = logit_delta_rel_l2 = prob_delta_l2 = prob_delta_rel_l2 = 0.
                        t_assoc_delta_l2 = fused_delta_l2 = 0.
                    changed_frac = float(np.mean((prob_np > 0.5) != ref["pred"])) if have_ref else 0.
                    if have_ref:
                        ref_alpha = ref["alpha"]
                        cur_alpha = alpha[j].cpu()
                        max_l = max(ref_alpha.shape[-1], cur_alpha.shape[-1])
                        ref_alpha_cmp = F.pad(ref_alpha, (0, max_l - ref_alpha.shape[-1]))
                        cur_alpha_cmp = F.pad(cur_alpha, (0, max_l - cur_alpha.shape[-1]))
                        alpha_delta_mae = stats(cur_alpha_cmp, ref_alpha_cmp)[0]
                        logits_delta_mae = valid_logit_mae(text_logits[j].cpu(), ref["text_logits"])
                    else:
                        alpha_delta_mae = logits_delta_mae = 0.0

                    pred_np = (prob_np[0] > 0.5)
                    pred_cc = count_cc(pred_np)
                    comp_np = comps_list[j].cpu().numpy().astype(bool)
                    comp_recalls = [float((pred_np & c).sum() / max(1, c.sum())) for c in comp_np]
                    mean_comp_recall = float(np.mean(comp_recalls)) if comp_recalls else float("nan")

                    row = {"condition": condition, "image": name, "text_original": texts[j], "text_used": cond_texts[j],
                           "text_changed": texts[j].strip().casefold() != cond_texts[j].strip().casefold(),
                           "token_ids_changed": valid_ids != reference.get(name, {}).get("valid_ids", valid_ids) if condition != "correct" else False,
                           "valid_token_count": len(valid_ids), "seg_dice": float(dice[j].cpu()), "seg_iou": float(iou[j].cpu()), "seg_loss": float(seg_loss_per_image[j].cpu()),
                           "L_match": float(mi["L_match"]), "N_gt": int(mi["N"]),
                           "gt_cc_count": int(comp_np.shape[0]), "pred_cc_count": pred_cc,
                           "pred_minus_gt_cc": pred_cc - int(comp_np.shape[0]),
                           "mean_component_recall": mean_comp_recall,
                           "assoc_cos_matched": assoc_matched,
                           "assoc_cos_supported": att["assoc_cos_supported"], "assoc_cos_orphan": att["assoc_cos_orphan"],
                           "n_orphan": att["n_orphan"],
                           "att_correct_mass": att["att_correct_mass"], "att_wrong_mass": att["att_wrong_mass"],
                           "att_null_mass": att["att_null_mass"], "att_argmax_unique_frac": att["att_argmax_unique_frac"],
                           "att_orphan_null_mass": att["att_orphan_null_mass"], "att_orphan_wrong_mass": att["att_orphan_wrong_mass"],
                           "att_unmatched_null_mass": att["att_unmatched_null_mass"], "att_unmatched_loc_mass": att["att_unmatched_loc_mass"],
                           "text_count_value": int(out["count_val"][j].item()),
                           "fuse_T_over_R_contrib": float(out["diag_fuse_T_over_R"][j].cpu()),
                           "text_attention_entropy_norm": float(out["diag_text_entropy_norm"][j].cpu()),
                           "t_assoc_query_pairwise_cos": float(out["diag_tassoc_query_cos"][j].cpu()),
                           "text_token_MAE_vs_correct": text_mae, "text_token_relative_L2_vs_correct": text_rel, "text_token_cosine_vs_correct": text_cos,
                           "pooled_text_MAE_vs_correct": pooled_mae, "pooled_text_relative_L2_vs_correct": pooled_rel, "pooled_text_cosine_vs_correct": pooled_cos,
                           "t_assoc_MAE_vs_correct": ta_mae, "t_assoc_relative_L2_vs_correct": ta_rel, "t_assoc_cosine_vs_correct": ta_cos,
                           "delta_T_assoc_L2_norm_vs_correct": t_assoc_delta_l2,
                           "fused_region_MAE_vs_correct": fu_mae, "fused_region_relative_L2_vs_correct": fu_rel, "fused_region_cosine_vs_correct": fu_cos,
                           "delta_H_fuse_L2_norm_vs_correct": fused_delta_l2,
                           "decoder16_MAE_vs_correct": d16[0], "decoder16_relative_L2_vs_correct": d16[1], "decoder16_cosine_vs_correct": d16[2],
                           "decoder8_MAE_vs_correct": d8[0], "decoder8_relative_L2_vs_correct": d8[1], "decoder8_cosine_vs_correct": d8[2],
                           "decoder4_MAE_vs_correct": d4[0], "decoder4_relative_L2_vs_correct": d4[1], "decoder4_cosine_vs_correct": d4[2],
                           "text_attention_logits_MAE_vs_correct_valid_only": logits_delta_mae,
                           "text_attention_weights_MAE_vs_correct_padded": alpha_delta_mae,
                           "region_attention_logits_MAE_vs_correct": stats(region_logits[j].cpu(), ref["region_logits"])[0] if have_ref else 0.,
                           "seg_probability_MAE_vs_correct": prob_mae, "seg_logit_MAE_vs_correct": logit_mae,
                           "delta_S_seg_logit_L2_norm_vs_correct": logit_delta_l2,
                           "delta_S_seg_logit_relative_L2_vs_correct": logit_delta_rel_l2,
                           "delta_seg_probability_L2_norm_vs_correct": prob_delta_l2,
                           "delta_seg_probability_relative_L2_vs_correct": prob_delta_rel_l2,
                           "predicted_pixel_change_fraction_vs_correct": changed_frac,
                           "mean_normalized_text_attention_entropy_raw": float(ent_q[j].mean().cpu()),
                           "min_normalized_text_attention_entropy_raw": float(ent_q[j].min().cpu()),
                           "max_normalized_text_attention_entropy_raw": float(ent_q[j].max().cpu()),
                           "mean_normalized_spatial_attention_entropy_raw": float(spatial_norm_ent[j].mean().cpu()),
                           "t_assoc_used_in_fuse": True, "fused_regions_passed_to_decoder16_8_4": True}
                    rows.append(row)
                    content_positions = torch.where(out["content_mask"][j])[0].tolist()
                    all_ids_j = ids[j].detach().cpu().tolist()
                    all_tokens_j = model.tokenizer.convert_ids_to_tokens(all_ids_j)
                    for qidx in range(alpha.shape[1]):
                        alpha_rows.append({"condition": condition, "image": name, "query_id": qidx + 1,
                                           "token_index_original": -1, "token_id": -1, "token": "[NULL]",
                                           "attention_logit_pre_softmax": float("nan"),
                                           "attention_weight": float(out["alpha_null"][j, qidx].cpu()),
                                           "is_cls_sep_pad": False})
                        for tidx in content_positions:
                            alpha_rows.append({"condition": condition, "image": name, "query_id": qidx + 1,
                                               "token_index_original": int(tidx), "token_id": int(all_ids_j[tidx]), "token": all_tokens_j[tidx],
                                               "attention_logit_pre_softmax": float(text_logits[j, qidx, tidx].cpu()),
                                               "attention_weight": float(alpha[j, qidx, tidx].cpu()),
                                               "is_cls_sep_pad": False})
                    for qidx in range(region_logits.shape[1]):
                        for pos in range(region_logits.shape[2]):
                            attn_summary_rows.append({"condition": condition, "image": name, "query_id": qidx + 1, "spatial_position": pos,
                                                      "spatial_logit_pre_softmax": float(region_logits[j, qidx, pos].cpu()),
                                                      "spatial_weight_raw": float(spatial_alpha[j, qidx, pos].cpu()),
                                                      "normalized_spatial_entropy": float(spatial_norm_ent[j, qidx].cpu())})

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "text_perturbation_per_image.csv", index=False)
    pd.DataFrame(alpha_rows).to_csv(out_dir / "text_association_weights.csv", index=False)
    pd.DataFrame(token_rows).to_csv(out_dir / "text_perturbation_token_audit.csv", index=False)
    pd.DataFrame(attn_summary_rows).to_csv(out_dir / "raw_spatial_attention_audit.csv", index=False)
    summary = df.groupby("condition").agg(
        n_images=("image", "count"), mean_seg_dice=("seg_dice", "mean"), std_seg_dice=("seg_dice", "std"),
        mean_seg_iou=("seg_iou", "mean"), std_seg_iou=("seg_iou", "std"), mean_seg_loss=("seg_loss", "mean"),
        mean_L_match=("L_match", "mean"), n_texts_changed=("text_changed", "sum"), n_token_ids_changed=("token_ids_changed", "sum"),
        mean_pred_cc_count=("pred_cc_count", "mean"), mean_gt_cc_count=("gt_cc_count", "mean"),
        mean_pred_minus_gt_cc=("pred_minus_gt_cc", "mean"), mean_component_recall=("mean_component_recall", "mean"),
        mean_assoc_cos_matched=("assoc_cos_matched", "mean"),
        mean_assoc_cos_supported=("assoc_cos_supported", "mean"), mean_assoc_cos_orphan=("assoc_cos_orphan", "mean"),
        n_orphan_queries=("n_orphan", "sum"),
        mean_att_correct_mass=("att_correct_mass", "mean"), mean_att_wrong_mass=("att_wrong_mass", "mean"),
        mean_att_null_mass=("att_null_mass", "mean"),
        mean_att_orphan_null_mass=("att_orphan_null_mass", "mean"), mean_att_orphan_wrong_mass=("att_orphan_wrong_mass", "mean"),
        mean_att_unmatched_null_mass=("att_unmatched_null_mass", "mean"), mean_att_unmatched_loc_mass=("att_unmatched_loc_mass", "mean"),
        mean_att_argmax_unique_frac=("att_argmax_unique_frac", "mean"),
        mean_text_count_value=("text_count_value", "mean"),
        mean_fuse_T_over_R_contrib=("fuse_T_over_R_contrib", "mean"),
        mean_text_attention_entropy_norm=("text_attention_entropy_norm", "mean"),
        mean_t_assoc_query_pairwise_cos=("t_assoc_query_pairwise_cos", "mean"),
        mean_text_token_relative_L2=("text_token_relative_L2_vs_correct", "mean"), mean_text_token_cosine=("text_token_cosine_vs_correct", "mean"),
        mean_pooled_text_relative_L2=("pooled_text_relative_L2_vs_correct", "mean"), mean_pooled_text_cosine=("pooled_text_cosine_vs_correct", "mean"),
        mean_t_assoc_relative_L2=("t_assoc_relative_L2_vs_correct", "mean"), mean_t_assoc_cosine=("t_assoc_cosine_vs_correct", "mean"),
        mean_delta_T_assoc_L2_norm=("delta_T_assoc_L2_norm_vs_correct", "mean"),
        mean_fused_region_relative_L2=("fused_region_relative_L2_vs_correct", "mean"), mean_fused_region_cosine=("fused_region_cosine_vs_correct", "mean"),
        mean_delta_H_fuse_L2_norm=("delta_H_fuse_L2_norm_vs_correct", "mean"),
        mean_decoder16_relative_L2=("decoder16_relative_L2_vs_correct", "mean"), mean_decoder8_relative_L2=("decoder8_relative_L2_vs_correct", "mean"),
        mean_decoder4_relative_L2=("decoder4_relative_L2_vs_correct", "mean"),
        mean_text_attention_logits_MAE_valid_only=("text_attention_logits_MAE_vs_correct_valid_only", "mean"),
        mean_text_attention_weights_MAE=("text_attention_weights_MAE_vs_correct_padded", "mean"),
        mean_region_attention_logits_MAE=("region_attention_logits_MAE_vs_correct", "mean"),
        mean_seg_probability_MAE=("seg_probability_MAE_vs_correct", "mean"), mean_seg_logit_MAE=("seg_logit_MAE_vs_correct", "mean"),
        mean_delta_S_seg_logit_L2_norm=("delta_S_seg_logit_L2_norm_vs_correct", "mean"),
        mean_delta_S_seg_logit_relative_L2=("delta_S_seg_logit_relative_L2_vs_correct", "mean"),
        mean_delta_seg_probability_L2_norm=("delta_seg_probability_L2_norm_vs_correct", "mean"),
        mean_delta_seg_probability_relative_L2=("delta_seg_probability_relative_L2_vs_correct", "mean"),
        mean_predicted_pixel_change_fraction=("predicted_pixel_change_fraction_vs_correct", "mean"),
        mean_normalized_text_attention_entropy_raw=("mean_normalized_text_attention_entropy_raw", "mean"),
        mean_normalized_spatial_attention_entropy_raw=("mean_normalized_spatial_attention_entropy_raw", "mean")).reset_index()
    order = {c: i for i, c in enumerate(conditions)}
    summary = summary.sort_values("condition", key=lambda s: s.map(order)).reset_index(drop=True)
    correct_row = summary[summary["condition"] == "correct"]
    if not correct_row.empty:
        summary["delta_dice_vs_correct"] = summary["mean_seg_dice"] - float(correct_row.iloc[0]["mean_seg_dice"])
        summary["delta_iou_vs_correct"] = summary["mean_seg_iou"] - float(correct_row.iloc[0]["mean_seg_iou"])
        summary["delta_assoc_cos_vs_correct"] = summary["mean_assoc_cos_matched"] - float(correct_row.iloc[0]["mean_assoc_cos_matched"])
    summary.to_csv(out_dir / "text_perturbation_summary.csv", index=False)
    paired = _paired_stats(df, conditions, out_dir / "text_perturbation_paired_stats.csv", metric="seg_dice")
    paired_assoc = _paired_stats(df, conditions, out_dir / "text_perturbation_paired_stats_assoc_cos.csv", metric="assoc_cos_matched")
    stage_cols = ["condition", "image", "t_assoc_relative_L2_vs_correct", "delta_T_assoc_L2_norm_vs_correct",
                  "fused_region_relative_L2_vs_correct", "delta_H_fuse_L2_norm_vs_correct",
                  "decoder16_relative_L2_vs_correct", "decoder8_relative_L2_vs_correct",
                  "decoder4_relative_L2_vs_correct", "delta_S_seg_logit_L2_norm_vs_correct",
                  "delta_S_seg_logit_relative_L2_vs_correct", "seg_probability_MAE_vs_correct",
                  "delta_seg_probability_L2_norm_vs_correct", "delta_seg_probability_relative_L2_vs_correct",
                  "predicted_pixel_change_fraction_vs_correct"]
    df[stage_cols].to_csv(out_dir / "text_path_stagewise_deltas.csv", index=False)
    with open(out_dir / "text_preprocessing_config.json", "w", encoding="utf-8") as f:
        json.dump({"text_cleanup": "replace common punctuation with spaces; collapse whitespace",
                   "region_attention_keys": "location-phrase tokens ([vertical] left|right lung) + NULL slot",
                   "global_tokens": "count, bilateral, infection, and ... (never enter per-region attention)",
                   "negative_controls": ["count_changed", "count_removed"],
                   "positive_conditions": ["location_removed", "one_side_removed", "left_side_removed", "mismatched"]},
                  f, ensure_ascii=False, indent=2)
    key_cols = ["condition", "n_images", "mean_seg_dice", "delta_dice_vs_correct",
                "mean_assoc_cos_matched", "mean_assoc_cos_supported", "mean_assoc_cos_orphan", "n_orphan_queries",
                "mean_att_correct_mass", "mean_att_wrong_mass", "mean_att_orphan_null_mass", "mean_att_orphan_wrong_mass",
                "mean_att_unmatched_null_mass",
                "mean_t_assoc_relative_L2", "mean_fused_region_relative_L2",
                "mean_decoder16_relative_L2", "mean_seg_probability_MAE"]
    print("Text perturbation audit summary (key columns; full table in text_perturbation_summary.csv):\n",
          summary[[c for c in key_cols if c in summary.columns]].to_string(index=False))
    print("Paired bootstrap Dice change vs correct:\n", paired.to_string(index=False))
    print("Paired bootstrap association-score change vs correct:\n", paired_assoc.to_string(index=False))
    print("NOTE: audit is evaluation-only; same weights and same images are used for all conditions.")
    print("NOTE: L_match is identical across conditions BY DESIGN (region masks are text-independent).")
    print("NOTE: count_* conditions are NEGATIVE CONTROLS (count is not an attention key).")
    return summary


def _gradient_norm_by_group(loss_component, named_parameter_groups):
    flat_params = []
    spans = {}
    for group_name, params in named_parameter_groups.items():
        params = [p for p in params if p.requires_grad]
        begin = len(flat_params)
        flat_params.extend(params)
        spans[group_name] = (begin, len(flat_params))
    if not flat_params or not loss_component.requires_grad:
        return {name: 0.0 for name in named_parameter_groups}
    grads = torch.autograd.grad(loss_component, flat_params, retain_graph=True,
                                allow_unused=True, create_graph=False)
    result = {}
    for name, (begin, end) in spans.items():
        sq = 0.0
        for g in grads[begin:end]:
            if g is not None:
                sq += float(g.detach().float().pow(2).sum().cpu())
        result[name] = math.sqrt(sq)
    return result


def _text_transfer_gradient_audit(model, comps):
    groups = {
        'text_adapter': list(model.text_project.parameters()),
        'assoc_v': list(model.assoc_v.parameters()),
        'null_slot': [model.null_key, model.null_val],
        'attn_log_scale': [model.attn_log_scale],
        'geom_ctx': (list(model.norm_ctx.parameters())
                     + (list(model.geom_mlp.parameters()) if model.geom_mlp is not None else [])
                     + ([model.act_on, model.act_off] if model.use_activity else [])),
        'fusion': list(model.fuse.parameters()) + list(model.norm_T.parameters()),
        'align_r': list(model.align_r.parameters()),
        'align_t': list(model.align_t.parameters()),
        'region_head': list(model.head.parameters()),
        'decoder16_cross_attn': list(model.decoder16.guide_layer.cross_attn.parameters()),
        'decoder8_cross_attn': list(model.decoder8.guide_layer.cross_attn.parameters()),
        'decoder4_cross_attn': list(model.decoder4.guide_layer.cross_attn.parameters()),
    }
    for tag, dec in (("16", model.decoder16), ("8", model.decoder8), ("4", model.decoder4)):
        if dec.film is not None:
            groups[f'film{tag}'] = list(dec.film.parameters())
    result = {}
    for loss_name, component in comps.items():
        norms = _gradient_norm_by_group(component, groups)
        for group_name, value in norms.items():
            result[f'{loss_name}_grad_{group_name}'] = value
    result['text_encoder_trainable_parameters'] = int(sum(p.numel() for p in model.text_encoder.parameters() if p.requires_grad))
    result['text_encoder_frozen'] = bool(all(not p.requires_grad for p in model.text_encoder.parameters()))
    return result


def _film_gate(dec):
    return float(dec.film.gate.detach().cpu()) if dec.film is not None else float("nan")


def main():
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, "| seed:", args.seed, "| output:", OUTPUT_DIR)
    print("Deterministic mode: ON | cuBLAS workspace:",
          os.environ.get("CUBLAS_WORKSPACE_CONFIG"), "| torch threads:", torch.get_num_threads())
    print(f"K={args.num_queries} | lambda_ra={args.lambda_ra} | lambda_att={args.lambda_att} | "
          f"ra_token_weight={args.ra_token_weight} | ra_null_weight={args.ra_null_weight} | "
          f"ra_unm_null_weight={args.ra_unm_null_weight} | use_activity_embed={args.use_activity_embed}")
    if int(args.num_queries) < 7:
        print(f"[info] K={args.num_queries}: images with N>K keep (N-K) GT components unmatched by design "
              "(they are ignored in L_match, not penalised).")

    global TEXT_LOOKUP
    TEXT_LOOKUP = load_sample_text_lookup()
    train_all, valid_all = build_mosmed_split()
    if bool(args.auto_orientation):
        resolve_orientation(train_all + valid_all, OUTPUT_DIR)
    elif str(args.patient_left_on_image_side).lower() == "auto":
        args.patient_left_on_image_side = "right"
        print("[info] auto_orientation=false and patient_left_on_image_side='auto' -> using 'right'.")
    render_orientation_check(train_all + valid_all, args, VIS_DIR / "orientation_check.png")

    per_count = int(args.pilot_train_per_count)
    train_recs, train_component_counts = select_balanced_train_records(train_all, args, per_count=per_count)
    valid_recs, valid_component_counts = select_multi_region_records(
        valid_all, args.pilot_valid_multi_region_samples, args, split_name="validation")
    if not train_recs or not valid_recs:
        raise RuntimeError("Could not select multi-region images from both train and validation splits. "
                           "Check mask paths, component threshold, and split CSVs.")
    print(f"Pilot selected -> balanced train: {len(train_recs)} ({per_count} each N=3,4,5,6) | "
          f"held-out validation multi-region: {len(valid_recs)} | total: {len(train_recs)+len(valid_recs)}")
    with open(OUTPUT_DIR / "selected_multi_region_samples.json", "w") as f:
        json.dump({"train_count": len(train_recs),
                   "train_target_distribution": {f"N={n}": per_count for n in (3, 4, 5, 6)},
                   "validation_count": len(valid_recs),
                   "total_count": len(train_recs) + len(valid_recs),
                   "train_component_counts": train_component_counts,
                   "validation_component_counts": valid_component_counts}, f, indent=2)

    ds_train = MosMedDataset(train_recs, mode='train', image_size=args.image_size)
    ds_valid = MosMedDataset(valid_recs, mode='valid', image_size=args.image_size)

    ds_probe = MosMedDataset(train_recs, mode='valid', image_size=args.image_size)
    n_bad = 0
    for idx in range(len(ds_probe)):
        _, _, comps_probe, name_probe, _ = ds_probe[idx]
        if int(comps_probe.shape[0]) != int(train_component_counts[name_probe]):
            n_bad += 1
    print(f"Selection-vs-Dataset component-count mismatches (train): {n_bad}/{len(ds_probe)}")
    if n_bad > 0:
        raise RuntimeError("Selection N differs from Dataset N; the N-balanced design would be violated.")

    train_pf = [{"name": str(r[2]), "text": str(TEXT_LOOKUP.get(_norm_image_key(r[2]), ""))} for r in train_recs]
    valid_pf = [{"name": str(r[2]), "text": str(TEXT_LOOKUP.get(_norm_image_key(r[2]), ""))} for r in valid_recs]
    preflight_train = validate_text_perturbations(train_pf, "Train")
    preflight_valid = validate_text_perturbations(valid_pf, "Validation")

    eqN = [float(caption_count(r["text"]) == train_component_counts[r["name"]]) for r in train_pf]
    print(f"Caption count == number of GT components: {np.mean(eqN) * 100:.1f}% of train captions")
    with_both = sum(1 for r in train_pf if caption_sides(r["text"]) == {"left", "right"})
    print(f"Captions with BOTH a left and a right phrase: {with_both}/{len(train_pf)}")

    _hn_probe = random.Random(0)
    hn_all = [make_location_hard_negatives(r["text"], _hn_probe, max_out=int(args.hn_per_image)) for r in train_pf]
    print(f"Location hard-negative coverage: {sum(len(h) > 0 for h in hn_all)}/{len(hn_all)} captions have >=1, "
          f"mean per caption = {np.mean([len(h) for h in hn_all]):.2f}")
    print("Example hard negatives (tag = which side the negative LACKS):")
    for r, h in list(zip(train_pf, hn_all))[:3]:
        print("  ORIGINAL:", r["text"])
        for t, lacks in h:
            print(f"     NEG(lacks {('left', 'right')[lacks - 1]:5s}): {t}")
    with open(OUTPUT_DIR / "text_perturbation_preflight.json", "w", encoding="utf-8") as f:
        json.dump({"Train": preflight_train, "Validation": preflight_valid,
                   "caption_count_equals_N_fraction": float(np.mean(eqN)),
                   "hard_negatives_per_caption_mean": float(np.mean([len(h) for h in hn_all]))},
                  f, ensure_ascii=False, indent=2)
    print("Text perturbation preflight passed:", preflight_train, preflight_valid)

    train_loader_generator = make_loader_generator(int(args.seed) + 1001)
    valid_loader_generator = make_loader_generator(int(args.seed) + 2001)
    train_eval_loader_generator = make_loader_generator(int(args.seed) + 3001)

    dl_train = DataLoader(
        ds_train, batch_size=args.train_batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        worker_init_fn=seed_worker, generator=train_loader_generator,
    )
    dl_valid = DataLoader(
        ds_valid, batch_size=args.valid_batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        worker_init_fn=seed_worker, generator=valid_loader_generator,
    )

    model = RegionQueryPilot(args.vision_type, args.project_dim, args).to(device)
    assert all(not p.requires_grad for p in model.text_encoder.parameters()), "Text encoder must be frozen for this experiment."

    _ids, _msk = model.tokenize([train_pf[0]["text"]], device)
    _lex = model.parse_lexicon(_ids, _msk)
    _toks = model.tokenizer.convert_ids_to_tokens(_ids[0].tolist())
    print("Phrase parser check:", [(t, int(s)) for t, s in zip(_toks, _lex["phrase_side"][0].tolist())],
          "| count_val =", int(_lex["count_val"][0]))
    if int(_lex["phrase_mask"].sum()) == 0:
        raise RuntimeError("Phrase parser found NO location phrase in the first caption; check the tokenizer/lexicon.")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=float(args.lr))
    print(f"Frozen text encoder: {sum(p.numel() for p in model.text_encoder.parameters()):,} parameters; "
          f"trainable parameters passed to optimizer: {sum(p.numel() for p in trainable_params):,}")
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.cosine_T_max, eta_min=1e-6)
    steps_per_epoch = math.ceil(len(ds_train) / int(args.train_batch_size))
    print(f"optimizer steps per epoch: {steps_per_epoch} | total steps: {steps_per_epoch * int(args.epochs)}")

    with open(OUTPUT_DIR / "pilot_config.json", "w") as f:
        config_dump = dict(args)
        config_dump["text_encoder_frozen"] = True
        config_dump["deterministic_mode"] = True
        config_dump["cublas_workspace_config"] = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        config_dump["optimizer_steps_total"] = steps_per_epoch * int(args.epochs)
        json.dump(config_dump, f, indent=2, default=str)

    history = []
    best_train_l_total = float("inf")
    best_train_epoch = 0
    best_train_ckpt_path = OUTPUT_DIR / "best_model_by_train_L_total.pt"
    gradient_audit_rows = []
    hn_rng = random.Random(int(args.seed) + 777)
    t0 = time.time()
    for epoch in range(args.epochs):
        args.current_epoch = epoch + 1
        model.train()
        run = {"loss": 0.0, "L_match_total": 0.0, "L_seg": 0.0, "L_RA": 0.0, "L_att": 0.0,
               "fuse_T_over_R": 0.0, "text_entropy": 0.0, "tassoc_qcos": 0.0,
               "L_RA_seq": 0.0, "L_RA_tok": 0.0, "L_RA_null_unm": 0.0, "L_RA_null_other": 0.0}
        ra_ep = {k: 0 for k in RA_HIT_KEYS + RA_COUNT_KEYS}
        act_m_ep, act_u_ep = [], []
        n_batches = 0
        for images, gts, comps_list, names, texts in tqdm(dl_train, desc=f"epoch {epoch + 1}/{args.epochs}", leave=False):
            images = images.to(device)
            bs = int(images.shape[0])
            input_ids, attention_mask = model.tokenize(texts, device)
            outputs = model(images, input_ids, attention_mask)
            A, logits = outputs["attention"], outputs["mask_logits"]
            l_seg = DiceCELoss(sigmoid=True)(outputs["seg_logits"], gts.to(device).float())
            losses, rows_b, cols_b = [], [], []
            for i in range(bs):
                comps = comps_list[i].to(device)
                loss_i, info = region_match_loss(logits[i], comps, A[i], args)
                losses.append(loss_i)
                rows_b.append(info["rows"]); cols_b.append(info["cols"])
                run["L_match_total"] += info["L_match"]
                am_, au_ = act_split(outputs["act"][i].detach().cpu(), info["rows"])
                act_m_ep += am_; act_u_ep += au_
            l_match_total = torch.stack(losses).mean()

            tl = compute_text_losses(model, outputs, rows_b, cols_b, comps_list, texts, args, hn_rng, device)
            l_ra, l_att = tl["l_ra"], tl["l_att"]
            for k in ra_ep:
                ra_ep[k] += tl["ra_stats"][k]
            loss = (l_match_total + float(args.lambda_seg) * l_seg + float(args.lambda_ra) * l_ra
                    + float(args.lambda_att) * l_att)
            run["L_RA"] += float(l_ra.detach()) * bs
            for name, key in (("L_RA_seq", "L_seq"), ("L_RA_tok", "L_tok"),
                              ("L_RA_null_unm", "L_null_unm"), ("L_RA_null_other", "L_null_other")):
                run[name] += float(tl["ra_stats"][key]) * bs
            run["L_att"] += float(l_att.detach()) * bs
            run["L_seg"] += float(l_seg.detach()) * bs
            run["loss"] += float(loss.detach()) * bs
            run["fuse_T_over_R"] += float(outputs["diag_fuse_T_over_R"].mean()) * bs
            run["text_entropy"] += float(outputs["diag_text_entropy_norm"].mean()) * bs
            run["tassoc_qcos"] += float(outputs["diag_tassoc_query_cos"].mean()) * bs
            if n_batches == 0:
                grad_row = {"epoch": epoch + 1, "batch_images": bs,
                            "L_match_value": float(l_match_total.detach()), "L_seg_value": float(l_seg.detach()),
                            "L_RA_value": float(l_ra.detach()), "L_att_value": float(l_att.detach()),
                            **_text_transfer_gradient_audit(model, {"L_match": l_match_total, "L_seg": l_seg,
                                                                   "L_RA": l_ra, "L_att": l_att})}
                gradient_audit_rows.append(grad_row)

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            n_batches += 1
        lr_scheduler.step()

        val, _, _, _ = evaluate(model, dl_valid, device, args)
        n_tr = max(1, len(ds_train))
        cur_total = run["loss"] / n_tr
        if np.isfinite(cur_total) and cur_total < best_train_l_total:
            best_train_l_total = cur_total
            best_train_epoch = epoch + 1
            torch.save({"epoch": best_train_epoch, "best_train_L_total": best_train_l_total,
                        "selection_definition": "image-weighted mean of the full training objective",
                        "model_state_dict": model.state_dict()}, best_train_ckpt_path)

        train_act_m = float(np.mean(act_m_ep)) if act_m_ep else float("nan")
        train_act_u = float(np.mean(act_u_ep)) if act_u_ep else float("nan")
        row = {"epoch": epoch + 1, "train_L_total": cur_total,
               "train_L_match": run["L_match_total"] / n_tr, "train_L_seg": run["L_seg"] / n_tr,
               "train_L_RA": run["L_RA"] / n_tr, "train_L_att": run["L_att"] / n_tr,
               "train_L_RA_seq": run["L_RA_seq"] / n_tr, "train_L_RA_tok": run["L_RA_tok"] / n_tr,
               "train_L_RA_null_unm": run["L_RA_null_unm"] / n_tr,
               "train_L_RA_null_other": run["L_RA_null_other"] / n_tr,
               "train_ra_acc_cross": ra_ep["hits_cross"] / max(1, ra_ep["n_cross"]),
               "train_ra_acc_batch": ra_ep["hits_batch"] / max(1, ra_ep["n_batch"]),
               "train_ra_acc_hard": ra_ep["hits_hard"] / max(1, ra_ep["n_hard"]),
               "train_ra_acc_tok": ra_ep["hits_tok"] / max(1, ra_ep["n_tok"]),
               "train_ra_acc_null": ra_ep["hits_null"] / max(1, ra_ep["n_null"]),
               "train_ra_acc_unm": ra_ep["hits_unm"] / max(1, ra_ep["n_unm"]),
               "train_n_unm_anchors": ra_ep["n_unm"],
               "train_act_matched": train_act_m, "train_act_unmatched": train_act_u,
               "train_positive_pairs": ra_ep["positive_pairs"], "train_neg_cross": ra_ep["neg_cross"],
               "train_neg_batch": ra_ep["neg_batch"], "train_neg_hard": ra_ep["neg_hard"],
               "train_tok_anchors": ra_ep["tok_anchors"], "train_null_anchors": ra_ep["null_anchors"],
               "train_anchors_skipped": ra_ep["anchors_skipped"],
               "train_fuse_T_over_R_contrib": run["fuse_T_over_R"] / n_tr,
               "train_text_attention_entropy_norm": run["text_entropy"] / n_tr,
               "train_t_assoc_query_pairwise_cos": run["tassoc_qcos"] / n_tr,
               "attn_logit_scale": float(model.attn_log_scale.detach().clamp(max=math.log(20.0)).exp().cpu()),
               "decoder16_gate": float(model.decoder16.guide_layer.scale.detach().cpu()),
               "decoder8_gate": float(model.decoder8.guide_layer.scale.detach().cpu()),
               "decoder4_gate": float(model.decoder4.guide_layer.scale.detach().cpu()),
               "film16_gate": _film_gate(model.decoder16), "film8_gate": _film_gate(model.decoder8),
               "film4_gate": _film_gate(model.decoder4),
               **{f"val_{k}": v for k, v in val.items()}}
        history.append(row)
        print(f"[{epoch + 1:03d}/{args.epochs}] L_total={cur_total:.3f} match={row['train_L_match']:.3f} "
              f"seg={row['train_L_seg']:.3f} RA={row['train_L_RA']:.3f} "
              f"[seq/tok/nullUnm/nullOther={row['train_L_RA_seq']:.3f}/{row['train_L_RA_tok']:.3f}/"
              f"{row['train_L_RA_null_unm']:.3f}/{row['train_L_RA_null_other']:.3f}] "
              f"(VAL RA={val['L_RA']:.3f} [{val['L_RA_seq']:.3f}/{val['L_RA_tok']:.3f}/{val['L_RA_null_unm']:.3f}/{val['L_RA_null_other']:.3f}]) "
              f"att(w={args.lambda_att})={row['train_L_att']:.3f} | "
              f"train RAacc cross/batch/hard/tok/null/unm={row['train_ra_acc_cross']:.2f}/{row['train_ra_acc_batch']:.2f}/"
              f"{row['train_ra_acc_hard']:.2f}/{row['train_ra_acc_tok']:.2f}/{row['train_ra_acc_null']:.2f}/{row['train_ra_acc_unm']:.2f} "
              f"(n_unm={ra_ep['n_unm']}) act matched/unmatched={train_act_m:.2f}/{train_act_u:.2f} | "
              f"VAL dice={val['matched_dice']:.3f} rec={val['gt_recall@0.5']:.3f} | "
              f"VAL att correct/wrong/null/unif={val['att_correct_mass']:.2f}/{val['att_wrong_mass']:.2f}/{val['att_null_mass']:.2f}/{val['att_uniform_baseline']:.2f} "
              f"unmatched NULL/loc={val['att_unmatched_null_mass']:.2f}/{val['att_unmatched_loc_mass']:.2f} "
              f"sidemap(R|L)={val['side_map_acc_conv_right']:.2f}|{val['side_map_acc_conv_left']:.2f} "
              f"pred_right_frac={val['att_pred_right_phrase_frac']:.2f} | "
              f"VAL RAacc c/b/h/t/n/u={val['ra_acc_cross']:.2f}/{val['ra_acc_batch']:.2f}/{val['ra_acc_hard']:.2f}/{val['ra_acc_tok']:.2f}/{val['ra_acc_null']:.2f}/{val['ra_acc_unm']:.2f} "
              f"VAL act m/u={val['act_matched']:.2f}/{val['act_unmatched']:.2f} | "
              f"fuseT/R={row['train_fuse_T_over_R_contrib']:.2f} ent={row['train_text_attention_entropy_norm']:.2f} "
              f"T_qcos={row['train_t_assoc_query_pairwise_cos']:.2f} scale={row['attn_logit_scale']:.1f}")
        pd.DataFrame(history).to_csv(OUTPUT_DIR / "pilot_history.csv", index=False)
        pd.DataFrame(gradient_audit_rows).to_csv(OUTPUT_DIR / "text_transfer_gradient_audit.csv", index=False)

    print(f"training time: {(time.time() - t0) / 60:.1f} min")
    torch.save(model.state_dict(), OUTPUT_DIR / "last_epoch_model.pt")
    if not best_train_ckpt_path.exists():
        raise RuntimeError("No checkpoint was saved; inspect train_L_total.")

    payload = torch.load(best_train_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    ds_train_eval = MosMedDataset(train_recs, mode='valid', image_size=args.image_size)
    dl_train_eval = DataLoader(
        ds_train_eval, batch_size=args.valid_batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        worker_init_fn=seed_worker, generator=train_eval_loader_generator,
    )
    train_eval, train_df, train_vis_store, train_match_df = evaluate(
        model, dl_train_eval, device, args, collect_vis=True, vis_limit_per_category=len(ds_train_eval))
    train_df.to_csv(OUTPUT_DIR / "pilot_train_per_image_diagnostics.csv", index=False)
    train_match_df.to_csv(OUTPUT_DIR / "pilot_train_per_match_iou_dice.csv", index=False)
    with open(OUTPUT_DIR / "pilot_train_summary.json", "w") as f:
        json.dump(train_eval, f, indent=2)

    train_vis_dir = VIS_DIR / "train_best_L_total_checkpoint"
    train_success_dir = train_vis_dir / "multi_region_success"
    train_failure_dir = train_vis_dir / "multi_region_failure"
    for d in (train_vis_dir, train_success_dir, train_failure_dir):
        d.mkdir(parents=True, exist_ok=True)
    train_scored = (train_match_df.groupby("image")["binary_dice"].mean().to_dict()
                    if not train_match_df.empty else {})
    train_n_fig = 0
    for category_recs in train_vis_store.values():
        for rec in category_recs:
            score = train_scored.get(rec["name"], float("nan"))
            safe = re.sub(r"[^A-Za-z0-9_.-]", "_", rec["name"])
            dest = train_success_dir if np.isfinite(score) and score >= 0.5 else train_failure_dir
            render_sample(rec, args, dest / f"{safe}__meanDice_{score:.3f}.png")
            render_sample(rec, args, train_vis_dir / f"{safe}__meanDice_{score:.3f}.png")
            render_query_mask_text_attention(
                rec, args, train_vis_dir / "query_mask_text_attention" / f"{safe}__meanDice_{score:.3f}__Qmask_text.png")
            train_n_fig += 1

    train_pert_dir = OUTPUT_DIR / "text_perturbation_train_best_L_total"
    train_pert_dir.mkdir(parents=True, exist_ok=True)
    train_pert_summary = evaluate_text_perturbations(model, dl_train_eval, device, args, train_pert_dir)

    val, df, vis_store, match_df = evaluate(
        model, dl_valid, device, args, collect_vis=True, vis_limit_per_category=len(ds_valid))
    df.to_csv(OUTPUT_DIR / "pilot_per_image_diagnostics.csv", index=False)
    match_df.to_csv(OUTPUT_DIR / "pilot_per_match_iou_dice.csv", index=False)
    val_pert_dir = OUTPUT_DIR / "text_perturbation_validation_same_train_L_total_checkpoint"
    val_pert_dir.mkdir(parents=True, exist_ok=True)
    val_pert_summary = evaluate_text_perturbations(model, dl_valid, device, args, val_pert_dir)
    torch.save(model.state_dict(), OUTPUT_DIR / "region_query_pilot.pt")

    with open(OUTPUT_DIR / "best_checkpoint_info.json", "w") as f:
        json.dump({
            "selection": "minimum image-weighted epoch training L_total",
            "best_train_epoch": best_train_epoch, "best_train_L_total": best_train_l_total,
            "objective": "L_match + lam_seg*L_seg + lam_ra*(beta*L_tok + gamma*(w_unm*L_null_unm + L_null_other)) + lam_att*L_att (L_seq diagnostic only, excluded from loss; lam_att default 0)",
            "orientation": {"rot90_k": int(args.rot90_k) % 4, "pil_reverse_indexing": bool(args.pil_reverse_indexing),
                            "patient_left_on_image_side": str(args.patient_left_on_image_side)},
            "same_checkpoint_used_for_train_and_validation": True,
            "checkpoint": str(best_train_ckpt_path),
            "train_summary": train_eval, "validation_summary": val,
            "train_text_perturbation_summary": train_pert_summary.to_dict(orient="records"),
            "validation_text_perturbation_summary": val_pert_summary.to_dict(orient="records"),
        }, f, indent=2)

    success_dir = VIS_DIR / "multi_region_success"
    failure_dir = VIS_DIR / "multi_region_failure"
    success_dir.mkdir(parents=True, exist_ok=True)
    failure_dir.mkdir(parents=True, exist_ok=True)
    val_qt_dir = VIS_DIR / "val_query_mask_text_attention"
    scored = match_df.groupby("image")["binary_dice"].mean().to_dict() if not match_df.empty else {}
    for category_recs in vis_store.values():
        for rec in category_recs:
            score = scored.get(rec["name"], float("nan"))
            safe = re.sub(r"[^A-Za-z0-9_.-]", "_", rec["name"])
            dest = success_dir if np.isfinite(score) and score >= 0.5 else failure_dir
            render_sample(rec, args, dest / f"{safe}__meanDice_{score:.3f}.png")
            render_query_mask_text_attention(rec, args, val_qt_dir / f"{safe}__meanDice_{score:.3f}__Qmask_text.png")
    with open(OUTPUT_DIR / "multi_region_pilot_summary.json", "w") as f:
        json.dump(val, f, indent=2)
    with open(OUTPUT_DIR / "pilot_final_summary.json", "w") as f:
        json.dump(val, f, indent=2)

    n_fig = 0
    for c, recs in vis_store.items():
        for rec in recs:
            safe = re.sub(r"[^A-Za-z0-9_.-]", "_", rec["name"])
            tag = f"N{c}" if c < 4 else "N4plus"
            render_sample(rec, args, VIS_DIR / f"{tag}__{safe}.png")
            n_fig += 1

    print("\n" + "=" * 90)
    print("MOSMED REGION-QUERY v11 (activity embedding + group-balanced NULL loss; token-level + NULL-supervised L_RA) SUMMARY")
    print("=" * 90)
    for k, v in val.items():
        print(f"{k:46s}: {v}")
    print(f"\nN distribution (GT components per image): {df['N'].value_counts().sort_index().to_dict()}")
    print(f"Validation figures saved: {n_fig}  ->  {VIS_DIR}")
    print(f"Best checkpoint epoch {best_train_epoch} (train L_total={best_train_l_total:.6f}); train figures: {train_n_fig}")
    print("KEY METRICS: att_correct_mass >> att_uniform_baseline and > att_wrong_mass; att_pred_right_phrase_frac ~ 0.5 "
          "(1.0 or 0.0 = collapse); side_map_acc_conv_right vs _left (one of them should be ~1, the other ~0; ~0.5 = no grounding); "
          "ra_acc_tok / ra_acc_cross; assoc_cos_supported >> assoc_cos_orphan; att_orphan_null_mass and "
          "att_unmatched_null_mass (higher = better); ra_acc_null / ra_acc_unm (NULL beats every location token); "
          "act_matched >> act_unmatched (unmatched masks must be (nearly) empty for the NULL decision to be learnable).")


if __name__ == "__main__":
    main()
