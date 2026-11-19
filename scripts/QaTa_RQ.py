import os
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
from transformers import AutoModel
from scipy.optimize import linear_sum_assignment
from skimage.measure import label
from tqdm.auto import tqdm
from monai.networks.blocks.unetr_block import UnetrUpBlock
from monai.networks.blocks.upsample import SubpixelUpsample
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.losses import DiceCELoss
from functools import partial
from PIL import Image
tqdm = partial(tqdm, disable=True)

from monai.transforms import (Compose, NormalizeIntensityd, RandZoomd,
                              Resized, ToTensord, LoadImaged, EnsureChannelFirstd)

import logging
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

# Experiment hyper-parameters and default paths live in configs/qata_rq.yaml
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
    ap = argparse.ArgumentParser(description="QaTa_RQ_clean.py -- region-query pilot.")
    ap.add_argument("--config", default="configs/qata_rq.yaml",
                    help="Experiment YAML (hyper-parameters + default paths).")
    ap.add_argument("--dataset_root", default=None, help="Override DATA.dataset_root from the config file.")
    ap.add_argument("--frames_dir", default=None, help="Override DATA.frames_dir from the config file.")
    ap.add_argument("--masks_dir", default=None, help="Override DATA.masks_dir from the config file.")
    ap.add_argument("--train_id_csv", default=None, help="Override DATA.train_id_csv from the config file.")
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
for _key in ("dataset_root", "frames_dir", "masks_dir", "train_id_csv", "output_dir", "bert_type", "vision_type",):
    _val = getattr(_CLI, _key)
    if _val is not None:
        args[_key] = _val

if not os.path.isdir(args.vision_type):
    print(f"[info] '{args.vision_type}' not found -> using 'facebook/convnext-tiny-224'")
    args.vision_type = "facebook/convnext-tiny-224"

OUTPUT_DIR = Path(f"{args.output_dir}_seed{args.seed}")
VIS_DIR = OUTPUT_DIR / "visualizations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VIS_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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

    return df.columns[0] if len(df.columns) else None


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


def build_qata_split():
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
            missing_frames.append(mask_path.name)
            continue
        paired_by_key[key] = (str(frame_path), str(mask_path), mask_path.name)

    train_keys = _read_split_keys(args.train_id_csv, "Train")

    unresolved = sorted(k for k in train_keys if k not in paired_by_key)
    if unresolved:
        print(f"[warning] Train: {len(unresolved)} Train_ID keys did not match "
              f"frame/mask files; examples: {unresolved[:8]}")

    train_records = [paired_by_key[k] for k in sorted(train_keys) if k in paired_by_key]
    valid_records = [paired_by_key[k] for k in sorted(paired_by_key) if k not in train_keys]

    if not train_records:
        raise RuntimeError(
            "No Train samples could be matched. "
            "Inspect the Train_ID CSV filename column and actual frame/mask names.")
    if not valid_records:
        raise RuntimeError(
            "Validation set is empty (every Train Set mask is listed in Train_ID).")

    train_ids = {r[2] for r in train_records}
    valid_ids = {r[2] for r in valid_records}
    overlap = train_ids & valid_ids
    if overlap:
        raise RuntimeError(
            f"Train/validation image overlap detected ({len(overlap)} samples), "
            f"e.g. {sorted(overlap)[:10]}. Check the Train_ID CSV.")

    print(f"Train paired samples: {len(train_records):,}")
    print(f"Validation paired samples: {len(valid_records):,}")
    print(f"Indexed frames: {len(frame_files):,} | masks: {len(mask_files):,} | "
          f"paired: {len(paired_by_key):,} | masks without frame: {len(missing_frames):,}")
    print(f"Final split -> train: {len(train_records):,} | validation: {len(valid_records):,} "
          "| test excluded")
    return train_records, valid_records


def count_gt_components_from_file(mask_path, image_size, min_area, connectivity):
    with Image.open(mask_path) as im:
        arr = np.array(im.convert("L").resize((int(image_size[1]), int(image_size[0])), Image.Resampling.NEAREST))
    binary = (arr > 0).astype(np.uint8)
    lab = label(binary, connectivity=connectivity)
    counts = np.bincount(lab.ravel())
    return int(sum(counts[i] >= min_area for i in range(1, int(lab.max()) + 1)))


def select_multi_region_records(records, limit, cfg, split_name="validation", exact_counts=None):
    buckets = {}
    multi_count = 0
    for rec in records:
        n_comp = count_gt_components_from_file(
            rec[1], cfg.image_size, cfg.min_component_area, cfg.connectivity
        )
        if n_comp >= 2:
            multi_count += 1
        buckets.setdefault(n_comp, []).append((rec, n_comp))

    rng = random.Random(int(cfg.data_seed) + (0 if split_name == "train" else 100003))
    for n in buckets:
        rng.shuffle(buckets[n])

    if exact_counts is not None:
        missing = {int(n): int(k) - len(buckets.get(int(n), []))
                   for n, k in exact_counts.items()
                   if len(buckets.get(int(n), [])) < int(k)}
        if missing:
            available = {int(n): len(buckets.get(int(n), [])) for n in exact_counts}
            raise RuntimeError(
                f"Cannot create the requested exact {split_name} subset. "
                f"Requested counts={exact_counts}; available counts={available}; "
                f"shortages={missing}. No fallback sampling was performed."
            )
        chosen_pairs = []
        for n, count in exact_counts.items():
            chosen_pairs.extend(buckets[int(n)][:int(count)])
        rng.shuffle(chosen_pairs)
        chosen = chosen_pairs
        print(f"{split_name}: found {multi_count} images with >=2 GT components; "
              f"selected exactly {len(chosen)} with distribution "
              f"{dict(sorted({n: sum(1 for _, nn in chosen if nn == n) for n in set(nn for _, nn in chosen)}.items()))}.")
    else:
        candidates = [pair for n in sorted(buckets) if n >= 2 for pair in buckets[n]]
        rng.shuffle(candidates)
        chosen = candidates[:min(int(limit), len(candidates))]
        dist = {}
        for _, n in chosen:
            dist[n] = dist.get(n, 0) + 1
        print(f"Multi-region {split_name}: found {multi_count} images with >=2 GT components; "
              f"selected {len(chosen)} / requested {limit}; N distribution={dict(sorted(dist.items()))}.")
        if len(chosen) < min(int(limit), 20):
            print("[warning] Fewer than 20 multi-region images are available in this split; "
                  "the pilot will use all that were found.")

    return [rec for rec, _ in chosen], {rec[2]: int(n) for rec, n in chosen}


class QaTaDataset(Dataset):

    def __init__(self, records, mode="train", image_size=(224, 224)):
        super().__init__()
        self.records = list(records)
        self.mode = mode
        self.image_size = list(image_size)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        image_path, mask_path, name = self.records[idx]
        data = {"image": image_path, "gt": mask_path}
        data = self.transform()(data)
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

        return image, gt.int(), comps, name

    def transform(self):
        common = [
            LoadImaged(["image", "gt"], reader="PILReader"),
            EnsureChannelFirstd(["image", "gt"]),
        ]
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
    return images, gts, comps, names


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

        x = x + nn.Parameter(self.pe[:, :x.size(1)], requires_grad=False)
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
        return R, A.view(B, self.K, side, side), logits


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
        self.scale = nn.Parameter(torch.tensor(0.01), requires_grad=True)

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
    def __init__(self, in_channels, out_channels, spatial_size):
        super().__init__()
        self.guide_layer = RegionGuidedDecoderLayer(in_channels)
        self.spatial_size = spatial_size
        self.decoder = UnetrUpBlock(2, in_channels, out_channels, 3, 2, norm_name='BATCH')

    def forward(self, vis, skip_vis, regions):
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
        self.spatial_dim = [7, 14, 28, 56]
        feature_dim = [768, 384, 192, 96]
        self.vis_pos = PositionalEncoding(768)
        self.head = RegionQueryHead(cfg.num_queries, in_dim=768, mask_in_dim=96,
                                    d_q=cfg.d_q, d=cfg.d, d_v=cfg.d_v, d_m=cfg.d_m,
                                    mask_pos_enc=bool(cfg.mask_pos_enc))

        self.decoder16 = RegionGuidedDecoder(feature_dim[0], feature_dim[1], self.spatial_dim[0])
        self.decoder8 = RegionGuidedDecoder(feature_dim[1], feature_dim[2], self.spatial_dim[1])
        self.decoder4 = RegionGuidedDecoder(feature_dim[2], feature_dim[3], self.spatial_dim[2])

        self.decoder1 = SubpixelUpsample(2, feature_dim[3], 24, 4)
        self.out = UnetOutBlock(2, in_channels=24, out_channels=1)

    def forward(self, image):
        if image.shape[1] == 1:
            image = repeat(image, 'b 1 h w -> b c h w', c=3)
        image_features = self.encoder(image)['feature']
        if len(image_features[0].shape) == 4:
            image_features = image_features[1:]
        e56 = image_features[0]
        tok = [rearrange(item, 'b c h w -> b (h w) c') for item in image_features]
        os32 = tok[3]

        regions, A, mask_logits = self.head(self.vis_pos(os32), e56, side=7)

        os16, attn16 = self.decoder16(os32, tok[2], regions)
        os8, attn8 = self.decoder8(os16, tok[1], regions)
        os4, attn4 = self.decoder4(os8, tok[0], regions)
        os4 = rearrange(os4, 'B (H W) C -> B C H W', H=self.spatial_dim[-1], W=self.spatial_dim[-1])
        os1 = self.decoder1(os4)
        seg_logits = self.out(os1)
        return {'regions': regions, 'attention': A, 'mask_logits': mask_logits,
                'seg_logits': seg_logits, 'decoder_attn': [attn16, attn8, attn4]}

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
    L_unmatched_gt_region = zero
    L_match = L_matched + float(cfg.lambda_u) * L_unmatched

    L_div = spatial_diversity_loss(logits, rows, eps)
    epoch_now = int(getattr(cfg, "current_epoch", 1))
    warmup_epochs = int(getattr(cfg, "div_warmup_epochs", 5))
    ramp_epochs = int(getattr(cfg, "div_ramp_epochs", 10))
    if epoch_now <= warmup_epochs:
        effective_lambda_div = 0.0
    else:
        ramp = min(1.0, (epoch_now - warmup_epochs) / max(1, ramp_epochs))
        effective_lambda_div = float(cfg.lambda_div) * ramp
    L_total = L_match + effective_lambda_div * L_div

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
        "effective_lambda_div": float(effective_lambda_div),
        "rows": rows,
        "cols": cols,
        "cost": cost_np,
        "N": N,
        "num_unassigned_gt": len(unassigned_gt),
    }
    return L_total, info

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
    if N > 0:
        inter_kg = Bm @ g.t()
        cover = inter_kg / g.sum(1).clamp(min=1.0)[None, :]
        contain = inter_kg / area.clamp(min=1.0)[:, None]
        covered = (cover >= cfg.cover_thr) & (area > 0)[:, None]
        out["gt_multi_covered"] = int((covered.sum(0) >= 2).sum().item())
        out["n_unm_dup"] = int(sum(bool((contain[k] >= cfg.dup_thr).any()) for k in unmatched_active))
    return out


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

@torch.no_grad()


def evaluate(model, loader, device, cfg, collect_vis=False, vis_limit_per_category=None):
    model.eval()
    sums = {"L_total": 0.0, "L_match": 0.0, "L_matched": 0.0,
            "L_unmatched": 0.0, "L_unmatched_bce": 0.0, "L_unmatched_fg": 0.0,
            "L_unmatched_gt_region": 0.0, "L_foreign_region": 0.0,
            "L_div": 0.0, "L_seg": 0.0}
    n_img = 0
    per_image = []
    per_match = []
    match_freq = np.zeros(cfg.num_queries)
    vis_store = {c: [] for c in range(5)}

    for images, gts, comps_list, names in tqdm(loader, desc="valid", leave=False):
        images = images.to(device)
        outputs = model(images)
        A, logits = outputs["attention"], outputs["mask_logits"]
        seg_loss = DiceCELoss(sigmoid=True)(outputs["seg_logits"], gts.to(device).float())
        for i in range(images.shape[0]):
            comps = comps_list[i].to(device)
            _, info = region_match_loss(logits[i], comps, A[i], cfg)
            diag = sample_diagnostics(logits[i], comps, A[i], info["rows"], info["cols"], cfg)

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
            per_image.append({"name": names[i], "N": info["N"], **{k: v for k, v in diag.items()},
                              "L_total": info["L_total"], "L_match": info["L_match"],
                              "L_div": info["L_div"]})
            if collect_vis:
                c = min(info["N"], 4)

                vis_limit = (cfg.vis_per_category if vis_limit_per_category is None
                             else int(vis_limit_per_category))
                if len(vis_store[c]) < vis_limit:
                    vis_store[c].append({
                        "name": names[i],
                        "image": images[i, 0].cpu().numpy(),
                        "comps": comps_list[i].cpu(),
                        "logits": logits[i].cpu(),
                        "A": A[i].cpu(),
                    })

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
    res["query_match_freq"] = (match_freq / max(1, match_freq.sum())).round(3).tolist()
    return res, df, vis_store, pd.DataFrame(per_match)


def main():
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, "| seed:", args.seed, "| output:", OUTPUT_DIR)

    train_all, valid_all = build_qata_split()
    train_recs, train_component_counts = select_multi_region_records(
        train_all, args.pilot_train_multi_region_samples, args, split_name="train",
        exact_counts={2: 25, 3: 25})
    assert len(train_recs) == 50, f"Expected 50 training images, got {len(train_recs)}"
    assert sum(n == 2 for n in train_component_counts.values()) == 25
    assert sum(n == 3 for n in train_component_counts.values()) == 25
    valid_recs, valid_component_counts = select_multi_region_records(
        valid_all, args.pilot_valid_multi_region_samples, args, split_name="validation")
    if not train_recs or not valid_recs:
        raise RuntimeError("Could not select multi-region images from both train and validation splits. "
                           "Check mask paths, component threshold, and split CSV.")
    print(f"Mini-pilot selected -> train multi-region: {len(train_recs)} | "
          f"held-out validation multi-region: {len(valid_recs)} | total: {len(train_recs)+len(valid_recs)}")
    with open(OUTPUT_DIR / "selected_multi_region_samples.json", "w") as f:
        json.dump({"train_count": len(train_recs), "validation_count": len(valid_recs),
                   "total_count": len(train_recs)+len(valid_recs),
                   "train_requested_distribution": {"N=2": 25, "N=3": 25},
                   "train_actual_distribution": {
                       "N=2": sum(n == 2 for n in train_component_counts.values()),
                       "N=3": sum(n == 3 for n in train_component_counts.values())},
                   "train_component_counts": train_component_counts,
                   "validation_component_counts": valid_component_counts}, f, indent=2)

    ds_train = QaTaDataset(train_recs, mode='train', image_size=args.image_size)
    ds_valid = QaTaDataset(valid_recs, mode='valid', image_size=args.image_size)

    dl_train = DataLoader(ds_train, batch_size=args.train_batch_size, shuffle=True,
                          num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
                          persistent_workers=True)
    dl_valid = DataLoader(ds_valid, batch_size=args.valid_batch_size, shuffle=False,
                          num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
                          persistent_workers=True)

    model = RegionQueryPilot(args.vision_type, args.project_dim, args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.cosine_T_max, eta_min=1e-6)

    with open(OUTPUT_DIR / "pilot_config.json", "w") as f:
        json.dump(dict(args), f, indent=2, default=str)

    history = []

    best_val_dice = -float("inf")
    best_val_loss = float("inf")
    best_epoch = 0
    best_ckpt_path = OUTPUT_DIR / "best_model_by_val_dice.pt"

    best_train_l_match = float("inf")
    best_train_l_match_epoch = 0
    best_train_ckpt_path = OUTPUT_DIR / "best_model_by_train_L_match.pt"
    t0 = time.time()
    for epoch in range(args.epochs):

        args.current_epoch = epoch + 1
        args.div_warmup_epochs = 5
        args.div_ramp_epochs = 10
        model.train()
        run = {"loss": 0.0, "L_match": 0.0, "L_matched": 0.0, "L_unmatched": 0.0,
               "L_unmatched_bce": 0.0, "L_unmatched_fg": 0.0,
               "L_unmatched_gt_region": 0.0, "L_foreign_region": 0.0,
               "L_div": 0.0, "L_seg": 0.0}
        n_batches = 0
        for images, gts, comps_list, names in tqdm(dl_train, desc=f"epoch {epoch + 1}/{args.epochs}", leave=False):
            images = images.to(device)
            outputs = model(images)
            A, logits = outputs["attention"], outputs["mask_logits"]
            l_seg = DiceCELoss(sigmoid=True)(outputs["seg_logits"], gts.to(device).float())
            losses = []
            for i in range(images.shape[0]):
                comps = comps_list[i].to(device)
                loss_i, info = region_match_loss(logits[i], comps, A[i], args)
                losses.append(loss_i)
                run["L_match"] += info["L_match"] / images.shape[0]
                run["L_matched"] += info["L_matched"] / images.shape[0]
                run["L_unmatched"] += info["L_unmatched"] / images.shape[0]
                run["L_unmatched_bce"] += info["L_unmatched_bce"] / images.shape[0]
                run["L_unmatched_fg"] += info["L_unmatched_fg"] / images.shape[0]
                run["L_unmatched_gt_region"] += info["L_unmatched_gt_region"] / images.shape[0]
                run["L_foreign_region"] += info["L_foreign_region"] / images.shape[0]
                run["L_div"] += info["L_div"] / images.shape[0]
            l_match_total = torch.stack(losses).mean()
            loss = l_match_total + float(args.lambda_seg) * l_seg
            run["L_seg"] += float(l_seg.detach())

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            run["loss"] += float(loss.detach())
            n_batches += 1
        lr_scheduler.step()

        val, _, _, _ = evaluate(model, dl_valid, device, args)

        current_train_l_match = float(run["L_match"] / max(1, n_batches))
        if np.isfinite(current_train_l_match) and current_train_l_match < best_train_l_match:
            best_train_l_match = current_train_l_match
            best_train_l_match_epoch = epoch + 1
            torch.save({
                "epoch": best_train_l_match_epoch,
                "best_train_L_match": best_train_l_match,
                "model_state_dict": model.state_dict(),
            }, best_train_ckpt_path)

        current_dice = float(val.get("matched_dice", float("nan")))
        current_loss = float(val.get("L_total", float("inf")))
        is_better = np.isfinite(current_dice) and (
            current_dice > best_val_dice + 1e-8 or
            (abs(current_dice - best_val_dice) <= 1e-8 and current_loss < best_val_loss)
        )
        if is_better:
            best_val_dice = current_dice
            best_val_loss = current_loss
            best_epoch = epoch + 1
            torch.save({
                "epoch": best_epoch,
                "best_val_matched_dice": best_val_dice,
                "best_val_L_total": best_val_loss,
                "model_state_dict": model.state_dict(),
            }, best_ckpt_path)

        row = {"epoch": epoch + 1,
               "train_loss": run["loss"] / n_batches,
               "train_L_match": run["L_match"] / n_batches,
               "train_L_matched": run["L_matched"] / n_batches,
               "train_L_unmatched": run["L_unmatched"] / n_batches,
               "train_L_unmatched_bce": run["L_unmatched_bce"] / n_batches,
               "train_L_unmatched_fg": run["L_unmatched_fg"] / n_batches,
               "train_L_unmatched_gt_region": run["L_unmatched_gt_region"] / n_batches,
               "train_L_foreign_region": run["L_foreign_region"] / n_batches,
               "train_L_div": run["L_div"] / n_batches,
               "train_L_seg": run["L_seg"] / n_batches,
               **{f"val_{k}": v for k, v in val.items()}}
        history.append(row)
        print(f"[{epoch + 1:02d}/{args.epochs}] train_loss={row['train_loss']:.4f} | "
              f"train_L_match={row['train_L_match']:.4f} | "
              f"train_L_foreign={row['train_L_foreign_region']:.4f} | train_L_div={row['train_L_div']:.4f} | train_L_seg={row['train_L_seg']:.4f} | val_L_total={val['L_total']:.4f} | "
              f"val_L_div={val['L_div']:.4f} | val_dice={val['matched_dice']:.3f} | "
              f"recall@0.5={val['gt_recall@0.5']:.3f} | overlap(N>=2)={val['frac_overlap_images_N>=2']:.3f} | "
              f"gt_cov>=2q={val['frac_gt_regions_covered_by_>=2_queries']:.3f} | "
              f"unm_dup={val['frac_unmatched_active_that_duplicate_a_GT']:.3f} | "
              f"val_L_unm_bce={val['L_unmatched_bce']:.4f} | val_L_unm_fg={val['L_unmatched_fg']:.4f} | "
              f"attn_cos={val['attn_cos_mean']:.3f} | match_freq={val['query_match_freq']}")
        pd.DataFrame(history).to_csv(OUTPUT_DIR / "pilot_history.csv", index=False)

    print(f"training time: {(time.time() - t0) / 60:.1f} min")

    torch.save(model.state_dict(), OUTPUT_DIR / "last_epoch_model.pt")
    if not best_ckpt_path.exists():
        raise RuntimeError("No valid best checkpoint was saved; inspect validation matched Dice.")
    best_payload = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(best_payload["model_state_dict"])

    torch.save(model.state_dict(), OUTPUT_DIR / "region_query_pilot.pt")
    with open(OUTPUT_DIR / "best_checkpoint_info.json", "w") as f:
        json.dump({
            "selected_by": "maximum validation matched_dice; ties broken by minimum validation L_total",
            "best_epoch": best_epoch,
            "best_val_matched_dice": best_val_dice,
            "best_val_L_total": best_val_loss,
            "checkpoint": str(best_ckpt_path),
        }, f, indent=2)
    print(f"Using BEST checkpoint from epoch {best_epoch} "
          f"(val_matched_dice={best_val_dice:.4f}, val_L_total={best_val_loss:.6f}) "
          "for final evaluation and visualizations.")

    val, df, vis_store, match_df = evaluate(model, dl_valid, device, args, collect_vis=True)
    df.to_csv(OUTPUT_DIR / "pilot_per_image_diagnostics.csv", index=False)
    match_df.to_csv(OUTPUT_DIR / "pilot_per_match_iou_dice.csv", index=False)

    if not best_train_ckpt_path.exists():
        raise RuntimeError("No best-by-train-L_match checkpoint was saved.")
    best_train_payload = torch.load(best_train_ckpt_path, map_location=device)
    model.load_state_dict(best_train_payload["model_state_dict"])

    ds_train_eval = QaTaDataset(train_recs, mode='valid', image_size=args.image_size)
    dl_train_eval = DataLoader(
        ds_train_eval, batch_size=args.valid_batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
        persistent_workers=(args.num_workers > 0))
    train_eval, train_df, train_vis_store, train_match_df = evaluate(
        model, dl_train_eval, device, args, collect_vis=True,
        vis_limit_per_category=len(ds_train_eval))
    train_df.to_csv(OUTPUT_DIR / "pilot_train_per_image_diagnostics.csv", index=False)
    train_match_df.to_csv(OUTPUT_DIR / "pilot_train_per_match_iou_dice.csv", index=False)
    with open(OUTPUT_DIR / "pilot_train_summary.json", "w") as f:
        json.dump(train_eval, f, indent=2)
    with open(OUTPUT_DIR / "best_train_L_match_checkpoint_info.json", "w") as f:
        json.dump({
            "selected_by": "lowest epoch-mean training L_match",
            "best_train_L_match_epoch": best_train_l_match_epoch,
            "best_train_L_match": best_train_l_match,
            "checkpoint": str(best_train_ckpt_path),
            "purpose": "training-set visualizations and diagnostics only; not the final validation-selected model"
        }, f, indent=2)

    train_vis_dir = VIS_DIR / "train_best_L_match_checkpoint"
    train_success_dir = train_vis_dir / "multi_region_success"
    train_failure_dir = train_vis_dir / "multi_region_failure"
    train_vis_dir.mkdir(parents=True, exist_ok=True)
    train_success_dir.mkdir(parents=True, exist_ok=True)
    train_failure_dir.mkdir(parents=True, exist_ok=True)
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
            train_n_fig += 1

    with open(OUTPUT_DIR / "train_vs_validation_best_checkpoint.json", "w") as f:
        json.dump({
            "checkpoint_epoch": best_train_l_match_epoch,
            "checkpoint_selection": "lowest epoch-mean training L_match (training visualization checkpoint)",
            "best_train_L_match": best_train_l_match,
            "validation_selected_checkpoint_epoch": best_epoch,
            "validation_checkpoint_selection": "maximum validation matched_dice; ties broken by minimum validation L_total",
            "train_selected_images": len(ds_train_eval),
            "validation_selected_images": len(ds_valid),
            "train_summary": train_eval,
            "validation_summary": val,
            "note": "Train-set outputs are a diagnostic of fitted examples, not evidence of generalization. Training loss/optimizer logic is unchanged."
        }, f, indent=2)

    success_dir = VIS_DIR / "multi_region_success"
    failure_dir = VIS_DIR / "multi_region_failure"
    success_dir.mkdir(parents=True, exist_ok=True)
    failure_dir.mkdir(parents=True, exist_ok=True)
    scored = match_df.groupby("image")["binary_dice"].mean().to_dict() if not match_df.empty else {}
    for category_recs in vis_store.values():
        for rec in category_recs:
            score = scored.get(rec["name"], float("nan"))
            safe = re.sub(r"[^A-Za-z0-9_.-]", "_", rec["name"])
            dest = success_dir if np.isfinite(score) and score >= 0.5 else failure_dir
            render_sample(rec, args, dest / f"{safe}__meanDice_{score:.3f}.png")
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
    print("QaTa-COV19 REGION-QUERY + REGION-GUIDED DECODER SUMMARY (validation subset, best checkpoint)")
    print("=" * 90)
    for k, v in val.items():
        print(f"{k:46s}: {v}")
    print(f"\nN distribution (GT components per image): "
          f"{df['N'].value_counts().sort_index().to_dict()}")
    print(f"Validation figures saved: {n_fig}  ->  {VIS_DIR}")
    print(f"Train figures (lowest train L_match, epoch {best_train_l_match_epoch}, "
          f"L_match={best_train_l_match:.6f}) saved: {train_n_fig} -> {train_vis_dir}")
    print(f"Train per-image diagnostics CSV: {OUTPUT_DIR / 'pilot_train_per_image_diagnostics.csv'}")
    print(f"Train per-match IoU/Dice CSV: {OUTPUT_DIR / 'pilot_train_per_match_iou_dice.csv'}")
    print(f"Per-match IoU/Dice CSV: {OUTPUT_DIR / 'pilot_per_match_iou_dice.csv'}")
    print(f"Selected train/validation samples: {OUTPUT_DIR / 'selected_multi_region_samples.json'}")
    print(f"Success/failure example figures: {success_dir} | {failure_dir}")
    print("Component threshold: >=15 pixels at 224x224; connectivity=8. "
          "Collapse check: 'frac_gt_regions_covered_by_>=2_queries' and "
          "'frac_unmatched_active_that_duplicate_a_GT' (direct), plus "
          "'frac_overlap_images_N>=2', 'attn_cos_mean', 'query_match_freq' and the figures. "
          "L_seg uses TeViA wrapper DiceCELoss with sigmoid on decoder logits. Text encoder and static prompts are removed; R_k drives region cross-attention. lambda_div=0 for this decoder feasibility pilot. Matched queries are supervised only against their assigned component; unassigned GT components are ignored only when N_gt>K. Unmatched queries receive all-zero BCE plus a foreground-focused penalty, and that extra penalty is applied only to unmatched queries.")

if __name__ == "__main__":
    main()
