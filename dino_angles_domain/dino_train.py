#!/usr/bin/env python3
# dino_train.py
from __future__ import annotations

import re, time
from pathlib import Path
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision.transforms as T
import torchvision.transforms.functional as TF # For discrete rotation
from sklearn.model_selection import StratifiedShuffleSplit

from utils import (
    seed_everything, image_group_id, parse_camera_id, split_csv_list,
    pad_to_square, parse_prob_list, parse_swap_map, IMAGENET_MEAN, IMAGENET_STD
)
from dataset import PigCropDataset
from dino_model import build_dino_backbone, DinoHead, build_angle_head, DinoAngleMlpHead
from losses import build_class_weight_tensor, build_criterion
from evaluate import evaluate_f1, save_history, evaluate_test_and_save
import random

# ─────────────────────────────────────────────────────────────────────────────
# Helper: caps per camera and/or class
# ─────────────────────────────────────────────────────────────────────────────
def apply_caps(df, seed, per_camera_cap=0, per_class_cap=0, per_camera_class_cap=0,
               camera_col="camera_id", class_col="class_id"):
    if df.empty:
        return df
    rng = np.random.default_rng(seed)
    out = df

    if per_camera_class_cap and per_camera_class_cap > 0:
        keep = []
        for (cam, cls), g in out.groupby([camera_col, class_col], sort=False):
            if len(g) <= per_camera_class_cap:
                keep.append(g.index.values)
            else:
                keep.append(rng.choice(g.index.values, size=per_camera_class_cap, replace=False))
        out = out.loc[np.concatenate(keep)].copy()

    if per_camera_cap and per_camera_cap > 0:
        keep = []
        for cam, g in out.groupby(camera_col, sort=False):
            if len(g) <= per_camera_cap:
                keep.append(g.index.values)
            else:
                keep.append(rng.choice(g.index.values, size=per_camera_cap, replace=False))
        out = out.loc[np.concatenate(keep)].copy()

    if per_class_cap and per_class_cap > 0:
        keep = []
        for cls, g in out.groupby(class_col, sort=False):
            if len(g) <= per_class_cap:
                keep.append(g.index.values)
            else:
                keep.append(rng.choice(g.index.values, size=per_class_cap, replace=False))
        out = out.loc[np.concatenate(keep)].copy()

    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Transforms (ConvNeXt-style) – flips handled by PigCropDataset
# ─────────────────────────────────────────────────────────────────────────────
def build_transforms(img_size: int, rand_affine_p: float, re_prob: float, pad_ratio: float):
    # Flips are done in PigCropDataset, not here.
    train_tfms = T.Compose([
        T.Lambda(lambda im: pad_to_square(im, img_size)),
        # T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BILINEAR),
        T.ColorJitter(0.40, 0.40, 0.35, 0.2),
        T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.10),
        T.RandomApply([T.RandomAffine(
            # degrees=0,
            degrees=5,
            translate=(0.03, 0.03),
            scale=(0.95, 1.05),
            interpolation=T.InterpolationMode.BILINEAR,
        )], p=rand_affine_p),
        # --- NEW: Discrete 45-degree rotation ---
        T.RandomApply([
            T.Lambda(lambda img: TF.rotate(
                img, 
                angle=random.choice([45, 90, 135, 180, 225, 270, 315]), # Exclude 0 here
                interpolation=TF.InterpolationMode.BILINEAR
            ))
        ], p=0.1),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        T.RandomErasing(p=re_prob, scale=(0.02, 0.12), ratio=(0.3, 0.3), value="random"),
        # T.RandomAutocontrast(p=0.3),
        # T.RandomAdjustSharpness(sharpness_factor=2.0, p=0.3),
    ])

    val_tfms = T.Compose([
        T.Lambda(lambda im: pad_to_square(im, img_size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tfms, val_tfms

def cutmix_only(x, y, alpha=0.1, prob=0.7, skip_class=4):
    """
    x: (B, C, H, W)
    y: (B,)
    Returns:
      x_aug,
      y_mix = None or (y1, y2, lam),
      mix_type = None or "cutmix"
    """
    if random.random() > prob:
        # no augmentation
        return x, None, None

    if (y == skip_class).float().mean().item() > 0.5:
        return x, None, None

    B, C, H, W = x.size()
    device = x.device

    # shuffle indices
    indices = torch.randperm(B, device=device)
    y2 = y[indices]

    # sample lambda from Beta
    lam = np.random.beta(alpha, alpha)

    # CutMix bounding box
    cut_w = int(W * np.sqrt(1. - lam))
    cut_h = int(H * np.sqrt(1. - lam))
    # uniform center
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2c = np.clip(cy + cut_h // 2, 0, H)

    x_aug = x.clone()
    x_aug[:, :, y1:y2c, x1:x2] = x[indices, :, y1:y2c, x1:x2]

    # adjust lam based on actual box area
    box_area = (x2 - x1) * (y2c - y1)
    lam_eff = 1.0 - float(box_area) / float(H * W)

    # return exactly (y1, y2, lam_eff)
    return x_aug, (y, y2, lam_eff), "cutmix"


# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    import argparse

    ap = argparse.ArgumentParser("DINOv2 posture training (ConvNeXt-style pipeline)")

    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--train-csv", type=str, default="train.csv", help="training csv file name (relative to data-root)")
    ap.add_argument("--test-csv", type=str, default="unseenVP_test.csv")
    ap.add_argument("--test-images", type=str, default="unseenVP_test_images")
    ap.add_argument("--out-dir", type=Path, default=Path("runs/dinov2_angle_domain"))

    # DINOv2 backbone + head
    ap.add_argument("--dino-weight", type=str, required=True,
                    help="HuggingFace model name or local path, e.g. 'facebook/dinov2-base'.")
    ap.add_argument("--head", type=str, default="mlp", choices=["linear", "mlp"])
    ap.add_argument("--mlp-hidden-ratio", type=float, default=1.0)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--finetune-backbone", action="store_true",
                    help="If set, unfreeze backbone and fine-tune. Default: frozen backbone.")
    ap.add_argument("--backbone-lr-mult", type=float, default=0.1,
                    help="Backbone LR = lr * backbone_lr_mult when fine-tuning.")
    ap.add_argument("--model-tag", type=str, default="dinov2",
                    help="Tag for checkpoint filenames (like convnext model name).")

    # General training
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--label-smooth", type=float, default=0.02)
    ap.add_argument("--pad-ratio", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=3557)
    ap.add_argument("--use-amp", action="store_true", default=True)
    ap.add_argument("--no-amp", action="store_false", dest="use_amp")
    ap.add_argument(
        "--square-crop",
        action="store_true",
        help="If set, convert bbox (xywh) to a square bbox before cropping."
    )

    # camera protocol (same as ConvNeXt)
    ap.add_argument("--train-cameras", type=str, default="")
    ap.add_argument("--val-cameras", type=str, default="")
    ap.add_argument("--test-cameras", type=str, default="")
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--no-stratified-split", action="store_true")

    ap.add_argument("--seen-test-csv", type=str, default="seenVP_test.csv",
                    help="If set, also evaluate on second test set (e.g. seenVP_test.csv).")
    ap.add_argument("--seen-test-images", type=str, default="seenVP_test_images",
                    help="Image dir for second test set (e.g. seenVP_test_images). Required if --seen-test-csv is set.")

    # caps
    ap.add_argument("--train-per-camera-cap", type=int, default=0)
    ap.add_argument("--train-per-class-cap", type=int, default=0)
    ap.add_argument("--train-per-camera-class-cap", type=int, default=0)

    ap.add_argument("--val-per-camera-cap", type=int, default=0)
    ap.add_argument("--val-per-class-cap", type=int, default=0)
    ap.add_argument("--val-per-camera-class-cap", type=int, default=0)

    # flip + swap (handled in PigCropDataset)
    ap.add_argument("--hflip-prob", type=float, default=0.0)
    ap.add_argument("--vflip-prob", type=float, default=0.0)
    ap.add_argument("--flip-swap-map", type=str, default='{"0":"1","1":"0"}')
    ap.add_argument("--swap-on", type=str, default="both", choices=["h", "v", "both", "none"])
    ap.add_argument("--class-hflip-prob", type=str, default="0.95,0.95,0.95,0.35,0.5")
    ap.add_argument("--class-vflip-prob", type=str, default="0.95,0.95,0.95,0.3,0.4")
    ap.add_argument("--class-id-offset", type=int, default=0)

    # sampler toggle
    ap.add_argument("--no-weighted-sampler", action="store_true")

    # loss
    ap.add_argument("--loss", type=str, default="ce", choices=["ce", "focal"])
    ap.add_argument("--class-weights", type=str, default="none",
                    choices=["none", "inv_freq", "inv_freq_norm"])
    ap.add_argument("--focal-alpha", type=float, default=1.0)
    ap.add_argument("--focal-gamma", type=float, default=2.0)

    # augment knobs
    ap.add_argument("--rand-affine-p", type=float, default=0.0)
    ap.add_argument("--re-prob", type=float, default=0.0)

    ap.add_argument(
        "--tta",
        action="store_true",
        help="Use horizontal flip TTA at test time (2-view average)."
    )
    ap.add_argument(
        "--tta-multi",
        action="store_true",
        help="4-view TTA: avg(original, hflip, vflip, rot180). Stronger than --tta alone."
    )

    # Mixup / CutMix
    ap.add_argument("--mixup-alpha", type=float, default=0.2)
    ap.add_argument("--cutmix-alpha", type=float, default=0.1)
    ap.add_argument("--mixup-cutmix-prob", type=float, default=0.7)

    ap.add_argument("--run-name", type=str, default="")

    # ---- ablation: angle features ----
    ap.add_argument("--use-angles", action="store_true",
                    help="Concatenate [az_sin, az_cos, el_sin, el_cos] with DINOv2 CLS for "
                         "angle-conditioned classification (requires _with_angles.csv).")
    # ap.add_argument("--angle-bottleneck", type=int, default=64,
    #                 help="Bottleneck size for angle-feature fusion layer.")

    # ---- ablation: rot90 augmentation ----
    ap.add_argument("--rot90-prob", type=float, default=0.0,
                    help="Probability of applying a 90/180/270-degree rotation augmentation.")

    
    # ---- angle MLP architecture (new) ----
    ap.add_argument("--angle-mlp-dim", type=int, default=128,
                    help="Output dimension of the AngleMLP branch (4 → dim → dim).")
    ap.add_argument("--shared-hidden-dim", type=int, default=128,
                    help="Shared hidden layer size before posture/domain heads.")

    # ---- domain adversarial (new) ----
    ap.add_argument("--use-domain-adv", action="store_true",
                    help="Enable domain adversarial branch with gradient reversal (GRL).")
    ap.add_argument("--grl-lambda", type=float, default=0.1,
                    help="GRL reversal coefficient (higher = stronger domain confusion).")
    ap.add_argument("--w-domain", type=float, default=0.05,
                    help="Weight for domain adversarial loss: total_loss = posture_loss + w_domain * domain_loss.")
    ap.add_argument("--domain-head-type", type=str, default="mlp", choices=["linear", "mlp"],
                    help="Architecture of the domain classification head after GRL.")
    ap.add_argument("--domain-hidden-dim", type=int, default=128,
                    help="Hidden dim for domain head when --domain-head-type=mlp.")

    # ---- random continuous rotation augmentation (new) ----
    ap.add_argument("--rand-rot-prob", type=float, default=0.7,
                    help="Probability of applying random continuous rotation augmentation each sample.")
    ap.add_argument("--rand-rot-max-deg", type=float, default=90.0,
                    help="Maximum rotation angle (degrees) for continuous rotation augmentation.")

    return ap.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    seed_everything(args.seed)

    data_root = args.data_root
    train_csv = data_root / args.train_csv if args.train_csv else data_root / "train.csv"
    
    train_img_dir = data_root / "train_images"

    test_csv = data_root / args.test_csv if args.test_csv else None
    test_img_dir = data_root / args.test_images if args.test_images else None

    seen_test_csv = data_root / args.seen_test_csv if args.seen_test_csv else None
    seen_test_img_dir = data_root / args.seen_test_images if args.seen_test_images else None


    for p in [train_csv, test_csv, train_img_dir, test_img_dir]:
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    train_df["class_id"] = train_df["class_id"].astype(int)
    train_df["image_group"] = train_df["image_id"].map(image_group_id)
    train_df["camera_id"] = train_df["image_id"].map(parse_camera_id)

    test_df["image_group"] = test_df["image_id"].map(image_group_id)
    test_df["camera_id"] = test_df["image_id"].map(parse_camera_id)

    classes_txt = data_root / "pig_posture_classes.txt"
    if classes_txt.exists():
        class_names = [ln.strip() for ln in classes_txt.read_text().splitlines() if ln.strip()]
    else:
        class_names = ["Lateral_lying_left", "Lateral_lying_right", "Sitting", "Standing", "Sternal_lying"]

    num_classes = int(train_df["class_id"].max()) + 1

    # camera filters
    train_cams = split_csv_list(args.train_cameras)
    val_cams = split_csv_list(args.val_cameras)
    test_cams = split_csv_list(args.test_cameras)

    if train_cams:
        train_df = train_df[train_df["camera_id"].isin(train_cams)].reset_index(drop=True)

    # build trn_df_full / val_df (camera holdout or random stratified, like ConvNeXt)
    if val_cams:
        val_df = train_df[train_df["camera_id"].isin(val_cams)].reset_index(drop=True)
        trn_df_full = train_df[~train_df["camera_id"].isin(val_cams)].reset_index(drop=True)
        if len(val_df) == 0:
            print("[WARN] empty val from --val-cameras; fallback to random split")
            val_cams = []
            trn_df_full = train_df.copy()
    if not val_cams:
        trn_df_full = train_df.copy()
        img_dom = (
            trn_df_full.groupby("image_group")
            .agg(dom_label=("class_id", lambda s: int(s.value_counts().idxmax())),
                 camera_id=("camera_id", lambda s: s.iloc[0]))
            .reset_index()
        )
        groups = img_dom["image_group"].tolist()
        n_val = max(1, int(round(len(groups) * float(args.val_split))))

        if args.no_stratified_split:
            rng = np.random.default_rng(args.seed)
            val_groups = set(rng.choice(groups, size=n_val, replace=False))
        else:
            img_dom["stratum"] = img_dom["camera_id"].astype(str) + "_" + img_dom["dom_label"].astype(str)
            cnt = img_dom["stratum"].value_counts()
            rare = set(cnt[cnt < 2].index.tolist())
            if rare:
                print(f"[WARN] {len(rare)} rare strata; fallback to non-stratified")
                rng = np.random.default_rng(args.seed)
                val_groups = set(rng.choice(groups, size=n_val, replace=False))
            else:
                sss = StratifiedShuffleSplit(n_splits=1, test_size=float(args.val_split), random_state=args.seed)
                _, va_idx = next(sss.split(img_dom["image_group"], img_dom["stratum"]))
                val_groups = set(img_dom.iloc[va_idx]["image_group"].tolist())

        val_df = trn_df_full[trn_df_full["image_group"].isin(val_groups)].reset_index(drop=True)
        trn_df_full = trn_df_full[~trn_df_full["image_group"].isin(val_groups)].reset_index(drop=True)

    if test_cams:
        test_df = test_df[test_df["camera_id"].isin(test_cams)].reset_index(drop=True)

    # caps
    if any([args.train_per_camera_cap, args.train_per_class_cap, args.train_per_camera_class_cap]):
        trn_df_full = apply_caps(
            trn_df_full, seed=args.seed,
            per_camera_cap=args.train_per_camera_cap,
            per_class_cap=args.train_per_class_cap,
            per_camera_class_cap=args.train_per_camera_class_cap,
        )
    if any([args.val_per_camera_cap, args.val_per_class_cap, args.val_per_camera_class_cap]):
        val_df = apply_caps(
            val_df, seed=args.seed + 1,
            per_camera_cap=args.val_per_camera_cap,
            per_class_cap=args.val_per_class_cap,
            per_camera_class_cap=args.val_per_camera_class_cap,
        )

    # domain ids (camera -> domain id) – for metadata / future DG tricks
    all_domains = sorted(trn_df_full["camera_id"].unique().tolist())
    dom2id = {d: i for i, d in enumerate(all_domains)}
    trn_df_full["domain_id"] = trn_df_full["camera_id"].map(dom2id).astype(int)
    val_df["domain_id"] = val_df["camera_id"].map(lambda d: dom2id.get(d, -1)).astype(int)
    test_df["domain_id"] = test_df["camera_id"].map(lambda d: dom2id.get(d, -1)).astype(int)
    print("[domain] dom2id:", dom2id)

    # transforms (same idea as ConvNeXt)
    train_tfms, val_tfms = build_transforms(
        img_size=args.img_size,
        rand_affine_p=args.rand_affine_p,
        re_prob=args.re_prob,
        pad_ratio=args.pad_ratio,
    )

    flip_swap_map = parse_swap_map(args.flip_swap_map)
    class_h = parse_prob_list(args.class_hflip_prob)
    class_v = parse_prob_list(args.class_vflip_prob)


    # datasets
    ds_tr = PigCropDataset(
        trn_df_full, mode="train",
        train_img_dir=train_img_dir, test_img_dir=test_img_dir,
        pad_ratio=args.pad_ratio,
        train_tfms=train_tfms, val_tfms=val_tfms,
        hflip_prob=args.hflip_prob, vflip_prob=args.vflip_prob,
        class_hflip_probs=class_h, class_vflip_probs=class_v,
        flip_swap_map=flip_swap_map, swap_on=args.swap_on,
        class_id_offset=args.class_id_offset,
        return_domain=args.use_domain_adv,
        square_crop=args.square_crop,
        rot90_prob=args.rot90_prob,
        rand_rot_prob=args.rand_rot_prob,
        rand_rot_max_deg=args.rand_rot_max_deg,
        use_angles=args.use_angles,
       
    )
   
    ds_va = PigCropDataset(
        val_df, mode="val",
        train_img_dir=train_img_dir, test_img_dir=test_img_dir,
        pad_ratio=args.pad_ratio,
        train_tfms=train_tfms, val_tfms=val_tfms,
        return_domain=False,
        square_crop=args.square_crop,
        use_angles=args.use_angles,
       
    )

    # sampler
    class_counts = trn_df_full["class_id"].value_counts().sort_index().to_dict()
    freq = np.array([class_counts.get(i, 0) for i in range(num_classes)], dtype=np.float32)

    sampler = None
    if not args.no_weighted_sampler:
        inv = 1.0 / np.clip(freq, 1.0, None)
        sample_w = trn_df_full["class_id"].map(lambda c: inv[int(c)]).values
        sampler = WeightedRandomSampler(weights=sample_w, num_samples=len(sample_w), replacement=True)
        print("[sampler] WeightedRandomSampler enabled")
    else:
        print("[sampler] shuffle enabled")

    dl_tr = DataLoader(
        ds_tr, batch_size=args.batch,
        sampler=sampler, shuffle=(sampler is None),
        num_workers=args.workers, pin_memory=True, drop_last=True,
        persistent_workers=(args.workers > 0)
    )
    dl_va = DataLoader(
        ds_va, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        persistent_workers=(args.workers > 0)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE:", device)

    # ─────────────────────────────────────────────────────────────────────
    # Build DINOv2 backbone + head
    # ─────────────────────────────────────────────────────────────────────
    bbundle = build_dino_backbone(args.dino_weight)
    print(f"[model] DINO backbone: {bbundle.model_name}, feat_dim={bbundle.feat_dim}")

    freeze_backbone = not bool(args.finetune_backbone)

    model = build_angle_head(
        bundle=bbundle,
        num_classes=num_classes,
        num_domains=len(all_domains),
        use_angles=args.use_angles,
        angle_mlp_dim=args.angle_mlp_dim,
        shared_hidden_dim=args.shared_hidden_dim,
        head=args.head,
        mlp_hidden_ratio=args.mlp_hidden_ratio,
        dropout=args.dropout,
        freeze_backbone=freeze_backbone,
        use_domain_adv=args.use_domain_adv,
        grl_lambda=args.grl_lambda,
        domain_head_type=args.domain_head_type,
        domain_hidden_dim=args.domain_hidden_dim,
    ).to(device)

    _in_dim_str = (f"{bbundle.feat_dim}+AngleMLP({args.angle_mlp_dim})" if args.use_angles
                   else f"{bbundle.feat_dim}(no angles)")
    print(f"[model] DinoAngleMlpHead | "
          f"in={_in_dim_str} → shared_hidden={args.shared_hidden_dim} → num_classes={num_classes} | "
          f"head={args.head} | freeze_backbone={freeze_backbone} | "
          f"domain_adv={args.use_domain_adv} (λ={args.grl_lambda}, w={args.w_domain})")

    # optimizer: all non-backbone params
    head_params = [p for name, p in model.named_parameters()
                   if not name.startswith("backbone.")]

    if freeze_backbone:
        optimizer = torch.optim.AdamW(
            [p for p in head_params if p.requires_grad],
            lr=args.lr,
            weight_decay=args.wd,
        )
        print(f"[opt] frozen backbone, head lr={args.lr}")
    else:
        backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": args.lr * args.backbone_lr_mult},
                {"params": [p for p in head_params if p.requires_grad], "lr": args.lr},
            ],
            lr=args.lr,
            weight_decay=args.wd,
        )
        print(f"[opt] finetune backbone: head lr={args.lr}, backbone lr={args.lr * args.backbone_lr_mult}")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.05
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(args.use_amp and device.type == "cuda"))

    # loss
    class_weight_t = build_class_weight_tensor(freq=freq, scheme=args.class_weights, device=device)
    criterion = build_criterion(
        loss_name=args.loss,
        label_smooth=args.label_smooth,
        class_weight_t=class_weight_t,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
    )
    domain_criterion = nn.CrossEntropyLoss()
    print(f"[loss] {args.loss} | class_weights={args.class_weights} | label_smooth={args.label_smooth} | "
          f"focal_alpha={args.focal_alpha} focal_gamma={args.focal_gamma}")

    run_tag = args.run_name.strip()
    if run_tag:
        run_tag = "_" + re.sub(r"[^a-zA-Z0-9_\-]+", "_", run_tag)

    tag = args.model_tag
    best_f1, bad = -1.0, 0
    best_train_loss = 1e9
    best_path = args.out_dir / f"best{run_tag}.pt"
    last_path = args.out_dir / f"last{run_tag}.pt"

    history = {"epoch": [], "train_loss": [], "val_f1": [], "lr": []}

    # ─────────────────────────────────────────────────────────────────────
    # Train loop
    # ─────────────────────────────────────────────────────────────────────
    for ep in range(args.epochs):
        t0 = time.time()
        model.train()
        if freeze_backbone:
            model.backbone.eval()

        running, seen = 0.0, 0

        for batch in dl_tr:
            domain_ids = None
            if args.use_angles and args.use_domain_adv:
                x, angles, y, domain_ids = batch[0], batch[1], batch[2], batch[3]
                angles = angles.to(device, non_blocking=True)
                domain_ids = domain_ids.to(device, non_blocking=True)
            elif args.use_angles:
                x, angles, y = batch[0], batch[1], batch[2]
                angles = angles.to(device, non_blocking=True)
            elif args.use_domain_adv:
                x, y, domain_ids = batch[0], batch[1], batch[2]
                angles = None
                domain_ids = domain_ids.to(device, non_blocking=True)
            else:
                x, y = batch[0], batch[1]
                angles = None
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda", enabled=(args.use_amp and device.type == "cuda")):
                x_aug, y_mix, mix_type = cutmix_only(x, y, alpha=float(args.mixup_alpha), prob=args.mixup_cutmix_prob)

                posture_logits, domain_logits = (
                    model(x_aug, angles) if angles is not None else model(x_aug)
                )

                if y_mix is None:
                    loss = criterion(posture_logits, y)
                else:
                    y1, y2, lam = y_mix
                    loss = lam * criterion(posture_logits, y1) + (1 - lam) * criterion(posture_logits, y2)

                if args.use_domain_adv and domain_logits is not None and domain_ids is not None:
                    loss = loss + args.w_domain * domain_criterion(domain_logits, domain_ids)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            running += float(loss.item()) * x.size(0)
            seen += x.size(0)


        scheduler.step()
        val_f1 = evaluate_f1(model, dl_va, device)
        train_loss = running / max(1, seen)
        dt = (time.time() - t0) / 60.0
        print(f"Epoch {ep+1:02d}/{args.epochs} | loss={train_loss:.4f} | val_f1={val_f1:.5f} | {dt:.2f} min")

        history["epoch"].append(ep + 1)
        history["train_loss"].append(train_loss)
        history["val_f1"].append(float(val_f1))
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))

        if val_f1 > best_f1 + 1e-6:
            best_f1 = val_f1
            bad = 0
            torch.save({"model": model.state_dict(), "meta": {"dom2id": dom2id}}, best_path)
            print("  -> saved best:", best_path)
        else:
            bad += 1
            if bad >= args.patience:
                print("Early stopping.")
                break

    # Save last epoch model
    # torch.save({"model": model.state_dict()}, last_path)
    # print("Saved last epoch model:", last_path)

    print("Best val macro-F1:", float(best_f1))
    save_history(args.out_dir, run_tag=f"_{run_tag}", history=history)

    # load best for test evaluation
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    if freeze_backbone:
        model.backbone.eval()

    # test evaluation (if labeled)
    test_is_labeled = ("class_id" in test_df.columns) and (test_df["class_id"].notna().all())
    if test_is_labeled:
        # Testing on Test set
        ds_te = PigCropDataset(
            test_df, mode="test",
            train_img_dir=train_img_dir, test_img_dir=test_img_dir,
            pad_ratio=args.pad_ratio,
            train_tfms=None, val_tfms=val_tfms,
            use_angles=args.use_angles,
           
        )
        flip_pairs = [(0, 1)] if (args.tta or args.tta_multi) else None

        print("Testing on Test set")
        evaluate_test_and_save(
            model=model,
            test_df=test_df,
            ds_test=ds_te,
            batch_size=args.batch,
            num_workers=args.workers,
            device=device,
            use_amp=args.use_amp,
            class_names=class_names,
            out_dir=args.out_dir,
            run_tag=f"{run_tag}",
            use_tta=args.tta or args.tta_multi,
            flip_swap_pairs=flip_pairs,
            tta_multi=args.tta_multi,
        )

        # Test on test set 2
        if seen_test_csv and seen_test_csv.exists() and seen_test_img_dir and seen_test_img_dir.exists():
            test2_df = pd.read_csv(seen_test_csv)
            if "camera_id" not in test2_df.columns and "image_id" in test2_df.columns:
                test2_df["camera_id"] = test2_df["image_id"].map(parse_camera_id)
            if "camera_id" in test2_df.columns:
                test2_df["domain_id"] = test2_df["camera_id"].map(lambda d: dom2id.get(d, -1)).astype(int)
            else:
                test2_df["domain_id"] = -1
            ds_te1 = PigCropDataset(
                test2_df, mode="test",
                train_img_dir=train_img_dir, test_img_dir=seen_test_img_dir,
                pad_ratio=args.pad_ratio,
                train_tfms=None, val_tfms=val_tfms,
                use_angles=args.use_angles,
               
            )
            flip_pairs = [(0, 1)] if (args.tta or args.tta_multi) else None

            evaluate_test_and_save(
                model=model,
                test_df=test2_df,
                ds_test=ds_te1,
                batch_size=args.batch,
                num_workers=args.workers,
                device=device,
                use_amp=args.use_amp,
                class_names=class_names,
                out_dir=args.out_dir,
                run_tag=f"{run_tag}_test2",
                use_tta=args.tta or args.tta_multi,
                flip_swap_pairs=flip_pairs,
                tta_multi=args.tta_multi,
            )

if __name__ == "__main__":
    main()


"""
Usage example (without masks, angles, or domain adv):
CUDA_VISIBLE_DEVICES=0 python dinov3_angles_domain/dinov3_train.py \
  --data-root dataset/multiview_pig_posture_recognition_latest/ \
  --train-set 1 \
  --test2-csv test2_gt.csv \
  --test2-images test2_images \
  --dinov2-weight facebook/dinov2-base \
  --out-dir runs/dinov2_angle_domain/nomask_nad \
  --dropout 0.1 \
  --hflip-prob 0.5 --vflip-prob 0.5 \
  --loss ce --class-weights none --no-weighted-sampler \
  --run-name nomask_nad \
  --finetune-backbone \
  --shared-hidden-dim 128 \
  --rand-rot-prob 0.7 --rand-rot-max-deg 90 \
  --epochs 30 --patience 8 --lr 1e-4 --wd 0.05

With angles but no masks or domain adv:
CUDA_VISIBLE_DEVICES=1 python dinov3_angles_domain/dinov3_train.py \
  --data-root dataset/multiview_pig_posture_recognition_latest/ \
  --train-set 1 \
  --test2-csv test2_gt.csv \
  --test2-images test2_images \
  --dinov3-weight facebook/dinov2-base \
  --out-dir runs/dinov3_angle_domain/angle_nomask_domain \
  --dropout 0.1 \
  --hflip-prob 0.5 --vflip-prob 0.5 \
  --loss ce --class-weights none --no-weighted-sampler \
  --run-name angle_nomask_domain \
  --finetune-backbone \
  --shared-hidden-dim 128 \
  --angle-mlp-dim 32 \
  --angle-bottleneck 32 \
  --rand-rot-prob 0.7 --rand-rot-max-deg 90 \
  --epochs 30 --patience 8 --lr 1e-4 --wd 0.05 --use-angles

With angles and masks but no domain adv:
CUDA_VISIBLE_DEVICES=0 python dinov3_angles_domain/dinov3_train.py \
  --data-root dataset/multiview_pig_posture_recognition_latest/ \
  --train-set 1 \
  --test2-csv test2_gt.csv \
  --test2-images test2_images \
  --dinov3-weight facebook/dinov2-base \
  --out-dir runs/dinov3_angle_domain/angle_mask_no_domain \
  --dropout 0.1 \
  --hflip-prob 0.5 --vflip-prob 0.5 --rand-rot-prob 0.7 --rand-rot-max-deg 90 \
  --loss ce --class-weights none --no-weighted-sampler \
  --run-name angle_mask_no_domain  \
  --use-mask \
  --mask-dir dataset/multiview_pig_posture_recognition_latest/masks/train1/ \
  --test-mask-dir dataset/multiview_pig_posture_recognition_latest/masks/test \
  --test2-mask-dir dataset/multiview_pig_posture_recognition_latest/masks/test2 \
  --mask-fill mean \
  --finetune-backbone \
  --epochs 30 --patience 8 --lr 1e-4 --wd 0.05 --use-angles

With angles and domain adv but no masks:
CUDA_VISIBLE_DEVICES=0 python dinov3_angles_domain/dinov3_train.py \
  --data-root dataset/multiview_pig_posture_recognition_latest/ \
  --train-set 1 \
  --dinov3-weight facebook/dinov2-base \
  --out-dir runs/dinov3_angle_domain/angle_domain_no_mask \
  --dropout 0.1 \
  --hflip-prob 0.5 --vflip-prob 0.5 \
  --loss ce --class-weights none --no-weighted-sampler \
  --run-name angle_domain_no_mask  \
  --finetune-backbone \
  --use-domain-adv \
  --grl-lambda 0.1 \
  --w-domain 0.05 \
  --epochs 30 --patience 8 --lr 1e-4 --wd 0.05 --use-angles
  
With angles, masks, and domain adv:
CUDA_VISIBLE_DEVICES=1 python dinov3_angles_domain/dinov3_train.py \
  --data-root dataset/multiview_pig_posture_recognition_latest/ \
  --train-set 1 \
  --test2-csv test2_gt.csv \
  --test2-images test2_images \
  --dinov3-weight facebook/dinov2-base \
  --out-dir runs/dinov3_angle_domain/angle_mask_domain \
  --dropout 0.1 \
  --hflip-prob 0.5 --vflip-prob 0.5 \
  --loss ce --class-weights none --no-weighted-sampler \
  --run-name angle_mask_domain  \
  --use-mask \
  --mask-dir dataset/multiview_pig_posture_recognition_latest/masks/train1/ \
  --test-mask-dir dataset/multiview_pig_posture_recognition_latest/masks/test \
  --test2-mask-dir dataset/multiview_pig_posture_recognition_latest/masks/test2 \
  --mask-fill mean \
  --finetune-backbone \
  --use-domain-adv \
  --grl-lambda 0.1 \
  --w-domain 0.05 \
  --epochs 30 --patience 8 --lr 1e-4 --wd 0.05 --use-angles

With
"""