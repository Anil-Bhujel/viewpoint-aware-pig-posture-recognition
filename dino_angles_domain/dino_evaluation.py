#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dino_evaluation.py

Evaluate a trained DINO posture model on the Kaggle-style dataset
using a saved checkpoint (e.g. best_dino_t2.pt), with optional flip-TTA,
and optionally save failed-case crops for visual inspection.
"""

from __future__ import annotations

import argparse
import ast
import math
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
from sklearn.metrics import f1_score, confusion_matrix, classification_report

from PIL import Image, ImageDraw, ImageFont

from dataset import PigCropDataset
from dino_model import build_dino_backbone, DinoHead, build_angle_head, DinoAngleMlpHead
from evaluate import plot_confmat
from utils import pad_to_square, IMAGENET_MEAN, IMAGENET_STD, parse_swap_map


# ---------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------
def build_val_transforms(img_size: int) -> T.Compose:
    val_tfms = T.Compose([
        T.Lambda(lambda im: pad_to_square(im, img_size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return val_tfms


# ---------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser("DINO posture evaluation (Kaggle dataset)")

    ap.add_argument("--data-root", type=Path, required=True,
                    help="Root with train.csv / test.csv etc.")
    ap.add_argument("--test-csv", type=str, default="unseenVP_test.csv",
                    help="CSV with test GT labels.")
    ap.add_argument("--test-images", type=str, default="unseenVP_test_images",
                    help="Folder with test images (relative to data-root).")

    ap.add_argument("--ckpt", type=Path, required=True,
                    help="Path to checkpoint saved by dino_train.py (best_dino_t*.pt).")
    ap.add_argument("--dino-weight", type=str, default="facebook/dinov2-base",
                    help="Backbone name used during training.")
    ap.add_argument("--head", type=str, default="mlp",
                    choices=["linear", "mlp"],
                    help="Head type used during training.")

    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--pad-ratio", type=float, default=0.03)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)

    ap.add_argument("--device", type=str, default="cuda",
                    help="'cuda' or 'cpu'.")
    ap.add_argument("--use-amp", action="store_true",
                    help="Use AMP during evaluation.")
    ap.add_argument("--no-amp", action="store_false", dest="use_amp")

    ap.add_argument("--tta", action="store_true",
                    help="Use horizontal flip TTA (with left/right swap).")
    ap.add_argument(
        "--flip-swap-map",
        type=str,
        default='{"0":"1","1":"0"}',
        help="JSON dict mapping class indices for horizontal flip "
             "(e.g. '{\"0\":\"1\",\"1\":\"0\"}' for left/right).",
    )

    ap.add_argument("--run-tag", type=str, default="Dino_eval",
                    help="Tag prefix for output filenames.")

    # ---- ablation: angle features ----
    ap.add_argument("--use-angles", action="store_true",
                    help="Model was trained with angle features (DinoAngleHead).")
    ap.add_argument("--angle-bottleneck", type=int, default=64,
                    help="Bottleneck size used at training time.")


    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory to save evaluation outputs. If omitted, uses checkpoint parent.",
    )

    # failed-case visualization
    ap.add_argument("--save-viz", action="store_true",
                    help="Save debug crops into viz/ folder")
    ap.add_argument("--viz-max", type=int, default=3000,
                    help="Max images to save")
    ap.add_argument("--viz-only-mistakes", action="store_true",
                    help="Save only misclassified examples")
    ap.add_argument("--viz-out", type=Path, default=None,
                    help="Override viz output folder (default: out_dir/viz)")
    ap.add_argument("--viz-font-size", type=int, default=18)

    return ap.parse_args()


# ---------------------------------------------------------------------
# Architecture auto-detection from checkpoint
# ---------------------------------------------------------------------
def detect_arch_from_state_dict(state_dict: dict) -> dict:
    """
    Inspect checkpoint keys to infer the model architecture so that
    evaluation works without needing to re-specify all training flags.
    """
    keys = set(state_dict.keys())
    is_angle_mlp_head = any(k.startswith("shared_hidden.") for k in keys)

    if not is_angle_mlp_head:
        # Plain DinoHead
        return {"is_angle_mlp_head": False}

    use_angles = any(k.startswith("angle_mlp.") for k in keys)
    use_domain_adv = any(k.startswith("domain_head.") for k in keys)

    angle_mlp_dim = int(state_dict["angle_mlp.0.weight"].shape[0]) if use_angles else 32
    shared_hidden_dim = int(state_dict["shared_hidden.0.weight"].shape[0])

    # Detect head type: if posture_head has only 2 tensors it's linear, >2 → mlp
    posture_head_keys = sorted(k for k in keys if k.startswith("posture_head."))
    head = "linear" if len(posture_head_keys) == 2 else "mlp"

    # Infer mlp_hidden_ratio from posture_head mlp if applicable
    mlp_hidden_ratio = 1.0
    if head == "mlp":
        w0_shape = state_dict["posture_head.0.weight"].shape  # (hidden, shared_hidden_dim)
        hd = w0_shape[0]
        mlp_hidden_ratio = hd / shared_hidden_dim

    num_domains = 1
    domain_head_type = "mlp"
    domain_hidden_dim = 128
    if use_domain_adv:
        domain_head_keys = sorted(k for k in keys if k.startswith("domain_head.") and "weight" in k)
        num_domains = int(state_dict[domain_head_keys[-1]].shape[0])
        domain_head_type = "linear" if len(domain_head_keys) == 1 else "mlp"
        if domain_head_type == "mlp":
            domain_hidden_dim = int(state_dict["domain_head.0.weight"].shape[0])

    return {
        "is_angle_mlp_head": True,
        "use_angles": use_angles,
        "use_domain_adv": use_domain_adv,
        "angle_mlp_dim": angle_mlp_dim,
        "shared_hidden_dim": shared_hidden_dim,
        "head": head,
        "mlp_hidden_ratio": mlp_hidden_ratio,
        "num_domains": num_domains,
        "domain_head_type": domain_head_type,
        "domain_hidden_dim": domain_hidden_dim,
    }


# ---------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------
def build_model(
    dino_weight: str,
    num_classes: int,
    head: str,
    device: torch.device,
    arch: dict | None = None,
) -> nn.Module:
    """
    Build the model.  If `arch` is provided (from detect_arch_from_state_dict),
    the architecture is matched exactly to the checkpoint.
    """
    bbundle = build_dino_backbone(dino_weight)

    if arch is not None and arch.get("is_angle_mlp_head"):
        model = build_angle_head(
            bundle=bbundle,
            num_classes=num_classes,
            use_angles=arch.get("use_angles", False),
            angle_mlp_dim=arch.get("angle_mlp_dim", 32),
            shared_hidden_dim=arch.get("shared_hidden_dim", 128),
            head=arch.get("head", head),
            mlp_hidden_ratio=arch.get("mlp_hidden_ratio", 1.0),
            freeze_backbone=False,
            use_domain_adv=arch.get("use_domain_adv", False),
            num_domains=arch.get("num_domains", 1),
            domain_head_type=arch.get("domain_head_type", "mlp"),
            domain_hidden_dim=arch.get("domain_hidden_dim", 128),
        ).to(device)
    else:
        model = DinoHead(
            backbone=bbundle.backbone,
            feat_dim=bbundle.feat_dim,
            num_classes=num_classes,
            freeze_backbone=False,
            head=head,
        ).to(device)

    return model


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def parse_bbox_xywh(bbox_val):
    if isinstance(bbox_val, str):
        bbox = ast.literal_eval(bbox_val)
    else:
        bbox = bbox_val
    if len(bbox) != 4:
        raise ValueError(f"Invalid bbox: {bbox_val}")
    x, y, w, h = map(float, bbox)
    return x, y, w, h


def apply_flip_swap_to_probs(probs: torch.Tensor, flip_swap_pairs: List[Tuple[int, int]]) -> torch.Tensor:
    probs = probs.clone()
    for a, b in flip_swap_pairs:
        tmp = probs[:, a].clone()
        probs[:, a] = probs[:, b]
        probs[:, b] = tmp
    return probs


@torch.no_grad()
def predict_probs(
    model: nn.Module,
    ds_test,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    use_amp: bool,
    use_tta: bool = False,
    flip_swap_pairs: List[Tuple[int, int]] | None = None,
):
    loader = DataLoader(
        ds_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    model.eval()
    all_probs = []
    all_y = []

    for batch in loader:
        # Support both (x, y) and (x, angles, y) batches
        if len(batch) == 3:
            x, angles, y = batch[0], batch[1], batch[2]
            angles = angles.to(device, non_blocking=True)
        else:
            x, y = batch[0], batch[1]
            angles = None
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.amp.autocast(
            device_type=("cuda" if device.type == "cuda" else "cpu"),
            enabled=(use_amp and device.type == "cuda")
        ):
            out = model(x, angles) if angles is not None else model(x)
            # DinoAngleMlpHead returns (posture_logits, domain_logits); DinoHead returns logits directly
            logits = out[0] if isinstance(out, (tuple, list)) else out
            probs = torch.softmax(logits, dim=1)

            if use_tta:
                x_flip = torch.flip(x, dims=[3])
                # For angle-conditioned models: apply hflip transform to az_sin
                if angles is not None:
                    angles_flip = angles.clone()
                    angles_flip[:, 0] = -angles_flip[:, 0]  # az_sin → -az_sin
                    out_flip = model(x_flip, angles_flip)
                else:
                    out_flip = model(x_flip)
                logits_flip = out_flip[0] if isinstance(out_flip, (tuple, list)) else out_flip
                probs_flip = torch.softmax(logits_flip, dim=1)

                if flip_swap_pairs:
                    probs_flip = apply_flip_swap_to_probs(probs_flip, flip_swap_pairs)

                probs = 0.5 * (probs + probs_flip)

        all_probs.append(probs.detach().cpu().numpy())
        all_y.append(y.detach().cpu().numpy())

    probs = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_y, axis=0)
    row_ids = ds_test.df["row_id"].tolist() if "row_id" in ds_test.df.columns else list(range(len(ds_test.df)))

    return row_ids, y_true, probs


def save_debug_crops(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    class_names: List[str],
    images_dir: Path,
    out_dir: Path,
    pad_ratio: float = 0.03,
    only_mistakes: bool = True,
    max_save: int = 3000,
    font_size: int = 18,
    banner_pad: int = 8,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    mistakes_dir = out_dir / "mistakes"
    correct_dir = out_dir / "correct"
    mistakes_dir.mkdir(parents=True, exist_ok=True)
    correct_dir.mkdir(parents=True, exist_ok=True)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    saved = 0
    for i, row in df.reset_index(drop=True).iterrows():
        gt = int(y_true[i])
        pred = int(y_pred[i])
        conf = float(probs[i, pred])

        is_mistake = (gt != pred)
        if only_mistakes and not is_mistake:
            continue
        if saved >= max_save:
            break

        img_path = images_dir / row["image_id"]
        if not img_path.exists():
            print(f"[WARN] missing image: {img_path}")
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] failed to open {img_path}: {e}")
            continue

        x, y, w, h = parse_bbox_xywh(row["bbox"])

        W, H = img.size
        pad_w = w * pad_ratio
        pad_h = h * pad_ratio

        x1 = max(0, int(math.floor(x - pad_w)))
        y1 = max(0, int(math.floor(y - pad_h)))
        x2 = min(W, int(math.ceil(x + w + pad_w)))
        y2 = min(H, int(math.ceil(y + h + pad_h)))

        crop = img.crop((x1, y1, x2, y2))

        gt_text = f"GT: {class_names[gt]} ({gt})"
        pred_text = f"PRED: {class_names[pred]} ({pred})"
        conf_text = f"Conf: {conf:.4f}"

        GT_COLOR = (0, 170, 0)
        PRED_COLOR = (200, 0, 0) if is_mistake else (0, 170, 0)
        CONF_COLOR = (0, 80, 200)

        dummy = Image.new("RGB", (10, 10))
        ddraw = ImageDraw.Draw(dummy)

        gt_bbox = ddraw.textbbox((0, 0), gt_text, font=font)
        pred_bbox = ddraw.textbbox((0, 0), pred_text, font=font)
        conf_bbox = ddraw.textbbox((0, 0), conf_text, font=font)

        gt_w, gt_h = gt_bbox[2] - gt_bbox[0], gt_bbox[3] - gt_bbox[1]
        pred_w, pred_h = pred_bbox[2] - pred_bbox[0], pred_bbox[3] - pred_bbox[1]
        conf_w, conf_h = conf_bbox[2] - conf_bbox[0], conf_bbox[3] - conf_bbox[1]

        text_h = max(gt_h, pred_h, conf_h)

        crop_w, crop_h = crop.size
        banner_h = text_h + 2 * banner_pad
        canvas_w = max(crop_w, gt_w + pred_w + conf_w + 6 * banner_pad)
        canvas_h = crop_h + banner_h

        canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))

        crop_x = (canvas_w - crop_w) // 2
        canvas.paste(crop, (crop_x, banner_h))

        draw = ImageDraw.Draw(canvas)
        draw.line([(0, banner_h - 1), (canvas_w, banner_h - 1)], fill=(180, 180, 180), width=1)

        x_cursor = banner_pad
        y_text = banner_pad

        draw.text((x_cursor, y_text), gt_text, fill=GT_COLOR, font=font)
        x_cursor += gt_w + banner_pad * 2

        draw.text((x_cursor, y_text), pred_text, fill=PRED_COLOR, font=font)
        x_cursor += pred_w + banner_pad * 2

        draw.text((x_cursor, y_text), conf_text, fill=CONF_COLOR, font=font)

        base_name = Path(row["image_id"]).stem
        row_id = row["row_id"] if "row_id" in row else f"{i:06d}"

        if is_mistake:
            save_dir = mistakes_dir / f"gt_{gt}_{class_names[gt]}__pred_{pred}_{class_names[pred]}"
        else:
            save_dir = correct_dir / f"class_{gt}_{class_names[gt]}"

        save_dir.mkdir(parents=True, exist_ok=True)

        out_path = save_dir / f"{row_id}_{base_name}.jpg"
        canvas.save(out_path, quality=95)
        saved += 1

    print(f"[viz] saved {saved} crops to {out_dir}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("DEVICE:", device)

    data_root: Path = args.data_root
    test_csv = data_root / args.test_csv
    test_img_dir = data_root / args.test_images
    train_img_dir = data_root / "train_images"

    for p in [test_csv, test_img_dir, train_img_dir]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required path: {p}")

    # -----------------------------
    # Load CSVs & class names
    # -----------------------------
    test_df = pd.read_csv(test_csv)
    test_df["class_id"] = test_df["class_id"].astype(int)

    classes_txt = data_root / "pig_posture_classes.txt"
    if classes_txt.exists():
        class_names = [ln.strip() for ln in classes_txt.read_text().splitlines() if ln.strip()]
    else:
        class_names = ["Lateral_lying_left", "Lateral_lying_right", "Sitting", "Standing", "Sternal_lying"]

    num_classes = len(class_names)
    print(f"[data] num_classes={num_classes} | classes={class_names}")

    # -----------------------------
    # Dataset
    # -----------------------------
    val_tfms = build_val_transforms(args.img_size)

    ds_te = PigCropDataset(
        test_df,
        mode="test",
        train_img_dir=train_img_dir,
        test_img_dir=test_img_dir,
        pad_ratio=args.pad_ratio,
        train_tfms=None,
        val_tfms=val_tfms,
        use_angles=args.use_angles,
       
    )

    # -----------------------------
    # Load checkpoint + auto-detect architecture
    # -----------------------------
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    if "model" in ckpt:
        state_dict = ckpt["model"]
    elif "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        raise KeyError(f"Checkpoint {args.ckpt} does not contain 'model' or 'model_state_dict'.")

    arch = detect_arch_from_state_dict(state_dict)
    print(f"[ckpt] Detected architecture: {arch}")

    # For angle-based models the dataset must also return angles
    if arch.get("use_angles") and not args.use_angles:
        print("[ckpt] Checkpoint uses angle features but --use-angles was not passed; enabling automatically.")
        args.use_angles = True
        # Rebuild dataset with angles enabled
        ds_te = PigCropDataset(
            test_df,
            mode="test",
            train_img_dir=train_img_dir,
            test_img_dir=test_img_dir,
            pad_ratio=args.pad_ratio,
            train_tfms=None,
            val_tfms=val_tfms,
            use_angles=True,
          
        )

    # Build model matching checkpoint architecture
    model = build_model(
        dino_weight=args.dino_weight,
        num_classes=num_classes,
        head=args.head,
        device=device,
        arch=arch,
    )

    # Try strict load first; if it fails, attempt flexible reconciliation
    def _flexible_load(model: nn.Module, state_dict: dict):
        try:
            model.load_state_dict(state_dict, strict=True)
            print(f"[ckpt] Strict load succeeded from {args.ckpt}")
            return
        except RuntimeError as e:
            print(f"[ckpt] Strict load failed: {e}")

        sd = dict(state_dict)  # copy
        msd = model.state_dict()

        # Handle older checkpoints that named a single Linear as 'classifier.weight/bias'
        # while the current model defines a Sequential at 'classifier' (mlp).
        # If shapes match the final layer (classifier.3) map them; otherwise try replacing
        # the model.classifier with a Linear to match the checkpoint.
        if "classifier.weight" in sd or "classifier.bias" in sd:
            w = sd.get("classifier.weight")
            b = sd.get("classifier.bias")
            # Prefer mapping to existing matching-shaped layer keys
            mapped = False
            for cand in ("classifier.3.weight", "classifier.0.weight", "classifier.weight"):
                if cand in msd and w is not None and msd[cand].shape == w.shape:
                    # map weight/bias to this candidate
                    sd[cand] = w
                    if b is not None:
                        sd[cand.replace("weight", "bias")] = b
                    sd.pop("classifier.weight", None)
                    sd.pop("classifier.bias", None)
                    mapped = True
                    print(f"[ckpt] Remapped 'classifier' -> '{cand.split('.')[0]}' by shape match")
                    break

            if not mapped and w is not None:
                # try replacing model.classifier (Sequential) with a single Linear
                try:
                    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
                        in_f = w.shape[1]
                        out_f = w.shape[0]
                        # create new linear on the same device as the model
                        model_dev = next(model.parameters()).device
                        new_cl = nn.Linear(in_f, out_f).to(model_dev)
                        # copy weights/bias to the module device
                        new_cl.weight.data.copy_(w.to(model_dev))
                        if b is not None:
                            new_cl.bias.data.copy_(b.to(model_dev))
                        model.classifier = new_cl
                        # rebuild msd after mutation
                        msd = model.state_dict()
                        # remove old keys from sd to avoid conflicts
                        sd.pop("classifier.weight", None)
                        sd.pop("classifier.bias", None)
                        print("[ckpt] Replaced Sequential 'classifier' with Linear to match checkpoint (moved to model device)")
                except Exception as e:
                    print(f"[ckpt] Failed to replace classifier: {e}")

        # Last resort: load non-strict and report missing/unexpected keys
        try:
            model.load_state_dict(sd, strict=False)
            print(f"[ckpt] Loaded weights with strict=False from {args.ckpt}")
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint even with flexible mapping: {e}")

    _flexible_load(model, state_dict)
    model.eval()

    # -----------------------------
    # TTA swap pairs
    # -----------------------------
    flip_pairs: List[Tuple[int, int]] = []
    if args.tta and args.flip_swap_map:
        swap_map = parse_swap_map(args.flip_swap_map)
        seen = set()
        for k, v in swap_map.items():
            a = int(k)
            b = int(v)
            if a < 0 or b < 0 or a >= num_classes or b >= num_classes or a == b:
                continue
            pair = (min(a, b), max(a, b))
            if pair in seen:
                continue
            seen.add(pair)
            flip_pairs.append(pair)
        flip_pairs.sort()
        print("[TTA] flip_pairs:", flip_pairs)

    # -----------------------------
    # Out directory
    # -----------------------------
    out_dir = args.out_dir if args.out_dir is not None else args.ckpt.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    print("[out] saving evaluation outputs to:", out_dir)

    # -----------------------------
    # Predict
    # -----------------------------
    row_ids, y_true, probs = predict_probs(
        model=model,
        ds_test=ds_te,
        batch_size=args.batch,
        num_workers=args.workers,
        device=device,
        use_amp=args.use_amp,
        use_tta=args.tta,
        flip_swap_pairs=flip_pairs,
    )

    y_pred = probs.argmax(axis=1).astype(int)
    test_f1 = f1_score(y_true, y_pred, average="macro")
    print(f"Test macro F1: {test_f1:.4f}")

    # -----------------------------
    # Reports
    # -----------------------------
    report_dict = classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    print(report_text)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    (out_dir / f"{args.run_tag}_classification_report.txt").write_text(report_text)
    (out_dir / f"{args.run_tag}_classification_report.json").write_text(json.dumps(report_dict, indent=2))
    cm_df.to_csv(out_dir / f"{args.run_tag}_confusion_matrix.csv")

    plot_confmat(
        cm_counts=cm,
        class_names=class_names,
        title=f"[POSTURE] TEST | macro-F1={test_f1:.4f}",
        out_png=out_dir / f"{args.run_tag}_confusion_matrix_norm.png",
        normalize=True,
        title_fontsize=18,
        label_fontsize=16,
        tick_fontsize=14,
        annot_fontsize=13,
    )

    pred_df = pd.DataFrame({
        "row_id": row_ids,
        "image_id": test_df["image_id"].values,
        "y_true": y_true,
        "y_pred": y_pred,
        "true_name": [class_names[i] for i in y_true],
        "pred_name": [class_names[i] for i in y_pred],
        "confidence": probs[np.arange(len(probs)), y_pred],
    })
    pred_df["correct"] = (pred_df["y_true"] == pred_df["y_pred"]).astype(int)
    pred_df.to_csv(out_dir / f"{args.run_tag}_predictions.csv", index=False)

    print("Saved:")
    print(" -", out_dir / f"{args.run_tag}_classification_report.txt")
    print(" -", out_dir / f"{args.run_tag}_classification_report.json")
    print(" -", out_dir / f"{args.run_tag}_confusion_matrix.csv")
    print(" -", out_dir / f"{args.run_tag}_confusion_matrix_norm.png")
    print(" -", out_dir / f"{args.run_tag}_predictions.csv")

    # -----------------------------
    # Save failed cases
    # -----------------------------
    if args.save_viz:
        viz_dir = args.viz_out if args.viz_out is not None else (out_dir / "viz")
        save_debug_crops(
            df=test_df,
            y_true=y_true,
            y_pred=y_pred,
            probs=probs,
            class_names=class_names,
            images_dir=test_img_dir,
            out_dir=viz_dir,
            pad_ratio=args.pad_ratio,
            only_mistakes=args.viz_only_mistakes,
            max_save=args.viz_max,
            font_size=args.viz_font_size,
        )


if __name__ == "__main__":
    main()

"""
python dinov3_angles_domain/dinov3_evaluation.py \
  --data-root dataset/multiview_pig_posture_recognition_latest \
  --train-set 2 \
  --test-csv test2_gt.csv \
  --test-images test2_images \
  --ckpt runs/dinov3_angles_domain/angle_domain_used_in_paper/best_dinov3_t1_angle_domain.pt \
  --dinov3-weight facebook/dinov2-base \
  --img-size 224 \
  --pad-ratio 0.03 \
  --batch 64 --workers 4 \
  --use-amp \
  --out-dir runs/dinov3_angles_domain/final/angle_domain_used_in_paper/eval_test2 \
  --save-viz \
  --viz-only-mistakes \
  --viz-max 3000
"""