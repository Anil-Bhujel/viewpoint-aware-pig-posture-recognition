#!/usr/bin/env python3
from __future__ import annotations

import time
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, confusion_matrix, classification_report

import matplotlib.pyplot as plt
import seaborn as sns


# -------------------------------------------------
# Basic val F1 (no TTA)
# -------------------------------------------------
@torch.no_grad()
def evaluate_f1(model, loader, device):
    model.eval()
    ys, ps = [], []
    for batch in loader:
        # Support both (x, y) and (x, angles, y) batches
        if len(batch) == 3:
            x, angles, y = batch[0], batch[1], batch[2]
            x = x.to(device, non_blocking=True)
            angles = angles.to(device, non_blocking=True)
            out = model(x, angles)
        else:
            x, y = batch[0], batch[1]
            x = x.to(device, non_blocking=True)
            out = model(x, None) if hasattr(model, "use_angles") else model(x)
        # DinoV3AngleMlpHead returns (posture_logits, domain_logits)
        logits = out[0] if isinstance(out, tuple) else out
        pred = logits.argmax(1).detach().cpu().numpy()
        ps.append(pred)
        ys.append(y.detach().cpu().numpy())
    return f1_score(np.concatenate(ys), np.concatenate(ps), average="macro")


# -------------------------------------------------
# Helper: build permutation for flip label swap
# -------------------------------------------------
def _build_flip_permutation(num_classes: int, flip_swap_pairs: List[Tuple[int, int]]) -> torch.Tensor:
    """
    Given pairs like [(0,1)] (left/right), build a permutation index
    that swaps those classes for the flipped logits.
    """
    perm = list(range(num_classes))
    for a, b in flip_swap_pairs:
        if 0 <= a < num_classes and 0 <= b < num_classes:
            perm[a], perm[b] = perm[b], perm[a]
    return torch.tensor(perm, dtype=torch.long)


# -------------------------------------------------
# Prediction with optional TTA
# -------------------------------------------------
@torch.no_grad()
def predict_labels(
    model,
    loader,
    device,
    use_amp: bool,
    use_tta: bool = False,
    flip_swap_pairs: Optional[List[Tuple[int, int]]] = None,
    tta_multi: bool = False,
):
    """
    use_tta:    2-view TTA — avg(original, hflip).
    tta_multi:  4-view TTA — avg(original, hflip, vflip, rot180).
                Implies use_tta.
    flip_swap_pairs: e.g. [(0,1)] swaps L/R class labels for hflip & rot180 branches.
    """
    if tta_multi:
        use_tta = True
    model.eval()
    preds = []

    amp_enabled = (use_amp and device.type == "cuda")
    flip_swap_pairs = flip_swap_pairs or []

    # we build perm lazily on first batch (when num_classes is known)
    perm_idx = None

    def _forward(img, ang):
        out = model(img, ang) if ang is not None else (
            model(img, None) if hasattr(model, "use_angles") else model(img)
        )
        # DinoAngleMlpHead returns (posture_logits, domain_logits); take first element
        if isinstance(out, tuple):
            return out[0]
        return out

    for batch in loader:
        x = batch[0]
        x = x.to(device, non_blocking=True)

        # Extract angles if batch is a 3-tuple (x, angles, label)
        angles = None
        if len(batch) == 3:
            angles = batch[1].to(device, non_blocking=True)

        with torch.amp.autocast(device_type=("cuda" if device.type == "cuda" else "cpu"),
                                 enabled=amp_enabled):
            logits1 = _forward(x, angles)

            if use_tta:
                # --- hflip branch ---
                x_h = torch.flip(x, dims=[3])
                ang_h = None
                if angles is not None:
                    ang_h = angles.clone()
                    ang_h[:, 0] = -ang_h[:, 0]   # az_sin negated
                logits_h = _forward(x_h, ang_h)

                if flip_swap_pairs:
                    if perm_idx is None:
                        num_classes = logits1.shape[1]
                        perm_idx = _build_flip_permutation(num_classes, flip_swap_pairs).to(device)
                    logits_h = logits_h[:, perm_idx]

                if tta_multi:
                    # --- vflip branch (no class swap needed for posture) ---
                    x_v = torch.flip(x, dims=[2])
                    ang_v = None
                    if angles is not None:
                        ang_v = angles.clone()
                        ang_v[:, 1] = -ang_v[:, 1]   # az_cos negated for vflip
                    logits_v = _forward(x_v, ang_v)

                    # --- rot180 branch = hflip + vflip (L/R swap same as hflip) ---
                    x_r180 = torch.flip(x, dims=[2, 3])
                    ang_r180 = None
                    if angles is not None:
                        ang_r180 = angles.clone()
                        ang_r180[:, 0] = -ang_r180[:, 0]   # hflip component
                        ang_r180[:, 1] = -ang_r180[:, 1]   # vflip component
                    logits_r180 = _forward(x_r180, ang_r180)
                    if flip_swap_pairs:
                        logits_r180 = logits_r180[:, perm_idx]  # same L/R swap as hflip

                    logits = 0.25 * (logits1 + logits_h + logits_v + logits_r180)
                else:
                    logits = 0.5 * (logits1 + logits_h)
            else:
                logits = logits1

        preds.append(logits.argmax(1).detach().cpu().numpy())

    return np.concatenate(preds)


# -------------------------------------------------
# Confusion matrix plotting
# -------------------------------------------------
def plot_confmat(cm_counts, class_names, title, out_png: Path, normalize=True,
                  title_fontsize=16, label_fontsize=14, tick_fontsize=12, annot_fontsize=11):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cm_counts = np.asarray(cm_counts, dtype=np.float64)
    support = cm_counts.sum(axis=1).astype(int)

    if normalize:
        denom = np.maximum(support.reshape(-1, 1), 1e-9)
        cm_norm = cm_counts / denom
    else:
        cm_norm = cm_counts

    annot = np.empty_like(cm_norm).astype(object)
    for i in range(cm_counts.shape[0]):
        for j in range(cm_counts.shape[1]):
            annot[i, j] = f"{cm_norm[i, j]:.2f}\n({int(cm_counts[i, j])})"

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        cm_norm, annot=annot, fmt="", cmap="Blues", cbar=True,
        xticklabels=class_names, yticklabels=class_names,
        vmin=0.0, vmax=1.0, ax=ax, annot_kws={"fontsize": annot_fontsize}
    )
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel("Predicted", fontsize=label_fontsize)
    ax.set_ylabel("Ground truth", fontsize=label_fontsize)
    ax.tick_params(axis="x", labelsize=tick_fontsize)
    ax.tick_params(axis="y", labelsize=tick_fontsize)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    # right-side GT count axis
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(np.arange(len(class_names)) + 0.5)
    ax2.set_yticklabels(support, fontsize=tick_fontsize)
    ax2.set_ylabel("GT count", fontsize=label_fontsize)

    fig.tight_layout()
    fig.savefig(out_png, dpi=250, bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------
# History save
# -------------------------------------------------
def save_history(out_dir: Path, run_tag: str, history: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    hist_df = pd.DataFrame(history)
    csv_path = out_dir / f"history{run_tag}.csv"
    val_png_path = out_dir / f"val_f1_history{run_tag}.png"

    hist_df.to_csv(csv_path, index=False)

    print("Saved history:", csv_path)

    fig = plt.figure(figsize=(10, 5))
    # plt.plot(hist_df["epoch"], hist_df["train_loss"], label="train_loss")
    plt.plot(hist_df["epoch"], hist_df["val_f1"], label="val_macro_f1")
    plt.xlabel("epoch")
    plt.legend()
    plt.tight_layout()
    fig.savefig(val_png_path, dpi=200)
    plt.close(fig)

    print("Saved plot:", val_png_path)

    train_loss_png_path = out_dir / f"train_loss_history{run_tag}.png"

    fig = plt.figure(figsize=(10, 5))
    plt.plot(hist_df["epoch"], hist_df["train_loss"], label="train_loss")
    # plt.plot(hist_df["epoch"], hist_df["val_f1"], label="val_macro_f1")
    plt.xlabel("epoch")
    plt.legend()
    plt.tight_layout()
    fig.savefig(train_loss_png_path, dpi=200)
    plt.close(fig)

    print("Saved plot:", train_loss_png_path)
    
# def save_history(out_dir: Path, run_tag: str, history: dict):
#     out_dir.mkdir(parents=True, exist_ok=True)
#     hist_df = pd.DataFrame(history)
#     csv_path = out_dir / f"history{run_tag}.csv"
#     png_path = out_dir / f"history{run_tag}.png"

#     hist_df.to_csv(csv_path, index=False)

#     fig = plt.figure(figsize=(10, 5))
#     plt.plot(hist_df["epoch"], hist_df["train_loss"], label="train_loss")
#     plt.plot(hist_df["epoch"], hist_df["val_f1"], label="val_macro_f1")
#     plt.xlabel("epoch")
#     plt.legend()
#     plt.tight_layout()
#     fig.savefig(png_path, dpi=200)
#     plt.close(fig)

#     print("Saved history:", csv_path)
#     print("Saved plot:", png_path)


# -------------------------------------------------
# Test evaluation (Kaggle-style) with optional TTA
# -------------------------------------------------
def evaluate_test_and_save(
    model,
    test_df,
    ds_test,
    batch_size,
    num_workers,
    device,
    use_amp,
    class_names,
    out_dir: Path,
    run_tag: str,
    use_tta: bool = False,
    flip_swap_pairs: Optional[List[Tuple[int, int]]] = None,
    tta_multi: bool = False,
):
    dl_te = DataLoader(
        ds_test, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0)
    )

    y_true = test_df["class_id"].astype(int).values
    y_pred = predict_labels(
        model=model,
        loader=dl_te,
        device=device,
        use_amp=use_amp,
        use_tta=use_tta,
        flip_swap_pairs=flip_swap_pairs,
        tta_multi=tta_multi,
    )

    test_f1 = f1_score(y_true, y_pred, average="macro")
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    rep = classification_report(
        y_true, y_pred, labels=labels, target_names=class_names,
        digits=4, zero_division=0
    )

    stamp = time.strftime("%Y%m%d_%H%M%S")
    base = f"{run_tag}_{stamp}" if run_tag else stamp

    out_dir.mkdir(parents=True, exist_ok=True)
    rpt_path = out_dir / f"{base}_test_report.txt"
    rpt_path.write_text(rep)

    cm_path = out_dir / f"{base}_test_confmat.png"
    plot_confmat(cm, class_names, title=f"TEST | F1={test_f1:.4f}", out_png=cm_path)

    print("\n=== TEST EVAL ===")
    print("TEST macro-F1:", float(test_f1))
    print("Saved report:", rpt_path)
    print("Saved confusion matrix:", cm_path)
    return float(test_f1)
