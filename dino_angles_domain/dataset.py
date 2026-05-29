# dataset.py
from __future__ import annotations
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset

from utils import parse_bbox_string, class_prob_for


# =============================================================================
# Angle transformation helpers
# =============================================================================
# Azimuth convention: az = atan2(dX, dY),  0°=+Y straight-ahead, +90°=+X(left).
# Elevation depends only on horizontal range r and z-offset → does NOT change
# under any in-plane image rotation or flip.

def _transform_angles_hflip(az_sin, az_cos, el_sin, el_cos):
    """Horizontal flip: dX → -dX  =>  az → -az.
    sin(-az)=-sin(az),  cos(-az)=cos(az)
    """
    return -az_sin, az_cos, el_sin, el_cos


def _transform_angles_vflip(az_sin, az_cos, el_sin, el_cos):
    """Vertical flip: dY → -dY  =>  az = atan2(dX,-dY).
    sin(new) = dX/r = sin(az)  [unchanged]
    cos(new) =-dY/r =-cos(az)
    """
    return az_sin, -az_cos, el_sin, el_cos


def _transform_angles_rot90cw(az_sin, az_cos, el_sin, el_cos):
    """90° clockwise image rotation  =>  az → az - 90°.
    sin(az-90°)=-cos(az),  cos(az-90°)=sin(az)
    """
    return -az_cos, az_sin, el_sin, el_cos


def _transform_angles_rot90ccw(az_sin, az_cos, el_sin, el_cos):
    """90° counter-clockwise image rotation  =>  az → az + 90°.
    sin(az+90°)=cos(az),  cos(az+90°)=-sin(az)
    """
    return az_cos, -az_sin, el_sin, el_cos


def _transform_angles_rot180(az_sin, az_cos, el_sin, el_cos):
    """180° rotation  =>  az → az + 180°.
    sin(az+180°)=-sin(az),  cos(az+180°)=-cos(az)
    """
    return -az_sin, -az_cos, el_sin, el_cos


def _transform_angles_rot_arbitrary(az_sin, az_cos, el_sin, el_cos, theta_deg):
    """Arbitrary CCW rotation by theta_deg  =>  az → az + theta.
    Using PIL rotate(theta_deg) which rotates CCW:
        sin(az+θ) = az_sin·cos(θ) + az_cos·sin(θ)
        cos(az+θ) = az_cos·cos(θ) - az_sin·sin(θ)
    Elevation is unaffected by in-plane rotation.
    """
    import math
    theta = math.radians(theta_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    new_az_sin = az_sin * cos_t + az_cos * sin_t
    new_az_cos = az_cos * cos_t - az_sin * sin_t
    return new_az_sin, new_az_cos, el_sin, el_cos


# =============================================================================
# Geometric helpers
# =============================================================================

def _clip(val, lo, hi):
    return max(lo, min(hi, val))


def bbox_xywh_to_square_xyxy(x, y, w, h, img_w, img_h):
    """Convert xywh bbox to a square xyxy bbox centred at bbox centre."""
    cx = x + w * 0.5
    cy = y + h * 0.5
    side = max(w, h)
    x1 = _clip(cx - side * 0.5, 0, img_w - 1)
    y1 = _clip(cy - side * 0.5, 0, img_h - 1)
    x2 = _clip(cx + side * 0.5, 1, img_w)
    y2 = _clip(cy + side * 0.5, 1, img_h)
    if x2 <= x1 + 1:
        x2 = min(img_w, x1 + 2)
    if y2 <= y1 + 1:
        y2 = min(img_h, y1 + 2)
    return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))


# Default angle columns produced by add_camera_angles_multicam.py
_DEFAULT_ANGLE_COLS = [
    "azimuth_sin", "azimuth_cos",
    "elevation_sin", "elevation_cos",
]


# =============================================================================
# Dataset
# =============================================================================

class PigCropDataset(Dataset):
    """
    Crops pig bboxes, applies augmentation, optionally returns angle features.

    Returns (train / val):
        use_angles=False : (img_tensor, label)
        use_angles=True  : (img_tensor, angle_tensor[4], label)

    Returns (test):
        use_angles=False : (img_tensor, label_or_row_id)
        use_angles=True  : (img_tensor, angle_tensor[4], label_or_row_id)

    angle_tensor = float32 [az_sin, az_cos, el_sin, el_cos].
    All zeros when angle_valid == 0 or columns are missing.

    Geometric augmentation order (train only):
        1. 90° rotation    (rot90_prob controls frequency; uniform over 90/180/270)
        2. Horizontal flip (hflip_prob / class_hflip_probs)
        3. Vertical flip   (vflip_prob / class_vflip_probs)
    All geometric transforms correctly update the angle tensor.

    """

    def __init__(
        self,
        df: pd.DataFrame,
        mode: str,
        train_img_dir: Path,
        test_img_dir: Path,
        pad_ratio: float,
        train_tfms=None,
        val_tfms=None,
        # --- flip augmentation ---
        hflip_prob: float = 0.0,
        vflip_prob: float = 0.0,
        class_hflip_probs: Optional[List[float]] = None,
        class_vflip_probs: Optional[List[float]] = None,
        flip_swap_map: Optional[Dict[int, int]] = None,
        swap_on: str = "h",          # h | v | both | none
        class_id_offset: int = 0,
        # --- 90-degree rotation augmentation ---
        rot90_prob: float = 0.0,
        # --- continuous random rotation augmentation ---
        rand_rot_prob: float = 0.0,
        rand_rot_max_deg: float = 30.0,
        # --- domain ---
        return_domain: bool = False,
        domain_col: str = "domain_id",
        # --- square crop ---
        square_crop: bool = False,
        # --- angle features (ablation) ---
        use_angles: bool = False,
        angle_cols: Optional[List[str]] = None,
        angle_valid_col: str = "angle_valid",
       
    ):
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.pad_ratio = float(pad_ratio)
        self.train_img_dir = Path(train_img_dir)
        self.test_img_dir = Path(test_img_dir)
        self.train_tfms = train_tfms
        self.val_tfms = val_tfms
        self.square_crop = bool(square_crop)

        self.hflip_prob = float(hflip_prob)
        self.vflip_prob = float(vflip_prob)
        self.class_hflip_probs = class_hflip_probs
        self.class_vflip_probs = class_vflip_probs
        self.flip_swap_map = flip_swap_map or {}
        self.swap_on = swap_on
        self.class_id_offset = int(class_id_offset)
        self.rot90_prob = float(rot90_prob)
        self.rand_rot_prob = float(rand_rot_prob)
        self.rand_rot_max_deg = float(rand_rot_max_deg)

        self.return_domain = bool(return_domain)
        self.domain_col = str(domain_col)

        self.use_angles = bool(use_angles)
        self.angle_cols = angle_cols if angle_cols is not None else _DEFAULT_ANGLE_COLS
        self.angle_valid_col = angle_valid_col
        self._has_angle_cols = (
            self.use_angles
            and all(c in df.columns for c in self.angle_cols)
        )
        if self.use_angles and not self._has_angle_cols:
            missing = [c for c in self.angle_cols if c not in df.columns]
            raise ValueError(
                f"use_angles=True but columns not found in DataFrame: {missing}\n"
                f"  -> Pass train1_with_angles.csv (from add_camera_angles_multicam.py)."
            )

        # ImageNet mean in uint8 format for PIL rotation fill color
        # IMAGENET_MEAN = (0.485, 0.456, 0.406), convert to 0-255 range
        self._imagenet_mean_u8 = np.array([0.485, 0.456, 0.406]) * 255.0
        self._imagenet_mean_u8 = self._imagenet_mean_u8.astype(np.uint8)

        
    def __len__(self):
        return len(self.df)



    # ------------------------------------------------------------------
    def _maybe_swap_label(self, y: int, mode: str) -> int:
        if self.swap_on == "none":
            return y
        if self.swap_on == "both" and mode in ("h", "v"):
            return self.flip_swap_map.get(y, y)
        if self.swap_on == mode:
            return self.flip_swap_map.get(y, y)
        return y

    def _sample_flip(self, y: int) -> Tuple[bool, bool]:
        ph = (class_prob_for(y, self.class_hflip_probs, self.class_id_offset)
              if self.class_hflip_probs else None)
        pv = (class_prob_for(y, self.class_vflip_probs, self.class_id_offset)
              if self.class_vflip_probs else None)
        if ph is None:
            ph = self.hflip_prob
        if pv is None:
            pv = self.vflip_prob
        do_h = (ph > 0) and (random.random() < ph)
        do_v = (pv > 0) and (random.random() < pv)
        return do_h, do_v

    def _read_angles(self, r) -> Tuple[float, float, float, float, bool]:
        """Return (az_sin, az_cos, el_sin, el_cos, is_valid)."""
        if not self._has_angle_cols:
            return 0.0, 0.0, 0.0, 0.0, False
        valid_col = self.angle_valid_col
        if valid_col in r.index:
            is_valid = bool(int(r[valid_col]))
        else:
            is_valid = False
        if not is_valid:
            return 0.0, 0.0, 0.0, 0.0, False
        az_sin = float(r[self.angle_cols[0]])
        az_cos = float(r[self.angle_cols[1]])
        el_sin = float(r[self.angle_cols[2]])
        el_cos = float(r[self.angle_cols[3]])
        if any(math.isnan(v) for v in (az_sin, az_cos, el_sin, el_cos)):
            return 0.0, 0.0, 0.0, 0.0, False
        return az_sin, az_cos, el_sin, el_cos, True

    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        image_id = r["image_id"]
        bbox = r["bbox"]

        img_path = (self.train_img_dir if self.mode in ("train", "val")
                    else self.test_img_dir) / image_id
        if not img_path.exists():
            raise FileNotFoundError(f"Missing image: {img_path}")

        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size
        x, y, w, h = parse_bbox_string(bbox)

        # ── Crop ────────────────────────────────────────────────────────
        if self.square_crop:
            x1, y1, x2, y2 = bbox_xywh_to_square_xyxy(x, y, w, h, img_w, img_h)
        else:
            # Apply pad_ratio symmetrically on all four sides
            pad = self.pad_ratio * max(w, h)
            x1 = int(round(_clip(x - pad,     0, img_w - 1)))
            y1 = int(round(_clip(y - pad,     0, img_h - 1)))
            x2 = int(round(_clip(x + w + pad, 1, img_w)))
            y2 = int(round(_clip(y + h + pad, 1, img_h)))
            if x2 <= x1 + 1:
                x2 = min(img_w, x1 + 2)
            if y2 <= y1 + 1:
                y2 = min(img_h, y1 + 2)
        crop = img.crop((x1, y1, x2, y2))

       
        # ── Angles ──────────────────────────────────────────────────────
        az_sin, az_cos, el_sin, el_cos, angle_valid = self._read_angles(r)

        # ================================================================
        # TRAIN ── geometric then photometric augmentation
        # ================================================================
        if self.mode == "train":
            y_t = int(r["class_id"])

            # 0. Random continuous rotation (arbitrary angle)
            if self.rand_rot_prob > 0 and random.random() < self.rand_rot_prob:
                theta_deg = random.uniform(-self.rand_rot_max_deg, self.rand_rot_max_deg)
                fill_color = (
                    tuple(self._imagenet_mean_u8.tolist())
                )
                crop = crop.rotate(theta_deg, expand=True,
                                   fillcolor=fill_color,
                                   resample=Image.BILINEAR)
                if angle_valid:
                    az_sin, az_cos, el_sin, el_cos = _transform_angles_rot_arbitrary(
                        az_sin, az_cos, el_sin, el_cos, theta_deg)

            # 1. Random 90-degree rotation
            if self.rot90_prob > 0 and random.random() < self.rot90_prob:
                k = random.randint(1, 3)   # 1=90CW, 2=180, 3=90CCW
                if k == 1:
                    crop = crop.transpose(Image.ROTATE_270)   # 90 CW
                    if angle_valid:
                        az_sin, az_cos, el_sin, el_cos = _transform_angles_rot90cw(
                            az_sin, az_cos, el_sin, el_cos)
                elif k == 2:
                    crop = crop.transpose(Image.ROTATE_180)
                    if angle_valid:
                        az_sin, az_cos, el_sin, el_cos = _transform_angles_rot180(
                            az_sin, az_cos, el_sin, el_cos)
                else:
                    crop = crop.transpose(Image.ROTATE_90)    # 90 CCW
                    if angle_valid:
                        az_sin, az_cos, el_sin, el_cos = _transform_angles_rot90ccw(
                            az_sin, az_cos, el_sin, el_cos)

            # 2. Horizontal / vertical flip
            do_h, do_v = self._sample_flip(y_t)

            if do_h:
                crop = ImageOps.mirror(crop)
                y_t = self._maybe_swap_label(y_t, "h")
                if angle_valid:
                    az_sin, az_cos, el_sin, el_cos = _transform_angles_hflip(
                        az_sin, az_cos, el_sin, el_cos)

            if do_v:
                crop = ImageOps.flip(crop)
                y_t = self._maybe_swap_label(y_t, "v")
                if angle_valid:
                    az_sin, az_cos, el_sin, el_cos = _transform_angles_vflip(
                        az_sin, az_cos, el_sin, el_cos)

            # 3. Photometric / tensor transforms
            x_t = self.train_tfms(crop)

            if self.use_angles:
                angle_t = torch.tensor(
                    [az_sin, az_cos, el_sin, el_cos], dtype=torch.float32)
                if self.return_domain:
                    return x_t, angle_t, y_t, int(r.get(self.domain_col, -1))
                return x_t, angle_t, y_t
            else:
                if self.return_domain:
                    return x_t, y_t, int(r.get(self.domain_col, -1))
                return x_t, y_t

        # ================================================================
        # VAL / TEST ── no geometric augmentation
        # ================================================================
        x_t = self.val_tfms(crop)
        angle_t = (torch.tensor([az_sin, az_cos, el_sin, el_cos], dtype=torch.float32)
                   if self.use_angles else None)

        if self.mode == "val":
            y_t = int(r["class_id"])
            if self.use_angles:
                if self.return_domain:
                    return x_t, angle_t, y_t, int(r.get(self.domain_col, -1))
                return x_t, angle_t, y_t
            else:
                if self.return_domain:
                    return x_t, y_t, int(r.get(self.domain_col, -1))
                return x_t, y_t

        # test
        label = (int(r["class_id"])
                 if ("class_id" in r.index and pd.notna(r.get("class_id")))
                 else str(r["row_id"]))
        return (x_t, angle_t, label) if self.use_angles else (x_t, label)
