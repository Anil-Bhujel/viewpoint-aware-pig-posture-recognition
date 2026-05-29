# utils.py
from __future__ import annotations
import os, re, random, math, json
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
from PIL import Image, ImageOps


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import torch.backends.cudnn as cudnn
    cudnn.deterministic = False
    cudnn.benchmark = True


def parse_bbox_string(bbox_str: str) -> tuple[float, float, float, float]:
    s = str(bbox_str).strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1].strip()
    s = s.replace(",", " ")
    parts = [p for p in s.split() if p]
    if len(parts) != 4:
        raise ValueError(f"Bad bbox: {bbox_str}")
    x, y, w, h = map(float, parts)
    return x, y, w, h


def image_group_id(image_id: str) -> str:
    return os.path.splitext(str(image_id))[0]


def pad_to_square(pil_img: Image.Image, size: int) -> Image.Image:
    return ImageOps.pad(pil_img, (size, size), method=Image.BICUBIC, color=(0, 0, 0))


def _normalize_cam_name(cam: str) -> str:
    cam = cam.lower()
    if cam in ("orbbec", "orb"):
        return "orb"
    if cam in ("turret", "tur"):
        return "tur"
    return cam


def parse_camera_id(image_id: str) -> str:
    s = str(image_id)
    base = os.path.basename(s)
    m = re.search(r"(pen\d+)_([a-zA-Z]+)_cam(\d+)_\d{8}_\d{6}", base)
    if not m:
        m2 = re.search(r"(pen\d+)_([a-zA-Z]+)_cam(\d+)_", base)
        if not m2:
            return "unk"
        pen, cam, num = m2.group(1), m2.group(2), m2.group(3)
    else:
        pen, cam, num = m.group(1), m.group(2), m.group(3)

    cam = _normalize_cam_name(cam)
    return f"{pen}_{cam}_cam{num}"


def split_csv_list(s: Optional[str]) -> list[str]:
    if s is None:
        return []
    s = str(s).strip()
    if not s:
        return []
    parts = re.split(r"[,\s]+", s)
    return [p for p in parts if p]


def parse_prob_list(s: str) -> Optional[List[float]]:
    s = str(s).strip()
    if not s:
        return None
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    return [float(p) for p in parts]


def parse_swap_map(s: str) -> Dict[int, int]:
    d = json.loads(s)
    return {int(k): int(v) for k, v in d.items()}


def class_prob_for(cid: int, probs: Optional[List[float]], class_id_offset: int) -> Optional[float]:
    if probs is None:
        return None
    idx = cid - int(class_id_offset)
    if idx < 0 or idx >= len(probs):
        raise ValueError(
            f"class_id={cid} offset={class_id_offset} -> idx={idx} out of range for prob list length {len(probs)}"
        )
    return float(probs[idx])
