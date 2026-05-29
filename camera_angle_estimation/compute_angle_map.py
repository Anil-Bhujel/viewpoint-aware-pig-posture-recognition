"""
compute_angle_map.py
--------------------
Dense per-pixel azimuth and elevation angle maps for the pen floor.

For each pixel (u, v):
  1. Back-project through K^-1 → ray in camera space
  2. Rotate to world space via R^T
  3. Intersect with the floor plane  z = plane_z (default 0)
  4. Clamp to pen floor bounds  [0, W] × [0, H]
  5. Compute:
       azimuth   = atan2(Yw − cam_y,  Xw − cam_x)   [0 = +X axis, CCW positive]
       elevation = atan2(cam_z − plane_z,  r)           [positive = overhead]

Outputs (in --out-dir):
    azimuth_map.npy        float32, degrees, NaN outside floor
    elevation_map.npy      float32, degrees, NaN outside floor
    valid_mask.npy         bool, True where floor ray is valid
    azimuth_vis.jpg        HSV-coloured overlay
    elevation_vis.jpg      viridis-coloured overlay
    angle_stats.json       summary statistics

Usage:
    python compute_angle_map.py \
        --image  frame.jpg \
        --camera camera_params.json \
        --out-dir angle_maps/

    # Also label individual pig bboxes from a CSV
    python compute_angle_map.py \
        --image  frame.jpg \
        --camera camera_params.json \
        --detections pigs.csv \
        --bbox-col bbox \
        --out-dir angle_maps/
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from estimate_camera import load_camera_params

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


# ── Core computation ──────────────────────────────────────────────────────────

def compute_angle_map(
    img_h: int,
    img_w: int,
    K: np.ndarray,
    R: np.ndarray,
    cam_pos: np.ndarray,
    floor_w: float,
    floor_h: float,
    plane_z: float = 0.0,
    elevation_convention: str = "elevation",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute dense azimuth and elevation maps for every pixel.

    Parameters
    ----------
    elevation_convention : str
        "elevation" (default): 0° = horizon, 90° = nadir (directly under camera)
        "zenith": 0° = nadir, 90° = horizon

    Returns
    -------
    az_map     : float32 (H, W) degrees, NaN where not on floor
    el_map     : float32 (H, W) degrees, NaN where not on floor
    valid_mask : bool    (H, W)
    """
    K_inv = np.linalg.inv(K)
    C = cam_pos.flatten()

    # Pixel grid  (centre of each pixel)
    u_grid = np.arange(img_w, dtype=np.float64)
    v_grid = np.arange(img_h, dtype=np.float64)
    uu, vv = np.meshgrid(u_grid, v_grid)   # (H, W)

    # Flatten → (H*W, 3) homogeneous pixels
    ones  = np.ones(img_h * img_w, dtype=np.float64)
    pixels_h = np.stack([uu.ravel(), vv.ravel(), ones], axis=1)   # (N, 3)

    # Rays in camera space:  d_cam = K^-1 p
    rays_cam = (K_inv @ pixels_h.T).T    # (N, 3)

    # Rays in world space:  d_world = R^T d_cam
    rays_world = (R.T @ rays_cam.T).T    # (N, 3)

    # ── Floor plane intersection  z = plane_z ─────────────────────────────────
    dz = rays_world[:, 2]
    # valid: ray not parallel to floor AND t > 0 (forward intersection)
    valid = (np.abs(dz) > 1e-9)
    t = np.where(valid, (plane_z - C[2]) / dz, np.nan)
    valid = valid & (t > 0)

    # World intersection points
    X = C[0] + t * rays_world[:, 0]
    Y = C[1] + t * rays_world[:, 1]

    # ── Clamp to pen floor  [0, floor_w] × [0, floor_h] ─────────────────────
    valid = valid & (X >= 0) & (X <= floor_w) & (Y >= 0) & (Y <= floor_h)

    # ── Azimuth  (standard math convention: 0 = +X axis, CCW positive) ──────────
    # atan2(dy, dx) — same as numpy/math convention.
    # 0°   = floor point directly along +X (right)
    # +90° = floor point along +Y (far wall direction)
    # NOTE: The GT turret precompute script uses a RELATIVE azimuth measured
    # from camera look-at direction (atan2(cross,dot)).  Scripts that compare
    # directly against *_azim_undist.npy should account for this offset.
    dx = X - C[0]
    dy = Y - C[1]
    az_rad  = np.arctan2(dy, dx)

    # ── Elevation ─────────────────────────────────────────────────────────────
    r = np.sqrt(dx**2 + dy**2)
    el_rad  = np.arctan2(C[2] - plane_z, r)

    az_deg  = np.where(valid, np.degrees(az_rad),  np.nan).astype(np.float32)
    el_deg  = np.where(valid, np.degrees(el_rad),  np.nan).astype(np.float32)

    # Convert elevation to zenith angle if requested
    if elevation_convention == "zenith":
        el_deg = np.where(np.isfinite(el_deg), 90.0 - el_deg, el_deg)

    az_map     = az_deg.reshape(img_h, img_w)
    el_map     = el_deg.reshape(img_h, img_w)
    valid_mask = valid.reshape(img_h, img_w)

    return az_map, el_map, valid_mask


def angles_at_pixel(
    u: float, v: float,
    K: np.ndarray,
    R: np.ndarray,
    cam_pos: np.ndarray,
    plane_z: float = 0.0,
    elevation_convention: str = "elevation",
) -> tuple[float, float, bool]:
    """
    Return (azimuth_deg, elevation_deg, valid) for a single pixel.
    Useful for per-bbox angle computation.
    
    Parameters
    ----------
    elevation_convention : str
        "elevation" (default): 0° = horizon, 90° = nadir
        "zenith": 0° = nadir, 90° = horizon
    """
    K_inv = np.linalg.inv(K)
    C = cam_pos.flatten()
    pixel_h = np.array([u, v, 1.0])
    ray_cam = K_inv @ pixel_h
    ray_world = R.T @ ray_cam
    dz = ray_world[2]
    if abs(dz) < 1e-9:
        return 0.0, 0.0, False
    t = (plane_z - C[2]) / dz
    if t <= 0:
        return 0.0, 0.0, False
    X = C[0] + t * ray_world[0]
    Y = C[1] + t * ray_world[1]
    dx = X - C[0]
    dy = Y - C[1]
    az_deg  = math.degrees(math.atan2(dy, dx))
    r       = math.hypot(dx, dy)
    el_deg  = math.degrees(math.atan2(C[2] - plane_z, r))
    
    # Convert to zenith angle if requested
    if elevation_convention == "zenith":
        el_deg = 90.0 - el_deg
    
    return az_deg, el_deg, True


# ── Visualisation ─────────────────────────────────────────────────────────────

def _colorize_map(values: np.ndarray, vmin: float, vmax: float,
                  colormap: str = "hsv") -> np.ndarray:
    """
    Convert a (H,W) float map with NaNs into a BGR uint8 image.
    """
    normed = np.where(
        np.isfinite(values),
        np.clip((values - vmin) / (vmax - vmin + 1e-9), 0, 1),
        -1.0,
    ).astype(np.float32)

    if _HAS_MPL:
        cmap = cm.get_cmap(colormap)
        rgba = cmap(normed)                 # (H, W, 4) float 0–1
        rgb  = (rgba[..., :3] * 255).astype(np.uint8)
        # Mask NaN pixels to black
        rgb[normed < 0] = 0
        bgr = rgb[..., ::-1]               # RGB → BGR for cv2
    else:
        # Fallback: HSV using OpenCV
        h_ch = np.where(normed >= 0, (normed * 120).astype(np.uint8), 0)
        s_ch = np.where(normed >= 0, np.uint8(220), np.uint8(0))
        v_ch = np.where(normed >= 0, np.uint8(230), np.uint8(0))
        hsv  = np.stack([h_ch, s_ch, v_ch], axis=-1)
        bgr  = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return bgr


def make_overlay(
    img_bgr: np.ndarray,
    angle_map: np.ndarray,
    vmin: float,
    vmax: float,
    colormap: str,
    alpha: float = 0.55,
    colorbar_label: str = "",
) -> np.ndarray:
    """Blend a colourised angle map over the original image."""
    color = _colorize_map(angle_map, vmin, vmax, colormap)
    color_resized = cv2.resize(color, (img_bgr.shape[1], img_bgr.shape[0]))

    valid = np.isfinite(angle_map).astype(np.float32)
    valid_full = cv2.resize(valid, (img_bgr.shape[1], img_bgr.shape[0]))
    mask = valid_full[..., np.newaxis]

    overlay = img_bgr.copy().astype(np.float32)
    overlay = overlay * (1 - alpha * mask) + color_resized.astype(np.float32) * alpha * mask
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    # Colour bar (right edge)
    if _HAS_MPL:
        _add_colorbar(overlay, vmin, vmax, colormap, colorbar_label)

    return overlay


def _add_colorbar(img: np.ndarray, vmin: float, vmax: float,
                  colormap: str, label: str):
    """Draw a vertical colour bar on the right edge of img (in-place)."""
    h, w = img.shape[:2]
    bar_w = 20
    for row in range(h):
        t = 1.0 - row / h               # top = high value
        if _HAS_MPL:
            cmap = cm.get_cmap(colormap)
            r, g, b, _ = cmap(t)
            bgr = (int(b * 255), int(g * 255), int(r * 255))
        else:
            bgr = (int(t * 180), int(t * 180), int((1 - t) * 180))
        img[row, max(0, w - bar_w):w] = bgr

    # Labels
    def _label(val, row):
        cv2.putText(img, f"{val:.0f}", (w - bar_w - 50, row),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    _label(vmax, 14)
    _label((vmin + vmax) / 2, h // 2)
    _label(vmin, h - 6)
    if label:
        cv2.putText(img, label, (w - bar_w - 10, h // 2 - 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                    cv2.LINE_AA)


# ── Per-bbox annotation ───────────────────────────────────────────────────────

def augment_detections_csv(
    csv_path: Path,
    out_path: Path,
    K: np.ndarray,
    R: np.ndarray,
    cam_pos: np.ndarray,
    bbox_col: str = "bbox",
    sample_mode: str = "lower_center",
    plane_z: float = 0.0,
    elevation_convention: str = "elevation",
):
    """
    Add azimuth_deg and elevation_deg columns to a detections CSV.

    sample_mode : 'center'       → (cx, cy)
                  'lower_center' → (cx, cy + 0.15*h) — matches posture model
    """
    import ast
    import csv

    rows_in = list(csv.DictReader(csv_path.read_text().splitlines()))
    if not rows_in:
        print("[WARN] Empty CSV")
        return

    fieldnames = list(rows_in[0].keys()) + ["azimuth_deg", "elevation_deg", "angle_valid"]

    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_in:
            try:
                bbox = list(map(float, ast.literal_eval(row[bbox_col])))
                bx, by, bw, bh = bbox[:4]
                if sample_mode == "lower_center":
                    su = bx + 0.5 * bw
                    sv = by + 0.65 * bh
                else:
                    su = bx + 0.5 * bw
                    sv = by + 0.5 * bh
                az, el, ok = angles_at_pixel(su, sv, K, R, cam_pos, plane_z, elevation_convention)
                row["azimuth_deg"]   = f"{az:.3f}" if ok else ""
                row["elevation_deg"] = f"{el:.3f}" if ok else ""
                row["angle_valid"]   = int(ok)
            except Exception as exc:
                row["azimuth_deg"]   = ""
                row["elevation_deg"] = ""
                row["angle_valid"]   = 0
                print(f"[WARN] Row skipped ({exc}): {row.get('image_id', '?')}")
            writer.writerow(row)

    print(f"[INFO] Annotated CSV saved -> {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Dense per-pixel azimuth/elevation maps for the pen floor.")
    ap.add_argument("--image",       required=True, help="Input image file")
    ap.add_argument("--camera",      required=True, help="camera_params.json")
    ap.add_argument("--out-dir",     default="angle_maps", help="Output directory")
    ap.add_argument("--plane-z",     type=float, default=0.0,
                    help="Floor plane height in world coords (default 0)")
    ap.add_argument("--alpha",       type=float, default=0.55,
                    help="Overlay transparency (0=full image, 1=full colourmap)")
    ap.add_argument("--detections",  default=None,
                    help="Optional CSV with bbox detections to annotate")
    ap.add_argument("--bbox-col",    default="bbox")
    ap.add_argument("--sample-mode", default="lower_center",
                    choices=["center", "lower_center"])
    ap.add_argument("--elevation-convention", default="elevation",
                    choices=["elevation", "zenith"],
                    help="elevation: 0°=horizon, 90°=nadir (default) | zenith: 0°=nadir, 90°=horizon")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load camera params
    params   = load_camera_params(Path(args.camera))
    K        = params["K"]
    R        = params["R"]
    cam_pos  = params["cam_pos"]
    floor_w  = params["floor_width_m"]
    floor_h  = params["floor_height_m"]

    # Load image
    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {args.image}")
    img_h, img_w = img.shape[:2]

    print(f"[INFO] Computing angle maps for {img_w}x{img_h} image …")
    print(f"[INFO] Elevation convention: {args.elevation_convention}")
    az_map, el_map, valid_mask = compute_angle_map(
        img_h, img_w, K, R, cam_pos, floor_w, floor_h, plane_z=args.plane_z,
        elevation_convention=args.elevation_convention)

    valid_px = int(valid_mask.sum())
    print(f"[INFO] Valid floor pixels: {valid_px} / {img_h * img_w} "
          f"({100 * valid_px / (img_h * img_w):.1f}%)")

    # ── Save arrays ───────────────────────────────────────────────────────────
    np.save(str(out_dir / "azimuth_map.npy"),   az_map)
    np.save(str(out_dir / "elevation_map.npy"), el_map)
    np.save(str(out_dir / "valid_mask.npy"),    valid_mask)

    az_valid = az_map[valid_mask]
    el_valid = el_map[valid_mask]

    stats = {
        "valid_pixels": valid_px,
        "total_pixels": img_h * img_w,
        "azimuth_deg":   {"min": float(az_valid.min()),
                          "max": float(az_valid.max()),
                          "mean": float(az_valid.mean()),
                          "std":  float(az_valid.std())},
        "elevation_deg": {"min": float(el_valid.min()),
                          "max": float(el_valid.max()),
                          "mean": float(el_valid.mean()),
                          "std":  float(el_valid.std())},
    }
    (out_dir / "angle_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"[INFO] Azimuth  {az_valid.min():.1f}..{az_valid.max():.1f} deg "
          f"(mean {az_valid.mean():.1f})")
    print(f"[INFO] Elevation {el_valid.min():.1f}..{el_valid.max():.1f} deg "
          f"(mean {el_valid.mean():.1f})")

    # ── Visualisations ────────────────────────────────────────────────────────
    az_vis = make_overlay(img, az_map, -180, 180, "hsv", args.alpha, "az (deg)")
    cv2.imwrite(str(out_dir / "azimuth_vis.jpg"),   az_vis,
                [cv2.IMWRITE_JPEG_QUALITY, 95])

    el_vis = make_overlay(img, el_map,
                          0.0,
                          90.0,
                          "plasma", args.alpha, "el (deg)")
    cv2.imwrite(str(out_dir / "elevation_vis.jpg"), el_vis,
                [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"[INFO] Visualisations saved -> {out_dir}")

    # ── Per-bbox angles ───────────────────────────────────────────────────────
    if args.detections:
        det_path = Path(args.detections)
        out_csv  = out_dir / (det_path.stem + "_angles.csv")
        augment_detections_csv(det_path, out_csv, K, R, cam_pos,
                               bbox_col=args.bbox_col,
                               sample_mode=args.sample_mode,
                               plane_z=args.plane_z,
                               elevation_convention=args.elevation_convention)


if __name__ == "__main__":
    main()
