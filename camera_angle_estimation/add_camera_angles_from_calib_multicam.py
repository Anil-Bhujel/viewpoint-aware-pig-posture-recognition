#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
add_camera_angles_from_calib_multicam.py
-----------------------------------------
Add camera angles that have a
``pnp_results_path`` in the config it uses a FULLY PnP-based pipeline:

  raw pixel → undistort → K_undist back-project → ray–floor intersection
            → az/el using PnP cam_pos (all in the same normalised world frame)

For cameras WITHOUT ``pnp_results_path`` the script falls back to the
original H + config cam_x/y/z pipeline (identical to original script).

Why fully PnP instead of mixing H with PnP cam_pos?
-------------------------------------------------------
The H map outputs world coordinates in the *config meter system* (Y origin at
one wall).  The PnP pipeline outputs world coordinates in a *normalised system*
(floor_w=1.0, Y origin at the far wall).  Mixing the two coordinate systems
for the camera position would produce wrong azimuth / elevation values.
Using the PnP pipeline end-to-end keeps world points and camera position in
the same coordinate frame, so angles are self-consistent and scale-invariant.

pipeline_calib.py saves camera_params.json with:
  K          – optimal camera matrix after undistortion
  R          – rotation  world → camera  (3×3)
  tvec       – translation  world → camera  (3,)
  cam_pos    – camera position in normalised world  (floor_w=1.0)
  floor_width_m  = 1.0   (normalisation reference)
  floor_height_m – floor depth / floor_width  (≈ 0.49 for these pens)

All angle geometry (az, el) uses normalised coordinates; the result is the
same in either unit system since angles are scale-invariant.

Config JSON extension
---------------------
Per camera, optionally add::

  "pen1_tur_cam1": {
      ...
      "pnp_results_path": "floor_angle_estimation/results/zenith/pen1_tur_cam1_test/camera_params.json",
      "plane_z_m": 0.5
  }

``plane_z_m`` is the pig back height above the real floor (default 0.5 m).
It is converted to normalised units automatically.

Output columns  (same as add_camera_angles_multicam.py + one extra)
----------------------------------------------------------------------
  world_x             pig floor X  [metres if config pipeline; norm*floor_w_m if PnP pipeline]
  world_y             pig floor Y  [same note]
  azimuth_deg         -180..+180  (0°=+X axis, CCW positive)
  elevation_deg       standard signed  (-90..0 for overhead camera)
                      OR zenith angle (0..+90) when --elevation-convention zenith
  elevation_down_deg  positive-down    (0..+90 for overhead camera)  [always stored raw]
  azimuth_sin / azimuth_cos
  elevation_sin / elevation_cos  (in the convention requested by --elevation-convention)
  angle_valid         1=OK  0=outside pen or undistort failed
  cam_pos_source      "pnp" or "config"   ← new column

Elevation conventions
---------------------
  elevation  (default) : standard signed elevation from horizontal
                         –90..0 for overhead cameras; negative = camera above pig
                         elevation_sin is negative for overhead views
  zenith               : angle from the nadir (straight-down) direction
                         0°=directly overhead, 90°=horizontal
                         elevation_sin grows from 0 (nadir) → 1 (horizontal)
                         matches --angle-convention zenith in visualize_viewpoint_angles.py

Usage
-----
  cd /path/to/floor_angle_estimation

  python add_camera_angles_from_calib_multicam.py \\
      --csv          train.csv \\
      --out-csv      train_with_angles_pnp.csv \\
      --camera-config camera_configs/zenith/pen1_camera_config.json \\
                      camera_configs/zenith/pen2_camera_config.json \\
      --sample-mode  center

"""

from __future__ import annotations

import argparse
import ast
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class PenFloor:
    width_m:  float
    height_m: float


@dataclass
class PnPParams:
    """Packed PnP result from pipeline_calib.py camera_params.json."""
    K_undist:     np.ndarray   # optimal camera matrix post-undistortion
    R:            np.ndarray   # world→camera rotation  (3×3)
    tvec:         np.ndarray   # world→camera translation  (3,)
    cam_pos_norm: np.ndarray   # camera position in normalised world  (floor_w=1.0)
    floor_h_norm: float        # floor height in normalised units
    floor_w_m:    float        # real floor width in metres (scale factor)
    plane_z_norm: float        # detection plane Z in normalised units


@dataclass
class CameraModel:
    prefix:       str
    model:        str           # "fisheye" | "pinhole"
    K:            np.ndarray
    D:            np.ndarray
    # Config pipeline fields (used when pnp is None)
    H:            np.ndarray    # undist pixel → (Xw, Yw) metres
    newK:         Optional[np.ndarray]
    img_pts:      Optional[np.ndarray]
    world_pts:    Optional[np.ndarray]
    calib_width:  float
    calib_height: float
    cam_x:        float         # config cam pos, metres
    cam_y:        float
    cam_z:        float
    plane_z:      float         # metres (pig back height above floor, config system)
    pen_floor:    PenFloor
    pen_bounds:   Optional[Tuple[float, float, float, float]] = field(default=None)
    # PnP pipeline fields (used when pnp is not None)
    pnp:          Optional[PnPParams] = field(default=None)
    cam_pos_source: str = "config"
    notes:        str = ""


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Like add_camera_angles_multicam.py but uses PnP-derived camera pose.",
    )
    ap.add_argument("--csv",           required=True)
    ap.add_argument("--out-csv",       required=True)
    ap.add_argument("--camera-config", nargs="+", required=True,
                    help="camera_config JSON files (same format as original script)")

    # CSV columns
    ap.add_argument("--csv-delim",     default=",")
    ap.add_argument("--bbox-col",      default="bbox")
    ap.add_argument("--image-id-col",  default="image_id")
    ap.add_argument("--width-col",     default="width")
    ap.add_argument("--height-col",    default="height")
    ap.add_argument("--sample-mode",
                    choices=["center", "lower_center", "bottom_center"],
                    default="center")
    ap.add_argument("--log-every",     type=int, default=5000)
    # Debug
    ap.add_argument("--image-root",    default=None)
    ap.add_argument("--debug",         action="store_true")
    ap.add_argument("--debug-dir",     default="debug_angles_pnp")
    ap.add_argument("--debug-samples", type=int, default=20)
    ap.add_argument("--debug-seed",    type=int, default=42)
    ap.add_argument("--allow-outside-bounds", action="store_true", default=False,
                    help="Compute angles even when the bbox sample point projects outside "
                         "the calibrated floor region (useful for standing pigs whose bbox "
                         "extends above the floor plane). angle_valid is still 1.")
    ap.add_argument("--elevation-convention", default="zenith",
                    choices=["elevation", "zenith"],
                    help="Convention for elevation_deg / elevation_sin / elevation_cos columns. "
                         "'elevation': standard signed from horizontal (-90..0 for overhead). "
                         "'zenith': angle from nadir (0=directly overhead, 90=horizontal); "
                         "use this when --angle-convention zenith in visualize_viewpoint_angles.py.")
    return ap.parse_args()


# =============================================================================
# Config loading
# =============================================================================

def load_camera_configs(json_paths: List[str]) -> Dict[str, CameraModel]:
    all_cameras: Dict[str, CameraModel] = {}

    for jp in json_paths:
        jp = Path(jp)
        if not jp.is_file():
            raise SystemExit(f"[ERROR] Config not found: {jp}")
        cfg = json.loads(jp.read_text())

        pen_floor = PenFloor(
            width_m  = float(cfg["pen_floor"]["width_m"]),
            height_m = float(cfg["pen_floor"]["height_m"]),
        )

        for prefix, cam_cfg in cfg["cameras"].items():
            if prefix in all_cameras:
                raise SystemExit(f"[ERROR] Duplicate camera prefix: {prefix}")

            if "pen_floor_override" in cam_cfg:
                cam_pen_floor = PenFloor(
                    width_m  = float(cam_cfg["pen_floor_override"]["width_m"]),
                    height_m = float(cam_cfg["pen_floor_override"]["height_m"]),
                )
            else:
                cam_pen_floor = pen_floor

            pnp_path_raw  = cam_cfg.get("pnp_results_path")
            has_homography = "homography_path" in cam_cfg

            if has_homography:
                # Full config pipeline: load K/D + homography + newK
                K, D, newK, H, img_pts, world_pts = _load_calib_and_homography(
                    prefix, cam_cfg["model"].lower(),
                    cam_cfg["calib_path"], cam_cfg["homography_path"],
                    float(cam_cfg["calib_width"]), float(cam_cfg["calib_height"]),
                    jp.parent,
                )
            else:
                # PnP-only camera: load just K/D for undistortion; no H needed
                calib_path = _resolve_path(cam_cfg["calib_path"], jp.parent)
                suffix = calib_path.suffix.lower()
                if suffix == ".npz":
                    K, D = _load_npz_calib(str(calib_path))
                elif suffix == ".ini":
                    K, D, _, _ = _load_orbbec_ini(str(calib_path))
                else:
                    raise SystemExit(f"[ERROR] {prefix}: unsupported calib format '{suffix}'")
                model_str = cam_cfg["model"].lower()
                D = _fix_fisheye_D(D) if model_str == "fisheye" else D.reshape(1, -1)
                newK = None
                H = np.eye(3)           # unused when pnp is active
                img_pts = world_pts = None

            bounds_tol = float(cam_cfg.get("bounds_tolerance_m", 0.05))
            pen_bounds = _derive_pen_bounds(world_pts, cam_pen_floor, tol=bounds_tol)

            # Config cam_pos fields — optional when using pure PnP
            cam_x   = float(cam_cfg.get("cam_x",   0.0))
            cam_y   = float(cam_cfg.get("cam_y",   0.0))
            cam_z   = float(cam_cfg.get("cam_z",   0.0))
            plane_z = float(cam_cfg.get("plane_z", 0.0))

            # PnP results — mandatory when homography_path is absent
            pnp_params: Optional[PnPParams] = None
            cam_source = "config"
            if pnp_path_raw is not None:
                pnp_path = _resolve_path(pnp_path_raw, jp.parent)
                pnp_params, err = _load_pnp_params(pnp_path, cam_pen_floor, plane_z)
                if pnp_params is not None:
                    cam_source = "pnp"
                else:
                    if not has_homography:
                        raise SystemExit(
                            f"[ERROR] {prefix}: PnP load failed ({err}) and no "
                            f"homography_path fallback available")
                    print(f"[WARN] {prefix}: PnP load failed ({err}) "
                          f"— falling back to config H pipeline")
            elif not has_homography:
                raise SystemExit(
                    f"[ERROR] {prefix}: neither 'pnp_results_path' nor "
                    f"'homography_path' is specified")

            model_obj = CameraModel(
                prefix     = prefix,
                model      = cam_cfg["model"].lower(),
                K=K, D=D, H=H, newK=newK,
                img_pts=img_pts, world_pts=world_pts,
                calib_width  = float(cam_cfg["calib_width"]),
                calib_height = float(cam_cfg["calib_height"]),
                cam_x=cam_x, cam_y=cam_y, cam_z=cam_z,
                plane_z    = plane_z,
                pen_floor  = cam_pen_floor,
                pen_bounds = pen_bounds,
                pnp        = pnp_params,
                cam_pos_source = cam_source,
                notes      = cam_cfg.get("notes", ""),
            )
            all_cameras[prefix] = model_obj

            newK_str   = "✓" if newK is not None else "✗"
            bounds_str = "disabled" if pen_bounds is None else (
                f"x=[{pen_bounds[0]:.2f},{pen_bounds[1]:.2f}] "
                f"y=[{pen_bounds[2]:.2f},{pen_bounds[3]:.2f}]m")
            if cam_source == "pnp":
                cp = pnp_params.cam_pos_norm
                cp_m = cp * pnp_params.floor_w_m
                print(
                    f"[OK]  {prefix}: model={model_obj.model}  [PNP] "
                    f"cam_norm=({cp[0]:.4f},{cp[1]:.4f},{cp[2]:.4f}) "
                    f"→ ~({cp_m[0]:.3f},{cp_m[1]:.3f},{cp_m[2]:.3f})m  "
                    f"plane_z_norm={pnp_params.plane_z_norm:.4f}  "
                    f"newK:{newK_str}  bounds:{bounds_str}"
                )
            else:
                print(
                    f"[OK]  {prefix}: model={model_obj.model}  [CFG] "
                    f"cam=({cam_x:.3f},{cam_y:.3f},{cam_z:.3f})m  "
                    f"plane_z={plane_z:.3f}m  "
                    f"newK:{newK_str}  bounds:{bounds_str}"
                )

    print(f"[INFO] Loaded {len(all_cameras)} camera(s) from {len(json_paths)} JSON file(s)")
    return all_cameras


def _load_pnp_params(
    pnp_path: Path,
    pen_floor: PenFloor,
    plane_z_m: float,
) -> Tuple[Optional["PnPParams"], str]:
    """
    Load pipeline_calib.py camera_params.json and return a PnPParams.

    The PnP world frame uses normalised coordinates (floor_width = 1.0 unit).
    All angle computations within this pipeline stay in that normalised frame,
    because angles are scale-invariant.  plane_z_m is the real pig back height
    above the floor; it is converted to normalised units here.

    Returns (PnPParams, "") on success, (None, error_msg) on failure.
    """
    if not pnp_path.is_file():
        return None, f"file not found: {pnp_path}"
    try:
        data = json.loads(pnp_path.read_text())
        K_undist = np.array(data["K"],       dtype=np.float64).reshape(3, 3)
        R        = np.array(data["R"],       dtype=np.float64).reshape(3, 3)
        tvec     = np.array(data["tvec"],    dtype=np.float64).flatten()
        cam_pos  = np.array(data["cam_pos"], dtype=np.float64).flatten()
        floor_h  = float(data.get("floor_height_m", data.get("floor_h", 0.5)))
        # floor_w_m: scale factor for displaying world_x/y in metres (cosmetic only).
        # Uses rect_width_m if stored (--rect-width-m flag in pipeline_calib.py).
        # Falls back to 1.0 (output world_x/y in normalised units).
        # Angles are scale-invariant and are NOT affected by this value.
        floor_w_m = float(data.get("rect_width_m", 1.0))
        # plane_z_norm: detection-plane height in normalised units.
        # Read directly from JSON if stored (fully scale-free).
        # Defaults to 0.0 (pig at floor level) — no real-world scale needed.
        plane_z_norm = float(data.get("plane_z_norm", 0.0))
        reproj = data.get("reprojection_error_px", float("nan"))
        print(f"      cam_pos_norm={cam_pos.round(4).tolist()}  "
              f"floor_h_norm={floor_h:.4f}  reproj={reproj:.2f}px  "
              f"plane_z_norm={plane_z_norm:.4f}")
        return PnPParams(
            K_undist=K_undist, R=R, tvec=tvec,
            cam_pos_norm=cam_pos, floor_h_norm=floor_h,
            floor_w_m=floor_w_m, plane_z_norm=plane_z_norm,
        ), ""
    except Exception as e:
        return None, str(e)


# =============================================================================
# Calibration / homography loading  (identical to original script)
# =============================================================================

def _resolve_path(p: str, json_dir: Path) -> Path:
    p = Path(p)
    if p.is_absolute():
        return p
    if p.exists():
        return p
    return json_dir / p


def _load_calib_and_homography(
    prefix, model, calib_path, h_path, calib_w, calib_h, json_dir
):
    calib_path = _resolve_path(calib_path, json_dir)
    h_path     = _resolve_path(h_path,     json_dir)

    suffix = calib_path.suffix.lower()
    if suffix == ".npz":
        K, D = _load_npz_calib(str(calib_path))
    elif suffix == ".ini":
        K, D, _, _ = _load_orbbec_ini(str(calib_path))
    else:
        raise SystemExit(f"[ERROR] {prefix}: unsupported calib format '{suffix}'")

    if model == "fisheye":
        D = _fix_fisheye_D(D)
    else:
        D = D.reshape(1, -1)

    H, img_pts, world_pts, newK = _load_homography_npz(str(h_path))
    if newK is None:
        print(f"[WARN] {prefix}: H .npz has no embedded newK — using K as P matrix.")
    return K, D, newK, H, img_pts, world_pts


def _load_npz_calib(path):
    d = np.load(path)
    K_key = next((k for k in ("camera_matrix","K","mtx") if k in d), None)
    D_key = next((k for k in ("dist_coeff","dist_coeffs","D","k") if k in d), None)
    if K_key is None:
        raise SystemExit(f"[ERROR] .npz has no camera_matrix key: {path}")
    K = d[K_key].astype(np.float64).reshape(3,3)
    D = d[D_key].astype(np.float64).flatten() if D_key else np.zeros(4)
    return K, D


def _load_orbbec_ini(path):
    import configparser
    cfg = configparser.ConfigParser()
    if not cfg.read(path):
        raise SystemExit(f"[ERROR] Cannot read .ini: {path}")

    ci_name = next((s for s in cfg.sections() if "colorintrinsic" in s.lower()), None)
    cd_name = next((s for s in cfg.sections() if "colordistortion" in s.lower()), None)
    if ci_name is None:
        raise SystemExit(
            f"[ERROR] {path}: missing [ColorIntrinsic] section. "
            f"Found sections: {cfg.sections()}"
        )

    ci = cfg[ci_name]
    fx = float(ci["fx"]); fy = float(ci["fy"])
    cx = float(ci["cx"]); cy = float(ci["cy"])
    w  = float(ci["width"]); h = float(ci["height"])
    K  = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float64)

    if cd_name is not None:
        cd = cfg[cd_name]
        def _g(k): return float(cd.get(k, 0))
        # Orbbec order k1,k2,k3,k4,k5,k6,p1,p2 → OpenCV k1,k2,p1,p2,k3,k4,k5,k6
        D = np.array([_g("k1"),_g("k2"),_g("p1"),_g("p2"),
                      _g("k3"),_g("k4"),_g("k5"),_g("k6")], dtype=np.float64)
    else:
        D = np.zeros(5, dtype=np.float64)

    return K, D, w, h


def _fix_fisheye_D(D):
    D = D.flatten()
    if len(D) < 4:
        D = np.pad(D, (0, 4-len(D)))
    return D[:4].reshape(-1, 1).astype(np.float64)


def _load_homography_npz(path):
    d = np.load(path, allow_pickle=True)
    H_key = next((k for k in ("H","homography","H_matrix") if k in d), None)
    if H_key is None:
        raise SystemExit(f"[ERROR] No H key in {path}")
    H = d[H_key].astype(np.float64)
    img_pts   = d["img_pts"].astype(np.float32)   if "img_pts"   in d else None
    world_pts = d["world_pts"].astype(np.float32) if "world_pts" in d else None
    newK = d["newK"].astype(np.float64) if "newK" in d else None
    return H, img_pts, world_pts, newK


def _derive_pen_bounds(world_pts, pen_floor, tol=0.05):
    if tol < 0:
        return None
    if world_pts is not None and len(world_pts) >= 4:
        wpts = world_pts.reshape(-1,2)
        return (wpts[:,0].min()-tol, wpts[:,0].max()+tol,
                wpts[:,1].min()-tol, wpts[:,1].max()+tol)
    return (0-tol, pen_floor.width_m+tol, 0-tol, pen_floor.height_m+tol)


# =============================================================================
# Per-pixel angle computation  (identical to original)
# =============================================================================

def _scale_K(K, from_w, from_h, to_w, to_h):
    sx = to_w / from_w; sy = to_h / from_h
    s = np.array([[sx,0,0],[0,sy,0],[0,0,1]], dtype=np.float64)
    return s @ K


def _undistort_pixel(px, py, K, D, P, model):
    pt = np.array([[[px, py]]], dtype=np.float32)
    if model == "fisheye":
        D4 = D[:4].reshape(-1,1).astype(np.float64)
        ud = cv2.fisheye.undistortPoints(pt, K, D4, P=P)
    else:
        ud = cv2.undistortPoints(pt, K, D, P=P)
    return float(ud[0,0,0]), float(ud[0,0,1])


def _apply_H(u, v, H):
    q = H @ np.array([u, v, 1.0], dtype=np.float64)
    if abs(q[2]) < 1e-12:
        return np.nan, np.nan, False
    q /= q[2]
    return float(q[0]), float(q[1]), True


def _world_to_angles(Xw, Yw, cam):
    dX = Xw - cam.cam_x
    dY = Yw - cam.cam_y
    dZ = cam.plane_z - cam.cam_z
    r     = math.hypot(dX, dY)
    az    = math.degrees(math.atan2(dY, dX))
    el    = math.degrees(math.atan2(dZ, r))
    el_dn = math.degrees(math.atan2(cam.cam_z - cam.plane_z, r))
    return az, el, el_dn, r


def _is_in_pen(Xw, Yw, bounds):
    if not math.isfinite(Xw) or not math.isfinite(Yw):
        return False
    if bounds is None:
        return True
    xmin, xmax, ymin, ymax = bounds
    return (xmin <= Xw <= xmax) and (ymin <= Yw <= ymax)


def _deg_to_sin_cos(deg):
    if deg is None or (isinstance(deg, float) and math.isnan(deg)):
        return np.nan, np.nan
    r = math.radians(deg)
    return math.sin(r), math.cos(r)


def _bbox_sample_point(x, y, w, h, mode):
    if mode == "center":
        return x + w*0.5, y + h*0.5
    elif mode in ("lower_center", "bottom_center"):
        return x + w*0.5, y + h*0.85
    return x + w*0.5, y + h*0.5


def _compute_arrow_direction(
    Xw_norm: float, Yw_norm: float,
    pnp: "PnPParams",
    hx: float, hy: float,
) -> tuple:
    """
    Image-space unit vector along the camera ray (camera → pig direction, undistorted calib-res frame).

    Method: project the camera nadir (floor point directly below camera) to the
    undistorted image plane.  The direction from the nadir pixel to the pig's
    undistorted pixel (hx, hy) is the image-space camera ray direction — i.e.,
    the direction the camera is looking at the pig, away from the camera.

    Returns (arrow_u, arrow_v) normalised to unit length, or (nan, nan) on failure.
    """
    cp = pnp.cam_pos_norm
    # Camera nadir: point on the detection plane directly below the camera
    nadir = np.array([cp[0], cp[1], pnp.plane_z_norm], dtype=np.float64)
    nadir_cam = pnp.R @ nadir + pnp.tvec
    if nadir_cam[2] <= 0:
        return np.nan, np.nan
    p = pnp.K_undist @ nadir_cam
    nadir_u = p[0] / p[2]
    nadir_v = p[1] / p[2]
    # Direction: along camera ray = from camera nadir toward pig (away from camera)
    du = hx - nadir_u
    dv = hy - nadir_v
    mag = math.hypot(du, dv)
    if mag < 1e-6:
        # Pig is directly under the camera — arrow undefined
        return np.nan, np.nan
    return du / mag, dv / mag


_NAN_RESULT = (np.nan, np.nan, np.nan, np.nan, np.nan, 0, np.nan, np.nan)


def compute_angles_for_pixel(px, py, img_w, img_h, cam, allow_outside_bounds=False):
    """
    Dispatch between two self-consistent pipelines:

    Config pipeline (cam.pnp is None):
      raw pixel → undistort → H → (Xw, Yw) metres → angles via config cam_pos

    PnP pipeline (cam.pnp is not None):
      raw pixel → undistort → K_undist^-1 back-project → ray–floor intersection
                → (Xw, Yw) in normalised PnP coords → angles via PnP cam_pos

    Both pipelines are fully self-consistent (world coords and camera position
    are in the same coordinate frame), so az/el are correct for each.
    The returned world_x / world_y differ in units (metres vs normalised×floor_w_m)
    but the angle columns are directly comparable.
    """
    if cam.pnp is not None:
        return _compute_angles_pnp(px, py, img_w, img_h, cam, allow_outside_bounds)
    return _compute_angles_config(px, py, img_w, img_h, cam, allow_outside_bounds)


def _compute_angles_config(px, py, img_w, img_h, cam, allow_outside_bounds=False):
    """Config H pipeline: identical to add_camera_angles_multicam.py."""
    try:
        cw, ch = cam.calib_width, cam.calib_height
        K_img = _scale_K(cam.K, cw, ch, img_w, img_h)
        P_src = cam.newK if cam.newK is not None else cam.K
        P_img = _scale_K(P_src, cw, ch, img_w, img_h)
        ux, uy = _undistort_pixel(px, py, K_img, cam.D, P_img, cam.model)
        hx = ux * (cw / img_w)
        hy = uy * (ch / img_h)
        Xw, Yw, ok = _apply_H(hx, hy, cam.H)
        if not ok or (not allow_outside_bounds and not _is_in_pen(Xw, Yw, cam.pen_bounds)):
            return _NAN_RESULT
        az, el, el_dn, _ = _world_to_angles(Xw, Yw, cam)
        return Xw, Yw, az, el, el_dn, 1, np.nan, np.nan
    except Exception:
        return _NAN_RESULT


def _compute_angles_pnp(px, py, img_w, img_h, cam, allow_outside_bounds=False):
    """
    PnP pipeline — fully self-consistent in the normalised world frame.

    Steps:
      1. Scale K to image resolution; undistort using K_undist as P matrix.
      2. Scale back to calib resolution → undistorted pixel in calib frame.
      3. Back-project through K_undist^-1 → normalised image ray in cam frame.
      4. Rotate ray to normalised world frame via R^T.
      5. Intersect with floor plane at Z = plane_z_norm.
      6. Compute az/el in normalised world frame (scale-invariant).
      7. Compute image-space arrow direction (pig → camera) via _compute_arrow_direction.
      8. Report world_x/y in metres = normalised × floor_w_m for readability.
    """
    try:
        pnp = cam.pnp
        cw, ch = cam.calib_width, cam.calib_height

        # Undistort using K_undist as the target (output) projection matrix
        K_img        = _scale_K(cam.K,        cw, ch, img_w, img_h)
        K_undist_img = _scale_K(pnp.K_undist, cw, ch, img_w, img_h)
        ux, uy = _undistort_pixel(px, py, K_img, cam.D, K_undist_img, cam.model)

        # Scale back to calib frame for K_undist back-projection
        hx = ux * (cw / img_w)
        hy = uy * (ch / img_h)

        # Back-project: normalised camera ray
        K_inv = np.linalg.inv(pnp.K_undist)
        ray_cam = K_inv @ np.array([hx, hy, 1.0], dtype=np.float64)

        # Rotate to normalised world frame (R is world→cam, R^T is cam→world)
        ray_world = pnp.R.T @ ray_cam

        # Intersect with floor plane at z = plane_z_norm
        # Parametric: P = cam_pos + t * ray_world,  P[2] = plane_z_norm
        dz = ray_world[2]
        if abs(dz) < 1e-9:
            return _NAN_RESULT   # ray parallel to floor
        cp = pnp.cam_pos_norm
        t  = (pnp.plane_z_norm - cp[2]) / dz
        if t < 0:
            return _NAN_RESULT   # floor behind camera

        Xw_norm = cp[0] + t * ray_world[0]
        Yw_norm = cp[1] + t * ray_world[1]

        # Bounds check in normalised coords — skip when --allow-outside-bounds
        in_bounds = (0.0 <= Xw_norm <= 1.0 + 1e-3 and
                     0.0 <= Yw_norm <= pnp.floor_h_norm + 1e-3)
        if not in_bounds and not allow_outside_bounds:
            return _NAN_RESULT

        # Angles in normalised world (scale-invariant)
        dX = Xw_norm - cp[0]
        dY = Yw_norm - cp[1]
        dZ_up   = pnp.plane_z_norm - cp[2]   # negative for overhead cam
        dZ_down = cp[2] - pnp.plane_z_norm   # positive for overhead cam
        r       = math.hypot(dX, dY)
        az      = math.degrees(math.atan2(dY, dX))
        el      = math.degrees(math.atan2(dZ_up, r))
        el_dn   = math.degrees(math.atan2(dZ_down, r))

        # Image-space arrow: unit vector from pig toward camera
        arrow_u, arrow_v = _compute_arrow_direction(Xw_norm, Yw_norm, pnp, hx, hy)

        # Convert world coords to metres for readability in CSV
        Xw_m = Xw_norm * pnp.floor_w_m
        Yw_m = Yw_norm * pnp.floor_w_m

        return Xw_m, Yw_m, az, el, el_dn, 1, arrow_u, arrow_v
    except Exception:
        return _NAN_RESULT


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    cam_models = load_camera_configs(args.camera_config)

    # Print per-camera summary
    print()
    print(f"  {'Prefix':<22}  {'cam_pos (config) m':<32}  Pipeline")
    print("  " + "-"*70)
    for prefix, cam in cam_models.items():
        tag = "PnP (ray–floor)" if cam.cam_pos_source == "pnp" else "Config H+cam_pos"
        if cam.pnp is not None:
            cp_m = cam.pnp.cam_pos_norm * cam.pnp.floor_w_m
            pos  = f"(cfg) ({cam.cam_x:.3f},{cam.cam_y:.3f},{cam.cam_z:.3f})  "\
                   f"(pnp) ({cp_m[0]:.3f},{cp_m[1]:.3f},{cp_m[2]:.3f})"
        else:
            pos = f"({cam.cam_x:.3f},{cam.cam_y:.3f},{cam.cam_z:.3f})"
        print(f"  {prefix:<22}  {pos:<48}  {tag}")
    print()

    delim = args.csv_delim
    df = pd.read_csv(args.csv, sep=delim, dtype=str)
    print(f"[INFO] Loaded {len(df):,} rows from {args.csv}")

    # Numeric columns
    df[args.width_col]  = pd.to_numeric(df[args.width_col],  errors="coerce").fillna(1920)
    df[args.height_col] = pd.to_numeric(df[args.height_col], errors="coerce").fillna(1080)

    cols = {
        "world_x":            [],
        "world_y":            [],
        "azimuth_deg":        [],
        "elevation_deg":      [],
        "elevation_down_deg": [],
        "azimuth_sin":        [],
        "azimuth_cos":        [],
        "elevation_sin":      [],
        "elevation_cos":      [],
        "angle_valid":        [],
        "cam_pos_source":     [],
        "arrow_u":            [],   # image-space direction toward camera (unit vec)
        "arrow_v":            [],
    }

    # Debug
    debug_set: set = set()
    if args.debug:
        random.seed(args.debug_seed)
        n = min(args.debug_samples, len(df))
        debug_set = set(random.sample(range(len(df)), n))
        Path(args.debug_dir).mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Debug → {args.debug_dir}  ({n} samples)")

    warned_prefixes: set = set()

    for idx, row in df.iterrows():
        if args.log_every > 0 and idx % args.log_every == 0:
            pct = 100 * idx / max(len(df), 1)
            print(f"[INFO] Row {idx:,}/{len(df):,}  ({pct:.1f}%) …")

        image_id = row[args.image_id_col]
        bbox     = ast.literal_eval(str(row[args.bbox_col]))
        x, y, w, h = map(float, bbox[:4])
        img_w    = float(row[args.width_col])
        img_h    = float(row[args.height_col])

        stem       = Path(str(image_id)).stem
        parts      = stem.split("_")
        cam_prefix = "_".join(parts[:3]) if len(parts) >= 3 else stem

        cam = cam_models.get(cam_prefix)
        if cam is None:
            if cam_prefix not in warned_prefixes:
                print(f"[WARN] No config for '{cam_prefix}' — skipping")
                warned_prefixes.add(cam_prefix)
            _append_nan(cols)
            continue

        x = max(0.0, min(x, img_w-1)); y = max(0.0, min(y, img_h-1))
        w = max(1.0, min(w, img_w-x)); h = max(1.0, min(h, img_h-y))

        spx, spy = _bbox_sample_point(x, y, w, h, args.sample_mode)
        spx = max(0.0, min(spx, img_w-1))
        spy = max(0.0, min(spy, img_h-1))

        Xw, Yw, az, el, el_dn, valid, arrow_u, arrow_v = compute_angles_for_pixel(
            spx, spy, img_w, img_h, cam, args.allow_outside_bounds)

        az_sin,  az_cos  = _deg_to_sin_cos(az)

        # Convert elevation to requested output convention
        if args.elevation_convention == "zenith":
            # zenith = angle from nadir (0=directly overhead, 90=horizontal)
            # el_dn is positive-down depression angle (0..90 for overhead cameras)
            # zenith = 90 - el_dn  →  sin(zenith) = cos(el_dn),  cos(zenith) = sin(el_dn)
            zenith_deg = 90.0 - el_dn if not (isinstance(el_dn, float) and math.isnan(el_dn)) else el_dn
            el_out = zenith_deg
            el_sin, el_cos = _deg_to_sin_cos(zenith_deg)
        else:
            el_out = el
            el_sin, el_cos = _deg_to_sin_cos(el)

        cols["world_x"].append(Xw);  cols["world_y"].append(Yw)
        cols["azimuth_deg"].append(az)
        cols["elevation_deg"].append(el_out)
        cols["elevation_down_deg"].append(el_dn)
        cols["azimuth_sin"].append(az_sin);  cols["azimuth_cos"].append(az_cos)
        cols["elevation_sin"].append(el_sin); cols["elevation_cos"].append(el_cos)
        cols["angle_valid"].append(valid)
        cols["cam_pos_source"].append(cam.cam_pos_source)
        cols["arrow_u"].append(arrow_u)
        cols["arrow_v"].append(arrow_v)

        if args.debug and idx in debug_set:
            _save_debug_image(args, idx, image_id, x, y, w, h,
                              img_w, img_h, cam, spx, spy, az, el, el_dn, valid,
                              arrow_u=arrow_u, arrow_v=arrow_v, Xw=Xw, Yw=Yw)

    for col_name, values in cols.items():
        df[col_name] = values

    df.to_csv(args.out_csv, sep=delim, index=False)

    n_valid  = sum(cols["angle_valid"])
    n_pnp    = sum(1 for s in cols["cam_pos_source"] if s == "pnp")
    n_config = sum(1 for s in cols["cam_pos_source"] if s == "config")
    pct      = 100 * n_valid / max(len(df), 1)
    print(f"\n[OK]  Saved → {args.out_csv}")
    print(f"      Valid angles : {n_valid:,}/{len(df):,} ({pct:.1f}%)")
    print(f"      PnP cam_pos  : {n_pnp:,} rows")
    print(f"      Config cam_pos: {n_config:,} rows")
    print()
    print("To compare angles with the original run:")
    print(f"  python scripts/compare_angle_csvs.py "
          f"--orig <original_with_angles.csv> --pnp {args.out_csv}")


# =============================================================================
# Helpers
# =============================================================================

def _append_nan(cols):
    for k in cols:
        if k == "angle_valid":
            cols[k].append(0)
        elif k == "cam_pos_source":
            cols[k].append("config")
        else:
            cols[k].append(np.nan)


def _project_world_to_frame(
    P_world: np.ndarray,
    pnp: "PnPParams",
    cam: "CameraModel",
    fw: int, fh: int,
) -> Optional[tuple]:
    """
    Project a world point to frame pixel coordinates using the *original*
    distorted camera model (K, D) so the result maps onto the raw image.
    Returns (px, py) in frame pixels, or None if behind camera or error.
    """
    # Quick depth check
    P_cam = pnp.R @ P_world.astype(np.float64) + pnp.tvec
    if P_cam[2] <= 0:
        return None

    rvec, _ = cv2.Rodrigues(pnp.R)
    tvec_col = pnp.tvec.reshape(3, 1)
    P_in = P_world.astype(np.float64).reshape(1, 1, 3)

    try:
        if cam.model == "fisheye":
            pts, _ = cv2.fisheye.projectPoints(
                P_in, rvec, tvec_col, cam.K, cam.D)
        else:
            D_cv = cam.D.reshape(-1, 1) if cam.D.ndim == 1 else cam.D
            pts, _ = cv2.projectPoints(
                P_in.reshape(1, 3), rvec, tvec_col, cam.K, D_cv)
        px = float(pts.reshape(2)[0])
        py = float(pts.reshape(2)[1])
    except Exception:
        return None

    # Scale from calibration resolution to frame resolution
    cx = px * (fw / cam.calib_width)
    cy = py * (fh / cam.calib_height)
    return int(round(cx)), int(round(cy))


def _draw_floor_minimap(
    frame: np.ndarray,
    pnp: "PnPParams",
    Xw_m: float, Yw_m: float,
    az_deg: float, el_deg: float,
    valid: int,
) -> None:
    """
    Draw a top-down floor minimap in the bottom-right corner of frame.

    Shows:
      - Floor rectangle (world origin = near-left corner, NL→NR = +X, NL→FL = +Y)
      - Camera position (cyan cross)
      - Pig world position (green dot)
      - Azimuth arrow from pig toward camera
      - World X/Y axis labels
      - az/el text
    """
    MAP  = 200          # canvas size (pixels)
    MG   = 22           # margin inside canvas
    fh, fw = frame.shape[:2]

    # Floor dimensions in normalised coords (floor_w = 1.0)
    floor_w_norm = 1.0
    floor_h_norm = pnp.floor_h_norm

    # Compute scale so the floor fits inside MAP - 2*MG, preserving aspect
    scale = min((MAP - 2*MG) / floor_w_norm, (MAP - 2*MG) / floor_h_norm)

    # Floor corners on canvas (Y=0 at top of floor = near row)
    fx0 = MG
    fy0 = MG
    fx1 = int(fx0 + floor_w_norm * scale)
    fy1 = int(fy0 + floor_h_norm * scale)

    canvas = np.zeros((MAP, MAP, 3), dtype=np.uint8)
    canvas[:] = (30, 30, 30)
    cv2.rectangle(canvas, (fx0, fy0), (fx1, fy1), (100, 100, 100), 1)

    # World → canvas pixel helper
    def w2c(xn: float, yn: float) -> tuple:
        return (int(round(fx0 + xn * scale)),
                int(round(fy0 + yn * scale)))

    # World axes arrows (from near-left = origin)
    org = w2c(0, 0)
    cv2.arrowedLine(canvas, org, w2c(0.18, 0), (80, 80, 255), 1, tipLength=0.35)   # +X right
    cv2.putText(canvas, "+X", (org[0]+int(0.2*scale), org[1]+3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, (80, 80, 255), 1)
    cv2.arrowedLine(canvas, org, w2c(0, 0.18), (80, 255, 80), 1, tipLength=0.35)   # +Y far
    cv2.putText(canvas, "+Y", (org[0]-18, org[1]+int(0.2*scale)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, (80, 255, 80), 1)

    # Camera position (convert from metres back to normalised)
    fw_m = pnp.floor_w_m if pnp.floor_w_m > 0 else 1.0
    cp_xn = pnp.cam_pos_norm[0]
    cp_yn = pnp.cam_pos_norm[1]
    cam_c = w2c(cp_xn, cp_yn)
    cv2.drawMarker(canvas, cam_c, (0, 255, 255), cv2.MARKER_CROSS, 14, 2)
    cv2.putText(canvas, "CAM", (cam_c[0] + 4, cam_c[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 255, 255), 1)

    # Pig position (metres → normalised)
    pig_xn = Xw_m / fw_m
    pig_yn = Yw_m / fw_m
    pig_c  = w2c(pig_xn, pig_yn)
    col_pig = (0, 200, 0) if valid else (0, 0, 200)
    cv2.circle(canvas, pig_c, 5, col_pig, -1)

    # Azimuth arrow: from pig toward camera, length proportional to floor
    if math.isfinite(az_deg):
        az_rad  = math.radians(az_deg)
        arr_len = int(0.15 * scale)
        # az = atan2(dY, dX) from cam; so direction FROM pig TOWARD cam = -az direction
        dxn = math.cos(az_rad + math.pi)
        dyn = math.sin(az_rad + math.pi)
        tip = (int(round(pig_c[0] + dxn * arr_len)),
               int(round(pig_c[1] + dyn * arr_len)))
        cv2.arrowedLine(canvas, pig_c, tip, (0, 255, 0), 2, tipLength=0.35)

    # Labels: az / el / world_pos
    cv2.putText(canvas, f"az={az_deg:.1f}deg", (4, MAP - 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1)
    cv2.putText(canvas, f"el={el_deg:.1f}deg", (4, MAP - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1)
    cv2.putText(canvas, f"pig ({Xw_m:.3f},{Yw_m:.3f})", (4, MAP - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, (180, 180, 180), 1)

    # Paste onto frame (bottom-right corner)
    oy = max(0, fh - MAP - 8)
    ox = max(0, fw - MAP - 8)
    frame[oy:oy + MAP, ox:ox + MAP] = canvas


def _save_debug_image(args, idx, image_id, x, y, w, h,
                      img_w, img_h, cam, spx, spy, az, el, el_dn, valid,
                      arrow_u=np.nan, arrow_v=np.nan, Xw=np.nan, Yw=np.nan):
    if args.image_root is None:
        return
    img_path = Path(args.image_root) / str(image_id)
    if not img_path.is_file():
        return
    frame = cv2.imread(str(img_path))
    if frame is None:
        return

    # Scale bbox to frame size
    fh, fw = frame.shape[:2]
    sx, sy = fw / img_w, fh / img_h
    bx1, by1 = int(x*sx), int(y*sy)
    bx2, by2 = int((x+w)*sx), int((y+h)*sy)
    spx_f, spy_f = int(spx*sx), int(spy*sy)

    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (200, 200, 200), 2)
    cv2.circle(frame, (spx_f, spy_f), 6, (0, 80, 255), -1)

    # ── Project camera nadir onto image ──────────────────────────────────────
    # Nadir = floor point directly below camera = (cam_pos_x, cam_pos_y, plane_z)
    nadir_px = None
    if cam.pnp is not None:
        nadir_world = np.array([
            cam.pnp.cam_pos_norm[0],
            cam.pnp.cam_pos_norm[1],
            cam.pnp.plane_z_norm,
        ], dtype=np.float64)
        nadir_px = _project_world_to_frame(
            nadir_world, cam.pnp, cam,
            fw, fh,
        )

    if nadir_px is not None:
        nx, ny = nadir_px
        # Camera nadir marker
        cv2.drawMarker(frame, (nx, ny), (0, 255, 255),
                       cv2.MARKER_CROSS, 24, 3, cv2.LINE_AA)
        cv2.circle(frame, (nx, ny), 12, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "CAM", (nx + 14, ny - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
        # Line from sample point to camera nadir
        cv2.line(frame, (spx_f, spy_f), (nx, ny), (0, 220, 255), 1, cv2.LINE_AA)
        # Azimuth angle label at midpoint of the line
        if math.isfinite(az):
            mx = (spx_f + nx) // 2
            my = (spy_f + ny) // 2
            cv2.putText(frame, f"az={az:.1f}",
                        (mx + 6, my), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 220, 255), 2, cv2.LINE_AA)

    # ── Image-space arrow (pig → camera, from arrow_u/arrow_v) ───────────────
    if math.isfinite(arrow_u) and math.isfinite(arrow_v):
        arrow_len = max(fw, fh) // 12
        ax2 = int(spx_f + arrow_u * arrow_len)
        ay2 = int(spy_f + arrow_v * arrow_len)
        cv2.arrowedLine(frame, (spx_f, spy_f), (ax2, ay2),
                        (0, 255, 0), 3, tipLength=0.25, line_type=cv2.LINE_AA)

    # ── Text labels ───────────────────────────────────────────────────────────
    src_color = (0, 255, 255) if cam.cam_pos_source == "pnp" else (180, 180, 0)
    label = (f"az={az:.1f}  el={el:.1f}  el_dn={el_dn:.1f}  "
             f"({'OK' if valid else 'INVALID'})  [{cam.cam_pos_source.upper()}]"
             if math.isfinite(az) else f"INVALID  [{cam.cam_pos_source.upper()}]")
    cv2.putText(frame, label, (bx1, max(20, by1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, src_color, 2, cv2.LINE_AA)

    if cam.pnp is not None:
        cp = cam.pnp.cam_pos_norm
        info = (f"cam_norm=({cp[0]:.3f},{cp[1]:.3f},{cp[2]:.3f})  "
                f"floor_h={cam.pnp.floor_h_norm:.3f}")
    else:
        info = f"cam_cfg=({cam.cam_x:.3f},{cam.cam_y:.3f},{cam.cam_z:.3f})m"
    cv2.putText(frame, info, (10, fh - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, src_color, 1, cv2.LINE_AA)

    # ── Floor minimap (bottom-right) ─────────────────────────────────────────
    if cam.pnp is not None and math.isfinite(Xw) and math.isfinite(Yw):
        _draw_floor_minimap(frame, cam.pnp, Xw, Yw, az, el, valid)

    out_path = Path(args.debug_dir) / f"dbg_{idx:06d}_{Path(str(image_id)).stem}.jpg"
    cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 88])


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    main()
