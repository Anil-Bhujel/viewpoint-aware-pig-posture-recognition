#!/usr/bin/env python3
"""
pipeline_calib.py
-----------------
Estimate per-pixel azimuth / elevation angles using ONLY:
  - A calibrated camera (K + D from checkerboard .npz or manufacturer JSON)
  - A single image
  - 4 manually annotated pen-floor corners

No physical floor dimensions, no camera height needed.

Approach
--------
The camera pose is recovered from the homography between the annotated
image corners and the unit-square floor model.  Because azimuth and elevation
are scale-invariant (they depend only on the *direction* of the camera-to-floor
ray, not its length), the normalised coordinate system

    world: floor spans [0, 1] × [0, H/W]   (H/W = floor aspect ratio)

produces geometrically correct angle maps.

Calibration file formats supported
------------------------------------
  .npz  (OpenCV checkerboard output)
        keys: camera_matrix (3×3), dist_coeff (4×1 or 5×1 or 8×1)
        — 4-coeff dist_coeff is auto-detected as fisheye

  .json manual spec:
        {
            "K": [[fx,0,cx],[0,fy,cy],[0,0,1]],
            "D": [k1,k2,p1,p2],            // optional
            "fisheye": true/false           // optional
        }

  .ini  (Orbbec/Femto manufacturer file)
        Reads [ColorIntrinsic] (fx,fy,cx,cy) and [ColorDistortion]
        (k1..k6, p1, p2 in Orbbec ordering → converted to OpenCV ordering).
        Auto-detected as regular (pinhole) camera.

Fisheye handling
-----------------
  If D has 4 coefficients OR --fisheye is passed, the image is undistorted
  using cv2.fisheye.  The remapped newK is used for all subsequent geometry.
  Corner annotation is performed on the already-undistorted image.

Usage
-----
    # Fisheye camera (auto-detected from 4-coeff D)
    python pipeline_calib.py \
        --image frame.jpg \
        --calib camera_parameters/pen1_tur_cam1_calibration.npz \
        --out-dir results/pen1_cam1

    # Re-use corners from a previous run
    python pipeline_calib.py \
        --image frame2.jpg \
        --calib camera_parameters/pen1_tur_cam1_calibration.npz \
        --corners results/pen1_cam1/corners.json \
        --out-dir results/pen1_cam1_frame2

    # For Orbbec RGB (regular pinhole camera, no fisheye)
    python floor_angle_estimation/pipeline_calib.py \
        --image frame_orb.jpg \
        --calib "orbbec_camera_parameter/P1C1_CameraParam_Femto Mega...ini" \
        --no-fisheye \
        --out-dir results/pen1_orb_cam1
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Calibration loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_orbbec_ini(path) -> tuple:
    """
    Parse an Orbbec / Femto Mega manufacturer .ini file and return (K, D).

    Expected sections:
      [ColorIntrinsic]  fx, fy, cx, cy, width, height
      [ColorDistortion] k1, k2, k3, k4, k5, k6, p1, p2

    Orbbec distortion ordering: k1,k2,k3 (radial), k4,k5,k6 (rational), p1,p2 (tangential)
    OpenCV 8-coeff ordering:    k1,k2,p1,p2,k3,k4,k5,k6

    Falls back gracefully if some keys are missing (uses 0).
    """
    import configparser
    cfg = configparser.ConfigParser()
    cfg.read(str(path))

    # ── Intrinsics ────────────────────────────────────────────────────────────
    # Support both naming conventions seen in the wild
    intr_section = next(
        (s for s in cfg.sections() if "colorintrinsic" in s.lower() or s == "RGB_0"),
        None,
    )
    if intr_section is None:
        raise ValueError(
            f"[ERROR] No [ColorIntrinsic] (or [RGB_0]) section found in {path}. "
            f"Sections found: {cfg.sections()}"
        )
    sec = cfg[intr_section]
    fx = float(sec["fx"]);  fy = float(sec["fy"])
    cx = float(sec["cx"]);  cy = float(sec["cy"])
    K  = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    print(f"[INFO] Orbbec intrinsics: fx={fx:.2f}  fy={fy:.2f}  cx={cx:.2f}  cy={cy:.2f}")

    # ── Distortion ────────────────────────────────────────────────────────────
    dist_section = next(
        (s for s in cfg.sections() if "colordistortion" in s.lower()),
        None,
    )
    if dist_section is not None:
        dsec = cfg[dist_section]
        # Orbbec order: k1, k2, k3, k4, k5, k6, p1, p2
        k1 = float(dsec.get("k1", 0)); k2 = float(dsec.get("k2", 0))
        k3 = float(dsec.get("k3", 0)); k4 = float(dsec.get("k4", 0))
        k5 = float(dsec.get("k5", 0)); k6 = float(dsec.get("k6", 0))
        p1 = float(dsec.get("p1", 0)); p2 = float(dsec.get("p2", 0))
        # Rearrange to OpenCV 8-coeff: k1, k2, p1, p2, k3, k4, k5, k6
        D = np.array([k1, k2, p1, p2, k3, k4, k5, k6], dtype=np.float64)
    elif "coeffs" in sec:
        # Old [RGB_0] style with a "coeffs" key (space-separated)
        import re
        D = np.array([float(x) for x in re.split(r"[\s,]+", sec["coeffs"].strip())],
                     dtype=np.float64)
    else:
        D = np.zeros(5, dtype=np.float64)
        print("[WARN] No distortion section found — using zero distortion")

    print(f"[INFO] Orbbec distortion (OpenCV order): {np.round(D, 6).tolist()}")
    return K, D


def load_calibration(path: str) -> dict:
    """
    Load K and D from a .npz or .json calibration file.

    Returns
    -------
    dict with keys:
        K       : (3,3) float64 intrinsic matrix
        D       : (N,) float64 distortion coefficients (may be empty)
        fisheye : bool — True if fisheye model should be used
    """
    p = Path(path)
    if not p.is_file():
        print(f"[ERROR] Calibration file not found: {p}")
        sys.exit(1)

    if p.suffix.lower() == ".npz":
        data = np.load(str(p), allow_pickle=True)
        # Support both common key naming conventions
        K_key = next((k for k in ("camera_matrix", "K", "mtx") if k in data), None)
        D_key = next((k for k in ("dist_coeff", "dist_coeffs", "D", "k") if k in data), None)
        if K_key is None:
            print(f"[ERROR] .npz file has no 'camera_matrix' or 'K' key. "
                  f"Found keys: {list(data.keys())}")
            sys.exit(1)
        K = data[K_key].astype(np.float64).reshape(3, 3)
        D = data[D_key].astype(np.float64).flatten() if D_key else np.zeros(4)
        # 4 distortion coefficients → likely fisheye (OpenCV fisheye model uses exactly 4)
        fisheye = (len(D) == 4)

    elif p.suffix.lower() == ".json":
        raw = json.loads(p.read_text())
        K = np.array(raw["K"], dtype=np.float64).reshape(3, 3)
        D = np.array(raw.get("D", []), dtype=np.float64).flatten()
        fisheye = bool(raw.get("fisheye", len(D) == 4))

    elif p.suffix.lower() == ".ini":
        K, D = _load_orbbec_ini(p)
        fisheye = False   # Orbbec RGB is a standard pinhole camera

    else:
        print(f"[ERROR] Unsupported calibration format: {p.suffix} (use .npz, .json, or .ini)")
        sys.exit(1)

    print(f"[INFO] Loaded calibration from {p.name}")
    print(f"       fx={K[0,0]:.2f}  fy={K[1,1]:.2f}  "
          f"cx={K[0,2]:.2f}  cy={K[1,2]:.2f}")
    print(f"       D={np.round(D, 5).tolist()}")
    print(f"       model={'fisheye' if fisheye else 'regular'}")
    return {"K": K, "D": D, "fisheye": fisheye}


# ─────────────────────────────────────────────────────────────────────────────
# Undistortion
# ─────────────────────────────────────────────────────────────────────────────

def undistort_image(img: np.ndarray, K: np.ndarray, D: np.ndarray,
                    fisheye: bool, balance: float = 0.0) -> tuple:
    """
    Undistort an image.

    Returns
    -------
    (undistorted_img, new_K)
        new_K is the effective intrinsic matrix of the undistorted image.
    """
    h, w = img.shape[:2]
    D4 = D[:4].reshape(-1, 1) if fisheye else D

    if fisheye:
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D4, (w, h), np.eye(3), balance=balance)
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D4, np.eye(3), new_K, (w, h), cv2.CV_16SC2)
        undist = cv2.remap(img, map1, map2, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT)
    else:
        new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha=0)
        undist = cv2.undistort(img, K, D, newCameraMatrix=new_K)

    return undist, new_K


# ─────────────────────────────────────────────────────────────────────────────
# Pose from homography (no metric floor size needed)
# ─────────────────────────────────────────────────────────────────────────────

def _floor_aspect(corners, K):
    """
    Compute floor_h (depth/width ratio) from homography column norms.
    Scale-invariant: uses |r2|/|r1| from the homography decomposition.
    Input corners = [NL, NR, FR, FL] (near-left, near-right, far-right, far-left).
    """
    src = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
    dst = np.float32(corners)
    H = cv2.getPerspectiveTransform(src, dst)
    K_inv = np.linalg.inv(K)
    r1_s = K_inv @ H[:, 0]
    r2_s = K_inv @ H[:, 1]
    lam_W = np.linalg.norm(r1_s)
    r2_sc = r2_s / lam_W
    return float(np.linalg.norm(r2_sc))


def pose_from_calibrated_corners(
    corners: list, K: np.ndarray
) -> tuple:
    """
    Recover camera rotation R, cam_pos (in normalised world coords), and
    normalised floor dimensions from 4 floor corners + known K.
    No physical floor dimensions or camera height required.

    World coordinate convention
    ---------------------------
    The FIRST corner clicked (near-left) is the ORIGIN (0, 0, 0).
    Corners are arranged in CLOCKWISE order (as viewed from above):

        near-left  → (0,        0,       0)   = ORIGIN (1st click)
        near-right → (floor_w,  0,       0)   (2nd click)
        far-right  → (floor_w,  floor_h, 0)   (3rd click)
        far-left   → (0,        floor_h, 0)   (4th click)

    Coordinate system: X = left-to-right, Y = near-to-far (away from camera),
    Z = X×Y points UPWARD (away from floor toward camera), so cam_pos_z > 0.

    Input corner order must be [NL, NR, FR, FL]:
        near-left, near-right, far-right, far-left

    This is equivalent to clicking in image order:
        bottom-left, bottom-right, top-right, top-left
    when the camera looks along the pen length (e.g. Orbbec side mount)
    and the near end of the pen appears at the bottom of the image.

    Returns
    -------
    R         : (3,3) rotation (world → camera)
    cam_pos   : (3,)  camera centre in normalised world coords  (z > 0)
    floor_w   : 1.0   (normalised width)
    floor_h   : float (normalised depth, near-to-far = depth/width ratio)
    reproj_err: float reprojection error in pixels (sanity check)
    """
    corners = list(corners)

    # ── Step 1: floor aspect ratio from homography column norms ───────────────
    # This is convention-independent (ratio of widths).
    floor_w = 1.0
    floor_h = _floor_aspect(corners, K)

    # ── Step 2: world points with 1st corner (NL) as origin ────────────────────
    # Corners input = [NL, NR, FR, FL]; mapped to world origin at NL
    # with X = left→right, Y = near→far (away from camera)
    image_pts = np.float32([corners[0], corners[1], corners[2], corners[3]])  # NL, NR, FR, FL
    world_pts = np.float32([[0, 0, 0], [floor_w, 0, 0], [floor_w, floor_h, 0], [0, floor_h, 0]])

    # ── Step 3: IPPE pose estimation ──────────────────────────────────────────
    # We enforce that the FIRST corner clicked is always the origin.
    # Try both potential normal directions (camera above or below floor)
    # but always keep the same corner ordering.
    retval, rvecs, tvecs, _ = cv2.solvePnPGeneric(
        world_pts, image_pts,
        K.astype(np.float64), np.zeros(4),
        flags=cv2.SOLVEPNP_IPPE,
    )
    
    best = dict(score=-1e9, R=None, tvec=None, cam_pos=None)
    
    for rv, tv in zip(rvecs, tvecs):
        R_i, _ = cv2.Rodrigues(rv)
        t_i = tv.flatten()
        cam_i = -(R_i.T @ t_i)
        min_depth = float((R_i @ world_pts.T + t_i[:, None]).T[:, 2].min())
        # Prefer solutions with cam_pos_z > 0 (camera above floor);
        # within that, prefer the highest min_depth (most unambiguous).
        score = (1e6 if cam_i[2] > 0 else 0) + min_depth
        if score > best["score"]:
            best.update(score=score, R=R_i, tvec=t_i, cam_pos=cam_i)

    R, tvec_n, cam_pos = best["R"], best["tvec"], best["cam_pos"]

    if cam_pos[2] <= 0:
        print(f"[WARN] cam_pos_z={cam_pos[2]:.4f} ≤ 0 — camera appears below floor. "
              "Check corner click order: 1st=near-left, 2nd=near-right, 3rd=far-right, 4th=far-left.")

    # ── Step 4: Reprojection sanity check ─────────────────────────────────────
    proj, _ = cv2.projectPoints(world_pts, cv2.Rodrigues(R)[0], tvec_n, K, np.zeros(4))
    proj = proj.reshape(-1, 2)
    reproj_err = float(np.sqrt(((proj - image_pts) ** 2).sum(axis=1)).mean())
    print(f"[INFO] Reprojection error (normalised coords): {reproj_err:.2f} px")
    if reproj_err > 5.0:
        print(f"[WARN] High reprojection error ({reproj_err:.1f} px) — "
              "check corner order: 1st=near-left, 2nd=near-right, 3rd=far-right, 4th=far-left")

    return R, cam_pos, floor_w, floor_h, reproj_err


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Floor angle estimation using only camera calibration (no floor dimensions).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--image",    required=True,
                    help="Source image (raw, before undistortion)")
    ap.add_argument("--calib",    required=True,
                    help="Calibration file: checkerboard .npz or manual .json with K (and D)")
    ap.add_argument("--corners",  default=None,
                    help="Existing corners.json on the UNDISTORTED image (skips annotation)")
    ap.add_argument("--fisheye",  action="store_true", default=None,
                    help="Force fisheye undistortion model (auto-detected from D size otherwise)")
    ap.add_argument("--no-fisheye", dest="fisheye", action="store_false",
                    help="Force regular (Brown-Conrady) distortion model")
    ap.add_argument("--balance",  type=float, default=1.0,
                    help="Fisheye undistort balance (0=crop-all, 1=keep-all, default 1 keeps all floor corners visible)")
    ap.add_argument("--out-dir",  default="results",
                    help="Output directory")
    ap.add_argument("--scale",    type=float, default=1.0,
                    help="Display scale for the annotation window")
    ap.add_argument("--alpha",    type=float, default=0.55,
                    help="Overlay transparency (0=image, 1=colour map, default 0.55)")
    ap.add_argument("--plane-z",  type=float, default=0.0,
                    help="Floor plane z in world coords (default 0)")
    ap.add_argument("--rect-width-m", type=float, default=None,
                    help="Real-world width of the clicked floor rectangle in metres "
                         "(NL→NR distance, left-to-right in image). E.g. 4.8768 for full-pen fisheye, "
                         "2.4384 for side-mounted Orbbec. If not provided, uses normalised coords (width=1.0)")
    ap.add_argument("--rect-height-m", type=float, default=None,
                    help="Real-world depth/height of the clicked floor rectangle in metres "
                         "(NL→FL distance, near-to-far in image). E.g. 2.3694 for full-pen turret, "
                         "3.2512 for side-mounted Orbbec (2/3 of full pen width). "
                         "If not provided, calculated from aspect ratio = rect_width_m × (height/width)")
    ap.add_argument("--shift-origin", 
                    choices=["none", "top-left", "top-right", "bottom-right", "bottom-left"],
                    default="none",
                    help="Translate camera position to express origin at a different corner. "
                         "Input corners are still [near-left, near-right, far-right, far-left] "
                         "and are always kept as origin initially for best geometry/reprojection. "
                         "This option just shifts cam_pos after estimation. "
                         "(default: none = keep origin at 1st clicked corner)")
    ap.add_argument("--flip-y-axis", action="store_true",
                    help="Flip Y-axis direction so it increases from far→near (aligns with image Y). "
                         "Without this, Y increases from near→far (away from camera). "
                         "Angles remain correct (translation and reflection invariant).")
    # Detection CSV (same as pipeline.py)
    ap.add_argument("--detections",  default=None,
                    help="CSV with pig bboxes to annotate with angles")
    ap.add_argument("--bbox-col",    default="bbox")
    ap.add_argument("--sample-mode", default="lower_center",
                    choices=["center", "lower_center"])
    ap.add_argument("--elevation-convention", default="elevation",
                    choices=["elevation", "zenith"],
                    help="Convention for the elevation angle map: "
                         "'elevation' (default) 0°=horizon, 90°=nadir; "
                         "'zenith' 0°=nadir (directly under camera), 90°=horizon")
    # Validation
    ap.add_argument("--gt-calib",    default=None,
                    help="Ground-truth calibration JSON for validation")

    args = ap.parse_args()

    # ── Imports ───────────────────────────────────────────────────────────────
    from annotate_corners  import annotate, load_corners
    from compute_angle_map import compute_angle_map, make_overlay, augment_detections_csv
    import validate_calibrated as vc

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_path = Path(args.image)
    if not image_path.is_file():
        print(f"[ERROR] Image not found: {image_path}")
        sys.exit(1)

    # ── Step 1: Load calibration ──────────────────────────────────────────────
    print("\n[STEP 1/5] Loading calibration …")
    calib = load_calibration(args.calib)
    K = calib["K"]
    D = calib["D"]
    fisheye = calib["fisheye"] if args.fisheye is None else args.fisheye

    # ── Step 2: Undistort image ───────────────────────────────────────────────
    print("\n[STEP 2/5] Undistorting image …")
    img_raw = cv2.imread(str(image_path))
    if img_raw is None:
        print(f"[ERROR] Cannot read image: {image_path}")
        sys.exit(1)

    has_distortion = len(D) > 0 and np.any(np.abs(D) > 1e-8)
    if has_distortion:
        img_undist, K_eff = undistort_image(img_raw, K, D, fisheye, balance=args.balance)
        undist_path = out_dir / f"undistorted{image_path.suffix}"
        cv2.imwrite(str(undist_path), img_undist, [cv2.IMWRITE_JPEG_QUALITY, 97])
        print(f"[INFO] Undistorted image saved: {undist_path}")
        print(f"[INFO] Effective K after undistortion: "
              f"fx={K_eff[0,0]:.2f}  fy={K_eff[1,1]:.2f}  "
              f"cx={K_eff[0,2]:.2f}  cy={K_eff[1,2]:.2f}")
    else:
        img_undist = img_raw
        K_eff = K
        print("[INFO] No significant distortion — skipping undistortion")

    img_h, img_w = img_undist.shape[:2]

    # ── Step 3: Annotate corners on undistorted image ─────────────────────────
    corners_json = out_dir / "corners.json"
    if args.corners:
        corners_json = Path(args.corners)
        if not corners_json.is_file():
            print(f"[ERROR] Corners file not found: {corners_json}")
            sys.exit(1)
        print(f"\n[STEP 3/5] Using existing corners: {corners_json}")
    else:
        print("\n[STEP 3/5] Annotate pen-floor corners on the undistorted image")
        print("  Corner order: near-left, near-right, far-right, far-left")
        print("  (Equivalently: bottom-left, bottom-right, top-right, top-left)")
        print("  Coordinate origin: the FIRST corner you click (near-left) = (0, 0, 0)")
        print("  Subsequent clicks go CLOCKWISE around the floor rectangle")
        print("  'near'/'bottom' = the floor edge closer to the camera in the image")
        print("  Controls: click=place  r=reset  Enter/s=save  q=quit\n")
        result = annotate(undist_path if has_distortion else image_path,
                          corners_json, scale=args.scale)
        if result is None:
            print("[INFO] Cancelled.")
            sys.exit(0)

    corners_raw = load_corners(corners_json)
    corners = corners_raw["corners"] if isinstance(corners_raw, dict) else corners_raw
    print(f"[INFO] Corners loaded: {[[round(x,1), round(y,1)] for x,y in corners]}")

    # ── Step 4: Pose from homography (no floor dimensions needed) ────────────
    print("\n[STEP 4/5] Estimating camera pose from homography + known K …")
    R, cam_pos, floor_w, floor_h, reproj_err = pose_from_calibrated_corners(corners, K_eff)

    print(f"[INFO] Normalised floor: {floor_w:.3f} × {floor_h:.3f}  "
          f"(aspect ratio W:H = 1:{floor_h:.3f})")
    print(f"[INFO] Camera position (normalised): "
          f"[{cam_pos[0]:.4f}, {cam_pos[1]:.4f}, {cam_pos[2]:.4f}]")

    # Save camera params (compatible with existing load_camera_params format)
    rvec, _ = cv2.Rodrigues(R)
    t_vec   = -(R @ cam_pos)
    
    # Apply real-world scaling if rect_width_m or rect_height_m is provided.
    # World units are isotropic: 1 world unit = physical floor width / floor_w in all directions.
    # scale_factor_w: metres per world unit in X (= rect_width_m when provided, else 1.0).
    # scale_factor_h: metres per world unit in Y — same as X for an isotropic floor;
    #                 if rect_height_m is provided explicitly, derive it as rect_height_m / floor_h
    #                 (because floor_h world units in Y = rect_height_m metres).
    scale_factor_w = args.rect_width_m if args.rect_width_m is not None else 1.0
    scale_factor_h = (args.rect_height_m / floor_h) if args.rect_height_m is not None else scale_factor_w

    cam_pos_scaled = cam_pos * np.array([scale_factor_w, scale_factor_h, scale_factor_w])
    floor_w_scaled = floor_w * scale_factor_w
    floor_h_scaled = floor_h * scale_factor_h
    tvec_scaled = -(R @ cam_pos_scaled)  # recompute from scaled cam_pos (element-wise scaling of t is wrong for non-uniform scale)
    
    # Apply origin shift if requested
    # Current origin is always at 1st clicked corner (near-left in image coords)
    # Corner positions (in world coords with 1st click as origin):
    #   [0,0,0]=NL(1st/bottom-left), [W,0,0]=NR(2nd/bottom-right), 
    #   [W,H,0]=FR(3rd/top-right), [0,H,0]=FL(4th/top-left)
    shift_deltas = {
        "none":           np.array([0, 0, 0]),
        "top-left":       np.array([0, floor_h_scaled, 0]),       # FL at (0, H, 0)
        "top-right":      np.array([floor_w_scaled, floor_h_scaled, 0]),  # FR at (W, H, 0)
        "bottom-right":   np.array([floor_w_scaled, 0, 0]),       # NR at (W, 0, 0)
        "bottom-left":    np.array([0, 0, 0]),                   # NL at (0, 0, 0)
    }
    
    shift = shift_deltas[args.shift_origin]
    cam_pos_final = cam_pos_scaled - shift
    tvec_final = tvec_scaled - shift
    origin_label = args.shift_origin if args.shift_origin != "none" else "1st clicked corner"
    
    # Apply Y-axis flip if requested (to align with image coordinates)
    if args.flip_y_axis:
        flip_matrix = np.diag([1, -1, 1])
        cam_pos_final = flip_matrix @ cam_pos_final
        tvec_final = flip_matrix @ tvec_final
        print(f"[INFO] Y-axis flipped (now increases from far→near)")
    
    cam_params = {
        "K":             K_eff.tolist(),
        "R":             R.tolist(),
        "rvec":          rvec.flatten().tolist(),
        "tvec":          tvec_final.tolist(),
        "cam_pos":       cam_pos_final.tolist(),
        "floor_width_m": float(floor_w_scaled),
        "floor_height_m": float(floor_h_scaled),
        "reprojection_error_px": reproj_err,
        "note": f"Real-world coordinate system (width={floor_w_scaled:.4f} m, origin at {origin_label}). "
                "Angles are metric-accurate (scale-invariant).",
    }
    if args.rect_width_m is not None or args.rect_height_m is not None:
        cam_params["rect_width_m"] = args.rect_width_m
        if args.rect_height_m is not None:
            cam_params["rect_height_m"] = args.rect_height_m
        print(f"[INFO] Scaled to real-world:")
        if args.rect_width_m is not None:
            print(f"       rect_width_m (left-to-right) = {args.rect_width_m:.4f} m")
        if args.rect_height_m is not None:
            print(f"       rect_height_m (near-to-far) = {args.rect_height_m:.4f} m")
        else:
            print(f"       rect_height_m (from aspect) = {floor_h_scaled:.4f} m")
    if args.shift_origin != "none":
        cam_params["origin"] = args.shift_origin
        print(f"[INFO] Origin shifted to {origin_label}: cam_pos = {cam_pos_final.tolist()}")
    if args.flip_y_axis:
        cam_params["y_axis_flipped"] = True
    cam_params["elevation_convention"] = args.elevation_convention
    cam_json = out_dir / "camera_params.json"
    cam_json.write_text(json.dumps(cam_params, indent=2))
    print(f"[INFO] Camera params saved: {cam_json}")

    # ── Step 5: Angle maps ────────────────────────────────────────────────────
    print("\n[STEP 5/5] Computing dense angle maps …")
    az_map, el_map, valid_mask = compute_angle_map(
        img_h, img_w, K_eff, R, cam_pos,
        floor_w, floor_h,
        plane_z=args.plane_z,
        elevation_convention=args.elevation_convention,
    )
    np.save(str(out_dir / "azimuth_map.npy"),   az_map)
    np.save(str(out_dir / "elevation_map.npy"), el_map)
    np.save(str(out_dir / "valid_mask.npy"),    valid_mask)

    az_valid = az_map[valid_mask]
    el_valid = el_map[valid_mask]

    stats = {
        "valid_pixels":  int(valid_mask.sum()),
        "azimuth_deg":   {"min": float(az_valid.min()),  "max": float(az_valid.max()),
                          "mean": float(az_valid.mean()), "std": float(az_valid.std())},
        "elevation_deg": {"min": float(el_valid.min()),  "max": float(el_valid.max()),
                          "mean": float(el_valid.mean()), "std": float(el_valid.std())},
    }
    (out_dir / "angle_stats.json").write_text(json.dumps(stats, indent=2))

    az_vis = make_overlay(img_undist, az_map, -180, 180, "hsv",
                          args.alpha, "az (deg)")
    # Colorbar always 0–90° so the same colour means the same angle across cameras.
    # Both conventions map to [0, 90]: elevation counts up from 0 (horizon) to 90 (nadir),
    # zenith counts up from 0 (nadir) to 90 (horizon).
    el_label = "zenith (deg)" if args.elevation_convention == "zenith" else "el (deg)"
    el_vis = make_overlay(img_undist, el_map, 0.0, 90.0, "plasma", args.alpha, el_label)
    cv2.imwrite(str(out_dir / "azimuth_vis.jpg"),   az_vis, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(str(out_dir / "elevation_vis.jpg"), el_vis, [cv2.IMWRITE_JPEG_QUALITY, 95])

    print(f"  Azimuth  range: {az_valid.min():.1f} … {az_valid.max():.1f} deg  "
          f"(mean {az_valid.mean():.1f})")
    print(f"  Elevation range: {el_valid.min():.1f} … {el_valid.max():.1f} deg  "
          f"(mean {el_valid.mean():.1f})")

    # Per-bbox detection CSV
    if args.detections:
        det_path = Path(args.detections)
        out_csv  = out_dir / (det_path.stem + "_angles.csv")
        augment_detections_csv(det_path, out_csv, K_eff, R, cam_pos,
                               bbox_col=args.bbox_col,
                               sample_mode=args.sample_mode,
                               plane_z=args.plane_z,
                               elevation_convention=args.elevation_convention)

    # ── Optional validation ───────────────────────────────────────────────────
    if args.gt_calib:
        print("\n[VALIDATION] Comparing against ground-truth calibration …")
        gt = vc.load_gt_calib(Path(args.gt_calib))
        param_cmp = vc.compare_params(cam_params, gt)
        print(f"  Focal error    : {param_cmp['focal_err_pct']:.1f}%  "
              f"({param_cmp['focal_err_abs_px']:.1f} px)")
        print(f"  Rotation error : {param_cmp['rotation_err_deg']:.2f} deg")

        K_gt = np.array(gt["K"])
        if gt.get("calib_width") and gt.get("calib_height"):
            sx = img_w / gt["calib_width"]; sy = img_h / gt["calib_height"]
            K_gt[0, 0] *= sx; K_gt[0, 2] *= sx
            K_gt[1, 1] *= sy; K_gt[1, 2] *= sy
        az_gt, el_gt, valid_gt = compute_angle_map(
            img_h, img_w, K_gt, np.array(gt["R"]), np.array(gt["cam_pos"]),
            floor_w, floor_h, plane_z=args.plane_z)

        map_cmp = vc.compare_angle_maps(az_map, el_map, valid_mask,
                                        az_gt,  el_gt,  valid_gt)
        if map_cmp.get("valid_overlap_px", 0) > 0:
            print(f"  Azimuth MAE    : {map_cmp['azimuth_mae_deg']:.2f} deg")
            print(f"  Elevation MAE  : {map_cmp['elevation_mae_deg']:.2f} deg")

        val_dir = out_dir / "validation"
        val_dir.mkdir(exist_ok=True)
        (val_dir / "validation_report.json").write_text(
            json.dumps({**param_cmp, "angle_map_comparison": map_cmp}, indent=2))

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n[DONE] All outputs written to {out_dir}/")
    print(f"  undistorted{image_path.suffix:<12} undistorted source image (if D was non-zero)")
    print(f"  corners.json          floor corner annotations")
    print(f"  camera_params.json    R, K, cam_pos (normalised coords)")
    print(f"  azimuth_map.npy       per-pixel azimuth  (degrees)")
    print(f"  elevation_map.npy     per-pixel elevation (degrees)")
    print(f"  valid_mask.npy        floor-visible pixel mask")
    print(f"  azimuth_vis.jpg       HSV-coloured azimuth overlay")
    print(f"  elevation_vis.jpg     plasma-coloured elevation overlay")
    print()


if __name__ == "__main__":
    main()
