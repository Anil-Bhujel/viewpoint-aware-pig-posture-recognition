#!/usr/bin/env python3
"""
visualize_viewpoint_angles.py
------------------------------
Visualize viewpoint angle information (azimuth and elevation) from bounding box 
annotations as a single viewing ray direction arrow.

Modes:
  - --crop (default): Each instance as a separate cropped bbox image
  - --no-crop: All instances on single full image with bbox overlays
  
Overlays a single arrow showing:
  - Direction: the ray direction determined by azimuth angle (sin_az, cos_az)
  - Length: proportional to elevation angle magnitude
  
This clearly shows how the same object appears at different angles from different 
camera viewpoints.

Usage
-----
    # Cropped bbox regions (default) - separate files for each instance
    python visualize_viewpoint_angles.py \
        --image-dir /path/to/images \
        --csv annotations.csv \
        --output-dir output_samples/ \
        --crop --arrow-scale 100

    # Full images with all instances (--no-crop mode)
    python visualize_viewpoint_angles.py \
        --image-dir /path/to/images \
        --csv annotations.csv \
        --output-dir output_samples/ \
        --no-crop --arrow-scale 80

    # Compare same posture from multiple cameras
    python visualize_viewpoint_angles.py \
        --image-dir pen1_turret_images/ \
        --csv pen1_turret_annotations.csv \
        --output-dir results/comparison_turret/ \
        --no-crop --arrow-scale 120

    python visualize_viewpoint_angles.py \
        --image-dir pen1_orbbec_images/ \
        --csv pen1_orbbec_annotations.csv \
        --output-dir results/comparison_orbbec/ \
        --no-crop --arrow-scale 120
"""

import argparse
import sys
from pathlib import Path
import csv
import json
import re

import cv2
import numpy as np


def parse_bbox(bbox_str):
    """
    Parse bounding box from various formats.
    Supports: "x,y,w,h" or "[x,y,w,h]" or "{'x':x,'y':y,'w':w,'h':h}"
    
    Returns (x, y, w, h) as integers.
    """
    bbox_str = str(bbox_str).strip()
    
    try:
        # Try JSON-like format
        if bbox_str.startswith('{') or bbox_str.startswith('['):
            data = json.loads(bbox_str.replace("'", '"'))
            if isinstance(data, dict):
                return int(data['x']), int(data['y']), int(data['w']), int(data['h'])
            elif isinstance(data, (list, tuple)) and len(data) >= 4:
                return int(data[0]), int(data[1]), int(data[2]), int(data[3])
    except:
        pass
    
    # Try comma-separated format
    try:
        parts = [float(x.strip()) for x in bbox_str.split(',')]
        if len(parts) >= 4:
            return int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
    except:
        pass
    
    raise ValueError(f"Cannot parse bbox: {bbox_str}")


def compute_padded_bbox(bbox, img_shape, pad_fraction=0.0):
    """
    Compute padded bounding box coordinates, clipped to image bounds.
    
    Parameters
    ----------
    bbox : tuple (x, y, w, h)
        Bounding box
    img_shape : tuple (height, width)
        Image shape
    pad_fraction : float
        Fraction of bbox width/height to add as padding on each side
        (e.g., 0.1 = 10% padding)
    
    Returns
    -------
    (x1, y1, x2, y2) : clipped to image bounds
    """
    x, y, w, h = map(float, bbox)
    h_img, w_img = img_shape[:2]
    
    # Compute padding
    pad_w = w * pad_fraction
    pad_h = h * pad_fraction
    
    # Compute padded coordinates
    x1 = max(0, int(x - pad_w))
    y1 = max(0, int(y - pad_h))
    x2 = min(w_img, int(x + w + pad_w))
    y2 = min(h_img, int(y + h + pad_h))
    
    return (x1, y1, x2, y2)


def compute_square_crop(bbox, img_shape, pad_fraction=0.0):
    """
    Compute square crop region centered on bbox, clipped to image bounds.
    
    Parameters
    ----------
    bbox : tuple (x, y, w, h)
        Bounding box
    img_shape : tuple (height, width)
        Image shape
    pad_fraction : float
        Fraction of bbox width/height to add as padding (applied before squaring)
    
    Returns
    -------
    (x1, y1, x2, y2) : square crop region clipped to image bounds
    """
    x, y, w, h = map(float, bbox)
    h_img, w_img = img_shape[:2]
    
    # Apply padding first if requested
    pad_w = w * pad_fraction
    pad_h = h * pad_fraction
    padded_w = w + 2 * pad_w
    padded_h = h + 2 * pad_h
    
    # Compute square side (use max of padded dimensions)
    side = max(padded_w, padded_h)
    
    # Center on bbox center
    cx = x + w / 2.0
    cy = y + h / 2.0
    
    # Compute square boundaries centered on bbox center
    x1 = cx - side / 2.0
    y1 = cy - side / 2.0
    x2 = x1 + side
    y2 = y1 + side
    
    # Clip to image bounds
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(w_img, int(x2))
    y2 = min(h_img, int(y2))
    
    return (x1, y1, x2, y2)


def crop_with_black_padding(img, bbox, pad_fraction=0.0, square_crop=True):
    """
    Crop region with black padding to maintain aspect ratio and keep bbox centered.
    
    Does NOT clip to image bounds. Instead, pads cropped region with black pixels
    where the crop extends beyond the image.
    
    Parameters
    ----------
    img : ndarray (H, W, 3)
        Image to crop
    bbox : tuple (x, y, w, h)
        Bounding box
    pad_fraction : float
        Fraction of bbox width/height to add as padding on each side
    square_crop : bool
        If True, crop to square; else crop to padded rectangular bbox
    
    Returns
    -------
    cropped_img : ndarray (side, side, 3) or (H, W, 3)
        Cropped image with black padding where needed
    crop_coords : tuple (x1_img, y1_img, x2_img, y2_img, x1_crop, y1_crop, x2_crop, y2_crop)
        Image bounds and crop bounds (for mapping coordinates back to original)
    center_in_crop : tuple (cx_crop, cy_crop)
        Bbox center in crop coordinates
    """
    h_img, w_img = img.shape[:2]
    x, y, w, h = map(float, bbox)
    
    # Apply padding
    pad_w = w * pad_fraction
    pad_h = h * pad_fraction
    padded_w = w + 2 * pad_w
    padded_h = h + 2 * pad_h
    
    if square_crop:
        # Compute square side
        side = max(padded_w, padded_h)
        crop_h, crop_w = int(side), int(side)
    else:
        crop_h, crop_w = int(padded_h), int(padded_w)
    
    # Center on bbox center (in image coordinates)
    cx = x + w / 2.0
    cy = y + h / 2.0
    
    # Desired crop boundaries in image coordinates (may extend beyond image)
    x1_desired = cx - crop_w / 2.0
    y1_desired = cy - crop_h / 2.0
    x2_desired = x1_desired + crop_w
    y2_desired = y1_desired + crop_h
    
    # Clipped boundaries (what we can actually extract from the image)
    x1_clipped = max(0, int(x1_desired))
    y1_clipped = max(0, int(y1_desired))
    x2_clipped = min(w_img, int(x2_desired))
    y2_clipped = min(h_img, int(y2_desired))
    
    # Extract the valid region from the image
    valid_crop = img[y1_clipped:y2_clipped, x1_clipped:x2_clipped].copy()
    
    # Create output image with black padding
    output_img = np.zeros((crop_h, crop_w, 3), dtype=img.dtype)
    
    # Calculate where to place the valid region in the output image
    offset_x = int(x1_clipped - x1_desired)
    offset_y = int(y1_clipped - y1_desired)
    
    # Place valid crop in padded image
    output_img[offset_y:offset_y + valid_crop.shape[0],
               offset_x:offset_x + valid_crop.shape[1]] = valid_crop
    
    # Bbox center in crop coordinates
    center_x_in_crop = (cx - x1_desired)
    center_y_in_crop = (cy - y1_desired)
    
    crop_coords = (x1_clipped, y1_clipped, x2_clipped, y2_clipped,
                   int(x1_desired), int(y1_desired), int(x2_desired), int(y2_desired))
    
    return output_img, crop_coords, (center_x_in_crop, center_y_in_crop)


def load_annotations(csv_path, image_col='image_path', bbox_col='bbox',
                    sin_az_col='sin_az', cos_az_col='cos_az',
                    sin_el_col='sin_el', cos_el_col='cos_el'):
    """
    Load annotations from CSV file.
    
    Returns list of dicts with keys: image_path, bbox, sin_az, cos_az, sin_el, cos_el
    """
    annotations = []
    
    has_image_arrows = False
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        _has_arrow_cols = 'arrow_u' in fieldnames and 'arrow_v' in fieldnames
        for i, row in enumerate(reader):
            try:
                # Parse bbox
                bbox = parse_bbox(row[bbox_col])
                
                # Get angle values
                sin_az = float(row[sin_az_col])
                cos_az = float(row[cos_az_col])
                sin_el = float(row[sin_el_col])
                cos_el = float(row[cos_el_col])

                # Image-space arrow direction (available in PnP-derived CSVs)
                arrow_u = float('nan')
                arrow_v = float('nan')
                if _has_arrow_cols:
                    try:
                        arrow_u = float(row['arrow_u'])
                        arrow_v = float(row['arrow_v'])
                        if np.isfinite(arrow_u) and np.isfinite(arrow_v):
                            has_image_arrows = True
                    except (ValueError, TypeError):
                        pass
                
                annotations.append({
                    'image_path': row[image_col],
                    'bbox': bbox,
                    'sin_az': sin_az,
                    'cos_az': cos_az,
                    'sin_el': sin_el,
                    'cos_el': cos_el,
                    'arrow_u': arrow_u,
                    'arrow_v': arrow_v,
                    'row_index': i,
                })
            except Exception as e:
                print(f"[WARN] Row {i}: {e}")
                continue

    if not has_image_arrows:
        print("[WARN] CSV has no valid arrow_u/arrow_v columns — falling back to "
              "world-space azimuth for arrow direction.")
        print("[WARN] This is only correct for overhead cameras where world X/Y "
              "aligns with image X/Y.")
        print("[WARN] For correct arrows, regenerate the CSV with "
              "add_camera_angles_from_calib_multicam.py (which outputs arrow_u/arrow_v).")
    
    return annotations


def draw_viewpoint_arrow(image, center, sin_az, cos_az, sin_el, cos_el,
                        arrow_scale=100, thickness=3, color=(0, 255, 0), font_scale=1.0,
                        angle_convention="elevation", display_convention="elevation",
                        arrow_u=None, arrow_v=None):
    """
    Draw single arrow showing 3D viewing direction at center of image.

    Direction determined by:
      - arrow_u / arrow_v (image-space unit vector, from PnP CSV) when available
      - otherwise sin_az / cos_az (world-space fallback for legacy CSVs)
    Length determined by elevation angle magnitude.

    Parameters
    ----------
    image : ndarray (H, W, 3) RGB or BGR image to draw on (modified in-place)
    center : tuple (cx, cy) center point in image coordinates
    sin_az, cos_az : float azimuth sine and cosine (world-space, fallback only)
    sin_el, cos_el : float elevation sine and cosine (ray vertical angle)
    arrow_scale : float scale factor for maximum arrow length in pixels
    thickness : int line thickness
    color : tuple BGR color for arrow
    font_scale : float text size scale (default 1.0)
    angle_convention : str  Convention of angles in input: "elevation" or "zenith"
    display_convention : str  "elevation" or "zenith"
    arrow_u, arrow_v : float or None
        Image-space unit vector toward camera (preferred over world-space azimuth)
    """
    cx, cy = int(center[0]), int(center[1])

    # Prefer image-space direction (arrow_u/arrow_v) when available and finite
    use_image_space = (
        arrow_u is not None and arrow_v is not None
        and np.isfinite(arrow_u) and np.isfinite(arrow_v)
    )
    if use_image_space:
        dir_x = float(arrow_u)
        dir_y = float(arrow_v)
    else:
        # Fallback: world-space azimuth (approximate — only correct for overhead
        # cameras where world X/Y aligns with image X/Y).  Arrow drawn in orange.
        az_mag = np.sqrt(sin_az**2 + cos_az**2)
        if az_mag < 1e-3:
            return False   # signal fallback used
        dir_x = sin_az / az_mag
        dir_y = cos_az / az_mag
        color = (0, 140, 255)   # orange — indicates world-space fallback
    
    # Convert input angles to display convention if needed
    if angle_convention != display_convention:
        if angle_convention == "elevation" and display_convention == "zenith":
            # Convert from elevation to zenith: sin(zenith) = cos(elevation), cos(zenith) = sin(elevation)
            sin_el, cos_el = cos_el, sin_el
        elif angle_convention == "zenith" and display_convention == "elevation":
            # Convert from zenith to elevation: sin(elevation) = cos(zenith), cos(elevation) = sin(zenith)
            sin_el, cos_el = cos_el, sin_el
    
    # Elevation angle determines arrow length and thickness
    # Extract elevation angle from sin_el (0° to 90°) and normalize to 0-1
    el_rad = np.arcsin(np.clip(sin_el, -1, 1))  # elevation angle in radians
    el_normalized = el_rad / (np.pi / 2)  # normalize: 0° → 0, 90° → 1
    arrow_length = arrow_scale * el_normalized
    
    # Scale arrow thickness with elevation angle but maintain visibility
    # minimum 2px at low angles, scales up to thickness at 90°
    arrow_thickness = max(2, int(2 + thickness * el_normalized))
    
    # Arrow endpoint
    end_x = cx + dir_x * arrow_length
    end_y = cy + dir_y * arrow_length
    
    # Draw arrow with scaled thickness
    cv2.arrowedLine(image, (cx, cy), (int(end_x), int(end_y)),
                   color, arrow_thickness, tipLength=0.25)
    
    # Draw circle at center
    cv2.circle(image, (cx, cy), 3, color, -1)
    
    # Compute angles for labels
    az_deg = np.degrees(np.arctan2(sin_az, cos_az))
    el_deg = np.degrees(np.arcsin(np.clip(sin_el, -1, 1)))
    
    # Labels
    # cv2.putText(image, f"Az: {az_deg:.1f}deg El: {el_deg:.1f}deg",
    #            (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2, cv2.LINE_AA)


def process_annotation(ann, image_dir, output_dir, arrow_scale=100, crop=True, font_scale=1.0, pad_fraction=0.0, square_crop=False,
                       angle_convention="elevation", display_convention="elevation"):
    """
    Process a single annotation: optionally crop bbox (with padding) and overlay angle arrow.
    
    Parameters
    ----------
    ann : dict
        Annotation with image_path, bbox, angles
    image_dir : str
        Directory with images
    output_dir : str
        Output directory
    arrow_scale : float
        Arrow length scale
    crop : bool
        Whether to crop to bbox region
    font_scale : float
        Font size for text
    pad_fraction : float
        Padding fraction for bbox
    square_crop : bool
        If True, crop to square region (max of width/height); else rectangular
    angle_convention : str
        Convention of angles in CSV: "elevation" or "zenith"
    display_convention : str
        Convention to display angles in: "elevation" or "zenith"
    
    Returns (success: bool, output_path: str)
    """
    image_name = ann['image_path']
    image_path = Path(image_dir) / image_name
    
    if not image_path.is_file():
        return False, f"Image not found: {image_path}"
    
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        return False, f"Cannot read image: {image_path}"
    
    h, w = img.shape[:2]
    x, y, bw, bh = ann['bbox']
    
    # Validate bbox
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w, x + bw), min(h, y + bh)
    
    if x2 <= x1 or y2 <= y1:
        return False, f"Invalid bbox: {ann['bbox']}"
    
    # Apply padding and/or square crop if requested
    if crop:
        if square_crop:
            # Use new black-padding method for square crops
            display_img, crop_coords, arrow_center = crop_with_black_padding(
                img, ann['bbox'], pad_fraction=pad_fraction, square_crop=True
            )
        else:
            # Original rectangular cropping with clipping to image bounds
            if pad_fraction > 0:
                x1, y1, x2, y2 = compute_padded_bbox(ann['bbox'], img.shape, pad_fraction)
            else:
                x, y, bw, bh = ann['bbox']
                x1, y1 = max(0, int(x)), max(0, int(y))
                x2, y2 = min(w, int(x + bw)), min(h, int(y + bh))
            
            display_img = img[y1:y2, x1:x2].copy()
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            arrow_center = ((x2 - x1) / 2, (y2 - y1) / 2)
    else:
        # Use full image (no cropping)
        display_img = img.copy()
        # Bbox center (in original image coordinates)
        x, y, bw, bh = ann['bbox']
        center_x = x + bw / 2
        center_y = y + bh / 2
        x1, y1 = max(0, int(x)), max(0, int(y))
        x2, y2 = min(w, int(x + bw)), min(h, int(y + bh))
        # Draw bbox rectangle
        cv2.rectangle(display_img, (x1, y1), (x2, y2), (200, 200, 200), 2)
        arrow_center = (center_x, center_y)
    
    # Draw angle arrow
    try:
        draw_viewpoint_arrow(display_img, arrow_center,
                            ann['sin_az'], ann['cos_az'],
                            ann['sin_el'], ann['cos_el'],
                            arrow_scale=arrow_scale, font_scale=font_scale,
                            angle_convention=angle_convention,
                            display_convention=display_convention,
                            arrow_u=ann.get('arrow_u'), arrow_v=ann.get('arrow_v'))
    except Exception as e:
        return False, f"Error drawing arrow: {e}"
    
    # Save output
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Output filename: original_name + _bbox + row_index (or _full if not cropped)
    stem = Path(image_name).stem
    suffix = Path(image_name).suffix
    crop_tag = "bbox" if crop else "full"
    output_name = f"{stem}_{crop_tag}_{ann['row_index']:04d}{suffix}"
    output_path = output_dir / output_name
    
    cv2.imwrite(str(output_path), display_img)
    return True, str(output_path)


def process_image_with_all_annotations(image_name, anns, image_dir, output_dir, arrow_scale=100, font_scale=1.0,
                                       angle_convention="elevation", display_convention="elevation"):
    """
    Process a full image with all annotations (used for --no-crop mode).
    Draw all instances and their arrows on a single image.
    
    Parameters
    ----------
    angle_convention : str
        Convention of angles in CSV: "elevation" or "zenith"
    display_convention : str
        Convention to display angles in: "elevation" or "zenith"
    
    Returns (success: bool, output_path: str, num_instances: int)
    """
    image_path = Path(image_dir) / image_name
    
    if not image_path.is_file():
        return False, f"Image not found: {image_path}", 0
    
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        return False, f"Cannot read image: {image_path}", 0
    
    display_img = img.copy()
    h, w = display_img.shape[:2]

    # Save output
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    error_count = 0
    for ann in anns:
        try:
            x, y, bw, bh = ann['bbox']
            
            # Validate bbox
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w, x + bw), min(h, y + bh)
            
            if x2 <= x1 or y2 <= y1:
                error_count += 1
                continue
            
            # Draw bbox rectangle
            cv2.rectangle(display_img, (x1, y1), (x2, y2), (200, 200, 200), 2)
            
            # Bbox center (in full image coordinates)
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Draw angle arrow
            draw_viewpoint_arrow(display_img, (center_x, center_y),
                                ann['sin_az'], ann['cos_az'],
                                ann['sin_el'], ann['cos_el'],
                                arrow_scale=arrow_scale, font_scale=font_scale,
                                angle_convention=angle_convention,
                                display_convention=display_convention,
                                arrow_u=ann.get('arrow_u'), arrow_v=ann.get('arrow_v'))
            
        except Exception as e:
            error_count += 1
            continue
    
    num_instances = len(anns) - error_count
    
    
    # Output filename: original_name + _annotated
    stem = Path(image_name).stem
    suffix = Path(image_name).suffix
    output_name = f"{stem}_annotated{suffix}"
    output_path = output_dir / output_name
    
    cv2.imwrite(str(output_path), display_img)
    return True, str(output_path), num_instances


def main():
    ap = argparse.ArgumentParser(
        description="Visualize viewpoint angles on cropped bounding box samples.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--image-dir", required=True,
                    help="Directory containing source images")
    ap.add_argument("--csv", required=True,
                    help="CSV file with bounding box and angle annotations")
    ap.add_argument("--output-dir", default="output_viewpoint_samples/",
                    help="Output directory for cropped samples with angle overlays")
    
    # CSV column names
    ap.add_argument("--csv-image-col", default="image_id",
                    help="CSV column name for image path")
    ap.add_argument("--csv-bbox-col", default="bbox",
                    help="CSV column name for bounding box (format: x,y,w,h or JSON)")
    ap.add_argument("--csv-sin-az-col", default="azimuth_sin",
                    help="CSV column name for sin(azimuth)")
    ap.add_argument("--csv-cos-az-col", default="azimuth_cos",
                    help="CSV column name for cos(azimuth)")
    ap.add_argument("--csv-sin-el-col", default="elevation_sin",
                    help="CSV column name for sin(elevation)")
    ap.add_argument("--csv-cos-el-col", default="elevation_cos",
                    help="CSV column name for cos(elevation)")
    
    # Filtering
    ap.add_argument("--filter-images", nargs="*", default=None,
                    help="Filter to only these image names (space-separated)")
    ap.add_argument("--filter-regex", default=None,
                    help="Filter image names by regex pattern")
    ap.add_argument("--sample-limit", type=int, default=None,
                    help="Limit number of samples to process")
    
    # Visualization
    ap.add_argument("--arrow-scale", type=float, default=50.0,
                    help="Scale factor for arrow length in pixels (default 100)")
    ap.add_argument("--arrow-thickness", type=int, default=2,
                    help="Arrow line thickness in pixels")
    ap.add_argument("--font-scale", type=float, default=1.0,
                    help="Font size scale for angle text (default 1.0)")
    ap.add_argument("--pad-fraction", type=float, default=0.0,
                    help="Padding fraction for bbox cropping (0.0-1.0). E.g., 0.2 = 20%% padding")
    ap.add_argument("--square-crop", action="store_true", default=False,
                    help="Crop to square region (max of width/height) for uniform aspect ratio")
    ap.add_argument("--crop", action="store_true", default=True,
                    help="Crop bounding box region (default True)")
    ap.add_argument("--no-crop", dest="crop", action="store_false",
                    help="Show full image instead of cropping bbox")
    
    # Angle conventions
    ap.add_argument("--angle-convention", default="elevation",
                    choices=["elevation", "zenith"],
                    help="Convention of angles stored in CSV: elevation (0°=horizon, 90°=nadir) | zenith (0°=nadir, 90°=horizon) [default: elevation]")
    ap.add_argument("--display-convention", default="elevation",
                    choices=["elevation", "zenith"],
                    help="Convention to display angles in visualization [default: elevation]. Arrow length scaled accordingly.")
    
    args = ap.parse_args()
    
    # ── Load annotations ──────────────────────────────────────────────────────
    print(f"\n[1/3] Loading annotations from {args.csv} …")
    try:
        annotations = load_annotations(
            args.csv,
            image_col=args.csv_image_col,
            bbox_col=args.csv_bbox_col,
            sin_az_col=args.csv_sin_az_col,
            cos_az_col=args.csv_cos_az_col,
            sin_el_col=args.csv_sin_el_col,
            cos_el_col=args.csv_cos_el_col,
        )
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    
    print(f"[INFO] Loaded {len(annotations)} annotations")
    
    # ── Filter annotations ────────────────────────────────────────────────────
    if args.filter_images:
        filter_set = set(args.filter_images)
        annotations = [a for a in annotations if a['image_path'] in filter_set]
        print(f"[INFO] Filtered to {len(annotations)} by image names")
    
    if args.filter_regex:
        pattern = re.compile(args.filter_regex)
        annotations = [a for a in annotations if pattern.search(a['image_path'])]
        print(f"[INFO] Filtered to {len(annotations)} by regex: {args.filter_regex}")
    
    if args.sample_limit:
        annotations = annotations[:args.sample_limit]
        print(f"[INFO] Limited to {len(annotations)} samples")
    
    # ── Process annotations ───────────────────────────────────────────────────
    print(f"\n[2/3] Processing {len(annotations)} samples …")
    print(f"[INFO] Angle convention: input={args.angle_convention}, display={args.display_convention}")
    success_count = 0
    fail_count = 0
    
    if args.crop:
        # Process each annotation individually (cropped mode)
        for i, ann in enumerate(annotations):
            success, result = process_annotation(
                ann, args.image_dir, args.output_dir,
                arrow_scale=args.arrow_scale,
                crop=True,
                font_scale=args.font_scale,
                pad_fraction=args.pad_fraction,
                square_crop=args.square_crop,
                angle_convention=args.angle_convention,
                display_convention=args.display_convention
            )
            
            if success:
                success_count += 1
                if (i + 1) % max(1, len(annotations) // 10) == 0:
                    print(f"  [{i+1}/{len(annotations)}] {result}")
            else:
                fail_count += 1
                print(f"  [WARN] Row {ann['row_index']}: {result}")
    else:
        # Group annotations by image and process each image once (full image mode)
        images_dict = {}
        for ann in annotations:
            img_name = ann['image_path']
            if img_name not in images_dict:
                images_dict[img_name] = []
            images_dict[img_name].append(ann)
        
        print(f"[INFO] Found {len(images_dict)} unique images with {len(annotations)} instances total")
        
        for i, (img_name, anns) in enumerate(images_dict.items()):
            success, result, num_inst = process_image_with_all_annotations(
                img_name, anns, args.image_dir, args.output_dir,
                arrow_scale=args.arrow_scale,
                font_scale=args.font_scale,
                angle_convention=args.angle_convention,
                display_convention=args.display_convention
            )
            
            if success:
                success_count += 1
                print(f"  [{i+1}/{len(images_dict)}] {num_inst} instances: {result}")
            else:
                fail_count += 1
                print(f"  [WARN] {img_name}: {result}")
    
    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n[3/3] Done!")
    print(f"  ✓ Processed: {success_count}")
    print(f"  ✗ Failed: {fail_count}")
    print(f"  Output: {args.output_dir}/")
    if args.crop:
        print(f"\nOutput mode: Cropped bbox regions")
        print(f"  Each file: one instance in bbox crop")
    else:
        print(f"\nOutput mode: Full images with all instances")
        print(f"  Each file: entire image with all annotated instances")
    print(f"\nVisualization:")
    print(f"  Single arrow from each bbox center showing viewing ray:")
    print(f"    - Direction: determined by azimuth angle (sin_az, cos_az)")
    print(f"    - Length: proportional to elevation angle (0°=no arrow, 90°=max length)")
    print(f"             Normalized scale: elevation_angle / 90° × arrow_scale")
    print(f"  Labels show: Azimuth (degrees) and Elevation (degrees)")


if __name__ == "__main__":
    main()
