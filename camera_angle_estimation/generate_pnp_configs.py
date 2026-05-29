#!/usr/bin/env python3
"""
generate_pnp_configs.py

Generate camera configuration files from PnP-estimated camera parameters.

This script generates JSON config files for multi-camera systems where each camera's
position and orientation have been estimated using PnP (Perspective-n-Point) calibration.

IMPORTANT: The pen_floor region is a REFERENCE calibration area. However, angle 
computation is NOT limited to this region. The camera parameters (rvec, tvec, etc.) 
from PnP estimation enable angle computation for ANY point in the image, including 
positions where the animal stands outside the calibrated floor region.

When computing angles:
  - Points within pen_floor: Validated against known ground truth
  - Points beyond pen_floor: Still computable using camera parameters
  - Example: A standing pig with bbox center outside the floor region can still have 
    its angle computed

Usage:
cd /path/to/floor_angle_estimation
    python generate_pnp_configs.py \\
        --output-dir ./configs \\
        --elevation-convention zenith \\
        --camera-params cam1:fisheye:path/to/cam1_calib.npz:path/to/cam1_params.json \\
                        cam2:fisheye:path/to/cam2_calib.npz:path/to/cam2_params.json \\
                        cam3:pinhole:path/to/cam3_calib.ini:path/to/cam3_params.json

Example (mixed camera types - fisheye and pinhole):
    python generate_pnp_configs.py \\
        --output-dir ./output \\
        --output-name pen1_config \\
        --elevation-convention zenith \\
        --camera-params pen1_tur_cam1:fisheye:gt_camera_parameters/pen1_tur_cam1_calibration.npz:results/pen1_tur_cam1/camera_params.json \\
                        pen1_tur_cam2:fisheye:gt_camera_parameters/pen1_tur_cam2_calibration.npz:results/pen1_tur_cam2/camera_params.json \\
                        pen1_orb_cam1:pinhole:gt_camera_parameters/p1c1_orb.ini:results/pen1_orb_cam1/camera_params.json \\
                        pen1_orb_cam2:pinhole:gt_camera_parameters/p1c2_orb.ini:results/pen1_orb_cam2/camera_params.json

Format: camera_name:model_type:path/to/calib_file:path/to/pnp/params.json
Supported models: fisheye, pinhole
Elevation convention: elevation or zenith (default: elevation)
Each camera specifies its own calibration file and PnP parameters path.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Tuple, List

# Image dimensions for each camera type
CALIB_DIMS = {
    "fisheye": (1280, 720),
    "pinhole": (1920, 1080),
}


def load_pnp_params(camera_params_path: str) -> dict:
    """Load PnP parameters from camera_params.json."""
    pnp_path = Path(camera_params_path)
    
    if not pnp_path.is_file():
        raise FileNotFoundError(f"Camera params file not found: {pnp_path}")
    
    with open(pnp_path) as f:
        data = json.load(f)
    
    return data


def build_camera_config(
    camera_name: str,
    camera_params_path: str,
    calib_path: str,
    calib_type: str
) -> dict:
    """Build camera config entry from PnP results.
    
    Args:
        camera_name: Name of the camera
        camera_params_path: Path to the PnP-estimated camera_params.json
        calib_path: Path to calibration file for this specific camera
        calib_type: Camera model type ('fisheye' or 'pinhole')
    
    The PnP results contain the camera's position and orientation (rvec, tvec, etc.)
    which enable angle computation for ANY point in the image, not just within the 
    calibrated floor region. The pnp_results_path references the full camera parameters
    needed for this computation.
    """
    if calib_type not in CALIB_DIMS:
        raise ValueError(f"Invalid calibration type '{calib_type}'. Must be one of: {list(CALIB_DIMS.keys())}")
    
    pnp_data = load_pnp_params(camera_params_path)
    
    # Extract floor reference dimensions from PnP results
    # Note: These are REFERENCE dimensions from the calibrated floor region.
    # Angle computation works beyond this region using the full camera parameters.
    calib_w, calib_h = CALIB_DIMS[calib_type]
    
    return {
        "model": calib_type,
        "calib_path": calib_path,
        "calib_width": calib_w,
        "calib_height": calib_h,
        "pnp_results_path": camera_params_path,
        "notes": f"PnP-estimated: {calib_type.upper()} camera. Angle computation enabled for entire image."
    }


def generate_camera_config_file(
    output_file: Path,
    camera_names: List[str],
    camera_params_map: Dict[str, Tuple[str, str, str]],
    elevation_convention: str = "elevation"
) -> dict:
    """Generate a comprehensive camera configuration file.
    
    Args:
        output_file: Output file path
        camera_names: List of camera names
        camera_params_map: Dict mapping camera_name -> (calib_type, calib_path, path_to_params_json)
        elevation_convention: Elevation angle convention ('elevation' or 'zenith'). Default: 'elevation'
    
    The pen_floor metadata is extracted from the calibration region but is ONLY
    a reference. The actual angle computation uses the full camera parameters 
    (rvec, tvec, etc.) and works for any point in the image, including positions 
    outside the calibrated floor region (e.g., when a pig stands up or moves beyond 
    the pen boundaries).
    """
    
    if elevation_convention not in ["elevation", "zenith"]:
        raise ValueError(f"Invalid elevation convention '{elevation_convention}'. Must be 'elevation' or 'zenith'")
    
    # Load first camera's PnP params to get floor reference dimensions
    first_camera_name = camera_names[0]
    first_calib_type, first_calib_path, first_params_path = camera_params_map[first_camera_name]
    first_camera_params = load_pnp_params(first_params_path)
    floor_width_m = float(first_camera_params.get("rect_width_m", 1.0))
    floor_height_m = float(first_camera_params.get("floor_height_m", 0.5))
    
    config = {
        "elevation_convention": elevation_convention,
        "pen_floor": {
            "width_m": floor_width_m,
            "height_m": floor_height_m,
            "note": "Reference calibration region. Angle computation enabled for entire image."
        },
        "cameras": {}
    }
    
    for cam_name in camera_names:
        if cam_name not in camera_params_map:
            raise ValueError(f"Camera params not provided for: {cam_name}")
        
        calib_type, calib_path, cam_params_path = camera_params_map[cam_name]
        config["cameras"][cam_name] = build_camera_config(
            cam_name, cam_params_path, calib_path, calib_type
        )
    
    return config



def main():
    """Main entry point with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Generate camera configuration files from PnP-estimated camera parameters. "
                    "Supports mixed camera types (fisheye and pinhole) in a single config.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""

        """
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for configuration files"
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="camera_config",
        help="Name for the output config file (without .json extension). Default: 'camera_config'"
    )
    parser.add_argument(
        "--elevation-convention",
        type=str,
        default="elevation",
        choices=["elevation", "zenith"],
        help="Elevation angle convention used in this configuration. Default: 'elevation'"
    )
    parser.add_argument(
        "--camera-params",
        type=str,
        nargs="+",
        required=True,
        help="Camera specs in format 'name:type:calib_path:params_path' (space-separated). "
             "Type: fisheye or pinhole. "
             "Example: cam1:fisheye:path/to/calib.npz:path/to/params.json"
    )
    
    args = parser.parse_args()
    
    # Parse camera params map: camera_name -> (calib_type, calib_path, params_path)
    camera_params_map = {}
    camera_names = []
    
    for item in args.camera_params:
        parts = item.split(":", 3)  # Split into max 4 parts (name, type, calib_path, params_path)
        
        if len(parts) != 4:
            parser.error(
                f"Invalid camera params format: '{item}'. "
                "Use 'camera_name:type:path/to/calib_file:path/to/params.json' "
                "(type must be 'fisheye' or 'pinhole')"
            )
        
        cam_name, cam_type, cam_calib_path, cam_params_path = parts
        
        if cam_type not in CALIB_DIMS:
            parser.error(
                f"Invalid camera type '{cam_type}' for {cam_name}. "
                f"Must be one of: {', '.join(CALIB_DIMS.keys())}"
            )
        
        camera_params_map[cam_name] = (cam_type, cam_calib_path, cam_params_path)
        camera_names.append(cam_name)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate config
    print(f"[INFO] Generating camera configuration file\n")
    print(f"  Elevation convention: {args.elevation_convention}")
    print(f"  Number of cameras: {len(camera_names)}")
    print(f"  Cameras:")
    
    for cam_name in camera_names:
        cam_type, cam_calib_path, cam_params_path = camera_params_map[cam_name]
        print(f"    - {cam_name:20s} ({cam_type:10s})")
        print(f"        calib:  {cam_calib_path}")
        print(f"        params: {cam_params_path}")
    print()
    
    try:
        config = generate_camera_config_file(
            output_dir / f"{args.output_name}.json",
            camera_names=camera_names,
            camera_params_map=camera_params_map,
            elevation_convention=args.elevation_convention
        )
        
        # Save config
        output_file = output_dir / f"{args.output_name}.json"
        with open(output_file, "w") as f:
            json.dump(config, f, indent=2)
        
        print(f"[INFO] Configuration details:")
        print(f"      ✓ Loaded {len(camera_names)} cameras")
        
        # Count by type
        type_counts = {}
        for cam_name in camera_names:
            cam_type, _, _ = camera_params_map[cam_name]
            type_counts[cam_type] = type_counts.get(cam_type, 0) + 1
        
        type_str = ", ".join(f"{count} {ctype}" for ctype, count in type_counts.items())
        print(f"      ✓ Camera types: {type_str}")
        print(f"      ✓ Elevation convention: {config['elevation_convention']}")
        
        floor_width = config["pen_floor"]["width_m"]
        floor_height = config["pen_floor"]["height_m"]
        print(f"      ✓ Floor dimensions: {floor_width:.4f}m × {floor_height:.4f}m")
        print(f"      ✓ Saved → {output_file}\n")
        
        print(f"[OK] Camera configuration generated successfully!")
        print(f"\nOutput file: {output_file}")
        
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return 1
    except ValueError as e:
        print(f"[ERROR] {e}")
        return 1
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
"""
python generate_pnp_configs.py \
    --output-dir camera_configs/zenith/ \
    --output-name pen1_camera_config_pnp_zenith \
    --elevation-convention zenith \
    --camera-params pen1_tur_cam1:fisheye:gt_camera_parameters/pen1_tur_cam1_calibration.npz:results/zenith/pen1_tur_cam1/camera_params.json \
                    pen1_tur_cam2:fisheye:gt_camera_parameters/pen1_tur_cam2_calibration.npz:results/zenith/pen1_tur_cam2/camera_params.json \
                    pen1_orb_cam1:pinhole:gt_camera_parameters/p1c1_orb.ini:results/zenith/pen1_orb_cam1/camera_params.json \
                    pen1_orb_cam2:pinhole:gt_camera_parameters/p1c2_orb.ini:results/zenith/pen1_orb_cam2/camera_params.json

# Generate with output name
python generate_pnp_configs.py \
    --output-dir camera_configs/zenith/ \
    --output-name pen2_camera_config_pnp_zenith \
    --elevation-convention zenith \
    --camera-params pen2_tur_cam1:fisheye:gt_camera_parameters/pen2_tur_cam1_calibration.npz:results/zenith/pen2_tur_cam1/camera_params.json \
                    pen2_tur_cam2:fisheye:gt_camera_parameters/pen2_tur_cam2_calibration.npz:results/zenith/pen2_tur_cam2/camera_params.json \
                    pen2_orb_cam1:pinhole:gt_camera_parameters/p2c1_orb.ini:results/zenith/pen2_orb_cam1/camera_params.json \
                    pen2_orb_cam2:pinhole:gt_camera_parameters/p2c2_orb.ini:results/zenith/pen2_orb_cam2/camera_params.json
"""