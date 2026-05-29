"""
annotate_corners.py
-------------------
Interactive tool: click the 4 pen-floor corners in an image and save them.

Corner ordering (important for pose estimation):
    p1 = near-left   (closest to camera, left side)
    p2 = near-right  (closest to camera, right side)
    p3 = far-right   (farthest from camera, right side)
    p4 = far-left    (farthest from camera, left side)

Controls:
    Left-click      : place the next corner (or drag to adjust)
    Right-click     : remove the last placed corner
    r               : reset all corners
    Enter / s       : save and exit
    q               : quit without saving

Usage:
    python annotate_corners.py --image frame.jpg --out corners.json
    python annotate_corners.py --image frame.jpg --out corners.json --scale 0.6
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

CORNER_LABELS = ["p1 near-left", "p2 near-right", "p3 far-right", "p4 far-left"]
# CORNER_LABELS = ["P1", "P2", "P3", "P4"] # Shorter labels for display
CORNER_COLORS = [
    (0,   255, 255),   # yellow  p1
    (0,   200, 255),   # orange  p2
    (0,   80,  255),   # red     p3
    (200, 80,  255),   # magenta p4
]
LINE_COLOR = (180, 180, 50)
DONE_COLOR = (80, 220, 80)


# ── State shared with mouse callback ──────────────────────────────────────────
_state: dict = {"corners": [], "drag_idx": None}


def _mouse_cb(event, x, y, flags, param):
    corners = _state["corners"]
    scale   = param["scale"]
    # Convert display coordinates back to original image coordinates
    ox = x / scale
    oy = y / scale

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(corners) < 4:
            corners.append([ox, oy])
        else:
            # Find closest corner to move it
            dists = [np.hypot(c[0] - ox, c[1] - oy) for c in corners]
            idx = int(np.argmin(dists))
            if dists[idx] < 20 / scale:
                _state["drag_idx"] = idx
    elif event == cv2.EVENT_MOUSEMOVE:
        if _state["drag_idx"] is not None:
            corners[_state["drag_idx"]] = [ox, oy]
    elif event == cv2.EVENT_LBUTTONUP:
        if _state["drag_idx"] is not None:
            corners[_state["drag_idx"]] = [ox, oy]
            _state["drag_idx"] = None
    elif event == cv2.EVENT_RBUTTONDOWN:
        if corners:
            corners.pop()


def _draw_guide(canvas: np.ndarray, scale: float) -> np.ndarray:
    """Draw a small ordering guide (mini-diagram) in the top-right corner."""
    h, w = canvas.shape[:2]
    gw, gh = 160, 100
    gx = w - gw - 8
    gy = 8
    # Background
    cv2.rectangle(canvas, (gx - 2, gy - 2), (gx + gw + 2, gy + gh + 2),
                  (30, 30, 30), -1)
    cv2.rectangle(canvas, (gx - 2, gy - 2), (gx + gw + 2, gy + gh + 2),
                  (180, 180, 180), 1)
    inset_pts = [
        (gx + 20,  gy + gh - 18),   # p1 near-left
        (gx + gw - 20, gy + gh - 18), # p2 near-right
        (gx + gw - 20, gy + 18),      # p3 far-right
        (gx + 20,  gy + 18),          # p4 far-left
    ]
    for i in range(4):
        cv2.line(canvas, inset_pts[i], inset_pts[(i + 1) % 4],
                 LINE_COLOR, 1, cv2.LINE_AA)
    for i, (px, py) in enumerate(inset_pts):
        cv2.circle(canvas, (px, py), 5, CORNER_COLORS[i], -1, cv2.LINE_AA)
        label = f"p{i+1}"
        cv2.putText(canvas, label, (px + 4, py - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, CORNER_COLORS[i], 2, cv2.LINE_AA)
    cv2.putText(canvas, "CAM", (gx + gw // 2 - 12, gy + gh + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
    return canvas


def _render(img_orig: np.ndarray, scale: float) -> np.ndarray:
    corners = _state["corners"]
    h, w = img_orig.shape[:2]
    canvas = cv2.resize(img_orig, (int(w * scale), int(h * scale)))

    def _s(pt):
        return (int(pt[0] * scale), int(pt[1] * scale))

    # Draw completed polygon sides
    if len(corners) >= 2:
        for i in range(len(corners)):
            if i == len(corners) - 1 and len(corners) < 4:
                break
            cv2.line(canvas, _s(corners[i]), _s(corners[(i + 1) % len(corners)]),
                     LINE_COLOR if len(corners) < 4 else DONE_COLOR, 2, cv2.LINE_AA)

    # Draw corner markers
    for i, c in enumerate(corners):
        col = CORNER_COLORS[i]
        cv2.circle(canvas, _s(c), 8, col, -1, cv2.LINE_AA)
        cv2.circle(canvas, _s(c), 10, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, CORNER_LABELS[i], (_s(c)[0] + 12, _s(c)[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 3, cv2.LINE_AA)

    # Prompt for next corner
    if len(corners) < 4:
        msg = f"Click  {CORNER_LABELS[len(corners)]}"
    else:
        msg = "All 4 corners set.  Press Enter/s to save,  r to reset"
    cv2.putText(canvas, msg, (10, canvas.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)

    _draw_guide(canvas, scale)
    return canvas


def annotate(image_path: Path, out_path: Path, scale: float = 1.0,
             existing: list | None = None) -> list | None:
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[ERROR] Cannot read image: {image_path}")
        return None

    if existing:
        _state["corners"] = [list(c) for c in existing]
    else:
        _state["corners"] = []

    win = "Annotate floor corners  (q=quit, r=reset, Enter=save)"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, _mouse_cb, {"scale": scale})

    while True:
        cv2.imshow(win, _render(img, scale))
        key = cv2.waitKey(20) & 0xFF
        if key in (13, ord("s")):          # Enter or s → save
            if len(_state["corners"]) == 4:
                break
            print("[WARN] 4 corners required before saving.")
        elif key == ord("r"):
            _state["corners"] = []
        elif key == ord("q"):
            cv2.destroyAllWindows()
            return None

    cv2.destroyAllWindows()
    corners = _state["corners"]
    data = {
        "image_path": str(image_path.resolve()),
        "corners": corners,
        "corner_order": ["near_left", "near_right", "far_right", "far_left"],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2))
    print(f"[INFO] Corners saved -> {out_path}")
    return corners


def load_corners(json_path: Path) -> dict:
    data = json.loads(Path(json_path).read_text())
    data["corners"] = [list(map(float, c)) for c in data["corners"]]
    return data


def main():
    ap = argparse.ArgumentParser(description="Annotate pen-floor corners.")
    ap.add_argument("--image",  required=True, help="Input image file")
    ap.add_argument("--out",    required=True, help="Output JSON path")
    ap.add_argument("--scale",  type=float, default=1.0,
                    help="Display scale factor (default 1.0)")
    ap.add_argument("--resume", action="store_true",
                    help="Load existing corners from --out and allow editing")
    args = ap.parse_args()

    image_path = Path(args.image)
    out_path   = Path(args.out)

    existing = None
    if args.resume and out_path.exists():
        existing = load_corners(out_path)["corners"]
        print(f"[INFO] Loaded existing corners from {out_path}")

    result = annotate(image_path, out_path, scale=args.scale, existing=existing)
    if result is None:
        print("[INFO] Annotation cancelled.")
        sys.exit(1)
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
