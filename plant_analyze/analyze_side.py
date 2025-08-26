from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional
from pathlib import Path

import cv2
import numpy as np
from plantcv import plantcv as pcv

from . import config as cfg
from .masking import ensure_binary


# ---------------------------- Data class (optional) ---------------------------- #
@dataclass
class SideBBox:
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    @property
    def width(self) -> int:
        return max(0, self.x_max - self.x_min + 1)

    @property
    def height(self) -> int:
        return max(0, self.y_max - self.y_min + 1)


# ---------------------------- Helpers ---------------------------- #
def _bbox_from_mask(m: np.ndarray) -> SideBBox:
    ys, xs = np.where(m > 0)
    if ys.size == 0 or xs.size == 0:
        return SideBBox(0, 0, 0, 0)
    return SideBBox(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def _skeletonize(mask: np.ndarray) -> np.ndarray:
    return pcv.morphology.skeletonize(mask=mask)


def _prune_robust(skel: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Iteratively prune; tolerate different PlantCV return signatures."""
    sizes = [10, 25, 50, 75, 100, 125, 150, 175, 200]
    pruned = skel.copy()
    for sz in sizes:
        try:
            ret = pcv.morphology.prune(skel_img=pruned, size=int(sz), mask=mask)
            if isinstance(ret, tuple):
                # (skel, seg_img, edge_objects) or (skel, edge_objects)
                pruned = ret[0]
            else:
                pruned = ret
        except Exception:
            pass
    return pruned


def _find_endpoints(skel: np.ndarray) -> np.ndarray:
    """Return Nx2 array of (y,x) coords for pixels with degree==1."""
    k = np.array([[1,1,1],[1,10,1],[1,1,1]], dtype=np.uint8)
    nb = cv2.filter2D((skel > 0).astype(np.uint8), -1, k)
    ys, xs = np.where(nb == 11)
    if ys.size == 0:
        return np.zeros((0, 2), dtype=int)
    return np.stack([ys, xs], axis=1)


def _px_to_mm(px: float) -> Optional[float]:
    """Convert pixel length to millimeters using cfg.MM_PER_PX (mm per pixel).
    If not available, return None (caller can skip writing mm values).
    """
    mm_per_px = getattr(cfg, "MM_PER_PX", None)
    if mm_per_px is None:
        return None
    try:
        mm_per_px = float(mm_per_px)
    except Exception:
        return None
    return float(px) * mm_per_px


# ---------------------------- Core API ---------------------------- #
# ---------------------------- Core API + Debug helpers ---------------------------- #

def _save_debug(img: np.ndarray, name: str) -> None:
    try:
        base = Path(pcv.params.debug_outdir or ".")
        base.mkdir(parents=True, exist_ok=True)
        pcv.print_image(img=img, filename=str(base / name))
    except Exception:
        pass

def _draw_size_overlay(mask: np.ndarray, rgb: np.ndarray, bbox: SideBBox,
                       height_px: float, length_px: float,
                       height_mm: Optional[float], length_mm: Optional[float]) -> np.ndarray:
    vis = rgb.copy()
    # mask outline
    cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cnts, -1, (0, 255, 255), 2)

    # bbox + dimensions text
    cv2.rectangle(vis, (bbox.x_min, bbox.y_min), (bbox.x_max, bbox.y_max), (0, 255, 0), 2)
    x_mid = int(0.5*(bbox.x_min + bbox.x_max))
    cv2.line(vis, (x_mid, bbox.y_min), (x_mid, bbox.y_max), (0, 255, 0), 2)

    # Annotations
    txt1 = f"H: {height_px:.1f}px" + (f" ({height_mm:.1f} mm)" if height_mm is not None else "")
    txt2 = f"L: {length_px:.1f}px" + (f" ({length_mm:.1f} mm)" if length_mm is not None else "")
    cv2.putText(vis, txt1, (bbox.x_min, max(0, bbox.y_min-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50,220,50), 2, cv2.LINE_AA)
    cv2.putText(vis, txt2, (bbox.x_min, min(vis.shape[0]-5, bbox.y_max+20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50,220,50), 2, cv2.LINE_AA)
    return vis
def _draw_shape_overlay(rgb: np.ndarray,
                        shape_height_px: float, shape_length_px: float,
                        shape_height_mm: Optional[float], shape_length_mm: Optional[float]) -> np.ndarray:
    """Overlay showing image-shape-based size (H,W) regardless of mask.
    Draws a border around the full image and annotates H/W in px (+mm if available).
    """
    vis = rgb.copy()
    H, W = vis.shape[:2]
    # full-frame border
    cv2.rectangle(vis, (1,1), (W-2, H-2), (255, 0, 200), 2)
    # dimension arrows
    # vertical
    cv2.arrowedLine(vis, (W-10, 5), (W-10, H-5), (255, 0, 200), 2, tipLength=0.02)
    # horizontal
    cv2.arrowedLine(vis, (5, H-10), (W-5, H-10), (255, 0, 200), 2, tipLength=0.02)
    # labels
    txtH = f"H_shape: {shape_height_px:.1f}px" + (f" ({shape_height_mm:.1f} mm)" if shape_height_mm is not None else "")
    txtW = f"W_shape: {shape_length_px:.1f}px" + (f" ({shape_length_mm:.1f} mm)" if shape_length_mm is not None else "")
    cv2.putText(vis, txtH, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,200), 2, cv2.LINE_AA)
    cv2.putText(vis, txtW, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,200), 2, cv2.LINE_AA)
    return vis

def analyze_one_side(slot_mask: np.ndarray, sample_name: str, rgb_img: np.ndarray) -> None:

    # Normalize mask to binary 0/255 with object=white
    slot_mask = ensure_binary(slot_mask, normalize_orientation=True)
    if slot_mask is None or cv2.countNonZero(slot_mask) == 0:
        raise ValueError("analyze_one_side: empty mask")

    H, W = slot_mask.shape[:2]

    # Save basic mask debug
    mask_vis = cv2.cvtColor(slot_mask, cv2.COLOR_GRAY2BGR)
    _save_debug(mask_vis, f"{sample_name}_side_mask.png")

    # Skeletonize & prune
    skel = _skeletonize(slot_mask)
    pruned = _prune_robust(skel, slot_mask)

    # Save skeleton debug (raw + pruned)
    skel_vis = rgb_img.copy()
    yy, xx = np.where(skel > 0)
    skel_vis[yy, xx] = (0, 255, 255)
    _save_debug(skel_vis, f"{sample_name}_side_skeleton.png")

    pruned_vis = rgb_img.copy()
    yy, xx = np.where(pruned > 0)
    pruned_vis[yy, xx] = (255, 255, 0)
    _save_debug(pruned_vis, f"{sample_name}_side_skeleton_pruned.png")

    # Endpoints
    endpoints = _find_endpoints(pruned)
    n_endpoints = int(endpoints.shape[0])

    endpoints_vis = pruned_vis.copy()
    for (ey, ex) in endpoints:
        cv2.circle(endpoints_vis, (int(ex), int(ey)), 3, (0, 0, 255), -1)
    _save_debug(endpoints_vis, f"{sample_name}_side_endpoints.png")

    # Size from mask bbox
    bbox = _bbox_from_mask(slot_mask)
    height_px = float(bbox.height)
    length_px = float(bbox.width)

    # mm conversion via cfg.MM_PER_PX (mm/px)
    height_mm = _px_to_mm(height_px)
    length_mm = _px_to_mm(length_px)

    # Area in px
    area_px = int(cv2.countNonZero(slot_mask))

    # Extra: Shape-based size (image dimensions)
    shape_H, shape_W = slot_mask.shape[:2]
    shape_height_px = float(shape_H)
    shape_length_px = float(shape_W)
    shape_height_mm = _px_to_mm(shape_height_px)
    shape_length_mm = _px_to_mm(shape_length_px)
    
    # OVERLAY: size/shape summary (like analyze.size style)
    size_overlay = _draw_size_overlay(slot_mask, rgb_img, bbox, height_px, length_px, height_mm, length_mm)
    _save_debug(size_overlay, f"{sample_name}_side_size_overlay.png")
    # (A) OVERLAY: custom size/shape summary (like analyze.size style)
    shape_overlay = _draw_shape_overlay(rgb_img, shape_height_px, shape_length_px, shape_height_mm, shape_length_mm)
    _save_debug(shape_overlay, f"{sample_name}_side_shape_overlay.png")

    # (B) PlantCV built-in: pcv.analyze.size (will also emit its own debug image)
    try:
        # รองรับความต่างของ signature ในบางเวอร์ชัน
        try:
            labeled_mask, n_labels = pcv.create_labels(bin_img=slot_mask)
        except TypeError:
            labeled_mask, n_labels = pcv.create_labels(slot_mask)

        pcv.analyze.size(
            img=rgb_img,                 # หรือใช้ cropped_img ถ้าคุณมี ROI แล้วครอปไว้
            labeled_mask=labeled_mask,
            label=sample_name
        )
    except Exception:
        # อย่าล้ม pipeline แม้ analyze.size จะ fail
        pass

    # ---------------- Record observations ---------------- #
    def _add(var, value, trait, method, scale, dt, label=None):
        pcv.outputs.add_observation(sample=sample_name, variable=var, trait=trait,
                                    method=method, scale=scale, datatype=dt,
                                    value=value, label=label or var)

    _add("height_px", height_px, "height", "bbox_y_span", "px", float)
    _add("length_px", length_px, "length", "bbox_x_span", "px", float)
    _add("area_px", area_px, "projected_area", "mask_nonzero", "px^2", int)
    _add("n_endpoints", n_endpoints, "skeleton_endpoints", "degree==1", "count", int)
    _add("bbox", f"({bbox.x_min},{bbox.y_min},{bbox.x_max},{bbox.y_max})", "bbox_xyxy", "mask_bbox", "px", str)

    if height_mm is not None:
        _add("height_mm", float(height_mm), "height", "px_to_mm", "mm", float)
    if length_mm is not None:
        _add("length_mm", float(length_mm), "length", "px_to_mm", "mm", float)
    
    # Record shape size too
    _add("shape_height_px", shape_height_px, "shape_height", "image_shape", "px", float)
    _add("shape_length_px", shape_length_px, "shape_length", "image_shape", "px", float)
    if shape_height_mm is not None:
        _add("shape_height_mm", shape_height_mm, "shape_height", "image_shape", "mm", float)
    if shape_length_mm is not None:
        _add("shape_length_mm", shape_length_mm, "shape_length", "image_shape", "mm", float)

    # Extra: save a consolidated debug with tiles
    try:
        gap = 8
        h2 = max(img.shape[0] for img in [rgb_img, size_overlay, endpoints_vis])
        w1 = rgb_img.shape[1]
        w2 = size_overlay.shape[1]
        w3 = endpoints_vis.shape[1]
        canvas = np.zeros((h2, w1 + w2 + w3 + gap*2, 3), dtype=np.uint8)
        canvas[:rgb_img.shape[0], :w1] = rgb_img
        canvas[:size_overlay.shape[0], w1+gap:w1+gap+w2] = size_overlay
        canvas[:endpoints_vis.shape[0], w1+gap+w2+gap:w1+gap+w2+gap+w3] = endpoints_vis
        _save_debug(canvas, f"{sample_name}_side_summary.png")
    except Exception:
        pass

    # No return; results are written into PlantCV outputs as in the existing pipeline
