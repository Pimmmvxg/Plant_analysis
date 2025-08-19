from typing import List, Tuple
import numpy as np
from math import floor
from plantcv import plantcv as pcv

def make_grid_rois(rgb_img, ROWS, COLS, ROI_RADIUS=None) -> Tuple[List, int]:
    if rgb_img is None:
        raise ValueError("rgb_img is None")

    H, W = rgb_img.shape[:2]
    if ROWS <= 0 or COLS <= 0:
        raise ValueError("ROWS and COLS must be > 0")

    # ขนาดช่อง
    cell_w = W / float(COLS)
    cell_h = H / float(ROWS)
    max_safe_r = max(1, floor(min(cell_w, cell_h) / 2) - 1)

    # เลือกรัศมี
    if ROI_RADIUS is None or ROI_RADIUS <= 0:
        eff_r = max(1, floor(min(cell_w, cell_h) / 3))
    else:
        eff_r = min(int(ROI_RADIUS), int(max_safe_r))

    # mask ขาวทั้งภาพ
    mask = np.full((H, W), 255, dtype=np.uint8)

    # Auto grid
    roi_contours, roi_hierarchy = pcv.roi.auto_grid(
        mask=mask,
        nrows=int(ROWS),
        ncols=int(COLS),
        radius=int(eff_r),
        img=rgb_img
    )

    rois = list(roi_contours) if roi_contours is not None else []
    return rois, int(eff_r)
