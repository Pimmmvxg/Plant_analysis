import numpy as np
import cv2
from plantcv import plantcv as pcv

def make_side_roi(rgb_img, mask_fill, USE_FULL_IMAGE_ROI, ROI_X, ROI_Y, ROI_W, ROI_H):
    H, W = rgb_img.shape[:2]
    if USE_FULL_IMAGE_ROI:
        x, y, w, h = 0, 0, W, H
    else:
        x = max(0, min(ROI_X, W - 1))
        y = max(0, min(ROI_Y, H - 1))
        w = max(1, min(ROI_W, W - x))
        h = max(1, min(ROI_H, H - y))


    _ = pcv.roi.rectangle(img=rgb_img, x=x, y=y, w=w, h=h) # debug only


    _fc = cv2.findContours(mask_fill.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = _fc[2] if len(_fc) == 3 else _fc[0]


    def intersects(b, r):
        bx, by, bw, bh = b
        rx, ry, rw, rh = r
        return not (bx + bw <= rx or rx + rw <= bx or by + bh <= ry or ry + rh <= by)


    roi_rect = (x, y, w, h)
    kept = []
    for cnt in contours:
        b = cv2.boundingRect(cnt)
        if intersects(b, roi_rect):
            kept.append(cnt)


    out = np.zeros_like(mask_fill, dtype=np.uint8)
    if kept:
        cv2.drawContours(out, kept, -1, 255, thickness=cv2.FILLED)
    return out, (x, y, w, h)