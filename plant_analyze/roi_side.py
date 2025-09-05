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

def make_side_rois_auto(rgb_img, mask_fill, min_area_px=800, merge_gap_px=10, debug_out_path=None):
    H, W = rgb_img.shape[:2]
    m = (mask_fill > 0).astype(np.uint8) * 255

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)
    m = pcv.fill(bin_img=m, size=int(min_area_px))
    if cv2.countNonZero(m) == 0:
        return []

    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    boxes = []
    for cid in range(1, num):
        x, y, w, h, area = stats[cid]
        if area >= min_area_px:
            boxes.append([int(x), int(y), int(w), int(h)])

    if not boxes:
        return []

    boxes = _merge_nearby_boxes(boxes, gap=int(merge_gap_px))
    boxes.sort(key=lambda b: b[0])

    rois = []
    dbg = rgb_img.copy()
    for i, (x, y, w, h) in enumerate(boxes, start=1):
        ret = pcv.roi.rectangle(img=rgb_img, x=x, y=y, w=w, h=h)
        roi_obj = ret[0] if isinstance(ret, tuple) else ret
        comp_mask = np.zeros((H, W), dtype=np.uint8)
        comp_mask[y:y+h, x:x+w] = m[y:y+h, x:x+w]
        rois.append({"idx": i, "bbox": (x, y, w, h), "roi_obj": roi_obj, "comp_mask": comp_mask})
        cv2.rectangle(dbg, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(dbg, f"#{i}", (x, max(0, y-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)

    if debug_out_path:
        cv2.imwrite(debug_out_path, dbg)
    return rois

def _merge_nearby_boxes(boxes, gap=8):
    if not boxes:
        return boxes
    boxes = sorted(boxes, key=lambda b: b[0])
    merged = [boxes[0]]
    def v_overlap(b1, b2):
        y1a, y1b = b1[1], b1[1]+b1[3]
        y2a, y2b = b2[1], b2[1]+b2[3]
        inter = max(0, min(y1b, y2b) - max(y1a, y2a))
        return inter / float(min(b1[3], b2[3]) + 1e-6)
    for b in boxes[1:]:
        x,y,w,h = b
        x0,y0,w0,h0 = merged[-1]
        horizontal_gap = x - (x0 + w0)
        if horizontal_gap <= gap and v_overlap(merged[-1], b) >= 0.3:
            nx = min(x0, x); ny = min(y0, y)
            nx2 = max(x0+w0, x+w); ny2 = max(y0+h0, y+h)
            merged[-1] = [nx, ny, nx2-nx, ny2-ny]
        else:
            merged.append(b)
    return merged