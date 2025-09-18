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

def make_side_rois_auto(
        rgb_img,
        mask_fill,
        cfg=None,               
        min_area_px=None,
        merge_gap_px=None,
        debug_out_path=None
    ):
    # ----- Resolve ค่าจาก cfg หรือใช้ค่า fallback -----
    if cfg is not None:
        if min_area_px is None:
            min_area_px = int(getattr(cfg, "MIN_PLANT_AREA", 800))
        if merge_gap_px is None:
            merge_gap_px = int(getattr(cfg, "SIDE_MERGE_GAP", 20))
        close_iters = int(getattr(cfg, "SIDE_CLOSE_ITERS", 1))
        v_overlap_min = float(getattr(cfg, "SIDE_V_OVERLAP_MIN", 0.3))
    else:
        min_area_px  = int(800 if min_area_px  is None else min_area_px)
        merge_gap_px = int(20  if merge_gap_px is None else merge_gap_px)
        close_iters  = 1
        v_overlap_min = 0.3

    H, W = rgb_img.shape[:2]
    m = (mask_fill > 0).astype(np.uint8) * 255

    # morphology ปรับจาก cfg ได้
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=close_iters)
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

    # ใช้ merge_gap_px
    boxes = _merge_nearby_boxes(boxes, gap=int(merge_gap_px), v_overlap_min=v_overlap_min)
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


def _merge_nearby_boxes(boxes, gap=100, v_overlap_min=0.0):
    """รวมกล่องที่ชิดกันในแนวแกน X และมี vertical overlap ถึงเกณฑ์"""
    boxes = sorted(boxes, key=lambda b: b[0])
    merged = [boxes[0]]

    for b in boxes[1:]:
        x,y,w,h = b
        x0,y0,w0,h0 = merged[-1]

        # ขยายกล่องล่าสุดออกซ้าย-ขวา gap พิกเซล
        ex = [x0-gap, y0, w0+2*gap, h0]

        # ถ้ามี overlap ใดๆ (ไม่สน vertical overlap)
        ex = [x0 - gap, y0, w0 + 2*gap, h0]  # expand horizontally by `gap`
        intersect = not (ex[0]+ex[2] <= x or x+w <= ex[0] or ex[1]+ex[3] <= y or y+h <= ex[1])
        if intersect:
            nx = min(x0, x); ny = min(y0, y)
            nx2 = max(x0+w0, x+w); ny2 = max(y0+h0, y+h)
            merged[-1] = [nx, ny, nx2-nx, ny2-ny]
        else:
            merged.append(b)
    return merged
