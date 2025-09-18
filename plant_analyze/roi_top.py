# roi_top.py (เวอร์ชันแก้แล้ว)
import cv2
import numpy as np

def make_grid_rois(rgb_img, rows, cols, roi_radius=None, margin=0.05, shape="circle"):
    """
    สร้างกริด ROI (rows x cols) แล้วคืนเป็น list ของคอนทัวร์ (ใช้กับ cv2.drawContours)
    - วางศูนย์กลางกลางเซลล์ (offset 0.5)
    - clamp รัศมีตามระยะจาก "ศูนย์กลางสุดขอบ" ถึงขอบภาพ
    """
    H, W = rgb_img.shape[:2]
    if rows <= 0 or cols <= 0:
        raise ValueError("rows/cols must be > 0")

    # กรอบใช้งานหลังหัก margin
    mx, my = int(W * margin), int(H * margin)
    x0, y0 = mx, my
    x1, y1 = W - mx - 1, H - my - 1

    # ขนาดเซลล์ (กว้าง/สูง) = แบ่งพื้นที่ใช้งานด้วยจำนวนคอลัมน์/แถว
    DX = (x1 - x0) / float(cols)   # << ต่างจากเดิม: หารด้วย cols (ไม่ใช่ cols-1)
    DY = (y1 - y0) / float(rows)  # << ต่างจากเดิม: หารด้วย rows

    # ศูนย์กลาง "สุดขอบ" เมื่อวางกลางเซลล์
    cx_min = x0 + 0.5 * DX
    cx_max = x0 + (cols - 0.5) * DX
    cy_min = y0 + 0.5 * DY
    cy_max = y0 + (rows - 0.5) * DY

    # รัศมีตั้งต้น
    if roi_radius is None:
        eff_r = int(max(2, round(0.45 * min(DX, DY))))
    else:
        eff_r = int(max(2, round(float(roi_radius))))

    # clamp รัศมีตาม "ระยะห่างของศูนย์กลางสุดขอบ" ถึงขอบภาพ
    eff_r = int(min(
        eff_r,
        cx_min,                 # ระยะจากศูนย์กลางซ้ายสุดถึงขอบซ้าย
        (W - 1) - cx_max,       # ระยะจากศูนย์กลางขวาสุดถึงขอบขวา
        cy_min,                 # ระยะจากศูนย์กลางบนสุดถึงขอบบน
        (H - 1) - cy_max        # ระยะจากศูนย์กลางล่างสุดถึงขอบล่าง
    ))
    if eff_r <= 0:
        raise ValueError("ROI grid does not fit the image. Adjust rows/cols/margin/radius.")

    # สร้างคอนทัวร์ ROI จากศูนย์กลาง "กลางเซลล์"
    rois = []
    for r in range(rows):
        cy = y0 + (r + 0.5) * DY
        for c in range(cols):
            cx = x0 + (c + 0.5) * DX
            if shape == "rect":
                half = eff_r
                cnt = np.array(
                    [[cx - half, cy - half],
                     [cx + half, cy - half],
                     [cx + half, cy + half],
                     [cx - half, cy + half]], dtype=np.int32
                ).reshape(-1, 1, 2)
            else:
                pts = cv2.ellipse2Poly((int(round(cx)), int(round(cy))),
                                       (eff_r, eff_r), 0, 0, 360, max(1, 360 // 96))
                cnt = pts.reshape(-1, 1, 2).astype(np.int32)
            rois.append(cnt)

    return rois, eff_r

def make_top_rois_auto(
    rgb_img,
    mask_fill,
    cfg=None,
    min_area_px=None,
    merge_gap_px=None,
    close_iters=None,
    debug_out_path=None,
):
    """
    จับคอมโพเนนต์จาก mask (top view) แล้ว 'รวมกล่องที่ชิดกัน' แบบเดียวกับ side
    คืน rois = [{"idx": i, "bbox": (x,y,w,h), "comp_mask": ...}, ...]
    """
    # ----- resolve params (เหมือนฝั่ง side) -----
    if cfg is not None:
        if min_area_px is None:
            min_area_px = int(getattr(cfg, "TOP_MIN_PLANT_AREA", getattr(cfg, "MIN_PLANT_AREA", 800)))
        if merge_gap_px is None:
            merge_gap_px = int(getattr(cfg, "TOP_MERGE_GAP", 5))
        if close_iters is None:
            close_iters = int(getattr(cfg, "TOP_CLOSE_ITERS", 1))
    else:
        min_area_px  = int(800 if min_area_px  is None else min_area_px)
        merge_gap_px = int(20  if merge_gap_px is None else merge_gap_px)
        close_iters  = int(1   if close_iters  is None else close_iters)

    H, W = rgb_img.shape[:2]
    m = (mask_fill > 0).astype(np.uint8) * 255

    # morphology ปรับแบบเดียวกับ side
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=close_iters)
    # กรองเศษเล็กออก (area-based)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros_like(m)
    boxes = []
    for cid in range(1, num):
        x, y, w, h, area = stats[cid]
        if area >= min_area_px:
            keep[labels == cid] = 255
            boxes.append([int(x), int(y), int(w), int(h)])

    if not boxes:
        return []

    # รวมกล่องที่ชิดกันแนวนอนตาม gap เหมือน side
    boxes = _merge_nearby_boxes_top(boxes, gap=int(merge_gap_px))
    boxes.sort(key=lambda b: b[0])

    rois = []
    dbg = rgb_img.copy()
    for i, (x, y, w, h) in enumerate(boxes, start=1):
        comp_mask = np.zeros((H, W), dtype=np.uint8)
        comp_mask[y:y+h, x:x+w] = keep[y:y+h, x:x+w]
        rois.append({"idx": i, "bbox": (x, y, w, h), "comp_mask": comp_mask})
        # วาด debug
        cv2.rectangle(dbg, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(dbg, f"#{i}", (x, max(0, y-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)

    if debug_out_path:
        cv2.imwrite(debug_out_path, dbg)
    return rois


def _merge_nearby_boxes_top(boxes, gap=8):
    """
    รวมกล่องที่ชิดกันในแนวนอน (logic เดียวกับ side’s _merge_nearby_boxes แต่ตัดเรื่อง v-overlap ออก)
    """
    if gap is None or gap <= 0:
        return sorted(boxes, key=lambda b: b[0])
    boxes = sorted(boxes, key=lambda b: b[0])
    merged = [boxes[0]]
    for b in boxes[1:]:
        x, y, w, h = b
        x0, y0, w0, h0 = merged[-1]
        ex = [x0 - gap, y0, w0 + 2*gap, h0]  # ขยายซ้าย-ขวา
        intersect = not (ex[0]+ex[2] <= x or x+w <= ex[0] or ex[1]+ex[3] <= y or y+h <= ex[1])
        if intersect:
            nx = min(x0, x); ny = min(y0, y)
            nx2 = max(x0+w0, x+w); ny2 = max(y0+h0, y+h)
            merged[-1] = [nx, ny, nx2 - nx, ny2 - ny]
        else:
            merged.append(b)
    return merged