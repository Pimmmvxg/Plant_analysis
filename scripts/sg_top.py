# sg_top.py — Top-view skeleton + แยกตาม partial ROI (ไม่ clip), มีสวิตช์ prune/no-prune และ auto-threshold แบบฉลาด
# ใช้ PlantCV v4+, OpenCV 4
import os, sys, json, traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
import pandas as pd
from plantcv import plantcv as pcv
import matplotlib.pyplot as plt


# ========================== CONFIG ==========================
# I/O
INPUT_PATH   = r"C:\Cantonese\topview_test.jpg"   # ใส่ไฟล์เดี่ยว หรือโฟลเดอร์
OUTPUT_DIR   = r".\results_topsingle"
EXTENSIONS   = ['.png', '.jpg', '.jpeg']
THREADS      = 1
DEBUG_MODE   = 'plot'     # 'none' | 'print' | 'plot'
SAVE_MASK    = True

# ROI กริด (ใช้วัดการ "ทับซ้อน" แบบ partial เท่านั้น — ไม่ clip รูปร่างพืช)
USE_ROI_GRID   = True
GRID_X, GRID_Y = 400, 200   # จุดเริ่มช่องซ้ายบน (พิกเซล)
DX, DY         = 500, 600   # ระยะศูนย์กลางถึงศูนย์กลาง
ROWS, COLS     = 2, 3
ROI_RADIUS     = 200         # รัศมีที่อยากได้; โค้ดจะคำนวณ eff_r ให้ปลอดภัยอีกที
ASSIGN_MODE    = 'multi'    # 'multi' = ต้นแตะหลายช่องนับทุกช่อง, 'best' = เลือกช่องที่ทับมากสุด
MIN_PLANT_AREA = 200        # px ขั้นต่ำให้ถือว่าเป็น "หนึ่งต้น"

# Auto-threshold
THRESH_STRATEGY = 'hybrid'   # 'hybrid' | 'exg' | 'lab' | 'hsv' | 'kmeans' | 'manual'
EXPECTED_OBJ_RATIO = (0.02, 0.60)  # สัดส่วนพื้นที่ "พืช" ที่คาดหวังภายในกริด (ภาพรวม)
USE_CLAHE  = True
CLAHE_CLIP = 2.0
CLAHE_TILE = 8
MANUAL_CHANNEL = 'lab_a'     # 'lab_a'|'lab_b'|'lab_l'|'hsv_s'|'hsv_v'|'exg'
MANUAL_T   = 128
MANUAL_OBJECT_TYPE = 'dark'  # 'dark' (พืชมืดกว่า) หรือ 'light'

# Cleaning (fail-safe) — ตั้ง CLEAN_MASK=False หากไม่อยาก clean
CLEAN_MASK          = True
KEEP_LARGEST_ONLY   = False
CLOSE_KSIZE         = 5
BASE_MIN_OBJ_SIZE   = 80
AREA_DROP_FAILSAFE  = 0.30   # ถ้าหลัง clean พื้นที่ลดเกิน 70% → ย้อนกลับมาสก์เดิม

# Prune/No-prune
DO_PRUNE            = True           # ปกติเปิด prune
FALLBACK_NO_PRUNE   = True           # ถ้า prune ไม่ผ่าน ให้ตกไปเมตริก no-prune

# Prune sizes (สัมพันธ์กับ ROI + ค่า absolute)
PRUNE_SIZE_FACTORS  = (0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.0)
PRUNE_SIZE_ABS      = (50, 75, 100, 125, 150, 175, 200, 250, 300, 350, 400, 500, 600, 800, 1000, 1200)
# ===========================================================


# -------------------- small utils --------------------
def safe_readimage(path: Path):
    """รองรับ pcv.readimage ที่อาจคืน 1/2/3 ค่า"""
    ri = pcv.readimage(filename=str(path))
    img, filename = None, path.name
    if isinstance(ri, tuple):
        if len(ri) == 3: img, _, filename = ri
        elif len(ri) == 2: img, p = ri; filename = Path(p).name if p else path.name
        else: img = ri[0]
    else:
        img = ri
    return img, filename

def _to_binary(img):
    if img is None: return img
    if img.dtype != np.uint8:
        img = img.astype('uint8')
    return (img > 0).astype('uint8') * 255

def _count(img):
    return int(cv2.countNonZero(img)) if img is not None else 0

def _rescale_0_255(img):
    img = img.astype('float32')
    mn, mx = float(img.min()), float(img.max())
    if mx <= mn:
        return np.zeros_like(img, dtype=np.uint8)
    return ((img - mn) * (255.0 / (mx - mn))).astype(np.uint8)

def _roi_mask_from_rect(shape, rect):
    (rx, ry, rw, rh) = rect
    roi = np.zeros(shape, dtype=np.uint8)
    cv2.rectangle(roi, (rx, ry), (rx + rw - 1, ry + rh - 1), 255, thickness=cv2.FILLED)
    return roi

def intersects(a_rect, b_rect):
    ax, ay, aw, ah = a_rect
    bx, by, bw, bh = b_rect
    return not (ax + aw <= bx or bx + bw <= ax or ay + ah <= by or by + bh <= ay)


# ------------------ cleaning (fail-safe) ------------------
def _connected_components_filter(bw, min_area):
    bw = _to_binary(bw)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    out = np.zeros_like(bw)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= max(1, int(min_area)):
            out[labels == i] = 255
    return out

def _largest_component(bw):
    bw = _to_binary(bw)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if num <= 2:
        return bw
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    out = np.zeros_like(bw)
    out[labels == idx] = 255
    return out

def _clean_mask_fail_safe(bw, close_ksize=5, base_min_obj_size=80,
                          roi_area_px=None, keep_largest=False,
                          area_drop_failsafe=0.30):
    if bw is None: return bw
    bw = _to_binary(bw)
    before = _count(bw)
    if before == 0: return bw

    out = bw.copy()
    if close_ksize and close_ksize >= 3:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(close_ksize), int(close_ksize)))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k, iterations=1)

    if roi_area_px is not None and roi_area_px > 0:
        min_obj_size = max(20, int(roi_area_px * 0.0015))
        min_obj_size = max(min_obj_size, base_min_obj_size // 2)
    else:
        min_obj_size = base_min_obj_size
    out = _connected_components_filter(out, min_obj_size)

    if keep_largest:
        out = _largest_component(out)

    after = _count(out)
    if after == 0 or after < int(before * float(area_drop_failsafe)):
        return bw  # ย้อนกลับ
    return out


# ------------------ auto threshold ------------------
def _channel_gray(rgb_img, name):
    if name.startswith('lab_'):
        ch = name.split('_')[1]
        return pcv.rgb2gray_lab(rgb_img=rgb_img, channel=ch)
    if name.startswith('hsv_'):
        ch = name.split('_')[1]
        return pcv.rgb2gray_hsv(rgb_img=rgb_img, channel=ch)
    if name == 'exg':
        R = rgb_img[:, :, 0].astype(np.int16)
        G = rgb_img[:, :, 1].astype(np.int16)
        B = rgb_img[:, :, 2].astype(np.int16)
        exg = 2 * G - R - B
        return _rescale_0_255(exg)
    raise ValueError(f"Unknown channel {name}")

def _apply_clahe(gray):
    if not USE_CLAHE: return _rescale_0_255(gray)
    clahe = cv2.createCLAHE(clipLimit=float(CLAHE_CLIP), tileGridSize=(CLAHE_TILE, CLAHE_TILE))
    return clahe.apply(_rescale_0_255(gray))

def _thresh(gray, method, object_type):
    if method == 'otsu':
        return pcv.threshold.otsu(gray_img=gray, object_type=object_type)
    if method == 'triangle':
        return pcv.threshold.triangle(gray_img=gray, object_type=object_type)
    if method == 'manual':
        t = int(MANUAL_T)
        if object_type == 'dark':
            return ((gray <= t).astype('uint8')) * 255
        else:
            return ((gray >= t).astype('uint8')) * 255
    raise ValueError(f"Unknown method {method}")

def _score_mask(mask, roi_rect, area_total):
    (rx, ry, rw, rh) = roi_rect
    roi_mask = _roi_mask_from_rect(mask.shape, roi_rect)
    in_roi = cv2.bitwise_and(mask, roi_mask)
    area_roi = max(1, rw * rh)
    ratio = _count(in_roi) / area_roi

    _fc = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = _fc[1] if len(_fc) == 3 else _fc[0]
    n_comp = len(cnts)
    solidity = 0.0
    if cnts:
        idx = int(np.argmax([cv2.contourArea(c) for c in cnts]))
        cnt = cnts[idx]
        a = float(cv2.contourArea(cnt))
        if a > 1.0:
            hull = cv2.convexHull(cnt); ah = float(cv2.contourArea(hull))
            if ah > 0: solidity = a / ah

    lo, hi = EXPECTED_OBJ_RATIO
    if ratio < lo: s_ratio = - (lo - ratio) * 2.0
    elif ratio > hi: s_ratio = - (ratio - hi) * 2.0
    else: s_ratio = 1.0 + (1.0 - min(ratio - lo, hi - ratio) / max(hi - lo, 1e-6))

    s_solidity = 0.7 * solidity
    s_comp = -0.15 * max(0, n_comp - 1)
    score = s_ratio + s_solidity + s_comp
    meta = dict(ratio_roi=float(ratio), n_components=int(n_comp), solidity=float(solidity))
    return score, meta

def auto_threshold(rgb_img, roi_rect):
    """คืน (mask, info) ที่ object=ขาว แน่นอน"""
    H, W = rgb_img.shape[:2]
    area_total = H * W

    def _candidate_masks():
        cands = []
        if THRESH_STRATEGY in ('hybrid', 'lab'):
            for ch in ('lab_a', 'lab_b', 'lab_l'):
                g = _apply_clahe(_channel_gray(rgb_img, ch))
                for meth, obj in [('otsu', 'dark'), ('otsu', 'light'), ('triangle', 'dark'), ('triangle', 'light')]:
                    try: cands.append((f'{ch}:{meth}:{obj}', _thresh(g, meth, obj)))
                    except Exception: pass
        if THRESH_STRATEGY in ('hybrid', 'hsv'):
            for ch in ('hsv_s', 'hsv_v'):
                g = _apply_clahe(_channel_gray(rgb_img, ch))
                for meth, obj in [('otsu', 'dark'), ('otsu', 'light'), ('triangle', 'dark'), ('triangle', 'light')]:
                    try: cands.append((f'{ch}:{meth}:{obj}', _thresh(g, meth, obj)))
                    except Exception: pass
        if THRESH_STRATEGY in ('hybrid', 'exg'):
            g = _apply_clahe(_channel_gray(rgb_img, 'exg'))
            for meth, obj in [('otsu', 'light'), ('triangle', 'light')]:
                try: cands.append((f'exg:{meth}:{obj}', _thresh(g, meth, obj)))
                except Exception: pass
        if THRESH_STRATEGY == 'manual':
            ch = MANUAL_CHANNEL; obj = MANUAL_OBJECT_TYPE
            g = _apply_clahe(_channel_gray(rgb_img, 'exg' if ch=='exg' else ch))
            cands = [(f'{ch}:manual:{obj}', _thresh(g, 'manual', obj))]
        return cands

    if THRESH_STRATEGY == 'kmeans':
        lab = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)
        hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
        A = lab[:, :, 1].astype('float32'); B = lab[:, :, 2].astype('float32'); S = hsv[:, :, 1].astype('float32')
        (rx, ry, rw, rh) = roi_rect
        roi = _roi_mask_from_rect(A.shape, roi_rect) > 0
        yy, xx = np.where(roi)
        if len(xx) < 500:
            idx = np.random.choice(len(xx), size=min(2000, len(xx)), replace=len(xx) < 2000)
        else:
            idx = np.random.choice(len(xx), size=2000, replace=False)
        X = np.stack([A[yy[idx], xx[idx]], B[yy[idx], xx[idx]], S[yy[idx], xx[idx]]], axis=1).astype('float32')
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        ret, labels, centers = cv2.kmeans(X, K=2, bestLabels=None, criteria=criteria, attempts=5,
                                          flags=cv2.KMEANS_PP_CENTERS)
        c0, c1 = centers
        score0 = -c0[0] - c0[1] + 0.5 * c0[2]
        score1 = -c1[0] - c1[1] + 0.5 * c1[2]
        plant_idx = 0 if score0 > score1 else 1
        flat = np.stack([A.ravel(), B.ravel(), S.ravel()], axis=1)
        d0 = np.sum((flat - centers[0]) ** 2, axis=1)
        d1 = np.sum((flat - centers[1]) ** 2, axis=1)
        lab_img = (d1 < d0).astype(np.uint8) if plant_idx == 1 else (d0 <= d1).astype(np.uint8)
        m = (lab_img.reshape(A.shape) * 255).astype(np.uint8)
        return _to_binary(m), {"strategy": "kmeans"}

    cands = _candidate_masks()
    if not cands:
        g = _apply_clahe(_channel_gray(rgb_img, 'lab_a'))
        m = _thresh(g, 'otsu', 'dark')
        return _to_binary(m), {"channel": "lab_a", "method": "otsu", "object_type": "dark", "strategy": "fallback"}

    best, best_meta, best_score = None, {}, -1e9
    for tag, m in cands:
        s, meta = _score_mask(_to_binary(m), roi_rect, area_total)
        if s > best_score:
            best_score, best, best_meta = s, (tag, m), meta
    tag, mask = best
    ch, meth, obj = tag.split(':')
    info = {"channel": ch, "method": meth, "object_type": obj, "strategy": THRESH_STRATEGY}
    info.update(best_meta)
    return _to_binary(mask), info


# ------------------ ROI grid (partial, no clip) ------------------
def _roi_grid_masks(rgb_img):
    """สร้างกริด ROI แล้วคืนหน้ากาก 'พื้นที่ช่อง' สำหรับวัดการทับซ้อน (ไม่ clip พืช)"""
    H, W = rgb_img.shape[:2]
    xN = GRID_X + (COLS - 1) * DX
    yN = GRID_Y + (ROWS - 1) * DY
    eff_r = min(
        ROI_RADIUS,
        max(DX // 2 - 1, 1),
        max(DY // 2 - 1, 1),
        GRID_X, GRID_Y,
        (W - 1) - xN, (H - 1) - yN
    )
    if eff_r <= 0:
        raise ValueError("ROI grid does not fit the image. ปรับ GRID_X/Y, DX/DY, ROWS/COLS, ROI_RADIUS")
    roi_obj = pcv.roi.multi(img=rgb_img, coord=(GRID_X, GRID_Y), radius=int(eff_r),
                            spacing=(DX, DY), nrows=ROWS, ncols=COLS)
    try: rois = list(roi_obj)
    except TypeError: rois = getattr(roi_obj, 'rois', None) or [roi_obj]

    fullwhite = np.full((H, W), 255, dtype=np.uint8)
    roi_masks = []
    for i, roi in enumerate(rois):
        roi_area = pcv.roi.filter(mask=fullwhite, roi=roi, roi_type='cutto')  # ใช้สร้างรูปร่างช่องเท่านั้น
        r = i // COLS + 1; c = i % COLS + 1
        roi_masks.append((roi_area, r, c))
    return roi_masks


# ------------------ per-plant analysis (with prune/no-prune) ------------------
def _analyze_plant_metrics(plant_mask, rw, rh, do_prune=DO_PRUNE, fallback_no_prune=FALLBACK_NO_PRUNE):
    """วิเคราะห์ 'ต้นเดียว' คืน dict เมตริก"""
    plant_mask = _to_binary(plant_mask)
    area_px = _count(plant_mask)
    skel = pcv.morphology.skeletonize(mask=plant_mask)

    def _easy_metrics(status_tag):
        branch_pts = pcv.morphology.find_branch_pts(skel_img=skel, mask=plant_mask, label="plant_tmp")
        skel_len_px = int(cv2.countNonZero(skel))
        return dict(
            status=status_tag,
            plant_area_px=area_px,
            skeleton_len_px=skel_len_px,
            num_branch_points=int(np.count_nonzero(branch_pts)) if branch_pts is not None else 0,
            num_leaf_segments=None, num_stem_segments=None,
        )

    if not do_prune:
        return _easy_metrics(status_tag="no_prune")

    eff_r = max(10, int(min(rw, rh) * 0.25))
    sizes = sorted(set([max(3, int(eff_r * t)) for t in PRUNE_SIZE_FACTORS] + list(PRUNE_SIZE_ABS)))

    pruned_skel = None; edge_objects = None; last_err = None
    for sz in sizes:
        try:
            ret = pcv.morphology.prune(skel_img=skel, size=sz, mask=plant_mask)
            if isinstance(ret, tuple):
                if len(ret) == 3: pruned_skel, _seg_img, edge_objects = ret
                elif len(ret) == 2: pruned_skel, edge_objects = ret
                else: pruned_skel = ret[0]; edge_objects = ret[1] if len(ret) > 1 else None
            else:
                pruned_skel = ret

            lo = pcv.morphology.segment_sort(skel_img=pruned_skel, objects=edge_objects, mask=plant_mask)
            leaf_obj = lo[0] if isinstance(lo, tuple) else lo
            stem_obj = lo[1] if isinstance(lo, tuple) and len(lo) > 1 else None
            sid = pcv.morphology.segment_id(skel_img=pruned_skel, objects=leaf_obj, mask=plant_mask)
            segmented_img = sid[0] if isinstance(sid, tuple) else sid

            _ = pcv.morphology.fill_segments(mask=plant_mask, objects=leaf_obj, label="plant_tmp")
            branch_pts_mask = pcv.morphology.find_branch_pts(skel_img=pruned_skel, mask=plant_mask, label="plant_tmp")
            _ = pcv.morphology.segment_euclidean_length(segmented_img=segmented_img, objects=leaf_obj, label="plant_tmp")
            _ = pcv.analyze.size(img=None, labeled_mask=plant_mask, label="plant_tmp")

            return dict(
                status="ok",
                plant_area_px=area_px,
                num_leaf_segments=int(len(leaf_obj)) if leaf_obj is not None else 0,
                num_stem_segments=int(len(stem_obj)) if stem_obj is not None else 0,
                num_branch_points=int(np.count_nonzero(branch_pts_mask)) if branch_pts_mask is not None else 0,
                skeleton_len_px=None,
            )
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            if "too many tips" in msg or "try pruning again" in msg:
                continue
            continue

    if fallback_no_prune:
        return _easy_metrics(status_tag="fallback_no_prune")
    return dict(status="prune_failed", plant_area_px=area_px,
                num_leaf_segments=0, num_stem_segments=0, num_branch_points=0, skeleton_len_px=None)


# ------------------ core pipeline ------------------
def segment(rgb_img, filename):
    H, W = rgb_img.shape[:2]

    # 1) กำหนดกรอบสำหรับประเมิน (ใช้ทั้งภาพ ถ้า USE_ROI_GRID=True แต่จะสร้างกริดภายหลัง)
    rx, ry, rw, rh = 0, 0, W, H

    # 2) สร้างมาสก์ object=ขาว ด้วย auto-threshold
    mask0, info = auto_threshold(rgb_img, roi_rect=(rx, ry, rw, rh))
    mask0 = _to_binary(mask0)

    # 3) Clean แบบ fail-safe (หรือข้าม หาก CLEAN_MASK=False)
    mask = mask0.copy()
    if CLEAN_MASK:
        roi_area_px = rw * rh
        mask = _clean_mask_fail_safe(mask, close_ksize=CLOSE_KSIZE,
                                     base_min_obj_size=BASE_MIN_OBJ_SIZE,
                                     roi_area_px=roi_area_px,
                                     keep_largest=KEEP_LARGEST_ONLY,
                                     area_drop_failsafe=AREA_DROP_FAILSAFE)
    mask = _to_binary(mask)
    if _count(mask) == 0:
        pcv.outputs.add_observation(sample='default', variable='status',
                                    trait='text', method='pipeline', scale='none',
                                    datatype=str, value='empty_mask', label='status')
        return {
            "filename": filename, "roi_x": int(rx), "roi_y": int(ry), "roi_w": int(rw), "roi_h": int(rh),
            "roi_type": 'partial', "plant_count": 0
        }, mask

    # 4) เตรียมกริด ROI พื้นที่ช่อง (ไว้คำนวณ partial overlap — ไม่ clip)
    if USE_ROI_GRID:
        roi_masks = _roi_grid_masks(rgb_img)   # [(roi_area_mask, r, c), ...]
        # ใช้ขนาดช่องเพื่อกะสเกล prune
        rw_slot, rh_slot = DX, DY
    else:
        roi_masks = [(np.full_like(mask, 255, dtype=np.uint8), 1, 1)]
        rw_slot, rh_slot = W, H

    # สร้าง union ของพื้นที่ ROI เพื่อ "คัดคอมโพเนนต์ที่ไม่ได้ทับช่อง" ออก (ไม่ clip รูปร่าง)
    union_roi = np.zeros_like(mask, dtype=np.uint8)
    for roi_area, r, c in roi_masks:
        union_roi = cv2.bitwise_or(union_roi, roi_area)

    # 5) แยก “ต้น” ด้วยคอมโพเนนต์ แล้วเก็บเฉพาะที่ทับ union ROI
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    plants = []
    for i in range(1, num):
        pm = (labels == i).astype('uint8') * 255
        if _count(pm) < int(MIN_PLANT_AREA):
            continue
        inter = cv2.bitwise_and(pm, union_roi)
        if _count(inter) == 0:
            continue  # อยู่นอกกริดทั้งหมด
        plants.append((i, pm, int(_count(pm))))

    if not plants:
        pcv.outputs.add_observation(sample='default', variable='status',
                                    trait='text', method='pipeline', scale='none',
                                    datatype=str, value='no_plants_in_grid', label='status')
        return {
            "filename": filename, "roi_x": int(rx), "roi_y": int(ry), "roi_w": int(rw), "roi_h": int(rh),
            "roi_type": 'partial', "plant_count": 0
        }, mask

    # 6) วิเคราะห์หนึ่งครั้งต่อหนึ่งต้น แล้ว "แจก" ผลให้ ROI ที่ทับแบบ partial
    metrics_by_pid = {pid: _analyze_plant_metrics(pm, rw_slot, rh_slot) for pid, pm, _ in plants}

    plant_index_in_slot = {(r, c): 0 for (_, r, c) in roi_masks}

    def _assign_rois_for_plant(pm):
        hits = []
        comp_area = int(cv2.countNonZero(pm))
        for roi_area, r, c in roi_masks:
            inter = cv2.bitwise_and(pm, roi_area)
            inter_area = int(cv2.countNonZero(inter))
            if inter_area > 0:
                ratio = inter_area / max(1, comp_area)
                hits.append(((r, c), inter_area, ratio))
        if not hits: return []
        if ASSIGN_MODE == 'best':
            return [max(hits, key=lambda x: x[1])]
        return hits  # 'multi'

    for pid, pm, area in plants:
        hits = _assign_rois_for_plant(pm)
        if not hits:  # เซฟตี้
            continue
        met = metrics_by_pid[pid]
        for (r, c), inter_area, ratio in hits:
            plant_index_in_slot[(r, c)] += 1
            k = plant_index_in_slot[(r, c)]
            sample_name = f"slot_{r}_{c}__plant_{k}"

            # เมตริกของ “ต้น”
            pcv.outputs.add_observation(sample=sample_name, variable='status',
                                        trait='text', method='pipeline', scale='none',
                                        datatype=str, value=met['status'], label='status')
            pcv.outputs.add_observation(sample=sample_name, variable='plant_area_px',
                                        trait='area', method='count_nonzero', scale='px',
                                        datatype=int, value=int(met['plant_area_px']), label='plant area')
            if 'skeleton_len_px' in met and met['skeleton_len_px'] is not None:
                pcv.outputs.add_observation(sample=sample_name, variable='skeleton_len_px',
                                            trait='length', method='skeleton_pixels', scale='px',
                                            datatype=int, value=int(met['skeleton_len_px']), label='skeleton length')
            if met.get('num_leaf_segments') is not None:
                pcv.outputs.add_observation(sample=sample_name, variable='num_leaf_segments',
                                            trait='count', method='segment_sort', scale='count',
                                            datatype=int, value=int(met['num_leaf_segments']), label='leaf segments')
            if met.get('num_stem_segments') is not None:
                pcv.outputs.add_observation(sample=sample_name, variable='num_stem_segments',
                                            trait='count', method='segment_sort', scale='count',
                                            datatype=int, value=int(met['num_stem_segments']), label='stem segments')
            pcv.outputs.add_observation(sample=sample_name, variable='num_branch_points',
                                        trait='count', method='find_branch_pts', scale='count',
                                        datatype=int, value=int(met['num_branch_points']), label='branch points')

            # เมตริกการซ้อนทับกับ ROI (partial)
            pcv.outputs.add_observation(sample=sample_name, variable='area_in_slot_px',
                                        trait='area', method='overlap', scale='px',
                                        datatype=int, value=int(inter_area), label='area inside slot')
            pcv.outputs.add_observation(sample=sample_name, variable='overlap_ratio',
                                        trait='ratio', method='overlap/plant_area', scale='ratio',
                                        datatype=float, value=float(ratio), label='overlap ratio (slot/plant)')

    # meta รวม
    pcv.outputs.add_observation(sample='default', variable='status',
                                trait='text', method='pipeline', scale='none',
                                datatype=str, value='ok', label='status')
    pcv.outputs.add_observation(sample='default', variable='auto_strategy',
                                trait='text', method='auto_select', scale='none',
                                datatype=str, value=info.get('strategy', ''), label='strategy')
    pcv.outputs.add_observation(sample='default', variable='auto_channel',
                                trait='text', method='auto_select', scale='none',
                                datatype=str, value=info.get('channel', ''), label='channel')
    pcv.outputs.add_observation(sample='default', variable='auto_method',
                                trait='text', method='auto_select', scale='none',
                                datatype=str, value=info.get('method', ''), label='method')
    pcv.outputs.add_observation(sample='default', variable='auto_object_type',
                                trait='text', method='auto_select', scale='none',
                                datatype=str, value=info.get('object_type', ''), label='object_type')
    pcv.outputs.add_observation(sample='default', variable='plant_count',
                                trait='count', method='connected_components', scale='count',
                                datatype=int, value=int(len(plants)), label='plant count')

    # คืนหน้ากากผลรวม (ไว้ดู/เซฟ)
    return {
        "filename": filename,
        "roi_x": int(rx), "roi_y": int(ry), "roi_w": int(rw), "roi_h": int(rh),
        "roi_type": 'partial',
    }, mask


# ------------------ runner ------------------
def process_one(path: Path, out_dir: Path, debug_mode: str, save_mask: bool):
    pcv.params.debug = debug_mode
    pcv.params.debug_outdir = str(out_dir / "debug")
    pcv.params.dpi = 150
    pcv.outputs.clear()

    img, filename = safe_readimage(path)
    extra, mask = segment(img, filename)

    # flatten outputs
    results_dict = pcv.outputs.observations.copy()
    flat = {}
    for sample, vars_ in results_dict.items():
        for var_name, record in vars_.items():
            col = f"{sample}__{var_name}" if sample != 'default' else var_name
            flat[col] = record.get('value', None)
    flat.update(extra)

    # write json
    json_dir = out_dir / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    out_json = json_dir / f"{Path(filename).stem}.json"
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(flat, f, ensure_ascii=False, indent=2)

    # write mask image
    if save_mask and mask is not None:
        (out_dir / "processed").mkdir(exist_ok=True, parents=True)
        pcv.print_image(img=mask, filename=str(out_dir / "processed" / f"{Path(filename).stem}_mask.png"))
        flat["mask_saved"] = True
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump(flat, f, ensure_ascii=False, indent=2)

    plt.close('all')
    return str(out_json)

def aggregate_json_to_csv(json_paths, out_csv: Path):
    rows = []
    for jp in json_paths:
        try:
            with open(jp, 'r', encoding='utf-8') as f:
                rows.append(json.load(f))
        except Exception as e:
            print(f"Error reading {jp}: {e}")
    if not rows:
        print("No valid JSON files found.")
        return
    df = pd.DataFrame(rows)
    cols = list(df.columns)
    if 'filename' in cols:
        cols.insert(0, cols.pop(cols.index('filename')))
        df = df[cols]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding='utf-8')
    print(f"Results written to {out_csv}")

def build_file_list(input_path: Path, extensions):
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() in extensions else []
    elif input_path.is_dir():
        return [p for p in input_path.glob('**/*') if p.suffix.lower() in extensions]
    else:
        return []

def main():
    in_path = Path(INPUT_PATH)
    out_dir = Path(OUTPUT_DIR)
    exts = set([e.lower() for e in EXTENSIONS])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "debug").mkdir(exist_ok=True)

    files = build_file_list(in_path, exts)
    if not files:
        print(f"No files found in {in_path} with extensions {exts}.")
        sys.exit(1)

    print(f"Found {len(files)} files to process.")

    json_paths = []
    if THREADS > 1:
        with ProcessPoolExecutor(max_workers=THREADS) as ex:
            futures = {ex.submit(process_one, f, out_dir, DEBUG_MODE, SAVE_MASK): f for f in files}
            for fut in as_completed(futures):
                f = futures[fut]
                try:
                    jp = fut.result()
                    json_paths.append(jp)
                except Exception as e:
                    print(f"Error processing {f}: {e}")
                    traceback.print_exc()
    else:
        for f in files:
            try:
                jp = process_one(f, out_dir, DEBUG_MODE, SAVE_MASK)
                json_paths.append(jp)
            except Exception as e:
                print(f"Error processing {f}: {e}")
                traceback.print_exc()

    out_csv = out_dir / "results.csv"
    aggregate_json_to_csv(json_paths, out_csv)
    print(f"Results written to {out_csv}")

if __name__ == '__main__':
    main()
