# multi_top.py — Top-view skeleton w/ fail-safe cleaning (no ROI clip, ROI TYPE unchanged)
import os, sys, json, traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
import pandas as pd
from plantcv import plantcv as pcv
import matplotlib.pyplot as plt


# ============ CONFIG ============
INPUT_PATH   = r"C:\Cantonese\topview_test.jpg"   # ไฟล์เดียว หรือโฟลเดอร์
OUTPUT_DIR   = r".\results_topview2"
EXTENSIONS   = ['.png', '.jpg', '.jpeg']
THREADS      = 1
DEBUG_MODE   = 'plot'     # 'none' | 'print' | 'plot'
SAVE_MASK    = True

# ROI สี่เหลี่ยมสำหรับ "วัดสัดส่วน / คัดคอนทัวร์" เท่านั้น (ไม่ clip รูปร่าง)
USE_FULL_IMAGE_ROI = True
ROI_X, ROI_Y, ROI_W, ROI_H = 100, 100, 500, 500
ROI_TYPE = 'partial'      # คงไว้เพื่อบันทึก meta; **ไม่ใช้ clip**

# Cleaning (fail-safe)
CLEAN_MASK = True               # ปิดได้ถ้าไม่อยาก clean
KEEP_LARGEST_ONLY = False       # True = เก็บก้อนใหญ่สุดอย่างเดียว
CLOSE_KSIZE = 5                 # 3–9 แล้วแต่ภาพ
BASE_MIN_OBJ_SIZE = 80          # พื้นที่ขั้นต่ำ (px) สำหรับลบเศษเล็ก
AREA_DROP_FAILSAFE = 0.30       # ถ้าพื้นที่หลัง clean < 30% ของก่อน clean → ย้อนกลับ
WHITE_INVERT_THRESHOLD = 0.50   # ถ้า white_ratio ใน ROI > 50% ให้ invert เพื่อให้ "พืช=ขาว"

# Prune sizes (สัมพันธ์กับ ROI + ค่าคงที่)
PRUNE_SIZE_FACTORS = (0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.0)
PRUNE_SIZE_ABS = (50, 75, 100, 125, 150, 175, 200, 250, 300, 350, 400, 500, 600, 800, 1000, 1200, 1500)

# ======= Auto-threshold config =======
THRESH_STRATEGY = 'hybrid'   # 'hybrid' | 'exg' | 'lab' | 'hsv' | 'kmeans' | 'manual'
EXPECTED_OBJ_RATIO = (0.02, 0.60)  # สัดส่วนพื้นที่วัตถุที่คาดหวังภายใน ROI (2–60%)
USE_CLAHE = True             # ช่วยแก้แสงไม่สม่ำเสมอ
CLAHE_CLIP = 2.0
CLAHE_TILE = 8
MANUAL_CHANNEL = 'lab_a'     # manual ใช้ 'lab_a' | 'lab_b' | 'lab_l' | 'hsv_s' | 'hsv_v' | 'exg'
MANUAL_T = 128               # manual threshold (0–255)
MANUAL_OBJECT_TYPE = 'dark'  # 'dark' ถ้าวัตถุมืดกว่าพื้นหลัง, 'light' ถ้าสว่างกว่า
# =====================================

def _rescale_0_255(img):
    img = img.astype('float32')
    mn, mx = float(img.min()), float(img.max())
    if mx <= mn:
        return np.zeros_like(img, dtype=np.uint8)
    out = (img - mn) * (255.0 / (mx - mn))
    return out.astype(np.uint8)

def _to_binary(img):
    return (img > 0).astype('uint8') * 255

def _count(img):
    return int(cv2.countNonZero(img)) if img is not None else 0

def _roi_mask_from_rect(shape, rect):
    (rx, ry, rw, rh) = rect
    roi = np.zeros(shape, dtype=np.uint8)
    cv2.rectangle(roi, (rx, ry), (rx + rw - 1, ry + rh - 1), 255, thickness=cv2.FILLED)
    return roi

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
    if not USE_CLAHE:
        return gray
    clahe = cv2.createCLAHE(clipLimit=float(CLAHE_CLIP), tileGridSize=(CLAHE_TILE, CLAHE_TILE))
    return clahe.apply(_rescale_0_255(gray))

def _thresh(gray, method, object_type):
    # คืนหน้ากากที่ 'วัตถุ = ขาว'
    if method == 'otsu':
        return pcv.threshold.otsu(gray_img=gray, object_type=object_type)
    if method == 'triangle':
        return pcv.threshold.triangle(gray_img=gray, object_type=object_type)
    if method == 'manual':
        t = int(MANUAL_T)
        if object_type == 'dark':
            bw = (gray <= t).astype('uint8') * 255
        else:
            bw = (gray >= t).astype('uint8') * 255
        return bw
    raise ValueError(f"Unknown method {method}")

def _score_mask(mask, roi_rect, area_total):
    # เกณฑ์: สัดส่วนใน ROI ใกล้ EXPECTED, ก้อนไม่แตกเป็นเศษ (คอมโพเนนต์น้อย), ก้อนใหญ่มี solidity สูง
    (rx, ry, rw, rh) = roi_rect
    roi_mask = _roi_mask_from_rect(mask.shape, roi_rect)
    in_roi = cv2.bitwise_and(mask, roi_mask)
    area_roi = max(1, rw * rh)
    ratio = _count(in_roi) / area_roi

    # คอมโพเนนต์+solidity
    _fc = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = _fc[1] if len(_fc) == 3 else _fc[0]
    n_comp = len(cnts)
    solidity = 0.0
    if cnts:
        idx = int(np.argmax([cv2.contourArea(c) for c in cnts]))
        cnt = cnts[idx]
        a = float(cv2.contourArea(cnt))
        if a > 1.0:
            hull = cv2.convexHull(cnt)
            ah = float(cv2.contourArea(hull))
            if ah > 0:
                solidity = a / ah

    # ให้คะแนนโดยระยะห่างจากช่วง EXPECTED_OBJ_RATIO
    lo, hi = EXPECTED_OBJ_RATIO
    if ratio < lo:
        s_ratio = - (lo - ratio) * 2.0
    elif ratio > hi:
        s_ratio = - (ratio - hi) * 2.0
    else:
        # อยู่ในช่วง → โบนัส
        s_ratio = 1.0 + (1.0 - min(ratio - lo, hi - ratio) / max(hi - lo, 1e-6))

    s_solidity = 0.7 * solidity   # ก้อนแน่นได้คะแนน
    s_comp = -0.15 * max(0, n_comp - 1)  # เศษเยอะโดนหัก
    score = s_ratio + s_solidity + s_comp
    meta = dict(ratio_roi=float(ratio), n_components=int(n_comp), solidity=float(solidity))
    return score, meta


# ---------- small utils ----------
def safe_readimage(path: Path):
    """รองรับ pcv.readimage ที่อาจคืน 1/2/3 ค่า"""
    ri = pcv.readimage(filename=str(path))
    img, filename = None, path.name
    if isinstance(ri, tuple):
        if len(ri) == 3:
            img, _, filename = ri
        elif len(ri) == 2:
            img, p = ri
            filename = Path(p).name if p else path.name
        else:
            img = ri[0]
    else:
        img = ri
    return img, filename

def _to_binary(img):
    """บังคับให้เป็น 0/255 (ไม่ถือเป็นการ clean)"""
    if img is None:
        return img
    if img.dtype != np.uint8:
        img = img.astype('uint8')
    return (img > 0).astype('uint8') * 255

def _count(img):
    return int(cv2.countNonZero(img)) if img is not None else 0

def intersects(a_rect, b_rect):
    """เช็ก bbox ตัดกัน: rect = (x, y, w, h)"""
    ax, ay, aw, ah = a_rect
    bx, by, bw, bh = b_rect
    return not (ax + aw <= bx or bx + bw <= ax or ay + ah <= by or by + bh <= ay)

def _connected_components_filter(bw, min_area):
    """ลบชิ้นเล็กด้วย connected components (ไม่พึ่งเวอร์ชัน PlantCV)"""
    bw = _to_binary(bw)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    out = np.zeros_like(bw)
    for i in range(1, num):  # 0 = background
        if stats[i, cv2.CC_STAT_AREA] >= max(1, int(min_area)):
            out[labels == i] = 255
    return out

def _largest_component(bw):
    """เก็บก้อนที่ใหญ่สุด (ใช้เมื่ออยากลดเสียงรบกวนหนัก ๆ)"""
    bw = _to_binary(bw)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if num <= 2:  # ไม่มีหรือมีก้อนเดียว
        return bw
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))  # avoid background idx 0
    out = np.zeros_like(bw)
    out[labels == idx] = 255
    return out

def _clean_mask_fail_safe(bw, close_ksize=5, base_min_obj_size=80,
                          roi_area_px=None, keep_largest=False,
                          area_drop_failsafe=0.30):
    """
    Clean แบบปลอดภัย:
      - ปิดรูเล็ก (closing) ด้วย kernel เล็ก
      - ลบชิ้นเล็กๆ ตามเกณฑ์ที่ 'ปรับตามขนาด ROI'
      - (ออปชัน) เก็บก้อนใหญ่สุดเท่านั้น
      - ถ้าพื้นที่หลัง clean ลดฮวบเกินเกณฑ์ → ย้อนกลับมาสก์ก่อน clean
    """
    if bw is None:
        return bw
    bw = _to_binary(bw)

    before = _count(bw)
    if before == 0:
        return bw

    out = bw.copy()

    # 1) ปิดรูเล็ก ๆ
    if close_ksize and close_ksize >= 3:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(close_ksize), int(close_ksize)))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k, iterations=1)

    # 2) เกณฑ์ลบชิ้นเล็กแบบปรับตาม ROI
    if roi_area_px is not None and roi_area_px > 0:
        min_obj_size = max(20, int(roi_area_px * 0.0015))  # 0.15% ของพื้นที่ ROI
        min_obj_size = max(min_obj_size, base_min_obj_size // 2)
    else:
        min_obj_size = base_min_obj_size

    out = _connected_components_filter(out, min_obj_size)

    # 3) (ออปชัน) เอาเฉพาะก้อนใหญ่สุด
    if keep_largest:
        out = _largest_component(out)

    # 4) Fail-safe: ถ้าพื้นที่หายเกินเกณฑ์ → ย้อนกลับ
    after = _count(out)
    if after == 0 or after < int(before * float(area_drop_failsafe)):
        return bw  # ย้อนกลับ

    return out


# ---------- auto mask ----------
def auto_threshold(rgb_img, roi_rect):
    """
    คืน (mask, info) โดย mask = binary ที่ 'วัตถุ = ขาว' แน่นอน
    กลยุทธ์:
      - hybrid: ทดลองหลาย channel/method แล้วใช้ ROI ให้คะแนน เลือกตัวดีที่สุด
      - exg/lab/hsv: ล็อกชุดช่องและวิธี
      - kmeans: แยก 2 กลุ่มจากพิกเซลใน ROI แล้ว generalize ไปทั้งภาพ
      - manual: ใช้ MANUAL_* ตาม config
    """
    H, W = rgb_img.shape[:2]
    area_total = H * W
    (rx, ry, rw, rh) = roi_rect

    def _candidate_masks():
        cands = []
        if THRESH_STRATEGY in ('hybrid', 'lab'):
            for ch in ('lab_a', 'lab_b', 'lab_l'):
                g = _apply_clahe(_channel_gray(rgb_img, ch))
                for meth, obj in [('otsu', 'dark'), ('otsu', 'light'), ('triangle', 'dark'), ('triangle', 'light')]:
                    try:
                        m = _thresh(g, meth, obj)
                        cands.append((f'{ch}:{meth}:{obj}', m))
                    except Exception:
                        pass
        if THRESH_STRATEGY in ('hybrid', 'hsv'):
            for ch in ('hsv_s', 'hsv_v'):
                g = _apply_clahe(_channel_gray(rgb_img, ch))
                for meth, obj in [('otsu', 'dark'), ('otsu', 'light'), ('triangle', 'dark'), ('triangle', 'light')]:
                    try:
                        m = _thresh(g, meth, obj)
                        cands.append((f'{ch}:{meth}:{obj}', m))
                    except Exception:
                        pass
        if THRESH_STRATEGY in ('hybrid', 'exg'):
            g = _apply_clahe(_channel_gray(rgb_img, 'exg'))
            for meth, obj in [('otsu', 'light'), ('triangle', 'light')]:
                try:
                    m = _thresh(g, meth, obj)
                    cands.append((f'exg:{meth}:{obj}', m))
                except Exception:
                    pass
        if THRESH_STRATEGY == 'manual':
            ch = MANUAL_CHANNEL
            obj = MANUAL_OBJECT_TYPE
            g = _channel_gray(rgb_img, 'exg' if ch == 'exg' else ch)
            g = _apply_clahe(g)
            m = _thresh(g, 'manual', obj)
            cands = [(f'{ch}:manual:{obj}', m)]
        return cands

    # k-means (2 คลัสเตอร์) จากพิกเซลใน ROI แล้ว label ทั้งภาพ
    def _kmeans_mask():
        # feature: [Lab a, Lab b, HSV s] (ปกติเขียวโดดใน a/b และ s สูง)
        lab = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)
        hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
        A = lab[:, :, 1].astype('float32'); B = lab[:, :, 2].astype('float32'); S = hsv[:, :, 1].astype('float32')
        roi = _roi_mask_from_rect(A.shape, roi_rect) > 0
        X = np.stack([A[roi], B[roi], S[roi]], axis=1)
        if X.shape[0] < 500:  # ข้อมูลน้อย → ขยาย sampling
            yy, xx = np.where(roi)
            idx = np.random.choice(len(xx), size=min(2000, len(xx)), replace=len(xx) < 2000)
            X = np.stack([A[yy[idx], xx[idx]], B[yy[idx], xx[idx]], S[yy[idx], xx[idx]]], axis=1).astype('float32')

        Z = X.copy()
        # KMeans OpenCV
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        ret, labels, centers = cv2.kmeans(Z, K=2, bestLabels=None, criteria=criteria, attempts=5,
                                          flags=cv2.KMEANS_PP_CENTERS)
        c0, c1 = centers
        # heuristic: พืชมักมี a,b ต่ำกว่า (เขียวกว่า) และ S สูงกว่า
        score0 = -c0[0] - c0[1] + 0.5 * c0[2]
        score1 = -c1[0] - c1[1] + 0.5 * c1[2]
        plant_center = c0 if score0 > score1 else c1

        # label ทั้งภาพด้วย nearest center
        flat = np.stack([A.ravel(), B.ravel(), S.ravel()], axis=1)
        d0 = np.sum((flat - centers[0]) ** 2, axis=1)
        d1 = np.sum((flat - centers[1]) ** 2, axis=1)
        plant_label = (d0 > d1).astype(np.uint8) if np.allclose(plant_center, centers[1]) else (d0 <= d1).astype(np.uint8)
        m = (plant_label.reshape(A.shape) * 255).astype(np.uint8)
        return m

    if THRESH_STRATEGY == 'kmeans':
        m = _kmeans_mask()
        info = {"strategy": "kmeans"}
        return _to_binary(m), info

    # hybrid / lab / hsv / exg / manual
    cands = _candidate_masks()
    # กันกรณีไม่มีผู้สมัคร
    if not cands:
        # fallback: Lab a, otsu,dark
        g = _apply_clahe(_channel_gray(rgb_img, 'lab_a'))
        m = _thresh(g, 'otsu', 'dark')
        return _to_binary(m), {"channel": "lab_a", "method": "otsu", "object_type": "dark", "strategy": "fallback"}

    # ให้คะแนนทุกผู้สมัคร
    best = None
    best_meta = {}
    best_score = -1e9
    for tag, m in cands:
        s, meta = _score_mask(_to_binary(m), roi_rect, area_total)
        if s > best_score:
            best_score = s
            best = (tag, m)
            best_meta = meta

    tag, mask = best
    ch, meth, obj = tag.split(':')
    info = {"channel": ch, "method": meth, "object_type": obj, "strategy": THRESH_STRATEGY}
    info.update(best_meta)
    return _to_binary(mask), info

# ---------- core ----------
def segment(rgb_img, filename):
    H, W = rgb_img.shape[:2]

   
    # ===== 1) Auto threshold แบบใหม่ =====
    if USE_FULL_IMAGE_ROI:
        rx, ry, rw, rh = 0, 0, W, H
    else:
        rx = max(0, min(ROI_X, W - 1))
        ry = max(0, min(ROI_Y, H - 1))
        rw = max(1, min(ROI_W, W - rx))
        rh = max(1, min(ROI_H, H - ry))

    mask0, info = auto_threshold(rgb_img, roi_rect=(rx, ry, rw, rh))
    # (ไม่ต้อง invert ซ้ำ เพราะ auto_threshold รับประกัน 'วัตถุ=ขาว' แล้ว)


    # 2) ให้ “พืช=ขาว” เสมอ (เช็กสัดส่วนเฉพาะพื้นที่ ROI; ไม่ clip รูปร่าง)
    if USE_FULL_IMAGE_ROI:
        rx, ry, rw, rh = 0, 0, W, H
    else:
        rx = max(0, min(ROI_X, W - 1))
        ry = max(0, min(ROI_Y, H - 1))
        rw = max(1, min(ROI_W, W - rx))
        rh = max(1, min(ROI_H, H - ry))

    roi_mask_measure = np.zeros_like(mask0, dtype=np.uint8)
    cv2.rectangle(roi_mask_measure, (rx, ry), (rx + rw - 1, ry + rh - 1), 255, thickness=cv2.FILLED)
    roi_only = cv2.bitwise_and(mask0, roi_mask_measure)
    white_ratio_roi = _count(roi_only) / max(1, rw * rh)
    if white_ratio_roi > WHITE_INVERT_THRESHOLD:
        mask0 = pcv.invert(gray_img=mask0)  # ให้พืชเป็นขาว

    # 3) Clean แบบ fail-safe (หรือข้าม หาก CLEAN_MASK=False)
    mask = mask0.copy()
    if CLEAN_MASK:
        roi_area_px = rw * rh
        mask = _clean_mask_fail_safe(
            mask,
            close_ksize=CLOSE_KSIZE,
            base_min_obj_size=BASE_MIN_OBJ_SIZE,
            roi_area_px=roi_area_px,
            keep_largest=KEEP_LARGEST_ONLY,
            area_drop_failsafe=AREA_DROP_FAILSAFE
        )
    mask = _to_binary(mask)

    if _count(mask) == 0:
        pcv.outputs.add_observation(sample='default', variable='status',
                                    trait='text', method='pipeline', scale='none',
                                    datatype=str, value='empty_mask', label='status')
        return {
            "filename": filename,
            "roi_x": int(rx), "roi_y": int(ry), "roi_w": int(rw), "roi_h": int(rh),
            "roi_type": ROI_TYPE, "num_leaf_segments": 0, "num_stem_segments": 0, "num_branch_points": 0
        }, mask

    # 4) หา contours และคัดเฉพาะที่ “bbox ตัดกับ ROI” (ไม่ clip รูปร่าง)
    _fc = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = _fc[1] if len(_fc) == 3 else _fc[0]
    roi_rect = (rx, ry, rw, rh)
    kept = []
    for cnt in contours:
        bx, by, bw, bh = cv2.boundingRect(cnt)
        if intersects((bx, by, bw, bh), roi_rect):
            kept.append(cnt)

    if not kept:
        pcv.outputs.add_observation(sample='default', variable='status',
                                    trait='text', method='pipeline', scale='none',
                                    datatype=str, value='empty_roi', label='status')
        return {
            "filename": filename, "roi_x": int(rx), "roi_y": int(ry), "roi_w": int(rw), "roi_h": int(rh),
            "roi_type": ROI_TYPE, "num_leaf_segments": 0, "num_stem_segments": 0, "num_branch_points": 0
        }, np.zeros_like(mask, dtype=np.uint8)

    # วาดคอนทัวร์ที่เลือกลงหน้ากากใหม่ (เต็มก้อน ไม่ตัดขอบ = ไม่ clip)
    kept_mask = np.zeros_like(mask, dtype=np.uint8)
    cv2.drawContours(kept_mask, kept, -1, 255, thickness=cv2.FILLED)

    # 5) สร้างสเกเลตัน + prune แบบไดนามิก
    skeleton = pcv.morphology.skeletonize(mask=kept_mask)
    eff_r = max(10, int(min(rw, rh) * 0.25))
    sizes = sorted(set([max(3, int(eff_r * t)) for t in PRUNE_SIZE_FACTORS] + list(PRUNE_SIZE_ABS)))

    pruned_skel = None
    edge_objects = None
    last_err = None
    leaf_obj = stem_obj = None
    segmented_img = None

    for sz in sizes:
        try:
            ret = pcv.morphology.prune(skel_img=skeleton, size=sz, mask=kept_mask)
            if isinstance(ret, tuple):
                if len(ret) == 3:
                    pruned_skel, _seg_img, edge_objects = ret
                elif len(ret) == 2:
                    pruned_skel, edge_objects = ret
                else:
                    pruned_skel = ret[0]
                    edge_objects = ret[1] if len(ret) > 1 else None
            else:
                pruned_skel = ret

            lo = pcv.morphology.segment_sort(skel_img=pruned_skel, objects=edge_objects, mask=kept_mask)
            if isinstance(lo, tuple):
                leaf_obj = lo[0]
                stem_obj = lo[1] if len(lo) > 1 else None
            else:
                leaf_obj, stem_obj = lo, None

            sid = pcv.morphology.segment_id(skel_img=pruned_skel, objects=leaf_obj, mask=kept_mask)
            segmented_img = sid[0] if isinstance(sid, tuple) else sid
            break  # ผ่านแล้ว
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            if "too many tips" in msg or "try pruning again" in msg:
                continue  # ลอง prune ใหญ่ขึ้น
            # ปัญหาอื่น ๆ — ลองค่าถัดไปเช่นกัน
            continue

    if pruned_skel is None:
        pcv.outputs.add_observation(sample='default', variable='status',
                                    trait='text', method='pipeline', scale='none',
                                    datatype=str, value=f'prune_failed:{last_err}', label='status')
        return {
            "filename": filename, "roi_x": int(rx), "roi_y": int(ry), "roi_w": int(rw), "roi_h": int(rh),
            "roi_type": ROI_TYPE, "num_leaf_segments": 0, "num_stem_segments": 0, "num_branch_points": 0
        }, kept_mask

    # 6) เมตริกต่อยอด
    _ = pcv.morphology.fill_segments(mask=kept_mask, objects=leaf_obj, label="default")
    branch_pts_mask = pcv.morphology.find_branch_pts(skel_img=pruned_skel, mask=kept_mask, label="default")
    _ = pcv.morphology.segment_euclidean_length(segmented_img=segmented_img, objects=leaf_obj, label="default")
    _ = pcv.analyze.size(img=rgb_img, labeled_mask=kept_mask, label="default")

    num_leaf_segments = int(len(leaf_obj)) if leaf_obj is not None else 0
    num_stem_segments = int(len(stem_obj)) if stem_obj is not None else 0
    num_branch_points = int(np.count_nonzero(branch_pts_mask)) if branch_pts_mask is not None else 0

    # log meta
    pcv.outputs.add_observation(sample='default', variable='status',
                                trait='text', method='pipeline', scale='none',
                                datatype=str, value='ok', label='status')
    pcv.outputs.add_observation(sample='default', variable='auto_channel',
                                trait='text', method='auto_select', scale='none',
                                datatype=str, value=info.get('channel', ''), label='channel')
    pcv.outputs.add_observation(sample='default', variable='auto_method',
                                trait='text', method='auto_select', scale='none',
                                datatype=str, value=info.get('method', ''), label='method')
    pcv.outputs.add_observation(sample='default', variable='auto_object_type',
                                trait='text', method='auto_select', scale='none',
                                datatype=str, value=info.get('object_type', ''), label='object_type')

    return {
        "filename": filename,
        "roi_x": int(rx), "roi_y": int(ry), "roi_w": int(rw), "roi_h": int(rh),
        "roi_type": ROI_TYPE,
        "num_leaf_segments": num_leaf_segments,
        "num_stem_segments": num_stem_segments,
        "num_branch_points": num_branch_points
    }, kept_mask


# ---------- runner ----------
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
