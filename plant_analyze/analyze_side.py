import numpy as np
from plantcv import plantcv as pcv
from .color import get_color_name, _hue_to_degrees
import cv2
from . import config as cfg
import os
from pathlib import Path

# --- HEIGHT helpers ---
def _height_mask(slot_mask, stem_obj, sample_name):
    """คืน mask ที่ใช้วัดความสูง: stem-only ถ้าตั้งค่าและมีข้อมูล, ไม่งั้นใช้ mask ทั้งหมด"""
    use_stem = bool(getattr(cfg, "HEIGHT_USE_STEM_ONLY", False))
    if use_stem and stem_obj is not None and len(stem_obj) > 0:
        try:
            m = pcv.morphology.fill_segments(mask=slot_mask, objects=stem_obj,
                                             label=f"{sample_name}_stem")
            if m is not None:
                m = (m > 0).astype(np.uint8) * 255
                if cv2.countNonZero(m) > 0:
                    return m, "stem_only"
        except Exception:
            pass
    return slot_mask, "full_mask"

def _analyze_color_side(rgb_img, mask, bins=36):
    """คำนวณสีหลักและสถิติสีจาก RGB + mask (วัตถุ=255)"""
    if rgb_img is None or mask is None:
        return None
    m = (mask > 0)

    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2HSV)   # H:0–179, S:0–255, V:0–255
    lab = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2LAB)   # L:0–255, A:0–255, B:0–255

    H = hsv[..., 0][m].astype(np.float32) * 2.0        # 0–179 -> 0–358 deg
    S = hsv[..., 1][m].astype(np.float32) / 255.0
    V = hsv[..., 2][m].astype(np.float32) / 255.0

    Lc = lab[..., 0][m].astype(np.float32) / 255.0
    Ac = lab[..., 1][m].astype(np.float32) - 128.0
    Bc = lab[..., 2][m].astype(np.float32) - 128.0

    if H.size == 0:
        return None

    # dominant hue from histogram
    hist, edges = np.histogram(H, bins=bins, range=(0, 360))
    peak = int(np.argmax(hist))
    hue_dom = float((edges[peak] + edges[peak + 1]) / 2.0)
    hue_dom = _hue_to_degrees(hue_dom)  # normalize/warp เผื่อ input เพี้ยน
    main_color = get_color_name(hue_dom)

    # circular mean hue
    rad = np.deg2rad(H)
    mean_angle = np.arctan2(np.mean(np.sin(rad)), np.mean(np.cos(rad)))
    mean_hue = float((np.rad2deg(mean_angle) + 360.0) % 360.0)

    # ค่าเฉลี่ย
    s_mean = float(np.mean(S))
    v_mean = float(np.mean(V))
    l_mean = float(np.mean(Lc))
    a_mean = float(np.mean(Ac))
    b_mean = float(np.mean(Bc))

    greenness_index = float(-a_mean)  # A บวก=แดง, ลบ=เขียว → -A ยิ่งสูงยิ่งเขียว

    return {
        "side_main_color": main_color,
        "side_main_hue_deg": round(hue_dom, 1),
        "side_mean_hue_deg": round(mean_hue, 1),
        "side_hsv_mean": {
            "hue_deg": round(mean_hue, 1),
            "sat": round(s_mean, 4),
            "val": round(v_mean, 4),
        },
        "side_lab_mean": {
            "L": round(l_mean, 4),
            "a": round(a_mean, 4),
            "b": round(b_mean, 4),
        },
        "side_greenness_index": round(greenness_index, 4),
    }
def _px_to_mm(px_len: float):
    if px_len is None:
        return None
    # ถ้ามีการกำหนด MM_PER_PX ให้ใช้
    if getattr(cfg, "MM_PER_PX", None):
        return float(px_len) * float(cfg.MM_PER_PX)
    # หรือถ้ามี DPI ให้คำนวณ
    if getattr(cfg, "DPI", None):
        px_per_mm = float(cfg.DPI) / 25.4
        return float(px_len) / px_per_mm
    # ถ้าไม่มีอะไรเลย ใช้ FAKE_MM_PER_PX ไว้ลอง
    k = float(getattr(cfg, "FAKE_MM_PER_PX", 0.02))
    return float(px_len) * k
    
def analyze_one_side(slot_mask, sample_name, rgb_img):
    # 1) skeletonize
    base_skel = pcv.morphology.skeletonize(mask=slot_mask)

    # 2) prune sizes
    sizes = [10, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 300, 350, 400, 450, 500]
    last_err = None
    pruned_skel, edge_objects = None, None
    
    for sz in sizes:
        try:
            ret = pcv.morphology.prune(skel_img=base_skel if pruned_skel is None else pruned_skel,
                                       size=sz, mask=slot_mask)
            if isinstance(ret, tuple) and len(ret) == 3:
                pruned_skel, seg_img, edge_objects = ret
            elif isinstance(ret, tuple) and len(ret) == 2:
                pruned_skel, edge_objects = ret
            else:
                pruned_skel = ret
                edge_objects = None
                
            lo = pcv.morphology.segment_sort(skel_img=pruned_skel, objects=edge_objects, mask=slot_mask)
            leaf_obj = lo[0] if isinstance(lo, tuple) else lo
            stem_obj = lo[1] if isinstance(lo, tuple) and len(lo) > 1 else None

            sid = pcv.morphology.segment_id(skel_img=pruned_skel, objects=leaf_obj, mask=slot_mask)
            segmented_img = sid[0] if isinstance(sid, tuple) else sid
            break
        except Exception as e:
            last_err = e
            continue
    else:
        #pruned_skel, segmented_img, leaf_obj, stem_obj = None, None, None, None #pruned false analyze color only
        raise last_err if last_err else RuntimeError("Failed to prune skeleton.")
    
    leaf_lengths = []
    stem_lengths = []

    try:
        if segmented_img is not None and leaf_obj is not None and len(leaf_obj) > 0:
            # จะวาดรูป debug ด้วย label = f"{sample_name}_leaf"
            _ = pcv.morphology.segment_euclidean_length(
                segmented_img=segmented_img, objects=leaf_obj, label=f"{sample_name}_leaf"
            )
    except Exception:
        pass

    try:
        if segmented_img is not None and stem_obj is not None and len(stem_obj) > 0:
            _ = pcv.morphology.segment_euclidean_length(
                segmented_img=segmented_img, objects=stem_obj, label=f"{sample_name}_stem"
            )
    except Exception:
        pass
    # --- HEIGHT (bbox along Y) ---
    hm, hm_src = _height_mask(slot_mask, stem_obj, sample_name)
    ys, xs = np.where(hm > 0)
    if ys.size > 0 and xs.size > 0:
        y_min, y_max = int(ys.min()), int(ys.max())
        x_min, x_max = int(xs.min()), int(xs.max())

        height_px = float(y_max - y_min + 1)
        length_px = float(x_max - x_min + 1)  # "ความยาว" ในที่นี้คือความกว้างตามแกน X

        height_mm = _px_to_mm(height_px)
        length_mm = _px_to_mm(length_px)

        # เก็บลง outputs → จะไป JSON
        pcv.outputs.add_observation(sample=sample_name, variable="height_px",
                                    trait="height", method=f"bbox_y ({hm_src})",
                                    scale="px", datatype=float, value=height_px, label="height_px")
        if height_mm is not None:
            pcv.outputs.add_observation(sample=sample_name, variable="height_mm",
                                        trait="height", method=f"px_to_mm ({hm_src})",
                                        scale="mm", datatype=float, value=float(height_mm), label="height_mm")

            pcv.outputs.add_observation(sample=sample_name, variable="length_px",
                                        trait="length", method=f"bbox_x ({hm_src})",
                                        scale="px", datatype=float, value=length_px, label="length_px")
        if length_mm is not None:
            pcv.outputs.add_observation(sample=sample_name, variable="length_mm",
                                        trait="length", method=f"px_to_mm ({hm_src})",
                                        scale="mm", datatype=float, value=float(length_mm), label="length_mm")

        # เซฟรูปดีบักให้เห็นเส้นวัด + ค่าตัวเลขบนภาพ
        # หมายเหตุ: rgb_img ที่ส่งเข้ามาใน analyze_one_side ตอนนี้เป็น "roi_img" แล้ว (จากข้อ 1)
        _save_size_debug(mask=slot_mask, roi_img=rgb_img,
                         xmin=x_min, xmax=x_max, ymin=y_min, ymax=y_max,
                         height_px=height_px, height_mm=height_mm,
                         length_px=length_px, length_mm=length_mm,
                         hm_src=hm_src, sample_name=sample_name)


    # === เพิ่มบล็อกสี ===
    color_feats = _analyze_color_side(rgb_img=rgb_img, mask=slot_mask, bins=36)
    if color_feats is not None:
        pcv.outputs.add_observation(sample=sample_name, variable='side_main_color',
                                    trait='main color', method='HSV histogram (side view)',
                                    scale='categorical', datatype=str, value=color_feats["side_main_color"],
                                    label='side main color')

        pcv.outputs.add_observation(sample=sample_name, variable='side_main_hue_deg',
                                    trait='dominant hue', method='HSV histogram (side view)',
                                    scale='degree', datatype=float, value=color_feats["side_main_hue_deg"],
                                    label='dominant hue (deg)')

        pcv.outputs.add_observation(sample=sample_name, variable='side_mean_hue_deg',
                                    trait='circular mean hue', method='circular mean (side view)',
                                    scale='degree', datatype=float, value=color_feats["side_mean_hue_deg"],
                                    label='mean hue (deg)')

        pcv.outputs.add_observation(sample=sample_name, variable='side_hsv_mean',
                                    trait='HSV mean', method='masked mean (side view)',
                                    scale='unit', datatype=dict, value=color_feats["side_hsv_mean"],
                                    label='mean HSV')

        pcv.outputs.add_observation(sample=sample_name, variable='side_lab_mean',
                                    trait='LAB mean', method='masked mean (side view)',
                                    scale='unit', datatype=dict, value=color_feats["side_lab_mean"],
                                    label='mean LAB')

        pcv.outputs.add_observation(sample=sample_name, variable='side_greenness_index',
                                    trait='greenness (-a*)', method='LAB a* proxy (side view)',
                                    scale='unit', datatype=float, value=color_feats["side_greenness_index"],
                                    label='greenness index')

def _save_size_debug(mask, roi_img, xmin, xmax, ymin, ymax,
                     height_px, height_mm, length_px, length_mm,
                     hm_src, sample_name):
    """เซฟรูปดีบักแสดงเส้นวัด H/W บน ROI (ถ้ามี) หรือบนหน้ากาก"""
    import cv2
    import numpy as np
    from . import config as cfg
    # เตรียมภาพแสดงผล
    if roi_img is None:
        vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    else:
        vis = roi_img.copy()
        # overlay โซนวัตถุจาง ๆ ให้เห็นขอบเขต
        overlay = vis.copy()
        overlay[mask > 0] = (0, 255, 0)
        vis = cv2.addWeighted(vis, 0.85, overlay, 0.15, 0)

    H, W = mask.shape[:2]
    cx = int((xmin + xmax) / 2)
    cy = int((ymin + ymax) / 2)

    # วาด "ความสูง" (แดง)
    cv2.line(vis, (cx, ymin), (cx, ymax), (0, 0, 255), 2)
    cv2.circle(vis, (cx, ymin), 4, (0, 0, 255), -1)
    cv2.circle(vis, (cx, ymax), 4, (0, 0, 255), -1)

    # วาด "ความยาว/กว้าง" (น้ำเงิน)
    cv2.line(vis, (xmin, cy), (xmax, cy), (255, 0, 0), 2)ชชชช
    cv2.circle(vis, (xmin, r,t
    cv2.circle(vis, (xmax, cy), 4, (255, 0, 0), -1)

    # ข้อความบนภาพ
    txt1 = f"H: {height_px:.1f}px" + (f" ({height_mm:.2f}mm)" if height_mm is not None else "")
    txt2 = f"W: {length_px:.1f}px" + (f" ({length_mm:.2f}mm)" if length_mm is not None else "")
    txt3 = f"source: {hm_src}"
    ytxt = max(20, ymin - 10)
    cv2.putText(vis, txt1, (10, ytxt), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, txt2, (10, ytxt + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(vis, txt3, (10, ytxt + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 160, 0), 2, cv2.LINE_AA)

    # ที่จะเซฟไฟล์
    base = getattr(cfg, "OUTPUT_DIR", None) or pcv.params.debug_outdir or "."
    Path(base, "processed").mkdir(parents=True, exist_ok=True)
    out_path = Path(base, "processed", f"{sample_name}_size_debug.png")
    cv2.imwrite(str(out_path), vis)
        