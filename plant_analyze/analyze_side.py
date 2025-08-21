import numpy as np
from plantcv import plantcv as pcv
from .color import get_color_name, _hue_to_degrees
import cv2
from . import config as cfg
import os

# --- HEIGHT helpers ---
def _endpoints_and_height_max_dy(skel_bin):
    """หา endpoints ของ skeleton แล้วเลือกคู่ที่มี Δy มากที่สุด; คืน (p_top, p_bot, height_px)"""
    K = np.array([[1,1,1],[1,10,1],[1,1,1]], dtype=np.uint8)
    sk = (skel_bin > 0).astype(np.uint8)
    nbr = cv2.filter2D(sk, -1, K)
    end_mask = ((sk > 0) & (nbr == 11)).astype(np.uint8)
    ys, xs = np.where(end_mask > 0)
    if ys.size >= 2:
        # เลือกคู่ที่ต่าง y มากสุด (ไม่ใช่ระยะเฉียง)
        idx_top = int(np.argmin(ys))
        idx_bot = int(np.argmax(ys))
        p_top = (int(xs[idx_top]), int(ys[idx_top]))
        p_bot = (int(xs[idx_bot]), int(ys[idx_bot]))
        return p_top, p_bot, float(abs(p_bot[1] - p_top[1]))
    # fallback: ใช้ช่วง y ของทั้ง skeleton
    ys_all, xs_all = np.where(sk > 0)
    if ys_all.size >= 2:
        y0, y1 = int(ys_all.min()), int(ys_all.max())
        x0 = int(xs_all[ys_all.argmin()])
        x1 = int(xs_all[ys_all.argmax()])
        return (x0, y0), (x1, y1), float(y1 - y0)
    return None, None, None

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

    # NOTE: ถ้าภาพของคุณเป็น BGR ให้เปลี่ยนเป็น COLOR_BGR2HSV / COLOR_BGR2LAB
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

    # dominant hue จากฮิสโตแกรม
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
def _px_to_mm(px_len:float):
    if px_len is None:
        return None
    k = float(getattr(cfg, "FAKE_MM_PER_PX", 0.5))  # เปลี่ยนเป็นค่าที่อยากลอง เช่น 0.42
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
        raise last_err if last_err else RuntimeError("Failed to prune skeleton.")
    
    _ = pcv.morphology.fill_segments(mask=slot_mask, objects=leaf_obj, label=sample_name)
    branch_pts_mask = pcv.morphology.find_branch_pts(skel_img=pruned_skel, mask=slot_mask, label=sample_name)
    _ = pcv.morphology.segment_euclidean_length(segmented_img=segmented_img, objects=leaf_obj, label=sample_name)
    _ = pcv.analyze.size(img=rgb_img, labeled_mask=slot_mask, label=sample_name)
    
    # ===== วัดความสูงแบบ Δy ของ endpoints (กันจับใบซ้าย-ขวา) =====
    # 1) เลือก mask สำหรับวัด: stem-only (ถ้าเปิดใช้และมี) หรือทั้งก้อน
    height_mask, height_scope = _height_mask(slot_mask, stem_obj, sample_name)

    # 2) ทำ skeleton เฉพาะส่วนที่จะวัด
    skel_h = pcv.morphology.skeletonize(mask=height_mask)

    # 3) หา endpoints และเลือกคู่ที่ Δy มากที่สุด (แนวดิ่ง)
    p_top, p_bot, height_px = _endpoints_and_height_max_dy(skel_h)

    # 4) บันทึกผล (px และ mm ถ้ามีตัวแปลง)
    if height_px is not None and height_px > 0:
        pcv.outputs.add_observation(sample=sample_name, variable='plant_height_px',
                                    trait='height', method=f'endpoints_max_dy ({height_scope})',
                                    scale='px', datatype=float, value=float(height_px),
                                    label='plant height (px)')
        height_mm = _px_to_mm(height_px)
        if height_mm is not None:
            pcv.outputs.add_observation(sample=sample_name, variable='plant_height_mm',
                                        trait='height', method=f'endpoints_max_dy + scale ({height_scope})',
                                        scale='mm', datatype=float, value=float(height_mm),
                                        label='plant height (mm)')

        # ===== DEBUG overlay =====
        try:
            overlay = rgb_img.copy()
            if p_top and p_bot:
                # เส้นแนวดิ่งที่วัด (จริงๆ เป็นเส้นตรงระหว่างจุดบน-ล่าง)
                cv2.line(overlay, p_top, p_bot, (0, 255, 0), 2)
                cv2.circle(overlay, p_top, 5, (0, 0, 255), -1)  # จุดบน (แดง)
                cv2.circle(overlay, p_bot, 5, (255, 0, 0), -1)  # จุดล่าง (ฟ้า)
            skel_rgb = cv2.cvtColor((skel_h > 0).astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
            overlay = cv2.addWeighted(overlay, 1.0, skel_rgb, 0.4, 0)
            dbg_dir = pcv.params.debug_outdir or "."
            os.makedirs(dbg_dir, exist_ok=True)
            out_name = os.path.join(dbg_dir, f"{sample_name}_side_height_debug.png")
            cv2.imwrite(out_name, overlay)
            pcv.outputs.add_observation(sample=sample_name, variable='plant_height_debug_image',
                                        trait='debug', method=f'overlay skeleton + vertical endpoints ({height_scope})',
                                        scale='path', datatype=str, value=out_name, label='height debug image')
        except Exception as e:
            print("WARN: save height debug failed:", e)
                
        num_leaf_segments = int(len(leaf_obj)) if leaf_obj is not None else 0
        num_stem_segments = int(len(stem_obj)) if stem_obj is not None else 0
        num_branch_points = int(np.count_nonzero(branch_pts_mask)) if branch_pts_mask is not None else 0


        pcv.outputs.add_observation(sample=sample_name, variable='num_leaf_segments',
                                    trait='count', method='pcv.morphology.segment_sort',
                                    scale='count', datatype=int, value=num_leaf_segments, label='leaf segments')
        pcv.outputs.add_observation(sample=sample_name, variable='num_stem_segments',
                                    trait='count', method='pcv.morphology.segment_sort',
                                    scale='count', datatype=int, value=num_stem_segments, label='stem segments')
        pcv.outputs.add_observation(sample=sample_name, variable='num_branch_points',
                                    trait='count', method='pcv.morphology.find_branch_pts',
                                    scale='count', datatype=int, value=num_branch_points, label='branch points')
        
        _ = pcv.analyze.size(img=rgb_img, labeled_mask=slot_mask, label=sample_name)

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
