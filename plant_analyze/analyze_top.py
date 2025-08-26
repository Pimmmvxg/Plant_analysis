import numpy as np
import cv2
from plantcv import plantcv as pcv
from .color import get_color_name

def _safe_starts(arr):
    if arr is None or len(arr) == 0:
        return (0.0, 0.0, 0.0)
    return (float(np.mean(arr)), float(np.std(arr)), float(np.median(arr)))

def add_global_density_and_color(rgb_img, mask_fill):
    total_px = mask_fill.size
    white_px = int(cv2.countNonZero(mask_fill))
    coverage_ratio = white_px / max(total_px, 1)
    _fc = cv2.findContours(mask_fill.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = _fc[2] if len(_fc) == 3 else _fc[0]
    n_comp = len(contours)
    areas = [cv2.contourArea(c) for c in contours] if contours else []
    a_med = float(np.median(areas)) if areas else 0.0
    a_mean = float(np.mean(areas)) if areas else 0.0

    if areas:
        big = contours[int(np.argmax(areas))]
        hull = cv2.convexHull(big)
        a_obj  = float(cv2.contourArea(big))
        a_hull = float(cv2.contourArea(hull)) if hull is not None else 0.0
        big_solidity = (a_obj / a_hull) if a_hull > 0 else 0.0
    else:
        big_solidity = 0.0
        
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2HSV)
    h, s, v = [ch[mask_fill > 0] for ch in cv2.split(hsv)]
    h_mean, h_std, h_med = _safe_starts(h.astype(np.float32) * 2.0)
    s_mean, s_std, s_med = _safe_starts(s.astype(np.float32))
    v_mean, v_std, v_med = _safe_starts(v.astype(np.float32))
    main_color = get_color_name(h_med)
    
    pcv.outputs.add_observation(sample='default', variable='all_coverage_ratio',
                                trait='ratio', method='count_nonzero/size', scale='none',
                                datatype=float, value=float(coverage_ratio), label='coverage')
    pcv.outputs.add_observation(sample='default', variable='all_n_components',
                                trait='count', method='findContours', scale='count',
                                datatype=int, value=int(n_comp), label='components')
    pcv.outputs.add_observation(sample='default', variable='all_comp_area_mean',
                                trait='area', method='mean(contourArea)', scale='px',
                                datatype=float, value=float(a_mean), label='mean component area')
    pcv.outputs.add_observation(sample='default', variable='all_comp_area_median',
                                trait='area', method='median(contourArea)', scale='px',
                                datatype=float, value=float(a_med), label='median component area')
    pcv.outputs.add_observation(sample='default', variable='all_big_solidity',
                                trait='ratio', method='largest(area)/convexHull', scale='none',
                                datatype=float, value=float(big_solidity), label='largest solidity')
    
    for (nm, val) in [
        ('hue_mean', h_mean), ('hue_std', h_std), ('hue_median', h_med),
        ('saturation_mean', s_mean), ('saturation_std', s_std), ('saturation_median', s_med),
        ('value_mean', v_mean), ('value_std', v_std), ('value_median', v_med)
    ]:
        pcv.outputs.add_observation(sample='default', variable=f'global_{nm}',
                                    trait='color', method='HSV stats(masked)', scale='unit',
                                    datatype=float, value=float(val), label=nm)
    pcv.outputs.add_observation(sample='default', variable='global_color_name',
                                trait='text', method='HSV median(name)', scale='none',
                                datatype=str, value=main_color, label='color name')

def analyze_one_top(slot_mask, sample_name, eff_r, rgb_img): 
    if slot_mask is None:
        raise RuntimeError("slot_mask is None (top)")
    if slot_mask.ndim == 3:
        slot_mask = cv2.cvtColor(slot_mask, cv2.COLOR_BGR2GRAY)
    if slot_mask.dtype != np.uint8:
        slot_mask = slot_mask.astype(np.uint8)
    slot_mask = np.where(slot_mask > 0, 255, 0).astype(np.uint8)
    
    area_px = int(cv2.countNonZero(slot_mask))
    roi_area_est = float(np.pi * (eff_r ** 2)) if (eff_r is not None and eff_r > 0) else 0.0
    coverage_local = (area_px / roi_area_est) if roi_area_est > 0 else 0.0
    
    _fc = cv2.findContours(slot_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = _fc[2] if len(_fc) == 3 else _fc[0]

    perim = 0.0
    convex_ratio = 0.0
    circularity = 0.0
    extent = 0.0
    n_comp = 0

    if contours:
        n_comp = len(contours)
        area_union = float(cv2.countNonZero(slot_mask))  # union area

        # เส้นรอบรูป = ผลรวมของทุกคอนทัวร์
        perim_sum = float(sum(cv2.arcLength(c, True) for c in contours))
        perim = perim_sum

        # hull รวม = hull(all_points)
        all_pts = np.vstack(contours)
        hull = cv2.convexHull(all_pts)
        a_hull = float(cv2.contourArea(hull)) if len(hull) >= 3 else 0.0
        convex_ratio = (area_union / a_hull) if a_hull > 0 else 0.0  # solidity ของ union

        # bbox รวมจาก hull
        x, y, w, h = cv2.boundingRect(hull)
        extent = (area_union / float(w * h)) if w * h > 0 else 0.0

        # circularity ของ union
        circularity = (4.0 * np.pi * area_union / (perim_sum ** 2)) if perim_sum > 0 else 0.0

        
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mask_idx = slot_mask > 0
    hh = h[mask_idx].astype(np.float32) * 2.0
    ss = s[mask_idx].astype(np.float32)
    vv = v[mask_idx].astype(np.float32)
    hue_med = float(np.median(hh)) if hh.size > 0 else 0.0
    col_name = get_color_name(hue_med)  
    
    pairs = [
        ('slot_area_px', area_px, 'area', 'px'),
        ('slot_coverage_ratio', coverage_local, 'ratio', 'none'),
        ('slot_perimeter', perim, 'length', 'px'),
        ('slot_convex_ratio', convex_ratio, 'ratio', 'none'),
        ('slot_circularity', circularity, 'ratio', 'none'),
        ('slot_extent', extent, 'ratio', 'none'),
        ('slot_n_components', float(n_comp), 'count', 'none'),
        ('hue_median', hue_med, 'color', 'unit'),
        ('sat_mean', float(np.mean(ss)) if ss.size>0 else 0.0, 'color', 'unit'),
        ('val_mean', float(np.mean(vv)) if vv.size>0 else 0.0, 'color', 'unit'),
    ]

    for var, val, trait, scale in pairs:
        pcv.outputs.add_observation(sample=sample_name, variable=var,
                                    trait=trait, method='no-skeleton stats', scale=scale,
                                    datatype=float, value=float(val), label=var)
    pcv.outputs.add_observation(sample=sample_name, variable='color_name',
                                trait='text', method='hue_median→name', scale='none',
                                datatype=str, value=col_name, label='dominant color')
    
def save_top_overlay(
    rgb_img,
    slot_mask,
    contours=None,
    eff_r=None,
    sample_name="default",
    mm_per_px: float | None = None,  # ตัวเลือก: ใส่สเกลจริง (มม./พิกเซล) เพื่อแปลงพื้นที่
):
    """
    วาด overlay รวมทุกคอนทัวร์ใน ROI:
      - contours ทั้งหมด (เขียว)
      - convex hull ของจุดรวมทั้งหมด (ฟ้า)
      - bounding box ของ hull รวม (เหลือง)
      - centroid ของ union mask (แดง)
      - วงกลม ROI (รัศมี eff_r) รอบ centroid
      - แถบข้อความสรุป: 'สีหลัก' และ 'พื้นที่หน่วยจริง' (ถ้ามี mm_per_px) มิฉะนั้นแสดงเป็น px²

    เซฟรูปเป็น: {pcv.params.debug_outdir}/{sample_name}_top_overlay.png
    คืนค่าเป็น path ไฟล์ที่เซฟ
    """
    import os
    import numpy as np
    import cv2
    from plantcv import plantcv as pcv
    try:
        from .color import get_color_name
    except Exception:
        # fallback เผื่อเรียกใช้แบบสแตนด์อโลน
        def get_color_name(hue_degree: float) -> str:
            return "unknown"

    if rgb_img is None or slot_mask is None:
        return None

    # --- ทำให้ mask เป็น binary uint8: วัตถุ = 255, ฉากหลัง = 0 ---
    m = slot_mask
    if m.ndim == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    if m.dtype != np.uint8:
        m = m.astype(np.uint8)
    m = np.where(m > 0, 255, 0).astype(np.uint8)

    # --- หา contours ถ้าไม่ส่งเข้ามา ---
    if contours is None:
        _fc = cv2.findContours(m.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = _fc[2] if len(_fc) == 3 else _fc[0]

    overlay = rgb_img.copy()

    # 1) วาดทุกคอนทัวร์ = เขียว (BGR: 0,255,0)
    if contours:
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

    # 2) วาด convex hull รวม = ฟ้า (BGR: 255,0,0) และ bbox = เหลือง (BGR: 0,255,255)
    hull = None
    if contours:
        all_pts = np.vstack(contours)  # (N,1,2)
        hull = cv2.convexHull(all_pts)
        cv2.drawContours(overlay, [hull], -1, (255, 0, 0), 2)  # ฟ้า
        x, y, w, h = cv2.boundingRect(hull)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 255), 2)  # เหลือง

    # 3) centroid ของ union mask + วงกลม ROI (ชมพู/ม่วง BGR: 255,0,255)
    M = cv2.moments(m, binaryImage=True)
    cx = cy = None
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.circle(overlay, (cx, cy), 4, (0, 0, 255), -1)  # แดง
        if eff_r is not None and eff_r > 0:
            cv2.circle(overlay, (cx, cy), int(eff_r), (255, 0, 255), 2)

    # 4) คำนวณสีหลัก (hue median ภายใน mask) -> ชื่อสี
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mask_idx = (m > 0)
    if np.count_nonzero(mask_idx) > 0:
        hue_deg = (h[mask_idx].astype(np.float32) * 2.0)  # OpenCV H: 0..179 → degree
        hue_med = float(np.median(hue_deg))
        color_name = get_color_name(hue_med)
    else:
        hue_med = 0.0
        color_name = "unknown"

    # 5) พื้นที่หน่วยจริง (ถ้ามี mm_per_px) ถ้าไม่มีเป็น px²
    area_px = int(cv2.countNonZero(m))
    if mm_per_px is not None and mm_per_px > 0:
        area_mm2 = float(area_px) * (mm_per_px ** 2)
        area_cm2 = area_mm2 / 100.0
        area_text = f"{area_mm2:,.2f} mm² ({area_cm2:,.2f} cm²)"
    else:
        area_text = f"{area_px:,} px²"

    # 6) แถบข้อความสรุป
    text = f"Main Color: {color_name} | Area: {area_text}"
    y0 = 30
    pad_w = max(10 + len(text) * 9, 260)  # กันข้อความโดนตัด
    cv2.rectangle(overlay, (10, y0 - 22), (10 + pad_w, y0 + 8), (0, 0, 0), -1)  # พื้นดำทึบ
    cv2.putText(overlay, text, (12, y0),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    # 7) บันทึกไฟล์
    outdir = pcv.params.debug_outdir or "."
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, f"{sample_name}_top_overlay.png")
    try:
        cv2.imwrite(out_path, overlay)
    except Exception:
        # เผื่อบางระบบเขียนไฟล์ตรง ๆ ไม่ได้
        pcv.print_image(img=overlay, filename=out_path)

    return out_path
