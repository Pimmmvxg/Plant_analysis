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
        
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
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
    roi_area_est = float(np.pi * (eff_r ** 2))
    coverage_local = area_px / max(roi_area_est, 1.0)
    
    perim = 0.0
    convex_ratio = 0.0
    circularity = 0.0
    extent = 0.0
    
    _fc = cv2.findContours(slot_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = _fc[2] if len(_fc) == 3 else _fc[0]
    if contours:
        big = contours[int(np.argmax([cv2.contourArea(c) for c in contours]))]
        a_obj = float(cv2.contourArea(big))
        p_obj = float(cv2.arcLength(big, True))
        perim = p_obj
        hull = cv2.convexHull(big)
        a_hull = float(cv2.contourArea(hull)) if hull is not None else 0.0
        convex_ratio = (a_obj / a_hull) if a_hull > 0 else 0.0
        circularity = (4.0 * np.pi * a_obj / (p_obj ** 2)) if p_obj > 0 else 0.0
        x, y, w, h = cv2.boundingRect(big)
        extent = (a_obj / float(w * h)) if w * h > 0 else 0.0
        
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