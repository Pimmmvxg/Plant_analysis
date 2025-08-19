import json
from pathlib import Path
import cv2
import numpy as np
from plantcv import plantcv as pcv
from . import config as cfg
from .io_utils import safe_readimage
from .masking import auto_select_mask, clean_mask
from .roi_top import make_grid_rois
from .roi_side import make_side_roi
from .analyze_top import analyze_one_top, add_global_density_and_color
from .analyze_side import analyze_one_side

def ensure_binary(mask):
    if mask is None:
        return None
    m = np.asarray(mask)
    if m.dtype != np.uint8:
        m = m.astype(np.uint8)
        
    m = np.where(m > 0, 255, 0).astype(np.uint8)
    return m

def run_one_image(rgb_img, filename):
    # 1) auto mask
    mask0, info = auto_select_mask(rgb_img)
    if cv2.countNonZero(mask0) > mask0.size / 2:
        mask0 = pcv.invert(gray_img=mask0)
    mask0 = ensure_binary(mask0)

    for k, v, trait, dt in [
        ('auto_channel', info['channel'], 'text', str),
        ('auto_method', info['method'], 'text', str),
        ('auto_object_type', info['object_type'], 'text', str),
        ('auto_ksize', str(info['ksize']), 'text', str),
        ('auto_area_ratio', float(info['area_ratio']), 'ratio', float),
        ('auto_n_components', int(info['n_components']), 'count', int),
        ('auto_solidity', float(info['solidity']), 'ratio', float),
    ]:
        pcv.outputs.add_observation(sample='default', variable=k, trait=trait,
                                    method='auto_select', scale='none',
                                    datatype=dt, value=v, label=k)

    # 2) clean mask
    mask_dilated = pcv.dilate(gray_img=mask0, ksize=2, i=1)
    try:
        mask_closed = pcv.close(mask=mask_dilated, ksize=5, shape='ellipse')
    except Exception:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_closed = cv2.morphologyEx(mask_dilated, cv2.MORPH_CLOSE, kernel)
        
    mask_closed = ensure_binary(mask_closed)
    mask_fill = pcv.fill(bin_img=mask_closed, size=30)
    mask_fill = clean_mask(mask_fill, close_ksize=5, min_obj_size=30)   
    mask_fill = ensure_binary(mask_fill)  
    
    if cfg.VIEW == "top":
        # ภาพรวมทั้งภาพ (ความหนาแน่น + สี)
        add_global_density_and_color(rgb_img, mask_fill)
        # วางกริดและวิเคราะห์รายช่อง
        rois, eff_r = make_grid_rois(rgb_img, cfg.ROWS, cfg.COLS, getattr(cfg, "ROI_RADIUS", None))
        union_mask = np.zeros_like(mask_fill, dtype=np.uint8)
        slots_with_obj = 0
        
        for i, roi_cnt in enumerate(rois):
            slot = np.zeros_like(mask_fill, dtype=np.uint8)
            cv2.drawContours(slot, [roi_cnt], -1, 255, thickness=cv2.FILLED)

            slot_mask = cv2.bitwise_and(mask_fill, slot)
            slot_mask = ensure_binary(slot_mask)

            if cv2.countNonZero(slot_mask) == 0:
                continue

            union_mask = cv2.bitwise_or(union_mask, slot_mask)

            r = i // cfg.COLS + 1
            c = i % cfg.COLS + 1
            sample_name = f"slot_{r}_{c}"

            analyze_one_top(slot_mask, sample_name, eff_r, rgb_img)
            slots_with_obj += 1
        
        if slots_with_obj == 0:
            raise RuntimeError("No objects inside any ROI cell.")
        extra = {
            "filename": filename,
            "view": "top",
            "roi_grid": f"{cfg.ROWS}x{cfg.COLS}",
            "roi_radius": int(eff_r),
            "roi_type": cfg.ROI_TYPE,
            "n_slots_with_objects": int(slots_with_obj),
        }
        return extra, union_mask
    
    else: # side
        slot_mask, (x, y, w, h) = make_side_roi(rgb_img, mask_fill, cfg.USE_FULL_IMAGE_ROI,
                                                cfg.ROI_X, cfg.ROI_Y, cfg.ROI_W, cfg.ROI_H)
        slot_mask = ensure_binary(slot_mask)
        if slot_mask is None or cv2.countNonZero(slot_mask) == 0:
            raise RuntimeError("No objects inside side ROI.")
        analyze_one_side(slot_mask, "default", rgb_img)
        extra = {
            "filename": filename,
            "view": "side",
            "roi_rect": f"({x},{y},{w},{h})",
        }
        return extra, slot_mask
    
def process_one(path: Path, out_dir: Path):
    pcv.params.debug = cfg.DEBUG_MODE
    pcv.params.debug_outdir = str(out_dir / "debug")
    pcv.params.dpi = 150
    pcv.outputs.clear()

    img, filename = safe_readimage(path)
    extra, mask = run_one_image(img, filename)

    # Flatten PlantCV observations → dict
    results_dict = pcv.outputs.observations.copy()
    flat = {}
    for sample, vars_ in results_dict.items():
        for var_name, record in vars_.items():
            col = f"{sample}_{var_name}" if sample != 'default' else var_name
            flat[col] = record.get('value', None)
    flat.update(extra)

    # Save per-image JSON
    json_dir = out_dir / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    out_json = json_dir / f"{Path(filename).stem}.json"
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(flat, f, ensure_ascii=False, indent=2)
        
    # Optional mask save
    if cfg.SAVE_MASK and mask is not None:
        (out_dir / "processed").mkdir(exist_ok=True, parents=True)
        pcv.print_image(img=mask, filename=str(out_dir / "processed" / f"{Path(filename).stem}_mask.png"))
        flat["mask_saved"] = True
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump(flat, f, ensure_ascii=False, indent=2)

    return str(out_json)