import json
from pathlib import Path
import cv2
import numpy as np
from plantcv import plantcv as pcv
from . import config as cfg
from .io_utils import safe_readimage
from .masking import clean_mask, ensure_binary, get_initial_mask
from .roi_top import make_grid_rois
from .roi_side import make_side_roi
from .analyze_top import analyze_one_top, add_global_density_and_color
from .analyze_side import analyze_one_side, _analyze_color_side
from .calibration import get_scale_from_checkerboard

_LAST_MM_PER_PX = getattr(cfg, "_LAST_MM_PER_PX", None)

def run_one_image(rgb_img, filename):
    global _LAST_MM_PER_PX
    
    mm_per_px, found_cb, scale_info = get_scale_from_checkerboard(
        image=rgb_img,
        square_size_mm=getattr(cfg, "CHECKER_SQUARE_MM", 8.0),
        pattern_size=getattr(cfg, "CHECKER_PATTERN", (4, 4)),
        previous_scale=_LAST_MM_PER_PX,
        fallback_scale=getattr(cfg, "FALLBACK_MM_PER_PX", 10.0 / 51.0),
        debug_name=f"{Path(filename).stem}_checker"
    )
    # อัปเดตค่าใช้งานจริง
    setattr(cfg, "MM_PER_PX", float(mm_per_px))
    _LAST_MM_PER_PX = float(mm_per_px)

    # log ลง outputs
    pcv.outputs.add_observation(sample='default', variable='mm_per_px',
                                trait='scale', method=scale_info, scale='mm/px',
                                datatype=float, value=float(mm_per_px), label='mm_per_px')
    
    # 1) initial mask (auto ถ้าไม่กำหนด, manual ถ้าตั้ง MASK_PATH/MASK_SPEC)
    mask0, info = get_initial_mask(rgb_img)
    mask0 = ensure_binary(mask0)
    
    H, W = rgb_img.shape[:2]
    
    # บันทึกที่มา (auto | manual_file | manual_spec)
    pcv.outputs.add_observation(sample='default', variable='mask_source',
                                trait='text', method='mask_select', scale='none',
                                datatype=str, value=info.get('source', 'auto'), label='mask_source')
    mm_per_px = getattr(cfg, "MM_PER_PX", None)
    if mm_per_px:
        height_mm = float(info.get('Height_shape', H)) * mm_per_px
        width_mm  = float(info.get('Width_shape', W)) * mm_per_px
    else:
        height_mm = None
        width_mm  = None
    # เก็บข้อมูลเมตาเดิมไว้เหมือนเคย (ใช้ key เดิมเพื่อความเข้ากันได้)
    for k, v, trait, dt in [
        ('auto_channel', info.get('channel'), 'text', str),
        ('auto_method', info.get('method'), 'text', str),
        ('auto_object_type', info.get('object_type'), 'text', str),
        ('auto_ksize', str(info.get('ksize')), 'text', str),
        ('auto_area_ratio', float(info.get('area_ratio', 0.0)), 'ratio', float),
        ('auto_n_components', int(info.get('n_components', 0)), 'count', int),
        ('auto_solidity', float(info.get('solidity', 0.0)), 'ratio', float),
        ('auto_height_px', int(info.get('Height_shape', H)), 'length', int),
        ('auto_width_px', int(info.get('Width_shape', W)), 'length', int),
        ('auto_height_mm', height_mm, 'length', float),
        ('auto_width_mm', width_mm, 'length', float),
    ]:
        pcv.outputs.add_observation(sample='default', variable=k, trait=trait,
                                    method='mask_select', scale='none',
                                    datatype=dt, value=v, label=k)

    # ถ้าเป็น manual_file ให้เก็บ path ไว้ด้วย (เผื่อ debug/trace)
    if 'mask_path' in info:
        pcv.outputs.add_observation(sample='default', variable='mask_path',
                                    trait='text', method='mask_select', scale='none',
                                    datatype=str, value=info['mask_path'], label='mask_path')


    # 2) clean mask
    mask_dilated = pcv.dilate(gray_img=mask0, ksize=2, i=1)
    try:
        mask_closed = pcv.close(mask=mask_dilated, ksize=5, shape='ellipse')
    except Exception:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_closed = cv2.morphologyEx(mask_dilated, cv2.MORPH_CLOSE, kernel)
        
    mask_closed = ensure_binary(mask_closed)
    _dbg = lambda *a: (print(*a) if getattr(cfg, "DEBUG_MODE", "none") == "print" else None)
    _dbg("DEBUG mask_closed:", mask_closed.dtype, np.unique(mask_closed)[:5])

    mask_fill = pcv.fill(bin_img=mask_closed, size=30)
    mask_fill = clean_mask(mask_fill, close_ksize=5, min_obj_size=30)
    mask_fill = ensure_binary(mask_fill)
    _dbg("DEBUG mask_fill:", mask_fill.dtype, np.unique(mask_fill)[:5])

    # --- ขนาดรวมทั้งภาพ (px) ---
    plant_size = int(cv2.countNonZero(mask_fill))
    pcv.outputs.add_observation(
        sample='default', variable='plant_size',
        trait='area', method='mask_pixel_count', scale='px',
        datatype=int, value=plant_size, label='plant_size'
    )

    def _area_mm2_from_px(px_area: int):
        if hasattr(cfg, "MM_PER_PX") and cfg.MM_PER_PX:
            mm_per_px = float(cfg.MM_PER_PX)
            return float(px_area) * (mm_per_px ** 2)
        if hasattr(cfg, "DPI") and cfg.DPI:
            px_per_mm = float(cfg.DPI) / 25.4
            return float(px_area) / (px_per_mm ** 2)
        return None

    mm2_total = _area_mm2_from_px(plant_size)
    if mm2_total is not None:
        pcv.outputs.add_observation(
            sample='default', variable='plant_area_mm2',
            trait='area', method='px_to_mm2', scale='mm2',
            datatype=float, value=float(mm2_total), label='plant_area_mm2'
        )

    #Top view
    if cfg.VIEW == "top":
        _dbg("DEBUG entering TOP pipeline")
        try:
            rois, eff_r = make_grid_rois(
                rgb_img, cfg.ROWS, cfg.COLS, getattr(cfg, "ROI_RADIUS", None)
            )
        except Exception as e:
            raise RuntimeError(f"make_grid_rois failed: {e}")
        _dbg("DEBUG rois:", len(rois), "eff_r:", eff_r)

        # (optional) เซฟ overlay
        overlay = rgb_img.copy()
        for cnt in rois:
            cv2.drawContours(overlay, [cnt], -1, (0, 255, 0), 2)
        base = getattr(cfg, "OUTPUT_DIR", None) or pcv.params.debug_outdir or "."
        (Path(base) / "processed").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(Path(base) / "processed" / f"{Path(filename).stem}_rois.png"), overlay)

        # ภาพรวมทั้งภาพ (density + color)
        add_global_density_and_color(rgb_img, mask_fill)

        union_mask = np.zeros_like(mask_fill, dtype=np.uint8)
        slots_with_obj = 0

        for i, roi_cnt in enumerate(rois):
            # หา center ของ contour
            M = cv2.moments(roi_cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # สร้าง ROI object (วงกลม)
            roi = pcv.roi.circle(img=rgb_img, x=cx, y=cy, r=int(eff_r))

            # filter ด้วย ROI (ใช้ค่าใน cfg)
            filtered_mask = pcv.roi.filter(
                mask=mask_fill, roi=roi,
                roi_type=getattr(cfg, "ROI_TYPE", "partial")
            )
            filtered_mask = ensure_binary(filtered_mask)
            if cv2.countNonZero(filtered_mask) == 0:
                continue

            # create_labels: รองรับทั้ง tuple และ single return
            try:
                result = pcv.create_labels(mask=filtered_mask, rois=None)
                if isinstance(result, tuple):
                    labeled_mask, n_labels = result
                else:
                    labeled_mask = result
                    n_labels = int(labeled_mask.max())
            except Exception as e:
                _dbg("WARN: create_labels failed:", e)
                continue

            if int(n_labels) <= 0:
                continue

            r = i // cfg.COLS + 1
            c = i %  cfg.COLS + 1
            per_slot_count = 0
            per_slot_area_sum = 0

            # วนวิเคราะห์รายต้น (plant)
            for j in range(1, int(n_labels) + 1):
                single = np.where(labeled_mask == j, 255, 0).astype(np.uint8)
                if cv2.countNonZero(single) < getattr(cfg, "MIN_PLANT_AREA", 200):
                    continue

                # ขนาดต่อ plant
                area_px = int(cv2.countNonZero(single))
                per_slot_area_sum += area_px
                pcv.outputs.add_observation(
                    sample=f"slot_{r}_{c}_obj{j}", variable="area_px",
                    trait="area", method="countNonZero", scale="px",
                    datatype=int, value=area_px, label="area_px"
                )
                mm2 = _area_mm2_from_px(area_px)
                if mm2 is not None:
                    pcv.outputs.add_observation(
                        sample=f"slot_{r}_{c}_obj{j}", variable="area_mm2",
                        trait="area", method="px_to_mm2", scale="mm2",
                        datatype=float, value=float(mm2), label="area_mm2"
                    )

                # รวมหน้ากาก/วิเคราะห์สี‑รูปร่างราย plant
                union_mask = cv2.bitwise_or(union_mask, single)
                union_mask = ensure_binary(union_mask)
                analyze_one_top(single, f"slot_{r}_{c}_obj{j}", eff_r, rgb_img)

                per_slot_count += 1
                slots_with_obj += 1

            # บันทึกจำนวน/พื้นที่รวมต่อ slot
            pcv.outputs.add_observation(
                sample=f"slot_{r}_{c}", variable="n_plants_in_slot",
                trait="count", method="roi_filter", scale="none",
                datatype=int, value=int(per_slot_count), label="n_plants"
            )
            pcv.outputs.add_observation(
                sample=f"slot_{r}_{c}", variable="slot_area_sum_px",
                trait="area", method="sum(label_area)", scale="px",
                datatype=int, value=int(per_slot_area_sum), label="slot_area_sum_px"
            )
            mm2_slot = _area_mm2_from_px(per_slot_area_sum)
            if mm2_slot is not None:
                pcv.outputs.add_observation(
                    sample=f"slot_{r}_{c}", variable="slot_area_sum_mm2",
                    trait="area", method="px_to_mm2", scale="mm2",
                    datatype=float, value=float(mm2_slot), label="slot_area_sum_mm2"
                )

        if slots_with_obj == 0:
            raise RuntimeError("No objects inside any ROI cell.")

        extra = {
            "filename": filename,
            "view": "top",
            "roi_grid": f"{cfg.ROWS}x{cfg.COLS}",
            "roi_radius": int(eff_r),
            "roi_type": getattr(cfg, "ROI_TYPE", "partial"),
            "n_slots_with_objects": int(slots_with_obj),
        }
        return extra, union_mask

    else: # side
        slot_mask, (x, y, w, h) = make_side_roi(
        rgb_img, mask_fill, cfg.USE_FULL_IMAGE_ROI,
        cfg.ROI_X, cfg.ROI_Y, cfg.ROI_W, cfg.ROI_H
    )

    # ถ้าเผลอเป็น 3-channel (H,W,3) บังคับแปลงเป็นเทา
    if slot_mask is None:
        raise RuntimeError("make_side_roi returned None for slot_mask.")
    if slot_mask.ndim == 3:
        slot_mask = cv2.cvtColor(slot_mask, cv2.COLOR_BGR2GRAY)

    slot_mask = ensure_binary(slot_mask)

    # DEBUG: ให้เห็น dtype, unique, shape, และจำนวนพิกเซลที่เป็น 1
    print("DEBUG side slot_mask:", slot_mask.dtype, np.unique(slot_mask)[:5],
          slot_mask.shape, "nz=", int(cv2.countNonZero(slot_mask)))

    if cv2.countNonZero(slot_mask) == 0:
        raise RuntimeError("No objects inside side ROI.")
        
    analyze_one_side(slot_mask, "default", rgb_img)
    roi_img = rgb_img[y:y+h, x:x+W].copy()
    analyze_one_side(slot_mask, "default", roi_img)
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