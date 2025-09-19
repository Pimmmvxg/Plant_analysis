import json
from pathlib import Path
import cv2
import numpy as np
from plantcv import plantcv as pcv
from . import config as cfg
from .io_utils import safe_readimage
from .masking import clean_mask, ensure_binary, get_initial_mask
from .roi_top import make_grid_rois
from .roi_side import make_side_roi ,make_side_rois_auto
from .analyze_top import analyze_one_top, add_global_density_and_color, combine_top_overlays, save_top_overlay
from .analyze_side import analyze_one_side, get_side_legend, combine_side_overlays, save_side_overlay
from .calibration import get_scale
from Thingsboard.send_data import publish_data
import multiprocessing
import threading
import concurrent.futures
import time
from .stem_rescue import add_v_connected_to_a

_LAST_MM_PER_PX = getattr(cfg, "_LAST_MM_PER_PX", None)
_global_lock = threading.Lock()

def run_one_image(rgb_img, filename):
    global _LAST_MM_PER_PX
    sample_name = Path(filename).stem

    # 0) คาลิเบรตสเกล 1 ครั้ง/ภาพ (จำค่าเดิมไว้เป็น previous)
    scale, found, scale_info = get_scale(
        image=rgb_img,
        prefer=("rectangle", "checkerboard"),
        previous_scale=_LAST_MM_PER_PX,
        fallback_scale= cfg.FALLBACK_MM_PER_PX,
        rectangle_kwargs={
            "rect_size_mm": cfg.RECT_SIZE_MM,
            "crop_top_ratio": 0.80 if cfg.VIEW == "side" else 1.0,
            "min_area": 50000,
            "eps_fraction": 0.04,
            "rect_tol": 0.50,
            "min_rectangularity": 0.30,
            "save_debug": True,
            "debug_name": f"{sample_name}_rect_scale",
        },
        checker_kwargs={
            "square_size_mm": cfg.CHECKER_SQUARE_MM,
            "pattern_size": cfg.CHECKER_PATTERN,
            "save_debug": True,
            "debug_name": f"{sample_name}_checker_scale",
        },
    )

    # 1) กระจายผลให้ทั้ง pipeline ใช้ร่วมกัน
    setattr(cfg, "MM_PER_PX", float(scale))
    _LAST_MM_PER_PX = float(scale)

    # 2) log ลง PlantCV outputs
    pcv.outputs.add_observation(
        sample="default", variable="mm_per_px",
        trait="scale", method=scale_info, scale="mm/px",
        datatype=float, value=float(scale), label="mm_per_px"
    )
    pcv.outputs.add_observation(
        sample="default", variable="scale_source",
        trait="text", method="scale_info", scale="none",
        datatype=str, value=scale_info, label="scale_source"
    )
    
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
        #('auto_area_ratio', float(info.get('area_ratio', 0.0)), 'ratio', float),
        ('auto_n_components', int(info.get('n_components', 0)), 'count', int),
        #('auto_solidity', float(info.get('solidity', 0.0)), 'ratio', float),
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

    mask_fill = pcv.fill(bin_img=mask_closed, size=300)
    mask_fill = clean_mask(mask_fill, close_ksize=5, min_obj_size=7000)
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
    
    # ========= STEM RESCUE (เฉพาะ TOP) =========
    if str(getattr(cfg, "VIEW", "top")).lower() == "top" and getattr(cfg, "ENABLE_STEM_RESCUE", True):
        old_leaf_mask = mask_fill.copy()
        rgb_for_stem = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

        mask_fill, dbg = add_v_connected_to_a(
            rgb=rgb_for_stem,
            base_a_mask=old_leaf_mask,
            method=getattr(cfg, "STEM_V_METHOD", "fixed"),   # "fixed"|"otsu"|"percentile"
            v_min=getattr(cfg, "STEM_V_MIN", 190),
            v_max=getattr(cfg, "STEM_V_MAX", 245),
            percentile=getattr(cfg, "STEM_V_PERCENTILE", 85),
            s_max=getattr(cfg, "STEM_S_MAX", None),         
            glare_v=getattr(cfg, "STEM_GLARE_V", 235),
            glare_s=getattr(cfg, "STEM_GLARE_S", 55),
            near_px=getattr(cfg, "STEM_NEAR_PX", 10),
            geo_iters=getattr(cfg, "STEM_GEO_ITERS", 100),
            open_k=getattr(cfg, "STEM_OPEN_K", 3),
            min_area_keep=getattr(cfg, "STEM_MIN_AREA_KEEP", 300),
            connect_mode=getattr(cfg, "STEM_CONNECT_MODE", "geo"),
            cc_close_k=getattr(cfg, "STEM_CC_CLOSE_K", 0),
        )

        # (optional) เซฟ debug
        base = getattr(cfg, "OUTPUT_DIR", None) or pcv.params.debug_outdir or "."
        dbgdir = Path(base) / "processed"; dbgdir.mkdir(parents=True, exist_ok=True)
        stub = Path(filename).stem
        cv2.imwrite(str(dbgdir / f"{stub}_leaf_before_stem.png"), ensure_binary(old_leaf_mask))
        cv2.imwrite(str(dbgdir / f"{stub}_leaf_after_stem.png"),  ensure_binary(mask_fill))
        cv2.imwrite(str(dbgdir / f"{stub}_vmask.png"), dbg["vmask"])
        cv2.imwrite(str(dbgdir / f"{stub}_v_connected.png"), dbg["v_connected"])
    #Top view
    if cfg.VIEW == "top":
        _dbg("DEBUG entering TOP pipeline")
        if getattr(cfg, "TOP_ROI_MODE", "grid") == "auto":
            # ---- โหมด auto (ใหม่) ----
            from .roi_top import make_top_rois_auto
            rois = make_top_rois_auto(
                rgb_img=rgb_img,
                mask_fill=mask_fill,
                cfg=cfg,
                min_area_px=getattr(cfg, "TOP_MIN_PLANT_AREA", getattr(cfg, "MIN_PLANT_AREA", 800)),
                merge_gap_px=getattr(cfg, "TOP_MERGE_GAP", 20),
                close_iters=getattr(cfg, "TOP_CLOSE_ITERS", 1),
                debug_out_path=str((Path(getattr(cfg, "OUTPUT_DIR", None) or pcv.params.debug_outdir or ".")
                                    / "processed" / f"{Path(filename).stem}_top_rois_auto.png"))
            )
            if not rois:
                raise RuntimeError("No objects detected for top view (auto).")
            
            # ====== ROW ORDER BY ZONES: TOP ROW FIRST, THEN BOTTOM (LEFT->RIGHT) ======
            # กำหนดเส้นแบ่งเป็นสัดส่วนของความสูงภาพ หรือกำหนดเป็นพิกเซลก็ได้ (ตั้งค่าได้ใน config)
            H = rgb_img.shape[0]
            split_ratio = float(getattr(cfg, "TOP_SPLIT_Y_RATIO", 0.50))   # 0.50 = ครึ่งภาพ
            split_y_px  = int(getattr(cfg, "TOP_SPLIT_Y_PX", split_ratio * H))

            # กันกรณี ROI อยู่ใกล้เส้นแบ่งมาก ๆ ด้วย buffer (พิกเซล)
            buffer_px = int(getattr(cfg, "TOP_ZONE_BUFFER_PX", 0))         # เช่น 10–25 ถ้าต้องการ

            def _center(bbox):
                x, y, w, h = bbox
                return (x + w * 0.5, y + h * 0.5)

            top_zone, bottom_zone = [], []
            for r in rois:
                cx, cy = _center(r["bbox"])
                # ใช้โซนบน/ล่างตามตำแหน่ง Y ของ center
                if cy <= (split_y_px - buffer_px):
                    top_zone.append((cx, r))
                elif cy >= (split_y_px + buffer_px):
                    bottom_zone.append((cx, r))
                else:
                    # ถ้าอยู่ในแถบ buffer: จัดโซนจาก y ด้านบนของ bbox จะเนียนกว่า
                    x, y, w, h = r["bbox"]
                    (top_zone if (y + h*0.5) <= split_y_px else bottom_zone).append((cx, r))

            # เรียงซ้าย->ขวาในแต่ละโซน แล้วรวม: แถวบนก่อน แถวล่างทีหลัง
            top_zone.sort(key=lambda t: t[0])
            bottom_zone.sort(key=lambda t: t[0])
            rois_sorted = [t[1] for t in top_zone] + [t[1] for t in bottom_zone]

            # รีเซ็ต index ให้เรียงใหม่ (ใช้ต่อในการตั้งชื่อ sample: top_1, top_2, ...)
            for i, r in enumerate(rois_sorted, start=1):
                r["idx"] = i

            rois = rois_sorted

            # ภาพรวมทั้งภาพ (density + color) จาก mask_fill เดิม
            add_global_density_and_color(rgb_img, mask_fill)

            union_mask = np.zeros_like(mask_fill, dtype=np.uint8)
            combined_masks = []
            combined_labels = []
            slots_with_obj = 0

            def process_top_auto(r):
                nonlocal slots_with_obj, union_mask, combined_masks, combined_labels
                
                x, y, w, h = r["bbox"]
                sub_img  = rgb_img[y:y+h, x:x+w].copy()
                sub_mask = r["comp_mask"][y:y+h, x:x+w].copy()
                sub_mask = ensure_binary(sub_mask)
                if cv2.countNonZero(sub_mask) == 0:
                    return

                sample = f"top_{r['idx']}"
                # eff_r ชั่วคราวสำหรับฟังก์ชันที่ต้องการ radius
                eff_r = int(max(4, round(0.45 * min(w, h))))

                # วิเคราะห์ระดับ ROI
                with _global_lock:
                    analyze_one_top(sub_mask, sample, eff_r, sub_img)
                    slots_with_obj += 1
                    add_global_density_and_color(sub_img, sub_mask)

                # แยกคอมโพเนนต์ภายใน ROI
                try:
                    _ret = pcv.create_labels(mask=sub_mask)
                except TypeError:
                    _ret = pcv.create_labels(bin_img=sub_mask)
                if isinstance(_ret, tuple):
                    labeled_mask, n_labels = _ret
                else:
                    labeled_mask = _ret
                    n_labels = int(labeled_mask.max()) if labeled_mask is not None else 0

                mode = getattr(cfg, "TOP_SUMMARY_MODE", "per_object")
                if mode == "per_object":
                    saved_any = False
                    # รวมทุกคอมโพเนนต์ใน ROI
                    merged = np.where(labeled_mask > 0, 255, 0).astype(np.uint8)
                    area_px = int(cv2.countNonZero(merged))
                    pcv.outputs.add_observation(
                        sample=f"{sample}",
                        variable="area_px",
                        trait="area", method="countNonZero",
                        scale="px", datatype=int, value=area_px, label="area_px"
                    )
                    mm2 = _area_mm2_from_px(area_px)
                    if mm2 is not None:
                        pcv.outputs.add_observation(
                            sample=f"{sample}",
                            variable="area_mm2",
                            trait="area", method="px_to_mm2",
                            scale="mm2", datatype=float, value=float(mm2), label="area_mm2"
                        )
                    if cv2.countNonZero(merged) >= getattr(cfg, "MIN_PLANT_AREA", 200):
                        save_top_overlay(
                            rgb_img=sub_img,
                            slot_mask=merged,
                            contours=None,
                            eff_r=eff_r,
                            sample_name=f"{sample}_union",
                            mm_per_px=getattr(cfg, "MM_PER_PX", None),
                            slot_label=f"{sample}_union"
                        )
                        full_mask = np.zeros(mask_fill.shape[:2], dtype=np.uint8)
                        full_mask[y:y+h, x:x+w] = merged
                        combined_masks.append(full_mask)
                        combined_labels.append(f"{sample}_union")

                    if not saved_any:
                        # fallback เป็น union ทั้ง ROI
                        save_top_overlay(
                            rgb_img=sub_img,
                            slot_mask=sub_mask,
                            contours=None,
                            eff_r=eff_r,
                            sample_name=f"{sample}_union",
                            mm_per_px=getattr(cfg, "MM_PER_PX", None),
                            slot_label=f"{sample}_union"
                        )
                        full_mask = np.zeros(mask_fill.shape[:2], dtype=np.uint8)
                        full_mask[y:y+h, x:x+w] = sub_mask
                        combined_masks.append(full_mask)
                        combined_labels.append(f"{sample}_union")

                else:  # "per_roi"
                    # log area_px + area_mm2 ต่อ ROI
                    area_px = int(cv2.countNonZero(sub_mask))
                    if area_px < getattr(cfg, "MIN_PLANT_AREA", 2000):
                        return        
                    pcv.outputs.add_observation(
                        sample=f"{sample}",
                        variable="area_px",
                        trait="area", method="countNonZero",
                        scale="px", datatype=int, value=area_px, label="area_px"
                    )
                    mm2 = _area_mm2_from_px(area_px)   # ใช้ helper ฟังก์ชันในไฟล์เดียวกัน
                    if mm2 is not None:
                        pcv.outputs.add_observation(
                            sample=f"{sample}",
                            variable="area_mm2",
                            trait="area", method="px_to_mm2",
                            scale="mm2", datatype=float, value=float(mm2), label="area_mm2"
                        )

                    # เซฟ overlay เป็นทั้ง ROI เหมือนเดิม
                    save_top_overlay(
                        rgb_img=sub_img,
                        slot_mask=sub_mask,
                        contours=None,
                        eff_r=eff_r,
                        sample_name=f"{sample}",
                        mm_per_px=getattr(cfg, "MM_PER_PX", None),
                        slot_label=f"{sample}"
                    )
                    full_mask = np.zeros(mask_fill.shape[:2], dtype=np.uint8)
                    full_mask[y:y+h, x:x+w] = sub_mask
                    combined_masks.append(full_mask)
                    combined_labels.append(f"{sample}")

            for r in rois:
                process_top_auto(r)

            if combined_masks:
                base = getattr(cfg, "OUTPUT_DIR", None) or pcv.params.debug_outdir or "."
                (Path(base) / "processed").mkdir(parents=True, exist_ok=True)
                combine_top_overlays(
                    rgb_img=rgb_img,
                    slot_masks=combined_masks,
                    labels=combined_labels,
                    eff_r=None, 
                    mm_per_px=getattr(cfg, "MM_PER_PX", None),
                    out_path=str(Path(base) / "processed" / "ALL_in_one_overlay.png"),
                )

            extra = {
                "filename": filename,
                "view": "top",
                "roi_mode": "auto",
                "n_top_rois": int(len(rois)),
            }
            return extra, ensure_binary(union_mask)

        else:
            # ---- โหมด grid (ของเดิม) ----
            rois, eff_r = make_grid_rois(
                rgb_img, cfg.ROWS, cfg.COLS, getattr(cfg, "ROI_RADIUS", None)
            )
        try:
            rois, eff_r = make_grid_rois(
                rgb_img, cfg.ROWS, cfg.COLS, getattr(cfg, "ROI_RADIUS", None)
            )
        except Exception as e:
            raise RuntimeError(f"make_grid_rois failed: {e}")
        _dbg("DEBUG rois:", len(rois), "eff_r:", eff_r)

        # (optional) เซฟ overlay
        overlay = rgb_img.copy()
        try:
            fname = Path(cfg.INPUT_PATH).stem
        except Exception:
            fname = "unknown"
        cv2.putText(overlay, f"File: {fname}", (12, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
        for cnt in rois:
            cv2.drawContours(overlay, [cnt], -1, (0, 255, 0), 2)
        base = getattr(cfg, "OUTPUT_DIR", None) or pcv.params.debug_outdir or "."
        (Path(base) / "processed").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(Path(base) / "processed" / f"{Path(filename).stem}_rois.png"), overlay)

        # ภาพรวมทั้งภาพ (density + color)
        add_global_density_and_color(rgb_img, mask_fill)

        union_mask = np.zeros_like(mask_fill, dtype=np.uint8)
        slots_with_obj = 0
        combined_masks = []
        combined_labels = []
        
        def process_roi(i, roi_cnt):
            nonlocal slots_with_obj, union_mask, combined_masks, combined_labels
            
            M = cv2.moments(roi_cnt)
            if M["m00"] == 0:
                return None
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
                return None

            # create_labels: รองรับทั้ง tuple และ single return
            try:
                result = pcv.create_labels(mask=filtered_mask, rois=None)
                if isinstance(result, tuple) and len(result) == 2:
                    labeled_mask, n_labels = result
                else:
                    labeled_mask = result
                    n_labels = int(labeled_mask.max()) if labeled_mask is not None else 0
            except Exception as e:
                _dbg("WARN: create_labels failed:", e)
                return None

            if int(n_labels) <= 0:
                return None

            r = i // cfg.COLS + 1
            c = i %  cfg.COLS + 1
            slot_union = np.zeros_like(filtered_mask, dtype=np.uint8)
            per_slot_count = 0
            per_slot_area_sum = 0

            if getattr(cfg, "MERGE_COMPONENTS_PER_SLOT", False):
                # --- โหมดรวมก้อนทั้งหมดใน ROI ---
                merged = np.where(labeled_mask > 0, 255, 0).astype(np.uint8)
                if cv2.countNonZero(merged) < getattr(cfg, "MIN_PLANT_AREA", 200):
                    return None

                area_px = int(cv2.countNonZero(merged))
                per_slot_area_sum = area_px

                with _global_lock:
                    pcv.outputs.add_observation(
                    sample=f"slot_{r}_{c}", variable="area_px",
                    trait="area", method="countNonZero(union)", scale="px",
                    datatype=int, value=area_px, label="area_px"
                )
            
                mm2 = _area_mm2_from_px(area_px)
                if mm2 is not None:
                    pcv.outputs.add_observation(
                        sample=f"slot_{r}_{c}", variable="area_mm2",
                        trait="area", method="px_to_mm2", scale="mm2",
                        datatype=float, value=float(mm2), label="area_mm2"
                    )
                    
                per_slot_count = 1
                with _global_lock:
                    slots_with_obj += 1

                # ใช้ analyze_one_top บน union ทั้งหมด
                union_mask = ensure_binary(cv2.bitwise_or(union_mask, merged))
                slot_union = ensure_binary(cv2.bitwise_or(slot_union, merged))

                if cv2.countNonZero(slot_union) > 0:
                    combined_masks.append(slot_union.copy())
                    combined_labels.append(f"slot_{r}_{c}")

            else:
                # --- โหมดเดิม: วนราย obj ---
                per_slot_count = 0
                for j in range(1, int(n_labels) + 1):
                    single = np.where(labeled_mask == j, 255, 0).astype(np.uint8)
                    if cv2.countNonZero(single) < getattr(cfg, "MIN_PLANT_AREA", 200):
                        continue

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
                        
                    with _global_lock:
                        analyze_one_top(single, f"slot_{r}_{c}_obj{j}", eff_r, rgb_img)
                        slots_with_obj += 1
                        per_slot_count += 1
                    
                    slot_union = ensure_binary(cv2.bitwise_or(slot_union, single))
                    union_mask = ensure_binary(cv2.bitwise_or(union_mask, single))

            if getattr(cfg, "SAVE_TOP_OVERLAY", True) and cv2.countNonZero(slot_union) > 0:
                save_top_overlay(
                    rgb_img=rgb_img,
                    slot_mask=slot_union,
                    contours=None,
                    eff_r=eff_r,
                    sample_name=f"slot_{r}_{c}",
                    mm_per_px=getattr(cfg, "MM_PER_PX", None),
                    slot_label=f"slot_{r}_{c}"   # ให้ขึ้นชื่อ slot ในแถบสรุป
                )
            with _global_lock:   
                # บันทึกจำนวน/พื้นที่รวมต่อ slot
                pcv.outputs.add_observation(
                    sample=f"slot_{r}_{c}", variable="n_plants_in_slot",
                    trait="count", method="roi_filter", scale="none",
                    datatype=int, value=int(per_slot_count), label="n_plants"
                )
                pcv.outputs.add_observation(
                    sample=f"slot_{r}_{c}", variable="slot_area_sum_px",
                    trait="area", method="sum(label_area)" if not getattr(cfg, "MERGE_COMPONENTS_PER_SLOT", False) else "countNonZero(union)",
                    scale="px", datatype=int, value=int(per_slot_area_sum), label="slot_area_sum_px"
                )
                mm2_slot = _area_mm2_from_px(per_slot_area_sum)
                if mm2_slot is not None:
                    pcv.outputs.add_observation(
                        sample=f"slot_{r}_{c}", variable="slot_area_sum_mm2",
                        trait="area", method="px_to_mm2", scale="mm2",
                        datatype=float, value=float(mm2_slot), label="slot_area_sum_mm2"
                    )

        for i, cnt in enumerate(rois):
            process_roi(i, cnt)
        if slots_with_obj == 0:
            raise RuntimeError("No objects inside any ROI cell.")
        
        if combined_masks:
            base = getattr(cfg, "OUTPUT_DIR", None) or pcv.params.debug_outdir or "."
            (Path(base) / "processed").mkdir(parents=True, exist_ok=True)
            combine_top_overlays(
                rgb_img=rgb_img,
                slot_masks=combined_masks,
                labels=combined_labels,
                eff_r=eff_r,
                mm_per_px=getattr(cfg, "MM_PER_PX", None),
                out_path=str(Path(base) / "processed" / "ALL_in_one_overlay.png"),
            ) 
        extra = {
            "filename": filename,
            "view": "top",
            "roi_grid": f"{cfg.ROWS}x{cfg.COLS}",
            "roi_radius": int(eff_r),
            "roi_type": getattr(cfg, "ROI_TYPE", "partial"),
            "n_slots_with_objects": int(slots_with_obj),
        }
        return extra, union_mask
    
    else:  # side
        #crop
        crop_img = pcv.crop(img=rgb_img, x=150, y=0, h=1800, w=3600)
        crop_mask = pcv.crop(img=mask_fill, x=150, y=0, h=1800, w=3600)
        
        crop_img_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        # === STEM RESCUE (SIDE) ด้วย V-channel ก่อนแยก ROI ===
        if getattr(cfg, "ENABLE_SIDE_STEM_RESCUE", True):
            crop_mask_before = ensure_binary(crop_mask)
            crop_mask, dbg_side = add_v_connected_to_a(
                rgb=crop_img_rgb,
                base_a_mask=crop_mask_before,
                method=getattr(cfg, "SIDE_V_METHOD", "fixed"),   # "fixed" | "otsu" | "percentile"
                v_min=getattr(cfg, "SIDE_V_MIN", 150),
                v_max=getattr(cfg, "SIDE_V_MAX", 255),
                percentile=getattr(cfg, "SIDE_V_PERCENTILE", 85),
                s_max=getattr(cfg, "SIDE_S_MAX", None),
                glare_v=getattr(cfg, "SIDE_GLARE_V", 255),
                glare_s=getattr(cfg, "SIDE_GLARE_S", 0),
                near_px=getattr(cfg, "SIDE_NEAR_PX", 30),
                geo_iters=getattr(cfg, "SIDE_GEO_ITERS", 120),
                open_k=getattr(cfg, "SIDE_OPEN_K", 3),
                min_area_keep=getattr(cfg, "SIDE_MIN_AREA_KEEP", 250),
                connect_mode=getattr(cfg, "STEM_CONNECT_MODE", "geo"),
                cc_close_k=getattr(cfg, "STEM_CC_CLOSE_K", 0),
            )
            crop_mask = ensure_binary(crop_mask)

            # (optional) เซฟ debug
            base = getattr(cfg, "OUTPUT_DIR", None) or pcv.params.debug_outdir or "."
            (Path(base) / "processed").mkdir(parents=True, exist_ok=True)
            stub = Path(filename).stem
            cv2.imwrite(str(Path(base) / "processed" / f"{stub}_sideV_before.png"), crop_mask_before)
            cv2.imwrite(str(Path(base) / "processed" / f"{stub}_sideV_after.png"),  crop_mask)
            if "vmask" in dbg_side:
                cv2.imwrite(str(Path(base) / "processed" / f"{stub}_sideV_vmask.png"), dbg_side["vmask"])
            if "v_connected" in dbg_side:
                cv2.imwrite(str(Path(base) / "processed" / f"{stub}_sideV_vconnected.png"), dbg_side["v_connected"])
        
        # หา ROI อัตโนมัติหลายต้น + ดีบัคภาพรวม
        rois = make_side_rois_auto(
            rgb_img=crop_img,
            mask_fill=crop_mask,
            min_area_px=getattr(cfg, "MIN_PLANT_AREA", 800),
            merge_gap_px=getattr(cfg, "SIDE_MERGE_GAP", 200),
            debug_out_path=str((Path(getattr(cfg, "OUTPUT_DIR", None) or pcv.params.debug_outdir or ".") 
                               / "processed" / f"{Path(filename).stem}_side_rois.png"))
        )
        if not rois:
            slot_mask, (x, y, w, h) = make_side_roi(
                crop_img, crop_mask, cfg.USE_FULL_IMAGE_ROI,
                cfg.ROI_X, cfg.ROI_Y, cfg.ROI_W, cfg.ROI_H
            )
            slot_mask = ensure_binary(slot_mask)
            if cv2.countNonZero(slot_mask) > 0:
                rois = [{"idx": 1, "bbox": (x, y, w, h), "comp_mask": slot_mask}]

        if not rois:
            raise RuntimeError("No objects detected for side view.")

        union_mask = np.zeros_like(crop_mask, dtype=np.uint8)
        union_masks = []
        union_labels = []
        
        def process_side_roi(r):
            x, y, w, h = r["bbox"]
            sub_img  = crop_img[y:y+h, x:x+w].copy()
            sub_mask = r["comp_mask"][y:y+h, x:x+w].copy()
            if sub_mask.ndim == 3:
                sub_mask = cv2.cvtColor(sub_mask, cv2.COLOR_BGR2GRAY)
            sub_mask = ensure_binary(sub_mask)
            if cv2.countNonZero(sub_mask) == 0:
                return

            # ชื่อ sample ต่อก้อน
            sample = f"side_{r['idx']}"
            
            with _global_lock:
                analyze_one_side(sub_mask, sample, sub_img)  # วิเคราะห์ต่อก้อน
                add_global_density_and_color(sub_img, sub_mask)  # metric สี/ความหนาแน่น

            # เซฟ overlay ต่อ object
            try:
                _ret = pcv.create_labels(mask=sub_mask)
            except TypeError:
                _ret = pcv.create_labels(bin_img=sub_mask)  # รองรับ PlantCV รุ่นเก่า
            if isinstance(_ret, tuple):
                labeled_mask, n_labels = _ret
            else:
                labeled_mask = _ret
                n_labels = int(labeled_mask.max()) if labeled_mask is not None else 0
                
            #หลายobject
            saved_any = False
            for j in range(1, int(n_labels) + 1):
                single = np.where(labeled_mask == j, 255, 0).astype(np.uint8)
                if cv2.countNonZero(single) < getattr(cfg, "MIN_PLANT_AREA", 200):
                    continue

                save_side_overlay(
                    rgb_img=sub_img,
                    slot_mask=single,
                    sample_name=f"{sample}_obj{j}",
                    mm_per_px=getattr(cfg, "MM_PER_PX", None)
                )
                saved_any = True
                
                full_mask = np.zeros(crop_img.shape[:2], dtype=np.uint8)
                full_mask[y:y+h, x:x+w] = single
                union_masks.append(full_mask)
                union_labels.append(f"{sample}_obj{j}")
                
            #แยกไม่ออกเลย
            if not saved_any:
                save_side_overlay(
                    rgb_img=sub_img,
                    slot_mask=sub_mask,      
                    sample_name=f"{sample}_union",
                    mm_per_px=getattr(cfg, "MM_PER_PX", None)
                )
                
                full_mask = np.zeros(crop_img.shape[:2], dtype=np.uint8)
                full_mask[y:y+h, x:x+w] = sub_mask
                union_masks.append(full_mask)
                union_labels.append(f"{sample}_union")
            # รวม union mask (สำหรับอ้างอิงรวม)
            union_mask[y:y+h, x:x+w] = cv2.bitwise_or(union_mask[y:y+h, x:x+w], sub_mask)

        for r in rois:
            process_side_roi(r)
        if not union_masks:
            raise RuntimeError("Processed side ROIs but no valid masks produced.")
        
        base = getattr(cfg, "OUTPUT_DIR", None) or pcv.params.debug_outdir or "."
        (Path(base) / "processed").mkdir(parents=True, exist_ok=True)
        combine_side_overlays(
            rgb_img=crop_img,
            masks=union_masks,
            labels=union_labels,
            mm_per_px=getattr(cfg, "MM_PER_PX", None),
            out_path=str(Path(base) / "processed" / f"{Path(filename).stem}_ALL_side_overlay.png"),
        )
        extra = {
            "filename": filename,
            "view": "side",
            "n_side_plants": int(len(rois)),
        }
        return extra, ensure_binary(union_mask)
    
def process_one(path: Path, out_dir: Path):
    pcv.params.debug = cfg.DEBUG_MODE
    pcv.params.debug_outdir = str(out_dir / "debug")
    pcv.params.dpi = 150
    
    with _global_lock:
        pcv.outputs.clear()

    img, filename = safe_readimage(path)
    extra, mask = run_one_image(img, filename)

    # Flatten PlantCV observations → dict
    with _global_lock:
        results_dict = pcv.outputs.observations.copy()
        flat = {}
    for sample, vars_ in results_dict.items():
        for var_name, record in vars_.items():
            col = f"{sample}_{var_name}" if sample != 'default' else var_name
            flat[col] = record.get('value', None)
    flat.update(extra)
    
    area_values = [v for k, v in flat.items() 
                   if isinstance(v, (int, float)) and "area_sum_mm2" in k and not str(k).startswith("default")]

    if area_values:
        avg_area = float(np.mean(area_values))
        flat["avg_area_mm2_per_image"] = avg_area
    else:
        flat["avg_area_mm2_per_image"] = None

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

    # ส่งข้อมูลขึ้น ThingsBoard
    try:
        publish_data(out_json)
        print(f"Published data from {out_json} to ThingsBoard.")
    except Exception as e:
        print(f"Failed to publish data from {out_json} to ThingsBoard: {e}")
    
    return str(out_json)

# ฟังก์ชันสำหรับประมวลผลหลายไฟล์ด้วย multiprocessing
def process_multiple(paths, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # แปลง paths เป็นรายการไฟล์ที่ต้องประมวลผล
    image_paths = []
    if isinstance(paths, (str, Path)):
        path = Path(paths)
        if path.is_dir():
            # รวบรวมไฟล์ภาพทั้งหมดในไดเร็กทอรี
            for ext in ['jpg', 'jpeg', 'png', 'tif', 'tiff']:
                image_paths.extend(list(path.glob(f"*.{ext}")))
                image_paths.extend(list(path.glob(f"*.{ext.upper()}")))
        else:
            image_paths = [path]
    else:
        # paths เป็น list อยู่แล้ว
        image_paths = [Path(p) for p in paths]
    
    # ตรวจสอบว่ามีไฟล์ให้ประมวลผลหรือไม่
    if not image_paths:
        print("No images found to process.")
        return []
    
    # ใช้ multiprocessing สำหรับการประมวลผลไฟล์
    num_processes = min(multiprocessing.cpu_count(), 16)
    print(f"Processing {len(image_paths)} images with {num_processes} processes...")
    
    # ใช้ ProcessPoolExecutor เพื่อประมวลผลแต่ละชุด
    start_time = time.time()
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as ex:
        futs = [ex.submit(process_one, p, out_dir) for p in image_paths]
        for i, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            results.append(fut.result())
            print(f"Completed {i}/{len(image_paths)}")

    print(f"All processing completed in {time.time() - start_time:.2f} seconds")
    return results