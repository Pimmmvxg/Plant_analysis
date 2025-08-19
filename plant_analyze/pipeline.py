import json
from pathlib import Path
import cv2
import numpy as np
from plantcv import plantcv as pcv
from . import config as cfg
from .io_utils import safe_readimage
from .masking import auto_select_mask, clean_mask, ensure_binary
from .roi_top import make_grid_rois
from .roi_side import make_side_roi
from .analyze_top import analyze_one_top, add_global_density_and_color
from .analyze_side import analyze_one_side

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
    print("DEBUG side mask_closed:",mask_closed.dtype, np.unique(mask_closed)[:5])
    mask_fill = pcv.fill(bin_img=mask_closed, size=30)
    mask_fill = clean_mask(mask_fill, close_ksize=5, min_obj_size=30)   
    mask_fill = ensure_binary(mask_fill) 
    print("DEBUG side mask_fill:",mask_fill.dtype, np.unique(mask_fill)[:5])
    plant_size = int(cv2.countNonZero(mask_fill))
    per_slot_area = 0  # ไว้ก่อนลูป labels

    for j in range(1, n_labels + 1):
        single = np.where(labeled_mask == j, 255, 0).astype(np.uint8)
        if cv2.countNonZero(single) < getattr(cfg, "MIN_PLANT_AREA", 200):
            continue

        # ขนาดต่อ plant (px)
        area_px = int(cv2.countNonZero(single))
        per_slot_area += area_px

        # เก็บเป็น observation ระดับ plant
        pcv.outputs.add_observation(
            sample=sample_name,                # f"slot_{r}_{c}_obj{j}"
            variable="area_px",
            trait="area",
            method="countNonZero",
            scale="px",
            datatype=int,
            value=area_px,
            label="area_px"
        )

        # (ของเดิม) union + analyze_one_top
        union_mask = cv2.bitwise_or(union_mask, single)
        union_mask = ensure_binary(union_mask)
        analyze_one_top(single, sample_name, eff_r, rgb_img)
        per_slot_count += 1
        slots_with_obj += 1

    # หลังจบลูป label ของ slot นี้ → บันทึกพื้นที่รวมต่อช่อง
    pcv.outputs.add_observation(
        sample=f"slot_{r}_{c}",
        variable="slot_area_sum_px",
        trait="area",
        method="sum(label_area)",
        scale="px",
        datatype=int,
        value=int(per_slot_area),
        label="slot_area_sum_px",
    )


    if cfg.VIEW == "top":
        print("DEBUG entering TOP pipeline")
        try:
            rois, eff_r = make_grid_rois(
                rgb_img, cfg.ROWS, cfg.COLS, getattr(cfg, "ROI_RADIUS", None)
            )
        except Exception as e:
            raise RuntimeError(f"make_grid_rois failed: {e}")
        print("DEBUG rois:", len(rois), "eff_r:", eff_r)
        
        overlay = rgb_img.copy()
        for cnt in rois:
            cv2.drawContours(overlay, [cnt], -1, (0, 255, 0), 2)

        base = getattr(cfg, "OUTPUT_DIR", None) or pcv.params.debug_outdir or "."
        save_dir = Path(base) / "processed"
        save_dir.mkdir(parents=True, exist_ok=True)

        out_path = save_dir / f"{Path(filename).stem}_rois.png"
        cv2.imwrite(str(out_path), overlay)
        print("Saved ROI overlay to", out_path)

        # วิเคราะห์ภาพรวมทั้งภาพ (ความหนาแน่น + สี)
        add_global_density_and_color(rgb_img, mask_fill)

        # เตรียมตัวแปรก่อนลูป
        union_mask = np.zeros_like(mask_fill, dtype=np.uint8)
        slots_with_obj = 0

        for i, roi_cnt in enumerate(rois):
            # หาศูนย์กลาง ROI จาก contour (เพื่อสร้าง ROI แบบวงกลมของ PlantCV)
            M = cv2.moments(roi_cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # 1) สร้าง ROI (Objects dataclass) ด้วย PlantCV
            roi = pcv.roi.circle(img=rgb_img, x=cx, y=cy, r=int(eff_r))

            # 2) คัดมาสก์ด้วย ROI แบบ "partial" (ทับบางส่วนก็เอาทั้งก้อน) — v4 แทนที่ roi_objects
            #    คืนค่าเป็น binary mask ของวัตถุที่ผ่านเกณฑ์ใน ROI นั้น
            filtered_mask = pcv.roi.filter(mask=mask_fill, roi=roi, roi_type="partial")  # v4
            filtered_mask = ensure_binary(filtered_mask)

            if cv2.countNonZero(filtered_mask) == 0:
                continue

            # 3) แยกเป็นรายออบเจ็กต์ด้วยฉลาก (label) — v4 วิธีใหม่สำหรับ multi-object
            labeled_mask, n_labels = pcv.create_labels(mask=filtered_mask, rois=None)    # v4
            try:
                result = pcv.create_labels(mask=filtered_mask, rois=None)
                if isinstance(result, tuple):
                    labeled_mask, n_labels = result
                else:
                    labeled_mask, n_labels = result, int(labeled_mask.max())
            except Exception as e:
                print("create_labels failed:", e)
                continue
            if n_labels <= 0:
                continue

            r = i // cfg.COLS + 1
            c = i % cfg.COLS + 1
            per_slot_count = 0

            # 4) วนวิเคราะห์รายต้น (label เริ่มที่ 1)
            for j in range(1, n_labels + 1):
                single = np.where(labeled_mask == j, 255, 0).astype(np.uint8)
                if cv2.countNonZero(single) < getattr(cfg, "MIN_PLANT_AREA", 200):
                    continue

                union_mask = cv2.bitwise_or(union_mask, single)
                union_mask = ensure_binary(union_mask)

                sample_name = f"slot_{r}_{c}_obj{j}"
                analyze_one_top(single, sample_name, eff_r, rgb_img)
                per_slot_count += 1
                slots_with_obj += 1

            pcv.outputs.add_observation(
                sample=f"slot_{r}_{c}",
                variable="n_plants_in_slot",
                trait="count",
                method="roi_partial_v4",
                scale="none",
                datatype=int,
                value=int(per_slot_count),
                label="count",
            )

        # หลังจบลูป
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