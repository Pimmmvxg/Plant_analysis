# multi_topview.py  — Top-view grid ROI + robust returns (PlantCV v4+)
import os, sys, json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
import pandas as pd
from plantcv import plantcv as pcv
import matplotlib.pyplot as plt

# ========== CONFIG ==========
INPUT_PATH   = r"C:\Cantonese\topview_test.jpg"   # ไฟล์เดียว หรือโฟลเดอร์
OUTPUT_DIR   = r".\results_topview"
EXTENSIONS   = ['.png', '.jpg', '.jpeg']
THREADS      = 1
DEBUG_MODE   = 'print'        # 'none'|'print'|'plot'  
SAVE_MASK    = True

# กริด ROI (จูนให้ตรงภาพคุณ)
GRID_X, GRID_Y = 400, 200    # กึ่งกลางช่องซ้ายบน (พิกเซล)
ROI_RADIUS     = 200          # รัศมีวงกลม ROI
DX, DY         = 500, 600    # ระยะกึ่งกลาง→กึ่งกลาง ช่องถัดไป (พิกเซล)
ROWS, COLS     = 2, 3        # จำนวนแถว/คอลัมน์
ROI_TYPE       = 'partial'   # 'partial' | 'cutto' | 'largest'
# ============================


# ---------- Utilities ----------
def safe_readimage(path: Path):
    """รองรับต่างเวอร์ชันของ pcv.readimage (2 หรือ 3 ค่าที่คืนมา)"""
    ri = pcv.readimage(filename=str(path))
    img, path_str, filename = None, None, None
    if isinstance(ri, tuple):
        if len(ri) == 3:
            img, path_str, filename = ri
        elif len(ri) == 2:
            img, path_str = ri
            filename = Path(path_str).name if path_str else path.name
        elif len(ri) == 1:
            img = ri[0]
            filename = path.name
        else:
            img = ri[0]; filename = path.name
    else:
        img = ri; filename = path.name
    return img, filename


def auto_select_mask(rgb_img):
    H, W = rgb_img.shape[:2]
    area_total = H * W

    chans = []
    try:  chans.append(("lab_a", pcv.rgb2gray_lab(rgb_img=rgb_img, channel='a')))
    except: pass
    try:  chans.append(("lab_b", pcv.rgb2gray_lab(rgb_img=rgb_img, channel='b')))
    except: pass
    try:  chans.append(("lab_l", pcv.rgb2gray_lab(rgb_img=rgb_img, channel='l')))
    except: pass
    try:  chans.append(("hsv_v", pcv.rgb2gray_hsv(rgb_img=rgb_img, channel='v')))
    except: pass
    try:  chans.append(("hsv_s", pcv.rgb2gray_hsv(rgb_img=rgb_img, channel='s')))
    except: pass
    if not chans:
        raise RuntimeError("No grayscale channels available for auto selection.")

    methods = []
    for name, g in chans:
        try:
            g = pcv.transform.rescale(gray_img=g, lower=0, upper=255)
        except Exception:
            pass
        methods += [
            (name, "otsu",     {"object_type": "dark"}),
            (name, "otsu",     {"object_type": "light"}),
            (name, "triangle", {"object_type": "dark"}),
            (name, "triangle", {"object_type": "light"}),
        ]
        for k in (31, 51, 101):
            methods += [(name, ("gaussian", k, 0, "dark"),  {}),
                        (name, ("gaussian", k, 0, "light"), {})]

    candidates = []
    for name, meth, params in methods:
        try:
            gray = next(g for nm, g in chans if nm == name)
            if meth == "otsu":
                m = pcv.threshold.otsu(gray_img=gray, **params)
            elif meth == "triangle":
                m = pcv.threshold.triangle(gray_img=gray, **params)
            else:
                _, k, off, obj = meth
                m = pcv.threshold.gaussian(gray_img=gray, ksize=k, offset=off, object_type=obj)

            # findContours แบบ robust (OpenCV3/4)
            _fc = cv2.findContours(m.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(_fc) == 3:
                _, contours, _hier = _fc
            else:
                contours, _hier = _fc

            n_comp = len(contours)
            if n_comp > 0:
                areas = [cv2.contourArea(c) for c in contours]
                cnt = contours[int(np.argmax(areas))]
                hull = cv2.convexHull(cnt)
                a_obj  = float(cv2.contourArea(cnt))
                a_hull = float(cv2.contourArea(hull))
                solidity = (a_obj / a_hull) if a_hull > 0 else 0.0
            else:
                solidity = 0.0

            area = int(np.count_nonzero(m))
            ratio = area / max(area_total, 1)

            def score_area(r):
                if 0.5 <= r <= 0.6:
                    return 2.0 - abs(0.30 - r)
                return 0.5 - abs(0.30 - r)
            s = score_area(ratio) - 0.1 * max(0, n_comp - 1) + 0.5 * solidity

            candidates.append((s, name, m, ratio, n_comp, solidity))
        except Exception:
            continue

    if not candidates:
        fallback = pcv.threshold.otsu(gray_img=pcv.rgb2gray_lab(rgb_img=rgb_img, channel='a'), object_type='dark')
        return fallback, {"channel":"lab_a","method":"otsu","object_type":"dark","ksize":None,
                          "area_ratio": float(np.count_nonzero(fallback))/max(area_total,1),
                          "n_components": 0, "solidity": 0.0}

    best = max(candidates, key=lambda x: x[0])
    _, name, mask, ratio, n_comp, solidity = best
    return mask, {"channel": name, "method": "auto", "object_type": "auto",
                  "ksize": None, "area_ratio": float(ratio),
                  "n_components": int(n_comp), "solidity": float(solidity)}

def _largest_component(bin_img):
    if bin_img is None or cv2.countNonZero(bin_img) == 0:
        return bin_img
    _fc = cv2.findContours(bin_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(_fc) == 3:
        _, cnts, _ = _fc
    else:
        cnts, _ = _fc
    if not cnts:
        return bin_img
    idx = int(np.argmax([cv2.contourArea(c) for c in cnts]))
    out = np.zeros_like(bin_img)
    cv2.drawContours(out, [cnts[idx]], -1, 255, thickness=cv2.FILLED)
    return out

def clean_mask(m, close_ksize=7, min_obj_size=120):
    if m is None:
        return m
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)
    m = _largest_component(m)
    m =pcv.fill(bin_img=m, size=min_obj_size)
    return m

def segment(rgb_img, filename):
    # 1) ทำ mask อัตโนมัติ
    mask0, info = auto_select_mask(rgb_img)
    if cv2.countNonZero(mask0) > mask0.size / 2:
        # ถ้า mask ใหญ่เกินไป ให้ใช้ largest component
        mask0 = pcv.invert(gray_img=mask0)
        
    white_ratio = cv2.countNonZero(mask0) / mask0.size
    if white_ratio > 0.5:
        mask0 = pcv.invert(gray_img=mask0)

    # เก็บค่าที่ auto เลือกไว้
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
    mask_fill = pcv.fill(bin_img=mask_closed, size=30)

    # 3) ROI grid (safe radius)
    H, W = rgb_img.shape[:2]
    eff_r = min(
        ROI_RADIUS,
        max(DX // 2 - 1, 1),
        max(DY // 2 - 1, 1)
    )
    x0, y0 = GRID_X, GRID_Y
    xN = GRID_X + (COLS - 1) * DX
    yN = GRID_Y + (ROWS - 1) * DY
    eff_r = min(eff_r, x0, y0, (W - 1) - xN, (H - 1) - yN)
    if eff_r <= 0:
        raise ValueError(
            "ROI grid does not fit the image. Adjust GRID_X/Y, DX/DY, ROWS/COLS, or ROI_RADIUS. "
            f"Image={W}x{H}, grid ({x0},{y0})->({xN},{yN})"
        )

    # v4+: pcv.roi.multi → ROI-collection (multi-ROI object)
    roi_obj = pcv.roi.multi(
        img=rgb_img,
        coord=(x0, y0),
        radius=int(eff_r),
        spacing=(DX, DY),
        nrows=ROWS,
        ncols=COLS
    )
    # ทำให้ iterable เสมอ (กันเวอร์ชันต่างกัน)
    try:
        rois = list(roi_obj)
    except TypeError:
        rois = getattr(roi_obj, 'roi', None) or getattr(roi_obj, 'rois', None) or [roi_obj]

    pcv.outputs.add_observation(
        sample='default', variable='roi_effective_radius',
        trait='length', method='precheck', scale='px',
        datatype=int, value=int(eff_r), label='effective ROI radius'
    )

    # 4) วนทีละ ROI → filter mask → วิเคราะห์รายช่อง
    union_mask = np.zeros_like(mask_fill, dtype=np.uint8)
    slots_with_obj = 0

    def analyze_one(slot_mask, sample_name, eff_r):
        """วิเคราะห์ skeleton/segments สำหรับ mask ของช่องเดียว"""
        # Skeletonize
        base_skel = pcv.morphology.skeletonize(mask=slot_mask)

        # 2) เตรียมชุดค่า prune ที่สัมพันธ์กับสเกลของ ROI (eff_r)
        #    เริ่มเล็ก → กลาง → ใหญ่ (กำจัดเสี้ยนเพิ่มขึ้นเรื่อยๆ)
        sizes = sorted(set(
            [max(3, int(eff_r * t)) for t in (0.15, 0.2, 0.25, 0.3, 0.35, 0.4)]
            + [50, 75, 100, 125, 150, 175, 200, 225, 250, 300, 350, 400, 500, 600]
        ))

        last_err = None
        pruned_skel = None
        edge_objects = None
        leaf_obj = stem_obj = None
        segmented_img = None

        for sz in sizes:
            try:
                # ใช้ skeleton ตั้งต้นทุกครั้ง (กันการ prune ทับซ้อนเกินไป)
                ret = pcv.morphology.prune(skel_img=base_skel, size=sz, mask=slot_mask)
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

                # จัดใบ/ก้าน
                lo = pcv.morphology.segment_sort(skel_img=pruned_skel, objects=edge_objects, mask=slot_mask)
                if isinstance(lo, tuple):
                    leaf_obj = lo[0]
                    stem_obj = lo[1] if len(lo) > 1 else None
                else:
                    leaf_obj, stem_obj = lo, None

                # ทำ ID ของ segment (จุดนี้ถ้า tips เยอะเกิน จะ throw)
                sid = pcv.morphology.segment_id(skel_img=pruned_skel, objects=leaf_obj, mask=slot_mask)
                segmented_img = sid[0] if isinstance(sid, tuple) else sid

                # ถ้าเดินมาถึงนี้ แปลว่าผ่าน → หยุดไล่ขนาด
                break

            except Exception as e:
                last_err = e
                # ถ้าสาเหตุคือ tips เยอะ ให้ลอง prune ใหญ่ขึ้น
                msg = str(e).lower()
                if "too many tips" in msg or "try pruning again" in msg:
                    continue
                # ถ้าไม่ใช่สาเหตุนี้ โยน error เดิม
                raise

        else:
            # ไล่จนหมดแล้วยังไม่ผ่าน → แจ้งสาเหตุล่าสุด
            raise last_err if last_err else RuntimeError("Failed to analyze slot: pruning did not converge.")

        # วิเคราะห์ย่อย
        _ = pcv.morphology.fill_segments(mask=slot_mask, objects=leaf_obj, label=sample_name)
        branch_pts_mask = pcv.morphology.find_branch_pts(skel_img=pruned_skel, mask=slot_mask, label=sample_name)
        _ = pcv.morphology.segment_euclidean_length(segmented_img=segmented_img, objects=leaf_obj, label=sample_name)
        _ = pcv.analyze.size(img=rgb_img, labeled_mask=slot_mask, label=sample_name)

        # ตัวเลขหลัก
        num_leaf_segments = int(len(leaf_obj)) if leaf_obj is not None else 0
        num_stem_segments = int(len(stem_obj)) if stem_obj is not None else 0
        num_branch_points = int(np.count_nonzero(branch_pts_mask)) if branch_pts_mask is not None else 0
        area_px = int(cv2.countNonZero(slot_mask))

        # บันทึกเป็น sample เฉพาะช่อง
        pcv.outputs.add_observation(sample=sample_name, variable='num_leaf_segments',
                                    trait='count', method='pcv.morphology.segment_sort',
                                    scale='count', datatype=int, value=num_leaf_segments, label='leaf segments')
        pcv.outputs.add_observation(sample=sample_name, variable='num_stem_segments',
                                    trait='count', method='pcv.morphology.segment_sort',
                                    scale='count', datatype=int, value=num_stem_segments, label='stem segments')
        pcv.outputs.add_observation(sample=sample_name, variable='num_branch_points',
                                    trait='count', method='pcv.morphology.find_branch_pts',
                                    scale='count', datatype=int, value=num_branch_points, label='branch points')
        pcv.outputs.add_observation(sample=sample_name, variable='slot_area_px',
                                    trait='area', method='count_nonzero', scale='px',
                                    datatype=int, value=area_px, label='slot mask area')

    for i, roi in enumerate(rois):
        # ฟิลเตอร์หน้ากากให้เหลือเฉพาะช่อง i
        slot_mask = pcv.roi.filter(mask=mask_fill, roi=roi, roi_type=ROI_TYPE)
        if slot_mask is None or cv2.countNonZero(slot_mask) == 0:
            continue

        # รวมหน้ากากของทุกช่อง (เผื่ออยากเซฟภาพเดียว)
        union_mask = cv2.bitwise_or(union_mask, slot_mask)

        # ชื่อ sample ต่อช่อง (เริ่มที่ 1)
        r = i // COLS + 1
        c = i %    COLS + 1
        sample_name = f"slot_{r}_{c}"

        analyze_one(slot_mask, sample_name, eff_r)
        slots_with_obj += 1

    if slots_with_obj == 0:
        raise RuntimeError("No objects inside any ROI cell.")

    # คืนค่า (รวมทั้งภาพ)
    return {
        "filename": filename,
        "roi_grid": f"{ROWS}x{COLS}",
        "roi_coord": f"({GRID_X},{GRID_Y})",
        "roi_radius": int(eff_r),
        "roi_spacing": f"({DX},{DY})",
        "roi_type": ROI_TYPE,
        "n_slots_with_objects": int(slots_with_obj)
    }, union_mask


def process_one(path: Path, out_dir: Path, debug_mode: str, save_mask: bool):
    pcv.params.debug = debug_mode
    pcv.params.debug_outdir = str(out_dir / "debug")
    pcv.params.dpi = 150
    pcv.outputs.clear()

    img, filename = safe_readimage(path)
    extra, mask = segment(img, filename)

    # รวม outputs
    results_dict = pcv.outputs.observations.copy()
    flat = {}
    for sample, vars_ in results_dict.items():
        for var_name, record in vars_.items():
           col = f"{sample}_{var_name}" if sample != 'default' else var_name
           flat[col] = record.get('value', None)
    flat.update(extra)

    # JSON
    json_dir = out_dir / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    out_json = json_dir / f"{Path(filename).stem}.json"
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(flat, f, ensure_ascii=False, indent=2)

    # หน้ากากตัวอย่าง
    if save_mask and mask is not None:
        (out_dir / "processed").mkdir(exist_ok=True, parents=True)
        pcv.print_image(img=mask, filename=str(out_dir / "processed" / f"{Path(filename).stem}_mask.png"))
        flat["mask_saved"] = True
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump(flat, f, ensure_ascii=False, indent=2)

    plt.close('all')  # กันเตือน figure overflow
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
    else:
        for f in files:
            try:
                jp = process_one(f, out_dir, DEBUG_MODE, SAVE_MASK)
                json_paths.append(jp)
            except Exception as e:
                print(f"Error processing {f}: {e}")

    out_csv = out_dir / "results.csv"
    aggregate_json_to_csv(json_paths, out_csv)
    print(f"Results written to {out_csv}")


if __name__ == '__main__':
    main()
