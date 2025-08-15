import os
import sys
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import cv2
import numpy as np
import pandas as pd
from plantcv import plantcv as pcv

INPUT_PATH   = r"C:\Cantonese\Test"         # folder OR single image path
OUTPUT_DIR   = r".\results2"            # output folder (relative to this script by default)
EXTENSIONS   = ['.png']                # allowed image extensions (lowercase recommended)       
THREADS      = 1                       # start with 1; increase after it works (Windows-safe)
DEBUG_MODE   = 'print'                 # 'none' | 'print' | 'plot
SAVE_MASK    = False                   # save mask/processed image example

def auto_select_mask(rgb_img):
    import numpy as np, cv2
    from plantcv import plantcv as pcv

    H, W = rgb_img.shape[:2]
    area_total = H * W

    # ----- 1) candidate gray channels -----
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

    # ----- 2) threshold methods to try -----
    methods = []
    for name, g in chans:
        try: g = pcv.transform.rescale(gray_img=g, lower=0, upper=255)
        except: pass
        methods += [
            (name, "otsu",     {"object_type":"dark"}),
            (name, "otsu",     {"object_type":"light"}),
            (name, "triangle", {"object_type":"dark"}),
            (name, "triangle", {"object_type":"light"}),
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
                used = ("otsu", params["object_type"], None)
            elif meth == "triangle":
                m = pcv.threshold.triangle(gray_img=gray, **params)
                used = ("triangle", params["object_type"], None)
            else:
                _, k, off, obj = meth
                m = pcv.threshold.gaussian(gray_img=gray, ksize=k, offset=off, object_type=obj)
                used = ("gaussian", obj, k)

            area = int(np.count_nonzero(m))
            ratio = area / max(area_total, 1)

            _fc = cv2.findContours(m.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = _fc[2] if len(_fc) == 3 else _fc[0]
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

            # ----- 3) scoring -----
            def score_area(r):
                if 0.5 <= r <= 0.6:
                    return 2.0 - abs(0.30 - r)
                return 0.5 - abs(0.30 - r)
            s = score_area(ratio)
            s -= 0.1 * max(0, n_comp - 1)
            s += 0.5 * solidity

            candidates.append((s, name, used, m, ratio, n_comp, solidity))
        except Exception:
            continue

    if not candidates:
        fallback = pcv.threshold.otsu(gray_img=pcv.rgb2gray_lab(rgb_img=rgb_img, channel='a'), object_type='dark')
        return fallback, {"channel":"lab_a","method":"otsu","object_type":"dark","ksize":None,
                          "area_ratio": float(np.count_nonzero(fallback))/max(area_total,1),
                          "n_components": 0, "solidity": 0.0}

    best = max(candidates, key=lambda x: x[0])
    _, name, used, mask, ratio, n_comp, solidity = best
    method, objtype, ksize = used
    return mask, {"channel":name,"method":method,"object_type":objtype,"ksize":ksize,
                  "area_ratio": float(ratio), "n_components": int(n_comp), "solidity": float(solidity)}

def segment(rgb_img, filename):
    # --- Tunable parameters (safe defaults) ---
    USE_FULL_IMAGE_ROI = False   # set True to use full image as ROI
    ROI_X, ROI_Y, ROI_W, ROI_H = 100, 320, 240, 150
    GAUSS_KSIZE   = 9001          # must be positive odd (e.g., 51/101/151)
    GAUSS_OFFSET  = 10
    DILATE_KSIZE  = 2
    DILATE_ITER   = 1
    FILL_SIZE     = 30
    PRUNE_SIZE_1  = 100
    PRUNE_SIZE_2  = 50
    ROI_TYPE      = 'partial'    # or 'cutto' / 'largest'

    mask0, info = auto_select_mask(rgb_img)
    
    pcv.outputs.add_observation(
    sample='default',
    variable='auto_channel',
    trait='text',
    method='auto_select',
    scale='none',
    datatype=str,               # หรือ 'str' ก็ได้ในบางเวอร์ชัน
    value=info['channel'],
    label='channel'
    )

    pcv.outputs.add_observation(
        sample='default',
        variable='auto_method',
        trait='text',
        method='auto_select',
        scale='none',
        datatype=str,
        value=info['method'],
        label='method'
    )

    pcv.outputs.add_observation(
        sample='default',
        variable='auto_object_type',
        trait='text',
        method='auto_select',
        scale='none',
        datatype=str,
        value=info['object_type'],
        label='object_type'
    )

    pcv.outputs.add_observation(
        sample='default',
        variable='auto_ksize',
        trait='text',
        method='auto_select',
        scale='none',
        datatype=str,
        value=str(info['ksize']),
        label='ksize'
    )

    pcv.outputs.add_observation(
        sample='default',
        variable='auto_area_ratio',
        trait='ratio',
        method='auto_select',
        scale='none',
        datatype=float,
        value=float(info['area_ratio']),
        label='area_ratio'
    )

    pcv.outputs.add_observation(
        sample='default',
        variable='auto_n_components',
        trait='count',
        method='auto_select',
        scale='count',
        datatype=int,
        value=int(info['n_components']),
        label='components'
    )

    pcv.outputs.add_observation(
        sample='default',
        variable='auto_solidity',
        trait='ratio',
        method='auto_select',
        scale='none',
        datatype=float,
        value=float(info['solidity']),
        label='solidity'
    )

    
    mask_dilated = pcv.dilate(gray_img=mask0, ksize=2, i=1)
    try:
        mask_closed = pcv.close(mask=mask_dilated, ksize=5, shape='ellipse')
    except Exception:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_closed = cv2.morphologyEx(mask_dilated, cv2.MORPH_CLOSE, kernel)
    mask_fill = pcv.fill(bin_img=mask_closed, size=30)
    
    #ROI selection
    H, W = rgb_img.shape[:2]
    if USE_FULL_IMAGE_ROI:
        x, y, w, h = 0, 0, W, H
    else:
        x = max(0, min(ROI_X, W - 1))
        y = max(0, min(ROI_Y, H - 1))
        w = max(1, min(ROI_W, W - x))
        h = max(1, min(ROI_H, H - y))

    _ = pcv.roi.rectangle(img=rgb_img, x=x, y=y, w=w, h=h)  # debug only

    _fc = cv2.findContours(mask_fill.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(_fc) == 3:  # OpenCV 3
        _, contours, hierarchy = _fc
    else:             # OpenCV 4
        contours, hierarchy = _fc

    roi_rect = (x, y, w, h)
    def intersects(bbox, roi):
        bx, by, bw, bh = bbox
        rx, ry, rw, rh = roi
        return not (bx + bw <= rx or rx + rw <= bx or by + bh <= ry or ry + rh <= by)

    kept_contours = []
    for cnt in contours:
        bx, by, bw, bh = cv2.boundingRect(cnt)     # ← ใช้ bx/by/bw/bh ไม่ให้ทับ ROI
        if intersects((bx, by, bw, bh), roi_rect): # ← เทียบ bbox ของ object กับ ROI
            kept_contours.append(cnt)

    partial_mask = np.zeros_like(mask_fill, dtype=np.uint8)
    if kept_contours:
        cv2.drawContours(partial_mask, kept_contours, -1, 255, thickness=cv2.FILLED)
    mask_fill = partial_mask
    
    # --- 5) Skeleton + prune twice ---
    skeleton = pcv.morphology.skeletonize(mask=mask_fill)
    sizes = [50, 75, 100, 125, 150, 175, 200, 250, 300, 350, 400]
    last_err = None
    pruned_skel, edge_objects = None, None
    for sz in sizes:
        try:
            ret = pcv.morphology.prune(
                skel_img=skeleton if pruned_skel is None else pruned_skel,
                size=sz,
                mask=mask_fill,
            )
            if isinstance(ret, tuple) and len(ret) == 3:
                pruned_skel, seg_img, edge_objects = ret
            elif isinstance(ret, tuple) and len(ret) == 2:
                pruned_skel, edge_objects = ret
                seg_img = None
            else:
                raise ValueError(f"Unexpected return type: {type(ret)}, len={len(ret) if hasattr(ret,'__len__') else 'n/a'}")
            # Try downstream steps; if they succeed, break
            leaf_obj, stem_obj = pcv.morphology.segment_sort(
                skel_img=pruned_skel, objects=edge_objects, mask=mask_fill
            )
            segmented_img, labeled_img = pcv.morphology.segment_id(
                skel_img=pruned_skel, objects=leaf_obj, mask=mask_fill
            )
            break
        except Exception as e:
            last_err = e
            continue
    else:
        raise last_err if last_err else RuntimeError("Failed to prune skeleton with all sizes.")
    # --- 6) Segment & sort into leaf vs. stem ---
    leaf_obj, stem_obj = pcv.morphology.segment_sort(
        skel_img=pruned_skel, objects=edge_objects, mask=mask_fill
    )
    _ = pcv.morphology.fill_segments(mask=mask_fill, objects=leaf_obj, label="default")
    branch_pts_mask = pcv.morphology.find_branch_pts(
        skel_img=pruned_skel, mask=mask_fill, label="default"
    )
    
    _ = pcv.analyze.size(img=rgb_img, labeled_mask=mask_fill, label="default")

    # --- 7) Optional fills/branch points/IDs/lengths ---
    _ = pcv.morphology.fill_segments(mask=mask_fill, objects=leaf_obj, label="default")

    branch_pts_mask = pcv.morphology.find_branch_pts(
        skel_img=pruned_skel, mask=mask_fill, label="default"
    )

    segmented_img, labeled_img = pcv.morphology.segment_id(
        skel_img=pruned_skel, objects=leaf_obj, mask=mask_fill
    )
    _ = pcv.morphology.segment_euclidean_length(
        segmented_img=segmented_img, objects=leaf_obj, label="default"
    )

    # Optional: size analysis (on whole image or ROI crop if you prefer)
    _ = pcv.analyze.size(img=rgb_img, labeled_mask=mask_fill, label="default")

    # --- 8) JSON‑safe custom metrics ---
    num_leaf_segments  = int(len(leaf_obj)) if leaf_obj is not None else 0
    num_stem_segments  = int(len(stem_obj)) if stem_obj is not None else 0
    num_branch_points  = int(np.count_nonzero(branch_pts_mask)) if branch_pts_mask is not None else 0

    pcv.outputs.add_observation(sample='default', variable='num_leaf_segments',
                                trait='count', method='pcv.morphology.segment_sort',
                                scale='count', datatype=int, value=num_leaf_segments, label='leaf segments')
    pcv.outputs.add_observation(sample='default', variable='num_stem_segments',
                                trait='count', method='pcv.morphology.segment_sort',
                                scale='count', datatype=int, value=num_stem_segments, label='stem segments')
    pcv.outputs.add_observation(sample='default', variable='num_branch_points',
                                trait='count', method='pcv.morphology.find_branch_pts',
                                scale='count', datatype=int, value=num_branch_points, label='branch points')

    # --- 9) Return dict + mask (JSON‑safe) ---
    return {
        "filename": filename,
        "roi_x": int(x), "roi_y": int(y), "roi_w": int(w), "roi_h": int(h),
        "roi_type": ROI_TYPE,
        "num_leaf_segments": num_leaf_segments,
        "num_stem_segments": num_stem_segments,
        "num_branch_points": num_branch_points
    }, mask_fill


def process_one(path: Path, out_dir: Path, debug_mode: str, save_mask: bool):
    pcv.params.debug = debug_mode
    pcv.params.debug_outdir = str(out_dir / "debug")
    pcv.params.dpi = 150
    pcv.outputs.clear()

    img, path_str, filename = pcv.readimage(filename=str(path))

    # segment now returns (dict, mask)
    extra, mask = segment(img, filename)

    # Collect PlantCV observations
    results_dict = pcv.outputs.observations.copy()

    flat = {}
    for sample, vars_ in results_dict.items():
        for var_name, record in vars_.items():
            flat[var_name] = record.get('value', None)

    flat.update(extra)

    # Save per-image JSON
    json_dir = out_dir / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    out_json = json_dir / f"{Path(filename).stem}.json"
    
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(flat, f, ensure_ascii=False, indent=2)

    if save_mask and mask is not None:
        (out_dir / "processed").mkdir(exist_ok=True, parents=True)
        pcv.print_image(img=mask, filename=str(out_dir / "processed" / f"{Path(filename).stem}_mask.png"))
        flat["mask_saved"] = True
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump(flat, f, ensure_ascii=False, indent=2)

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