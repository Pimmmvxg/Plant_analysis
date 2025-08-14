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
OUTPUT_DIR   = r".\results"            # output folder (relative to this script by default)
EXTENSIONS   = ['.png']                # allowed image extensions (lowercase recommended)       
THREADS      = 1                       # start with 1; increase after it works (Windows-safe)
DEBUG_MODE   = 'print'                 # 'none' | 'print' | 'plot
SAVE_MASK    = False                   # save mask/processed image example

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

    # --- 1) Colorspace + 'a' channel + histogram (debug only) ---
    _ = pcv.visualize.colorspaces(rgb_img=rgb_img, original_img=False)
    a = pcv.rgb2gray_lab(rgb_img=rgb_img, channel='a')
    _ = pcv.visualize.histogram(img=a)

    # --- 2) Threshold on 'a' channel ---
    thresh = pcv.threshold.gaussian(
        gray_img=a, ksize=GAUSS_KSIZE, offset=GAUSS_OFFSET, object_type='dark'
    )
    mask_dilated = pcv.dilate(gray_img=thresh, ksize=DILATE_KSIZE, i=DILATE_ITER)

    # ถ้ามี PlantCV morphology.close ใช้ได้เลย; ถ้าไม่มีใช้ OpenCV แทน
    try:
        mask_closed = pcv.morphology.close(mask=mask_dilated, ksize=5, shape='ellipse')
    except Exception:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_closed = cv2.morphologyEx(mask_dilated, cv2.MORPH_CLOSE, kernel)

    mask_fill = pcv.fill(bin_img=mask_closed, size=FILL_SIZE)

    # --- 3) Safe ROI (clamped) & crop view (optional) ---
    H, W = rgb_img.shape[:2]
    if USE_FULL_IMAGE_ROI:
        x, y, w, h = 0, 0, W, H
    else:
        x = max(0, min(ROI_X, W - 1))
        y = max(0, min(ROI_Y, H - 1))
        w = max(1, min(ROI_W, W - x))
        h = max(1, min(ROI_H, H - y))

    # draw/display ROI for debug
    _ = pcv.roi.rectangle(img=rgb_img, x=x, y=y, w=w, h=h)
    
    _fc = cv2.findContours(mask_fill.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(_fc) == 3:
        _, contours, hierarchy = _fc
    else:
        contours, hierarchy = _fc
        
    roi_rect = (x, y, w, h)
    def intersects(bbox, roi): # check if bbox intersects with roi
        bx, by, bw, bh = bbox
        rx, ry, rw, rh = roi
        return not (bx + bw <= rx or rx + rw <= bx or by + bh <= ry or ry + rh <= by)
    
    kept_contours = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if intersects((x, y, w, h), roi_rect):
            kept_contours.append(cnt)
            
    partial_mask = np.zeros_like(mask_fill, dtype=np.uint8)
    if kept_contours:
        cv2.drawContours(partial_mask, kept_contours, -1, 255, thickness=cv2.FILLED)
        
    mask_fill = partial_mask

    # --- 5) Skeleton + prune twice ---
    skeleton = pcv.morphology.skeletonize(mask=mask_fill)
    pruned_skel, seg_img, edge_objects = pcv.morphology.prune(
        skel_img=skeleton, size=PRUNE_SIZE_1, mask=mask_fill
    )
    pruned_skel, seg_img, edge_objects = pcv.morphology.prune(
        skel_img=pruned_skel, size=PRUNE_SIZE_2, mask=mask_fill
    )

    # --- 6) Segment & sort into leaf vs. stem ---
    leaf_obj, stem_obj = pcv.morphology.segment_sort(
        skel_img=pruned_skel, objects=edge_objects, mask=mask_fill
    )

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