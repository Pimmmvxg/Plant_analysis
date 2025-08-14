
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PlantCV Batch Pipeline Template
--------------------------------
- Walks an input folder for images (by extension)
- Runs a PlantCV analysis function on each image
- Saves per-image JSON results
- Optionally saves debug/processed images
- Aggregates all JSONs into one CSV

Usage:
  python plantcv_batch.py --input /path/to/images --output ./results --ext .jpg .png --threads 4 --debug print

Notes:
  1) Edit `analyze_image(img, filename)` to match your pipeline.
  2) PlantCV docs: https://plantcv.danforthcenter.org/
"""

import os
import sys
import json
import glob
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from plantcv import plantcv as pcv

# -----------------------------
# Your analysis pipeline here
# -----------------------------
def analyze_image(img, filename):
    """Edit this function with your final PlantCV pipeline.
    Return a dict of results that will be merged into the output JSON/CSV.
    """
    # Example steps (replace with your own):
    # 1) Convert to LAB and use 'a' channel to separate green-magenta
    a = pcv.rgb2gray_lab(rgb_img=img, channel='a')

    # 2) Threshold (example Otsu). Adjust as needed.
    a_thresh = pcv.threshold.otsu(gray_img=a, object_type='light')

    # 3) Clean up mask
    a_fill = pcv.fill(bin_img=a_thresh, size=100)
    a_open = pcv.morphology.open(mask=a_fill, ksize=5, shape='ellipse')

    # 4) Find objects and get ROI (example: everything in mask)
    id_objects, obj_hierarchy = pcv.find_objects(img=img, mask=a_open)
    roi_contour, roi_hierarchy = pcv.roi.rectangle(img=img, x=0, y=0, h=img.shape[0], w=img.shape[1])

    # 5) Keep objects in ROI
    roi_objects, hierarchy, kept_mask, obj_area = pcv.roi_objects(img=img, roi_contour=roi_contour,
                                                                  roi_hierarchy=roi_hierarchy,
                                                                  object_contour=id_objects,
                                                                  obj_hierarchy=obj_hierarchy,
                                                                  roi_type='partial')
    # 6) Combine and analyze
    if len(roi_objects) > 0:
        obj, mask = pcv.object_composition(img=img, contours=roi_objects, hierarchy=hierarchy)
        # Shape measurements
        shape_img = pcv.analyze_object(img=img, obj=obj, mask=mask)
        # Color measurements (in HSV as example)
        pcv.analyze_color(img=img, mask=mask, hist_plot_type=None)
    else:
        mask = a_open

    # 7) Optional morphology features (skeleton as example)
    _ = pcv.morphology.skeletonize(mask=mask)

    # 8) You can add your own custom metrics and observations
    pcv.outputs.add_observation(sample='default', variable='num_objects', trait='count',
                                method='pcv.count', scale='count', datatype=int, value=len(roi_objects),
                                label='objects')

    # Anything you want to return as flat key-values for the CSV
    return {
        "filename": filename,
        "num_objects": int(len(roi_objects)),
    }


def process_one(path, out_dir, debug_mode):
    pcv.params.debug = debug_mode  # 'none' | 'print' | 'plot'
    pcv.params.debug_outdir = str(Path(out_dir) / "debug")
    pcv.params.dpi = 150  # optional
    pcv.outputs.clear()

    img, path_str, filename = pcv.readimage(filename=str(path))

    # Run your analysis
    extra = analyze_image(img, filename)

    # Collect PlantCV's internal measurements into a dict
    results_dict = pcv.outputs.observations.copy()

    # Flatten observations
    flat = {}
    for sample, vars_ in results_dict.items():
        for var_name, record in vars_.items():
            flat[var_name] = record.get('value', None)

    # Merge with extra fields (filename, etc.)
    flat.update(extra)

    # Save per-image JSON
    json_dir = Path(out_dir) / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    out_json = json_dir / f"{Path(filename).stem}.json"
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(flat, f, ensure_ascii=False, indent=2)

    # Optionally save a key image (e.g., the mask) for quick QA
    # pcv.print_image(img=img, filename=str(Path(out_dir) / 'processed' / filename))

    return str(out_json)


def aggregate_json_to_csv(json_paths, out_csv):
    rows = []
    for jp in json_paths:
        try:
            with open(jp, 'r', encoding='utf-8') as f:
                rows.append(json.load(f))
        except Exception as e:
            print(f"[WARN] Failed to read {jp}: {e}")
    if not rows:
        print("[WARN] No JSON files to aggregate.")
        return
    df = pd.DataFrame(rows)
    # Move filename to the first column if present
    cols = list(df.columns)
    if 'filename' in cols:
        cols.insert(0, cols.pop(cols.index('filename')))
        df = df[cols]
    df.to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f"[OK] Wrote CSV -> {out_csv}")


def parse_args():
    p = argparse.ArgumentParser(description="PlantCV batch pipeline runner")
    p.add_argument('--input', '-i', required=True, help="Input folder of images")
    p.add_argument('--output', '-o', default="./results", help="Output folder")
    p.add_argument('--ext', nargs='+', default=['.jpg', '.png', '.jpeg', '.tif', '.tiff'],
                   help="Image extensions to include (space-separated)")
    p.add_argument('--threads', type=int, default=1, help="Number of worker processes")
    p.add_argument('--debug', choices=['none', 'print', 'plot'], default='none',
                   help="PlantCV debug mode: none | print | plot")
    return p.parse_args()


def main():
    args = parse_args()
    in_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'debug').mkdir(exist_ok=True)

    # Collect files
    exts = set([e.lower() for e in args.ext])
    files = [p for p in in_dir.rglob('*') if p.suffix.lower() in exts]
    if not files:
        print(f"[ERR] No images found in {in_dir} with extensions: {sorted(exts)}")
        sys.exit(1)

    print(f"[INFO] Found {len(files)} images. Processing with {args.threads} process(es)...")

    json_paths = []
    if args.threads > 1:
        with ProcessPoolExecutor(max_workers=args.threads) as ex:
            futures = {ex.submit(process_one, f, out_dir, args.debug): f for f in files}
            for fut in as_completed(futures):
                try:
                    jp = fut.result()
                    json_paths.append(jp)
                except Exception as e:
                    print(f"[ERR] Failed on {futures[fut]}: {e}")
    else:
        for f in files:
            try:
                jp = process_one(f, out_dir, args.debug)
                json_paths.append(jp)
            except Exception as e:
                print(f"[ERR] Failed on {f}: {e}")

    # Aggregate
    out_csv = out_dir / 'results.csv'
    aggregate_json_to_csv(json_paths, out_csv)
    print("[DONE] All finished.")


if __name__ == '__main__':
    main()
