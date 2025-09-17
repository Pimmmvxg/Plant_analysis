import cv2
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional

from skimage.segmentation import slic, mark_boundaries, watershed
from skimage.measure import regionprops
from skimage.morphology import remove_small_objects, disk
from skimage.feature import peak_local_max
from skimage.filters import sobel, gaussian
from skimage.util import img_as_float
import matplotlib.pyplot as plt


@dataclass
class Params:
    # --- SLIC superpixels ---
    n_segments: int = 800          # จำนวน superpixels (เพิ่มถ้าใบเล็ก/รายละเอียดเยอะ)
    compactness: float = 10.0      # ความกลมของ superpixel (สูง = กลมมาก)
    slic_sigma: float = 0.8         # blur ก่อน SLIC เบา ๆ

    # --- Plant mask (Lab-a threshold) ---
    # OpenCV Lab: a≈128 คือ "กลาง" (เขียวค่าต่ำกว่า 128)
    a_thresh: int = 135            # 135 ~ เขียวจัด (ลองช่วง 132–142)
    open_ksize: int = 3            # เปิดรูเล็ก ๆ (morph open)
    close_ksize: int = 7           # ปิดรูพรุนในใบ (morph close)
    min_plant_area: int = 3000     # กรองวัตถุเล็ก ๆ ที่ไม่ใช่พืช

    # --- Distance/Seeds ---
    suppress_edges: bool = True    # กดทับร่องขอบก่อนคำนวณ distance
    seed_min_distance: int = 12    # ระยะห่างขั้นต่ำระหว่าง seed (pixels)
    seed_threshold_rel: float = 0.40  # เกณฑ์เลือกยอด dist (0.30–0.50)
    use_superpixel_centroid_filter: bool = True  # คัดเฉพาะ seed ที่อยู่กลางใบจริง ๆ

    # --- Watershed ---
    min_leaf_area: int = 300       # ลบใบจิ๋ว/ชิ้นส่วนรบกวนหลัง watershed
    smooth_before_ws_sigma: float = 0.0  # 0=ไม่ smooth; 0.5–1.0 ช่วยลด over-seg

    # --- Debug/output ---
    save_dir: str = "results_nottingham_like"
    show_plots: bool = True
    save_fig: bool = True


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def read_image(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def get_lab(img_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)
    return L, A, B


def get_initial_plant_mask_from_lab_a(A: np.ndarray, p: Params) -> np.ndarray:
    # ช่อง a ต่ำ (=เขียว) → พืช
    m = (A < p.a_thresh).astype(np.uint8) * 255
    # เปิด-ปิดรูพรุนเล็ก ๆ
    if p.open_ksize > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p.open_ksize, p.open_ksize))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
    if p.close_ksize > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p.close_ksize, p.close_ksize))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)
    # กรองคอมโพเนนต์เล็ก
    num, lbl = cv2.connectedComponents(m)
    keep = np.zeros_like(m, dtype=np.uint8)
    for i in range(1, num):
        area = int((lbl == i).sum())
        if area >= p.min_plant_area:
            keep[lbl == i] = 255
    return keep


def superpixels_lab(img_rgb: np.ndarray, p: Params) -> np.ndarray:
    lab_f = img_as_float(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB))
    seg = slic(
        lab_f,
        n_segments=p.n_segments,
        compactness=p.compactness,
        sigma=p.slic_sigma,
        start_label=1,
        channel_axis=2,
    )
    return seg.astype(np.int32)


def plant_mask_from_superpixels(A: np.ndarray, seg: np.ndarray, p: Params) -> np.ndarray:
    # เฉลี่ยค่า a ต่อ superpixel แล้ว threshold อีกชั้น (ทำให้ mask สะอาดขึ้น)
    max_lbl = seg.max()
    a_mean = np.zeros(max_lbl + 1, dtype=np.float32)
    for i in range(1, max_lbl + 1):
        a_mean[i] = A[seg == i].mean()
    plant_sp = (a_mean < p.a_thresh).astype(np.uint8)
    m = plant_sp[seg] * 255
    m = m.astype(np.uint8)

    # ทำความสะอาดเบา ๆ
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p.close_ksize, p.close_ksize))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)
    return m


def compute_distance_map(plant_mask: np.ndarray, p: Params, img_rgb: Optional[np.ndarray]=None) -> np.ndarray:
    m_bin = (plant_mask > 0).astype(np.uint8)
    m_bin = cv2.copyMakeBorder(m_bin, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)  # กันขอบ
    if p.suppress_edges and img_rgb is not None:
        # ลดอิทธิพลเส้นขอบก่อน distance: ใช้ Sobel gradient จาก mask/ภาพ
        grad = sobel(m_bin.astype(float))
        grad = (grad / (grad.max() + 1e-6)).astype(np.float32)
        # กดน้ำหนักบริเวณขอบ
        m_weight = (m_bin.astype(np.float32) * (1.0 - grad)).astype(np.float32)
        dist = cv2.distanceTransform((m_weight > 0.2).astype(np.uint8), cv2.DIST_L2, 5)
    else:
        dist = cv2.distanceTransform(m_bin, cv2.DIST_L2, 5)
    # ตัดขอบคืน
    dist = dist[1:-1, 1:-1]
    return dist


def seeds_from_distance_and_superpixels(dist: np.ndarray, seg: np.ndarray, plant_mask: np.ndarray, p: Params) -> np.ndarray:
    # หา local maxima จาก distance
    peaks = peak_local_max(
        dist,
        min_distance=p.seed_min_distance,
        threshold_rel=p.seed_threshold_rel,
        footprint=None,
        exclude_border=False,
    )
    if peaks.dtype == bool and peaks.shape == dist.shape:
        seed_mask = peaks & (plant_mask > 0)
    else:
        seed_mask = np.zeros_like(dist, dtype=bool)
        if peaks.size > 0:
            seed_mask[peaks[:, 0], peaks[:, 1]] = True
        seed_mask &= (plant_mask > 0)

    if not p.use_superpixel_centroid_filter:
        markers = cv2.connectedComponents(seed_mask.astype(np.uint8))[1]
        return markers

    # กรองให้เหลือ seed ที่ "เป็นจุดศูนย์กลางของ superpixel" ที่อยู่ในใบ
    markers = np.zeros_like(seg, dtype=np.int32)
    lbl = 0
    for sp_label in np.unique(seg):
        if sp_label <= 0:
            continue
        sp_mask = (seg == sp_label)
        # centroid ของ superpixel
        props = regionprops(sp_mask.astype(np.uint8))
        if not props:
            continue
        cy, cx = props[0].centroid
        cy, cx = int(round(cy)), int(round(cx))
        if 0 <= cy < dist.shape[0] and 0 <= cx < dist.shape[1]:
            if seed_mask[cy, cx]:  # centroid ซ้อนทับยอดของ distance และอยู่ในพืช
                lbl += 1
                markers[cy, cx] = lbl

    # ถ้าได้ seed น้อยเกินไป ให้ fallback ใช้เม็ดจาก seed_mask ตรง ๆ
    if lbl < 2:
        markers = cv2.connectedComponents(seed_mask.astype(np.uint8))[1]

    return markers


def watershed_leaves(gradient: np.ndarray, markers: np.ndarray, plant_mask: np.ndarray, p: Params) -> np.ndarray:
    # ปรับ smooth gradient เพื่อกัน over-seg (ตามต้องการ)
    g = gradient.copy()
    if p.smooth_before_ws_sigma > 0:
        g = gaussian(g, sigma=p.smooth_before_ws_sigma)
    lbl = watershed(g, markers=markers, mask=(plant_mask > 0))
    # ลบชิ้นเล็ก ๆ
    lbl = remove_small_objects(lbl, min_size=p.min_leaf_area)
    return lbl.astype(np.int32)


def run_pipeline(image_path: str, p: Params):
    img = read_image(image_path)
    out_dir = Path(p.save_dir) / Path(image_path).stem
    ensure_dir(out_dir)

    # === 1) Superpixels (Lab space) ===
    seg = superpixels_lab(img, p)

    # === 2) Plant mask (Lab-a threshold + superpixel refine) ===
    _, A, _ = get_lab(img)
    plant_m0 = get_initial_plant_mask_from_lab_a(A, p)
    plant_m = plant_mask_from_superpixels(A, seg, p)

    # รวม (union) เพื่อให้ mask แน่นขึ้น
    plant_mask = cv2.bitwise_or(plant_m0, plant_m)

    # === 3) Distance map (optional suppress edges) ===
    dist = compute_distance_map(plant_mask, p, img_rgb=img)

    # === 4) Seeds from distance + (optional) superpixel centroid filtering ===
    markers = seeds_from_distance_and_superpixels(dist, seg, plant_mask, p)

    # === 5) Gradient (edge) for watershed ===
    # ใช้ Sobel จากภาพ grayscale (เฉพาะบริเวณพืช) + เติม edges ของ mask
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    grad_img = sobel(gray)
    grad_mask = sobel((plant_mask > 0).astype(float))
    gradient = (grad_img * 0.7 + grad_mask * 0.3)

    # === 6) Watershed ===
    labels_ws = watershed_leaves(gradient, markers, plant_mask, p)

    # === 7) บันทึกผล ===
    overlay = mark_boundaries(img, labels_ws, color=None, mode='thick')
    plant_mask_u8 = (plant_mask > 0).astype(np.uint8) * 255

    cv2.imwrite(str(out_dir / "plant_mask.png"), plant_mask_u8)
    cv2.imwrite(str(out_dir / "distance_map.png"), cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
    cv2.imwrite(str(out_dir / "labels_ws.png"), (labels_ws.astype(np.uint16)))
    cv2.imwrite(str(out_dir / "overlay_ws.jpg"), cv2.cvtColor((overlay * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

    if p.show_plots or p.save_fig:
        fig, axs = plt.subplots(2, 3, figsize=(14, 9))
        axs[0, 0].set_title("Original")
        axs[0, 0].imshow(img); axs[0, 0].axis('off')

        axs[0, 1].set_title("SLIC Superpixels")
        axs[0, 1].imshow(mark_boundaries(img, seg)); axs[0, 1].axis('off')

        axs[0, 2].set_title("Plant Mask")
        axs[0, 2].imshow(plant_mask, cmap='gray'); axs[0, 2].axis('off')

        axs[1, 0].set_title("Distance Map")
        axs[1, 0].imshow(dist, cmap='viridis'); axs[1, 0].axis('off')

        axs[1, 1].set_title("Seeds (markers)")
        axs[1, 1].imshow(mark_boundaries(img, (markers > 0).astype(int))); axs[1, 1].axis('off')

        axs[1, 2].set_title("Watershed (Leaves)")
        axs[1, 2].imshow(overlay); axs[1, 2].axis('off')

        plt.tight_layout()
        if p.save_fig:
            fig.savefig(str(out_dir / "debug_panel.png"), dpi=200)
        if p.show_plots:
            plt.show()
        plt.close(fig)

    # รายงานสั้น ๆ
    n_leaves = int(labels_ws.max())
    print(f"[DONE] {image_path}")
    print(f"  - Leaves segmented: {n_leaves}")
    print(f"  - Outputs saved to: {out_dir}")
    
def segment_component(img, seg, plant_mask, p, global_leaf_id_start=0):
    """
    ทำใบรายใบบน mask หนึ่ง component (ต้นเดียว) แล้วคืน labels กับจำนวนใบที่เพิ่ม
    """
    dist = compute_distance_map(plant_mask, p, img_rgb=img)
    markers = seeds_from_distance_and_superpixels(dist, seg, plant_mask, p)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    grad_img = sobel(gray)
    grad_mask = sobel((plant_mask > 0).astype(float))
    gradient = (grad_img * 0.7 + grad_mask * 0.3)

    labels_ws = watershed_leaves(gradient, markers, plant_mask, p)

    # reindex ให้ไม่ชนกัน (เลื่อนค่า label ให้ต่อจาก global counter)
    if labels_ws.max() > 0 and global_leaf_id_start > 0:
        labels_ws = np.where(labels_ws > 0, labels_ws + global_leaf_id_start, 0)

    n_leaves = int(labels_ws.max() - global_leaf_id_start)
    return labels_ws, n_leaves


def run_pipeline(image_path: str, p: Params):
    img = read_image(image_path)
    out_dir = Path(p.save_dir) / Path(image_path).stem
    ensure_dir(out_dir)

    # === 1) Superpixels (Lab space) ===
    seg = superpixels_lab(img, p)

    # === 2) Plant mask (Lab-a threshold + superpixel refine) ===
    _, A, _ = get_lab(img)
    plant_m0 = get_initial_plant_mask_from_lab_a(A, p)
    plant_m = plant_mask_from_superpixels(A, seg, p)
    plant_mask = cv2.bitwise_or(plant_m0, plant_m)
    plant_bin = (plant_mask > 0).astype(np.uint8)

    # === 2.1) แยก "หลายต้น" ด้วย connected components ===
    num_all, lbl_all = cv2.connectedComponents(plant_bin)
    # กรองเฉพาะ component ที่มีขนาดพอสมควร (กันตะไคร่/เศษเล็ก)
    comp_ids = []
    comp_areas = []
    for k in range(1, num_all):
        area = int((lbl_all == k).sum())
        if area >= max(p.min_plant_area, 1500):  # เกณฑ์กันสัญญาณรบกวน
            comp_ids.append(k)
            comp_areas.append(area)

    if len(comp_ids) == 0:
        # ไม่มีต้นที่ผ่านเกณฑ์
        cv2.imwrite(str(out_dir / "plant_mask.png"), plant_bin * 255)
        print(f"[DONE] {image_path}\n  - Leaves segmented: 0 (no valid plants)\n  - Outputs: {out_dir}")
        return

    # === 3) วนทีละต้น: ทำใบรายใบภายในแต่ละ component แล้วประกอบผลกลับ ===
    H, W = img.shape[:2]
    labels_global = np.zeros((H, W), dtype=np.int32)
    global_leaf_id = 0
    debug_seeds = np.zeros((H, W), dtype=np.uint8)  # optional แสดง seeds รวมหากอยาก plot

    for cid in comp_ids:
        comp_mask = (lbl_all == cid).astype(np.uint8) * 255

        # ทำงานใน bounding box ของต้น เพื่อลดเวลาและกัน buffer overlap
        ys, xs = np.where(comp_mask > 0)
        y0, y1 = max(0, ys.min()-5), min(H, ys.max()+6)
        x0, x1 = max(0, xs.min()-5), min(W, xs.max()+6)

        img_roi = img[y0:y1, x0:x1]
        seg_roi = seg[y0:y1, x0:x1]
        mask_roi = comp_mask[y0:y1, x0:x1]

        # เพื่อกันการติดกันของสองต้นที่ชิดกันมาก ให้ "หด" ขอบเล็กน้อยก่อน segment ใบ
        # จะช่วยลดโอกาส watershed ล้นข้ามต้น
        k_er = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_roi_tight = cv2.erode(mask_roi, k_er, iterations=1)

        # ทำ segment เฉพาะใน component นี้
        labels_roi, n_new = segment_component(img_roi, seg_roi, mask_roi_tight, p, global_leaf_id_start=global_leaf_id)

        # วางกลับภาพใหญ่
        labels_global[y0:y1, x0:x1] = np.where(labels_roi > 0, labels_roi, labels_global[y0:y1, x0:x1])

        global_leaf_id += n_new

    # === 4) บันทึกผลรวมทั้งภาพ ===
    overlay = mark_boundaries(img, labels_global, color=None, mode='thick')
    cv2.imwrite(str(out_dir / "plant_mask.png"), plant_bin * 255)
    cv2.imwrite(str(out_dir / "labels_ws.png"), labels_global.astype(np.uint16))
    cv2.imwrite(str(out_dir / "overlay_ws.jpg"), cv2.cvtColor((overlay * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

    # แผง debug รวม (เลือกจะแสดง/เซฟก็ได้)
    if p.show_plots or p.save_fig:
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].set_title("Original")
        axs[0].imshow(img); axs[0].axis('off')

        axs[1].set_title("Plant Mask (multi-plant)")
        axs[1].imshow(plant_bin, cmap='gray'); axs[1].axis('off')

        axs[2].set_title(f"Watershed Leaves (All Plants) | N={labels_global.max()}")
        axs[2].imshow(overlay); axs[2].axis('off')

        plt.tight_layout()
        if p.save_fig:
            fig.savefig(str(out_dir / "debug_panel_multi.png"), dpi=200)
        if p.show_plots:
            plt.show()
        plt.close(fig)

    print(f"[DONE] {image_path}")
    print(f"  - Plants detected: {len(comp_ids)} (areas: {comp_areas})")
    print(f"  - Leaves segmented (total): {int(labels_global.max())}")
    print(f"  - Outputs saved to: {out_dir}")



if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Nottingham-like leaf segmentation test (SLIC + distance + watershed)")
    ap.add_argument("image", help="Path to input top-view plant image (RGB)")
    ap.add_argument("--out", default=None, help="Override output dir")
    ap.add_argument("--show", action="store_true", help="Show debug plots")
    ap.add_argument("--no-save-fig", action="store_true", help="Do not save debug panel figure")
    ap.add_argument("--a-thresh", type=int, default=None, help="Lab-a threshold (default 135)")
    ap.add_argument("--segments", type=int, default=None, help="SLIC n_segments (default 800)")
    ap.add_argument("--seed-rel", type=float, default=None, help="seed threshold_rel (default 0.40)")
    ap.add_argument("--min-leaf", type=int, default=None, help="min leaf area (default 300)")
    args = ap.parse_args()

    p = Params()
    if args.out:
        p.save_dir = args.out
    if args.show:
        p.show_plots = True
    if args.no_save_fig:
        p.save_fig = False
    if args.a_thresh is not None:
        p.a_thresh = args.a_thresh
    if args.segments is not None:
        p.n_segments = args.segments
    if args.seed_rel is not None:
        p.seed_threshold_rel = args.seed_rel
    if args.min_leaf is not None:
        p.min_leaf_area = args.min_leaf

    run_pipeline(args.image, p)
