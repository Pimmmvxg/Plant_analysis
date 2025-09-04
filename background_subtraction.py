# test_bg_subtract.py
import os
import cv2
import numpy as np

# ----- ตั้งค่าไฟล์ทดสอบ -----
BACKGROUND = r"C:\Cantonese\real\bg.jpg"
FOREGROUND = r"C:\Cantonese\real\fg2.jpg"
OUTDIR = "debug_bg_subtract2"  # โฟลเดอร์ผลลัพธ์

os.makedirs(OUTDIR, exist_ok=True)

def _resize_to_match(a, b):
    if a.shape[:2] != b.shape[:2]:
        # resize รูปที่ "ใหญ่กว่า (ตามพื้นที่พิกเซล)" ให้เท่ารูปเล็กกว่า
        area_a, area_b = a.shape[0]*a.shape[1], b.shape[0]*b.shape[1]
        if area_a >= area_b:
            b_h, b_w = b.shape[:2]
            a = cv2.resize(a, (b_w, b_h), interpolation=cv2.INTER_AREA)
        else:
            a_h, a_w = a.shape[:2]
            b = cv2.resize(b, (a_w, a_h), interpolation=cv2.INTER_AREA)
    return a, b

def _align_ecc(ref_bgr, mov_bgr, warp_mode=cv2.MOTION_EUCLIDEAN, n_iters=200, eps=1e-7):
    ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
    mov_gray = cv2.cvtColor(mov_bgr, cv2.COLOR_BGR2GRAY)
    warp_matrix = np.eye(2, 3, dtype=np.float32) if warp_mode != cv2.MOTION_HOMOGRAPHY else np.eye(3, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, n_iters, eps)
    try:
        _, warp_matrix = cv2.findTransformECC(ref_gray, mov_gray, warp_matrix, warp_mode, criteria, None, 5)
        h, w = ref_gray.shape[:2]
        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            aligned = cv2.warpPerspective(mov_bgr, warp_matrix, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        else:
            aligned = cv2.warpAffine(mov_bgr, warp_matrix, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        return aligned
    except cv2.error:
        print("[WARN] ECC alignment failed; use unaligned foreground.")
        return mov_bgr

def _remove_small(bin_img, min_size=300):
    nb, labels, stats, _ = cv2.connectedComponentsWithStats(bin_img, connectivity=8)
    if nb <= 1:
        return bin_img
    keep = np.zeros_like(bin_img)
    for i in range(1, nb):
        if stats[i, cv2.CC_STAT_AREA] >= int(min_size):
            keep[labels == i] = 255
    return keep

def background_subtraction(bg_path, fg_path,
                           method="absdiff", align=True,
                           gaussian_blur=5, threshold=0,
                           object_type="light", morph_open=3, morph_close=7,
                           min_obj_size=500):
    bg = cv2.imread(bg_path, cv2.IMREAD_COLOR)
    fg = cv2.imread(fg_path, cv2.IMREAD_COLOR)
    if bg is None or fg is None:
        raise FileNotFoundError("อ่านภาพไม่ได้ ตรวจสอบพาธ BACKGROUND/FOREGROUND")

    bg, fg = _resize_to_match(bg, fg)
    if align:
        fg = _align_ecc(bg, fg)

    if method.lower() == "mog2":
        bgsub = cv2.createBackgroundSubtractorMOG2(history=2, varThreshold=16, detectShadows=False)
        _ = bgsub.apply(bg, learningRate=1.0)
        diff_gray = bgsub.apply(fg, learningRate=0.0)
    else:
        diff = cv2.absdiff(bg, fg)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    if gaussian_blur and gaussian_blur % 2 == 1 and gaussian_blur >= 3:
        diff_gray = cv2.GaussianBlur(diff_gray, (gaussian_blur, gaussian_blur), 0)

    if threshold and 1 <= int(threshold) <= 255:
        th = int(threshold)
        flag = cv2.THRESH_BINARY_INV if object_type == "dark" else cv2.THRESH_BINARY
        _, bin_img = cv2.threshold(diff_gray, th, 255, flag)
    else:
        flag = cv2.THRESH_BINARY_INV if object_type == "dark" else cv2.THRESH_BINARY
        _, bin_img = cv2.threshold(diff_gray, 0, 255, flag + cv2.THRESH_OTSU)

    def _mk(k): return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    if morph_open and morph_open >= 2:
        bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, _mk(morph_open), iterations=1)
    if morph_close and morph_close >= 2:
        bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, _mk(morph_close), iterations=1)

    bin_img = _remove_small(bin_img, min_obj_size)
    fgmask = (bin_img > 0).astype(np.uint8) * 255
    return bg, fg, fgmask, diff_gray

def _save_preview(bg, fg, mask, diff_gray, outdir):
    # บันทึกไฟล์หลัก
    cv2.imwrite(os.path.join(outdir, "00_bg.jpg"), bg)
    cv2.imwrite(os.path.join(outdir, "01_fg_aligned.jpg"), fg)
    cv2.imwrite(os.path.join(outdir, "02_diff_gray.png"), diff_gray)
    cv2.imwrite(os.path.join(outdir, "03_mask.png"), mask)

    # ทำ overlay หน้ากากเป็นเส้นขอบ
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = fg.copy()
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(outdir, "04_overlay_contours.jpg"), overlay)

def _print_stats(mask):
    H, W = mask.shape[:2]
    area = int(cv2.countNonZero(mask))
    cov = area / float(H * W + 1e-9)
    nb, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    nobj = max(0, nb - 1)
    print(f"[STATS] Size: {W}x{H}  area_px={area}  coverage={cov:.4f}  n_objects={nobj}")
    if nobj > 0:
        largest = stats[1:, cv2.CC_STAT_AREA].max()
        print(f"[STATS] largest_object_px={largest}")

if __name__ == "__main__":
    bg, fg, mask, diff_gray = background_subtraction(
        BACKGROUND, FOREGROUND,
        method="absdiff",     # ลอง "mog2" ได้ถ้าต้องการ
        align=True,
        gaussian_blur=5,
        threshold=0,          # 0 = ใช้ Otsu อัตโนมัติ
        object_type="light",  # ถ้าวัตถุเข้มกว่าพื้นหลังให้เปลี่ยนเป็น "dark"
        morph_open=3,
        morph_close=7,
        min_obj_size=500
    )
    _save_preview(bg, fg, mask, diff_gray, OUTDIR)
    _print_stats(mask)
    print(f"[OK] Saved results to: {os.path.abspath(OUTDIR)}")
