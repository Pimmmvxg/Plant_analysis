# mask_tuner.py
import argparse, json, os, cv2, numpy as np

# ====== พยายาม import ฟังก์ชันจากไฟล์ของคุณ ======
try:
    from custom_masking import auto_thresh_lab_a_otsu_guard
except Exception as e:
    print("[WARN] import from custom_masking.py ไม่สำเร็จ:", e)
    print("[INFO] จะใช้เวอร์ชัน fallback ภายในสคริปต์แทน")
    def auto_thresh_lab_a_otsu_guard(
            rgb_img,
            object_type="dark",
            v_bg=45,
            blur_ksize=3,
            s_min=30,
            v_shadow_max=115,
            v_high_guard=255,
            green_h=(35, 95),
            use_bottom_roi=True,
            bottom_roi_ratio=0.80,
            min_cc_area=200,
            open_ksize=3,
            close_ksize=8
        ):
        hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)
        Hh, Ss, Vv = cv2.split(hsv)
        L, Aa, Bb = cv2.split(lab)
        if blur_ksize and blur_ksize > 1:
            Aa = cv2.medianBlur(Aa, blur_ksize)
        H, W = Aa.shape
        fg = (Vv >= v_bg)
        a_fg = Aa[fg]
        if a_fg.size < 100:
            return 123, np.zeros((H, W), np.uint8)
        hist, _ = np.histogram(a_fg, bins=256, range=(0,255))
        p = hist.astype(np.float64); p /= (p.sum() + 1e-12)
        omega = np.cumsum(p)
        mu = np.cumsum(p * np.arange(256))
        mu_t = mu[-1]
        sigma_b2 = (mu_t*omega - mu)**2 / (omega*(1.0-omega) + 1e-12)
        t = int(np.nanargmax(sigma_b2))
        mask = (Aa <= t) if object_type=="dark" else (Aa >= t)
        mask &= fg
        shadow_gray = (Ss <= s_min) & (Vv <= v_shadow_max)
        mask &= ~shadow_gray
        h0, h1 = green_h
        green_band = (Hh >= h0) & (Hh <= h1)
        mask &= green_band
        if use_bottom_roi:
            y0 = int((1.0 - bottom_roi_ratio) * H)
            roi = np.zeros((H, W), bool); roi[y0:H, :] = True
            mask &= roi
        mask_u8 = (mask.astype(np.uint8) * 255)
        if open_ksize > 1:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, k, iterations=1)
        if close_ksize > 1:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, k, iterations=2)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
        cleaned = np.zeros_like(mask_u8)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_cc_area:
                cleaned[labels == i] = 255
        return t, cleaned

def odd_from_slider(v: int, min_odd: int = 1) -> int:
    v = max(min_odd, v)
    return v if v % 2 == 1 else v + 1

def overlay_image(img_bgr, mask_u8, alpha=0.5):
    # สีเขียวสำหรับ mask
    color = np.zeros_like(img_bgr)
    color[:, :, 1] = 255
    mask_bool = mask_u8 > 0
    out = img_bgr.copy()
    out[mask_bool] = cv2.addWeighted(out[mask_bool], (1-alpha), color[mask_bool], alpha, 0)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True, help="path ภาพต้นฉบับ (BGR)")
    ap.add_argument("--save_dir", default=".", help="โฟลเดอร์เซฟผลลัพธ์")
    args = ap.parse_args()

    bgr = cv2.imread(args.img, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"อ่านรูปไม่ได้: {args.img}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    base = os.path.splitext(os.path.basename(args.img))[0]

    cv2.namedWindow("original", cv2.WINDOW_NORMAL)
    cv2.namedWindow("mask", cv2.WINDOW_NORMAL)
    cv2.namedWindow("overlay", cv2.WINDOW_NORMAL)
    cv2.namedWindow("controls", cv2.WINDOW_NORMAL)

    # ===== Trackbars =====
    # Green hue range
    cv2.createTrackbar("H_low", "controls", 40, 179, lambda x: None)
    cv2.createTrackbar("H_high", "controls", 80, 179, lambda x: None)
    # Saturation/Value guards (สำหรับเลือกใบที่ “สด”)
    cv2.createTrackbar("S_min_green", "controls", 60, 255, lambda x: None)
    cv2.createTrackbar("V_min_green", "controls", 50, 255, lambda x: None)

    # Shadow/gray guard & background guard
    cv2.createTrackbar("S_min_shadow", "controls", 40, 255, lambda x: None)     # เดิมในไฟล์ 30
    cv2.createTrackbar("V_shadow_max", "controls", 130, 255, lambda x: None)    # เดิมในไฟล์ 115
    cv2.createTrackbar("V_bg_min", "controls", 45, 255, lambda x: None)

    # Blur/Morph/Area
    cv2.createTrackbar("Blur_k", "controls", 3, 15, lambda x: None)             # odd
    cv2.createTrackbar("Open_k", "controls", 3, 21, lambda x: None)             # odd
    cv2.createTrackbar("Close_k", "controls", 8, 21, lambda x: None)            # odd
    cv2.createTrackbar("Min_cc_area", "controls", 200, 20000, lambda x: None)

    # Bottom ROI
    cv2.createTrackbar("Use_bottom_roi", "controls", 1, 1, lambda x: None)
    cv2.createTrackbar("Bottom_roi_%", "controls", 80, 100, lambda x: None)     # 0–100

    # Object type: 0=dark (Aa<=t), 1=light (Aa>=t)
    cv2.createTrackbar("Object_type", "controls", 0, 1, lambda x: None)

    # Preview scale (เพื่อดูไวขึ้น)
    cv2.createTrackbar("Preview_%", "controls", 60, 100, lambda x: None)

    while True:
        # อ่านค่า
        h0 = cv2.getTrackbarPos("H_low", "controls")
        h1 = cv2.getTrackbarPos("H_high", "controls")
        if h1 < h0: h1 = h0

        s_min_green = cv2.getTrackbarPos("S_min_green", "controls")
        v_min_green = cv2.getTrackbarPos("V_min_green", "controls")

        s_min_shadow = cv2.getTrackbarPos("S_min_shadow", "controls")
        v_shadow_max = cv2.getTrackbarPos("V_shadow_max", "controls")
        v_bg_min = cv2.getTrackbarPos("V_bg_min", "controls")

        blur_k = odd_from_slider(cv2.getTrackbarPos("Blur_k", "controls"))
        open_k = odd_from_slider(cv2.getTrackbarPos("Open_k", "controls"))
        close_k = odd_from_slider(cv2.getTrackbarPos("Close_k", "controls"))
        min_cc_area = cv2.getTrackbarPos("Min_cc_area", "controls")

        use_bottom_roi = cv2.getTrackbarPos("Use_bottom_roi", "controls") == 1
        bottom_roi_pct = cv2.getTrackbarPos("Bottom_roi_%", "controls")
        bottom_roi_ratio = max(0.0, min(1.0, bottom_roi_pct/100.0))

        obj_type = "dark" if cv2.getTrackbarPos("Object_type", "controls") == 0 else "light"
        preview_pct = max(10, cv2.getTrackbarPos("Preview_%", "controls"))
        scale = preview_pct / 100.0

        # เรียกฟังก์ชันหลัก (เพิ่ม S/V guard ต่อจาก green_h ภายนอก)
        t, mask_u8 = auto_thresh_lab_a_otsu_guard(
            rgb_img=rgb,
            object_type=obj_type,
            v_bg=v_bg_min,
            blur_ksize=blur_k,
            s_min=s_min_shadow,
            v_shadow_max=v_shadow_max,
            green_h=(h0, h1),
            use_bottom_roi=use_bottom_roi,
            bottom_roi_ratio=bottom_roi_ratio,
            min_cc_area=min_cc_area,
            open_ksize=open_k,
            close_ksize=close_k
        )

        # ===== เพิ่ม “Green S/V guard” ภายนอก (เข้มขึ้นอีกชั้น) =====
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        Hh, Ss, Vv = cv2.split(hsv)
        green_sv_guard = ((Ss >= s_min_green) & (Vv >= v_min_green)).astype(np.uint8)*255
        mask_u8 = cv2.bitwise_and(mask_u8, green_sv_guard)

        # ===== Overlay =====
        overlay = overlay_image(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), mask_u8, alpha=0.45)

        # Resize preview
        if scale != 1.0:
            h, w = mask_u8.shape
            mask_show = cv2.resize(mask_u8, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_NEAREST)
            orig_show = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (int(w*scale), int(h*scale)))
            overlay_show = cv2.resize(overlay, (int(w*scale), int(h*scale)))
        else:
            mask_show = mask_u8
            orig_show = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            overlay_show = overlay

        cv2.imshow("original", orig_show)
        cv2.imshow("mask", mask_show)
        cv2.imshow("overlay", overlay_show)

        key = cv2.waitKey(20) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            os.makedirs(args.save_dir, exist_ok=True)
            p_mask = os.path.join(args.save_dir, f"{base}_mask.png")
            p_ovr  = os.path.join(args.save_dir, f"{base}_overlay.png")
            cv2.imwrite(p_mask, mask_u8)
            cv2.imwrite(p_ovr, overlay)
            print(f"[SAVE] {p_mask}")
            print(f"[SAVE] {p_ovr}")
        elif key == ord('p'):
            os.makedirs(args.save_dir, exist_ok=True)
            params = {
                "object_type": obj_type,
                "v_bg": v_bg_min,
                "blur_ksize": blur_k,
                "s_min_shadow": s_min_shadow,
                "v_shadow_max": v_shadow_max,
                "green_h": [h0, h1],
                "s_min_green": s_min_green,
                "v_min_green": v_min_green,
                "use_bottom_roi": use_bottom_roi,
                "bottom_roi_ratio": bottom_roi_ratio,
                "min_cc_area": min_cc_area,
                "open_ksize": open_k,
                "close_ksize": close_k,
                "otsu_t_on_A": int(t)
            }
            p_json = os.path.join(args.save_dir, f"{base}_params.json")
            with open(p_json, "w", encoding="utf-8") as f:
                json.dump(params, f, ensure_ascii=False, indent=2)
            print(f"[SAVE] {p_json}")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
