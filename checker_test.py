import cv2, numpy as np, math, json
from pathlib import Path
from statistics import median

IMG_PATH = r"R:\01-Organize\01-Management\01-Data Center\Brisk\06-AI & Machine Learning (D0340)\04-IOT_Smartfarm\picture_original_sideview_smartfarm\picture_sideview_F_11112025_160434.jpg"  
RECT_W_MM = 48.0   # กว้างจริงของสี่เหลี่ยม (mm)  <-- แก้ให้ตรงของจริง
RECT_H_MM = 48.0   # สูงจริงของสี่เหลี่ยม (mm)   <-- แก้ให้ตรงของจริง

OUT_JSON = "mm_per_px.json"

# ---- พารามิเตอร์จูนเล็กน้อย ----
CROP_TOP_RATIO = 0.80      # ใช้เฉพาะส่วนบนของภาพ 
MIN_AREA = 20000           # พื้นที่คอนทัวร์ขั้นต่ำ (px^2) ปรับตามขนาดสี่เหลี่ยมในภาพ
EPS_FRACTION = 0.04       # epsilon ของ approxPolyDP 
RECT_TOL = 0.6            # ยอมให้ aspect ratio เพี้ยนจากของจริงได้ 
MIN_RECTANGULARITY = 0.3  # area / (w*h) ขั้นต่ำ
MAX_AREA = 1000000          # กรองคอนทัวร์ที่ใหญ่เกินไป (px^2)

def order_box(pts4):
    # รับ 4 จุด (x,y) -> เรียงเป็น TL, TR, BR, BL
    pts = np.array(pts4, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]; br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]; bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def side_lengths(box4):
    tl, tr, br, bl = box4
    w1 = np.linalg.norm(tr - tl)
    w2 = np.linalg.norm(br - bl)
    h1 = np.linalg.norm(bl - tl)
    h2 = np.linalg.norm(br - tr)
    width_px  = 0.5 * (w1 + w2)
    height_px = 0.5 * (h1 + h2)
    return width_px, height_px

def compute_scale(width_px, height_px, area_px):
    # จับคู่ด้านยาว/สั้นของภาพกับ mm จริงอัตโนมัติ
    long_px, short_px = (width_px, height_px) if width_px >= height_px else (height_px, width_px)
    long_mm, short_mm = (max(RECT_W_MM, RECT_H_MM), min(RECT_W_MM, RECT_H_MM))

    s_long  = long_mm  / long_px
    s_short = short_mm / short_px
    s_area  = math.sqrt((RECT_W_MM * RECT_H_MM) / area_px)  # จากพื้นที่
    s_final = median([s_long, s_short, s_area])
    return s_final, {"mm_per_px_long": s_long, "mm_per_px_short": s_short, "mm_per_px_area": s_area}

def main():
    img = cv2.imread(IMG_PATH)
    if img is None:
        raise FileNotFoundError(IMG_PATH)
    H, W = img.shape[:2]
    roi = img[:int(H*CROP_TOP_RATIO)].copy()
    # 1) L-channel + CLAHE + threshold (พื้นหลังดำ -> วัตถุสว่าง)
    L = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)[:, :, 0]
    L = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(L)
    _, th = cv2.threshold(L, 120, 255, cv2.THRESH_BINARY)

    # clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    #th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, 1)
    #th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, 1)

    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_score = -1.0
    target_aspect = max(RECT_W_MM, RECT_H_MM) / min(RECT_W_MM, RECT_H_MM)

    for c in cnts:
        area = cv2.contourArea(c)
        if area < MIN_AREA:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, EPS_FRACTION * peri, True)
        if len(approx) != 4:
            continue
        if not cv2.isContourConvex(approx):
            continue

        # กล่องหมุนได้
        rect = cv2.minAreaRect(c)  # (center,(w,h),angle)
        (rw, rh) = rect[1]
        if rw <= 1 or rh <= 1:
            continue
        rect_area = rw * rh
        rectangularity = float(area) / float(rect_area + 1e-6)
        if rectangularity < MIN_RECTANGULARITY:
            continue

        # เช็คอัตราส่วน
        aspect = max(rw, rh) / min(rw, rh)
        ratio_err = abs(aspect - target_aspect) / target_aspect
        if ratio_err > RECT_TOL:
            continue

        # ให้คะแนน: ยิ่งสี่เหลี่ยมชัด/อัตราส่วนใกล้จริง/พื้นที่ใหญ่ → คะแนนสูง
        score = rectangularity * (1.0 - ratio_err) * math.sqrt(area)
        if score > best_score:
            best_score = score
            box = cv2.boxPoints(rect)
            box = order_box(box)
            best = {"box": box, "area": area, "rectangularity": rectangularity,
                    "aspect": aspect, "ratio_err": ratio_err}

    if best is None:
        print("[FAIL] ไม่พบสี่เหลี่ยมอ้างอิง ลองเพิ่ม/ลด MIN_AREA, EPS_FRACTION หรือปรับแสงให้คมขึ้น")
        return

    # 2) คำนวณ mm/px
    width_px, height_px = side_lengths(best["box"])
    s_final, parts = compute_scale(width_px, height_px, best["area"])

    print(f"width_px≈{width_px:.2f}, height_px≈{height_px:.2f}, area_px≈{best['area']:.1f}")
    print(f"aspect_px≈{best['aspect']:.3f} (target≈{target_aspect:.3f}, err≈{best['ratio_err']*100:.1f}%)")
    print(f"mm/px  -> final={s_final:.6f}  (long={parts['mm_per_px_long']:.6f}, "
          f"short={parts['mm_per_px_short']:.6f}, area={parts['mm_per_px_area']:.6f})")

    # 3) เซฟผลไว้ใช้ใน pipeline
    payload = {
    "mm_per_px": float(s_final),
    "mm_per_px_long": float(parts["mm_per_px_long"]),
    "mm_per_px_short": float(parts["mm_per_px_short"]),
    "mm_per_px_area": float(parts["mm_per_px_area"]),
    "rect_w_mm": float(RECT_W_MM),
    "rect_h_mm": float(RECT_H_MM),
    "source_img": str(IMG_PATH)
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[OK] saved -> {Path(OUT_JSON).resolve()}")

    # 4) วาด overlay ให้ตรวจสอบ
    vis = img.copy()
    tl, tr, br, bl = best["box"].astype(int)
    cv2.polylines(vis, [np.array([tl,tr,br,bl])], True, (0,255,255), 2)
    txt = f"mm/px={s_final:.6f}"
    cv2.rectangle(vis, (10,10), (10+len(txt)*9, 40), (0,0,0), -1)
    cv2.putText(vis, txt, (14,32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
    vis = cv2.resize(vis, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA)

    cv2.imshow("Rect scale", vis); cv2.waitKey(0); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
