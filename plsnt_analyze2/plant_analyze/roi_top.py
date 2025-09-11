# roi_top.py (เวอร์ชันแก้แล้ว)
import cv2
import numpy as np

def make_grid_rois(rgb_img, rows, cols, roi_radius=None, margin=0.05, shape="circle"):
    """
    สร้างกริด ROI (rows x cols) แล้วคืนเป็น list ของคอนทัวร์ (ใช้กับ cv2.drawContours)
    - วางศูนย์กลางกลางเซลล์ (offset 0.5)
    - clamp รัศมีตามระยะจาก "ศูนย์กลางสุดขอบ" ถึงขอบภาพ
    """
    H, W = rgb_img.shape[:2]
    if rows <= 0 or cols <= 0:
        raise ValueError("rows/cols must be > 0")

    # กรอบใช้งานหลังหัก margin
    mx, my = int(W * margin), int(H * margin)
    x0, y0 = mx, my
    x1, y1 = W - mx - 1, H - my - 1

    # ขนาดเซลล์ (กว้าง/สูง) = แบ่งพื้นที่ใช้งานด้วยจำนวนคอลัมน์/แถว
    DX = (x1 - x0) / float(cols)   # << ต่างจากเดิม: หารด้วย cols (ไม่ใช่ cols-1)
    DY = (y1 - y0) / float(rows)  # << ต่างจากเดิม: หารด้วย rows

    # ศูนย์กลาง "สุดขอบ" เมื่อวางกลางเซลล์
    cx_min = x0 + 0.5 * DX
    cx_max = x0 + (cols - 0.5) * DX
    cy_min = y0 + 0.5 * DY
    cy_max = y0 + (rows - 0.5) * DY

    # รัศมีตั้งต้น
    if roi_radius is None:
        eff_r = int(max(2, round(0.45 * min(DX, DY))))
    else:
        eff_r = int(max(2, round(float(roi_radius))))

    # clamp รัศมีตาม "ระยะห่างของศูนย์กลางสุดขอบ" ถึงขอบภาพ
    eff_r = int(min(
        eff_r,
        cx_min,                 # ระยะจากศูนย์กลางซ้ายสุดถึงขอบซ้าย
        (W - 1) - cx_max,       # ระยะจากศูนย์กลางขวาสุดถึงขอบขวา
        cy_min,                 # ระยะจากศูนย์กลางบนสุดถึงขอบบน
        (H - 1) - cy_max        # ระยะจากศูนย์กลางล่างสุดถึงขอบล่าง
    ))
    if eff_r <= 0:
        raise ValueError("ROI grid does not fit the image. Adjust rows/cols/margin/radius.")

    # สร้างคอนทัวร์ ROI จากศูนย์กลาง "กลางเซลล์"
    rois = []
    for r in range(rows):
        cy = y0 + (r + 0.5) * DY
        for c in range(cols):
            cx = x0 + (c + 0.5) * DX
            if shape == "rect":
                half = eff_r
                cnt = np.array(
                    [[cx - half, cy - half],
                     [cx + half, cy - half],
                     [cx + half, cy + half],
                     [cx - half, cy + half]], dtype=np.int32
                ).reshape(-1, 1, 2)
            else:
                pts = cv2.ellipse2Poly((int(round(cx)), int(round(cy))),
                                       (eff_r, eff_r), 0, 0, 360, max(1, 360 // 96))
                cnt = pts.reshape(-1, 1, 2).astype(np.int32)
            rois.append(cnt)

    return rois, eff_r
