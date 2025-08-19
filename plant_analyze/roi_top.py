
import numpy as np
import cv2

def _circle_contour(cx, cy, r, n_pts=96):
    # สร้างคอนทัวร์วงกลมด้วยจุดโพลิกอน
    pts = cv2.ellipse2Poly((int(cx), int(cy)), (int(r), int(r)), 0, 0, 360, max(1, 360 // n_pts))
    return pts.reshape(-1, 1, 2)

def _rect_contour(x, y, w, h):
    x, y, w, h = map(int, (x, y, w, h))
    cnt = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=np.int32)
    return cnt.reshape(-1, 1, 2)

def make_grid_rois(rgb_img, rows, cols, roi_radius=None, roi_shape="circle", margin=0.05):
    """
    สร้างรายการ ROI เป็นคอนทัวร์ (list of contours) แบบกริดขนาด rows x cols
    โดยไม่เรียกใช้ PlantCV ใด ๆ เพื่อตัดปัญหา 'Input binary mask is not binary!'

    Parameters
    ----------
    rgb_img : np.ndarray
        ภาพต้นฉบับ (ใช้แค่ขนาด H,W)
    rows, cols : int
        จำนวนแถว/คอลัมน์ในกริด
    roi_radius : float|None
        รัศมี ROI ถ้าไม่กำหนด จะคำนวณอัตโนมัติตามขนาดภาพ
    roi_shape : str
        "circle" หรือ "rect"
    margin : float
        เว้นขอบภาพเป็นสัดส่วน เช่น 0.05 = 5%

    Returns
    -------
    rois : list[np.ndarray]
        รายการคอนทัวร์ของ ROI แต่ละช่อง (เหมาะกับ cv2.drawContours)
    eff_r : int
        รัศมีที่ใช้จริง (มีประโยชน์เวลาคุณส่งไปคิด coverage)
    """
    H, W = rgb_img.shape[:2]
    # พื้นที่ใช้งาน หลังหัก margin
    mx = int(W * margin)
    my = int(H * margin)
    x0, y0 = mx, my
    x1, y1 = W - mx, H - my
    avail_w = max(1, x1 - x0)
    avail_h = max(1, y1 - y0)

    # ระยะห่างจุดศูนย์กลางแต่ละช่อง
    step_x = avail_w / float(cols)
    step_y = avail_h / float(rows)

    # คำนวณรัศมีอัตโนมัติถ้าไม่กำหนด
    if roi_radius is None:
        # ครึ่งหนึ่งของครึ่งช่อง (เผื่อระยะซ้อน) แล้วเอาน้อยสุดของแกน X/Y
        auto_r = 0.45 * min(step_x, step_y)
        eff_r = int(max(2, round(auto_r)))
    else:
        eff_r = int(max(2, round(float(roi_radius))))

    rois = []
    for r in range(rows):
        cy = y0 + (r + 0.5) * step_y
        for c in range(cols):
            cx = x0 + (c + 0.5) * step_x
            if roi_shape == "rect":
                # สร้างสี่เหลี่ยมขนาด 2r x 2r
                cnt = _rect_contour(cx - eff_r, cy - eff_r, 2 * eff_r, 2 * eff_r)
            else:
                # วงกลม (ค่าเริ่มต้น)
                cnt = _circle_contour(cx, cy, eff_r)
            rois.append(cnt.astype(np.int32))
    return rois, eff_r
