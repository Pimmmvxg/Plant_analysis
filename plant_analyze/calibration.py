import cv2
import numpy as np
from pathlib import Path
from . import config as cfg
from plantcv import plantcv as pcv

def get_scale_from_checkerboard(
    image,
    square_size_mm=2.5,
    pattern_size=(7, 7),
    previous_scale=None,
    fallback_scale=10.0 / 51.0,
    refine=True,
    save_debug=True,
    debug_name="checkerboard_scale"
):
    """
    คืน mm_per_px จาก checkerboard + flag ว่าพบหรือไม่ + ข้อความอธิบาย
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)

    if found and refine:
        # ปรับจุดให้คมขึ้น
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
        cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), term)

    if found:
        # คำนวณระยะต่อช่องแบบ robust: เฉลี่ยทั้งแนว x และแนว y
        w, h = pattern_size
        # แนวนอน: แต่ละแถว มี (w-1) ช่อง
        horiz = []
        for r in range(h):
            for c in range(w - 1):
                p1 = corners[r * w + c][0]
                p2 = corners[r * w + (c + 1)][0]
                horiz.append(np.linalg.norm(p1 - p2))
        # แนวตั้ง: แต่ละคอลัมน์ มี (h-1) ช่อง
        vert = []
        for c in range(w):
            for r in range(h - 1):
                p1 = corners[r * w + c][0]
                p2 = corners[(r + 1) * w + c][0]
                vert.append(np.linalg.norm(p1 - p2))

        dpx = float(np.median(horiz + vert)) if (horiz or vert) else None
        if dpx and dpx > 0:
            scale = float(square_size_mm) / dpx   # mm/px
            scale_info = "scale from checkerboard"
            # debug image (optional)
            if save_debug and (pcv.params.debug_outdir or getattr(cfg, "OUTPUT_DIR", None)):
                dbg = image.copy()
                cv2.drawChessboardCorners(dbg, pattern_size, corners, True)
                base = getattr(cfg, "OUTPUT_DIR", None) or pcv.params.debug_outdir or "."
                Path(base, "processed").mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(Path(base, "processed", f"{debug_name}.png")), dbg)
            return scale, True, scale_info

    # ไม่เจอ checkerboard → ใช้ previous หรือ fallback
    if previous_scale is not None and previous_scale > 0:
        return float(previous_scale), False, "using previous scale"
    return float(fallback_scale), False, "fallback scale"
