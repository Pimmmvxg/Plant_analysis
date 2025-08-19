import cv2
import numpy as np 
from plantcv import plantcv as pcv

def ensure_binary(mask):
    if mask is None:
        return None
    m = np.asarray(mask)
    if m.dtype != np.uint8:
        m = m.astype(np.uint8)
    # บังคับให้มีแค่ 0/255
    m = np.where(m > 0, 255, 0).astype(np.uint8)
    return m

def _keep_top_k_components(m: np.ndarray, k: int) -> np.ndarray:
    #เก็บ k ก้อนใหญ่ที่สุด
    m = ensure_binary(m)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num <= 1:
        return m
    # stats[:, cv2.CC_STAT_AREA] = area ของแต่ละ label (label 0 = background)
    areas = stats[1:, cv2.CC_STAT_AREA]
    order = np.argsort(areas)[::-1]  # ใหญ่ → เล็ก
    keep = order[:k] + 1            # ขยับ +1 เพราะเราตัด background ออกแล้ว
    out = np.zeros_like(m)
    for lab in keep:
        out[labels == lab] = 255
    return ensure_binary(out)

def clean_mask(m, close_ksize=7, min_obj_size=120, keep_largest=False, keep_top_k=None):
    """
    ทำความสะอาดมาสก์:
    - ปิดรู/เชื่อมช่องว่างด้วย morphological close (ขนาด kernel = close_ksize)
    - กรองสิ่งเล็กกว่า min_obj_size ออก (ด้วย pcv.fill)
    - เลือกคงเฉพาะก้อนใหญ่สุด หรือ k ก้อนใหญ่สุดได้ (ถ้าต้องการ)
    """
    if m is None:
        return m
    m = ensure_binary(m)

    # 1) close
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(close_ksize), int(close_ksize)))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)

    # 2) ตัดสิ่งเล็ก ๆ/อุดรู (ปรับเกณฑ์ได้ด้วย min_obj_size)
    m = pcv.fill(bin_img=m, size=int(min_obj_size))

    # 3) (ทางเลือก) คงเฉพาะก้อนใหญ่สุด หรือ k ก้อนใหญ่สุด
    if keep_top_k is not None and int(keep_top_k) >= 1:
        m = _keep_top_k_components(m, int(keep_top_k))
    elif keep_largest:
        m = _keep_top_k_components(m, 1)

    return ensure_binary(m)

def auto_select_mask(rgb_img):
    H, W = rgb_img.shape[:2]
    area_total = H * W

    # 1) เตรียมช่องสี
    chans = []
    for nm, fn, kw in [
        ("lab_a", pcv.rgb2gray_lab, {'channel': 'a'}),
        ("lab_b", pcv.rgb2gray_lab, {'channel': 'b'}),
        ("lab_l", pcv.rgb2gray_lab, {'channel': 'l'}),
        ("hsv_v", pcv.rgb2gray_hsv, {'channel': 'v'}),
        ("hsv_s", pcv.rgb2gray_hsv, {'channel': 's'}),
    ]:
        try:
            g = fn(rgb_img=rgb_img, **kw)
            try:
                g = pcv.transform.rescale(gray_img=g, min_value=0, max_value=255)
                if g.dtype != np.uint8:
                    g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)
                    g = g.astype(np.uint8)

            except Exception:
                pass
            chans.append((nm, g))
        except Exception:
            pass

    if not chans:
        raise RuntimeError("No grayscale channels available for auto selection.")

    # 2) เตรียมชุด threshold ที่จะลอง
    methods = []
    for name, _g in chans:
        methods += [
            (name, ("otsu",     None,  None, "dark")),
            (name, ("otsu",     None,  None, "light")),
            (name, ("triangle", None,  None, "dark")),
            (name, ("triangle", None,  None, "light")),
        ]
        for ksz in (31, 51, 101):
            methods += [
                (name, ("gaussian", ksz, 0, "dark")),
                (name, ("gaussian", ksz, 0, "light")),
            ]

    # 3) ลองทุก candidate + ให้คะแนน + เก็บ metadata (method,obj_type,ksize)
    candidates = []
    for name, spec in methods:
        try:
            gray = next(g for nm, g in chans if nm == name)
            kind, ksz, off, obj_type = spec

            if kind == "otsu":
                m = pcv.threshold.otsu(gray_img=gray, object_type=obj_type)
            elif kind == "triangle":
                m = pcv.threshold.triangle(gray_img=gray, object_type=obj_type)
            else:  # gaussian
                m = pcv.threshold.gaussian(gray_img=gray, ksize=ksz, offset=off, object_type=obj_type)

            m = ensure_binary(m)

            _fc = cv2.findContours(m.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = _fc[2] if len(_fc) == 3 else _fc[0]
            n_comp = len(contours)
            if n_comp > 0:
                areas = [cv2.contourArea(c) for c in contours]
                cnt = contours[int(np.argmax(areas))]
                hull = cv2.convexHull(cnt)
                a_obj  = float(cv2.contourArea(cnt))
                a_hull = float(cv2.contourArea(hull)) if hull is not None else 0.0
                solidity = (a_obj / a_hull) if a_hull > 0 else 0.0
            else:
                solidity = 0.0

            ratio = int(np.count_nonzero(m)) / max(area_total, 1)

            def score_area(r):
                return 1.0 - abs(0.3 - r)

            score = score_area(ratio) - 0.1 * max(0, n_comp - 1) + 0.5 * solidity
            # เก็บ method meta ไว้ใน candidate ด้วย
            candidates.append((score, name, m, ratio, n_comp, solidity, kind, obj_type, ksz))
        except Exception:
            continue

    # 4) เลือกตัวที่ดีที่สุด + คืน metadata ใ
    if not candidates:
        fb = pcv.threshold.otsu(gray_img=pcv.rgb2gray_lab(rgb_img=rgb_img, channel='a'), object_type='dark')
        fb = ensure_binary(fb)
        return fb, {
            "channel": "lab_a", "method": "otsu", "object_type": "dark", "ksize": None,
            "area_ratio": float(np.count_nonzero(fb)) / max(area_total, 1),
            "n_components": 0, "solidity": 0.0
        }

    best = max(candidates, key=lambda x: x[0])
    _, name, mask, ratio, n_comp, solidity, kind, obj_type, ksz = best
    return ensure_binary(mask), {
        "channel": name,
        "method": kind,
        "object_type": obj_type,
        "ksize": ksz,
        "area_ratio": float(ratio),
        "n_components": int(n_comp),
        "solidity": float(solidity),
    }
