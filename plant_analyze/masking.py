import cv2
import numpy as np 
from plantcv import plantcv as pcv

'''Auto-select a mask from an RGB image using various grayscale channels and thresholding methods.'''
def ensure_binary(mask):
    if mask is None:
        return None
    m = np.asarray(mask)
    if m.dtype != np.uint8:
        m = m.astype(np.uint8)
        
    m = np.where(m > 0, 255, 0).astype(np.uint8)
    return m

def _largest_component(bin_img):
    bin_img = ensure_binary(bin_img)
    if bin_img is None or cv2.countNonZero(bin_img) == 0:
        return bin_img
    _fc = cv2.findContours(bin_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(_fc) == 3:
        _, cnts, _ = _fc
    else:
        cnts, _ = _fc
    if not cnts:
        return bin_img
    idx = int(np.argmax([cv2.contourArea(c) for c in cnts]))
    out = np.zeros_like(bin_img)
    cv2.drawContours(out, [cnts[idx]], -1, 255, thickness=cv2.FILLED)
    return ensure_binary(out)

def clean_mask(m, close_ksize=7, min_obj_size=120):
    if m is None:
        return m
    m = ensure_binary(m)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)
    m = _largest_component(m)
    m = pcv.fill(bin_img=m, size=min_obj_size)
    m = ensure_binary(m)
    return m
def auto_select_mask(rgb_img):
    H, W = rgb_img.shape[:2]
    area_total = H * W

    chans = []
    try:  chans.append(("lab_a", pcv.rgb2gray_lab(rgb_img=rgb_img, channel='a')))
    except: pass
    try:  chans.append(("lab_b", pcv.rgb2gray_lab(rgb_img=rgb_img, channel='b')))
    except: pass
    try:  chans.append(("lab_l", pcv.rgb2gray_lab(rgb_img=rgb_img, channel='l')))
    except: pass
    try:  chans.append(("hsv_v", pcv.rgb2gray_hsv(rgb_img=rgb_img, channel='v')))
    except: pass
    try:  chans.append(("hsv_s", pcv.rgb2gray_hsv(rgb_img=rgb_img, channel='s')))
    except: pass
    if not chans:
        raise RuntimeError("No grayscale channels available for auto selection.")

    methods = []
    for name, g in chans:
        try:
            g = pcv.transform.rescale(gray_img=g, lower=0, upper=255)
        except Exception:
            pass
        methods += [
            (name, "otsu",     {"object_type": "dark"}),
            (name, "otsu",     {"object_type": "light"}),
            (name, "triangle", {"object_type": "dark"}),
            (name, "triangle", {"object_type": "light"}),
        ]
        for k in (31, 51, 101):
            methods += [(name, ("gaussian", k, 0, "dark"),  {}),
                        (name, ("gaussian", k, 0, "light"), {})]

    candidates = []
    for name, meth, params in methods:
        try:
            gray = next(g for nm, g in chans if nm == name)
            if meth == "otsu":
                m = pcv.threshold.otsu(gray_img=gray, **params)
            elif meth == "triangle":
                m = pcv.threshold.triangle(gray_img=gray, **params)
            else:
                _, k, off, obj = meth
                m = pcv.threshold.gaussian(gray_img=gray, ksize=k, offset=off, object_type=obj)
                
            m = ensure_binary(m)

            # findContours แบบ robust (OpenCV3/4)
            _fc = cv2.findContours(m.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(_fc) == 3:
                _, contours, _hier = _fc
            else:
                contours, _hier = _fc

            n_comp = len(contours)
            if n_comp > 0:
                areas = [cv2.contourArea(c) for c in contours]
                cnt = contours[int(np.argmax(areas))]
                hull = cv2.convexHull(cnt)
                a_obj  = float(cv2.contourArea(cnt))
                a_hull = float(cv2.contourArea(hull))
                solidity = (a_obj / a_hull) if a_hull > 0 else 0.0
            else:
                solidity = 0.0

            area = int(np.count_nonzero(m))
            ratio = area / max(area_total, 1)

            def score_area(r):
                if 0.5 <= r <= 0.6:
                    return 2.0 - abs(0.30 - r)
                return 0.5 - abs(0.30 - r)
            s = score_area(ratio) - 0.1 * max(0, n_comp - 1) + 0.5 * solidity

            candidates.append((s, name, m, ratio, n_comp, solidity))
        except Exception:
            continue

    if not candidates:
        fallback = pcv.threshold.otsu(gray_img=pcv.rgb2gray_lab(rgb_img=rgb_img, channel='a'), object_type='dark')
        fallback = ensure_binary(fallback)
        return fallback, {"channel":"lab_a","method":"otsu","object_type":"dark","ksize":None,
                          "area_ratio": float(np.count_nonzero(fallback))/max(area_total,1),
                          "n_components": 0, "solidity": 0.0}

    best = max(candidates, key=lambda x: x[0])
    _, name, mask, ratio, n_comp, solidity = best
    return ensure_binary(mask), {
        "channel": name,
        "method": meth,
        "object_type": obj,
        "ksize": k,
        "area_ratio": float(ratio),
        "n_components": int(n_comp),
        "solidity": float(solidity),
    }