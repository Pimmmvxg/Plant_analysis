import cv2, os, argparse
import numpy as np

def connect_mask_holes(m, gap_x=0, gap_y=0, iterations=1):
    m = (m>0).astype(np.uint8)*255
    out = m.copy()
    for _ in range(max(1, int(iterations))):
        if gap_x>0:
            kx = cv2.getStructuringElement(cv2.MORPH_RECT, (2*gap_x+1,1))
            out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kx)
        if gap_y>0:
            ky = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2*gap_y+1))
            out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, ky)
    return (out>0).astype(np.uint8)*255

def to_mask_from_image(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    V = hsv[...,2]
    m = (V>40).astype(np.uint8)*255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, k)

def overlay_diff(b, a):
    b = (b>0).astype(np.uint8); a = (a>0).astype(np.uint8)
    o = np.zeros((*b.shape,3), np.uint8)
    o[...,1] = b*255              # เขียว = เดิม
    o[...,2] = ((a==1)&(b==0))*255# แดง = เติมใหม่
    return o

def synthetic_mask(h=350,w=600):
    m = np.zeros((h,w),np.uint8)
    cv2.rectangle(m,(60,200),(220,340),255,-1)
    cv2.rectangle(m,(300,210),(460,350),255,-1)
    cv2.rectangle(m,(220,250),(300,300),0,-1)
    return m

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default=None)
    ap.add_argument("--gapx", type=int, default=30)
    ap.add_argument("--gapy", type=int, default=80)
    ap.add_argument("--iters", type=int, default=1)
    ap.add_argument("--outdir", type=str, default="_bridge_tests")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if args.input and os.path.exists(args.input):
        img = cv2.imread(args.input, cv2.IMREAD_COLOR)
        m0 = to_mask_from_image(img)
        stem = os.path.splitext(os.path.basename(args.input))[0]
    else:
        m0 = synthetic_mask()
        stem = "synthetic"

    before = int(cv2.countNonZero(m0))
    m1 = connect_mask_holes(m0, args.gapx, args.gapy, args.iters)
    after = int(cv2.countNonZero(m1))
    ov = overlay_diff(m0, m1)

    pref = os.path.join(args.outdir, f"{stem}_gx{args.gapx}_gy{args.gapy}_it{args.iters}")
    cv2.imwrite(pref+"_before.png", m0)
    cv2.imwrite(pref+"_after.png",  m1)
    cv2.imwrite(pref+"_overlay.png", ov)

    print(f"[bridge] area {before} -> {after} (Δ={after-before}), saved -> {pref}_*.png")    