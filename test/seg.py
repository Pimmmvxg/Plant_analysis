import cv2, numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_multiotsu
from pathlib import Path

IMG_PATH = r"R:\01-Organize\01-Management\01-Data Center\Brisk\06-AI & Machine Learning (D0340)\04-IOT_Smartfarm\picture_original_topview_smartfarm\picture_topview_09092025_134232.jpg"  

# ดัชนีที่จะใช้กับ Otsu: "exg" หรือ "vari"
INDEX_NAME = "exg"

# Multi-Otsu
CLASSES = 4                 # 3 หรือ 4 (ใบสว่างมาก แนะนำ 4)
SMOOTH_SIGMA = 1.5          # 0=ปิด, 1.2–2.0 ช่วยให้ฮิสโตแกรมนิ่งขึ้น

# รวมคลาสกลางเมื่อ CLASSES=4 (กันใบสว่าง/ใบมืดแยกชั้น)
MERGE_MIDDLE_IF_4 = False

# จัดการไฮไลต์ก่อน Otsu (ลดอิทธิพล pixel S ต่ำ + V สูง)
HANDLE_HIGHLIGHT = True
SPEC_S_MAX, SPEC_V_MIN = 30, 242   # ยิ่งเข้มงวด → S_MAX สูงขึ้น, V_MIN สูงขึ้น

AUTO_PICK_CLASS = True
S_MIN_PICK = 55            
V_MAX_PICK = 252           
A_MAX_PICK = 125           # เดิม 128   ← กันโทนน้ำตาล/เหลืองมากขึ้นเล็กน้อย

# --- เติมใบสว่างกลับ (กว้างขึ้น) ---
REFILL_BRIGHT_LEAF = True
H_RANGE = (35, 105)        # เดิม (40,100)  ← เผื่อม่วง/เหลืองปนเขียว
S_MIN_REFILL = 5           # เดิม 15        ← ใบสว่างมัก S ต่ำมาก
V_MIN_REFILL = 212         # เดิม 220       ← เผื่อไฮไลต์จ้าแต่ไม่ถึง 220
V_MAX_REFILL = 255         # เดิม 252
A_MAX_REFILL = 135         # เดิม 130       ← เผื่อใบอมเหลือง

# เกณฑ์ช่วยเลือกคลาสพืชแบบอัตโนมัติ (ใช้เมื่อไม่ merge middle)
AUTO_PICK_CLASS = True
S_MIN_PICK = 50             # ความอิ่มสีขั้นต่ำสำหรับเลือกคลาส
V_MAX_PICK = 245            # เพดานสว่างตอนพิจารณาเลือกคลาส
A_MAX_PICK = 128            # a* เพดานตอนพิจารณาเลือกคลาส

HANDLE_HIGHLIGHT = True
SPEC_S_MAX, SPEC_V_MIN = 25, 24

# Morphology + กรองชิ้นเล็ก
OPEN_K, CLOSE_K = 3, 7
MIN_OBJ_RATIO = 2e-4        # 0.02% ของพื้นที่ภาพ

# กดความจ้า L* ก่อนคำนวณดัชนี (ถ้าจ้าแรงมาก)
USE_L_GAMMA = False
L_GAMMA = 0.7               # 0.65–0.85 (ยิ่งต่ำยิ่งกดไฮไลต์)
DILATE_REFILL = 3   

def imread_any(path_str: str):
    data = np.fromfile(path_str, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    # Gray-world WB
    b,g,r = cv2.split(img_bgr.astype(np.float32))
    avg = (b.mean() + g.mean() + r.mean()) / 3.0
    b = np.clip(b * avg / (b.mean() + 1e-6), 0, 255)
    g = np.clip(g * avg / (g.mean() + 1e-6), 0, 255)
    r = np.clip(r * avg / (r.mean() + 1e-6), 0, 255)
    wb = cv2.merge([b,g,r]).astype(np.uint8)

    # CLAHE ที่ L*
    lab = cv2.cvtColor(wb, cv2.COLOR_BGR2Lab)
    L, a, b2 = cv2.split(lab)
    L = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(L)
    eq = cv2.cvtColor(cv2.merge([L, a, b2]), cv2.COLOR_Lab2BGR)
    return eq

def apply_L_gamma(img_bgr: np.ndarray, gamma=0.7) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    L, a, b = cv2.split(lab)
    Lg = np.clip((L/255.0)**gamma * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([Lg, a, b]), cv2.COLOR_Lab2BGR)

def compute_index(img_bgr: np.ndarray, index_name="exg") -> np.ndarray:
    B,G,R = cv2.split(img_bgr.astype(np.float32))
    if index_name.lower() == "vari":
        den  = (G + R - B)
        vari = (G - R) / (den + 1e-6)
        vari = np.nan_to_num(vari, nan=0.0, posinf=0.0, neginf=0.0)
        idx  = cv2.normalize(vari, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        exg  = 2*G - R - B
        idx  = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return idx

def suppress_highlight_in_index(img_eq: np.ndarray, src: np.ndarray,
                                s_max=30, v_min=242) -> np.ndarray:
    hsv = cv2.cvtColor(img_eq, cv2.COLOR_BGR2HSV)
    H,S,V = cv2.split(hsv)
    spec = (S < s_max) & (V > v_min)
    src_fixed = src.copy()
    if (~spec).any():
        median_val = np.median(src[~spec])
    else:
        median_val = int(np.median(src))
    src_fixed[spec] = median_val
    return src_fixed

def pick_vegetation_mask_by_rules(img_eq: np.ndarray, regions: np.ndarray,
                                  h_range=(40,100), s_min=50, v_max=245, a_max=128):
    H,S,V = cv2.split(cv2.cvtColor(img_eq, cv2.COLOR_BGR2HSV))
    _,a2,_ = cv2.split(cv2.cvtColor(img_eq, cv2.COLOR_BGR2Lab))
    cands = []
    for k in np.unique(regions):
        idx = (regions==k)
        if idx.sum()==0:
            cands.append((-1,-1,k)); continue
        hue_ok = (H[idx].mean()>=h_range[0]) and (H[idx].mean()<=h_range[1])
        sat_ok = (S[idx].mean()>=s_min)
        val_ok = (V[idx].mean()<=v_max)
        a_ok   = (a2[idx].mean()<=a_max)
        score = int(hue_ok) + int(sat_ok) + int(val_ok) + int(a_ok)
        cands.append((score, idx.sum(), k))
    veg_k = sorted(cands, key=lambda t:(t[0], t[1]))[-1][2]
    mask = (regions==veg_k).astype(np.uint8)*255
    return mask

def bright_leaf_refill(img_eq: np.ndarray, mask: np.ndarray,
                       h_range=(40,100), s_min=15, v_min=220, v_max=252, a_max=130):
    H,S,V = cv2.split(cv2.cvtColor(img_eq, cv2.COLOR_BGR2HSV))
    _,a2,_ = cv2.split(cv2.cvtColor(img_eq, cv2.COLOR_BGR2Lab))
    refill = (
        (mask == 0) &
        (H >= h_range[0]) & (H <= h_range[1]) &
        (S >= s_min) &
        (V >= v_min) & (V <= v_max) &
        (a2 <= a_max)
    )
    out = mask.copy()
    out[refill] = 255
    return out, refill

def morph_and_filter(mask: np.ndarray, open_k=3, close_k=7, min_obj_ratio=2e-4):
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(open_k), int(open_k)))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(close_k), int(close_k)))
    m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  k_open)

    min_obj = int(min_obj_ratio * m.size)
    cnts,_  = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean   = np.zeros_like(m)
    for c in cnts:
        if cv2.contourArea(c) >= min_obj:
            cv2.drawContours(clean, [c], -1, 255, cv2.FILLED)
    return clean

def show_results(img: np.ndarray, idx_img: np.ndarray, mask: np.ndarray, refill_mask=None):
    overlay = img.copy()
    overlay[mask > 0] = (0.5 * overlay[mask > 0] + [40, 200, 40]).astype(np.uint8)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, cnts, -1, (0, 255, 255), 2)

    if refill_mask is not None:
        # ไฮไลต์พิกเซลที่ถูก "เติม" ให้เห็นชัด (สีส้ม)
        overlay[refill_mask] = [0, 180, 255]

    # 4 ช่อง: Original | Index | Overlay | Binary(plant=white)
    plt.figure(figsize=(20, 5))
    plt.subplot(1, 4, 1); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); plt.title("Original"); plt.axis('off')
    plt.subplot(1, 4, 2); plt.imshow(idx_img, cmap='gray'); plt.title(f"Index for Otsu ({INDEX_NAME.upper()})"); plt.axis('off')
    plt.subplot(1, 4, 3); plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)); plt.title("Overlay (orange = refilled)"); plt.axis('off')

    # ช่องที่ต้องการ: ไบนารีพื้นดำ-พืชขาว
    plt.subplot(1, 4, 4); plt.imshow(mask, cmap='gray', vmin=0, vmax=255)
    plt.title("Binary (plant = white)"); plt.axis('off')

    plt.tight_layout(); plt.show()

    veg_px = int((mask > 0).sum())
    pct = 100.0 * veg_px / mask.size
    print(f"Vegetation pixels: {veg_px}  ({pct:.2f}%)")


def run_pipeline(img_path: str):
    img = imread_any(img_path)
    if img is None:
        raise FileNotFoundError(img_path)

    # 1) Preprocess
    img_eq = preprocess(img)

    # (optional) กดความจ้า L* ก่อนคำนวณดัชนี
    if USE_L_GAMMA:
        img_eq = apply_L_gamma(img_eq, L_GAMMA)

    # 2) เลือกดัชนี (ExG / VARI)
    src = compute_index(img_eq, INDEX_NAME)

    # 3) ตัดอิทธิพลไฮไลต์ (ก่อน Otsu)
    src_fixed = suppress_highlight_in_index(img_eq, src, SPEC_S_MAX, SPEC_V_MIN) if HANDLE_HIGHLIGHT else src.copy()

    # 4) Multi-Otsu
    src_for_otsu = cv2.GaussianBlur(src_fixed, (0,0), SMOOTH_SIGMA) if SMOOTH_SIGMA>0 else src_fixed
    thresholds = threshold_multiotsu(src_for_otsu, classes=CLASSES)
    regions = np.digitize(src_for_otsu, bins=thresholds)  # 0..CLASSES-1
    print("Otsu thresholds:", thresholds)

    # 5) เลือกคลาสพืช
    if CLASSES == 4 and MERGE_MIDDLE_IF_4:
        mask = ((regions==1) | (regions==2)).astype(np.uint8)*255
    else:
        if AUTO_PICK_CLASS:
            mask = pick_vegetation_mask_by_rules(img_eq, regions,
                                                 h_range=H_RANGE, s_min=S_MIN_PICK,
                                                 v_max=V_MAX_PICK, a_max=A_MAX_PICK)
        else:
            mask = (regions==(CLASSES-1)).astype(np.uint8)*255

    # 6) เติมใบสว่างกลับ
    refill_mask = None
    if REFILL_BRIGHT_LEAF:
        mask, refill_mask = bright_leaf_refill(
            img_eq, mask,
            h_range=H_RANGE,
            s_min=S_MIN_REFILL,
            v_min=V_MIN_REFILL,
            v_max=V_MAX_REFILL,
            a_max=A_MAX_REFILL
        )
        # ขยาย refill ให้เนียนขึ้น (เฉพาะส่วนที่เพิ่งเติม)
        if DILATE_REFILL > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DILATE_REFILL, DILATE_REFILL))
            # dilate เฉพาะ “ส่วนที่เติม” แล้ว union กลับเข้า mask
            refill_only = np.zeros_like(mask)
            refill_only[refill_mask] = 255
            refill_only = cv2.dilate(refill_only, k, iterations=1)
            mask = cv2.bitwise_or(mask, refill_only)

    # 7) ทำความสะอาด
    clean = morph_and_filter(mask, open_k=OPEN_K, close_k=CLOSE_K, min_obj_ratio=MIN_OBJ_RATIO)

    # 8) แสดงผล
    show_results(img, src, clean, refill_mask)

if __name__ == "__main__":
    run_pipeline(IMG_PATH)
