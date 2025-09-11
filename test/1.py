import cv2
import numpy as np
import matplotlib.pyplot as plt

# ===================== CONFIG =====================
IMG_PATH = r"R:\01-Organize\01-Management\01-Data Center\Brisk\06-AI & Machine Learning (D0340)\04-IOT_Smartfarm\picture_original_topview_smartfarm\picture_topview_10092025_134654.jpg"

USE_HSV_RULE       = True   # เกณฑ์ V สูง & S ต่ำ (ค่าคงที่)
USE_LAB_RULE       = True   # เกณฑ์ L สูง & chroma ต่ำ (ค่าคงที่)
USE_PERCENTILE     = False  # ใช้เปอร์เซ็นไทล์แทนค่าคงที่ (ทนต่อภาพมืด/สว่างไม่เท่ากัน)

USE_PLANT_BAND     = True
USE_GREENISH_GUARD = True

# ค่าจูนเกณฑ์แบบคงที่ (เมื่อ USE_PERCENTILE=False)
V_LOW   = 95   # ลดลง = จับแสงได้มากขึ้น
S_HIGH  = 55    # เพิ่มขึ้น = จับพื้นที่ซีดมากขึ้น
L_LOW   = 95   # LAB - ความสว่างจัด
C_HIGH  = 20    # LAB - chroma ต่ำ (15–25)

# ค่าจูนเปอร์เซ็นไทล์ (เมื่อ USE_PERCENTILE=True)
V_LOW_P   = 45     # ใช้ np.percentile(V, V_LOW_P)
S_HIGH_P  = 85     # ใช้ np.percentile(S, S_HIGH_P)
L_LOW_P   = 95     # ใช้ np.percentile(L, L_LOW_P)
C_HIGH_P  = 20     # ใช้ np.percentile(chroma, C_HIGH_P)

# Morphology และ inpaint
K_DILATE        = (3, 3)   # ขยาย/ปิดรู
DILATE_ITERS    = 3        
CLOSE_ITERS     = 1        # 0–2
INPAINT_RADIUS  = 5        # 5–9

def gray_world(bgr):
    b, g, r = cv2.split(bgr.astype(np.float32))
    mb, mg, mr = b.mean(), g.mean(), r.mean()
    m = (mb + mg + mr) / 3.0
    kb, kg, kr = m/(mb+1e-6), m/(mg+1e-6), m/(mr+1e-6)
    out = cv2.merge((b*kb, g*kg, r*kr))
    return np.clip(out, 0, 255).astype(np.uint8)
img = cv2.imread(IMG_PATH)

img = gray_world(img)   

H, W = img.shape[:2]
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
Hh, S, V = cv2.split(hsv)
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
L, A, B_ = cv2.split(lab)

# ----- Helper: chroma ใน LAB (ระยะจากจุดกลาง 128,128) -----
chroma = np.sqrt((A.astype(np.float32)-128.0)**2 + (B_.astype(np.float32)-128.0)**2)

# ---------- 1) สร้าง mask ตามกฎ HSV ----------
mask_hsv = np.zeros((H, W), np.uint8)
if USE_HSV_RULE:
    if USE_PERCENTILE:
        v_low  = np.percentile(V, V_LOW_P)
        s_high = np.percentile(S, S_HIGH_P)
    else:
        v_low, s_high = V_LOW, S_HIGH
    mV = cv2.inRange(V, int(v_low), 255)
    mS = cv2.inRange(S, 0, int(s_high))
    mask_hsv = cv2.bitwise_and(mV, mS)

# ---------- 2) สร้าง mask ตามกฎ LAB ----------
mask_lab = np.zeros((H, W), np.uint8)
if USE_LAB_RULE:
    if USE_PERCENTILE:
        l_low   = np.percentile(L, L_LOW_P)
        c_high  = np.percentile(chroma, C_HIGH_P)
    else:
        l_low, c_high = L_LOW, C_HIGH
    cond = ((L.astype(np.float32) >= l_low) & (chroma <= c_high)).astype(np.uint8) * 255
    mask_lab = cond

# ---------- 3) รวมกฎ (OR) ----------
glare_raw = cv2.bitwise_or(mask_hsv, mask_lab)

# ---------- 4) สร้าง plant band (โซนใบแบบหลวม) ----------
plant_band = np.ones((H, W), np.uint8)*255  # ค่าเริ่ม: ไม่จำกัด
if USE_PLANT_BAND:
    Bf, Gf, Rf = cv2.split(img.astype(np.float32))
    exg0 = 2*Gf - Rf - Bf
    exg0 = cv2.normalize(exg0, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # ใช้ Triangle ให้กินกว้าง
    _, loose = cv2.threshold(exg0, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
    loose = cv2.morphologyEx(loose, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)), 1)
    plant_band = cv2.dilate(loose, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7)), 1)

# จำกัดขอบเขตให้เฉพาะบนใบ
glare_raw = cv2.bitwise_and(glare_raw, plant_band)

if USE_GREENISH_GUARD:
    greenish = ((img[:,:,1].astype(np.int16) > img[:,:,2].astype(np.int16) + 12) &
                (img[:,:,1].astype(np.int16) > img[:,:,0].astype(np.int16) + 12))
    greenish = greenish.astype(np.uint8) * 255
    glare_raw = cv2.bitwise_and(glare_raw, cv2.bitwise_not(greenish))

# ---------- 6) ทำให้ mask เนียน + โตขึ้นเล็กน้อย ----------
ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, K_DILATE)
if CLOSE_ITERS > 0:
    glare_mask = cv2.morphologyEx(glare_raw, cv2.MORPH_CLOSE, ker, iterations=CLOSE_ITERS)
else:
    glare_mask = glare_raw.copy()
if DILATE_ITERS > 0:
    glare_mask = cv2.dilate(glare_mask, ker, iterations=DILATE_ITERS)

# ---------- 7) INPAINT ----------
img_fixed = cv2.inpaint(img, glare_mask, INPAINT_RADIUS, cv2.INPAINT_TELEA)

# ---------- 8) EXG + OTSU (before/after) ----------
def exg_otsu(bgr):
    Bc, Gc, Rc = cv2.split(bgr.astype(np.float32))
    exg = 2*Gc - Rc - Bc
    exg_norm = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(exg_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return exg_norm, mask

exg_before, plant_mask_before = exg_otsu(img)
exg_after,  plant_mask_after  = exg_otsu(img_fixed)

# 1) ทำ plant_band (โซนพืชแบบหลวม) จากภาพเดิม
B0, G0, R0 = cv2.split(img.astype(np.float32))
exg0 = 2*G0 - R0 - B0
exg0 = cv2.normalize(exg0, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
_, loose = cv2.threshold(exg0, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)

# 2) จำกัด glare ให้เฉพาะบนใบ แล้วขยายขอบนิดหน่อย
glare_in_leaf = cv2.bitwise_and(glare_mask, plant_band)
glare_in_leaf = cv2.dilate(glare_in_leaf,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)),
                           iterations=1)

# 3) รวมเข้าไปใน mask หลัง inpaint (บังคับเติมให้เป็นใบ)
plant_mask_fixed = cv2.bitwise_or(plant_mask_after, glare_in_leaf)

# (ออปชัน) 4) เติมรูภายในใบ (fill holes) เพื่อไม่ให้มีรูตรงไฮไลต์
ff = plant_mask_fixed.copy()
h, w = ff.shape[:2]
mask_ff = np.zeros((h+2, w+2), np.uint8)
cv2.floodFill(ff, mask_ff, (0,0), 255)        # เติมพื้นหลัง
holes = cv2.bitwise_not(ff)                    # ส่วนที่เป็น "รู" ภายในวัตถุ
plant_mask_filled = cv2.bitwise_or(plant_mask_fixed, holes)

# 5) (ถ้ากลัวล้ำไปนอกใบ) บังคับให้อยู่ในโซนพืชอีกชั้น
plant_mask_filled = cv2.bitwise_and(plant_mask_filled, plant_band)


# ---------- 9) สร้าง overlay จุด glare (สีส้ม) ----------
overlay = img.copy()
overlay[glare_mask > 0] = (0, 128, 255)  # BGR ส้ม

# ---------- 10) แสดงผล ----------
plt.figure(figsize=(12, 10))

plt.subplot(3,3,1); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); plt.title("Original"); plt.axis('off')
plt.subplot(3,3,2); plt.imshow(glare_mask, cmap='gray', vmin=0, vmax=255); plt.title("Glare mask (final)"); plt.axis('off')
plt.subplot(3,3,3); plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)); plt.title("Overlay (orange = glare)"); plt.axis('off')

plt.subplot(3,3,4); plt.imshow(exg_before, cmap='gray'); plt.title("Index before (ExG)"); plt.axis('off')
plt.subplot(3,3,5); plt.imshow(plant_mask_before, cmap='gray', vmin=0, vmax=255); plt.title("Binary before inpaint"); plt.axis('off')
plt.subplot(3,3,6); plt.imshow(plant_mask_after, cmap='gray', vmin=0, vmax=255); plt.title("Binary after inpaint"); plt.axis('off')

plt.subplot(3,3,7); plt.imshow(plant_band, cmap='gray', vmin=0, vmax=255); plt.title("Plant band (loose)"); plt.axis('off')
plt.subplot(3,3,8); plt.imshow(glare_in_leaf, cmap='gray', vmin=0, vmax=255); plt.title("Glare ∧ Plant band"); plt.axis('off')
plt.subplot(3,3,9); plt.imshow(plant_mask_filled, cmap='gray', vmin=0, vmax=255); plt.title("Binary fixed (A + fill)"); plt.axis('off')

plt.tight_layout(); plt.show()

veg_px_fixed = int((plant_mask_filled > 0).sum())
print(f"Vegetation AFTER fix : {veg_px_fixed}")

