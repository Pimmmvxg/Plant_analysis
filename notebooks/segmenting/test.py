import cv2
import matplotlib.pyplot as plt

# --- Input: side-view ROI ---
roi = cv2.imread(r"R:\01-Organize\01-Management\01-Data Center\Brisk\06-AI & Machine Learning (D0340)\04-IOT_Smartfarm\picture_original_sideview_smartfarm\picture_sideview_01102025_153446.jpg")   # แทนด้วย path จริง

# Step 1: แปลงไป Lab และดึงช่อง L
L = cv2.cvtColor(roi, cv2.COLOR_BGR2Lab)[:,:,0]

# Step 2: ทำ CLAHE บน L
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
L_clahe = clahe.apply(L)

# Step 3: Threshold (Otsu)
_, th = cv2.threshold(L_clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Step 4: Morphology
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
th_close = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, 1)
th_open  = cv2.morphologyEx(th_close, cv2.MORPH_OPEN, kernel, 1)

# แสดง debug images
titles = [
    "Original ROI",
    "L channel",
    "CLAHE on L",
    "Threshold (Otsu)",
    "After Morph Close",
    "After Morph Open (final)"
]
images = [roi, L, L_clahe, th, th_close, th_open]

plt.figure(figsize=(15,8))
for i, (img, title) in enumerate(zip(images, titles)):
    plt.subplot(2,3,i+1)
    if len(img.shape)==2:
        plt.imshow(img, cmap="gray")
    else:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
plt.tight_layout()
plt.show()
