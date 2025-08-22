from pathlib import Path

#selected view for the plant analysis
#options: "top", "side"
VIEW = "side"

#Debug
DEBUG_MODE = 'print'  # 'none'|'print'|'plot'
THREADS = 1  # Number of threads to use for processing
SAVE_MASK = True  # Save the mask image

#I/O
INPUT_PATH = Path(r"C:\Cantonese\sideview_mm.png")  # Single file or folder
OUTPUT_DIR = Path(r".\results_sideview_mm")  # Output directory for results
EXTENSIONS = ['.png', '.jpg', '.jpeg']  # Supported image file extensions

#TOP
ROWS, COLS = 2, 3
ROI_TYPE = "partial" # 'partial' | 'cutto' | 'largest'
ROI_RADIUS = 200 

#SIDE (rectangle ROI)
USE_FULL_IMAGE_ROI = True
ROI_X, ROI_Y, ROI_W, ROI_H = 100, 100, 240, 240

#Mask Selection
#ใช้ไฟล์ Mask(binary)
MASK_PATH = None # เช่น r"C:\path\my_mask.png" ; ถ้า None จะไม่ใช้โหมดนี้

#กำหนดThresholdเอง
#MASK_SPEC = None

MASK_SPEC = {
     "channel": "lab_a",        # "lab_a"|"lab_b"|"lab_l"|"hsv_h"|"hsv_s"|"hsv_v"
     "method": "mean",      # "otsu"|"triangle"|"gaussian"
     "object_type": "dark",     # "dark"|"light"
     "ksize": 2001,               # ใช้เมื่อ method="gaussian"
     "offset": 5                # ใช้เมื่อ method="gaussian"
}

#Calibretion scale
CHECKER_SQUARE_MM = 8.0
CHECKER_PATTERN = (4, 4)
FALLBACK_MM_PER_PX = 10.0 / 51.0

