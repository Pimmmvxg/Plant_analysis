from pathlib import Path

#selected view for the plant analysis
#options: "top", "side"
#VIEW = "top"
TOP_EXPECT_N_MIN = 4
TOP_EXPECT_N_MAX = 10
SIDE_EXPECT_N_MIN = 3
SIDE_EXPECT_N_MAX = 20
COVERAGE_TARGET = 0.05
MERGE_COMPONENTS_PER_SLOT = True

VIEW = "side"

#Debug
DEBUG_MODE = 'print'  # 'none'|'print'|'plot'
THREADS = 1  # Number of threads to use for processing
SAVE_MASK = True  # Save the mask image
SAVE_TOP_OVERLAY = True

#I/O
INPUT_PATH = Path(r"C:\Users\admin\Downloads\05.jpg")  # Single file or folder
OUTPUT_DIR = Path(r".\results_side05_new")  # Output directory for results
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
    "method": "binary",         
    "threshold": 124,          
    "object_type": "dark",      # วัตถุเข้มกว่าพื้นหลัง
    
}

'''
MASK_SPEC = {
     "channel": "lab_a",        # "lab_a"|"lab_b"|"lab_l"|"hsv_h"|"hsv_s"|"hsv_v"
     "method": "otsu",      # "otsu"|"triangle"|"gaussian"
     "object_type": "dark",     # "dark"|"light"
     "ksize": 2001,               # ใช้เมื่อ method="gaussian"
     "offset": 5                # ใช้เมื่อ method="gaussian"
}
'''
#Calibretion scale
CHECKER_SQUARE_MM = 10.0
CHECKER_PATTERN = (3, 3)
FALLBACK_MM_PER_PX = 10.0 / 51.0

# ---- Mask scoring weights ----
W_COVERAGE   = 1.0   # น้ำหนักความใกล้เคียง coverage เป้าหมาย
W_COMPONENTS = 1.2   # น้ำหนักจำนวนก้อนตามที่คาดหวัง
W_SOLIDITY   = 0.5   # น้ำหนักความแน่น (solidity)
W_BORDER     = 1.0   # น้ำหนักโทษการแตะขอบเฟรม
FORCE_OBJECT_WHITE = True