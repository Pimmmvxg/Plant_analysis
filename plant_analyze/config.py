from pathlib import Path
import re

#selected view for the plant analysis
#options: "top", "side"
#VIEW = "top"
#VIEW = "side"

#Debug
DEBUG_MODE = 'print'  # 'none'|'print'|'plot'
THREADS = 1  # Number of threads to use for processing
SAVE_MASK = True  # Save the mask image
SAVE_TOP_OVERLAY = True
SAVE_SIDE_ROIS_OVERVIEW = True  # เซฟรูปภาพรวมกรอบ ROI (#1, #2, ...)

#I/O
INPUT_PATH = Path(
    r"R:\01-Organize\01-Management\01-Data Center\Brisk\06-AI & Machine Learning (D0340)\04-IOT_Smartfarm\picture_original_sideview_smartfarm\picture_sideview_04092025_100132.jpg",
    )  # Single file or folder
EXTENSIONS = ['.png', '.jpg', '.jpeg']  # Supported image file extensions

_SIDE_PATTERNS = [r"sideview", r"side[_\- ]?view", r"\bside\b"]
_TOP_PATTERNS = [r"topview", r"top[_\- ]?view", r"\btop\b"]

def _detect_view_from_path(p: Path) -> str:
    s = str(p).lower()
    for pat in _SIDE_PATTERNS:
        if re.search(pat, s):
            return "side"
    for pat in _TOP_PATTERNS:
        if re.search(pat, s):
            return "top"
    #ไม่พบ pattern
    return "unknown"

VIEW = _detect_view_from_path(INPUT_PATH)
if VIEW == "unknown":
    print("Warning: Cannot detect view (top/side) from input path. Default to 'side'.")
    VIEW = "side"
    
_target_name = INPUT_PATH.stem if INPUT_PATH.is_file() else INPUT_PATH.name
OUTPUT_DIR = Path(rf".\results_{_target_name}")  # Output directory for results

#TOP
ROWS, COLS = 2, 4
ROI_TYPE = "partial" # 'partial' | 'cutto' | 'largest'
ROI_RADIUS = 400 
TOP_EXPECT_N_MIN = 4
TOP_EXPECT_N_MAX = 10

#SIDE (rectangle ROI)
USE_FULL_IMAGE_ROI = False
#ROI_X, ROI_Y, ROI_W, ROI_H = 100, 100, 240, 240
MIN_PLANT_AREA = 500      # พิกเซลขั้นต่ำของก้อนที่จะนับเป็น 1 ต้น 
SIDE_MERGE_GAP = 12        # ระยะช่องว่างแนวนอน (px) สำหรับรวมกล่องที่ชิดกันมาก
SIDE_EXPECT_N_MIN = 3
SIDE_EXPECT_N_MAX = 20

#Mask Selection
#ใช้ไฟล์ Mask(binary)
MASK_PATH = None # เช่น r"C:\path\my_mask.png" ; ถ้า None จะไม่ใช้โหมดนี้

#กำหนดThresholdเอง
#MASK_SPEC = None

if VIEW == "side":
    MASK_SPEC = {
        "channel": "lab_a",        # "lab_a"|"lab_b"|"lab_l"|"hsv_h"|"hsv_s"|"hsv_v"
        "method": "binary",        # ใช้ binary threshold ตรง ๆ
        "threshold": 125,          # ค่า threshold ตัดต้นไม้กับพื้นหลัง
        "object_type": "dark",     # วัตถุเข้มกว่าพื้นหลัง
    }

elif VIEW == "top":
    MASK_SPEC = {
        "channel": "lab_a",        # LAB channel a
        "method": "otsu",          # auto threshold (Otsu)
        "object_type": "dark",     # วัตถุเข้มกว่าพื้นหลัง
        "ksize": 2001,             # ใช้กรณี method="gaussian"
        "offset": 5,               # ใช้กรณี method="gaussian"
    }

else:
    MASK_SPEC = None   # fallback → auto select
#Calibretion scale
CHECKER_SQUARE_MM = 12.0
CHECKER_PATTERN = (3, 3)
FALLBACK_MM_PER_PX = 12.0 / 146.3

# ---- Mask scoring weights ----
W_COVERAGE   = 1.0   # น้ำหนักความใกล้เคียง coverage เป้าหมาย
W_COMPONENTS = 1.2   # น้ำหนักจำนวนก้อนตามที่คาดหวัง
W_SOLIDITY   = 0.5   # น้ำหนักความแน่น (solidity)
W_BORDER     = 1.0   # น้ำหนักโทษการแตะขอบเฟรม
FORCE_OBJECT_WHITE = True
COVERAGE_TARGET = 0.05
MERGE_COMPONENTS_PER_SLOT = True