from pathlib import Path
import re
from typing import Optional

#Debug
DEBUG_MODE = 'print'  # 'none'|'print'|'plot'
THREADS = 1  # Number of threads to use for processing
SAVE_MASK = True  # Save the mask image
SAVE_TOP_OVERLAY = True
SAVE_SIDE_ROIS_OVERVIEW = True  # เซฟรูปภาพรวมกรอบ ROI (#1, #2, ...)

#I/O
INPUT_PATH: Optional[Path] = None
#INPUT_PATH = Path(rf".\input\sideview_1.png") 
OUTPUT_DIR: Optional[Path] = None
VIEW: Optional[str] = None  # "side" | "top" | None 
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

def _default_output_dir(p: Path) -> Path:
    name = p.stem if p.is_file() else p.name
    return Path(f"./results_{name}")

def resolve_runtime(input_path: str|Path,
                    output_dir: Optional[str|Path] = None,
                    view: Optional[str] = None):
    global INPUT_PATH, OUTPUT_DIR, VIEW, MASK_SPEC
    INPUT_PATH = Path(input_path)
    VIEW = view or _detect_view_from_path(INPUT_PATH)
    OUTPUT_DIR = Path(output_dir) if output_dir else _default_output_dir(INPUT_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    #Mask Selection
    #ใช้ไฟล์ Mask(binary)
    MASK_PATH = None # เช่น r"C:\path\my_mask.png" ; ถ้า None จะไม่ใช้โหมดนี้
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