# config.py
from pathlib import Path
import re
from typing import Optional

# Debug / I/O configuration
DEBUG_MODE = 'print'      # 'none' | 'print' | 'plot'
THREADS = 1               # Number of threads to use for processing
SAVE_MASK = True          # Save the mask image
SAVE_TOP_OVERLAY = True
SAVE_SIDE_ROIS_OVERVIEW = True  # เซฟรูปภาพรวมกรอบ ROI (#1, #2, ...)

# I/O
INPUT_PATH: Optional[Path] = None
#INPUT_PATH: Optional[Path] = Path(r"R:\01-Organize\01-Management\01-Data Center\Brisk\06-AI & Machine Learning (D0340)\04-IOT_Smartfarm\picture_original_topview_smartfarm\picture_topview_03092025_140927.jpg")
OUTPUT_DIR: Optional[Path] = None
VIEW: Optional[str] = None       # "side" | "top" | None
EXTENSIONS = ['.png', '.jpg', '.jpeg']  # Supported image file extensions

# Folder-name view patterns
_SIDE_PATTERNS = [r"sideview", r"side[_\- ]?view", r"\bside\b"]
_TOP_PATTERNS  = [r"topview",  r"top[_\- ]?view",  r"\btop\b"]

# External Binary Mask file (optional)
MASK_PATH: Optional[Path] = None   # เช่น r"C:\path\to\my_mask.png"
MASK_SPEC: Optional[dict] = None
USE_EXTERNAL_MASK: bool = False    # True เมื่อ MASK_PATH ถูกตั้งค่า

# Utilities
def _detect_view_from_path(p: Path) -> str:
    s = str(p).lower()
    for pat in _SIDE_PATTERNS:
        if re.search(pat, s):
            return "side"
    for pat in _TOP_PATTERNS:
        if re.search(pat, s):
            return "top"
    return "unknown"

def safe_target_name(p: Optional[Path]) -> str:
    if p is None:
        return "input"
    q = Path(p)
    # ไฟล์: มี suffix หรือ is_file() → ใช้ stem
    if q.suffix or q.is_file():
        return q.stem or "input"
    # โฟลเดอร์ → ใช้ name
    return q.name or "input"

def _default_output_dir(p: Optional[Path]) -> Path:
    if p is None:
        return Path("./results_input")  # fallback กรณีไม่ได้ส่ง path

    path = Path(p)
    try:
        if path.is_file():  
            name = path.stem
        else:
            name = path.name
        if not name:
            name = "input"
    except Exception:
        name = "input"

    return Path(f"./results_{name}")

# -------------------------------------------------
# Main entry to resolve runtime parameters
# -------------------------------------------------
def resolve_runtime(input_path: str | Path,
                    output_dir: Optional[str | Path] = None,
                    view: Optional[str] = None):
   
    global INPUT_PATH, OUTPUT_DIR, VIEW, MASK_SPEC, USE_EXTERNAL_MASK, MASK_PATH
    # --- 1) INPUT_PATH ---
    if input_path is None:
        raise ValueError("resolve_runtime: 'input_path' is required.")
    INPUT_PATH = Path(input_path)

    # --- 2) VIEW ---
    if view is None or view not in ("side", "top"):
        VIEW = _detect_view_from_path(INPUT_PATH)
        if VIEW == "unknown":
            raise ValueError(
                "Cannot detect view type from input path. "
                "Please specify --view explicitly as 'side' or 'top'."
            )
    else:
        VIEW = view  

    # --- 3) OUTPUT_DIR ---
    OUTPUT_DIR = Path(output_dir) if output_dir else _default_output_dir(INPUT_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- 4) External mask mode (ถ้าตั้ง MASK_PATH) ---
    USE_EXTERNAL_MASK = False
    if MASK_PATH:
        if not isinstance(MASK_PATH, Path):
            MASK_PATH = Path(MASK_PATH)
        if not MASK_PATH.exists():
            raise FileNotFoundError(f"External MASK_PATH not found: {MASK_PATH}")
        USE_EXTERNAL_MASK = True
        MASK_SPEC = None
        return

    # --- 5) ตั้ง MASK_SPEC ตาม VIEW (เมื่อไม่ได้ใช้ external mask) ---
    if VIEW == "side":
        '''MASK_SPEC = {
            "channel": "lab_a",
            "method": "binary",
            "threshold": 123,
            "object_type": "dark",
        }
        '''
        MASK_SPEC = {
            "channel": "lab_a",
            "method": "side_auto",  # ใช้ auto + guard
            "object_type": "dark",
            "s_min": 30,
            "v_bg": 45,
            "v_shadow_max": 115,
            "green_h": (20, 95),
            "bottom_roi_ratio": 0.60,
            "min_cc_area": 200,
        } 
    elif VIEW == "top":
        MASK_SPEC = {
            "channel": "lab_a",
            "method": "otsu",
            "object_type": "dark",
            "ksize": 2001,  # ใช้เมื่อ method="gaussian"
            "offset": 5,    # ใช้เมื่อ method="gaussian"
        }
        '''
        MASK_SPEC = {
            "channel": "hsv_v",
            "method": "binary",
            "threshold": "130",
            "object_type": "light" 
        }'''
    else:
        MASK_SPEC = None  

# TOP view parameters
TOP_ROI_MODE = "auto" 
TOP_SUMMARY_MODE = "per_roi"
TOP_MIN_PLANT_AREA = 4000      
TOP_MERGE_GAP = 0             # px; มาก = รวมง่าย
TOP_CLOSE_ITERS = 1            # ปิดรูเล็ก ๆ ก่อนจับกล่อง

STEM_CONNECT_MODE = "cc_touch"
STEM_CC_CLOSE_K   = 3

ROWS, COLS = 2, 3
ROI_TYPE = "partial"  # 'partial' | 'cutto' | 'largest'
ROI_RADIUS = 400
TOP_EXPECT_N_MIN = 4
TOP_EXPECT_N_MAX = 10

# ---------- STEM RESCUE (TOP view) ----------
ENABLE_STEM_RESCUE        = True   # เปิด/ปิดฟีเจอร์กู้ก้าน
RESCUE_B_OTSU_FALLBACK    = 135
RESCUE_L_MIN_FOR_STEM     = 120
RESCUE_S_MIN_FOR_STEM     = 12
RESCUE_A_WHITE_MAX        = 150
RESCUE_HUE_YELLOW         = (15, 45)   # OpenCV H: 0–179
RESCUE_GEO_ITERS          = 40
RESCUE_VERT_DILATE_K      = 9
RESCUE_MIN_AREA_PX        = 800        # ปรับตามความละเอียดภาพ

# -------------------------------------------------
# SIDE view parameters (rectangle ROI)
USE_FULL_IMAGE_ROI = False
#ROI_X, ROI_Y, ROI_W, ROI_H = 100, 100, 240, 240
MIN_PLANT_AREA = 600
SIDE_MERGE_GAP = 200
SIDE_EXPECT_N_MIN = 3
SIDE_EXPECT_N_MAX = 20
SIDE_BRIDGE_GAP_X = 5
SIDE_BRIDGE_GAP_Y = 30

# -------------------------------------------------
# Calibration scale
CHECKER_SQUARE_MM = 12.0 # Size of one square in checkerboard (mm)
RECT_SIZE_MM = (48, 48) # (w, h) ของสี่เหลี่ยมอ้างอิงจริง (mm)
CHECKER_PATTERN = (3, 3)
FALLBACK_MM_PER_PX = 48.0 /340.0

# -------------------------------------------------
# Mask scoring weights
W_COVERAGE   = 1.0
W_COMPONENTS = 1.2
W_SOLIDITY   = 0.5
W_BORDER     = 1.0
FORCE_OBJECT_WHITE = True
COVERAGE_TARGET = 0.05
MERGE_COMPONENTS_PER_SLOT = True
