# config.py
from pathlib import Path
import re
from typing import Optional

# -------------------------
# Debug / I/O configuration
# -------------------------
DEBUG_MODE = "none"    # 'none' | 'print' | 'plot'
THREADS = 1               # Number of threads to use for processing
SAVE_MASK = True          # Save the mask image
SAVE_TOP_OVERLAY = True
SAVE_SIDE_ROIS_OVERVIEW = True  # เซฟรูปภาพรวมกรอบ ROI (#1, #2, ...)

# -------------------------
# I/O
# -------------------------
#INPUT_PATH: Optional[Path] = None
INPUT_PATH: Optional[Path] = Path(r"R:\01-Organize\01-Management\01-Data Center\Brisk\06-AI & Machine Learning (D0340)\04-IOT_Smartfarm\picture_original_topview_smartfarm\picture_topview_10092025_134654.jpg")
#OUTPUT_DIR: Optional[Path] = Path(r"D:\smartfarm\picture_topview")
VIEW: Optional[str] = "top"       # "side" | "top" | None
EXTENSIONS = ['.png', '.jpg', '.jpeg']  # Supported image file extensions

# -------------------------
# Folder-name view patterns
# -------------------------
_SIDE_PATTERNS = [r"sideview", r"side[_\- ]?view", r"\bside\b"]
_TOP_PATTERNS  = [r"topview",  r"top[_\- ]?view",  r"\btop\b"]

# -------------------------
# External Binary Mask file (optional)
# -------------------------
MASK_PATH: Optional[Path] = None   # เช่น r"C:\path\to\my_mask.png"
MASK_SPEC: Optional[dict] = None
USE_EXTERNAL_MASK: bool = False    # True เมื่อ MASK_PATH ถูกตั้งค่า

# -------------------------------------------------
# Utilities
# -------------------------------------------------
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
        MASK_SPEC = {
            "channel": "lab_a",
            "method": "binary",
            "threshold": 125,
            "object_type": "dark",
        }
    elif VIEW == "top":
        MASK_SPEC = {
            "channel": "lab_a",
            "method": "otsu",
            "object_type": "dark",
            "ksize": 2001,  # ใช้เมื่อ method="gaussian"
            "offset": 5,    # ใช้เมื่อ method="gaussian"
        }
    else:
        MASK_SPEC = None  

# -------------------------------------------------
# TOP view parameters
# -------------------------------------------------
ROWS, COLS = 2, 4
ROI_TYPE = "partial"  # 'partial' | 'cutto' | 'largest'
ROI_RADIUS = 400
TOP_EXPECT_N_MIN = 4
TOP_EXPECT_N_MAX = 10

# -------------------------------------------------
# SIDE view parameters (rectangle ROI)
# -------------------------------------------------
USE_FULL_IMAGE_ROI = False
# ROI_X, ROI_Y, ROI_W, ROI_H = 100, 100, 240, 240
MIN_PLANT_AREA = 500
SIDE_MERGE_GAP = 12
SIDE_EXPECT_N_MIN = 3
SIDE_EXPECT_N_MAX = 20

# -------------------------------------------------
# Calibration scale
# -------------------------------------------------
CHECKER_SQUARE_MM = 12.0
CHECKER_PATTERN = (3, 3)
FALLBACK_MM_PER_PX = 12.0 / 146.3

# -------------------------------------------------
# Mask scoring weights
# -------------------------------------------------
W_COVERAGE   = 1.0
W_COMPONENTS = 1.2
W_SOLIDITY   = 0.5
W_BORDER     = 1.0
FORCE_OBJECT_WHITE = True
COVERAGE_TARGET = 0.05
MERGE_COMPONENTS_PER_SLOT = True
