from pathlib import Path

#selected view for the plant analysis
#options: "top", "side"
VIEW = "top"

#Debug
DEBUG_MODE = 'print'  # 'none'|'print'|'plot'
THREADS = 1  # Number of threads to use for processing
SAVE_MASK = True  # Save the mask image

#I/O
INPUT_PATH = Path(r"C:\Cantonese\topview_test.jpg")  # Single file or folder
OUTPUT_DIR = Path(r".\results_topview")  # Output directory for results
EXTENSIONS = ['.png', '.jpg', '.jpeg']  # Supported image file extensions

#TOP
ROWS, COLS = 2, 3
ROI_TYPE = "partial" # 'partial' | 'cutto' | 'largest'


# SIDE (rectangle ROI)
USE_FULL_IMAGE_ROI = False
ROI_X, ROI_Y, ROI_W, ROI_H = 100, 320, 240, 150

