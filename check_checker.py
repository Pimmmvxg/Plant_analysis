import cv2
import numpy as np

# ---- 1) ตั้งค่าจำนวน "ช่อง" บนกระดาน (not inner corners) ----
# ถ้ากระดาน 5x5 ช่อง ให้ใส่ 5,5 แล้วคำนวณ inner corners เป็น 4x4
NUM_SQUARES = (4,45)  # (cols, rows) จำนวน "ช่อง"
pattern_size = (NUM_SQUARES[0] - 1, NUM_SQUARES[1] - 1)

# ---- 2) โหลดภาพ ----
img = cv2.imread(r"R:\01-Organize\01-Management\01-Data Center\Brisk\06-AI & Machine Learning (D0340)\04-IOT_Smartfarm\picture_original_topview_smartfarm\picture_topview_A_05112025_090038.jpg")
if img is None:
    raise FileNotFoundError("ไม่พบภาพที่ระบุ")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ---- 3) เพิ่มคอนทราสต์ (ช่วยมากกับภาพแสงไม่สม่ำเสมอ) ----
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
gray_eq = clahe.apply(gray)

# ---- 4) ลองด้วย findChessboardCornersSB (ใหม่/ทนกว่า) ----
found = False
corners = None
try:
    found, corners = cv2.findChessboardCornersSB(
        gray_eq, pattern_size, flags=cv2.CALIB_CB_EXHAUSTIVE
    )
except AttributeError:
    # OpenCV build บางตัวอาจไม่มีฟังก์ชัน SB; จะไปทางคลาสสิกแทน
    pass

# ---- 5) ถ้ายังไม่เจอ ใช้แบบคลาสสิก + flags ที่ช่วยให้ทนขึ้น ----
if not found or corners is None:
    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH |
        cv2.CALIB_CB_NORMALIZE_IMAGE |
        cv2.CALIB_CB_EXHAUSTIVE |     # ลองหลายสเกล/ pattern วางเอียง
        cv2.CALIB_CB_ACCURACY
    )
    found, corners = cv2.findChessboardCorners(gray_eq, pattern_size, flags=flags)

# ---- 6) ถ้าเจอ มาปรับมุมให้คมขึ้นด้วย cornerSubPix ----
if found and corners is not None:
    # เกณฑ์หยุดเมื่อถึง 30 รอบหรือ error < 0.001
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    # ต้องเป็น float32
    gray_f = gray_eq.copy()
    corners = cv2.cornerSubPix(gray_f, np.float32(corners), (5,5), (-1,-1), criteria)

    out = img.copy()
    cv2.drawChessboardCorners(out, pattern_size, corners, found)
    cv2.imshow("Chessboard Corners", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("✅ Found corners:", len(corners))
else:
    print("❌ Chessboard corners not found.")
