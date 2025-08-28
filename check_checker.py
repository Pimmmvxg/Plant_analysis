import cv2
img  = cv2.imread(r"C:\Users\admin\Downloads\05.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

pattern = (3, 3)  # 5x5 ช่อง -> 4x4 มุมด้านใน
flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE  # ชัวร์กว่า FAST_CHECK
found, corners = cv2.findChessboardCorners(gray, pattern, flags)
print("found:", found, "corners:", None if corners is None else corners.shape)
