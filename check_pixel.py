import cv2
import numpy as np

# โหลดภาพต้นฉบับ
orig = cv2.imread(r"R:\01-Organize\01-Management\01-Data Center\Brisk\06-AI & Machine Learning (D0340)\04-IOT_Smartfarm\picture_original_topview_smartfarm\picture_topview_A_05112025_090038.jpg")
h0, w0 = orig.shape[:2]

# resize สำหรับแสดงผล
scale_width = 800
ratio = scale_width / w0
new_dim = (scale_width, int(h0 * ratio))
resized = cv2.resize(orig, new_dim, interpolation=cv2.INTER_AREA)

points = []

def click_event(event, x, y, flags, param):
    global points, resized, orig, ratio

    if event == cv2.EVENT_LBUTTONDOWN:
        # แปลงพิกัดจาก resized -> original
        x_orig = int(x / ratio)
        y_orig = int(y / ratio)

        points.append((x_orig, y_orig))

        # วาดจุดบนภาพ resized ที่โชว์
        cv2.circle(resized, (x, y), 5, (0, 0, 255), -1)

        if len(points) == 2:
            x1, y1 = points[0]
            x2, y2 = points[1]

            #คำนวณระยะจากภาพต้นฉบับ (จริง)
            dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

            # วาดเส้นบน resized
            cv2.line(resized,
                     (int(x1 * ratio), int(y1 * ratio)),
                     (int(x2 * ratio), int(y2 * ratio)),
                     (255, 0, 0), 2)

            text = f"{dist:.2f} px (จริง)"
            cv2.putText(resized, text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 2)

            print("ระยะทางจริง:", dist, "px")

        cv2.imshow("image", resized)


# แสดงภาพที่ resize
cv2.imshow("image", resized)
cv2.setMouseCallback("image", click_event)

cv2.waitKey(0)
cv2.destroyAllWindows()
