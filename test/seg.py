import cv2, numpy as np

# --- โหลดภาพ ---
img_bgr = cv2.imread(r"R:\01-Organize\01-Management\01-Data Center\Brisk\06-AI & Machine Learning (D0340)\04-IOT_Smartfarm\picture_original_sideview_smartfarm\picture_sideview_08092025_100850.jpg")
if img_bgr is None:
    raise FileNotFoundError("อ่านภาพไม่สำเร็จ ตรวจ path อีกครั้ง")

# สำหรับ PlantCV ควรเก็บเป็น RGB ไว้ใช้ต่อ
img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# --- คุมขนาดหน้าต่างแสดงผล (รีไซส์เฉพาะตอนโชว์) ---
H, W = img_bgr.shape[:2]
MAX_W, MAX_H = 1000, 800  # ปรับได้ตามหน้าจอคุณ
scale = min(MAX_W / W, MAX_H / H, 1.0)
disp_w, disp_h = int(W * scale), int(H * scale)

disp_base = cv2.resize(img_bgr, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
disp = disp_base.copy()

# --- ช่วยวาดข้อความแนะนำ ---
def draw_hints(canvas):
    cv2.putText(canvas, "CLICK: plant then background | 'r' undo | 'ESC' to exit",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 220, 20), 2, cv2.LINE_AA)

# --- เก็บคลิก (พิกัดจริงของภาพ) ---
clicks = []  # [(x_real, y_real), ...]
def on_mouse(event, x, y, flags, param):
    global disp
    if event == cv2.EVENT_LBUTTONDOWN:
        # พิกัดที่คลิกมาจากภาพที่ถูกย่อ -> แปลงกลับเป็นพิกัดจริง
        xr = int(round(x / scale))
        yr = int(round(y / scale))
        # ป้องกันเลยขอบ
        xr = max(0, min(W - 1, xr))
        yr = max(0, min(H - 1, yr))
        clicks.append((xr, yr))

        # วาดจุดลงบนภาพแสดงผล
        disp = disp_base.copy()
        draw_hints(disp)
        for i, (xx, yy) in enumerate(clicks):
            # แปลงพิกัดจริง -> พิกัดแสดง
            xd, yd = int(round(xx * scale)), int(round(yy * scale))
            color = (0, 255, 0) if i == 0 else (0, 165, 255)  # จุดที่ 1=เขียว (พืช), จุดที่ 2=ส้ม (พื้นหลัง)
            cv2.circle(disp, (xd, yd), 6, color, -1, cv2.LINE_AA)
            cv2.putText(disp, f"{i+1}", (xd+8, yd-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

# --- เปิดหน้าต่างและรอคลิก ---
cv2.namedWindow("CLICK (resized display)")
cv2.setMouseCallback("CLICK (resized display)", on_mouse)

disp = disp_base.copy()
draw_hints(disp)
while True:
    cv2.imshow("CLICK (resized display)", disp)
    k = cv2.waitKey(20) & 0xFF
    if k == 27:  # ESC
        break
    if k == ord('r') and clicks:
        clicks.pop()                         # undo ล่าสุด
        disp = disp_base.copy()
        draw_hints(disp)
        for i, (xx, yy) in enumerate(clicks):
            xd, yd = int(round(xx * scale)), int(round(yy * scale))
            color = (0, 255, 0) if i == 0 else (0, 165, 255)
            cv2.circle(disp, (xd, yd), 6, color, -1, cv2.LINE_AA)
            cv2.putText(disp, f"{i+1}", (xd+8, yd-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    if len(clicks) >= 2:
        break

cv2.destroyAllWindows()

if len(clicks) < 2:
    raise RuntimeError("ต้องคลิกอย่างน้อย 2 จุด: จุดที่ 1=พืช, จุดที่ 2=พื้นหลัง")

# --- คำนวณ points (x=b, y=a) ตามที่ PlantCV ต้องการ ---
lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
L, a, b = cv2.split(lab)

(px1, py1), (px2, py2) = clicks[:2]  # จุดที่ 1=พืช, จุดที่ 2=พื้นหลัง (สลับได้ตามต้องการ)
plant_point = (int(b[py1, px1]), int(a[py1, px1]))  # (x=b, y=a)
bg_point    = (int(b[py2, px2]), int(a[py2, px2]))

points = [plant_point, bg_point]
print("points =", points)
