import paho.mqtt.client as mqtt
import json
import time

# ข้อมูลของ ThingsBoard
THINGSBOARD_HOST = "191.20.110.47"  # หรือ IP Server ของเรา
ACCESS_TOKEN = "image_token"  # Token จากอุปกรณ์ใน ThingsBoard
PORT = 1883
# สร้าง client
client = mqtt.Client()
client.username_pw_set(ACCESS_TOKEN)
client.connect(THINGSBOARD_HOST, PORT, 60)
client.loop_start()

while True:
    attributes = {
        "global_color_name": "Warm Green",
        "auto_height_mm": 312.17019822282978,
        "auto_width_mm": 215.9692412850307,
        "slot_1_1_color_name": "Warm Green",
        "slot_1_1_slot_area_sum_mm2": 692.6410637773947,
        "side_1_height_mm": 21.71636390686035,
        "side_1_length_mm": 26.551648057997227,
        "side_2_height_mm": 33.592500418424606,
        "side_2_length_mm": 31.726250395178795,
    }

    client.publish("v1/devices/me/attributes", json.dumps(attributes), qos=1)
    print("Published attributes:", attributes)
    
    time.sleep(10)  # รอ 10 วินาที ก่อนส่งข้อมูลใหม่