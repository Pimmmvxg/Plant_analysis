def get_color_name(hue):
    if hue < 15 or hue >= 345:
        return "แดง"
    elif hue < 45:
        return "ส้ม"
    elif hue < 75:
        return "เหลือง"
    elif hue < 105:
        return "เขียวอ่อน"
    elif hue < 135:
        return "เขียว"
    elif hue < 165:
        return "เขียวฟ้า"
    elif hue < 195:
        return "ฟ้า"
    elif hue < 225:
        return "น้ำเงิน"
    elif hue < 255:
        return "น้ำเงินเข้ม"
    elif hue < 285:
        return "ม่วง"
    elif hue < 315:
        return "ชมพูม่วง"
    else:
        return 'Unknown'
    
import json

with open(r"C:\Plant_analysis\notebooks\analyze\output.json", "r") as f:
    data = json.load(f)

for i in range(1, 7):
    key = f"default_{i}"
    try:
        hue_median = data["observations"][key]["hue_median"]["value"]
        color_name = get_color_name(hue_median)
        print(f"{key}:Main Color = {color_name}(Hue = {hue_median})")
    except KeyError:
        print(f"{key}: No data")