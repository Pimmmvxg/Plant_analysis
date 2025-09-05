import json, time, math
from pathlib import Path
from .tb_client import ThingsboardClient

def publish_data(json_path: str | Path,
                 attribute_keys = ("side_1_height_mm", "side_1_length_mm",
                                  "side_2_height_mm", "side_2_length_mm",
                                  "global_color_name", 
                                  "slot_1_1_color_name", "slot_1_1_slot_area_sum_mm2",
                                  "slot_1_2_color_name", "slot_1_2_slot_area_sum_mm2",
                                  "slot_1_3_color_name", "slot_1_3_slot_area_sum_mm2",
                                  "slot_1_4_color_name", "slot_1_4_slot_area_sum_mm2",
                                  "slot_2_1_color_name", "slot_2_1_slot_area_sum_mm2",
                                  "slot_2_2_color_name", "slot_2_2_slot_area_sum_mm2",
                                  "slot_2_3_color_name", "slot_2_3_slot_area_sum_mm2",
                                  "slot_2_4_color_name", "slot_2_4_slot_area_sum_mm2"
                                  ),
                 ):
    p = Path(json_path)
    if not p.is_file():
        raise FileNotFoundError(f"Result JSON not found: {p}")
    
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
        
    attrs = {}
    missing = []
    for key in attribute_keys:
        if key in data:
            val = data[key]
            # กัน NaN/Inf
            if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                continue
            attrs[key] = val
        else:
            missing.append(key)

    if not attrs:
        raise ValueError(f"No attribute keys found in JSON. Expected any of: {attribute_keys}")

    # 4) ส่งขึ้น ThingsBoard
    cli = ThingsboardClient()
    try:
        cli.publish_attributes(attrs)
    finally:
        try:
            cli.stop()
        except Exception:
            pass

    return {"attributes": attrs, "missing": missing, "json_path": str(p)}