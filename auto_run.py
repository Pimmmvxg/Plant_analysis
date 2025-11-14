import argparse
from pathlib import Path
from ultralytics import YOLO
import time
import sys
import json

from plant_analyze import cli
from plant_analyze import config as cfg  
from plant_anomaly import infer_with_mask
from plant_anomaly.infer_with_mask import CFG as anomaly_cfg

try:
    from Thingsboard.tb_client import ThingsboardClient, ACCESS_TOKEN
    TB_CLIENT_ENABLED = True
except ImportError:
    print("ไม่พบ tb_client.py ข้ามการส่งข้อมูล")
    TB_CLIENT_ENABLED = False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=False, help="Path to input image file or folder")
    ap.add_argument("--output", help="Path to output directory (optional)")
    ap.add_argument("--view", choices=["side", "top"], help="View type: 'side' or 'top' (optional)")
    args = ap.parse_args()

    if args.input:
        input_path = Path(args.input)
    elif cfg.INPUT_PATH:
        input_path = Path(cfg.INPUT_PATH) 
    else:
        ap.error("--input is required (or set config.INPUT_PATH in config.py)")

    if not input_path.exists():
        ap.error(f"Input path not found: {input_path}")
        
    print(f" - Input file: {input_path.name}")
    
    # Config Plant Analysis
    output_dir = Path(args.output) if args.output else None
    cfg.resolve_runtime(input_path=input_path, output_dir=output_dir, view=args.view)
    
    # Config & Load : Anomaly Detection
    anomaly_out_dir = Path(cfg.OUTPUT_DIR) / "anomaly_results"
    infer_with_mask.ensure_dir(anomaly_out_dir)
    
    print(f" - Loading YOLO model from: {anomaly_cfg['weights']}")
    try:
        anomaly_model = YOLO(anomaly_cfg["weights"])
        if anomaly_cfg.get("force_class_names"):
            anomaly_model.names = anomaly_cfg["force_class_names"]
    except Exception as e:
        print(f"[Pipeline ERROR] ไม่สามารถโหลดโมเดล YOLO ได้: {e}", file=sys.stderr)
        sys.exit(1)
        
    # Run Plant analysis (Generate Mask)
    print(f"กำลังวิเคราะห์พืช ...(Plant Analysis)...")
    try:
        cli.main()
    except Exception as e:
        print(f"[Pipeline EROR] Plant Analysis failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc() 
        sys.exit(1) #Plant Analysis failed หยุดทำงาน
    
    print(f"Plant Analysis success")
    
    # find mask for anomaly detection
    print(f"find mask for anomaly detection")
    
    # Path Mask
    mask_path = Path(cfg.OUTPUT_DIR) / "processed" / f"{input_path.stem}_mask.png"
    
    print(f" - พบ {mask_path.name}")
    
    # Run Anomaly Detection
    print(f"กำลังตรวจจับความผิดปกติ (Anomaly Detection)")
    
    try:
        summary = infer_with_mask.infer_one(
            model=anomaly_model,
            cfg=anomaly_cfg,
            image_path=input_path,
            out_dir=anomaly_out_dir,
            mask_input=mask_path
        )
    except Exception as e:
        print(f"[Pipeline ERROR] Anomaly Detection failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    print(f"Anomaly detection success")
    print(f"  - พบ {summary.get('num_detections', 0)} จุด")
    print(f"  - บันทึกผลลัพธ์ไปที่: {anomaly_out_dir}")
        
    if TB_CLIENT_ENABLED:    
        print("ส่งข้อมูลความผิดปกติ ขึ้น Thingsboard")
        tb_client = None
        try:
            telemetry_data = {
                "anomaly_detections": summary.get('num_detections', 0)
            }
            tb_client = ThingsboardClient(token=ACCESS_TOKEN)
            tb_client.publish_telemetry(telemetry_data)
            
            print(f"send data success")
        
        except Exception as e:
            print(f"send data failed:{e}", file=sys.stderr)

        finally:
            if tb_client:
                tb_client.stop()
                
    print("="*50 + "\n")
    
    

if __name__ == "__main__":
    main()
