import argparse
from pathlib import Path

from plant_analyze import cli
from plant_analyze import config as cfg  

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

    output_dir = Path(args.output) if args.output else None

    cfg.resolve_runtime(input_path=input_path, output_dir=output_dir, view=args.view)

    cli.main()

if __name__ == "__main__":
    main()
