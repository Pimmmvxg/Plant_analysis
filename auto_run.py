import argparse
from pathlib import Path
from plant_analyze.config import resolve_runtime
from plant_analyze import cli

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to input image file")
    ap.add_argument("--output", help="Path to output directory (optional)")
    ap.add_argument("--view", choices=["side", "top"], help="View type: 'side' or 'top' (optional)")
    args = ap.parse_args()
    
    resolve_runtime(input_path=Path(args.input),
                    output_dir=Path(args.output),
                    view=args.view)
    cli.main()

if __name__ == "__main__":
    cli.main()

