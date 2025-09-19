from pathlib import Path
from watchdog.events import FileSystemEventHandler, FileSystemEvent, FileMovedEvent
from watchdog.observers.polling import PollingObserver as Observer

import subprocess
import time
import sys

WATCHED_DIR = [
    Path(r"R:\01-Organize\01-Management\01-Data Center\Brisk\06-AI & Machine Learning (D0340)\04-IOT_Smartfarm\picture_original_sideview_smartfarm"),
    Path(r"R:\01-Organize\01-Management\01-Data Center\Brisk\06-AI & Machine Learning (D0340)\04-IOT_Smartfarm\picture_original_topview_smartfarm"),
]
EXTS = {".png", ".jpg", ".jpeg"}
SKIP_SUFFIXES = {".tmp", ".partial", ".crdownload"}

# ป้องกันยิงซ้ำไฟล์เดิมในเวลาไล่เลี่ย
_LAST_RUN_TS: dict[str, float] = {}
MIN_GAP_SEC = 2.5

def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{ts}] {msg}", flush=True)

def is_stable_file(p: Path, checks: int = 2, wait_sec: float = 1.0) -> bool:
    """ไฟล์ต้องนิ่ง (ขนาดไม่เปลี่ยน) ต่อเนื่อง 'checks' ครั้ง"""
    if not p.exists():
        return False
    last = -1
    for i in range(checks):
        try:
            sz = p.stat().st_size
        except FileNotFoundError:
            return False
        if sz <= 0:
            log(f"  - size=0 (try {i+1}/{checks}) -> wait")
            time.sleep(wait_sec)
            continue
        if last != -1 and sz != last:
            log(f"  - size changed {last} -> {sz} (try {i+1}/{checks}) -> wait")
            last = sz
            time.sleep(wait_sec)
            continue
        last = sz
        time.sleep(wait_sec)
    try:
        stable = p.exists() and p.stat().st_size == last
    except FileNotFoundError:
        stable = False
    log(f"  - stable={stable}, final size={last}")
    return stable

def infer_view_from_path(p: Path) -> str | None:
    s = str(p).lower()
    parent_names = [pp.name.lower() for pp in list(p.parents)[:4]]
    joined = " ".join([s] + parent_names)
    if any(k in joined for k in ["sideview", "side_view", "sideview_smartfarm"]) or "side" in parent_names:
        return "side"
    if any(k in joined for k in ["topview", "top_view", "topview_smartfarm"]) or "top" in parent_names:
        return "top"
    return None

def should_skip(p: Path) -> bool:
    # ตัดไฟล์ที่ต่อท้ายด้วย suffix ชั่วคราว หรือสกุลไม่ตรง
    low = str(p).lower()
    if any(low.endswith(suf) for suf in SKIP_SUFFIXES):
        return True
    if p.suffix.lower() not in EXTS:
        return True
    return False

def debounce(path_str: str) -> bool:
    now = time.time()
    last = _LAST_RUN_TS.get(path_str, 0.0)
    if now - last < MIN_GAP_SEC:
        return True  # skip
    _LAST_RUN_TS[path_str] = now
    return False

def run_pipeline(p: Path):
    log(f"Event for: {p}")
    if should_skip(p):
        log("  - skip: suffix/ext not allowed")
        return
    if not p.exists():
        log("  - skip: path not exists (race)")
        return
    if debounce(str(p)):
        log("  - skip: debounced")
        return
    if not is_stable_file(p, checks=3, wait_sec=2.0):
        log("  - skip: not stable")
        return
    cmd = [sys.executable, "auto_run.py", "--input", str(p)]
    view = infer_view_from_path(p)
    if view:
        cmd += ["--view", view]  # ให้ pipeline ชัดเจนว่า top/side
        
    project_root = Path(__file__).resolve().parent
    log(f"  - RUN: {' '.join(cmd)} (cwd={project_root})")
    try:
        subprocess.run(cmd, check=False, cwd=project_root)
    except Exception as e:
        log(f"  - ERROR running pipeline: {e}")
        
class Handler(FileSystemEventHandler):
    def on_created(self, event: FileSystemEvent):
        if event.is_directory:
            return
        log(f"on_created: {event.src_path}")
        try:
            run_pipeline(Path(event.src_path))
        except Exception as e:
            log(f"  - ERROR in on_created: {e}")

    def on_moved(self, event: FileMovedEvent):
        if event.is_directory:
            return
        log(f"on_moved: {event.src_path} -> {event.dest_path}")
        try:
            run_pipeline(Path(event.dest_path))
        except Exception as e:
            log(f"  - ERROR in on_moved: {e}")

    '''def on_modified(self, event: FileSystemEvent):
        if event.is_directory:
            return
        log(f"on_modified: {event.src_path}")
        try:
            run_pipeline(Path(event.src_path))
        except Exception as e:
            log(f"  - ERROR in on_modified: {e}")'''


if __name__ == "__main__":
    observer = Observer()
    for d in WATCHED_DIR:
        d.mkdir(parents=True, exist_ok=True)
        log(f"Watching directory: {d}")
        observer.schedule(Handler(), str(d), recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()