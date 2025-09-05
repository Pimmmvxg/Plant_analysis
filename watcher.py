from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from pathlib import Path
import subprocess
import time

WATCHED_DIR = Path(r"R:\01-Organize\01-Management\01-Data Center\Brisk\06-AI & Machine Learning (D0340)\04-IOT_Smartfarm\picture_original_sideview_smartfarm"), 
(r"R:\01-Organize\01-Management\01-Data Center\Brisk\06-AI & Machine Learning (D0340)\04-IOT_Smartfarm\picture_original_topview_smartfarm")

EXTS = {".png", ".jpg", ".jpeg"}

class Handler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory:
            p = Path(event.src_path)
            if p.suffix.lower() in EXTS:
                time.sleep(1)  # wait for file to be fully written
                subprocess.run(["python", "auto_run.py", "--input", str(p)], check=False)
                
if __name__ == "__main__":
    observer = Observer()
    observer.schedule(Handler(), str(WATCHED_DIR), recursive=True)
    observer.start()
    print(f"Watching directory: {WATCHED_DIR}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()