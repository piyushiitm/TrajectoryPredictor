"""Download the recordings from the shared Google Drive folder into data/recordings/.

Runs at deploy time (Render's build command), and again from the server on
startup if the build left data/recordings/ empty:

    python server/fetch_data.py

The folder must be shared as "Anyone with the link -> Viewer". Override it with
DATA_FOLDER_URL. CSVs are picked up anywhere in the folder tree, and .zip files
are unpacked. Everything that happens is written to data/fetch_log.txt, which
the server shows at /api/status.
"""
import os
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path

DEFAULT_URL = "https://drive.google.com/drive/folders/1t_RECqwdEEg5rAYrYMfqriabc7RVvW0l"
ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "data" / "recordings"
LOG = ROOT / "data" / "fetch_log.txt"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def main():
    url = os.environ.get("DATA_FOLDER_URL", DEFAULT_URL)
    DEST.mkdir(parents=True, exist_ok=True)
    log(f"fetching {url}")
    try:
        import gdown
    except ImportError:
        log("gdown is not installed -- skipping download")
        return 0
    with tempfile.TemporaryDirectory() as tmp:
        try:
            got = gdown.download_folder(url=url, output=tmp, quiet=False, use_cookies=False)
            log(f"gdown returned {len(got or [])} files")
        except Exception as e:                      # not shared, rate limited, ...
            log(f"download failed: {type(e).__name__}: {e}")
        files = [p for p in Path(tmp).rglob("*") if p.is_file()]
        log("downloaded: " + (", ".join(p.name for p in files[:60]) or "nothing"))
        for z in [p for p in files if p.suffix.lower() == ".zip"]:
            try:
                with zipfile.ZipFile(z) as zf:
                    zf.extractall(z.parent / (z.stem + "_unzipped"))
                log(f"unzipped {z.name}")
            except zipfile.BadZipFile:
                log(f"{z.name} is not a valid zip")
        n = 0
        for p in Path(tmp).rglob("*.csv"):          # flatten any sub-folders
            if p.name.startswith("._"):
                continue                            # macOS metadata inside zips
            target = DEST / p.name
            if not target.exists():
                shutil.move(str(p), target)
                n += 1
    have = sorted(q.name for q in DEST.glob("*.csv"))
    log(f"{n} new, {len(have)} recordings in {DEST}: {', '.join(have)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
