"""Download the recordings from the shared Google Drive folder into data/recordings/.

Run at deploy time (Render's build command) so the replay service has the trips
that GitHub can't carry (~1.1 GB, see data/MANIFEST.csv):

    python server/fetch_data.py

The folder must be shared as "Anyone with the link -> Viewer". Override the
folder with DATA_FOLDER_URL. A failed download does not fail the deploy: the
service still starts, and judges can upload a recording from the page instead.
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

DEFAULT_URL = "https://drive.google.com/drive/folders/1t_RECqwdEEg5rAYrYMfqriabc7RVvW0l"
ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "data" / "recordings"


def main():
    url = os.environ.get("DATA_FOLDER_URL", DEFAULT_URL)
    DEST.mkdir(parents=True, exist_ok=True)
    try:
        import gdown
    except ImportError:
        print("fetch_data: gdown is not installed -- skipping download")
        return 0
    with tempfile.TemporaryDirectory() as tmp:
        try:
            gdown.download_folder(url=url, output=tmp, quiet=False, use_cookies=False,
                                  remaining_ok=True)
        except Exception as e:                      # rate limit, not shared, ...
            print(f"fetch_data: download failed ({e}); continuing without recordings")
        n = 0
        for p in Path(tmp).rglob("*.csv"):          # flatten any sub-folders
            target = DEST / p.name
            if not target.exists():
                shutil.move(str(p), target)
                n += 1
    have = sorted(q.name for q in DEST.glob("*.csv"))
    print(f"fetch_data: {n} new, {len(have)} recordings in {DEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
