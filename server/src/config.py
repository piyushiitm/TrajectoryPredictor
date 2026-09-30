"""Repository paths and the recording roster, in one place.

Scripts used to hardcode `data/raw/mount_a.csv` and `results/models/v3/speed`,
which tied them to the working tree they grew up in. Everything resolves from
the repository root now, so a checkout runs without editing paths.

Recording names changed with the move: sessions captured simultaneously share a
suffix, so `s07_mount`, `s07_pocket` and `s07_hand` are one ride recorded on
three phones at once. That is what makes mount placement separable from route,
traffic and rider -- with the old timestamped names the grouping was invisible.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "recordings"
UPLOADS = ROOT / "data" / "uploads"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
CACHE = ROOT / "data" / "cache"
for _d in (UPLOADS, CACHE, RESULTS):
    _d.mkdir(parents=True, exist_ok=True)

SPEED_TUNED = MODELS / "speed_tuned"          # bike only -- sharper on that bike
SPEED_GENERAL = MODELS / "speed_general"      # + IO-VNBD cars -- generalises
HEADING = MODELS / "heading_correction_gyro"  # corrects the gyro's residual

# Every mounted, pocketed and hand-held recording used for training.
MOUNTS = [f"s{n:02d}_mount" for n in range(1, 10)]
POCKETS = [f"s{n:02d}_pocket" for n in range(1, 10)]
HANDS = ["s07_hand", "s08_hand", "s09_hand", "s10_hand"]
LEGACY = ["s11_bike", "s12_bike", "s13_bike"]
TRAIN = MOUNTS + POCKETS + HANDS + LEGACY

# Held out of every pool. t03_other is a different phone, mount and vehicle --
# the hardest case, and the one worth quoting.
TEST = ["t01_mount", "t02_mount", "t03_other"]
HOLDOUT = "t03_other"


def recording(name):
    """Path to a recording, whether it shipped with the repo or was uploaded."""
    for d in (DATA, UPLOADS):
        p = d / f"{name}.csv"
        if p.exists():
            return p
    raise FileNotFoundError(f"no recording named {name} in {DATA} or {UPLOADS}")
