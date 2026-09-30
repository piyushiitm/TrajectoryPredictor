"""Which recording is the "9.4 km single-anchor run" (gyro 96.2%, mag 15.8%, fused 9.1%)?

The app README and IdrEngine/MagCompass comments quote that run, but no log in
results/ records which recording it was. This re-runs the replay engine on every
recording with ONE anchor -- 300 s of calibration, then no GNSS to the end of the
file -- and prints trip length and end-of-trip drift for each heading source.

    ./venv/bin/python server/find_single_anchor.py            # all recordings
    ./venv/bin/python server/find_single_anchor.py s05_mount  # just some

Rows closest to 9.4 km come first. The original number came from an earlier
speed model, so expect the percentages to be close to, not exactly, 96/16/9.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import server as S                                  # noqa: E402  (loads the models)

TARGET_KM = 9.4
names = sys.argv[1:] or [t["name"] for t in S.trips()]
rows = []
for nm in names:
    try:
        r = S.run_engine(nm, cal_s=300.0, start_s=None, duration_s=0.0, model="tuned")
    except Exception as e:                          # too short, no GPS, calibration failed
        print(f"  {nm:12s} skipped: {getattr(e, 'detail', e)}")
        continue
    f = r["final"]
    pct = lambda k: f[k]["pct"] if k in f else float("nan")
    rows.append((abs(r["distance_m"] / 1000 - TARGET_KM), nm, r["distance_m"] / 1000,
                 r["cal_r2"], pct("gyro"), pct("mag"), pct("fused"), pct("mlgyro")))
    print(f"  {nm:12s} done", flush=True)

print(f"\n{'recording':12s} {'km':>6s} {'cal R2':>7s} {'gyro':>7s} {'mag':>7s} {'fused':>7s} {'ML gyro':>8s}")
for _, nm, km, r2, g, m, fu, ml in sorted(rows):
    tag = "   <-- 96.2 / 15.8 / 9.1 ?" if abs(km - TARGET_KM) < 0.6 else ""
    print(f"{nm:12s} {km:6.2f} {r2:7.3f} {g:6.1f}% {m:6.1f}% {fu:6.1f}% {ml:7.1f}%{tag}")
