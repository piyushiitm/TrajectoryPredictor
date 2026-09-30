"""Animated replay of trip1 through the v3 engine, markers moving live.

trip1 is the honest case: its phone, mount and vehicle were excluded from every
training pool, so nothing here has seen this ride.

Four tracks are integrated from the SAME anchor, differing only in heading, so
the spread between them is exactly what each component contributes:

  green   GNSS truth
  purple  gyro-only heading          (what drifts)
  orange  gyro + hard-iron compass   (what the fusion fixes)
  red     fusion + learned correction (the v3 ML head)

Speed comes from the v3 model throughout. One anchor at the start, nothing from
GNSS afterwards -- a full-length blackout, not a rolling window.

Writes an animated GIF, plus a PNG of the final frame.
"""
import config
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.insert(0, str(Path(__file__).parent))
from sim_compass_pipeline import compass, tau_for
from idr_core import (load_session_gps, genuine_fixes, gyro_matrix, calibrate_yaw,
                      yaw_rate_from_cal, truth_xy, model_speed_track2, wrap, FS)

ap = argparse.ArgumentParser()
ap.add_argument("--trip", default="t03_other")
ap.add_argument("--speed-model", default=str(config.SPEED_GENERAL))
ap.add_argument("--head-model", default=str(config.HEADING))
ap.add_argument("--cal-s", type=float, default=300.0)
ap.add_argument("--seconds", type=float, default=0.0, help="0 = to the end")
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--out", default="results/trip1_live")
a = ap.parse_args()

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

st = json.load(open(f"{a.speed_model}/norm_stats.json"))
MU, SD, CL = np.array(st["mean"]), np.array(st["std"]), st["input_cols"]
NET = tf.keras.models.load_model(f"{a.speed_model}/model.keras")
hs = json.load(open(f"{a.head_model}/norm_stats.json"))
HMU, HSD = np.array(hs["mean"]), np.array(hs["std"])
HNET = tf.keras.models.load_model(f"{a.head_model}/model.keras")

p = f"data/raw/{a.trip}.csv"
rs, truth = load_session_gps(p)
if rs is None:
    raise SystemExit(f"{a.trip}: unusable")
psi, _ = compass(p, rs)
t = rs["t_ms"].values / 1000.0
n = len(rs)
hdg, spd = truth["hdg"], truth["spd"]
te_, tn_ = truth_xy(truth)
fx = genuine_fixes(rs)
W = gyro_matrix(rs)

A = int(a.cal_s * FS)
B = n - 1 if a.seconds <= 0 else min(n - 1, A + int(a.seconds * FS))
w4, q = calibrate_yaw(W, t, hdg, fx, 0, A)
if w4 is None:
    raise SystemExit("calibration failed")
m = B - A
rate = yaw_rate_from_cal(W[A:B], w4)

v = model_speed_track2(rs, NET, MU, SD, CL, A, A, B, spd[A])
if v is None:
    v = np.full(m, spd[A])

h_gyro = hdg[A] + np.cumsum(rate) / FS
if psi is not None:
    off = wrap(hdg[A] - psi[A])
    hm = psi[A:B] + off
    h_fus = np.empty(m)
    cur = hdg[A]
    for i in range(m):
        al = (1 / FS) / (tau_for(i / FS) + 1 / FS)
        cur += rate[i] / FS
        cur += al * wrap(hm[i] - cur)
        h_fus[i] = cur
else:
    h_fus = h_gyro.copy()

# learned correction on top of the fusion, applied at 2 Hz as on the phone
h_ml = h_fus.copy()
try:
    from run_head_correct import build_head
    r = build_head(a.trip, hop=10)
    if r is not None:
        Xh, Yh, Th = r
        c = HNET.predict((Xh - HMU) / HSD, batch_size=16384, verbose=0).ravel()
        idx = np.clip(((Th - t[A]) * FS).astype(int), 0, m - 1)
        corr = np.zeros(m)
        ok = (Th >= t[A]) & (Th <= t[B - 1])
        if ok.sum() > 10:
            corr = np.interp(np.arange(m), idx[ok], np.clip(c[ok], -0.5, 0.5))
        h_ml = h_fus + corr
except Exception as e:
    print(f"  heading correction unavailable: {e}")

tracks = {}
for nm, h in (("gyro", h_gyro), ("fused", h_fus), ("ml", h_ml)):
    e = np.cumsum(v * np.sin(h)) / FS
    nn = np.cumsum(v * np.cos(h)) / FS
    tracks[nm] = (e, nn)
E_t = te_[A:B] - te_[A]
N_t = tn_[A:B] - tn_[A]
dist = np.cumsum(spd[A:B]) / FS

print(f"  {a.trip}: {m/FS:.0f}s blackout, {dist[-1]:.0f} m travelled, cal R2 {q:.2f}")
for nm, (e, nn) in tracks.items():
    err = np.hypot(e[-1] - E_t[-1], nn[-1] - N_t[-1])
    print(f"    {nm:6s} final drift {err:8.1f} m  ({100*err/dist[-1]:5.1f}%)")

STEP = max(1, m // (a.fps * 25))
fr = range(1, m, STEP)
plt.style.use("dark_background")
fig, (ax, bx) = plt.subplots(1, 2, figsize=(14, 6.4),
                             gridspec_kw={"width_ratios": [1.5, 1]})
ax.set_aspect("equal")
ax.set_title(f"{a.trip} — full GNSS blackout, single anchor")
ax.set_xlabel("east (m)"); ax.set_ylabel("north (m)"); ax.grid(alpha=.15)
bx.set_title("drift from truth"); bx.set_xlabel("distance travelled (m)")
bx.set_ylabel("error (m)"); bx.grid(alpha=.15)

allx = np.r_[E_t, tracks["gyro"][0], tracks["fused"][0], tracks["ml"][0]]
ally = np.r_[N_t, tracks["gyro"][1], tracks["fused"][1], tracks["ml"][1]]
pad = 0.08 * max(np.ptp(allx), np.ptp(ally), 50)
ax.set_xlim(allx.min() - pad, allx.max() + pad)
ax.set_ylim(ally.min() - pad, ally.max() + pad)

COL = {"truth": "#2ecc71", "gyro": "#9b59b6", "fused": "#e67e22", "ml": "#e74c3c"}
LBL = {"truth": "GNSS truth", "gyro": "gyro only", "fused": "gyro+compass",
       "ml": "fusion + ML"}
lines, dots, elines = {}, {}, {}
for k in ("truth", "gyro", "fused", "ml"):
    lines[k], = ax.plot([], [], color=COL[k], lw=3 if k == "truth" else 2, label=LBL[k])
    dots[k], = ax.plot([], [], "o", color=COL[k], ms=10 if k == "truth" else 7)
    if k != "truth":
        elines[k], = bx.plot([], [], color=COL[k], lw=2, label=LBL[k])
ax.legend(loc="best", framealpha=.25)
bx.legend(loc="upper left", framealpha=.25)
errs = {k: np.hypot(tracks[k][0] - E_t, tracks[k][1] - N_t) for k in tracks}
bx.set_xlim(0, dist[-1])
bx.set_ylim(0, max(e.max() for e in errs.values()) * 1.1 + 1)
txt = bx.text(.97, .04, "", transform=bx.transAxes, ha="right", family="monospace")


def draw(i):
    lines["truth"].set_data(E_t[:i], N_t[:i])
    dots["truth"].set_data([E_t[i - 1]], [N_t[i - 1]])
    for k in ("gyro", "fused", "ml"):
        e, nn = tracks[k]
        lines[k].set_data(e[:i], nn[:i])
        dots[k].set_data([e[i - 1]], [nn[i - 1]])
        elines[k].set_data(dist[:i], errs[k][:i])
    txt.set_text(f"t {i/FS:6.0f}s   {dist[i-1]:7.0f} m\n"
                 f"gyro  {errs['gyro'][i-1]:8.1f} m\n"
                 f"fused {errs['fused'][i-1]:8.1f} m\n"
                 f"ML    {errs['ml'][i-1]:8.1f} m")
    return list(lines.values()) + list(dots.values()) + list(elines.values()) + [txt]


Path(a.out).parent.mkdir(parents=True, exist_ok=True)
draw(m)
fig.tight_layout()
fig.savefig(a.out + ".png", dpi=130)
print(f"  wrote {a.out}.png")
an = FuncAnimation(fig, draw, frames=fr, blit=True)
an.save(a.out + ".gif", writer=PillowWriter(fps=a.fps))
print(f"  wrote {a.out}.gif  ({len(list(fr))} frames)")
