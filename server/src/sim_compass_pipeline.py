"""End-to-end drift with the LEARNED COMPASS FILTER in the loop.

Five heading sources, same speed model throughout, so only heading varies:

  gyro     calibrated-gyro integration
  mag      hard-iron compass, spike-rejected            (what the app ships)
  fused    gyro + mag, tau shrinking with elapsed time  (what the app ships)
  mag_f    compass with the learned filter applied
  fused_f  gyro + filtered compass

The filter failed its offline check -- +47.3% on held-out tails but -3.6%
leave-one-out, and -22.6% to -75.5% on unseen rides -- which says it memorised
route-specific magnetic landmarks rather than a general disturbance law. This
run asks the only question that still matters: does that failure actually show
up as worse POSITION, or does the fusion absorb it?

trip11 is the honest test: never in the filter's training pool. mount_e was,
so its numbers are optimistic and are marked.
"""
import config
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from run_full_job import log
from run_compass_filter import COLS
from idr_core import (load_session_gps, genuine_fixes, gyro_matrix, calibrate_yaw,
                      yaw_rate_from_cal, truth_xy, model_speed_track2, wrap, FS)

B_TOL = 0.15
import os
SPEED_MODEL = os.environ.get("IDR_SPEED", "results/models/mount_only")
FILTER_MODEL = "results/models/compass_filter"
TRIPS = sys.argv[1:] or ["s11_bike", "s05_mount", "t03_other"]
LEAKED = {"s05_mount"}


def compass(path, rs, filt=None, fmu=None, fsd=None):
    """Compass heading on the sample grid: raw and (optionally) filtered."""
    raw = pd.read_csv(path)
    raw.columns = [c.strip() for c in raw.columns]
    try:
        M = np.column_stack([raw[f'MAGNETIC FIELD {k} (μT)'].values for k in 'XYZ'])
        G = np.column_stack([raw[f'GRAVITY {k} (m/s²)'].values for k in 'XYZ'])
        W = np.column_stack([raw['GYROSCOPE Yaw (rad/s)'].values,
                             raw['GYROSCOPE Pitch (rad/s)'].values,
                             raw['GYROSCOPE Roll (rad/s)'].values])
    except KeyError:
        return None, None
    tr = raw['TIME SINCE START (ms)'].values / 1000.0
    ok = np.isfinite(M).all(1) & np.isfinite(G).all(1) & (np.abs(M).sum(1) > 1e-6)
    if ok.sum() < 2000:
        return None, None
    c = np.linalg.lstsq(np.column_stack([2 * M, np.ones(len(M))])[ok],
                        np.sum(M[ok] ** 2, axis=1), rcond=None)[0][:3]
    Mc = M - c
    b = np.linalg.norm(Mc, axis=1)
    b0 = np.median(b[ok])
    gn = G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-9)
    h = Mc - np.sum(Mc * gn, axis=1)[:, None] * gn
    e1 = np.column_stack([1 - gn[:, 0] ** 2, -gn[:, 0] * gn[:, 1], -gn[:, 0] * gn[:, 2]])
    e1 = e1 / (np.linalg.norm(e1, axis=1, keepdims=True) + 1e-9)
    e2 = np.cross(gn, e1)
    hx = np.sum(h * e1, axis=1)
    hy = np.sum(h * e2, axis=1)
    psi = np.arctan2(hy, hx)
    hz = np.sum(Mc * gn, axis=1)
    inc = np.arctan2(hz, np.hypot(hx, hy))

    t_rs = rs["t_ms"].values / 1000.0
    clean = ok & (np.abs(b - b0) / b0 <= B_TOL)       # spike rejection
    if clean.sum() < 500:
        return None, None
    psi_raw = np.interp(t_rs, tr[clean], np.unwrap(psi[clean]))

    psi_filt = None
    if filt is not None:
        fs = 1.0 / np.median(np.diff(tr))
        win = max(2, int(round(fs)))
        b_std = pd.Series(b).rolling(win, min_periods=1, center=True).std().to_numpy()
        db = np.gradient(b, tr)
        spd = np.interp(tr, t_rs, rs["gps_speed_kmh"].values / 3.6)
        X = np.column_stack([
            b, (b - b0) / max(b0, 1e-6), db, np.nan_to_num(b_std),
            hz, inc, inc - np.median(inc[ok]), np.hypot(hx, hy), hx, hy,
            gn[:, 0], gn[:, 1], gn[:, 2],
            np.arccos(np.clip(gn[:, 2], -1, 1)),
            np.linalg.norm(W, axis=1), W[:, 0], W[:, 1], W[:, 2], spd,
        ]).astype("float32")
        X = np.nan_to_num(X)
        corr = filt.predict((X - fmu) / fsd, batch_size=32768, verbose=0).ravel()
        # the filter predicts the compass's error, so ADD it back
        psi_filt = np.interp(t_rs, tr[ok], np.unwrap((psi + corr)[ok]))
    return psi_raw, psi_filt


def tau_for(el):
    return 10.0 if el < 60 else max(2.0, 10.0 - (el - 60) * 8.0 / 240.0)


def main():
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')

    st = json.load(open(f"{SPEED_MODEL}/norm_stats.json"))
    MU = np.array(st["mean"]); SD = np.array(st["std"]); CL = st["input_cols"]
    NET = tf.keras.models.load_model(f"{SPEED_MODEL}/speed_model.keras")
    fn = json.load(open(f"{FILTER_MODEL}/norm.json"))
    FMU = np.array(fn["mean"]); FSD = np.array(fn["std"])
    FILT = tf.keras.models.load_model(f"{FILTER_MODEL}/compass.keras")
    log(f"speed model {SPEED_MODEL}   compass filter {FILTER_MODEL}")

    for T in TRIPS:
        p = str(config.recording(T))
        if not Path(p).exists():
            log(f"{T}: missing"); continue
        rs, truth = load_session_gps(p)
        if rs is None:
            log(f"{T}: unusable"); continue
        psi, psi_f = compass(p, rs, FILT, FMU, FSD)
        if psi is None:
            log(f"{T}: no compass"); continue
        t = rs["t_ms"].values / 1000.0; n = len(rs)
        hdg = truth["hdg"]; sp = truth["spd"]
        te_, tn_ = truth_xy(truth); fx = genuine_fixes(rs); W = gyro_matrix(rs)
        cal = int(300 * FS)
        tag = "  (*) in the filter's training pool" if T in LEAKED else "  [CLEAN]"
        log("")
        log("=" * 72)
        log(f"=== {T}{tag}")
        log("=" * 72)
        for D in (10.0, 30.0, 60.0, 0.0):
            step = (n - cal - 1) if D == 0 else int(D * FS)
            if step < 20 or cal + step >= n:
                continue
            acc = {}
            dists = []
            for a in range(cal, max(cal + 1, n - step), step):
                b_ = a + step
                if b_ >= n or sp[a:b_].mean() < 3.0:
                    continue
                dist = np.trapezoid(sp[a:b_], dx=1 / FS)
                if dist < 30:
                    continue
                w4, _ = calibrate_yaw(W, t, hdg, fx, a - cal, a)
                if w4 is None:
                    continue
                rate = yaw_rate_from_cal(W[a:b_], w4)
                hs = {"gyro": hdg[a] + np.cumsum(rate) / FS}
                for nm, ps in (("mag", psi), ("mag_f", psi_f)):
                    if ps is None:
                        continue
                    off = wrap(hdg[a] - ps[a])
                    hm = ps[a:b_] + off
                    hs[nm] = hm
                    cur = hdg[a]; hf = np.empty(step)
                    for i in range(step):
                        al = (1 / FS) / (tau_for(i / FS) + 1 / FS)
                        cur += rate[i] / FS
                        cur += al * wrap(hm[i] - cur)
                        hf[i] = cur
                    hs["fused" if nm == "mag" else "fused_f"] = hf
                v = model_speed_track2(rs, NET, MU, SD, CL, a, a, b_, sp[a])
                if v is None:
                    v = np.full(step, sp[a])
                dE, dN = te_[b_ - 1] - te_[a], tn_[b_ - 1] - tn_[a]
                dists.append(dist)
                for k, hh in hs.items():
                    e = np.cumsum(v * np.sin(hh)) / FS
                    nn_ = np.cumsum(v * np.cos(hh)) / FS
                    err = np.hypot(e[-1] - dE, nn_[-1] - dN)
                    acc.setdefault(k, []).append(100 * err / dist)
                    acc.setdefault(k + "#m", []).append(err)
            if not acc:
                continue
            lbl = "FULL" if D == 0 else f"{D:.0f}s"
            log(f"  --- {lbl}  n={len(dists)}  median distance {np.median(dists):.0f} m ---")
            for k in ("gyro", "mag", "fused", "mag_f", "fused_f"):
                if k in acc:
                    log(f"    {k:9s} {np.median(acc[k]):7.1f}%  {np.median(acc[k+'#m']):8.1f} m")
    log("DONE")


if __name__ == "__main__":
    main()
