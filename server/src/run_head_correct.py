"""PART A: head-to-head speed comparison, app_bike vs bike_v2, same test sets.
PART B: learn to CORRECT the gyro+compass fused heading, with GPS bearing as truth.

Part A matters because the two models were previously scored on different test
sets, which is not a comparison. Here both are run on the same held-out tails
and on trip1.

Part B is the one open avenue left. Speed is learned well (corr vy 0.8-0.95),
the lateral velocity component never was (corr vx ~0 across seven attempts), but
HEADING has never been attacked directly on top of the compass fusion. The
fusion already works -- on trip1 it took full-trip drift from 74.6% to 26.5% --
so the question is whether its RESIDUAL against GPS bearing is predictable.

Baselines the correction must beat:
  gyro    calibrated-gyro integration alone
  fused   gyro + hard-iron compass, tau shrinking with elapsed time
A model predicting zero residual reproduces `fused` exactly.
"""
import config
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from run_full_job import BIKE, HOLD, TAIL, GAP_S, collect, log
from preprocess_speed import window_features
from idr_core import (load_session_gps, genuine_fixes, gyro_matrix, calibrate_yaw,
                      yaw_rate_from_cal, wrap, FS)

SCALES = (2.0, 5.0, 10.0)
ELAPSED = (5.0, 15.0, 30.0, 60.0)
B_TOL = 0.15


def mag_psi(path, rs):
    """Hard-iron corrected, tilt-compensated compass heading on the sample grid.

    Readings whose field magnitude departs more than B_TOL from the running norm
    are dropped: a passing vehicle or a steel structure bends the field and
    swings apparent north by up to 90 deg.
    """
    raw = pd.read_csv(path)
    raw.columns = [c.strip() for c in raw.columns]
    try:
        M = np.column_stack([raw[f'MAGNETIC FIELD {k} (μT)'].values for k in 'XYZ'])
        G = np.column_stack([raw[f'GRAVITY {k} (m/s²)'].values for k in 'XYZ'])
    except KeyError:
        return None
    tr = raw['TIME SINCE START (ms)'].values / 1000.0
    ok = np.isfinite(M).all(1) & np.isfinite(G).all(1) & (np.abs(M).sum(1) > 1e-6)
    if ok.sum() < 500:
        return None
    b = np.linalg.norm(M, axis=1)
    c = np.linalg.lstsq(np.column_stack([2 * M, np.ones(len(M))])[ok],
                        np.sum(M[ok] ** 2, axis=1), rcond=None)[0][:3]
    m0 = b[ok].mean()
    clean = ok & (np.abs(b - m0) / m0 <= B_TOL)
    if clean.sum() < 500:
        return None
    Mc = M - c
    gn = G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-9)
    h = Mc - np.sum(Mc * gn, axis=1)[:, None] * gn
    e1 = np.column_stack([1 - gn[:, 0] ** 2, -gn[:, 0] * gn[:, 1], -gn[:, 0] * gn[:, 2]])
    e1 = e1 / (np.linalg.norm(e1, axis=1, keepdims=True) + 1e-9)
    e2 = np.cross(gn, e1)
    psi = np.arctan2(np.sum(h * e2, axis=1), np.sum(h * e1, axis=1))
    return np.interp(rs["t_ms"].values / 1000.0, tr[clean], np.unwrap(psi[clean]))


def tau_for(el):
    """Swept on four recordings: short windows want ~10 s, long ones ~2 s."""
    return 10.0 if el < 60 else max(2.0, 10.0 - (el - 60) * 8.0 / 240.0)


def build_head(nm, hop=5):
    """Window features + gyro and fused heading estimates + the GPS truth."""
    p = config.recording(nm)
    if not p.exists():
        return None
    rs, truth = load_session_gps(p)
    if rs is None:
        return None
    psi = mag_psi(str(p), rs)
    t = rs["t_ms"].values / 1000.0
    n = len(rs)
    hdg = truth["hdg"]
    spd = truth["spd"]
    fx = genuine_fixes(rs)
    W = gyro_matrix(rs)
    lax = rs["accel_x"].values - rs["grav_x"].values
    lay = rs["accel_y"].values - rs["grav_y"].values
    laz = rs["accel_z"].values - rs["grav_z"].values
    gx, gy, gz = rs["gyro_roll"].values, rs["gyro_pitch"].values, rs["gyro_yaw"].values
    cal_n = int(300 * FS)
    wmax = int(max(SCALES) * FS)
    dt = 1.0 / FS
    X, Y, TT = [], [], []
    cache = {}
    for e in range(max(wmax, cal_n), n, hop):
        if not (np.isfinite(hdg[e - 1]) and spd[e - 1] > 3.0):
            continue
        f, bad = [], False
        for s in SCALES:
            sl = slice(e - int(s * FS), e)
            seg = [lax[sl], lay[sl], laz[sl], gx[sl], gy[sl], gz[sl]]
            if not all(np.isfinite(v).all() for v in seg):
                bad = True
                break
            f += window_features(*seg, dt)
        if bad:
            continue
        for el in ELAPSED:
            a = e - int(el * FS)
            if a < cal_n or not np.isfinite(hdg[a]):
                continue
            key = a // (60 * int(FS))
            if key not in cache:
                w4, _ = calibrate_yaw(W, t, hdg, fx, a - cal_n, a)
                cache[key] = w4
            w4 = cache[key]
            if w4 is None:
                continue
            rate = yaw_rate_from_cal(W[a:e], w4)
            dh_g = float(np.sum(rate) / FS)
            if psi is None:
                dh_f = dh_g
            else:
                off = wrap(hdg[a] - psi[a])
                cur = hdg[a]
                for i in range(len(rate)):
                    al = (1 / FS) / (tau_for(i / FS) + 1 / FS)
                    cur += rate[i] / FS
                    cur += al * wrap(psi[a + i] + off - cur)
                dh_f = wrap(cur - hdg[a])
            X.append(f + [float(spd[a]), el, dh_g, dh_f, abs(dh_f),
                          float(rate[-1]) if len(rate) else 0.0])
            Y.append([wrap(hdg[e - 1] - hdg[a]), dh_g, dh_f])
            TT.append(e / FS)
    if not X:
        return None
    return np.asarray(X, "float32"), np.asarray(Y, "float32"), np.asarray(TT)


def main():
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.keras.utils.set_random_seed(1)
    L = tf.keras.layers
    deg = np.rad2deg

    # ---------------- PART A ----------------
    log("#" * 70)
    log("PART A - SPEED: app_bike vs bike_v2 on the SAME test sets")
    log("#" * 70)
    X, Y, S, T = collect(BIKE)
    te = np.zeros(len(X), bool)
    for nm in sorted(set(S.tolist())):
        k = S == nm
        te |= k & (T >= T[k].max() * (1 - TAIL))
    g1 = collect([HOLD])
    X1, Y1 = (g1[0], g1[1]) if g1 else (None, None)
    for tag, mdir in (("app_bike", str(config.SPEED_GENERAL)),
                      ("bike_v2", "results/models/bike_v2")):
        st = json.load(open(f"{mdir}/norm_stats.json"))
        mu = np.array(st["mean"])
        sd = np.array(st["std"])
        net = tf.keras.models.load_model(f"{mdir}/speed_model.keras")
        p = np.clip(Y[te, 2] + net.predict((X[te] - mu) / sd, batch_size=16384,
                                           verbose=0).ravel(), 0, None)
        h = np.abs(Y[te, 2] - Y[te, 0]).mean()
        m = np.abs(p - Y[te, 0]).mean()
        log(f"  {tag:9s} TAILS  hold {h:6.3f}  model {m:6.3f} m/s ({100*(h-m)/h:+6.1f}%)"
            f"  corr {np.corrcoef(p, Y[te, 0])[0, 1]:+.3f}")
        if X1 is not None:
            p1 = np.clip(Y1[:, 2] + net.predict((X1 - mu) / sd, batch_size=16384,
                                                verbose=0).ravel(), 0, None)
            h1 = np.abs(Y1[:, 2] - Y1[:, 0]).mean()
            m1 = np.abs(p1 - Y1[:, 0]).mean()
            log(f"  {tag:9s} trip1  hold {h1:6.3f}  model {m1:6.3f} m/s "
                f"({100*(h1-m1)/h1:+6.1f}%)  corr {np.corrcoef(p1, Y1[:, 0])[0, 1]:+.3f}")
    del X, Y, S, T

    # ---------------- PART B ----------------
    log("")
    log("#" * 70)
    log("PART B - HEADING: learn the residual of the gyro+compass fusion")
    log("#" * 70)
    Xs, Ys, Ss, Ts = [], [], [], []
    for nm in BIKE:
        r = build_head(nm)
        if r is None:
            log(f"  {nm}: no heading samples")
            continue
        Xs.append(r[0])
        Ys.append(r[1])
        Ss.append(np.full(len(r[0]), nm))
        Ts.append(r[2])
        log(f"  {nm}: {len(r[0])} rows")
    if not Xs:
        log("no heading data")
        return
    X = np.concatenate(Xs)
    Y = np.concatenate(Ys)
    S = np.concatenate(Ss)
    T = np.concatenate(Ts)
    te = np.zeros(len(X), bool)
    tr = np.zeros(len(X), bool)
    for nm in sorted(set(S.tolist())):
        k = S == nm
        cut = T[k].max() * (1 - TAIL)
        te |= k & (T >= cut)
        tr |= k & (T < cut - GAP_S)
    log(f"train {tr.sum()} | tails {te.sum()}")

    mu = X[tr].mean(0)
    sd = X[tr].std(0)
    sd = np.where(sd < 1e-6, 1.0, sd) + 1e-8
    Xn = (X - mu) / sd
    resid = wrap(Y[:, 0] - Y[:, 2])          # truth minus FUSED

    m = tf.keras.Sequential([L.Input((X.shape[1],)),
                             L.Dense(256, activation="relu"), L.Dropout(.1),
                             L.Dense(128, activation="relu"), L.Dropout(.1),
                             L.Dense(64, activation="relu"), L.Dense(1)])
    m.compile(tf.keras.optimizers.Adam(1e-3), loss="huber")
    log("training heading-correction head ...")
    m.fit(Xn[tr], resid[tr], validation_split=0.1, epochs=80, batch_size=2048,
          verbose=2, callbacks=[
              tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
              tf.keras.callbacks.ReduceLROnPlateau(patience=4, factor=.5)])

    def sc(mask, tag, Xa=None, Ya=None):
        if Xa is None:
            xs = Xn[mask]
            y = Y[mask]
        else:
            xs = (Xa - mu) / sd
            y = Ya
        pr = y[:, 2] + m.predict(xs, batch_size=16384, verbose=0).ravel()
        eg = deg(np.abs(wrap(y[:, 0] - y[:, 1]))).mean()
        ef = deg(np.abs(wrap(y[:, 0] - y[:, 2]))).mean()
        em = deg(np.abs(wrap(y[:, 0] - pr))).mean()
        best = "MODEL" if em < min(eg, ef) else ("fused" if ef < eg else "gyro")
        log(f"    {tag:12s} gyro {eg:6.2f}  fused {ef:6.2f}  model {em:6.2f} deg"
            f"   vs fused {100*(ef-em)/ef:+6.1f}%   best={best}")

    log("")
    log("=== held-out TAILS ===")
    sc(te, "ALL TAILS")
    for nm in sorted(set(S[te].tolist())):
        k = te & (S == nm)
        if k.sum() > 30:
            sc(k, nm)
    log("")
    log("=== trip1 (never trained on) ===")
    r = build_head(HOLD)
    if r:
        sc(None, "t03_other", r[0], r[1])

    out = config.MODELS / ("head_v2")
    out.mkdir(parents=True, exist_ok=True)
    m.save(out / "head.keras")
    json.dump({"mean": mu.tolist(), "std": sd.tolist()}, open(out / "norm.json", "w"))
    log(f"saved {out}")
    log("DONE")


if __name__ == "__main__":
    main()
