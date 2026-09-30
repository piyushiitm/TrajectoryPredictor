"""Learn to filter LOCAL MAGNETIC ATTRACTION out of the compass heading.

Hard-iron correction removes a CONSTANT offset -- the phone's own magnets, a
fixed bracket. It cannot touch a TRANSIENT one: a passing truck, a steel bridge,
rebar in an underpass, overhead lines. Those were measured swinging the compass
by up to 90 deg, and they announce themselves in the field magnitude (18.5 uT/s
during a spurious jump against 0.72 uT/s otherwise, a 26x tell).

The app currently REJECTS such samples. That is safe but wasteful: on the mount
recording a third of all readings were discarded. If the disturbance signature
predicts the heading bias it causes, those samples could be corrected instead.

Target
------
Instantaneous compass heading error against GPS bearing, with the per-session
CONSTANT offset removed first. That removal is essential: the compass frame has
an arbitrary origin, so without it the model would simply memorise one offset
per recording and look excellent while learning nothing. In deployment the
offset is anchored from GNSS at the moment of loss, exactly as it is here.

Inputs are magnetic and attitude only -- no window statistics of acceleration --
because the question is specifically whether the FIELD reveals its own
distortion.

Evaluated two ways, as requested: leave-one-recording-out (honest, no shared
ride) and held-out tails (optimistic, shares the ride and its magnetic
surroundings). The gap between them is informative in itself.
"""
import config
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from run_full_job import log
from idr_core import load_session_gps, wrap, FS

MOUNTS = ["s01_mount", "s02_mount", "s03_mount", "s04_mount", "s05_mount"]
UNSEEN = ["t03_other", "s11_bike", "s05_pocket"]
TAIL = 0.10
GAP_S = 150.0
deg = np.rad2deg


def build(nm, hop=5):
    """Per-sample magnetic features and the compass's own heading error."""
    p = config.recording(nm)
    if not p.exists():
        return None
    rs, truth = load_session_gps(p)
    if rs is None:
        return None
    raw = pd.read_csv(p)
    raw.columns = [c.strip() for c in raw.columns]
    try:
        M = np.column_stack([raw[f'MAGNETIC FIELD {k} (μT)'].values for k in 'XYZ'])
        G = np.column_stack([raw[f'GRAVITY {k} (m/s²)'].values for k in 'XYZ'])
        W = np.column_stack([raw['GYROSCOPE Yaw (rad/s)'].values,
                             raw['GYROSCOPE Pitch (rad/s)'].values,
                             raw['GYROSCOPE Roll (rad/s)'].values])
    except KeyError:
        return None
    tr = raw['TIME SINCE START (ms)'].values / 1000.0
    ok = np.isfinite(M).all(1) & np.isfinite(G).all(1) & (np.abs(M).sum(1) > 1e-6)
    if ok.sum() < 2000:
        return None

    # hard-iron: the centre of the sphere the field traces, fitted once
    c = np.linalg.lstsq(np.column_stack([2 * M, np.ones(len(M))])[ok],
                        np.sum(M[ok] ** 2, axis=1), rcond=None)[0][:3]
    Mc = M - c
    b = np.linalg.norm(Mc, axis=1)
    b0 = np.median(b[ok])                       # the undisturbed norm

    gn = G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-9)
    h = Mc - np.sum(Mc * gn, axis=1)[:, None] * gn
    e1 = np.column_stack([1 - gn[:, 0] ** 2, -gn[:, 0] * gn[:, 1], -gn[:, 0] * gn[:, 2]])
    e1 = e1 / (np.linalg.norm(e1, axis=1, keepdims=True) + 1e-9)
    e2 = np.cross(gn, e1)
    hx = np.sum(h * e1, axis=1)
    hy = np.sum(h * e2, axis=1)
    psi = np.arctan2(hy, hx)
    hz = np.sum(Mc * gn, axis=1)                # vertical field component
    inc = np.arctan2(hz, np.hypot(hx, hy))      # magnetic inclination (dip)

    fs = 1.0 / np.median(np.diff(tr))
    win = max(2, int(round(fs)))                # ~1 s
    def roll(v, f):
        s = pd.Series(v)
        return f(s.rolling(win, min_periods=1, center=True)).to_numpy()
    db = np.gradient(b, tr)
    b_std = roll(b, lambda r: r.std()).astype(float)
    b_dev = (b - b0) / max(b0, 1e-6)
    inc_dev = inc - np.median(inc[ok])
    tilt = np.arccos(np.clip(gn[:, 2], -1, 1))
    wmag = np.linalg.norm(W, axis=1)

    # truth on the raw grid
    t_rs = rs["t_ms"].values / 1000.0
    hdg = np.interp(tr, t_rs, np.unwrap(truth["hdg"]))
    spd = np.interp(tr, t_rs, truth["spd"])

    sel = np.where(ok & (spd > 3.0))[0][::hop]
    if len(sel) < 500:
        return None
    err = wrap(hdg[sel] - psi[sel])
    # remove the arbitrary constant frame offset; anchored from GNSS in the app
    off = np.arctan2(np.mean(np.sin(err)), np.mean(np.cos(err)))
    y = wrap(err - off)

    X = np.column_stack([
        b[sel], b_dev[sel], db[sel], b_std[sel],
        hz[sel], inc[sel], inc_dev[sel],
        np.hypot(hx, hy)[sel],
        hx[sel], hy[sel],
        gn[sel, 0], gn[sel, 1], gn[sel, 2], tilt[sel],
        wmag[sel], W[sel, 0], W[sel, 1], W[sel, 2],
        spd[sel],
    ]).astype("float32")
    return X, y.astype("float32"), tr[sel]


COLS = ["b", "b_dev", "db_dt", "b_std", "hz", "inc", "inc_dev", "h_horiz",
        "hx", "hy", "gx", "gy", "gz", "tilt", "wmag", "wx", "wy", "wz", "speed"]


def main():
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    L = tf.keras.layers

    log("building MOUNT-ONLY compass pool")
    Xs, Ys, Ss, Ts = [], [], [], []
    for nm in MOUNTS:
        r = build(nm)
        if r is None:
            log(f"  {nm}: unusable")
            continue
        Xs.append(r[0]); Ys.append(r[1]); Ss.append(np.full(len(r[0]), nm)); Ts.append(r[2])
        log(f"  {nm}: {len(r[0])} rows   raw compass err "
            f"{deg(np.abs(r[1])).mean():.2f} deg")
    if not Xs:
        log("no data")
        return
    X = np.concatenate(Xs); Y = np.concatenate(Ys)
    S = np.concatenate(Ss); T = np.concatenate(Ts)
    log(f"pool: {len(X)} rows, {len(COLS)} features")

    def fit(xt, yt, seed=1):
        tf.keras.utils.set_random_seed(seed)
        m = tf.keras.Sequential([L.Input((xt.shape[1],)),
                                 L.Dense(128, activation="relu"), L.Dropout(.2),
                                 L.Dense(64, activation="relu"), L.Dropout(.2),
                                 L.Dense(32, activation="relu"), L.Dense(1)])
        m.compile(tf.keras.optimizers.Adam(1e-3), loss="huber")
        h = m.fit(xt, yt, validation_split=0.15, epochs=100, batch_size=1024,
                  verbose=0,
                  callbacks=[tf.keras.callbacks.EarlyStopping(patience=12,
                                                              restore_best_weights=True),
                             tf.keras.callbacks.ReduceLROnPlateau(patience=6, factor=.5)])
        return m, len(h.history["loss"])

    def report(y, pred, tag, n, ep=None):
        raw = deg(np.abs(y)).mean()
        cor = deg(np.abs(wrap(y - pred))).mean()
        e = f" ep={ep:3d}" if ep else ""
        log(f"  {tag:10s} n={n:6d}{e}   raw {raw:6.2f}  filtered {cor:6.2f} deg"
            f"   {100*(raw-cor)/raw:+6.1f}%   {'BETTER' if cor < raw else 'worse'}")
        return 100 * (raw - cor) / raw

    log("")
    log("#" * 70)
    log("A. LEAVE-ONE-RECORDING-OUT  (no shared ride -- the honest number)")
    log("#" * 70)
    g = []
    for held in MOUNTS:
        tr = S != held; te = S == held
        if te.sum() < 100:
            continue
        mu = X[tr].mean(0); sd = X[tr].std(0); sd = np.where(sd < 1e-6, 1.0, sd) + 1e-8
        m, ep = fit((X[tr] - mu) / sd, Y[tr])
        p = m.predict((X[te] - mu) / sd, batch_size=16384, verbose=0).ravel()
        g.append(report(Y[te], p, held, te.sum(), ep))
    if g:
        log(f"  LORO MEAN {np.mean(g):+6.1f}%")

    log("")
    log("#" * 70)
    log("B. HELD-OUT TAILS  (shares the ride and its magnetic surroundings)")
    log("#" * 70)
    te = np.zeros(len(X), bool); tr = np.zeros(len(X), bool)
    for nm in sorted(set(S.tolist())):
        k = S == nm
        cut = T[k].max() * (1 - TAIL)
        te |= k & (T >= cut); tr |= k & (T < cut - GAP_S)
    mu = X[tr].mean(0); sd = X[tr].std(0); sd = np.where(sd < 1e-6, 1.0, sd) + 1e-8
    m, ep = fit((X[tr] - mu) / sd, Y[tr])
    p = m.predict((X[te] - mu) / sd, batch_size=16384, verbose=0).ravel()
    report(Y[te], p, "ALL TAILS", te.sum(), ep)
    for nm in sorted(set(S[te].tolist())):
        k = te & (S == nm)
        if k.sum() > 50:
            pk = m.predict((X[k] - mu) / sd, batch_size=16384, verbose=0).ravel()
            report(Y[k], pk, nm, k.sum())

    log("")
    log("#" * 70)
    log("C. UNSEEN CONDITIONS (final model, all 5 mounts)")
    log("#" * 70)
    mu = X.mean(0); sd = X.std(0); sd = np.where(sd < 1e-6, 1.0, sd) + 1e-8
    m, ep = fit((X - mu) / sd, Y)
    for nm in UNSEEN:
        r = build(nm)
        if r is None:
            log(f"  {nm}: unusable")
            continue
        pk = m.predict((r[0] - mu) / sd, batch_size=16384, verbose=0).ravel()
        report(r[1], pk, nm, len(r[0]))

    out = config.MODELS / ("compass_filter")
    out.mkdir(parents=True, exist_ok=True)
    m.save(out / "compass.keras")
    json.dump({"cols": COLS, "mean": mu.tolist(), "std": sd.tolist()},
              open(out / "norm.json", "w"))
    log(f"saved {out}")
    log("DONE")


if __name__ == "__main__":
    main()
