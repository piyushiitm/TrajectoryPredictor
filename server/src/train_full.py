"""Train BOTH heads on every sample, then test with magnetometer heading.

Data
----
Every 10 Hz sample carries a label, so hop = 1 sample (no decimation). Sources:
IO-VNBD (V-file CAN truth), comma2k19 (CAN + GNSS pose), and the phone trips
(their own GPS). Split 90/10 BY SESSION -- row-wise would leak, since adjacent
windows overlap almost completely.

Heads
-----
speed   residual against the anchor speed; predicting 0 == "hold last speed"
velvec  anchor-frame (vx, vy); predicting (0, v0) == hold-anchor velocity

Testing
-------
Positional drift on the most recent trips, with heading taken from the
HARD-IRON-CORRECTED magnetometer rather than GPS bearing. The compass is
drift-free, which is what the gyro is not, and its offset is anchored once at
the start of the outage from the last GPS fix -- nothing during the blackout.
"""
import config
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from idr_core import (load_session, load_session_gps, genuine_fixes, gyro_matrix,
                      calibrate_yaw, yaw_rate_from_cal, truth_xy, wrap, FS)
from preprocess_speed import window_features, feature_names

SCALES = (2.0, 5.0, 10.0)
ELAPSED = (5.0, 15.0, 30.0, 60.0)
T0 = time.time()


def log(*a):
    print(f"[{time.time()-T0:7.0f}s]", *a, flush=True)


def names():
    n = []
    for s in SCALES:
        n += [f"{k}_{int(s)}s" for k in feature_names()]
    return n + ["v0", "elapsed"]


def build(rs, truth, hop=1):
    lax = rs["accel_x"].values - rs["grav_x"].values
    lay = rs["accel_y"].values - rs["grav_y"].values
    laz = rs["accel_z"].values - rs["grav_z"].values
    gx, gy, gz = rs["gyro_roll"].values, rs["gyro_pitch"].values, rs["gyro_yaw"].values
    spd = np.asarray(truth["spd"], float)
    hdg = truth.get("hdg")
    hdg = np.asarray(hdg, float) if hdg is not None else None
    n = len(rs); dt = 1.0 / FS; wmax = int(max(SCALES) * FS)
    X, Y = [], []
    for e in range(wmax, n, hop):
        if not np.isfinite(spd[e - 1]):
            continue
        f, bad = [], False
        for s in SCALES:
            sl = slice(e - int(s * FS), e)
            seg = [lax[sl], lay[sl], laz[sl], gx[sl], gy[sl], gz[sl]]
            if not all(np.isfinite(v).all() for v in seg):
                bad = True; break
            f += window_features(*seg, dt)
        if bad:
            continue
        for el in ELAPSED:
            a = e - int(el * FS)
            if a < 0 or not np.isfinite(spd[a]):
                continue
            if hdg is None or not (np.isfinite(hdg[a]) and np.isfinite(hdg[e - 1])):
                dh = np.nan
            else:
                dh = wrap(hdg[e - 1] - hdg[a])
            X.append(f + [spd[a], el])
            Y.append([spd[e - 1], dh, spd[a]])
    if not X:
        return None
    return np.asarray(X, "float32"), np.asarray(Y, "float32")


def hard_iron(M):
    A = np.column_stack([2 * M, np.ones(len(M))]); b = np.sum(M ** 2, axis=1)
    return np.linalg.lstsq(A, b, rcond=None)[0][:3]


def mag_heading(path, rs):
    raw = pd.read_csv(path); raw.columns = [c.strip() for c in raw.columns]
    try:
        M = np.column_stack([raw[f'MAGNETIC FIELD {k} (μT)'].values for k in 'XYZ'])
        G = np.column_stack([raw[f'GRAVITY {k} (m/s²)'].values for k in 'XYZ'])
    except KeyError:
        return None
    tr = raw['TIME SINCE START (ms)'].values / 1000.0
    ok = np.isfinite(M).all(1) & np.isfinite(G).all(1) & (np.abs(M).sum(1) > 1e-6)
    if ok.sum() < 500:
        return None
    M = M - hard_iron(M[ok])
    gn = G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-9)
    mh = M - np.sum(M * gn, axis=1)[:, None] * gn
    ex = np.zeros_like(gn); ex[:, 0] = 1.0
    e1 = ex - np.sum(ex * gn, axis=1)[:, None] * gn
    nn = np.linalg.norm(e1, axis=1, keepdims=True); bad = nn[:, 0] < 1e-6
    if bad.any():
        ey = np.zeros_like(gn); ey[:, 1] = 1.0
        e1[bad] = (ey - np.sum(ey * gn, axis=1)[:, None] * gn)[bad]
        nn = np.linalg.norm(e1, axis=1, keepdims=True)
    e1 /= (nn + 1e-9); e2 = np.cross(gn, e1)
    psi = np.unwrap(np.arctan2(np.sum(mh * e2, axis=1), np.sum(mh * e1, axis=1))[ok])
    return np.interp(rs['t_ms'].values / 1000.0, tr[ok], psi)


def main():
    cache = Path("data/processed/full_everysample.npz")
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        X, Y, S = z["X"], z["Y"], z["S"]
        log(f"cached {len(X)} rows, {len(set(S.tolist()))} sessions")
    else:
        Xs, Ys, Ss = [], [], []
        io = sorted(Path("data/raw/iovnbd_data").rglob("S-*.csv"))
        log(f"IO-VNBD {len(io)} files (every sample)")
        for i, p in enumerate(io):
            try:
                rs, truth = load_session(p)
            except Exception:
                continue
            if rs is None:
                continue
            r = build(rs, truth)
            if r is None:
                continue
            Xs.append(r[0]); Ys.append(r[1]); Ss.append(np.full(len(r[0]), f"IO:{p.stem}"))
            if (i + 1) % 10 == 0:
                log(f"  {i+1}/{len(io)}  rows so far {sum(len(x) for x in Xs)}")
        try:
            from comma_loader import find_routes, load_route
            rt = find_routes("data/raw/comma2k19")
            log(f"comma2k19 {len(rt)} routes")
            for j, (k, segs) in enumerate(sorted(rt.items())):
                try:
                    rs, truth = load_route(segs)
                except Exception:
                    continue
                if rs is None:
                    continue
                r = build(rs, truth)
                if r is None:
                    continue
                Xs.append(r[0]); Ys.append(r[1]); Ss.append(np.full(len(r[0]), f"CM:{j}"))
        except Exception as ex:
            log(f"comma unavailable: {ex}")
        trips = [f"trip{i}" for i in range(1, 14)] + \
                ["s01_mount", "s02_mount", "s01_pocket", "s02_pocket"]
        log("phone trips")
        for nm in trips:
            p = config.recording(nm)
            if not p.exists():
                continue
            try:
                rs, truth = load_session_gps(p)
            except Exception:
                continue
            if rs is None:
                log(f"  {nm}: unusable"); continue
            r = build(rs, truth)
            if r is None:
                continue
            Xs.append(r[0]); Ys.append(r[1]); Ss.append(np.full(len(r[0]), f"TR:{nm}"))
            log(f"  {nm}: {len(r[0])} rows")
        X = np.concatenate(Xs); Y = np.concatenate(Ys); S = np.concatenate(Ss)
        del Xs, Ys, Ss
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, X=X, Y=Y, S=S)
        log(f"built {len(X)} rows, {len(set(S.tolist()))} sessions")

    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.keras.utils.set_random_seed(1)

    sess = np.array(sorted(set(S.tolist())))
    rng = np.random.default_rng(0); rng.shuffle(sess)
    ncut = max(1, int(0.10 * len(sess)))
    te = np.isin(S, sess[:ncut]); tr = ~te
    log(f"split: train {tr.sum()} rows / test {te.sum()} rows "
        f"({len(sess)-ncut} vs {ncut} sessions)")

    v0 = Y[:, 2]; spd = Y[:, 0]; dh = Y[:, 1]
    mu = X[tr].mean(0); sd = X[tr].std(0); sd = np.where(sd < 1e-6, 1.0, sd) + 1e-8
    Xn = (X - mu) / sd
    L = tf.keras.layers

    def mlp(nout):
        m = tf.keras.Sequential([L.Input((X.shape[1],)),
                                 L.Dense(256, activation="relu"), L.Dropout(.1),
                                 L.Dense(128, activation="relu"), L.Dropout(.1),
                                 L.Dense(64, activation="relu"), L.Dense(nout)])
        m.compile(tf.keras.optimizers.Adam(1e-3), loss="huber")
        return m

    cb = [tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
          tf.keras.callbacks.ReduceLROnPlateau(patience=4, factor=.5)]

    log("=== SPEED head ===")
    ms = mlp(1)
    ms.fit(Xn[tr], (spd - v0)[tr], validation_split=0.1, epochs=60, batch_size=2048,
           verbose=2, callbacks=cb)
    ps = np.clip(v0[te] + ms.predict(Xn[te], batch_size=16384, verbose=0).ravel(), 0, None)
    hs = np.abs(v0[te] - spd[te]).mean(); msE = np.abs(ps - spd[te]).mean()
    log(f"SPEED test: hold {hs:.3f}  model {msE:.3f} m/s ({100*(hs-msE)/hs:+.1f}%)"
        f"  corr {np.corrcoef(ps, spd[te])[0,1]:.3f}")

    log("=== VELVEC head ===")
    ok = np.isfinite(dh)
    trv = tr & ok; tev = te & ok
    vx = spd * np.sin(dh); vy = spd * np.cos(dh)
    mv = mlp(2)
    mv.fit(Xn[trv], np.column_stack([vx, vy - v0])[trv], validation_split=0.1,
           epochs=60, batch_size=2048, verbose=2, callbacks=cb)
    pv = mv.predict(Xn[tev], batch_size=16384, verbose=0)
    px = pv[:, 0]; py = v0[tev] + pv[:, 1]
    hv = np.hypot(vx[tev], vy[tev] - v0[tev]).mean()
    mvE = np.hypot(vx[tev] - px, vy[tev] - py).mean()
    log(f"VELVEC test: hold {hv:.3f}  model {mvE:.3f} m/s ({100*(hv-mvE)/hv:+.1f}%)"
        f"  corr vx {np.corrcoef(px, vx[tev])[0,1]:+.3f}  vy {np.corrcoef(py, vy[tev])[0,1]:+.3f}")

    out = config.MODELS / ("full"); out.mkdir(parents=True, exist_ok=True)
    ms.save(out / "speed.keras"); mv.save(out / "velvec.keras")
    json.dump({"mean": mu.tolist(), "std": sd.tolist(), "cols": names()},
              open(out / "norm.json", "w"))
    del X, Xn

    log("=== DRIFT on recent trips: GYRO vs MAGNETOMETER heading ===")
    for nm in ["s01_mount", "s02_mount", "s01_pocket", "s02_pocket", "s11_bike", "s12_bike", "s13_bike"]:
        p = config.recording(nm)
        if not p.exists():
            continue
        try:
            rs, truth = load_session_gps(p)
        except Exception:
            continue
        if rs is None:
            log(f"  {nm}: unusable"); continue
        psi = mag_heading(str(p), rs)
        t = rs["t_ms"].values / 1000.0; n = len(rs)
        hdg = truth["hdg"]; sp = truth["spd"]
        te_, tn_ = truth_xy(truth); fx = genuine_fixes(rs); W = gyro_matrix(rs)
        cal = int(300 * FS)
        for D in (10.0, 30.0, 60.0):
            step = int(D * FS); res = {"gyro": [], "mag": [], "fused": []}
            for a in range(cal, n - step, step):
                if sp[a:a + step].mean() < 3.0:
                    continue
                dist = np.trapezoid(sp[a:a + step], dx=1 / FS)
                if dist < 30:
                    continue
                w4, q = calibrate_yaw(W, t, hdg, fx, a - cal, a)
                if w4 is None:
                    continue
                idx = np.arange(a, a + step)
                f2 = []
                lax = rs["accel_x"].values - rs["grav_x"].values
                # speed from the model, re-anchored at the outage start
                XX = []
                for e in idx:
                    pass
                rate = yaw_rate_from_cal(W[a:a + step], w4)
                h_g = hdg[a] + np.cumsum(rate) / FS
                hs_ = {"gyro": h_g}
                if psi is not None:
                    off = wrap(hdg[a] - psi[a])
                    h_m = psi[a:a + step] + off
                    alpha = (1 / FS) / (10.0 + 1 / FS)
                    h_f = np.empty(step); cur = hdg[a]
                    for i in range(step):
                        cur += rate[i] / FS
                        cur += alpha * wrap(h_m[i] - cur)
                        h_f[i] = cur
                    hs_["mag"] = h_m; hs_["fused"] = h_f
                v = np.full(step, sp[a])          # hold speed: isolates heading
                dE, dN = te_[a + step - 1] - te_[a], tn_[a + step - 1] - tn_[a]
                for k, h in hs_.items():
                    e = np.cumsum(v * np.sin(h)) / FS; nn_ = np.cumsum(v * np.cos(h)) / FS
                    res[k].append(100 * np.hypot(e[-1] - dE, nn_[-1] - dN) / dist)
            if res["gyro"]:
                log(f"  {nm:9s} {D:3.0f}s n={len(res['gyro']):3d}  " +
                    "  ".join(f"{k} {np.median(v):5.1f}%" for k, v in res.items() if v))
    log("done")


if __name__ == "__main__":
    main()
