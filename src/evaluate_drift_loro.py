"""Positional drift, measured leave-one-recording-out. The benchmark number.

Everything reported so far has been one of two unsatisfying things: model-level
metrics (speed MAE, heading degrees) that are honest but are not drift, or drift
figures measured on recordings the speed head had trained on.

Here, for each held-out recording: train the speed head on the other 24, then run
the FULL pipeline on the held-out one -- model speed, gyro/compass fusion, real
outage windows -- and measure the positional drift the PS actually asks about.
Nothing the model has seen is scored.

Heading is not learned, so the fusion needs no holdout; only the speed head is
refitted per fold.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from run_matrix_all import ALLB, cached, log
from sim_compass_pipeline import compass, tau_for
from idr_core import (load_session_gps, genuine_fixes, gyro_matrix, calibrate_yaw,
                      yaw_rate_from_cal, truth_xy, model_speed_track2, wrap, FS)

FOLDS = ["mount_a", "mount_d", "mount_g", "mount_i", "hand_a", "hand_c",
         "pocket_b", "pocket_i", "trip12"]
DURS = (10.0, 30.0, 60.0)
CAL_S = 300.0


def main():
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    L = tf.keras.layers

    store = {}
    for nm in ALLB:
        r = cached(nm, "speed")
        if r is not None:
            store[nm] = r
    have = list(store)
    z = np.load(Path("data/processed/cache") / "__IO__speed.npz")
    Xio, Yio = z["X"], z["Y"]
    log(f"{len(have)} recordings cached, IO-VNBD {len(Xio)} rows")
    cols = json.load(open("results/models/v3/speed/norm_stats.json"))["input_cols"]

    agg = {d: {k: [] for k in ("gyro", "mag", "fused", "hold")} for d in DURS}
    for held in FOLDS:
        if held not in store:
            continue
        tr = [n for n in have if n != held]
        X = np.concatenate([store[n][0] for n in tr] + [Xio])
        Y = np.concatenate([store[n][1] for n in tr] + [Yio])
        mu = X.mean(0); sd = X.std(0); sd = np.where(sd < 1e-6, 1.0, sd) + 1e-8
        tf.keras.utils.set_random_seed(1)
        m = tf.keras.Sequential([L.Input((X.shape[1],)),
                                 L.Dense(256, activation="relu"), L.Dropout(.1),
                                 L.Dense(128, activation="relu"), L.Dropout(.1),
                                 L.Dense(64, activation="relu"), L.Dense(1)])
        m.compile(tf.keras.optimizers.Adam(1e-3), loss="huber")
        m.fit((X - mu) / sd, (Y[:, 0] - Y[:, 2]), validation_split=0.1, epochs=60,
              batch_size=2048, verbose=0,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=8,
                                                          restore_best_weights=True),
                         tf.keras.callbacks.ReduceLROnPlateau(patience=4, factor=.5)])

        p = Path(f"data/raw/{held}.csv")
        rs, truth = load_session_gps(p)
        if rs is None:
            log(f"  {held}: unusable"); continue
        psi, _ = compass(str(p), rs)
        t = rs["t_ms"].values / 1000.0; n = len(rs)
        hdg, sp = truth["hdg"], truth["spd"]
        te_, tn_ = truth_xy(truth); fx = genuine_fixes(rs); W = gyro_matrix(rs)
        cal = int(CAL_S * FS)
        line = [f"  {held:9s}"]
        for D in DURS:
            step = int(D * FS)
            res = {k: [] for k in ("gyro", "mag", "fused", "hold")}
            for a in range(cal, n - step, step):
                b = a + step
                if sp[a:b].mean() < 3.0:
                    continue
                dist = np.trapezoid(sp[a:b], dx=1 / FS)
                if dist < 30:
                    continue
                w4, _ = calibrate_yaw(W, t, hdg, fx, a - cal, a)
                if w4 is None:
                    continue
                rate = yaw_rate_from_cal(W[a:b], w4)
                hs = {"gyro": hdg[a] + np.cumsum(rate) / FS}
                if psi is not None:
                    off = wrap(hdg[a] - psi[a]); hm = psi[a:b] + off
                    cur = hdg[a]; hf = np.empty(step)
                    for i in range(step):
                        al = (1 / FS) / (tau_for(i / FS) + 1 / FS)
                        cur += rate[i] / FS
                        cur += al * wrap(hm[i] - cur)
                        hf[i] = cur
                    hs["mag"] = hm; hs["fused"] = hf
                v = model_speed_track2(rs, m, mu, sd, cols, a, a, b, sp[a])
                if v is None:
                    v = np.full(step, sp[a])
                dE, dN = te_[b - 1] - te_[a], tn_[b - 1] - tn_[a]
                for k, hh in hs.items():
                    e = np.cumsum(v * np.sin(hh)) / FS
                    nn = np.cumsum(v * np.cos(hh)) / FS
                    res[k].append(100 * np.hypot(e[-1] - dE, nn[-1] - dN) / dist)
                hh = hs.get("fused", hs["gyro"])
                e = np.cumsum(np.full(step, sp[a]) * np.sin(hh)) / FS
                nn = np.cumsum(np.full(step, sp[a]) * np.cos(hh)) / FS
                res["hold"].append(100 * np.hypot(e[-1] - dE, nn[-1] - dN) / dist)
            for k, vv in res.items():
                if vv:
                    agg[D][k].append(np.median(vv))
            if res["fused"]:
                line.append(f"{D:.0f}s fused {np.median(res['fused']):5.1f}%"
                            f" gyro {np.median(res['gyro']):5.1f}%")
        log("  ".join(line))

    log("")
    log("=" * 62)
    log("LEAVE-ONE-RECORDING-OUT DRIFT  (median over recordings)")
    log("=" * 62)
    log(f"  {'outage':8s} {'gyro':>8s} {'compass':>8s} {'fused':>8s} {'hold':>8s}"
        f" {'<10%':>6s}")
    for D in DURS:
        row = {k: (np.median(v) if v else float('nan')) for k, v in agg[D].items()}
        best = min(v for k, v in row.items() if k != "hold" and np.isfinite(v))
        under = sum(1 for x in agg[D]["fused"] if x < 10)
        log(f"  {D:5.0f}s   {row['gyro']:7.1f}% {row['mag']:7.1f}% {row['fused']:7.1f}%"
            f" {row['hold']:7.1f}%   {under}/{len(agg[D]['fused'])}")
    log("")
    log("The last column counts recordings whose MEDIAN fused drift is under 10%.")
    log("DONE")


if __name__ == "__main__":
    main()
