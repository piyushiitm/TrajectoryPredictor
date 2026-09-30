"""Full matrix: heading source x map matching, on the road-test recordings.

Heading sources are gyro-only, magnetometer-only and the complementary fusion,
each run with map matching off and on, at 10/30/60 s and the full single-anchor
trip. Speed always comes from the model -- this is the fully-on-IMU case, with
nothing from GNSS after the anchor.

The magnetometer here applies the SAME spike rejection the app now does: a
reading whose field magnitude departs more than 15% from the running norm is
dropped, because a passing vehicle or a steel structure bends the field and
swings apparent north by up to 90 deg.

Map matching is HMM/Viterbi (Newson & Krumm) against the offline OSM extract,
and its own trust gate can reject a match, in which case the unsnapped track
is kept -- a confident snap to the wrong road costs more than it saves.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
sys.path.insert(0, str(Path(__file__).parent))
from idr_core import (load_session_gps, genuine_fixes, gyro_matrix, calibrate_yaw,
                      yaw_rate_from_cal, truth_xy, model_speed_track2, wrap, FS,
                      EARTH_RADIUS_M)

import os
_MD = os.environ.get("IDR_MODEL", "results/models/app_bike")
_st = json.load(open(f"{_MD}/norm_stats.json"))
_MU = np.array(_st["mean"]); _SD = np.array(_st["std"]); _COLS = _st["input_cols"]
_NET = tf.keras.models.load_model(f"{_MD}/speed_model.keras")
PBF = "data/osm/pbf/central-zone.osm.pbf"
B_TOL = 0.15


def mag_heading(path, rs):
    raw = pd.read_csv(path); raw.columns = [c.strip() for c in raw.columns]
    M = np.column_stack([raw[f'MAGNETIC FIELD {k} (μT)'].values for k in 'XYZ'])
    G = np.column_stack([raw[f'GRAVITY {k} (m/s²)'].values for k in 'XYZ'])
    tr = raw['TIME SINCE START (ms)'].values / 1000.0
    ok = np.isfinite(M).all(1) & np.isfinite(G).all(1) & (np.abs(M).sum(1) > 1e-6)
    if ok.sum() < 500:
        return None, None, 0
    b = np.linalg.norm(M, axis=1)
    bstd = float(b[ok].std())
    A = np.column_stack([2 * M, np.ones(len(M))])
    c = np.linalg.lstsq(A[ok], np.sum(M[ok] ** 2, axis=1), rcond=None)[0][:3]
    m0 = b[ok].mean()
    clean = ok & (np.abs(b - m0) / m0 <= B_TOL)      # spike rejection
    rej = int(ok.sum() - clean.sum())
    if clean.sum() < 500:
        return None, bstd, rej
    Mc = M - c
    gn = G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-9)
    h = Mc - np.sum(Mc * gn, axis=1)[:, None] * gn
    e1 = np.column_stack([1 - gn[:, 0] ** 2, -gn[:, 0] * gn[:, 1], -gn[:, 0] * gn[:, 2]])
    e1 = e1 / (np.linalg.norm(e1, axis=1, keepdims=True) + 1e-9)
    e2 = np.cross(gn, e1)
    psi = np.arctan2(np.sum(h * e2, axis=1), np.sum(h * e1, axis=1))
    return (np.interp(rs["t_ms"].values / 1000.0, tr[clean], np.unwrap(psi[clean])),
            bstd, rej)


def main():
    from osm_pbf import roads_from_pbf
    from map_match import RoadNetwork, match
    for T in sys.argv[1:]:
        p = f"data/raw/tests/{T}.csv"
        if not Path(p).exists():
            p = f"data/raw/{T}.csv"
        rs, truth = load_session_gps(p)
        if rs is None:
            print(f"{T}: unusable", flush=True); continue
        psi, bstd, rej = mag_heading(p, rs)
        t = rs["t_ms"].values / 1000.0; n = len(rs)
        hdg = truth["hdg"]; sp = truth["spd"]
        te_, tn_ = truth_xy(truth); fx = genuine_fixes(rs); W = gyro_matrix(rs)
        cal = int(300 * FS)
        print(f"\n{'='*76}\n=== {T}   {n/FS:.0f}s   |B| std {bstd:.1f} uT   "
              f"spikes rejected {rej}   model {_MD.split('/')[-1]}\n{'='*76}", flush=True)
        net = None
        try:
            ways = roads_from_pbf(PBF, np.nanmin(truth["lat"]), np.nanmax(truth["lat"]),
                                  np.nanmin(truth["lon"]), np.nanmax(truth["lon"]),
                                  verbose=False)
            net = RoadNetwork(ways, float(truth["lat"][0]), float(truth["lon"][0]))
            print(f"  map: {len(ways)} ways loaded", flush=True)
        except Exception as ex:
            print(f"  map unavailable: {ex}", flush=True)

        for D in (10.0, 30.0, 60.0, 0.0):
            step = (n - cal - 1) if D == 0 else int(D * FS)
            if step < 20 or cal + step >= n:
                continue
            acc = {}
            dists = []
            nwin = 0
            for a in range(cal, max(cal + 1, n - step), step):
                b = a + step
                if b >= n or sp[a:b].mean() < 3.0:
                    continue
                dist = np.trapezoid(sp[a:b], dx=1 / FS)
                if dist < 30:
                    continue
                w4, q = calibrate_yaw(W, t, hdg, fx, a - cal, a)
                if w4 is None:
                    continue
                rate = yaw_rate_from_cal(W[a:b], w4)
                hs = {"gyro": hdg[a] + np.cumsum(rate) / FS}
                if psi is not None:
                    off = wrap(hdg[a] - psi[a]); hm = psi[a:b] + off
                    hf = np.empty(step); cur = hdg[a]
                    for i in range(step):
                        el = i / FS
                        tau = 10.0 if el < 60 else max(2.0, 10.0 - (el - 60) * 8.0 / 240.0)
                        al = (1 / FS) / (tau + 1 / FS)
                        cur += rate[i] / FS; cur += al * wrap(hm[i] - cur); hf[i] = cur
                    hs["mag"] = hm; hs["fused"] = hf
                v = model_speed_track2(rs, _NET, _MU, _SD, _COLS, a, a, b, sp[a])
                if v is None:
                    v = np.full(step, sp[a])
                dE, dN = te_[b - 1] - te_[a], tn_[b - 1] - tn_[a]
                nwin += 1
                dists.append(dist)
                for k, h in hs.items():
                    e = np.cumsum(v * np.sin(h)) / FS; nn_ = np.cumsum(v * np.cos(h)) / FS
                    err_m = np.hypot(e[-1] - dE, nn_[-1] - dN)
                    acc.setdefault(f"{k}", []).append(100 * err_m / dist)
                    acc.setdefault(f"{k}#m", []).append(err_m)
                    if net is None:
                        continue
                    la = truth["lat"][a] + np.degrees(nn_ / EARTH_RADIUS_M)
                    lo = truth["lon"][a] + np.degrees(
                        e / (EARTH_RADIUS_M * np.cos(np.radians(truth["lat"][a]))))
                    try:
                        mla, mlo = match(net, la, lo, h, np.arange(step) / FS,
                                         stride=max(1, step // 60))
                        mE = np.radians(mlo[-1] - truth["lon"][a]) * np.cos(
                            np.radians(truth["lat"][a])) * EARTH_RADIUS_M
                        mN = np.radians(mla[-1] - truth["lat"][a]) * EARTH_RADIUS_M
                        em = np.hypot(mE - dE, mN - dN)
                        acc.setdefault(f"{k}+map", []).append(100 * em / dist)
                        acc.setdefault(f"{k}+map#m", []).append(em)
                    except Exception:
                        pass
            if not acc:
                continue
            lbl = "FULL" if D == 0 else f"{D:.0f}s"
            dmed = np.median(dists) if dists else float('nan')
            print(f"\n  --- {lbl}  n={nwin}   median distance travelled {dmed:.0f} m ---",
                  flush=True)
            print(f"    {'heading':8s} {'map OFF':>18s} {'map ON':>18s}", flush=True)
            for k in ("gyro", "mag", "fused"):
                if k not in acc:
                    continue
                o_p = np.median(acc[k]); o_m = np.median(acc[f"{k}#m"])
                if f"{k}+map" in acc:
                    n_p = np.median(acc[f"{k}+map"]); n_m = np.median(acc[f"{k}+map#m"])
                    on = f"{n_p:7.1f}% {n_m:7.1f} m"
                else:
                    on = f"{'--':>17s}"
                print(f"    {k:8s} {o_p:7.1f}% {o_m:7.1f} m  {on}", flush=True)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
