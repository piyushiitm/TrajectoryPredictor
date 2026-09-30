"""
IDR navigator -- the deployable pipeline, end to end.

    position(t) = p0 + integral of  v(t) * [sin h(t), cos h(t)] dt

Acceleration is never integrated. Measured on this data, the accelerometer's
1-second velocity increment correlates ~0.02-0.07 with the true increment,
because 2-5 deg/s of mount tilt wobble fabricates 0.34-0.86 m/s^2 of phantom
horizontal acceleration against a real signal of 0.09-0.48 m/s^2. Integrating
that diverges to 140-400% path error. Everything here avoids it.

HEADING -- calibrate, then coast.
  While GNSS is available, least-squares fit the three gyro axes plus a bias
  to the heading change seen between GPS fixes. This discovers the axis
  permutation, scale and sign together, which matters because the channel
  named "GYROSCOPE Yaw" correlates 0.006 with true yaw rate while the one
  named "GYROSCOPE Pitch" correlates 0.95. A 300s window reaches R^2 ~0.99 and
  holds heading to ~9 deg over a 60s blackout.

SPEED -- anchored multi-scale regression.
  30 mount-independent window statistics at 2s/5s/10s, conditioned on the
  speed known at the outage start, predicting the RESIDUAL from it. Optional
  per-session recalibration fits a scale+offset on the pre-outage window,
  correcting this vehicle's vibration response without retraining.

Usage:
    python3 src/idr_navigate.py --input <S-file or dir> --model models/speed2 \
        --durations 10,30,60,120 --recal
"""
import config
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')

from idr_core import (FS, wrap, load_session, load_session_gps, truth_xy,
                      gyro_matrix, calibrate_yaw, yaw_rate_from_cal,
                      model_speed_track2, genuine_fixes, EARTH_RADIUS_M)


class IDR:
    """Calibrate on GNSS-available data, then dead-reckon through a blackout."""

    def __init__(self, model_dir, cal_s=300.0, recal=True, blend_tau=0.0):
        st = json.load(open(Path(model_dir) / "norm_stats.json"))
        self.mu = np.array(st["mean"]); self.sd = np.array(st["std"])
        self.cols = st["input_cols"]
        self.model = tf.keras.models.load_model(Path(model_dir) / "speed_model.keras")
        self.cal_s = cal_s
        self.recal = recal
        # The model beats hold-speed only once the anchor has gone stale
        # (worse at 5s, 26% better at 60s). Blending toward the model with
        # this time constant keeps the anchor's accuracy early on.
        self.blend_tau = blend_tau

    def calibrate(self, rs, t, bearing, fx, lo, hi, true_speed=None):
        """Returns a dict of calibration state, or None if it cannot be trusted."""
        w4, q = calibrate_yaw(gyro_matrix(rs), t, bearing, fx, lo, hi)
        if w4 is None:
            return None
        cal = {"w4": w4, "r2": q, "scale": 1.0, "offset": 0.0}
        if self.recal and true_speed is not None and hi - lo > 100:
            cv = model_speed_track2(rs, self.model, self.mu, self.sd, self.cols,
                                    lo, lo, hi, true_speed[lo], hop_s=2.0)
            if cv is not None and len(cv) > 100:
                y = true_speed[lo:hi][:len(cv)]
                k = np.isfinite(cv) & np.isfinite(y) & (y > 2)
                if k.sum() > 50 and np.std(cv[k]) > 0.2:
                    A = np.column_stack([cv[k], np.ones(k.sum())])
                    w, *_ = np.linalg.lstsq(A, y[k], rcond=None)
                    if 0.3 < w[0] < 3.0:
                        cal["scale"], cal["offset"] = float(w[0]), float(w[1])
        return cal

    def run(self, rs, cal, a, b, h0, v0):
        """Dead-reckon [a,b). Returns (east, north, heading, speed)."""
        W = gyro_matrix(rs)
        rate = yaw_rate_from_cal(W[a:b], cal["w4"])
        h = h0 + np.cumsum(rate) / FS
        v = model_speed_track2(rs, self.model, self.mu, self.sd, self.cols,
                               a, a, b, v0)
        if v is None:
            v = np.full(b - a, v0)
        v = np.clip(cal["scale"] * v + cal["offset"], 0.0, None)
        if self.blend_tau > 0:
            w = np.exp(-(np.arange(len(v)) / FS) / self.blend_tau)
            v = w * v0 + (1 - w) * v
        e = np.cumsum(v * np.sin(h)) / FS
        n = np.cumsum(v * np.cos(h)) / FS
        return e, n, h, v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--durations", default="10,30,60,120")
    ap.add_argument("--cal-s", type=float, default=300.0)
    ap.add_argument("--recal", action="store_true")
    ap.add_argument("--blend-tau", type=float, default=0.0,
                    help="seconds; blend from the anchor speed toward the model")
    ap.add_argument("--max-files", type=int, default=0)
    ap.add_argument("--map-match", default=None,
                    help="path to a .osm.pbf; snaps the dead-reckoned track to "
                         "the road network and reports both scores")
    ap.add_argument("--gps-truth", action="store_true",
                    help="recording has no V-file; use its own GPS as truth")
    ap.add_argument("--dataset", default="auto",
                    choices=["auto", "iovnbd", "trip", "pune", "stride", "comma"],
                    help="which loader to use for --input")
    args = ap.parse_args()

    root = Path(args.input)
    ds = args.dataset
    if ds == "auto":
        ds = "trip" if args.gps_truth else "iovnbd"

    # each loader returns (rs, truth) in the same shape, so the engine below
    # is identical across datasets
    if ds == "pune":
        from pune_loader import find_files, load_pune
        items = find_files(root); loader = load_pune
    elif ds == "stride":
        from stride_loader import find_sessions, load_stride
        items = find_sessions(root); loader = load_stride
    elif ds == "comma":
        from comma_loader import find_routes, load_route
        items = [v for _, v in sorted(find_routes(root).items())]
        loader = load_route
    elif ds == "trip":
        items = sorted(root.glob("*.csv")) if root.is_dir() else [root]
        loader = load_session_gps
    else:
        items = sorted(root.rglob("S-*.csv")) if root.is_dir() else [root]
        loader = load_session
    if args.max_files:
        items = items[:args.max_files]
    files = items
    idr = IDR(args.model, args.cal_s, args.recal, args.blend_tau)
    durs = [float(x) for x in args.durations.split(",")]
    res = {d: [] for d in durs}
    meters = {d: [] for d in durs}
    mm_res = {d: [] for d in durs}
    mm_m = {d: [] for d in durs}
    nsess = 0
    matcher = None
    if args.map_match:
        from osm_pbf import roads_from_pbf
        from map_match import RoadNetwork, match
        matcher = (roads_from_pbf, RoadNetwork, match)

    for p in files:
        rs, truth = loader(p)
        if rs is None:
            continue
        n = len(rs); t = rs["t_ms"].values / 1000.0
        fx = genuine_fixes(rs)
        if len(fx) < 25:
            continue
        gb = np.deg2rad(pd.Series(rs["gps_bearing_deg"]).ffill().bfill().values)
        gs = pd.Series(rs["gps_speed_kmh"]).ffill().bfill().values / 3.6
        gs = np.nan_to_num(gs)
        te, tn = truth_xy(truth); tv = truth["spd"]
        nsess += 1

        net = None
        if matcher is not None:
            roads_from_pbf, RoadNetwork, match = matcher
            try:
                ways = roads_from_pbf(args.map_match,
                                      np.nanmin(truth["lat"]), np.nanmax(truth["lat"]),
                                      np.nanmin(truth["lon"]), np.nanmax(truth["lon"]),
                                      verbose=(nsess == 1))
                net = RoadNetwork(ways, float(truth["lat"][0]), float(truth["lon"][0]))
            except Exception as ex:
                print(f"  [map] unavailable for this session: {ex}")
        cal_n = int(args.cal_s * FS)
        for d in durs:
            step = (n - cal_n - 1) if d == 0 else int(d * FS)
            if step < 20 or cal_n + step >= n:
                continue
            for a in range(cal_n, max(cal_n + 1, n - step), step):
                b = a + step
                if tv[a:b].mean() < 3.0:
                    continue
                dist = np.trapezoid(tv[a:b], dx=1 / FS)
                if dist < 30:
                    continue
                c = idr.calibrate(rs, t, gb, fx, a - cal_n, a, gs)
                if c is None:
                    continue
                e, nn, h, _ = idr.run(rs, c, a, b, gb[a], gs[a])
                dE, dN = te[b - 1] - te[a], tn[b - 1] - tn[a]
                err = np.hypot(e[-1] - dE, nn[-1] - dN)
                res[d].append(100 * err / dist); meters[d].append(err)

                if net is not None:
                    # DR displacement -> absolute lat/lon, snap, re-score
                    la0, lo0 = float(truth["lat"][a]), float(truth["lon"][a])
                    dlat = la0 + np.rad2deg(nn / EARTH_RADIUS_M)
                    dlon = lo0 + np.rad2deg(e / (EARTH_RADIUS_M *
                                                 np.cos(np.deg2rad(la0))))
                    try:
                        mlat, mlon = match(net, dlat, dlon, h,
                                           np.arange(len(e)) / FS)
                        mE = np.deg2rad(mlon[-1] - lo0) * np.cos(np.deg2rad(la0)) * EARTH_RADIUS_M
                        mN = np.deg2rad(mlat[-1] - la0) * EARTH_RADIUS_M
                        me = np.hypot(mE - dE, mN - dN)
                        mm_res[d].append(100 * me / dist); mm_m[d].append(me)
                    except Exception:
                        pass

    if mm_res and any(mm_res.values()):
        print("\n  --- with map matching ---")
        print(f"  {'outage':>7s} {'n':>5s} {'drift %':>9s} {'error m':>9s} {'<15%':>7s} {'<10%':>7s}")
        for d in durs:
            v = np.array(mm_res.get(d, []))
            if not len(v):
                continue
            mm = np.array(mm_m[d])
            tag = "FULL" if d == 0 else f"{d:.0f}s"
            print(f"  {tag:>6s} {len(v):5d} {np.median(v):8.1f}% {np.median(mm):8.1f} "
                  f"{100*(v<15).mean():6.0f}% {100*(v<10).mean():6.0f}%")

    print(f"dataset: {ds}   sessions: {nsess}   model {Path(args.model).name}   "
          f"cal {args.cal_s:g}s   recal {args.recal}")
    print(f"\n  {'outage':>7s} {'n':>5s} {'drift %':>9s} {'error m':>9s} {'<15%':>7s} {'<10%':>7s}")
    for d in durs:
        if not res[d]:
            continue
        v = np.array(res[d]); mm = np.array(meters[d])
        tag = "FULL" if d == 0 else f"{d:.0f}s"
        print(f"  {tag:>6s} {len(v):5d} {np.median(v):8.1f}% {np.median(mm):8.1f} "
              f"{100*(v<15).mean():6.0f}% {100*(v<10).mean():6.0f}%")
    print("\n  median positional drift as % of distance travelled.")


if __name__ == "__main__":
    main()
