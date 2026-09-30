"""
IDR Engine -- window-feature SPEED regression, and the fixed-rate resampling
shared with the velocity builder.

Why speed magnitude rather than the vx/vy vector: heading is not observable
from these recordings (check_heading.py measures the integrated gyro at gain
-0.005 and the device-fused orientation at 0.109, where 1.0 is correct), so
anything that rotates acceleration into the world frame inherits a corrupted
heading. Speed magnitude never needs heading -- it is regressed directly from
window statistics of the IMU, with no integration at all, which also sidesteps
the gravity-leak problem that makes double integration unusable.

The 30 features are magnitude- and variance-based so they do not depend on how
the phone is mounted: overall linear-acceleration and gyro statistics, per-axis
spreads and mean absolute derivatives, and five spectral bands of the linear
acceleration (road-noise vibration scales with speed -- this is the "vibration
filter" the problem statement asks for).

Usage:
    python3 preprocess_speed.py --input iovnbd_data --output data/speed.csv

Resampling notes follow.

The IO-VNBD logs are not evenly sampled: the IMU arrives in bursts and the
smartphone GPS updates roughly every 9 seconds, carrying the previous fix
forward in between. Dead reckoning integrates acceleration against elapsed
time, so an uneven rate silently biases every integrated velocity. resample()
puts the whole session on an exact grid before any physics runs.

Two column families are treated differently on purpose:
  * IMU channels are bin-AVERAGED -- averaging is the correct low-pass for a
    noisy continuous signal.
  * GPS channels take the LAST value in the bin, never an average. Averaging
    a position that is held constant for 90 rows and then jumps would invent
    fixes that never happened. Carried-forward and (0,0) placeholders are
    turned into NaN here so downstream code can tell a genuine fix from a
    repeat.
"""
from pathlib import Path

import numpy as np
import pandas as pd

# IMU channels: safe to bin-average.
AVG_COLS = ("accel_x", "accel_y", "accel_z",
            "grav_x", "grav_y", "grav_z",
            "gyro_roll", "gyro_pitch", "gyro_yaw",
            "orient_pitch_deg", "orient_roll_deg")

# GPS: take the last value in the bin, and blank it when the fix is stale.
GPS_LAST_COLS = ("gps_lat", "gps_lon", "gps_speed_kmh", "gps_bearing_deg")

# Also last-in-bin, but NOT tied to GPS freshness: orient_yaw_deg is a device
# channel that updates every row. It is "last" rather than averaged only
# because it is an angle that wraps at 360 -- averaging 359 and 1 gives 180,
# pointing the car backwards. Blanking it alongside a stale GPS fix (an
# earlier bug here) silently destroyed the fused-heading channel.
LAST_COLS = GPS_LAST_COLS + ("orient_yaw_deg",)


def resample(df, target_hz, extra_avg=()):
    """Bin-average the IMU to exactly target_hz. GPS columns take the last
    GENUINE value in each bin (a carried-forward or 0,0 placeholder is not a
    real fix). extra_avg names further columns to bin-average -- used to carry
    V-file ground-truth position through with the sensor data.
    """
    df = df.copy()
    dt = 1000.0 / target_hz                       # bin width in milliseconds
    df["_bin"] = (df["t_ms"].values / dt).astype(np.int64)
    g = df.groupby("_bin", sort=True)

    avg = [c for c in list(AVG_COLS) + list(extra_avg) if c in df.columns]
    A = g[avg].mean()

    last = [c for c in LAST_COLS if c in df.columns]
    L = g[last].last()

    # A position repeated from the previous bin is the phone holding its last
    # fix, not a new one; (0,0) is the no-fix placeholder. Blank both so the
    # dead-reckoning loop only re-anchors on real fixes.
    if "gps_lat" in L.columns and "gps_lon" in L.columns:
        lat, lon = L["gps_lat"].values, L["gps_lon"].values
        dead = (lat == 0) & (lon == 0)
        same = np.zeros(len(L), dtype=bool)
        same[1:] = (lat[1:] == lat[:-1]) & (lon[1:] == lon[:-1])
        stale = dead | same
        gps_cols = [c for c in GPS_LAST_COLS if c in L.columns]
        L.loc[stale, gps_cols] = np.nan

    out = pd.concat([A, L], axis=1).reset_index(drop=True)
    # Rebuild time from the bin index so it is exactly on the grid.
    out.insert(0, "t_ms", (A.index.values * dt))
    return out


# ---------------------------------------------------------------- features

FEATURE_NAMES = [
    "linacc_mean", "linacc_std", "linacc_p90", "linacc_p10", "linacc_range",
    "gyro_mean", "gyro_std", "gyro_p90", "gyro_p10", "gyro_range",
    "lax_std", "lax_absderiv", "lay_std", "lay_absderiv",
    "laz_std", "laz_absderiv",
    "grx_std", "grx_absderiv", "gry_std", "gry_absderiv",
    "grz_std", "grz_absderiv",
    "linacc_derivvar", "gyro_derivvar", "laz_absmean",
    "band0", "band1", "band2", "band3", "band4",
]
N_BANDS = 5


def feature_names():
    return list(FEATURE_NAMES)


def _spread(v):
    """std, 90th pct, 10th pct, and the range between them."""
    p90, p10 = np.percentile(v, 90), np.percentile(v, 10)
    return v.std(), p90, p10, p90 - p10


def window_features(lax, lay, laz, gx, gy, gz, dt):
    """30 mount-angle-independent statistics over one window of IMU samples.

    Everything is a magnitude, a spread or a rate of change, so rotating the
    phone in its cradle does not change the features -- which is what lets a
    model trained on one mounting work on another.
    """
    lin = np.sqrt(lax ** 2 + lay ** 2 + laz ** 2)
    gyr = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)

    f = []
    f.append(lin.mean()); f.extend(_spread(lin))          # linacc_* (5)
    f.append(gyr.mean()); f.extend(_spread(gyr))          # gyro_*   (5)
    for v in (lax, lay, laz, gx, gy, gz):                 # per-axis (12)
        f.append(v.std())
        f.append(np.abs(np.diff(v)).mean() / dt if len(v) > 1 else 0.0)
    f.append(np.var(np.diff(lin)) if len(lin) > 1 else 0.0)   # linacc_derivvar
    f.append(np.var(np.diff(gyr)) if len(gyr) > 1 else 0.0)   # gyro_derivvar
    f.append(np.abs(laz).mean())                              # laz_absmean

    # spectral energy of the linear-acceleration magnitude, split into N_BANDS
    # equal slices up to Nyquist. Road vibration grows with speed.
    x = lin - lin.mean()
    P = np.abs(np.fft.rfft(x)) ** 2
    P = P[1:] if len(P) > 1 else P                        # drop DC
    if len(P) >= N_BANDS:
        for b in np.array_split(P, N_BANDS):
            f.append(float(np.log1p(b.sum())))
    else:
        f.extend([0.0] * N_BANDS)
    return f


def gps_speed_truth(rs):
    """True speed (m/s) per row interpolated between GENUINE GPS fixes.

    For recordings with no vehicle CAN bus beside them (the user's own phone
    trips), GPS speed is the only truth available. It is weaker than a V-file
    -- claude.md finding #6 puts trip1's self-consistency floor at 5.8-16.1%
    -- so treat small differences measured against it as noise.
    """
    lat, lon = rs["gps_lat"].values, rs["gps_lon"].values
    t = rs["t_ms"].values / 1000.0
    sp = rs["gps_speed_kmh"].values / 3.6
    idx, plat, plon = [], None, None
    for i in range(len(rs)):
        if not np.isfinite(lat[i]) or not np.isfinite(lon[i]) or (lat[i] == 0 and lon[i] == 0):
            continue
        if not np.isfinite(sp[i]):
            continue
        if plat is None or lat[i] != plat or lon[i] != plon:
            idx.append(i); plat, plon = lat[i], lon[i]
    if len(idx) < 20:
        return None
    idx = np.array(idx, int)
    return np.interp(t, t[idx], sp[idx])


def speed_truth(s_path, n):
    """True speed (m/s) per row from the row-aligned V-file, else None."""
    v = Path(s_path).parent / Path(s_path).name.replace("S-", "V-", 1)
    if not v.exists():
        return None
    V = pd.read_csv(v, encoding="latin-1")
    V.columns = [c.strip() for c in V.columns]
    sc = [c for c in V.columns if c.startswith("Velocity")]
    if not sc:
        return None
    sp = pd.to_numeric(V[sc[0]], errors="coerce").values[:n] / 3.6   # km/h -> m/s
    if len(sp) < n:
        sp = np.concatenate([sp, np.full(n - len(sp), np.nan)])
    return sp


def process_session(rs, sid, src, driver, dt, window_s, hop_s, truth):
    """One row per window: 30 features -> the true speed at the window's end."""
    lax = rs["accel_x"].values - rs["grav_x"].values
    lay = rs["accel_y"].values - rs["grav_y"].values
    laz = rs["accel_z"].values - rs["grav_z"].values
    gx, gy, gz = rs["gyro_roll"].values, rs["gyro_pitch"].values, rs["gyro_yaw"].values
    t = rs["t_ms"].values / 1000.0

    wn = max(4, int(round(window_s / dt)))
    hop = max(1, int(round(hop_s / dt)))
    out = []
    for e in range(wn, len(rs), hop):
        sl = slice(e - wn, e)
        y = truth[e - 1]
        if not np.isfinite(y):
            continue
        seg = [lax[sl], lay[sl], laz[sl], gx[sl], gy[sl], gz[sl]]
        if not all(np.isfinite(v).all() for v in seg):
            continue
        row = {"session_id": sid, "source_file": src, "driver": driver,
               "t_end_s": t[e - 1], "dt_s": dt, "window_s": window_s,
               "speed_ms": float(y)}
        row.update(dict(zip(FEATURE_NAMES, window_features(*seg, dt))))
        out.append(row)
    return out


def main():
    import argparse
    from preprocess import (load_raw, split_sessions, find_input_files,
                            driver_from_path, fix_speed_units_vfile, _fix_speed_units)

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--target-hz", type=float, default=10.0)
    ap.add_argument("--window", type=float, default=2.0)
    ap.add_argument("--hop", type=float, default=0.5)
    ap.add_argument("--truth", default="vfile", choices=["vfile", "gps"],
                    help="vfile = row-aligned CAN speed (IO-VNBD); gps = interpolate "
                         "the phone's own GPS speed (for recordings with no V-file)")
    args = ap.parse_args()
    dt = 1.0 / args.target_hz

    rows, n_ok, n_skip = [], 0, 0
    for path in find_input_files(args.input):
        try:
            # With GPS truth the speed column IS the label, so its units must be
            # right; with V-file truth the label comes from elsewhere and the
            # column is unused here. Prefer the V-file unit check when a V-file
            # exists -- the GPS-displacement fallback needs >=20 genuine fixes
            # and silently gives up on 39 of 72 IO-VNBD files.
            df = load_raw(str(path), fix_units=False)
            if args.truth == "gps":
                vp = path.parent / path.name.replace("S-", "V-", 1)
                df, verdict = fix_speed_units_vfile(df, vp)
                if verdict == "unusable":
                    df = _fix_speed_units(df)
        except Exception as e:
            print(f"  SKIPPED {path.name}: {e}"); continue
        drv = driver_from_path(path)
        got_any = False
        for si, s in enumerate(split_sessions(df)):
            rs = resample(s.reset_index(drop=True), args.target_hz)
            if len(rs) < 500:
                continue
            tr = (gps_speed_truth(rs) if args.truth == "gps"
                  else speed_truth(path, len(rs)))
            if tr is None:
                continue
            got = process_session(rs, f"{path.name}_{si}", path.name, drv,
                                  dt, args.window, args.hop, tr)
            if got:
                print(f"  {path.name} session {si}: {len(s)} raw rows -> {len(got)} windows")
                rows.extend(got); got_any = True
        n_ok += got_any; n_skip += (not got_any)

    D = pd.DataFrame(rows).dropna().reset_index(drop=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    D.to_csv(args.output, index=False)
    print(f"\nWrote {len(D)} windows from {D.session_id.nunique() if len(D) else 0} "
          f"sessions ({n_ok} files with truth, {n_skip} without) to {args.output}")
    if len(D):
        print(f"  speed: mean {D.speed_ms.mean():.2f} m/s, "
              f"std {D.speed_ms.std():.2f}, max {D.speed_ms.max():.2f}")


if __name__ == "__main__":
    main()
