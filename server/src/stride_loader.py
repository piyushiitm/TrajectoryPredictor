"""
STRIDE (Rajshahi, Bangladesh) loader -- South Asian road conditions.

Figshare 10.6084/m9.figshare.25460755, CC BY 4.0. Sensor Logger app on a
Xiaomi Poco X2, ~99.6 Hz IMU and 1 Hz GPS, 23 sessions totalling 59 minutes.

Why this dataset. The speed model only works near its training distribution:
IO-VNBD averages 13.0 m/s and comma2k19 22.6, while the user's trips run
6-10 m/s and Pune 3.9 -- and the model over-predicts those by 3-5x. STRIDE's
GPS-weighted mean is 7.76 m/s, squarely in the target range, on roads
physically comparable to India's (Bangladesh, with a bump/pothole subset).

Layout: one directory per session, one CSV per sensor, each with `time`
(ns) and `seconds_elapsed`. Note the columns are ordered z, y, x.
  TotalAcceleration  raw accelerometer INCLUDING gravity   -> accel_*
  Gravity            gravity vector                        -> grav_*
  Gyroscope          rad/s                                 -> gyro_*
  Location           speed (m/s), bearing, lat, lon        -> truth
Sensor Logger's `Accelerometer.csv` is gravity-REMOVED, so it is not used:
taking TotalAcceleration minus Gravity reproduces it and keeps the same
convention as IO-VNBD.
"""
from pathlib import Path

import numpy as np
import pandas as pd

FS = 10.0
MIN_SECONDS = 100.0          # anomaly clips are a few seconds; skip them


def _read(d, name):
    p = Path(d) / name
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None


def find_sessions(root, min_seconds=MIN_SECONDS):
    out = []
    for loc in sorted(Path(root).rglob("Location.csv")):
        try:
            L = pd.read_csv(loc)
        except Exception:
            continue
        if L["seconds_elapsed"].max() >= min_seconds and len(L) >= 60:
            out.append(loc.parent)
    return out


def load_stride(session, target_hz=FS):
    """Return (rs, truth) shaped like idr_core.load_session."""
    A = _read(session, "TotalAcceleration.csv")
    G = _read(session, "Gravity.csv")
    W = _read(session, "Gyroscope.csv")
    L = _read(session, "Location.csv")
    if any(x is None for x in (A, G, W, L)):
        return None, None

    t0 = max(A.seconds_elapsed.min(), G.seconds_elapsed.min(),
             W.seconds_elapsed.min(), L.seconds_elapsed.min())
    t1 = min(A.seconds_elapsed.max(), G.seconds_elapsed.max(),
             W.seconds_elapsed.max(), L.seconds_elapsed.max())
    if t1 - t0 < MIN_SECONDS:
        return None, None
    t = np.arange(t0, t1, 1.0 / target_hz)

    def itp(df, col):
        return np.interp(t, df.seconds_elapsed.values, df[col].values)

    spd = itp(L, "speed")
    br = np.deg2rad(L.bearing.values)
    hx = np.interp(t, L.seconds_elapsed.values, np.sin(br))
    hy = np.interp(t, L.seconds_elapsed.values, np.cos(br))
    hdg = np.arctan2(hx, hy)

    rs = pd.DataFrame({
        "t_ms": (t - t[0]) * 1000.0,
        "accel_x": itp(A, "x"), "accel_y": itp(A, "y"), "accel_z": itp(A, "z"),
        "grav_x": itp(G, "x"), "grav_y": itp(G, "y"), "grav_z": itp(G, "z"),
        "gyro_roll": itp(W, "x"), "gyro_pitch": itp(W, "y"), "gyro_yaw": itp(W, "z"),
        "gps_lat": itp(L, "latitude"), "gps_lon": itp(L, "longitude"),
        "gps_speed_kmh": spd * 3.6,
        "gps_bearing_deg": np.rad2deg(hdg) % 360,
    })
    truth = dict(lat=rs.gps_lat.values, lon=rs.gps_lon.values,
                 spd=spd, hdg=hdg, yaw=None)
    return rs, truth


if __name__ == "__main__":
    import sys
    tot = 0.0
    for s in find_sessions(sys.argv[1]):
        rs, tr = load_stride(s)
        if rs is None:
            continue
        tot += len(rs) / FS
        g = np.linalg.norm(rs[["grav_x", "grav_y", "grav_z"]].values, axis=1).mean()
        lin = np.linalg.norm((rs[["accel_x", "accel_y", "accel_z"]].values -
                              rs[["grav_x", "grav_y", "grav_z"]].values), axis=1).mean()
        print(f"  {str(s).split('Road Data/')[-1]:34s} {len(rs):6d} rows  "
              f"spd {tr['spd'].mean():5.2f}  |g| {g:5.2f}  |lin| {lin:5.2f}")
    print(f"  TOTAL {tot/60:.1f} min")
