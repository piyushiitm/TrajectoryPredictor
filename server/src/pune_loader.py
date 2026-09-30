"""
Pune (India) driving dataset loader -- an independent Indian test set.

Mendeley 5stn873wft, CC BY 4.0. Dashboard-mounted Android phone on Pune city
roads: Lat, Lon, Speed, Heading, 3-axis accel and 3-axis gyro at ~10 Hz.
About 24 minutes over two usable files (a third has no GPS).

Why it matters here: it is the right DOMAIN (Indian urban, dash-mounted,
mean speed 3.9 m/s) but the wrong SIZE (2% of our training data), so it is
used for evaluation, not training. It also sits in the low-speed regime that
comma2k19 pulled the model away from.

Two format differences from IO-VNBD:
  * Acc X/Y/Z is ALREADY linear acceleration (gravity removed -- the mean
    vector has magnitude 0.19, not 9.81), so grav_* is set to zero and the
    accel columns carry the linear signal directly. Nothing downstream needs
    the gravity direction: the speed features use accel-minus-gravity, and
    the yaw calibration uses the gyro alone.
  * Timestamps have 1-second resolution with ~10 samples inside each second,
    so the grid is reconstructed as uniform 10 Hz rather than parsed.
"""
from pathlib import Path

import numpy as np
import pandas as pd

FS = 10.0


def load_pune(path, target_hz=FS):
    """Return (rs, truth) shaped like idr_core.load_session."""
    d = pd.read_csv(path)
    d.columns = [c.strip() for c in d.columns]
    need = {"Longitude", "Latitude", "Speed", "Acc X", "Acc Y", "Acc Z",
            "Heading", "gyro_x", "gyro_y", "gyro_z"}
    if not need <= set(d.columns):
        return None, None
    for c in d.columns:
        if c != "Time":
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=list(need)).reset_index(drop=True)
    n = len(d)
    if n < 600:
        return None, None

    spd = d["Speed"].values.astype(float)                 # m/s
    hdg = np.deg2rad(d["Heading"].values.astype(float))
    rs = pd.DataFrame({
        "t_ms": np.arange(n) * (1000.0 / target_hz),
        "accel_x": d["Acc X"].values, "accel_y": d["Acc Y"].values,
        "accel_z": d["Acc Z"].values,
        "grav_x": 0.0, "grav_y": 0.0, "grav_z": 0.0,      # already gravity-free
        "gyro_roll": d["gyro_x"].values, "gyro_pitch": d["gyro_y"].values,
        "gyro_yaw": d["gyro_z"].values,
        "gps_lat": d["Latitude"].values, "gps_lon": d["Longitude"].values,
        "gps_speed_kmh": spd * 3.6,
        "gps_bearing_deg": d["Heading"].values,
    })
    truth = dict(lat=d["Latitude"].values, lon=d["Longitude"].values,
                 spd=spd, hdg=hdg, yaw=None)
    return rs, truth


def find_files(root):
    root = Path(root)
    out = []
    for p in sorted(root.glob("*.csv")):
        try:
            cols = set(c.strip() for c in pd.read_csv(p, nrows=1).columns)
        except Exception:
            continue
        if {"Speed", "Latitude", "Heading"} <= cols:
            out.append(p)
    return out


if __name__ == "__main__":
    import sys
    for p in find_files(sys.argv[1]):
        rs, tr = load_pune(p)
        if rs is None:
            print(f"  SKIP {p.name}"); continue
        print(f"  {p.name}: {len(rs)} rows ({len(rs)/FS/60:.1f} min)  "
              f"speed mean {tr['spd'].mean():.2f} max {tr['spd'].max():.2f} m/s  "
              f"|acc| {np.linalg.norm(rs[['accel_x','accel_y','accel_z']].values,axis=1).mean():.2f}")
