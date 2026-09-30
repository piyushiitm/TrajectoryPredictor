"""
comma2k19 loader -- a second platform for the speed model.

Why this dataset. Every attempt to improve the speed head by adding capacity
failed (2x data, a 4th feature scale, four boosting frameworks, a deep TCN),
which pointed at DOMAIN SHIFT across vehicles rather than model limits.
comma2k19 is a different device, fleet and logger with the same sensor class,
and it is sustained highway driving -- which fills the speed range where our
IO-VNBD-trained model degraded worst (MAE tripled above 25 m/s, and 36% of
test windows sat above the 95th percentile of anything in training).

Format (from the dataset docs): each segment holds processed_log/ with
paired time/value numpy arrays.
    IMU        accelerometer (m/s^2), gyro (rad/s), magnetometer -- in a
               forward/right/down DEVICE frame. The EON is windscreen
               mounted, so this is a rigid mount.
    CAN        car_speed (m/s)  <- ground truth, same grade as IO-VNBD V-files
    GNSS       live_gnss_* = [lat, lon, speed, utc, alt, bearing]

Two differences from IO-VNBD that the loader must absorb:
  * There is NO gravity channel. IO-VNBD ships one; here gravity is estimated
    by low-passing the accelerometer, then subtracted to get linear
    acceleration. Our features are magnitudes and spreads, so a slow-varying
    offset matters little, but it is removed for consistency.
  * Sample rates are irregular and differ per stream, so everything is
    resampled onto the project's common 10 Hz grid.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

FS = 10.0
GRAV_CUT = 0.05          # Hz; slow enough to track tilt, not vehicle motion


def _load_pair(d, name):
    """Return (t, value) for a processed_log entry, or (None, None)."""
    p = Path(d) / name
    if not p.exists():
        return None, None
    try:
        t = np.load(p / "t")
        v = np.load(p / "value")
    except Exception:
        return None, None
    return np.asarray(t, float), np.asarray(v, float)


def find_segments(root):
    """Every segment directory under a chunk (one per minute of driving)."""
    root = Path(root)
    return sorted({p.parent.parent for p in root.rglob("processed_log/CAN")}) or \
           sorted({p.parent for p in root.rglob("processed_log")})


def load_segment(seg, target_hz=FS, min_seconds=45.0):
    """Resample one segment onto the 10 Hz grid.

    Returns (rs, truth) shaped like idr_core.load_session so the existing
    feature builders work unchanged, or (None, None) if unusable.
    """
    pl = Path(seg) / "processed_log"
    ta, acc = _load_pair(pl, "IMU/accelerometer")
    tg, gyr = _load_pair(pl, "IMU/gyro")
    if gyr is None:
        tg, gyr = _load_pair(pl, "IMU/gyro_uncalibrated")
    tc, spd = _load_pair(pl, "CAN/speed")
    if tc is None:
        tc, spd = _load_pair(pl, "CAN/car_speed")
    if acc is None or gyr is None or spd is None:
        return None, None
    spd = spd[:, 0] if spd.ndim > 1 else spd

    t0 = max(ta[0], tg[0], tc[0]); t1 = min(ta[-1], tg[-1], tc[-1])
    if t1 - t0 < min_seconds:
        return None, None
    t = np.arange(t0, t1, 1.0 / target_hz)
    n = len(t)

    A = np.column_stack([np.interp(t, ta, acc[:, i]) for i in range(3)])
    G = np.column_stack([np.interp(t, tg, gyr[:, i]) for i in range(3)])
    V = np.interp(t, tc, spd)

    # gravity by low-pass, since there is no gravity channel here
    b, a = butter(2, GRAV_CUT / (target_hz / 2), "low")
    GR = np.column_stack([filtfilt(b, a, A[:, i]) for i in range(3)])

    tl, gnss = _load_pair(pl, "GNSS/live_gnss_qcom")
    if gnss is None:
        tl, gnss = _load_pair(pl, "GNSS/live_gnss_ublox")
    if gnss is not None and gnss.shape[1] >= 6:
        lat = np.interp(t, tl, gnss[:, 0]); lon = np.interp(t, tl, gnss[:, 1])
        brg = np.rad2deg(np.arctan2(np.interp(t, tl, np.sin(np.deg2rad(gnss[:, 5]))),
                                    np.interp(t, tl, np.cos(np.deg2rad(gnss[:, 5])))))
        gspd = np.interp(t, tl, gnss[:, 2])
    else:
        lat = lon = brg = gspd = np.full(n, np.nan)

    rs = pd.DataFrame({
        "t_ms": (t - t[0]) * 1000.0,
        "accel_x": A[:, 0], "accel_y": A[:, 1], "accel_z": A[:, 2],
        "grav_x": GR[:, 0], "grav_y": GR[:, 1], "grav_z": GR[:, 2],
        "gyro_roll": G[:, 0], "gyro_pitch": G[:, 1], "gyro_yaw": G[:, 2],
        "gps_lat": lat, "gps_lon": lon,
        "gps_speed_kmh": gspd * 3.6,          # this project stores km/h
        "gps_bearing_deg": brg,
    })
    hx = np.interp(t, tc, np.zeros(len(tc)))   # placeholder, heading from GNSS
    truth = dict(lat=lat, lon=lon, spd=V,
                 hdg=np.deg2rad(brg), yaw=None)
    return rs, truth


if __name__ == "__main__":
    import sys
    segs = find_segments(sys.argv[1])
    print(f"{len(segs)} segments found")
    ok = 0
    for s in segs[:5]:
        rs, tr = load_segment(s)
        if rs is None:
            print(f"  SKIP {s.name}")
            continue
        ok += 1
        print(f"  {'/'.join(s.parts[-3:]):40s} {len(rs):5d} rows  "
              f"speed mean {tr['spd'].mean():5.2f} max {tr['spd'].max():5.2f} m/s")
    print(f"{ok}/5 loaded")


def find_routes(root):
    """Group segments by drive. Each segment is one minute; consecutive
    numbered segments of a route are contiguous in time, so stitching them
    gives multi-minute sessions. Without this the longest usable anchor age
    would be ~50s, and the model's biggest wins are at 60-120s."""
    segs = find_segments(root)
    routes = {}
    for s in segs:
        try:
            idx = int(s.name)
        except ValueError:
            continue
        routes.setdefault(s.parent, []).append((idx, s))
    return {k: [s for _, s in sorted(v)] for k, v in routes.items()}


def load_route(segs, target_hz=FS, max_gap_s=2.0):
    """Concatenate consecutive segments into one session on the 10 Hz grid."""
    parts, last_idx = [], None
    for s in segs:
        idx = int(s.name)
        rs, truth = load_segment(s, target_hz)
        if rs is None:
            last_idx = None
            continue
        if last_idx is not None and idx != last_idx + 1:
            break                      # a gap in the drive: stop, keep it contiguous
        parts.append((rs, truth)); last_idx = idx
    if not parts:
        return None, None
    rs = pd.concat([p[0] for p in parts], ignore_index=True)
    dt = 1000.0 / target_hz
    rs["t_ms"] = np.arange(len(rs)) * dt
    truth = {k: np.concatenate([p[1][k] for p in parts])
             for k in ("lat", "lon", "spd", "hdg")}
    truth["yaw"] = None
    return rs, truth
