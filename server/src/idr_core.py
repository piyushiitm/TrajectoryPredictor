"""
IDR core: shared loaders and the calibrate-then-coast heading/speed engine.

Design follows what the measurements support, and what the literature does
(AVNet, AI-IMU, DMDVDR): a learned SPEED pseudo-measurement plus a calibrated
gyro heading, integrated ONCE to position. Acceleration is never integrated --
measured SNR for that is below 1 (tilt wobble fabricates 0.34-0.86 m/s^2
against a true 0.09-0.48 m/s^2 signal).

Truth comes from the row-aligned V-file: survey GPS position, Velocity,
Heading and the CAN Yaw Rate, all at 10Hz. Its self-consistency floor is
0.5-0.8%, so a <15% target is actually measurable against it.
"""
import numpy as np
import pandas as pd
from pathlib import Path

from TrajectoryPredictor.server.src.preprocess import (load_raw, split_sessions, EARTH_RADIUS_M,
                        fix_speed_units_vfile, _fix_speed_units)
from TrajectoryPredictor.server.src.preprocess_speed import resample, feature_names, window_features

FS = 10.0
WINDOW_S, HOP_S = 2.0, 0.5


def wrap(a):
    return np.angle(np.exp(1j * a))


def load_session(s_path, target_hz=FS):
    """Return (rs, truth) where truth carries V-file position/velocity/heading."""
    p = Path(s_path)
    vp = p.parent / p.name.replace("S-", "V-", 1)
    if not vp.exists():
        return None, None
    V = pd.read_csv(vp, encoding="latin-1")
    V.columns = [c.strip() for c in V.columns]

    def col(pref):
        c = [x for x in V.columns if x.startswith(pref)]
        return pd.to_numeric(V[c[0]], errors="coerce").values if c else None

    lat, lon = col("Latitude"), col("Longitude")
    vel, hdg, yrt = col("Velocity"), col("Heading"), col("Yaw Rate")
    if lat is None or vel is None or hdg is None:
        return None, None

    # The S-file's own GPS speed column is used as the ANCHOR speed at the
    # start of an outage, so its units must be right. IO-VNBD stores m/s in a
    # column labelled km/h; the V-file check resolves it on 68 of 72 files,
    # where the GPS-displacement fallback manages only 33.
    df = load_raw(str(p), fix_units=False)
    df, verdict = fix_speed_units_vfile(df, vp, verbose=False)
    if verdict == "unusable":
        df = _fix_speed_units(df, verbose=False)
    ses = split_sessions(df)
    if not ses:
        return None, None
    rs = resample(ses[0].reset_index(drop=True), target_hz)
    n = len(rs)
    if n < 1000 or len(vel) < n:
        return None, None
    truth = dict(lat=lat[:n], lon=lon[:n], spd=vel[:n] / 3.6,
                 hdg=np.deg2rad(hdg[:n]),
                 yaw=np.deg2rad(yrt[:n]) if yrt is not None else None)
    if not np.isfinite(truth["lat"]).all():
        good = np.isfinite(truth["lat"]) & np.isfinite(truth["lon"])
        if good.sum() < n * 0.8:
            return None, None
        idx = np.where(good)[0]
        for k in ("lat", "lon"):
            truth[k] = np.interp(np.arange(n), idx, truth[k][idx])
    return rs, truth


def truth_xy(truth):
    """Local east/north metres from the V-file lat/lon."""
    la, lo = np.deg2rad(truth["lat"]), np.deg2rad(truth["lon"])
    e = (lo - lo[0]) * np.cos(la[0]) * EARTH_RADIUS_M
    n = (la - la[0]) * EARTH_RADIUS_M
    return e, n


def gyro_matrix(rs):
    return np.nan_to_num(np.stack([rs["gyro_roll"].values,
                                   rs["gyro_pitch"].values,
                                   rs["gyro_yaw"].values], 1))


def calibrate_yaw(W, t, anchor_hdg, fix_idx, lo, hi):
    """Least-squares fit of the 3 gyro axes + bias to the heading change seen
    between GPS fixes inside [lo, hi).

    This is the honest replacement for trusting channel names OR projecting
    onto gravity: it discovers the axis permutation, the scale and the sign
    in one step, from data that is available before the outage begins.
    Returns (w4, quality) with quality = R^2 of the fit.
    """
    rows, targ = [], []
    f = fix_idx[(fix_idx >= lo) & (fix_idx < hi)]
    # Some datasets ship GPS already interpolated onto the IMU grid, so every
    # row looks like a new fix and every interval is 0.1s -- below the 0.5s
    # minimum, leaving nothing to fit. Decimate to ~1s spacing in that case.
    if len(f) > 2:
        step = np.median(np.diff(t[f]))
        if 0 < step < 0.5:
            keep = max(1, int(round(1.0 / step)))
            f = f[::keep]
    for a, b in zip(f[:-1], f[1:]):
        dt = t[b] - t[a]
        if not (0.5 < dt < 15.0):
            continue
        rows.append(np.append(W[a + 1:b + 1].sum(0) / FS, dt))
        targ.append(wrap(anchor_hdg[b] - anchor_hdg[a]))
    if len(rows) < 8:
        return None, 0.0
    A, y = np.array(rows), np.array(targ)
    w, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ w
    ss = 1.0 - np.sum((y - pred) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-9)
    return w, float(ss)


def yaw_rate_from_cal(W, w4):
    """Apply a calibration to get yaw rate per sample (bias term included)."""
    return W @ w4[:3] + w4[3]


def model_speed_track(rs, model, mu, sd):
    """Speed predicted every HOP_S, interpolated onto the sample grid."""
    lax = rs["accel_x"].values - rs["grav_x"].values
    lay = rs["accel_y"].values - rs["grav_y"].values
    laz = rs["accel_z"].values - rs["grav_z"].values
    gx, gy, gz = rs["gyro_roll"].values, rs["gyro_pitch"].values, rs["gyro_yaw"].values
    dt = 1.0 / FS
    wn, hop = int(WINDOW_S * FS), int(HOP_S * FS)
    ends, feats = [], []
    for e in range(wn, len(rs), hop):
        sl = slice(e - wn, e)
        seg = [lax[sl], lay[sl], laz[sl], gx[sl], gy[sl], gz[sl]]
        if not all(np.isfinite(v).all() for v in seg):
            continue
        ends.append(e - 1); feats.append(window_features(*seg, dt))
    if not feats:
        return None
    X = ((np.array(feats, "float32") - mu) / sd).astype("float32")
    p = np.clip(model.predict(X, batch_size=8192, verbose=0).ravel(), 0, None)
    return np.interp(np.arange(len(rs)), np.array(ends), p)


def genuine_fixes(rs):
    lat, lon = rs["gps_lat"].values, rs["gps_lon"].values
    idx, pl, po = [], None, None
    for i in range(len(rs)):
        if not (np.isfinite(lat[i]) and np.isfinite(lon[i])) or (lat[i] == 0 and lon[i] == 0):
            continue
        if pl is None or lat[i] != pl or lon[i] != po:
            idx.append(i); pl, po = lat[i], lon[i]
    return np.array(idx, int)


def model_speed_track2(rs, model, mu, sd, cols, anchor_idx, lo, hi, v0, hop_s=1.0):
    """Anchored multi-scale speed over [lo,hi), for the speed2 model.

    Unlike model_speed_track, this needs the anchor: the model predicts the
    RESIDUAL from the speed known at the outage start, so v0 and the elapsed
    time are inputs, not context.
    """
    from src.preprocess_speed2 import SCALES, feature_names2
    from TrajectoryPredictor.server.src.preprocess_speed import window_features
    lax = rs["accel_x"].values - rs["grav_x"].values
    lay = rs["accel_y"].values - rs["grav_y"].values
    laz = rs["accel_z"].values - rs["grav_z"].values
    gx, gy, gz = rs["gyro_roll"].values, rs["gyro_pitch"].values, rs["gyro_yaw"].values
    dt = 1.0 / FS
    wmax = int(max(SCALES) * FS)
    hop = max(1, int(hop_s * FS))
    names = feature_names2()

    ends, rows = [], []
    for e in range(max(lo, wmax), hi, hop):
        feats, bad = [], False
        for s in SCALES:
            sl = slice(e - int(s * FS), e)
            seg = [lax[sl], lay[sl], laz[sl], gx[sl], gy[sl], gz[sl]]
            if not all(np.isfinite(v).all() for v in seg):
                bad = True; break
            feats += window_features(*seg, dt)
        if bad:
            continue
        full = dict(zip(names, feats + [float(v0), (e - anchor_idx) / FS]))
        rows.append([full[c] for c in cols]); ends.append(e)
    if not rows:
        return None
    X = ((np.array(rows, "float32") - mu) / sd).astype("float32")
    res = model.predict(X, batch_size=8192, verbose=0).ravel()
    v = np.clip(v0 + res, 0.0, None)
    return np.interp(np.arange(lo, hi), np.array(ends), v)


def mag_heading_track(s_path, rs):
    """Tilt-compensated compass heading on the sample grid, or None.

    Drift-free but noisy (~10 deg). Its value is that its error does NOT grow
    with outage length, so it bounds the gyro's drift over long blackouts.
    """
    raw = pd.read_csv(s_path, encoding="latin-1")
    raw.columns = [c.strip() for c in raw.columns]

    def c(pref):
        h = [x for x in raw.columns if x.startswith(pref)]
        return pd.to_numeric(raw[h[0]], errors="coerce").values if h else None

    mx, my, mz = c("MAGNETIC FIELD X"), c("MAGNETIC FIELD Y"), c("MAGNETIC FIELD Z")
    gx, gy, gz = c("GRAVITY X"), c("GRAVITY Y"), c("GRAVITY Z")
    if mx is None or gx is None:
        return None
    M = np.column_stack([mx, my, mz]); G = np.column_stack([gx, gy, gz])
    ok = np.isfinite(M).all(1) & np.isfinite(G).all(1)
    if ok.sum() < 500:
        return None
    M, G = M[ok], G[ok]
    E = np.cross(M, G); E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
    N = np.cross(G, E); N /= np.linalg.norm(N, axis=1, keepdims=True) + 1e-9
    h = np.arctan2(E[:, 1], N[:, 1])
    src = np.linspace(0, len(rs) - 1, len(h))
    hx = np.interp(np.arange(len(rs)), src, np.sin(h))
    hy = np.interp(np.arange(len(rs)), src, np.cos(h))
    return np.arctan2(hx, hy)


def fuse_heading(h_gyro, h_mag, offset, tau_s, dt=1.0 / FS):
    """Complementary blend: gyro early (smooth, accurate short-term), compass
    late (noisy, but its error does not grow). tau sets the crossover."""
    t = np.arange(len(h_gyro)) * dt
    w = np.exp(-t / tau_s)
    return h_gyro + (1.0 - w) * wrap(h_mag + offset - h_gyro)


def load_session_gps(s_path, target_hz=FS):
    """Load a recording that has no V-file: truth comes from its own GPS.

    Weaker truth than a V-file (trip1's self-consistency floor is 5.8-16.1%,
    trip3's about 1.2-3.2%), and only as dense as the GPS -- 1Hz on the user's
    recordings, versus 10Hz survey data in a V-file. Speed and heading are
    interpolated between GENUINE fixes, never from carried-forward values.
    """
    df = load_raw(str(s_path))            # units auto-detected
    ses = split_sessions(df)
    if not ses:
        return None, None
    rs = resample(ses[0].reset_index(drop=True), target_hz)
    n = len(rs)
    if n < 1000:
        return None, None
    fx = genuine_fixes(rs)
    if len(fx) < 20:
        return None, None
    t = rs["t_ms"].values / 1000.0
    lat = np.interp(t, t[fx], rs["gps_lat"].values[fx])
    lon = np.interp(t, t[fx], rs["gps_lon"].values[fx])
    sp = np.interp(t, t[fx], rs["gps_speed_kmh"].values[fx] / 3.6)
    br = np.deg2rad(rs["gps_bearing_deg"].values[fx])
    hx = np.interp(t, t[fx], np.sin(br)); hy = np.interp(t, t[fx], np.cos(br))
    truth = dict(lat=lat, lon=lon, spd=sp, hdg=np.arctan2(hx, hy), yaw=None)
    return rs, truth
