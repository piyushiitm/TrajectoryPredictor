"""
IDR Engine -- preprocessing pipeline for IO-VNBD-style GPS+IMU logs.

Turns a raw per-100ms sensor log (GPS repeated between fixes, IMU dense)
into training samples for the velocity-correction model:

    input  = IMU-integrated velocity DRIFT since the last GPS fix
             (world frame, north=+Y, east=+X)
    target = TRUE velocity drift over that same window, from GPS

Design decisions locked in with the user:
  - Sessions are split at TIME SINCE START resets (concatenated trips).
  - Roll & pitch: computed instantaneously from the gravity vector every
    row (no drift -- gravity always points down, so no anchoring needed).
  - Yaw: anchored to GPS bearing at each GPS fix, then propagated forward
    between fixes by integrating gyro yaw-rate (same reset-then-integrate
    pattern as velocity, applied to heading instead of speed).
  - Linear acceleration = ACCEL - GRAVITY in phone frame (using the
    dataset's own gravity columns, not an assumed constant), then rotated
    into world frame with the per-row rotation matrix from (roll, pitch, yaw(t)).
  - Training target is the DRIFT (delta since anchor), not the absolute
    velocity -- isolates the actual sensor error from "how fast were you
    going anyway", which is the better-posed regression problem.

Usage:
    python scripts/preprocess.py --input data/S-M.csv --output data/samples.csv
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

EARTH_RADIUS_M = 6_371_000.0


def latlon_to_local_xy(lat, lon, lat0, lon0):
    """Equirectangular local-tangent-plane projection -- fine for the short
    (few-hundred-meter) distances inside one window. x = east (m), y = north (m)."""
    lat_r, lon_r, lat0_r, lon0_r = map(np.deg2rad, (lat, lon, lat0, lon0))
    x = (lon_r - lon0_r) * np.cos(lat0_r) * EARTH_RADIUS_M
    y = (lat_r - lat0_r) * EARTH_RADIUS_M
    return x, y


# ---------------------------------------------------------------------------
# Loading & cleanup
# ---------------------------------------------------------------------------

def load_raw(path, fix_units=True):
    df = pd.read_csv(path, encoding="latin-1")
    df.columns = [c.strip() for c in df.columns]
    # normalize the handful of columns we touch by name
    rename = {}
    for c in df.columns:
        if c.startswith("GPS LATITUDE"):
            rename[c] = "gps_lat"
        elif c.startswith("GPS LONGITUDE"):
            rename[c] = "gps_lon"
        elif c.startswith("GPS SPEED"):
            rename[c] = "gps_speed_kmh"
        elif c.startswith("GPS ORIENTATION"):
            rename[c] = "gps_bearing_deg"
        elif c.startswith("GPS SATELLITES"):
            rename[c] = "gps_satellites_raw"
        elif c.startswith("TIME SINCE START"):
            rename[c] = "t_ms"
        elif c.startswith("ACCELEROMETER X"):
            rename[c] = "accel_x"
        elif c.startswith("ACCELEROMETER Y"):
            rename[c] = "accel_y"
        elif c.startswith("ACCELEROMETER Z"):
            rename[c] = "accel_z"
        elif c.startswith("GRAVITY X"):
            rename[c] = "grav_x"
        elif c.startswith("GRAVITY Y"):
            rename[c] = "grav_y"
        elif c.startswith("GRAVITY Z"):
            rename[c] = "grav_z"
        elif c.startswith("GYROSCOPE Yaw"):
            rename[c] = "gyro_yaw"
        elif c.startswith("GYROSCOPE Pitch"):
            rename[c] = "gyro_pitch"
        elif c.startswith("GYROSCOPE Roll"):
            rename[c] = "gyro_roll"
        # Device-fused absolute orientation (Android rotation-vector: gyro +
        # accel + magnetometer). Unlike integrated gyro yaw this does NOT
        # drift, which matters enormously over a long GNSS outage.
        elif c.startswith("ORIENTATION (Yaw)"):
            rename[c] = "orient_yaw_deg"
        elif c.startswith("ORIENTATION (Pitch)"):
            rename[c] = "orient_pitch_deg"
        elif c.startswith("ORIENTATION (Roll"):
            rename[c] = "orient_roll_deg"
    df = df.rename(columns=rename)
    for c in ("orient_yaw_deg", "orient_pitch_deg", "orient_roll_deg"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if fix_units:
        df = _fix_speed_units(df)
    return df


def _fix_speed_units(df, verbose=True):
    """IO-VNBD's smartphone logs label their speed column 'GPS SPEED (Kmh)'
    but actually store METRES PER SECOND -- verified two independent ways:
    against the row-aligned V-*.csv CAN-bus speed (ratio 3.607) and against
    speed derived from consecutive GPS positions (ratio 0.999). Other loggers
    (e.g. the app used for Trip1.csv) really do write km/h (ratio 3.553).

    Taking the label at face value makes every speed 3.6x too small, which
    corrupts the velocity anchor, the training targets and the
    distance-travelled denominator all at once. So detect the real unit from
    the data: compare the column against displacement between consecutive
    GENUINE GPS fixes, and rescale to true km/h if it is actually m/s.
    """
    if "gps_speed_kmh" not in df.columns:
        return df
    lat, lon = df.get("gps_lat"), df.get("gps_lon")
    if lat is None or lon is None:
        return df
    lat, lon = lat.values, lon.values
    col = df["gps_speed_kmh"].values
    t_s = df["t_ms"].values / 1000.0

    # indices where GPS reported a genuinely NEW position
    fx, plat, plon = [], None, None
    for i in range(len(df)):
        if np.isnan(lat[i]) or np.isnan(lon[i]) or (lat[i] == 0 and lon[i] == 0):
            continue
        if plat is None or lat[i] != plat or lon[i] != plon:
            fx.append(i); plat, plon = lat[i], lon[i]
    if len(fx) < 20:
        return df
    fx = np.array(fx)
    la, lo = np.deg2rad(lat[fx]), np.deg2rad(lon[fx])
    dist = np.hypot(np.diff(lo) * np.cos(la[:-1]) * EARTH_RADIUS_M,
                    np.diff(la) * EARTH_RADIUS_M)
    dt = np.diff(t_s[fx])
    k = (dt > 0.3) & (dt < 15.0) & (dist > 3.0)
    if k.sum() < 10:
        return df
    v_pos = dist[k] / dt[k]                 # true speed in m/s
    v_col = col[fx][:-1][k]
    moving = v_pos > 2.0
    if moving.sum() < 10:
        return df
    ratio = float(np.median(v_col[moving] / v_pos[moving]))

    if abs(ratio - 1.0) < 0.35:             # column is m/s despite its name
        df["gps_speed_kmh"] = df["gps_speed_kmh"] * 3.6
        if verbose:
            print(f"    [units] speed column is m/s (ratio {ratio:.2f}) -> rescaled to km/h")
    elif abs(ratio - 3.6) > 1.0 and verbose:
        print(f"    [units] WARNING: speed column matches neither m/s nor km/h "
              f"(ratio {ratio:.2f}); leaving as-is")
    return df


def fix_speed_units_vfile(df, v_path, verbose=True):
    """Unit-fix the speed column against the row-aligned V-*.csv CAN speed.

    Preferred over _fix_speed_units() whenever a V-file exists, because it
    needs no GPS fixes at all. The displacement method requires >=20 genuine
    fixes and, with IO-VNBD's ~9s GPS update rate, silently gave up on 39 of
    72 files -- leaving raw m/s in a column everything downstream reads as
    km/h, so every anchor velocity on those files came out 3.6x too small.

    Returns (df, verdict) where verdict is one of:
      'vfile_ms'    rescaled, confirmed m/s against CAN speed
      'vfile_kmh'   already km/h, left alone
      'unusable'    V-file gave no usable overlap -- caller should fall back
    """
    if "gps_speed_kmh" not in df.columns or not Path(v_path).exists():
        return df, "unusable"
    V = pd.read_csv(v_path, encoding="latin-1")
    V.columns = [c.strip() for c in V.columns]
    sc = [c for c in V.columns if c.startswith("Velocity")]
    if not sc:
        return df, "unusable"
    vs = pd.to_numeric(V[sc[0]], errors="coerce").values      # CAN speed, km/h
    cs = pd.to_numeric(df["gps_speed_kmh"], errors="coerce").values
    n = min(len(vs), len(cs))
    vs, cs = vs[:n], cs[:n]
    # only compare while genuinely moving; at rest the ratio is 0/0 noise
    m = (vs > 10.0) & np.isfinite(cs) & (cs > 0)
    if m.sum() < 50:
        return df, "unusable"
    ratio = float(np.median(vs[m] / cs[m]))
    if 2.5 < ratio < 5.0:                    # column is m/s despite its name
        df["gps_speed_kmh"] = df["gps_speed_kmh"] * 3.6
        if verbose:
            print(f"    [units] speed column is m/s (V-file ratio {ratio:.2f})"
                  f" -> rescaled to km/h")
        return df, "vfile_ms"
    if 0.7 < ratio < 1.5:                    # genuinely km/h already
        if verbose:
            print(f"    [units] speed column already km/h (V-file ratio {ratio:.2f})")
        return df, "vfile_kmh"
    if verbose:
        print(f"    [units] V-file ratio {ratio:.2f} matches neither m/s nor km/h"
              f" (GPS speed column likely dead on this file)")
    return df, "unusable"


def split_sessions(df):
    """Split at TIME SINCE START resets -- these are separate concatenated trips."""
    dt = df["t_ms"].diff()
    reset_idx = df.index[dt < 0].tolist()
    bounds = [0] + reset_idx + [len(df)]
    sessions = []
    for i in range(len(bounds) - 1):
        s = df.iloc[bounds[i]:bounds[i + 1]].reset_index(drop=True)
        if len(s) > 20:  # drop degenerate slivers
            sessions.append(s)
    return sessions


# ---------------------------------------------------------------------------
# Orientation
# ---------------------------------------------------------------------------

def roll_pitch_from_gravity(grav_x, grav_y, grav_z):
    """Instantaneous tilt angles from the gravity vector (phone frame).
    No drift, no integration -- valid every single row.
    Convention: roll about the Y axis, pitch about the X axis, phone flat = 0,0.
    """
    roll = np.arctan2(grav_y, grav_z)
    pitch = np.arctan2(-grav_x, np.sqrt(grav_y ** 2 + grav_z ** 2))
    return roll, pitch


def yaw_rate_from_body_gyro(gyro_roll_rate, gyro_pitch_rate, gyro_yaw_rate, roll, pitch):
    """True rate of change of world-frame heading (yaw), computed from the
    full 3-axis body-frame gyro reading plus the CURRENT tilt.

    Naively treating the gyro's own "yaw" channel as the heading-change
    rate is only correct when the phone is perfectly flat (roll=pitch=0).
    At real tilt angles the three raw gyro axes mix together once you
    project them onto the world's vertical axis. This is the standard
    body-rate -> Euler-rate formula (ZYX / yaw-pitch-roll sequence) that
    accounts for that mixing, using this row's fresh gravity-derived tilt
    -- so the tilt correction is effectively continuous, not just applied
    at GPS-fix resets.
    """
    p, q, r = gyro_roll_rate, gyro_pitch_rate, gyro_yaw_rate
    return (q * np.sin(roll) + r * np.cos(roll)) / np.cos(pitch)


def yaw_axis_from_gyro(gyro_xyz, moving=None, lat_accel=None):
    """Estimate the vehicle's vertical (yaw) axis in the GYRO's own frame.

    Do not trust the channel names. IO-VNBD labels its gyro axes
    Roll/Pitch/Yaw, but the channel that actually tracks vehicle yaw is the one
    labelled Pitch (corr 0.95 against the CAN Yaw Rate, versus 0.006 for the
    one labelled Yaw). The gyro triad is permuted relative to the
    accelerometer/gravity triad, so projecting onto the gravity direction --
    the textbook move -- returns the wrong channel and reads ~0.

    While a vehicle drives, almost all of its angular velocity is yaw: it turns
    far more than it pitches or rolls. So the PRINCIPAL AXIS of the angular
    velocity is the vertical, and it can be found without labels, without the
    gravity channel and without ground truth.

    The principal axis has no sign, so it is resolved with the non-holonomic
    relation a_lateral = v * yaw_rate: lateral acceleration and yaw rate must
    agree in sign. Pass lat_accel to enable that; otherwise the sign is
    arbitrary and only |correlation| is meaningful.

    Returns a unit 3-vector u such that (omega @ u) is the yaw rate.
    """
    W = np.asarray(gyro_xyz, float)
    m = np.isfinite(W).all(1)
    if moving is not None:
        m &= np.asarray(moving, bool)
    if m.sum() < 500:
        return None
    X = W[m] - W[m].mean(0)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    u = Vt[0]
    if lat_accel is not None:
        # lat_accel may be one channel or several candidates (N,) or (N,C).
        # Which body axis is "lateral" depends on how the phone is turned in
        # its mount, so take whichever candidate responds most strongly to
        # turning and use that one to fix the sign.
        A = np.asarray(lat_accel, float)
        if A.ndim == 1:
            A = A[:, None]
        yr = W @ u
        best_c = 0.0
        for j in range(A.shape[1]):
            k = m & np.isfinite(A[:, j])
            if k.sum() < 500 or np.std(A[k, j]) < 1e-9 or np.std(yr[k]) < 1e-9:
                continue
            c = np.corrcoef(A[k, j], yr[k])[0, 1]
            if abs(c) > abs(best_c):
                best_c = c
        if best_c < 0:
            u = -u
    return u


def rotation_matrix(yaw, pitch, roll):
    """Build phone-frame -> world-frame rotation matrix (ZYX Euler convention).
    World frame: X = east, Y = north, Z = up.
    """
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)

    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])

    return Rz @ Ry @ Rx


# ---------------------------------------------------------------------------
# GPS fix detection + ground-truth velocity
# ---------------------------------------------------------------------------

def detect_gps_fixes(df):
    """Return indices where GPS lat/lon actually changed (a genuine new fix),
    not just a repeated stale value."""
    changed = (df["gps_lat"].diff() != 0) | (df["gps_lon"].diff() != 0)
    changed.iloc[0] = True  # first row counts as a fix
    return df.index[changed].tolist()


def gps_velocity_ne(speed_kmh, bearing_deg):
    """Decompose GPS speed+bearing into north/east components.
    Compass bearing convention: 0 deg = North, 90 deg = East (clockwise)."""
    speed_ms = speed_kmh / 3.6
    bearing_rad = np.deg2rad(bearing_deg)
    vx_east = speed_ms * np.sin(bearing_rad)
    vy_north = speed_ms * np.cos(bearing_rad)
    return vx_east, vy_north


# ---------------------------------------------------------------------------
# Core per-session processing
# ---------------------------------------------------------------------------

def process_session(df, session_id, source_file="", driver="", session_index_in_file=0,
                     realtime_interval_s=None, max_window_s=90.0):
    """Build training samples for one session (one continuous drive).

    Default behavior (realtime_interval_s=None): one sample per window,
    emitted only at the window's END -- the drift accumulated over the
    *whole* gap between two consecutive GPS fixes. This is what every model
    so far has been trained on.

    realtime_interval_s (e.g. 0.5 or 1.0, or 0 for every raw row): also emit
    a sample roughly every that many seconds *inside* each window, not just
    at the end -- so the model is trained on what it will actually see at
    inference: a partial, growing drift at an arbitrary point in time since
    the last GPS fix, not only the final tally. There is no ground-truth GPS
    position at those intermediate instants (that's the whole reason we need
    IMU at all), so the intermediate target is an ASSUMPTION: the true
    correction grows linearly with elapsed time across the window
    (target(t) = target(window_end) * elapsed_s / window_dt_s). This is an
    approximation -- real drift/turning isn't perfectly linear inside a
    window -- but it's the only ground truth we can construct without new
    data, and it directly trains the model for continuous, real-time
    correction instead of one-shot end-of-window correction.

    max_window_s caps which windows get this intermediate-sample treatment.
    GPS-fix gaps in this data range from under a second to 600+ seconds
    (rare blackout-length outliers). Applying realtime_interval_s uniformly
    means a single 600s window can contribute thousands of rows (at native
    ~10Hz) while a typical 9s window contributes ~90 -- so a handful of rare
    long windows end up dominating the training set both in row COUNT and in
    magnitude (drift keeps growing the longer you integrate), which showed
    up empirically as training loss diverging instead of decreasing.
    Windows longer than max_window_s fall back to the original
    endpoint-only behavior (one real-GPS-anchored sample), regardless of
    realtime_interval_s, so real-time expansion only applies to windows in
    the range it's actually meant to help with -- ordinary GPS update gaps,
    not rare extended blackouts.
    """
    roll, pitch = roll_pitch_from_gravity(df["grav_x"].values, df["grav_y"].values, df["grav_z"].values)
    lin_ax = df["accel_x"].values - df["grav_x"].values
    lin_ay = df["accel_y"].values - df["grav_y"].values
    lin_az = df["accel_z"].values - df["grav_z"].values

    gyro_roll_rate = df["gyro_roll"].values   # rad/s, body-frame X axis
    gyro_pitch_rate = df["gyro_pitch"].values  # rad/s, body-frame Y axis
    gyro_yaw_rate = df["gyro_yaw"].values     # rad/s, body-frame Z axis
    t_s = df["t_ms"].values / 1000.0

    fixes = detect_gps_fixes(df)
    if len(fixes) < 2:
        return []

    gps_vx_all, gps_vy_all = gps_velocity_ne(df["gps_speed_kmh"].values, df["gps_bearing_deg"].values)
    bearing_rad_all = np.deg2rad(df["gps_bearing_deg"].values)

    samples = []

    for w in range(len(fixes) - 1):
        i0, i1 = fixes[w], fixes[w + 1]
        if i1 - i0 < 2:
            continue  # need at least one IMU step inside the window

        # GPS bearing/speed can be NaN at a real fix -- most commonly right
        # after a cold-start acquisition, or while stationary (heading is
        # undefined at zero speed). A window anchored on or ending at such a
        # fix has no valid ground truth, so skip it rather than let NaN
        # propagate through the whole window's integration.
        if (np.isnan(bearing_rad_all[i0]) or np.isnan(gps_vx_all[i0]) or np.isnan(gps_vy_all[i0])
                or np.isnan(gps_vx_all[i1]) or np.isnan(gps_vy_all[i1])):
            continue

        yaw0 = bearing_rad_all[i0]
        v0x, v0y = gps_vx_all[i0], gps_vy_all[i0]

        yaw = yaw0
        vx, vy = v0x, v0y
        pos_x, pos_y = 0.0, 0.0  # IMU-integrated position drift, local frame anchored at i0
        prev_t = t_s[i0]
        window_dt_s = t_s[i1] - t_s[i0]

        # window-end ground truth (the GPS fix we're integrating up to) --
        # computed up front since intermediate samples need it too
        gps_vx_end, gps_vy_end = gps_vx_all[i1], gps_vy_all[i1]
        gps_drift_x_end = gps_vx_end - v0x
        gps_drift_y_end = gps_vy_end - v0y
        gps_pos_x_end, gps_pos_y_end = latlon_to_local_xy(
            df["gps_lat"].values[i1], df["gps_lon"].values[i1],
            df["gps_lat"].values[i0], df["gps_lon"].values[i0],
        )

        window_gets_realtime_samples = (
            realtime_interval_s is not None and window_dt_s <= max_window_s
        )
        next_emit_t = (prev_t + realtime_interval_s) if window_gets_realtime_samples and realtime_interval_s else None

        # integrate row by row through the window: yaw via gyro, velocity via
        # world-frame linear acceleration built from that running yaw, and
        # position via trapezoidal integration of that same velocity
        for k in range(i0 + 1, i1 + 1):
            dt = t_s[k] - prev_t
            if dt <= 0 or dt > 2.0:  # guard against bad/missing timestamps
                prev_t = t_s[k]
                continue

            yaw_dot = yaw_rate_from_body_gyro(
                gyro_roll_rate[k], gyro_pitch_rate[k], gyro_yaw_rate[k],
                roll[k], pitch[k],
            )
            yaw = yaw + yaw_dot * dt
            R = rotation_matrix(yaw, pitch[k], roll[k])
            a_world = R @ np.array([lin_ax[k], lin_ay[k], lin_az[k]])
            new_vx = vx + a_world[0] * dt
            new_vy = vy + a_world[1] * dt
            pos_x += 0.5 * (vx + new_vx) * dt
            pos_y += 0.5 * (vy + new_vy) * dt
            vx, vy = new_vx, new_vy
            prev_t = t_s[k]

            # real-time intermediate sample. realtime_interval_s > 0: emit
            # roughly every that many seconds of elapsed time since the
            # window started. realtime_interval_s == 0: emit for EVERY raw
            # row in the window (native sampling rate, no subsampling at
            # all) -- one training example per IMU sample in the dataset.
            # Either way the target uses the linear-interpolation assumption
            # (see docstring). Skipped entirely when realtime_interval_s is
            # None, which reproduces the original one-sample-per-window
            # behavior.
            emit_this_row = window_gets_realtime_samples and k != i1 and (
                realtime_interval_s == 0 or t_s[k] >= next_emit_t
            )
            if emit_this_row:
                elapsed_s = t_s[k] - t_s[i0]
                frac = elapsed_s / window_dt_s if window_dt_s > 0 else 0.0
                samples.append({
                    "session_id": session_id,
                    "source_file": source_file,
                    "session_index_in_file": session_index_in_file,
                    "driver": driver,
                    "window_start_idx": i0,
                    "window_end_idx": i1,
                    "window_dt_s": elapsed_s,           # time since last real fix (this sample's "age")
                    "window_total_dt_s": window_dt_s,    # full gap to the next real fix (context only)
                    "is_window_end": False,
                    "v0x": v0x, "v0y": v0y,
                    "imu_drift_x": vx - v0x, "imu_drift_y": vy - v0y,
                    "gps_drift_x": gps_drift_x_end * frac, "gps_drift_y": gps_drift_y_end * frac,
                    "imu_end_vx": vx, "imu_end_vy": vy,
                    "gps_end_vx": v0x + gps_drift_x_end * frac, "gps_end_vy": v0y + gps_drift_y_end * frac,
                    "imu_pos_drift_x": pos_x, "imu_pos_drift_y": pos_y,
                    "gps_pos_drift_x": gps_pos_x_end * frac, "gps_pos_drift_y": gps_pos_y_end * frac,
                })
                if realtime_interval_s:  # periodic mode only -- every-row mode has no timer to advance
                    next_emit_t += realtime_interval_s

        # always emit the window-end sample too -- this one has REAL GPS
        # ground truth (not interpolated), so every window is still anchored
        # by a genuine measurement regardless of realtime_interval_s
        samples.append({
            "session_id": session_id,
            "source_file": source_file,
            "session_index_in_file": session_index_in_file,
            "driver": driver,
            "window_start_idx": i0,
            "window_end_idx": i1,
            "window_dt_s": window_dt_s,
            "window_total_dt_s": window_dt_s,
            "is_window_end": True,
            "v0x": v0x, "v0y": v0y,
            "imu_drift_x": vx - v0x, "imu_drift_y": vy - v0y,
            "gps_drift_x": gps_drift_x_end, "gps_drift_y": gps_drift_y_end,
            "imu_end_vx": vx, "imu_end_vy": vy,
            "gps_end_vx": gps_vx_end, "gps_end_vy": gps_vy_end,
            "imu_pos_drift_x": pos_x, "imu_pos_drift_y": pos_y,
            "gps_pos_drift_x": gps_pos_x_end, "gps_pos_drift_y": gps_pos_y_end,
        })

    return samples


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def find_input_files(input_path):
    """--input can be a single CSV file, or a directory to search recursively
    for S-*.csv files (the smartphone-sensor logs; V-*.csv vehicle CAN-bus
    files are ignored -- wrong format for this pipeline)."""
    p = Path(input_path)
    if p.is_file():
        return [p]
    if p.is_dir():
        files = sorted(p.rglob("S-*.csv")) + sorted(p.rglob("s-*.csv"))
        files = sorted(set(files))
        return files
    raise FileNotFoundError(f"--input path not found: {input_path}")


def driver_from_path(path):
    """Best-effort driver/vehicle label pulled from the folder structure,
    e.g. '.../Categorised IOVNB Dataset/Vta (Driver E)/Vta01a/S-Vta1a.csv'
    -> 'Vta (Driver E)'. Falls back to the immediate parent folder name."""
    parts = path.parts
    for part in parts:
        if "Driver" in part:
            return part
    return path.parent.name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                     help="A single CSV file, or a directory to search recursively for S-*.csv files")
    ap.add_argument("--output", required=True)
    ap.add_argument("--realtime-interval", type=float, default=None,
                     help="If set, also emit training samples INSIDE each GPS-fix window, not "
                          "just at the window's end -- trains the model for continuous "
                          "real-time correction instead of one-shot end-of-window correction. "
                          "Pass a positive number of seconds (e.g. 1.0) to sample periodically, "
                          "or 0 to emit a sample for EVERY raw row in the dataset (native "
                          "sampling rate, no subsampling). Intermediate targets are linearly "
                          "interpolated (see process_session docstring). Omit this flag entirely "
                          "for the original one-sample-per-window behavior.")
    ap.add_argument("--max-window-s", type=float, default=90.0,
                     help="Only windows (gaps between consecutive real GPS fixes) up to this "
                          "many seconds get the --realtime-interval treatment. Longer windows "
                          "(rare blackout-length outliers) fall back to a single endpoint sample, "
                          "same as the default behavior, so a handful of extreme-length windows "
                          "can't dominate the dataset by row count and drift magnitude. Ignored "
                          "if --realtime-interval isn't set. Default 90s.")
    args = ap.parse_args()

    input_files = find_input_files(args.input)
    if not input_files:
        print(f"No S-*.csv files found under {args.input}")
        return
    print(f"Found {len(input_files)} input file(s)")

    all_samples = []
    global_session_id = 0
    files_failed = []

    for path in input_files:
        try:
            df = load_raw(str(path))
        except Exception as e:
            print(f"  SKIPPED {path.name}: failed to load ({e})")
            files_failed.append(str(path))
            continue

        sessions = split_sessions(df)
        driver = driver_from_path(path)
        print(f"{path.name} [{driver}]: {len(df)} rows -> {len(sessions)} session(s)")

        for local_idx, s in enumerate(sessions):
            samples = process_session(s, session_id=global_session_id,
                                       source_file=path.name, driver=driver,
                                       session_index_in_file=local_idx,
                                       realtime_interval_s=args.realtime_interval,
                                       max_window_s=args.max_window_s)
            print(f"    session {global_session_id}: {len(s)} rows -> {len(samples)} training windows")
            all_samples.extend(samples)
            global_session_id += 1

    if files_failed:
        print(f"\n{len(files_failed)} file(s) skipped due to load errors:")
        for f in files_failed:
            print(f"  {f}")

    out_df = pd.DataFrame(all_samples)
    out_df.to_csv(args.output, index=False)
    print(f"\nWrote {len(out_df)} total samples from {global_session_id} sessions to {args.output}")
    if len(out_df):
        print(out_df[["window_dt_s", "imu_drift_x", "imu_drift_y", "gps_drift_x", "gps_drift_y"]].describe())


if __name__ == "__main__":
    main()