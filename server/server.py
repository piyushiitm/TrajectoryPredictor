"""IDR replay service: upload a recording, watch the engine drive it live.

Same engine as the phone and the offline scripts -- the v3 speed head, the
hard-iron compass with magnetic spike rejection, the complementary fusion with
elapsed-dependent tau, and the learned heading correction. Nothing here
re-implements the pipeline; it imports it, so the browser shows what the app
would have done on that ride.

    ./venv/bin/python web/server.py          then open http://127.0.0.1:8000

Uploads land in data/raw/uploads/ and are listed alongside every recording
already in data/raw/, so existing trips can be replayed without re-uploading.
"""
import io
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
UPLOADS = ROOT / "data" / "raw" / "uploads"
UPLOADS.mkdir(parents=True, exist_ok=True)

# Only a TFLite interpreter is needed at serve time. Prefer the small LiteRT
# runtime (fits a 512 MB host); fall back to full TensorFlow if that's what's installed.
try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    Interpreter = tf.lite.Interpreter

from idr_core import (load_session_gps, genuine_fixes, gyro_matrix, calibrate_yaw,
                      yaw_rate_from_cal, model_speed_track2, wrap, FS)
from sim_compass_pipeline import compass, tau_for


class TFLiteModel:
    """Wraps a .tflite interpreter behind the same .predict() the engine
    expects from a Keras model -- no .keras checkpoints ship in the repo
    (gitignored as TFLite build artefacts), only the exported .tflite."""

    def __init__(self, path):
        self._interp = Interpreter(model_path=str(path))
        self._interp.allocate_tensors()
        self._in = self._interp.get_input_details()[0]
        self._out = self._interp.get_output_details()[0]

    def predict(self, X, batch_size=None, verbose=0):
        X = np.asarray(X, dtype=self._in["dtype"])
        if tuple(self._interp.get_input_details()[0]["shape"]) != X.shape:
            self._interp.resize_tensor_input(self._in["index"], X.shape)
            self._interp.allocate_tensors()
        self._interp.set_tensor(self._in["index"], X)
        self._interp.invoke()
        return self._interp.get_tensor(self._out["index"])


# HEAD_DIR corrects the gyro heading (h["mlgyro"] below), so it's the "gyro"
# baseline head, not "fused" -- see train_heading_correction.py's BASE dict.
HEAD_DIR = ROOT / "models" / "heading_correction_gyro"

# Two speed heads, the same pair the phone carries.
#   tuned    the 25 bike recordings only -- sharper on this bike
#   general  those plus the IO-VNBD car pool -- holds up on other vehicles
SPEED_DIRS = {"tuned": ROOT / "models" / "speed_tuned",
              "general": ROOT / "models" / "speed_general"}
SPEED = {}
for _k, _d in SPEED_DIRS.items():
    _st = json.load(open(_d / "norm_stats.json"))
    SPEED[_k] = (np.array(_st["mean"]), np.array(_st["std"]), _st["input_cols"],
                 TFLiteModel(_d / "model.tflite"))
_hs = json.load(open(HEAD_DIR / "norm_stats.json"))
HMU, HSD = np.array(_hs["mean"]), np.array(_hs["std"])
HNET = TFLiteModel(HEAD_DIR / "model.tflite")

# Recordings the speed heads trained on. Their numbers are optimistic and the UI
# says so: a model scored on its own training data is not evidence of anything.
import config
TRAINED = set(
    [f"mount_{c}" for c in "abcdefghi"] +
    [f"pocket_{c}" for c in "abcdefghi"] +
    [f"hand_{c}" for c in "abcd"] +
    ["trip11", "trip12", "trip13"] + config.TRAIN)

# Where recordings are looked up: uploads first, then the legacy data/raw/, then
# data/recordings/ (what server/fetch_data.py fills from the shared Drive folder).
DATA_DIRS = ((UPLOADS, "upload"), (ROOT / "data" / "raw", "local"),
             (ROOT / "data" / "recordings", "local"))

app = FastAPI(title="IDR replay")

# If the deploy's build step left no recordings (Drive hiccup), fetch them in the
# background on startup so the service heals itself without a redeploy.
import threading
_RECS = ROOT / "data" / "recordings"
if not any(_RECS.glob("*.csv")) and os.environ.get("DATA_FOLDER_URL"):
    import fetch_data
    threading.Thread(target=fetch_data.main, daemon=True).start()


@app.get("/api/status")
def status():
    """What data the service has, and what the last Drive fetch did."""
    logf = ROOT / "data" / "fetch_log.txt"
    return {d.name if d != UPLOADS else "uploads":
            sorted(p.name for p in d.glob("*.csv")) for d, _ in DATA_DIRS} | {
        "fetch_log": logf.read_text().splitlines()[-40:] if logf.exists() else []}
CACHE = {}
SESSIONS = {}


def find(name):
    for d, _ in DATA_DIRS:
        p = d / f"{name}.csv"
        if p.exists():
            return p
    raise HTTPException(404, f"no recording named {name}")


@app.get("/api/trips")
def trips():
    seen, out = set(), []
    for d, tag in DATA_DIRS:
        for p in sorted(d.glob("*.csv")):
            if p.stem in seen:
                continue
            seen.add(p.stem)
            out.append({"name": p.stem, "source": tag,
                        "mb": round(p.stat().st_size / 1e6, 1),
                        "trained": p.stem in TRAINED})
    return out


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "expected a .csv recording")
    dest = UPLOADS / file.filename
    dest.write_bytes(await file.read())
    CACHE.pop(dest.stem, None)
    return {"name": dest.stem, "mb": round(dest.stat().st_size / 1e6, 1)}


def run_engine(name, cal_s=300.0, start_s=None, duration_s=0.0, model="tuned"):
    """Dead-reckon a blackout and return every track, in lat/lon."""
    p = find(name)
    rs, truth = load_session_gps(p)
    if rs is None:
        raise HTTPException(422, f"{name}: no usable GPS or too short")
    psi, _bstd = compass(str(p), rs)
    t = rs["t_ms"].values / 1000.0
    n = len(rs)
    hdg, spd = truth["hdg"], truth["spd"]
    fx = genuine_fixes(rs)
    W = gyro_matrix(rs)

    A = int(cal_s * FS) if start_s is None else int(start_s * FS)
    A = max(A, int(cal_s * FS))
    if start_s is None:
        # Skip forward to a window where the vehicle is actually moving. A
        # stationary stretch makes the distance denominator tiny, so the drift
        # PERCENTAGE explodes even when the position is fine -- the ceiling
        # track shows the same number, which is the giveaway.
        probe = int(max(duration_s, 30.0) * FS)
        while A + probe < n and spd[A:A + probe].mean() < 3.0:
            A += int(10 * FS)
    B = n - 1 if duration_s <= 0 else min(n - 1, A + int(duration_s * FS))
    if B - A < 50:
        raise HTTPException(422, "outage window too short for this recording")
    w4, q = calibrate_yaw(W, t, hdg, fx, max(0, A - int(cal_s * FS)), A)
    if w4 is None:
        raise HTTPException(422, "gyro calibration failed on this recording")

    m = B - A
    rate = yaw_rate_from_cal(W[A:B], w4)
    MU, SD, CL, NET = SPEED.get(model, SPEED["tuned"])
    v = model_speed_track2(rs, NET, MU, SD, CL, A, A, B, spd[A])
    if v is None:
        v = np.full(m, spd[A])

    # GPS bearing with the model's speed: heading is perfect, so whatever error
    # remains is the SPEED head's alone. It is a ceiling, not a deployable mode --
    # bearing is exactly what a real blackout takes away.
    h = {"gpshdg": np.asarray(hdg[A:B], float),
         "gyro": hdg[A] + np.cumsum(rate) / FS}
    if psi is not None:
        off = wrap(hdg[A] - psi[A])
        hm = psi[A:B] + off
        cur, hf = hdg[A], np.empty(m)
        for i in range(m):
            al = (1 / FS) / (tau_for(i / FS) + 1 / FS)
            cur += rate[i] / FS
            cur += al * wrap(hm[i] - cur)
            hf[i] = cur
        h["fused"] = hf
        h["mag"] = hm
    else:
        h["fused"] = h["gyro"].copy()

    # The correction head is fitted to the GYRO's residual (true - dh_gyro), so
    # it belongs on the gyro heading. A head trained on the FUSED residual was
    # tried and scored -7.1% leave-one-out against +21.3% for this one: once the
    # compass has taken the drift out, what remains is compass noise, and there
    # is nothing learnable in it.
    h["mlgyro"] = h["gyro"].copy()
    try:
        from run_head_correct import build_head
        r = build_head(name, hop=10)
        if r is not None:
            Xh, _Yh, Th = r
            c = HNET.predict((Xh - HMU) / HSD, batch_size=16384, verbose=0).ravel()
            ok = (Th >= t[A]) & (Th <= t[B - 1])
            if ok.sum() > 10:
                idx = np.clip(((Th[ok] - t[A]) * FS).astype(int), 0, m - 1)
                corr = np.interp(np.arange(m), idx, np.clip(c[ok], -0.5, 0.5))
                h["mlgyro"] = h["gyro"] + corr
    except Exception:
        traceback.print_exc()

    lat0, lon0 = float(truth["lat"][A]), float(truth["lon"][A])
    kx = np.cos(np.radians(lat0)) * 6371000.0
    tracks = {}
    for k, hh in h.items():
        e = np.cumsum(v * np.sin(hh)) / FS
        nn = np.cumsum(v * np.cos(hh)) / FS
        tracks[k] = (lat0 + np.degrees(nn / 6371000.0),
                     lon0 + np.degrees(e / kx))
    tracks["hold"] = ((lat0 + np.degrees(np.cumsum(spd[A] * np.cos(h["fused"])) / FS / 6371000.0)),
                      (lon0 + np.degrees(np.cumsum(spd[A] * np.sin(h["fused"])) / FS / kx)))
    tlat, tlon = truth["lat"][A:B], truth["lon"][A:B]

    # decimate to about 1 Hz so the payload stays small and plays smoothly
    step = max(1, int(FS))
    sl = slice(0, m, step)
    dist = np.cumsum(spd[A:B]) / FS

    def ll(a):
        return [[round(float(x), 6), round(float(y), 6)]
                for x, y in zip(a[0][sl], a[1][sl])]

    out = {"name": name, "model": model, "trained_on": name in TRAINED,
           "start_s": round(A / FS, 1),
           "hz": FS / step, "cal_r2": round(float(q), 3),
           "seconds": round(m / FS, 1), "distance_m": round(float(dist[-1]), 1),
           "speed": [round(float(x), 2) for x in v[sl]],
           "dist": [round(float(x), 1) for x in dist[sl]],
           "truth": [[round(float(a), 6), round(float(b), 6)]
                     for a, b in zip(tlat[sl], tlon[sl])],
           "tracks": {k: ll(a) for k, a in tracks.items()},
           "final": {}}
    for k, a in tracks.items():
        dy = (a[0][-1] - tlat[-1]) * 6371000.0 * np.pi / 180
        dx = (a[1][-1] - tlon[-1]) * kx * np.pi / 180
        err = float(np.hypot(dx, dy))
        out["final"][k] = {"m": round(err, 1),
                           "pct": round(100 * err / max(dist[-1], 1), 1)}
    return out


@app.get("/api/replay/{name}")
def replay(name: str, cal_s: float = 300.0, start_s: float = -1,
           duration_s: float = 0.0, model: str = "tuned"):
    key = (name, cal_s, start_s, duration_s, model)
    if key not in CACHE:
        CACHE[key] = run_engine(name, cal_s,
                                None if start_s < 0 else start_s, duration_s, model)
        if len(CACHE) > 12:
            CACHE.pop(next(iter(CACHE)))
    return JSONResponse(CACHE[key])


@app.get("/api/windows/{name}")
def windows(name: str, duration_s: float = 60.0, cal_s: float = 300.0,
            model: str = "tuned", track: str = "fused"):
    """Score every outage window in a recording, so the UI can jump to the
    best and worst ones instead of the operator hunting for them.

    Picking windows by their own score is cherry-picking, which is exactly why
    both ends are offered: the worst case is the honest half of the story.
    """
    key = ("win", name, duration_s, cal_s, model)
    if key in CACHE:
        return JSONResponse(CACHE[key])
    p = find(name)
    rs, truth = load_session_gps(p)
    if rs is None:
        raise HTTPException(422, f"{name}: no usable GPS")
    psi, _ = compass(str(p), rs)
    t = rs["t_ms"].values / 1000.0
    n = len(rs)
    hdg, spd = truth["hdg"], truth["spd"]
    from idr_core import truth_xy
    te_, tn_ = truth_xy(truth)
    fx = genuine_fixes(rs)
    W = gyro_matrix(rs)
    MU, SD, CL, NET = SPEED.get(model, SPEED["tuned"])
    cal = int(cal_s * FS)
    step = int(duration_s * FS)
    rows = []
    for a in range(cal, n - step, step):
        b = a + step
        if spd[a:b].mean() < 3.0:
            continue
        dist = float(np.trapezoid(spd[a:b], dx=1 / FS))
        if dist < 30:
            continue
        w4, q = calibrate_yaw(W, t, hdg, fx, a - cal, a)
        if w4 is None:
            continue
        rate = yaw_rate_from_cal(W[a:b], w4)
        hs = {"gyro": hdg[a] + np.cumsum(rate) / FS}
        if psi is not None:
            off = wrap(hdg[a] - psi[a])
            hm = psi[a:b] + off
            cur, hf = hdg[a], np.empty(step)
            for i in range(step):
                al = (1 / FS) / (tau_for(i / FS) + 1 / FS)
                cur += rate[i] / FS
                cur += al * wrap(hm[i] - cur)
                hf[i] = cur
            hs["fused"] = hf
            hs["mag"] = hm
        else:
            hs["fused"] = hs["gyro"]
        v = model_speed_track2(rs, NET, MU, SD, CL, a, a, b, spd[a])
        if v is None:
            v = np.full(step, spd[a])
        dE, dN = te_[b - 1] - te_[a], tn_[b - 1] - tn_[a]
        r = {"start_s": round(a / FS, 1), "dist_m": round(dist, 1),
             "cal_r2": round(float(q), 3)}
        for k, hh in hs.items():
            e = np.cumsum(v * np.sin(hh)) / FS
            nn = np.cumsum(v * np.cos(hh)) / FS
            r[k] = round(100 * float(np.hypot(e[-1] - dE, nn[-1] - dN)) / dist, 1)
        rows.append(r)
    if not rows:
        raise HTTPException(422, "no scorable windows in this recording")
    key_track = track if track in rows[0] else "gyro"
    rows.sort(key=lambda z: z[key_track])
    res = {"name": name, "duration_s": duration_s, "track": key_track,
           "n": len(rows), "best": rows[:5], "worst": rows[-5:][::-1],
           "median": rows[len(rows) // 2][key_track]}
    CACHE[key] = res
    if len(CACHE) > 12:
        CACHE.pop(next(iter(CACHE)))
    return JSONResponse(res)


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"),
          name="static")

if __name__ == "__main__":
    import os
    import uvicorn
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", 8000))
    print(f"  open http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")
