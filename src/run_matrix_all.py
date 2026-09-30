"""Full training matrix: 5 pools x 4 heads x {leave-one-out, tails}.

POOLS
  NEW     only the latest sessions (mount_f-i, pocket_f-i, hand_a-d)
  MOUNT   every mounted recording
  HAND    the hand-held recordings only
  ALL     every bike recording
  ALL+IO  ALL plus the IO-VNBD car pool (speed and velvec only -- IO-VNBD is
          loaded with V-file truth and no usable magnetometer path here)

HEADS
  speed   residual against the anchor speed; predicting 0 == hold-speed
  velvec  anchor-frame (vx, vy); predicting (0, v0) == hold-anchor velocity
  gyroc   correction to the calibrated-gyro heading, GPS bearing as truth
  magc    correction to the hard-iron compass heading, GPS bearing as truth

EVALUATION
  LORO    leave-one-recording-out. Honest: the test ride is never trained on.
  TAILS   last 10% of each recording, 150 s gap. Optimistic: shares the ride,
          its route and its magnetic surroundings. Reported alongside LORO
          precisely so the gap between them is visible -- on the compass filter
          that gap was 51 points, which is what exposed it as memorisation.

Per-recording features are cached to data/processed/cache/, so the pools reuse
one build rather than repeating the expensive part five times.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from train_full import build as build_speed
from run_head_correct import build_head
from run_compass_filter import build as build_mag
from idr_core import load_session_gps, wrap

CACHE = Path("data/processed/cache")
CACHE.mkdir(parents=True, exist_ok=True)
T0 = time.time()

MOUNTS = [f"mount_{c}" for c in "abcdefghi"]
POCKETS = [f"pocket_{c}" for c in "abcdefghi"]
HANDS = [f"hand_{c}" for c in "abcd"]
OLD = ["trip11", "trip12", "trip13"]
NEW = [f"mount_{c}" for c in "fghi"] + [f"pocket_{c}" for c in "fghi"] + HANDS
ALLB = MOUNTS + POCKETS + HANDS + OLD
HOLD = "trip1"                     # never in any pool

POOLS = {"NEW": NEW, "MOUNT": MOUNTS, "HAND": HANDS, "ALL": ALLB}
HEADS = ["speed", "velvec", "gyroc", "magc"]
MAX_FOLDS = 6
TAIL, GAP_S = 0.10, 150.0
deg = np.rad2deg


def log(*a):
    print(f"[{time.time()-T0:7.0f}s]", *a, flush=True)


def cached(nm, head):
    """Build (or load) the feature/target arrays for one recording and head."""
    f = CACHE / f"{nm}__{head}.npz"
    if f.exists():
        z = np.load(f)
        return z["X"], z["Y"], z["T"]
    out = None
    if head in ("speed", "velvec"):
        p = Path(f"data/raw/{nm}.csv")
        if not p.exists():
            return None
        rs, truth = load_session_gps(p)
        if rs is None:
            return None
        r = build_speed(rs, truth, hop=5)
        if r is not None:
            X, Y = r
            out = (X, Y, np.linspace(0, len(rs) / 10.0, len(X)))
    elif head == "gyroc":
        r = build_head(nm, hop=10)
        if r is not None:
            out = (r[0], r[1], r[2])
    elif head == "magc":
        r = build_mag(nm, hop=10)
        if r is not None:
            out = (r[0], r[1].reshape(-1, 1), r[2])
    if out is None:
        return None
    np.savez(f, X=out[0], Y=out[1], T=out[2])
    return out


def targets(Y, head):
    """(input target, a function scoring predictions against the baseline)."""
    if head == "speed":
        v0, spd = Y[:, 2], Y[:, 0]
        def sc(p):
            pr = np.clip(v0 + p.ravel(), 0, None)
            h = np.abs(v0 - spd).mean(); m = np.abs(pr - spd).mean()
            return h, m, 100 * (h - m) / h
        return (spd - v0)[:, None], sc
    if head == "velvec":
        v0, spd, dh = Y[:, 2], Y[:, 0], Y[:, 1]
        vx, vy = spd * np.sin(dh), spd * np.cos(dh)
        def sc(p):
            px, py = p[:, 0], v0 + p[:, 1]
            h = np.hypot(vx, vy - v0).mean()
            m = np.hypot(vx - px, vy - py).mean()
            return h, m, 100 * (h - m) / h
        return np.column_stack([vx, vy - v0]), sc
    if head == "gyroc":
        true, gyro = Y[:, 0], Y[:, 1]
        def sc(p):
            h = deg(np.abs(wrap(true - gyro))).mean()
            m = deg(np.abs(wrap(true - (gyro + p.ravel())))).mean()
            return h, m, 100 * (h - m) / h
        return wrap(true - gyro)[:, None], sc
    # magc: Y is the compass's own heading error, offset removed
    err = Y[:, 0]
    def sc(p):
        h = deg(np.abs(err)).mean()
        m = deg(np.abs(wrap(err - p.ravel()))).mean()
        return h, m, 100 * (h - m) / h
    return err[:, None], sc


def main():
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    L = tf.keras.layers

    def fit(xt, yt, nout, seed=1):
        tf.keras.utils.set_random_seed(seed)
        m = tf.keras.Sequential([L.Input((xt.shape[1],)),
                                 L.Dense(128, activation="relu"), L.Dropout(.15),
                                 L.Dense(64, activation="relu"), L.Dropout(.15),
                                 L.Dense(32, activation="relu"), L.Dense(nout)])
        m.compile(tf.keras.optimizers.Adam(1e-3), loss="huber")
        m.fit(xt, yt, validation_split=0.12, epochs=60, batch_size=2048, verbose=0,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=8,
                                                          restore_best_weights=True),
                         tf.keras.callbacks.ReduceLROnPlateau(patience=4, factor=.5)])
        return m

    # ---- build / load every recording once per head ----
    store = {}
    for head in HEADS:
        for nm in sorted(set(ALLB + [HOLD])):
            r = cached(nm, head)
            if r is not None:
                store[(nm, head)] = r
        got = sum(1 for k in store if k[1] == head)
        log(f"cached {head}: {got} recordings")

    io = None
    for head in ("speed", "velvec"):
        f = CACHE / f"__IO__{head}.npz"
        if f.exists():
            z = np.load(f)
            io = (z["X"], z["Y"])
            log(f"IO-VNBD cache for {head}: {len(io[0])} rows")
            break
    if io is None:
        from idr_core import load_session
        Xs, Ys = [], []
        files = sorted(Path("data/raw/iovnbd_data").rglob("S-*.csv"))
        for p in files[:60]:
            try:
                rs, truth = load_session(p)
            except Exception:
                continue
            if rs is None:
                continue
            r = build_speed(rs, truth, hop=10)
            if r:
                Xs.append(r[0]); Ys.append(r[1])
        if Xs:
            io = (np.concatenate(Xs), np.concatenate(Ys))
            np.savez(CACHE / "__IO__speed.npz", X=io[0], Y=io[1])
            log(f"IO-VNBD built: {len(io[0])} rows")

    results = []
    for head in HEADS:
        pools = dict(POOLS)
        if head in ("speed", "velvec") and io is not None:
            pools = dict(POOLS, **{"ALL+IO": ALLB})
        for pname, members in pools.items():
            have = [n for n in members if (n, head) in store]
            if len(have) < 2:
                log(f"{head:6s} {pname:7s}  skipped (only {len(have)} recordings)")
                continue
            X = np.concatenate([store[(n, head)][0] for n in have])
            Y = np.concatenate([store[(n, head)][1] for n in have])
            S = np.concatenate([np.full(len(store[(n, head)][0]), n) for n in have])
            T = np.concatenate([store[(n, head)][2] for n in have])
            extra = None
            if pname == "ALL+IO" and io is not None:
                extra = io
            Yt, scorer = targets(Y, head)
            nout = Yt.shape[1]

            # ---- LORO ----
            folds = have if len(have) <= MAX_FOLDS else \
                list(np.array(have)[np.linspace(0, len(have) - 1, MAX_FOLDS).astype(int)])
            gains = []
            for held in folds:
                tr = S != held
                te = S == held
                if te.sum() < 50:
                    continue
                Xtr, Ytr = X[tr], Yt[tr]
                if extra is not None:
                    Ye, _ = targets(extra[1], head)
                    Xtr = np.concatenate([Xtr, extra[0]])
                    Ytr = np.concatenate([Ytr, Ye])
                mu = Xtr.mean(0); sd = Xtr.std(0)
                sd = np.where(sd < 1e-6, 1.0, sd) + 1e-8
                m = fit((Xtr - mu) / sd, Ytr, nout)
                p = m.predict((X[te] - mu) / sd, batch_size=16384, verbose=0)
                _, _, g = targets(Y[te], head)[1](p)
                gains.append(g)
            loro = np.mean(gains) if gains else float("nan")

            # ---- TAILS ----
            te = np.zeros(len(X), bool); tr = np.zeros(len(X), bool)
            for nm in have:
                k = S == nm
                cut = T[k].max() * (1 - TAIL)
                te |= k & (T >= cut); tr |= k & (T < cut - GAP_S)
            Xtr, Ytr = X[tr], Yt[tr]
            if extra is not None:
                Ye, _ = targets(extra[1], head)
                Xtr = np.concatenate([Xtr, extra[0]])
                Ytr = np.concatenate([Ytr, Ye])
            mu = Xtr.mean(0); sd = Xtr.std(0); sd = np.where(sd < 1e-6, 1.0, sd) + 1e-8
            m = fit((Xtr - mu) / sd, Ytr, nout)
            p = m.predict((X[te] - mu) / sd, batch_size=16384, verbose=0)
            _, _, tails = targets(Y[te], head)[1](p)

            # ---- fully unseen vehicle ----
            un = float("nan")
            if (HOLD, head) in store:
                Xh, Yh, _ = store[(HOLD, head)]
                ph = m.predict((Xh - mu) / sd, batch_size=16384, verbose=0)
                _, _, un = targets(Yh, head)[1](ph)

            results.append((head, pname, len(have), len(X), loro, tails, un))
            log(f"{head:6s} {pname:7s} rec={len(have):2d} rows={len(X):7d}   "
                f"LORO {loro:+7.1f}%   TAILS {tails:+7.1f}%   trip1 {un:+7.1f}%   "
                f"gap {tails-loro:+6.1f}")

    log("")
    log("=" * 86)
    log(f"{'head':7s} {'pool':8s} {'rec':>3s} {'rows':>8s} {'LORO':>9s} {'TAILS':>9s}"
        f" {'trip1':>9s} {'gap':>7s}")
    log("=" * 86)
    for h, p, nr, nx, lo, ta, un in results:
        log(f"{h:7s} {p:8s} {nr:3d} {nx:8d} {lo:+9.1f} {ta:+9.1f} {un:+9.1f} {ta-lo:+7.1f}")
    log("")
    log("LORO is the honest number. A large TAILS-minus-LORO gap means the model")
    log("memorised the ride rather than learning a transferable rule.")
    json.dump([{"head": h, "pool": p, "recordings": nr, "rows": int(nx),
                "loro": float(lo), "tails": float(ta), "trip1": float(un)}
               for h, p, nr, nx, lo, ta, un in results],
              open("results/logs/matrix_all.json", "w"), indent=1)
    log("DONE")


if __name__ == "__main__":
    main()
