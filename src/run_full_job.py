"""Stage 1: score the SHIPPED model on the new recordings.
Stage 2: retrain on every sample of all bike data, then test on held-out data.

Held out, and never seen in training:
  - trip1 entirely (a different phone, different mount, different vehicle)
  - the last 10% of every bike recording, with a 150 s gap either side

The tail split is deliberate: consecutive windows overlap almost completely, so
a random 10% of rows would put near-duplicates on both sides and report
memorisation as accuracy.
"""
import json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from train_full import build, names
from idr_core import load_session_gps, FS

BIKE = ["trip11", "trip12", "trip13",
        "mount_a", "mount_b", "mount_c", "mount_d", "mount_e",
        "pocket_a", "pocket_b", "pocket_c", "pocket_d", "pocket_e"]
NEW = ["mount_c", "mount_d", "mount_e", "pocket_c", "pocket_d", "pocket_e"]
HOLD = "trip1"
TAIL, GAP_S = 0.10, 150.0
T0 = time.time()


def log(*a):
    print(f"[{time.time()-T0:7.0f}s]", *a, flush=True)


def collect(nms, hop=1):
    Xs, Ys, Ss, Ts = [], [], [], []
    for nm in nms:
        p = Path(f"data/raw/{nm}.csv")
        if not p.exists():
            log(f"  {nm}: missing"); continue
        try:
            rs, truth = load_session_gps(p)
        except Exception as e:
            log(f"  {nm}: load failed ({e})"); continue
        if rs is None:
            log(f"  {nm}: unusable (no GPS?)"); continue
        r = build(rs, truth, hop=hop)
        if r is None:
            log(f"  {nm}: no samples"); continue
        X, Y = r
        Xs.append(X); Ys.append(Y); Ss.append(np.full(len(X), nm))
        Ts.append(np.linspace(0, len(rs) / FS, len(X)))
        log(f"  {nm}: {len(X):7d} rows  ({len(rs)/FS:.0f}s)")
    if not Xs:
        return None
    return (np.concatenate(Xs), np.concatenate(Ys),
            np.concatenate(Ss), np.concatenate(Ts))


def score(pred, v0, spd, tag):
    h = np.abs(v0 - spd).mean(); m = np.abs(pred - spd).mean()
    c = np.corrcoef(pred, spd)[0, 1] if np.std(pred) > 1e-9 else float("nan")
    log(f"    {tag:12s} hold {h:6.3f}  model {m:6.3f} m/s  ({100*(h-m)/h:+6.1f}%)  corr {c:+.3f}")
    return m


def main():
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.keras.utils.set_random_seed(1)
    L = tf.keras.layers

    log("#" * 70)
    log("STAGE 1  —  SHIPPED model (results/models/app_bike) on the NEW recordings")
    log("#" * 70)
    st = json.load(open("results/models/app_bike/norm_stats.json"))
    mu0 = np.array(st["mean"]); sd0 = np.array(st["std"])
    net0 = tf.keras.models.load_model("results/models/app_bike/speed_model.keras")
    got = collect(NEW)
    if got:
        Xn_, Yn_, Sn_, _ = got
        for nm in NEW:
            k = Sn_ == nm
            if k.sum() < 50:
                continue
            p = np.clip(Yn_[k, 2] + net0.predict((Xn_[k] - mu0) / sd0,
                        batch_size=16384, verbose=0).ravel(), 0, None)
            score(p, Yn_[k, 2], Yn_[k, 0], nm)
        del Xn_, Yn_, Sn_

    log("")
    log("#" * 70)
    log("STAGE 2  —  RETRAIN on every sample of all bike data")
    log("#" * 70)
    log("building training pool (trip1 excluded entirely)")
    got = collect(BIKE)
    if not got:
        log("no data"); return
    X, Y, S, T = got
    te = np.zeros(len(X), bool); tr = np.zeros(len(X), bool)
    for nm in sorted(set(S.tolist())):
        k = S == nm
        cut = T[k].max() * (1 - TAIL)
        te |= k & (T >= cut)
        tr |= k & (T < cut - GAP_S)
    log(f"train {tr.sum()} rows | held-out tails {te.sum()} rows "
        f"({100*te.sum()/len(X):.1f}%) | {len(X)-tr.sum()-te.sum()} dropped to the {GAP_S:.0f}s gap")

    v0 = Y[:, 2]; spd = Y[:, 0]; dh = Y[:, 1]
    mu = X[tr].mean(0); sd = X[tr].std(0); sd = np.where(sd < 1e-6, 1.0, sd) + 1e-8
    Xn = (X - mu) / sd

    def mlp(nout):
        m = tf.keras.Sequential([L.Input((X.shape[1],)),
                                 L.Dense(256, activation="relu"), L.Dropout(.1),
                                 L.Dense(128, activation="relu"), L.Dropout(.1),
                                 L.Dense(64, activation="relu"), L.Dense(nout)])
        m.compile(tf.keras.optimizers.Adam(1e-3), loss="huber")
        return m

    cb = [tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
          tf.keras.callbacks.ReduceLROnPlateau(patience=4, factor=.5)]

    log("training SPEED head …")
    ms = mlp(1)
    ms.fit(Xn[tr], (spd - v0)[tr], validation_split=0.1, epochs=80,
           batch_size=2048, verbose=2, callbacks=cb)

    log("")
    log("=== held-out TAILS (never trained on) ===")
    p = np.clip(v0[te] + ms.predict(Xn[te], batch_size=16384, verbose=0).ravel(), 0, None)
    score(p, v0[te], spd[te], "ALL TAILS")
    for nm in sorted(set(S[te].tolist())):
        k = te & (S == nm)
        if k.sum() < 30:
            continue
        pk = np.clip(v0[k] + ms.predict(Xn[k], batch_size=16384, verbose=0).ravel(), 0, None)
        score(pk, v0[k], spd[k], nm)

    log("")
    log("=== trip1 (fully unseen: never in the training pool) ===")
    g1 = collect([HOLD])
    if g1:
        X1, Y1, _, _ = g1
        p1 = np.clip(Y1[:, 2] + ms.predict((X1 - mu) / sd,
                     batch_size=16384, verbose=0).ravel(), 0, None)
        score(p1, Y1[:, 2], Y1[:, 0], "trip1")

    out = Path("results/models/bike_v2"); out.mkdir(parents=True, exist_ok=True)
    ms.save(out / "speed_model.keras")
    json.dump({"input_cols": names(), "mean": mu.tolist(), "std": sd.tolist(),
               "scales": [2.0, 5.0, 10.0], "fs": 10.0}, open(out / "norm_stats.json", "w"))
    conv = tf.lite.TFLiteConverter.from_keras_model(ms)
    (out / "speed_model.tflite").write_bytes(conv.convert())
    log(f"saved {out}  (keras + tflite + norm_stats)")
    log("DONE")


if __name__ == "__main__":
    main()
