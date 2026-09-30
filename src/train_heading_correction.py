"""Two heading-correction heads, each fitted to the estimate it corrects.

There was a mismatch in the pipeline: the one head shipped was trained on the
residual against the GYRO (true - dh_gyro) but was being added to the FUSED
heading. That is the wrong quantity, and it showed -- ML_FUSED came out slightly
worse than fused, and ML_GYRO slightly worse than gyro, on mount_h.

So: one head per baseline, each trained on its own residual.

  head_gyro    target true - dh_gyro    ->  applied to the gyro heading
  head_fused   target true - dh_fused   ->  applied to the fused heading

Both are validated leave-one-recording-out, which is the only number worth
trusting here; a held-out tail shares the ride and has overstated every heading
result so far by 8-51 points.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from run_matrix_all import ALLB, HOLD, cached, log
from train_full import names
from idr_core import wrap

MAX_FOLDS = 8
deg = np.rad2deg
COLS = names() + ["dh_gyro", "dh_fused", "abs_dh_fused", "yawrate_now"]


def main():
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    L = tf.keras.layers

    log("loading cached heading features")
    store = {}
    for nm in sorted(set(ALLB + [HOLD])):
        r = cached(nm, "gyroc")
        if r is not None:
            store[nm] = r
    have = [n for n in ALLB if n in store]
    X = np.concatenate([store[n][0] for n in have])
    Y = np.concatenate([store[n][1] for n in have])
    S = np.concatenate([np.full(len(store[n][0]), n) for n in have])
    log(f"{len(have)} recordings, {len(X)} rows, {X.shape[1]} inputs")

    # Y = [dh_true, dh_gyro, dh_fused]
    BASE = {"gyro": 1, "fused": 2}

    def fit(xt, yt, seed=1):
        tf.keras.utils.set_random_seed(seed)
        m = tf.keras.Sequential([L.Input((xt.shape[1],)),
                                 L.Dense(256, activation="relu"), L.Dropout(.1),
                                 L.Dense(128, activation="relu"), L.Dropout(.1),
                                 L.Dense(64, activation="relu"), L.Dense(1)])
        m.compile(tf.keras.optimizers.Adam(1e-3), loss="huber")
        m.fit(xt, yt, validation_split=0.1, epochs=80, batch_size=2048, verbose=0,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=10,
                                                          restore_best_weights=True),
                         tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=.5)])
        return m

    for bname, bi in BASE.items():
        resid = wrap(Y[:, 0] - Y[:, bi])
        log("")
        log("#" * 68)
        log(f"HEAD for '{bname}'   target = true - dh_{bname}")
        log("#" * 68)
        folds = have if len(have) <= MAX_FOLDS else \
            list(np.array(have)[np.linspace(0, len(have) - 1, MAX_FOLDS).astype(int)])
        gains = []
        for held in folds:
            tr = S != held
            te = S == held
            if te.sum() < 50:
                continue
            mu = X[tr].mean(0); sd = X[tr].std(0)
            sd = np.where(sd < 1e-6, 1.0, sd) + 1e-8
            m = fit((X[tr] - mu) / sd, resid[tr])
            p = m.predict((X[te] - mu) / sd, batch_size=16384, verbose=0).ravel()
            b = deg(np.abs(wrap(Y[te, 0] - Y[te, bi]))).mean()
            c = deg(np.abs(wrap(Y[te, 0] - (Y[te, bi] + p)))).mean()
            gains.append(100 * (b - c) / b)
            log(f"  {held:9s} n={te.sum():6d}   {bname} {b:6.2f}  corrected {c:6.2f} deg"
                f"   {gains[-1]:+6.1f}%")
        if gains:
            log(f"  LORO MEAN {np.mean(gains):+6.1f}%")

        # final head on everything, exported
        mu = X.mean(0); sd = X.std(0); sd = np.where(sd < 1e-6, 1.0, sd) + 1e-8
        m = fit((X - mu) / sd, resid)
        if HOLD in store:
            Xh, Yh, _ = store[HOLD]
            p = m.predict((Xh - mu) / sd, batch_size=16384, verbose=0).ravel()
            b = deg(np.abs(wrap(Yh[:, 0] - Yh[:, bi]))).mean()
            c = deg(np.abs(wrap(Yh[:, 0] - (Yh[:, bi] + p)))).mean()
            log(f"  trip1 (unseen)  {bname} {b:6.2f}  corrected {c:6.2f} deg"
                f"   {100*(b-c)/b:+6.1f}%")
        out = Path(f"results/models/v3/head_{bname}")
        out.mkdir(parents=True, exist_ok=True)
        m.save(out / "model.keras")
        json.dump({"input_cols": COLS, "mean": mu.tolist(), "std": sd.tolist(),
                   "baseline": bname}, open(out / "norm_stats.json", "w"))
        tfl = tf.lite.TFLiteConverter.from_keras_model(m).convert()
        (out / "model.tflite").write_bytes(tfl)
        log(f"  exported {out} ({len(tfl)/1024:.0f} KB)")
    log("DONE")


if __name__ == "__main__":
    main()
