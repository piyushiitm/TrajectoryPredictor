"""Final v3 models: trained on ALL data, then tested on a vehicle in no pool.

The matrix picked the winners by leave-one-recording-out:
  speed   ALL+IO   LORO +39.3%, trip1 +25.6%   (IO-VNBD is what buys transfer)
  gyroc   ALL      LORO +17.2%, gap +4.0       (first heading head to survive LORO)

Those runs each held a recording back. These do not: every bike recording plus
the IO-VNBD pool goes into training, which is what should ship. The LORO figures
above remain the honest estimate of how they generalise; the point of retraining
on everything is to use the data, not to produce a new score.

trip1 is the only honest test left -- it was excluded from every pool
throughout, so it is a genuinely unseen phone, mount and vehicle.

Both heads are exported to TFLite. The gyro-correction head takes the 92 speed
features plus [dh_gyro, dh_fused, |dh_fused|, yawrate] = 98 inputs, in that
order; the app must build them identically or the mapping is silently wrong.
"""
import config
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from run_matrix_all import (ALLB, HOLD, cached, targets, log, CACHE)
from train_full import names

OUT = config.MODELS / ("v3")


def main():
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    L = tf.keras.layers
    OUT.mkdir(parents=True, exist_ok=True)

    def fit(xt, yt, nout, seed=1, epochs=80):
        tf.keras.utils.set_random_seed(seed)
        m = tf.keras.Sequential([L.Input((xt.shape[1],)),
                                 L.Dense(256, activation="relu"), L.Dropout(.1),
                                 L.Dense(128, activation="relu"), L.Dropout(.1),
                                 L.Dense(64, activation="relu"), L.Dense(nout)])
        m.compile(tf.keras.optimizers.Adam(1e-3), loss="huber")
        m.fit(xt, yt, validation_split=0.1, epochs=epochs, batch_size=2048,
              verbose=0,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=10,
                                                          restore_best_weights=True),
                         tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=.5)])
        return m

    def export(m, mu, sd, cols, sub, extra=None):
        d = OUT / sub
        d.mkdir(parents=True, exist_ok=True)
        m.save(d / "model.keras")
        meta = {"input_cols": cols, "mean": mu.tolist(), "std": sd.tolist(),
                "scales": [2.0, 5.0, 10.0], "fs": 10.0}
        if extra:
            meta.update(extra)
        json.dump(meta, open(d / "norm_stats.json", "w"))
        tfl = tf.lite.TFLiteConverter.from_keras_model(m).convert()
        (d / "model.tflite").write_bytes(tfl)
        # parity: a silent TFLite/Keras divergence would be invisible on device
        it = tf.lite.Interpreter(model_content=tfl); it.allocate_tensors()
        i0 = it.get_input_details()[0]["index"]; o0 = it.get_output_details()[0]["index"]
        probe = np.zeros((1, mu.shape[0]), "float32")
        it.set_tensor(i0, probe); it.invoke()
        k = m.predict(probe, verbose=0)
        log(f"  exported {d}  ({len(tfl)/1024:.0f} KB)  tflite-vs-keras "
            f"{np.abs(k - it.get_tensor(o0)).max():.2e}")

    # ---------------- SPEED: ALL bike + IO-VNBD ----------------
    log("SPEED head — pool ALL + IO-VNBD (everything)")
    have = [n for n in ALLB if (n, "speed") in _store("speed")]
    X = np.concatenate([_store("speed")[(n, "speed")][0] for n in have])
    Y = np.concatenate([_store("speed")[(n, "speed")][1] for n in have])
    z = np.load(CACHE / "__IO__speed.npz")
    Xio, Yio = z["X"], z["Y"]
    Yt, _ = targets(Y, "speed")
    Yio_t, _ = targets(Yio, "speed")
    Xa = np.concatenate([X, Xio]); Ya = np.concatenate([Yt, Yio_t])
    log(f"  {len(have)} bike recordings + IO-VNBD = {len(Xa)} rows")
    mu = Xa.mean(0); sd = Xa.std(0); sd = np.where(sd < 1e-6, 1.0, sd) + 1e-8
    ms = fit((Xa - mu) / sd, Ya, 1)
    export(ms, mu, sd, names(), "speed")

    Xh, Yh, _ = _store("speed")[(HOLD, "speed")]
    ph = ms.predict((Xh - mu) / sd, batch_size=16384, verbose=0)
    h, m_, g = targets(Yh, "speed")[1](ph)
    log(f"  TEST trip1 (in no pool):  hold {h:.3f}  model {m_:.3f} m/s  ({g:+.1f}%)")

    # ---------------- GYRO CORRECTION: ALL bike ----------------
    log("")
    log("GYRO-CORRECTION head — pool ALL")
    st = _store("gyroc")
    have = [n for n in ALLB if (n, "gyroc") in st]
    Xg = np.concatenate([st[(n, "gyroc")][0] for n in have])
    Yg = np.concatenate([st[(n, "gyroc")][1] for n in have])
    Ygt, _ = targets(Yg, "gyroc")
    log(f"  {len(have)} recordings = {len(Xg)} rows, {Xg.shape[1]} inputs")
    mug = Xg.mean(0); sdg = Xg.std(0); sdg = np.where(sdg < 1e-6, 1.0, sdg) + 1e-8
    mg = fit((Xg - mug) / sdg, Ygt, 1)
    export(mg, mug, sdg, names() + ["dh_gyro", "dh_fused", "abs_dh_fused",
                                    "yawrate_now"], "gyroc",
           {"note": "92 speed features, then v0/elapsed are inside them; "
                    "trailing 4 are heading-state inputs"})
    if (HOLD, "gyroc") in st:
        Xh, Yh, _ = st[(HOLD, "gyroc")]
        ph = mg.predict((Xh - mug) / sdg, batch_size=16384, verbose=0)
        h, m_, g = targets(Yh, "gyroc")[1](ph)
        log(f"  TEST trip1 (in no pool):  gyro {h:.2f}  corrected {m_:.2f} deg  ({g:+.1f}%)")
    log("DONE")


_CACHE_STORE = {}


def _store(head):
    """Recording -> cached arrays, loaded once per head."""
    if head not in _CACHE_STORE:
        d = {}
        for nm in sorted(set(ALLB + [HOLD])):
            r = cached(nm, head)
            if r is not None:
                d[(nm, head)] = r
        _CACHE_STORE[head] = d
    return _CACHE_STORE[head]


if __name__ == "__main__":
    main()
