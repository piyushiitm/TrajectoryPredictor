"""Bike-only speed head, for a like-for-like comparison against v3.

v3 was trained on 25 bike recordings PLUS the IO-VNBD car pool, because in the
pool matrix that combination generalised best: ALL+IO scored +25.6% on trip1
against ALL's +13.0%, at no cost to its leave-one-out score.

This head sees the same 25 bike recordings and nothing else. Tested on new_a and
new_b -- both recorded after every pool was fixed, so neither model has seen
them -- the difference between the two is the IO-VNBD contribution, measured
rather than argued.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from run_matrix_all import ALLB, cached, log
from train_full import names

OUT = Path("results/models/v3/speed_bikeonly")


def main():
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    L = tf.keras.layers

    store = {}
    for nm in ALLB:
        r = cached(nm, "speed")
        if r is not None:
            store[nm] = r
    have = list(store)
    X = np.concatenate([store[n][0] for n in have])
    Y = np.concatenate([store[n][1] for n in have])
    log(f"bike-only pool: {len(have)} recordings, {len(X)} rows (no IO-VNBD)")

    mu = X.mean(0); sd = X.std(0); sd = np.where(sd < 1e-6, 1.0, sd) + 1e-8
    tf.keras.utils.set_random_seed(1)
    m = tf.keras.Sequential([L.Input((X.shape[1],)),
                             L.Dense(256, activation="relu"), L.Dropout(.1),
                             L.Dense(128, activation="relu"), L.Dropout(.1),
                             L.Dense(64, activation="relu"), L.Dense(1)])
    m.compile(tf.keras.optimizers.Adam(1e-3), loss="huber")
    m.fit((X - mu) / sd, Y[:, 0] - Y[:, 2], validation_split=0.1, epochs=80,
          batch_size=2048, verbose=0,
          callbacks=[tf.keras.callbacks.EarlyStopping(patience=10,
                                                      restore_best_weights=True),
                     tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=.5)])

    OUT.mkdir(parents=True, exist_ok=True)
    m.save(OUT / "speed_model.keras")
    m.save(OUT / "model.keras")
    json.dump({"input_cols": names(), "mean": mu.tolist(), "std": sd.tolist(),
               "scales": [2.0, 5.0, 10.0], "fs": 10.0},
              open(OUT / "norm_stats.json", "w"))
    tfl = tf.lite.TFLiteConverter.from_keras_model(m).convert()
    (OUT / "model.tflite").write_bytes(tfl)
    log(f"exported {OUT} ({len(tfl)/1024:.0f} KB)")

    # speed MAE on the two fresh recordings, both heads
    st = json.load(open("results/models/v3/speed/norm_stats.json"))
    vmu, vsd = np.array(st["mean"]), np.array(st["std"])
    vnet = tf.keras.models.load_model("results/models/v3/speed/model.keras")
    for nm in ("new_a", "new_b"):
        r = cached(nm, "speed")
        if r is None:
            log(f"  {nm}: no cached features"); continue
        Xa, Ya, _ = r
        v0, spd = Ya[:, 2], Ya[:, 0]
        hold = np.abs(v0 - spd).mean()
        pb = np.clip(v0 + m.predict((Xa - mu) / sd, batch_size=16384,
                                    verbose=0).ravel(), 0, None)
        pv = np.clip(v0 + vnet.predict((Xa - vmu) / vsd, batch_size=16384,
                                       verbose=0).ravel(), 0, None)
        eb, ev = np.abs(pb - spd).mean(), np.abs(pv - spd).mean()
        log(f"  {nm}: hold {hold:.3f} | bike-only {eb:.3f} ({100*(hold-eb)/hold:+.1f}%)"
            f" | v3 bike+IO {ev:.3f} ({100*(hold-ev)/hold:+.1f}%) m/s")
    log("DONE")


if __name__ == "__main__":
    main()
