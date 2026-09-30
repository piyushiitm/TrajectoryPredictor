"""Where does the velocity-vector model's gain actually come from?

The matrix reported velvec at +13.8% to +18.1% leave-one-out, positive in every
pool, which looks like the head finally works. But the metric is the combined
error |(vx, vy) - truth|, and vy (along-track) is just speed wearing different
clothes -- the speed head already does that at +39.3%.

The question that decides whether MORE DATA would help is narrower: has the
LATERAL component vx started to carry signal? vx is where turn information
lives, and it has sat at corr ~0 through every earlier attempt, including
614k samples of IO-VNBD with dense CAN truth.

So this reports the two components separately, leave-one-recording-out, and
also scores vx against the only baseline that matters for it: predicting zero.
If corr(vx) is still ~0, the lateral signal is absent and more of the same
recordings will not conjure it.
"""
import config
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from run_matrix_all import ALLB, HOLD, cached, log

MAX_FOLDS = 8


def main():
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    L = tf.keras.layers

    store = {}
    for nm in sorted(set(ALLB + [HOLD])):
        r = cached(nm, "velvec")
        if r is not None:
            store[nm] = r
    have = [n for n in ALLB if n in store]
    X = np.concatenate([store[n][0] for n in have])
    Y = np.concatenate([store[n][1] for n in have])
    S = np.concatenate([np.full(len(store[n][0]), n) for n in have])
    log(f"{len(have)} recordings, {len(X)} rows")

    v0, spd, dh = Y[:, 2], Y[:, 0], Y[:, 1]
    vx = spd * np.sin(dh)          # lateral: where turn information lives
    vy = spd * np.cos(dh)          # along-track: essentially speed
    T = np.column_stack([vx, vy - v0])

    def fit(xt, yt, seed=1):
        tf.keras.utils.set_random_seed(seed)
        m = tf.keras.Sequential([L.Input((xt.shape[1],)),
                                 L.Dense(256, activation="relu"), L.Dropout(.1),
                                 L.Dense(128, activation="relu"), L.Dropout(.1),
                                 L.Dense(64, activation="relu"), L.Dense(2)])
        m.compile(tf.keras.optimizers.Adam(1e-3), loss="huber")
        m.fit(xt, yt, validation_split=0.1, epochs=60, batch_size=2048, verbose=0,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=8,
                                                          restore_best_weights=True),
                         tf.keras.callbacks.ReduceLROnPlateau(patience=4, factor=.5)])
        return m

    folds = have if len(have) <= MAX_FOLDS else \
        list(np.array(have)[np.linspace(0, len(have) - 1, MAX_FOLDS).astype(int)])
    log("")
    log(f"{'held out':10s} {'corr vx':>8s} {'corr vy':>8s} {'vx MAE':>8s} "
        f"{'vx zero':>8s} {'vx gain':>8s} {'vy gain':>8s}")
    log("-" * 62)
    cx, cy, gx, gy = [], [], [], []
    for held in folds:
        tr = S != held
        te = S == held
        if te.sum() < 50:
            continue
        mu = X[tr].mean(0); sd = X[tr].std(0)
        sd = np.where(sd < 1e-6, 1.0, sd) + 1e-8
        m = fit((X[tr] - mu) / sd, T[tr])
        p = m.predict((X[te] - mu) / sd, batch_size=16384, verbose=0)
        px, py = p[:, 0], v0[te] + p[:, 1]
        c1 = np.corrcoef(px, vx[te])[0, 1] if np.std(px) > 1e-9 else np.nan
        c2 = np.corrcoef(py, vy[te])[0, 1] if np.std(py) > 1e-9 else np.nan
        # vx against predicting zero -- the only honest baseline for a lateral term
        z = np.abs(vx[te]).mean()
        e = np.abs(vx[te] - px).mean()
        # vy against holding the anchor speed
        zh = np.abs(vy[te] - v0[te]).mean()
        eh = np.abs(vy[te] - py).mean()
        cx.append(c1); cy.append(c2)
        gx.append(100 * (z - e) / z); gy.append(100 * (zh - eh) / zh)
        log(f"{held:10s} {c1:+8.3f} {c2:+8.3f} {e:8.3f} {z:8.3f} "
            f"{gx[-1]:+7.1f}% {gy[-1]:+7.1f}%")
    log("-" * 62)
    log(f"{'MEAN':10s} {np.nanmean(cx):+8.3f} {np.nanmean(cy):+8.3f} "
        f"{'':8s} {'':8s} {np.mean(gx):+7.1f}% {np.mean(gy):+7.1f}%")
    log("")
    log("corr vx near 0 means the lateral (turn) component carries no signal, and")
    log("the head's headline gain is its along-track half -- i.e. the speed model.")
    log("DONE")


if __name__ == "__main__":
    main()
