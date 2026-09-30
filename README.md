# TrajectoryPredictor

Dead reckoning for a GNSS blackout, using nothing but a smartphone's own
sensors. No OBD-II, no vehicle CAN bus, no external hardware at inference time.

Built for SIH 2026 problem statement **SIH26168**, whose benchmark is **under
10% positional drift** during the blackout.

---

## Where it stands

Measured **leave-one-recording-out**: the speed head is retrained without the
recording it is scored on, so nothing is tested on its own training data.

| outage | gyro | compass | **fused** | hold speed |
|--------|------|---------|-----------|------------|
| 10 s   | 13.4% | 15.2% | **13.8%** | 15.8% |
| 30 s   | 19.2% | 17.0% | **16.1%** | 25.2% |
| 60 s   | 21.7% | 17.0% | **16.8%** | 32.2% |

On the two most recent unseen recordings the same pipeline reaches **8.8% and
10.5% at 10 s**, so the benchmark is met on good rides and missed on poor ones.
The spread is dominated by **how the phone is mounted**, not by the models.

Component scores, also leave-one-recording-out:

| head | metric | score |
|------|--------|-------|
| speed (bike + IO-VNBD) | MAE vs hold-speed | **+39.3%** |
| speed, unseen vehicle  | MAE vs hold-speed | **+25.6%** |
| heading correction     | degrees vs gyro   | **+21.3%** |

Figures measured on a held-out *tail* of a recording run 8–51 points higher and
are not used here: a tail shares its ride, route and magnetic surroundings with
the training data. That gap is the reason every number above is LORO.

---

## How it works

```
position(t) = p0 + ∫ v(t) · [sin h(t), cos h(t)] dt
```

One integration. **Acceleration is never integrated.** Measured on this data the
accelerometer's 1 s velocity increment correlates 0.02–0.07 with the truth,
because mount motion fabricates more horizontal acceleration than the vehicle
produces — 2.0–3.4 m/s² against a real signal of 0.4–0.6 m/s². Integrating it
diverges to 140–400% path error.

**Speed** is regressed directly from window statistics of the IMU. Vibration
amplitude scales with speed, so the vibration *is* the signal — low-pass
filtering the accelerometer doubled the drift.

**Heading** comes from a gyro/compass complementary filter, not from a model.
The gyro is smooth but drifts; the compass is noisy but does not. τ shrinks with
elapsed time, so long blackouts lean on the compass. The compass gets online
hard-iron correction and rejects readings whose field magnitude departs >15%
from its running norm — a passing truck or a steel bridge swings apparent north
by up to 90°, and those disturbances announce themselves as an 18.5 µT/s spike
against 0.72 µT/s normally.

A single learned **heading correction** sits on top of the gyro (+21.3% LORO).

---

## Layout

```
src/        engine and training scripts
models/     exported heads, .tflite + normalisation
app/        Android app (Kotlin) and built APKs
server/     replay service: upload a recording, watch it drive the engine
data/       recordings (gitignored, see MANIFEST.csv)
results/    logs behind the numbers above
```

### Training entry points

| script | what it produces |
|--------|------------------|
| `train_speed_general.py`      | speed head, bike + IO-VNBD cars — generalises |
| `train_speed_tuned.py`        | speed head, bike only — sharper on that bike |
| `train_heading_correction.py` | one correction head per baseline, both validated |
| `evaluate_drift_loro.py`      | the benchmark table at the top of this file |
| `run_matrix_all.py`           | 5 pools × 4 heads × {LORO, tails} |
| `analyse_velocity_vector.py`  | splits the velocity head into along-track vs lateral |

### Running it

```bash
python -m venv venv && ./venv/bin/pip install -r requirements.txt
./venv/bin/python server/server.py       # then open http://127.0.0.1:8000
```

The replay service takes a recording, runs the real engine over it, and animates
every heading source against GNSS truth with live drift readouts. Best/worst
window buttons rank every outage window in the file, so the good cases cannot be
shown without the bad ones being one click away.

---

## Data

28 recordings, ~1.1 GB, **not in the repository** — `MANIFEST.csv` lists them.

Sessions that were captured simultaneously share a suffix: `s07_mount`,
`s07_pocket` and `s07_hand` are one ride recorded on three phones at once, which
is what makes mount placement separable from route and traffic.

| prefix | meaning |
|--------|---------|
| `sNN_*` | training sessions |
| `tNN_*` | held out from every pool — the honest tests |

`t03_other` is a different phone, mount and vehicle; it is the hardest case and
the one worth quoting.

---

## What was tried and did not work

Roughly 60 experiments. The failures are load-bearing — several of them are why
the shipped design is shaped the way it is.

- **Velocity-vector regression, eleven formulations.** The along-track component
  works (corr 0.74); the lateral one never has — corr **+0.077** across a 20×
  range of data volume, from 386k bike samples to 614k IO-VNBD samples with
  dense CAN truth. The turn signal is not in the IMU, and more data does not
  conjure it.
- **Map matching**, HMM, HMM with road topology, and a learned matcher. At 10 s
  the dead reckoning is already more accurate (3.2 m) than the road network's
  8 m granularity, so snapping *adds* error; at 60 s the 63 m error cannot
  identify the right road among streets 100–200 m apart. No window in between.
- **A learned compass filter.** +47.3% on held-out tails, **−3.6%** leave-one-out.
  It had memorised where particular bridges and parked vehicles were.
- **Accelerometer denoising and integration**, four ways. The signal sits 4–6×
  below the mount-motion noise floor, and integrating makes it worse, not better.
- **Anti-alias filtering** of the accelerometer: drift doubled, because the
  vibration being filtered out is the speed cue.
