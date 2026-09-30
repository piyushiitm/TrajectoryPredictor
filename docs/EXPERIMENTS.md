# IDR Engine — Experiment Log

SIH 2026, PS **SIH26168** — AI-ML Intelligent Dead Reckoning for GNSS-denied
navigation. Every number here was measured in this repository and is
reproducible with the command given.

---

## 1. Executive summary

**What works.** A dead-reckoning engine that integrates *once*:

    position(t) = p₀ + ∫ v(t)·[sin h(t), cos h(t)] dt

with **speed from an ML model** (never integrated) and **heading from a
calibrated gyro** (one integration, of angular rate). On IO-VNBD held-out
vehicles this gives **15.1–16.9% positional drift**, flat from 10 s to 300 s,
with no GNSS during the outage.

**What does not work, and why.** Integrating the accelerometer. Measured on
this data, the accelerometer's 1-second velocity increment correlates
**0.02–0.07** with the true increment, because 2–5°/s of mount tilt wobble
fabricates **0.34–0.86 m/s²** of phantom horizontal acceleration against a real
signal of 0.09–0.48 m/s². Integrated over a path this diverges to
**140–400% error**. No model, filter or framework recovers it.

**The binding constraint.** Not architecture — *data coverage*. Four model
families, a deep sequence model, 2× data and extra feature scales all failed to
improve held-out performance; re-weighting the training distribution produced a
~100× error reduction. There is roughly **2.3 hours of public driving data in
the 5–12 m/s band** that the Indian use case needs.

---

## 2. Measurement infrastructure

Nothing below is meaningful without knowing the floor — how much the ground
truth disagrees with itself.

| truth source | self-consistency floor | usable for <10% claim? |
|---|---|---|
| IO-VNBD V-file (CAN + survey GPS, 10 Hz) | **0.2–0.8%** | yes |
| trip3 own GPS (1 Hz) | 1.2–3.2% | yes |
| trip1 own GPS | 5.8–9.1% | marginal |
| trip2 own GPS | 11.3–18.3% | **no** below 120 s |
| IO-VNBD phone GPS (9 s updates) | 25.9–38.0% | no |

Consequence: **the <10% benchmark cannot be demonstrated on trip2 below 120 s** —
the yardstick is wobblier than the thing being measured. All headline results
are quoted against V-file truth.

---

## 3. Data discoveries (each one changed downstream results)

### 3.1 IO-VNBD speed units are m/s, not km/h
The column labelled `GPS SPEED (Kmh)` stores **m/s**. Verified two ways: ratio
3.607 against the V-file CAN speed, and 0.999 against speed derived from
consecutive GPS positions. **68 of 70 measurable files are m/s; none are
genuinely km/h.**

The original displacement-based detector needed ≥20 genuine GPS fixes and
silently gave up on **39 of 72 files** — leaving raw m/s in a column everything
downstream read as km/h, making those anchors 3.6× too small. Replaced with a
V-file-based check that resolves 68/72. *(This bug class recurred three times in
different places; see §8.)*

### 3.2 The gyro channel names are wrong
IO-VNBD's `GYROSCOPE Yaw` correlates **0.006** with the CAN `Yaw Rate`. The
channel labelled `GYROSCOPE Pitch` correlates **0.95**. The gyro triad is
permuted relative to the accelerometer/gravity triad.

Worse, `ω·ĝ` — the textbook projection that *should* be label-agnostic — also
fails, because **IO-VNBD's GRAVITY channel is (0, 0, 9.8065)** with std
0.02–0.04: every recording had the phone lying flat, so the vertical axis *is*
device-z, and the projection returns exactly the wrong channel.

Fix: `yaw_axis_from_gyro()` finds the yaw axis as the **principal axis of
angular velocity** (a driving vehicle yaws far more than it pitches or rolls) —
no labels, no gravity channel, no ground truth. Per-session |corr| went
**0.118 → 0.424**, with 3 sessions above 0.9.

### 3.3 GPS updates every 9.00 seconds
Measured across 49 files: **46 files at exactly 9.00 s**, 3 at 1 Hz. This
explains the unit-detector failures, why GPS-truth speed data covered only 40 of
66 sessions, and why the V-files are a necessity rather than a convenience.

### 3.4 The V-files carry channels we were re-deriving badly
`Yaw Rate (deg/sec)`, `Indicated Longitudinal/Lateral Acceleration (g)`,
`Steering Angle`, and four `Wheel Speed` channels. We had been differencing
`Heading` and GPS velocity to reconstruct quantities that were present as direct
measurements.

### 3.5 Conventions verified (no bug, but load-bearing)
- GPS bearing and V-file `Heading` are both **clockwise from north** (1.4–8.1°
  against position-derived bearing; the mathematical convention is off by ~100°).
- Speed decomposition `E = v·sin(θ)`, `N = v·cos(θ)` verified against
  displacement: 1.7–15.7% error, versus 135–148% for the swapped assignment.
- `roll_pitch_from_gravity` ↔ `rotation_matrix` are mutually consistent:
  rotating gravity to world gives horizontal leak of **0.0000 m/s²**.

---

## 4. Why acceleration is never integrated

This is the single most consequential finding, established five independent ways.

| test | result |
|---|---|
| corr(accel 1 s velocity increment, true increment) | **0.02–0.07** across 4 trips, all intervals 1–30 s |
| with *perfect* GPS heading in the rotation matrix | unchanged — `trust-IMU` 3.543 vs `hold-anchor` 2.629 m/s |
| tilt wobble (change in gravity direction per second) | **2–5°**, fabricating **0.34–0.86 m/s²** |
| true vehicle \|dv/dt\| for comparison | **0.09–0.48 m/s²** — SNR below 1 |
| path error from integrating | **140–400%** over a full trip |

The error is **not a bias** (mean 0.01 m/s² against noise std 1.0 — removing it
perfectly changes MAE by 0.001) and **not high-frequency** (low-pass at 5/2/1/0.5
Hz changes correlation by <0.01, and anti-aliasing at the raw 91 Hz rate before
decimation changes nothing). It is broadband noise from the mount, overlapping
the signal band.

**A model trained to predict the residual behaves correctly**: it learns to output
only 12–16% of the true velocity change, which is the right response to an input
whose correlation with the target is zero.

---

## 5. Heading

### 5.1 What was tried
| method | corr | gain (1.0 = correct) |
|---|---|---|
| integrated gyro, Euler formula | 0.118 median | 0.013 |
| device-fused `ORIENTATION (Yaw)` | 0.161 | 0.172 |
| tilt-compensated magnetometer | ~10° median absolute, drift-free | — |
| PCA-aligned accelerometer (NHC, `a_lat = v·ω`) | 0.082 | 0.062 |
| **PCA yaw-axis gyro** | **0.424** | **0.344** |
| **least-squares calibration vs GPS bearing** | **R² 0.988** | — |

### 5.2 The calibration window is the dominant knob
| window | fit R² | heading error @60 s | drift with true speed |
|---|---|---|---|
| 60 s | 0.75 | 25.3° | 25.2% |
| **300 s** | **0.988** | **9.7°** | **8.1%** |
| 600 s | 0.989 | 11.3° | 8.9% |

At 9 s GPS updates a 60 s window yields only ~6 fit points. 300 s is the sweet
spot. This one change took heading-limited drift from 25.2% to 8.1%.

### 5.3 Heading cannot be improved by a model
- Predicting the calibration **residual**: R² **−0.006** on held-out sessions
  (worse than the mean), −0.3% change. It is noise.
- Predicting **angular displacement** directly: 9.51°/32.76°/40.82° for
  integration vs 10.23°/23.19°/43.09° for a GBM at 10/30/60 s — one win in three,
  inside noise.

**Reason:** `θ = ∫ω dt` is *exact*, not an approximation. Once the linear fit
recovers axis order, scale, bias and sign, nothing remains but noise. Use exact
mathematics where the physics is known; use learning only where it isn't.

### 5.4 Compass fusion made it worse
Complementary-filtering the drift-free compass into the gyro heading: 8.5° →
30.7° at 30 s. The magnetometer is too disturbed inside a vehicle.

---

## 6. Speed — the part that works

### 6.1 Formulation history
| formulation | held-out-vehicle MAE | vs hold-speed |
|---|---|---|
| `imu_vx, imu_vy` only (the literal 2→32→32→32→2 spec) | 3.089 | −80% |
| + anchor and elapsed, no `yaw_rel` | 2.022 | −18% |
| absolute target, window statistics | 4.65 | worse |
| **residual target, anchored, multi-scale (2/5/10 s)** | **2.688** | **+21%** |

Two changes did the work: predicting the **residual** from the known anchor
speed rather than the absolute value, and **multi-scale** windows. The advantage
grows with outage length — 26% better at 60 s, 29% at 120 s — which is exactly
where it is needed.

### 6.2 Architecture and framework sweep — all negative
Same 92 features, same residual target, same held-out vehicles:

| learner | MAE | corr |
|---|---|---|
| **Keras MLP (incumbent)** | **2.697** | **0.903** |
| Ensemble (top 3) | 2.783 | 0.902 |
| CatBoost | 2.872 | 0.900 |
| LightGBM | 2.933 | 0.893 |
| XGBoost | 2.934 | 0.894 |
| hold-speed baseline | 3.395 | — |
| sklearn HistGBR (tuned) | 3.416 | 0.871 |
| **Deep 1D-CNN (dilated TCN) on raw IMU** | **3.421** | 0.878 |

The deep sequence model was tested at matched data volume (98.6k windows) after
a first run at 22k; scaling 10× moved it 0.14, so it was not starved. The
hand-crafted features already encode mount-rotation invariance; the CNN must
learn it from ~45 sessions and overfits instead.

Also negative: **2× data** (2.742 vs 2.688), a **4th feature scale** at 20 s
(2.830), **per-session recalibration** (16.4% → 19.9% drift), **blending toward
the anchor** (15.2% → 17.1%), and **scale-bias correction** (bias is 1%).

**Spread across frameworks is ~8%; the gap to baseline is 21%.** The features do
the work; the learner barely matters.

---

## 7. Data is the real lever

### 7.1 Datasets integrated
| dataset | country | usable | mean speed | truth | loader |
|---|---|---|---|---|---|
| IO-VNBD | Nigeria | ~20 h | 13.9 m/s | CAN V-file | `idr_core.load_session` |
| comma2k19 Chunk_1 | USA | ~2 h | 22.6 | CAN | `comma_loader.py` |
| STRIDE | Bangladesh | 53 min | 8.3 | GPS | `stride_loader.py` |
| Pune | India | 24 min | 3.9 | GPS | `pune_loader.py` |
| trip1–4 | India | ~60 min | 1–10 | GPS | `load_session_gps` |

**Rejected after inspection:** RoadSens-4M (Bangladesh — no speed channel;
positions interpolated to 100 Hz; sessions 18–27 s), Mendeley 9vr83n7z5j
(Indian, accelerometer+gyro only, no GPS), Nigerian alcohol-driving set (no
public repository link), Thailand two-wheeler set (headerless CSVs, schema
undocumented).

### 7.2 Adding a platform helps; adding the wrong speed range hurts
Controlled test, identical held-out IO-VNBD vehicles:

| training | MAE | corr |
|---|---|---|
| IO-VNBD only | 2.865 | 0.892 |
| **+ comma2k19** | **2.766** | **0.897** |

First genuine generalisation gain of the project — from a single extra dongle.
But on the *slow* trips the same addition is harmful: the model's mean
prediction tracks its **training** mean, not the input.

| model | training mean | trip2 pred / true | Pune pred / true |
|---|---|---|---|
| IO-VNBD | 13.0 | 21.6 / 8.0 | 11.3 / 3.9 |
| + comma2k19 | 15.6 | 21.5 / 8.0 | 18.7 / 3.9 |

### 7.3 Oversampling the in-range data is decisive
Six training pools, **3 seeds each, 54 runs** (seed std is 0.11–0.15 MAE, so
differences under ~5% are noise):

| pool | mean speed | trip1 | trip2 | trip3 | mean |
|---|---|---|---|---|---|
| IO-VNBD only | 13.9 | +3.5% | **−368%** | −48% | −138% |
| + comma2k19 | 15.7 | **+10.8%** | −222% | −50% | −87% |
| + STRIDE + Pune | 13.2 | −11.1% | −21% | +11.6% | −6.9% |
| **+ South Asia ×6** | **10.9** | −8.6% | −2.5% | +10.4% | **−0.2%** |
| + comma, SA ×6 | 12.4 | −15.8% | −14.5% | **+17.1%** | −4.4% |
| South Asia only (21k) | 6.9 | −77% | −5.1% | −2.5% | −28% |

**IO-VNBD alone scores −368% on trip2; the same data with South Asian sessions
repeated 6× scores −2.5%.** A ~100× error reduction from re-weighting alone —
larger than every architectural change combined.

**Caution:** single-seed runs of this sweep showed apparent wins (+19.7% on
trip2) that did not survive 3 seeds (−5.1%). Always average seeds here.

---

## 8. Bugs found and fixed (do not reintroduce)

1. **m/s vs km/h, three times** — the S-file speed column (§3.1); the anchor
   speed in `idr_navigate` (inflated drift to 48%); and a units slip in a
   diagnostic. Detected each time by a cross-check disagreeing, never by
   inspection.
2. **Yaw sign convention** — `rotation_matrix` expects mathematical yaw
   (X=east, CCW) while GPS bearing is a compass angle (CW from north). The
   pipeline seeded `yaw` with the raw bearing and then added CCW increments, so
   heading integrated backwards. Symptom: a *stable negative* correlation, which
   is what exposed it. Fix: `yaw = π/2 − bearing`.
3. **`resample()` blanked `orient_yaw_deg`** whenever a GPS fix was stale,
   destroying the fused-heading channel entirely.
4. **Zero-variance features** — `yaw_rel` was identically 0 under some
   configurations; its training std of 1e-8 made normalisation multiply by 1e8
   and produce 10⁹% errors. Guard added to drop such columns with a warning.
5. **`tensorflow-metal`** produces diverging gradients on this pipeline. It is
   deliberately **not installed**; every trainer also calls
   `set_visible_devices([], 'GPU')`.
6. **The Uncategorised IO-VNBD branch duplicates the Categorised one** — same 72
   recordings flattened. Using both puts one drive in train *and* test.

---

## 9. Honest limitations

- **The <10% benchmark is not met.** Best is 15.1–16.9% on IO-VNBD held-out
  vehicles; the PS asks for <10%.
- **The user's own recordings do not work** — 35–92% drift. The cause is
  mounting, not the model: trip3's gyro calibration fits at **R² 0.599** against
  IO-VNBD's 0.988, with 5.02°/s tilt wobble and 5.0 m/s² vibration against
  IO-VNBD's 1–2.
- **No training pool beats hold-speed on all three trips.** trip1 (10.4 m/s) and
  trip2/trip3 (~6 m/s) cannot be served by one training distribution, because the
  model is steering a *prior* rather than inferring speed from the IMU.
- **Map matching is unbuilt.** The PS names it explicitly, and it is the
  strongest remaining lever — it constrains cross-track error, roughly half the
  remaining budget, and needs no new data.
- **The mobile application is unbuilt.** Models export to TFLite; the Android app
  does not exist.

---

## 10. Reproducing

```bash
source venv/bin/activate
IOV="data/raw/iovnbd_data/Synchronised V abd S datasets/Categorised IOVNB Dataset"

# features and training
python src/preprocess_speed2.py --input "$IOV" --output data/processed/speed2.csv --hop 1.0
python src/train_speed2.py --input data/processed/pool_IO_SAx6.csv --outdir models/speed_best

# full pipeline drift, any dataset
python src/idr_navigate.py --input "$IOV" --model models/speed_best \
    --dataset iovnbd --cal-s 300 --durations 10,30,60,120,300

# diagnostics
python src/check_heading.py --input "$IOV"        # heading observability
python src/idr_budget.py    --input "$IOV" --model models/speed_best   # error budget
python src/speed_bench.py   --input data/processed/speed2.csv          # learner sweep
```
