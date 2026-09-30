# IDR Engine — Experiment Log

SIH 2026, PS **SIH26168** — AI-ML Intelligent Dead Reckoning for GNSS-denied
navigation. Every number here was measured in this repository and is
reproducible with the command given.

> **How to read this log.** Sections 1–10 are **phase 1**, run on IO-VNBD before
> the team's own recordings existed. Sections 11–14 are **phase 2**, on the 28
> recordings in `data/MANIFEST.csv`, and they overturn several phase-1
> conclusions — heading *can* be improved by a model (§11.3), the compass helps
> on long outages (§11.4), and map matching and the Android app now exist.
> Where the two disagree, phase 2 is current; phase-1 text is kept because the
> reasons behind each reversal are part of the record. Superseded statements
> are marked **[superseded → §n]**.

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

### 5.3 Heading cannot be improved by a model **[superseded → §11.3]**
- Predicting the calibration **residual**: R² **−0.006** on held-out sessions
  (worse than the mean), −0.3% change. It is noise.
- Predicting **angular displacement** directly: 9.51°/32.76°/40.82° for
  integration vs 10.23°/23.19°/43.09° for a GBM at 10/30/60 s — one win in three,
  inside noise.

**Reason:** `θ = ∫ω dt` is *exact*, not an approximation. Once the linear fit
recovers axis order, scale, bias and sign, nothing remains but noise. Use exact
mathematics where the physics is known; use learning only where it isn't.

### 5.4 Compass fusion made it worse **[superseded → §11.4]**
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

Phase-2 bugs (§13) continue this list: 7–12.

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
- **[superseded → §11.6]** **Map matching is unbuilt.** The PS names it explicitly, and it is the
  strongest remaining lever — it constrains cross-track error, roughly half the
  remaining budget, and needs no new data.
- **[superseded → §11.7]** **The mobile application is unbuilt.** Models export to TFLite; the Android app
  does not exist.

---

## 10. Reproducing (phase 1)

These scripts belong to the phase-1 layout and are no longer in the tree; the
current commands are in §14.

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

---

## 11. Phase 2 — the team's own recordings

28 recordings (`data/MANIFEST.csv`, ~1.1 GB, not in git): 25 training sessions
captured on up to three phones at once (`sNN_mount`, `sNN_pocket`, `sNN_hand`),
so placement is separable from route and traffic, plus 3 held out of every pool
(`t01_mount`, `t02_mount`, and `t03_other` — a different phone, mount and
vehicle). **Every score below is leave-one-recording-out (LORO)**: the model is
retrained without the recording it is scored on.

### 11.1 Positional drift, full pipeline — `results/loro_drift.log`

Median over 9 held-out recordings; model speed, gyro/compass fusion, real
outage windows, no GNSS after the anchor.

| outage | gyro | compass | **fused** | hold speed | recordings < 10% |
|--------|------|---------|-----------|------------|------------------|
| 10 s   | 13.4% | 15.2% | **13.8%** | 15.8% | 2 / 9 |
| 30 s   | 19.2% | 17.0% | **16.1%** | 25.2% | 1 / 9 |
| 60 s   | 21.7% | 17.0% | **16.8%** | 32.2% | 0 / 9 |

Per recording at 10 s the spread is 9.1% (`mount_d`) to 20.4% (`pocket_b`); at
60 s one mounted ride reaches 38.1% (`mount_i`). **The spread is set by how the
phone is held, not by the model** — the app README measures 1.4°/s of mount
wobble giving 8.3% drift against 5.7°/s giving 39.3%. The <10% target is met on
good rides and missed on poor ones.

### 11.2 Head × pool matrix — `results/matrix_all.log`

Four heads, five training pools, each scored three ways: LORO (the honest
number), held-out tails, and trip1 (in no pool). Gain is against the matching
baseline (hold-speed for speed heads, the raw source for heading heads);
`gap` = tails − LORO, i.e. how much a model memorised its rides.

| head | NEW (12) | MOUNT (9) | HAND (4) | ALL (25) | ALL+IO |
|------|---------|-----------|----------|----------|--------|
| speed          | +40.7 | +39.9 | +38.7 | +34.9 | **+39.3** (gap 9.2) |
| velocity vector| +14.8 | +18.1 | +13.8 | +15.7 | +17.1 |
| gyro correction| +21.9 | **−12.3** | +11.5 | **+17.2** (gap 4.0) | — |
| compass filter | −48.5 (gap 51.0) | −13.3 | −8.4 | −6.2 | — |

- The **speed** head holds up everywhere; adding IO-VNBD (ALL+IO) is what buys
  transfer — trip1 +25.6% against +13.0% for ALL. Ships as `speed_general`.
- The **gyro correction** fails on the MOUNT-only pool and passes on mixed
  pools: placement diversity, not volume, is what it needs. Ships (ALL).
- The **compass filter** looks positive on tails (+2.6 to +12.0) and is negative
  on every LORO — gaps up to 51 points. It had memorised where particular
  bridges and parked vehicles distort the field. **Not shipped.** This one table
  is why every headline number in the project is LORO.

### 11.3 Heading *can* be improved by a model (reverses §5.3)

§5.3 predicted the *calibration residual* and got R² −0.006. The fix was the
target: predict the **total heading error accumulated since the anchor**,
`true − dh_gyro`, from 96 inputs (the 90 window statistics, v0, elapsed,
dh_gyro, dh_fused, |dh_fused|, current yaw rate), and apply it once as an
offset on the gyro heading.

- LORO +17.2% (ALL pool, gap 4.0); the shipped retrain reports +21.3% against
  the raw gyro (README).
- trip1, never trained on: 24.73° → 20.75° (+16.1%) — `results/v3_train.log`.
- The same head trained against the **fused** residual scored −7.1% LORO: once
  the compass has removed what is learnable, the remainder is noise. Only the
  gyro variant ships.
- A residual head on the fused estimate (`results/head_correct.log`) also lost:
  −3.6% on tails overall, best on only 1 of 14 held-out tails (and on trip1
  by 0.6%).

### 11.4 Compass fusion: loses short, wins long (refines §5.4)

§5.4 saw the compass hurt at 30 s. With hard-iron correction, a 15%
field-magnitude gate (spikes read 18.5 µT/s against 0.72 µT/s normally) and a
fusion time constant τ that shrinks from 10 s to 2 s as the outage runs on:

- Under ~10 s the gyro has not drifted yet and the compass only adds noise —
  gyro wins at 10 s in §11.1 (13.4% vs 13.8% fused).
- From 30 s it wins: 16.1% fused vs 19.2% gyro at 30 s, 16.8% vs 21.7% at 60 s.
- On long single-anchor runs it is decisive: one 9.4 km run went 96% (gyro) →
  9.1% (fused) (app README); a 6.2 km full trip in `results/compass_pipeline.log`
  went 76.1% → 38.7%.
- A learned compass filter on top (`mag_f`, `fused_f` in that log) never beat
  the plain fusion — consistent with §11.2.

### 11.5 Velocity-vector head: along-track works, lateral never does

Eleven formulations predicted the full 2-D velocity (forward + sideways)
instead of speed. It is positive in every pool (+13.8% to +18.1%, §11.2), but
`analyse_velocity_vector.py` splits it: the **along-track** component correlates
0.74 with truth; the **lateral** one +0.077, flat across a 20× range of data
volume (386k bike samples to 614k IO-VNBD samples with dense CAN truth). More
data does not help — the turn is not observable from the IMU window. It stays an
ongoing experiment; heading comes from gyro + compass + the correction head.

### 11.6 Map matching: built, and off by default

HMM (Newson & Krumm), HMM with road topology, and a learned matcher were all
built. None beat plain dead reckoning: at 10 s the DR error (3.2 m) is already
finer than the road network's ~8 m granularity, so snapping *adds* error
(3.2 → 5.7 m); at 60 s (~63 m) the point is often nearer the wrong road among
streets 100–200 m apart (63.2 m vs 62.7 m). Measured gain ~2 m in 50 m. The
matcher ships trust-gated and off: it refuses to run until calibration R² ≥ 0.8,
and is worth enabling once 60 s drift is under ~20–30 m.

### 11.7 The Android app exists

`app/` — IDR Navigator: both speed heads (tuned / general) and the heading head
on-device as TFLite (<300 KB each; export parity 1.8×10⁻⁷ speed, 3.7×10⁻⁹
heading), Outage/Restore to simulate a blackout while GNSS keeps arriving for
comparison, GYRO / MAG / FUSED / ML_GYRO heading modes, a live mount-quality
check, an offline road network (231,692 points) and in-app trip recording.

---

## 12. Honest limitations (phase 2)

- **<10% is met on good rides only** — 2 of 9 held-out recordings at 10 s; the
  median is 13.8%. Mounting dominates (§11.1).
- **Turns are the weak point.** Turns drift ~2× straight running, and the
  lateral velocity component is unobservable (§11.5).
- **The gyro correction needs placement-diverse training** — it fails on a
  mount-only pool (§11.2).
- **Map matching gives nothing yet** at current drift levels (§11.6).
- **Thin in-range data remains** — the phase-1 constraint (§7) still holds for
  public data; own recordings are the lever.

---

## 13. Bugs found and fixed in phase 2 (do not reintroduce)

7. **App: ML heading correction applied cumulatively.** The head predicts the
   *total* error since the anchor; `IdrEngine` added it into `drHeading` every
   0.5 s, re-applying the whole correction twice a second. In `Hdg: ML_GYRO` the
   heading spun (2,104° in a 60 s simulation) and every dead-reckoned marker
   circled on the spot while AI speed kept updating. Fix: keep the integrated
   heading separate and apply the latest correction once, clipped to ±0.5 rad —
   exactly as the replay server does.
8. **App: `YawCalibrator.solve` read the wrong column.** After Gauss–Jordan it
   returned `m[i][i+1] / m[i][i]` instead of the augmented column
   `m[i][n] / m[i][i]`, so all three gyro weights came out 0 and only the bias
   was used. Simulated recovery went from R² −0.001 to 1.000.
9. **Replay server pointed at dead paths** — `results/models/v3/*` `.keras`
   checkpoints that are gitignored, and `<root>/src` instead of `server/src`.
   It now loads the shipped `.tflite` files under `models/`.
10. **`idr_core.py` imported itself through a package path**
    (`TrajectoryPredictor.server.src…`) that only exists on one laptop.
11. **`config.ROOT` was one level too shallow** after `config.py` moved into
    `server/src/`.
12. **Drive fetch broke on gdown 6**, which removed `remaining_ok`; the call
    raised, was caught, and deploys came up with no recordings. The fetch now
    logs every step to `data/fetch_log.txt`, shown at `/api/status`.

---

## 14. Reproducing (current)

Run from the repository root. Recordings go in `data/recordings/`
(`python server/fetch_data.py` pulls them from the shared Drive folder).

```bash
python -m venv venv && ./venv/bin/pip install -r requirements.txt

# training matrix: 5 pools x 4 heads, LORO + tails + trip1  -> results/matrix_all.log
./venv/bin/python server/src/run_matrix_all.py
# full-pipeline positional drift, LORO                     -> results/loro_drift.log
./venv/bin/python server/src/evaluate_drift_loro.py
# along-track vs lateral split of the velocity-vector head
./venv/bin/python server/src/analyse_velocity_vector.py
# shipped models
./venv/bin/python server/src/train_speed_general.py
./venv/bin/python server/src/train_speed_tuned.py
./venv/bin/python server/src/train_heading_correction.py

# replay service (only needs a TFLite runtime)
./venv/bin/pip install -r server/requirements-render.txt
./venv/bin/python server/server.py          # http://127.0.0.1:8000
```
