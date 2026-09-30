IDR Navigator v2 — installable APK
==================================
File     : IDRNavigator-v2.apk   (23 MB, debug-signed)
Package  : com.teamua.idr.v2     <- DIFFERENT from v1, so BOTH install side by side
Label    : IDR Navigator v2
ABIs     : arm64-v8a, armeabi-v7a, x86, x86_64 (any phone)

WHAT CHANGED FROM v1
  speed model   bike_v2: trained on every sample of 13 bike recordings
                (709,659 rows). On held-out tails +43.4% over hold-speed
                (corr 0.854) vs v1's +27.9%. On the fully unseen trip1 it is
                the better of the two through the whole pipeline: 60 s drift
                20.9% against 43.8%, full-trip 26.5% against 39.3%.
  compass       hard-iron correction + magnetic spike rejection. Readings whose
                field magnitude departs >15% from the running norm are dropped:
                a passing truck or a steel bridge bends the field and swings
                apparent north by up to 90 deg. Measured 18.5 uT/s during those
                jumps against 0.72 uT/s normally.
  fusion        gyro + compass, tau shrinking with elapsed time (10 s early,
                2 s past 5 min). Gyro is smooth but drifts; the compass does
                not drift. On trip1 the full-trip drift went 74.6% (gyro alone)
                to 26.5% (compass).
  controls      Hdg: GYRO / MAG / FUSED cycle, and a Map on/off toggle.

CONTROLS
  Row 1  Outage | Record | AI speed <-> Hold speed
  Row 2  Hdg: FUSED (tap to cycle) | Map: off

WHAT TO EXPECT (measured, fully on IMU, nothing from GNSS after the anchor)
  Mounted phone, 10 s outage : about 9-12% of distance travelled
  Mounted phone, 30 s        : about 12-15%
  Mounted phone, 60 s        : about 14-16%
  Long single-anchor runs    : the compass matters most here; gyro alone
                               reached 96% drift on one 9.4 km run, fused 9.1%

  Heading modes are within half a point of each other below ~10 s -- the gyro
  has not drifted yet. Test them on LONG outages, that is where they separate.

MAP MATCHING IS OFF BY DEFAULT, deliberately
  Measured gain is about 2 m out of 50 m (~4%). At current drift levels
  (30-65 m after 60 s, against roads 100-200 m apart) the dead-reckoned point
  is often nearer the WRONG road, and snapping then moves you confidently onto
  it. A learned matcher and a topology-routing matcher were both built and both
  failed to beat plain dead reckoning. Matching becomes worth switching on once
  drift is under roughly 20-30 m at 60 s.
  It also refuses to run until the gyro->yaw calibration reaches R2 >= 0.8, and
  the road data covers BHOPAL ONLY.

BEFORE A TEST RIDE
  1. Clamp the phone rigidly. Mount quality dominates everything: 0.46 deg of
     tilt jitter mounted vs 4.15 deg in a pocket, and pocket drift is roughly
     double at 30-60 s.
  2. Keep it away from anything magnetic. Two phones in one pocket measured
     37.7 uT of field noise against 4-5 uT on a clean mount, and the compass
     gate will simply refuse to engage there.
  3. Wait for "compass READY" and "calibration R2 >= 0.8" in the panel. Both
     need a few minutes of riding WITH turns.
  4. Check the mount line says GOOD while MOVING, not while parked.

KNOWN LIMITS
  - Trained on one bike and one rider. On a different vehicle (trip1) it is
    only marginally better than holding the last speed.
  - Turns are still the weak case: roughly double the drift of straight running.
  - Map tiles are fetched online; the basemap greys out in a real tunnel, though
    dead reckoning keeps running.
  - Debug-signed. On Realme/Oppo, if install fails, turn off "Verify apps over
    USB" in Developer options.
