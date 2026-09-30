IDR Navigator — installable APK
================================
File        : IDRNavigator-v1.apk   (24 MB, debug-signed)
Package     : com.teamua.idr
Label       : IDR Navigator
Min/target  : Android 8.0+ / compileSdk 36
ABIs        : arm64-v8a, armeabi-v7a, x86, x86_64  (universal — any phone)
Bundled     : assets/speed_model.tflite (89 KB, unquantized)
              trained on bike trip11 (oversampled x22) + IO-VNBD pool
              assets/roads.bin (2.7 MB) -- 231,692 road points over Bhopal,
              bbox 23.17-23.24 N, 77.38-77.47 E, offline, no network needed

INSTALL — on the phone
  1. Copy the .apk to the phone (USB, Drive, WhatsApp to self, etc.)
  2. Open it in Files; allow "install unknown apps" for that app when asked.
  3. On Realme/Oppo/ColorOS, if it fails with a verification error, turn OFF
     Settings > Additional Settings > Developer options > "Verify apps over USB".

INSTALL — over adb
  adb install -r IDRNavigator-v1.apk
  If INSTALL_FAILED_VERIFICATION_FAILURE:
  adb shell settings put global verifier_verify_adb_installs 0

PERMISSIONS — grant at first launch
  Location: choose "While using the app" AND set accuracy to Precise.
  ColorOS blocks `adb pm grant`, so this must be done on-screen.

USING IT
  Markers are concentric: green = GNSS, red = IMU dead reckoning,
  blue = hold-speed. They sit on top of each other while GNSS is healthy.
  Purple = map-matched, and appears ONLY during an outage and ONLY when the
  match passes its trust gate. Purple missing is normal and meaningful: it
  means the matcher could not tell which road you are on, and showing a
  confident wrong road would be worse than showing nothing.

  1. Mount the phone rigidly (dash/windscreen clamp). Loose mounting is the
     single largest error source — 1.4 deg/s wobble gave 8.3% drift in our
     data, 5.7 deg/s gave 39.3%.
  2. Drive 2-3 min with GNSS on. Wait for "calibration R2" to reach >= 0.8;
     below that the gyro->yaw fit has not converged and heading is unreliable.
  3. Check the mount line reads GOOD *while moving*, not while parked.
  4. Press "Outage" to simulate a GNSS blackout. Red/blue separate from green.
  5. Press "Restore" to re-anchor and read the gap.

  "AI speed"  toggles the TFLite speed model on/off. Turn it ON: the retrained
  model beats the hold-speed baseline by 45.9% on held-out bike data
  (MAE 2.255 vs 4.171 m/s) and by 7.5% on trip1. It is still 23% WORSE than
  hold-speed on trip2, which is a near-constant-speed drive where holding the
  last speed is very hard to beat -- so compare both on your own route.

  "Record"    writes a trip in the 24-column IO-VNBD format for further
  training. Files land in the app's external files dir:
  /sdcard/Android/data/com.teamua.idr/files/

MAP MATCHING
  Snaps the drifting IMU path onto the road grid (HMM / Viterbi, Newson &
  Krumm 2009), correcting CROSS-TRACK error only -- it cannot fix along-track
  (speed) error, and it cannot rescue a heading that is already wrong, because
  a wrong heading makes it snap confidently to the wrong road.

  OFF BY DEFAULT. On-road testing on a motorcycle showed it made drift WORSE:
  with a noisy heading the matcher snaps confidently to the wrong road, which
  costs more than the cross-track error it removes. It now refuses to run at
  all until the gyro->yaw calibration reaches R2 >= 0.8, and the trust gate was
  tightened (margin > 3.0, snap < 60 m, cost < 20). Only enable it once the
  panel shows a converged calibration and the mount reads GOOD while moving.

  The OUTAGE panel line "map" shows the snapped-vs-GNSS distance and either
  the snap/margin figures or why the match was rejected.

  Roads cover BHOPAL ONLY. Outside that bbox the panel reads "outside map
  extract" and matching switches off; dead reckoning continues as before. To
  cover another city, re-run:
    python src/export_roads_asset.py --pbf <region>.osm.pbf \
        --bbox lat_min,lat_max,lon_min,lon_max \
        --out apps/IDRNavigator/app/src/main/assets/roads.bin

KNOWN LIMITS
  - Map tiles are fetched online; the basemap greys out in a real tunnel.
    Dead reckoning keeps running — only the background blanks.
  - Debug-signed, so it will not update over a Play Store build.
  - Turn detection is unreliable on a MOTORCYCLE. Heading is integrated from a
    fixed 4-parameter gyro fit, which assumes the phone's tilt relative to the
    world is constant; a bike leans into every turn, so the mapping from gyro
    axes to world-vertical yaw changes mid-turn. Map matching does not fix
    this -- it makes a wrong heading snap to a wrong road.
