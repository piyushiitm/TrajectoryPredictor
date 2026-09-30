package com.teamua.idr

import android.content.Context
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sin

/**
 * The deployable pipeline, on-device.
 *
 *     position(t) = p0 + integral of v(t) * [sin h(t), cos h(t)] dt
 *
 * ONE integration. Acceleration is never integrated: measured on this data the
 * accelerometer's 1 s velocity increment correlates 0.02-0.07 with the true
 * increment, because mount tilt wobble fabricates more phantom horizontal
 * acceleration than the vehicle actually produces. Integrating it diverges to
 * 140-400% path error.
 *
 * Speed comes from the TFLite head; heading from the calibrated gyro. Both are
 * anchored at the moment GNSS is lost and coast from there.
 *
 * On a well-mounted phone at short outages, simply HOLDING the last known
 * speed beat the model (8.3% vs 15.3% drift on trip6), so both are computed
 * and exposed -- the UI shows each, and `useModel` selects which drives the
 * displayed track.
 */
class IdrEngine(ctx: Context) {

    companion object {
        const val FS = 10.0
        const val R_EARTH = 6_371_000.0
        private val SCALES = doubleArrayOf(2.0, 5.0, 10.0)
        private const val BUF = 120                     // 12 s at 10 Hz
    }

    /**
     * Two speed heads, switchable on the road.
     *
     *   TUNED    trained on the 25 bike recordings only. Sharper here: on the
     *            two most recent unseen rides it scored +39.5% and +31.9%
     *            against hold-speed, versus +35.0% and +27.2% for GENERAL.
     *            It degrades on other vehicles, though -- a mount-only variant
     *            of the same idea fell to -47.5% on a car recording.
     *
     *   GENERAL  the same bike data plus the IO-VNBD car pool. Slightly behind
     *            on this bike, clearly ahead off it: +25.6% versus +13.0% on a
     *            phone, mount and vehicle it had never seen.
     */
    enum class SpeedHead { TUNED, GENERAL }

    private val tuned = SpeedModel(ctx, "speed_tuned")
    private val general = SpeedModel(ctx, "speed_general")
    var speedHead = SpeedHead.TUNED
    private val model get() = if (speedHead == SpeedHead.TUNED) tuned else general
    val calib = YawCalibrator(180.0)

    /**
     * Magnetic compass, fused into the heading. The gyro alone drifted to 96.2%
     * positional error on a 9.4 km single-anchor run; with this fusion the same
     * run came out at 9.1%. The gain grows with outage length -- under about
     * 10 s the gyro has not drifted yet and the compass only adds noise.
     */
    val compass = MagCompass()

    /** Which heading source drives the dead reckoning. */
    enum class Heading { GYRO, MAG, FUSED, ML_GYRO }

    /**
     * FUSED by default. Measured on the mounted recordings: at 10 s the three
     * are within half a point of each other, because the gyro has not drifted
     * yet. The difference appears at length -- on a 9.4 km single-anchor run
     * GYRO gave 96.2 % drift, MAG 15.8 % and FUSED 9.1 %.
     */
    // GYRO by default for this build. On the two most recent unseen recordings
    // gyro beat fused at 10 s, 30 s and 60 s; the compass only wins on very long
    // blackouts (one 8.5 km run went 101% gyro to 20% fused). Switch to FUSED
    // when the outage is minutes rather than seconds.
    var headingMode = Heading.GYRO

    /** Learned heading correction; survived leave-one-recording-out at +17.2%. */
    private val headModel = HeadingModel(ctx)

    /**
     * Reference track: TRUE GPS bearing with the AI speed. Heading is perfect,
     * so whatever this track's error is belongs to the speed head alone. It is a
     * CEILING for comparison during a simulated outage, not a deployable mode --
     * a real blackout is exactly the loss of that bearing.
     */
    var refLat = 0.0; var refLon = 0.0; private set
    /** Offset from the compass frame to true bearing, anchored once at GNSS loss. */
    private var magOffset = 0.0
    private var magOk = false

    /** null when roads.bin is absent or the route leaves the extract. */
    val roads: RoadNetwork? = RoadNetwork.load(ctx)
    /**
     * Off by default. Map matching only removes error when the heading feeding
     * it is roughly right; measured on a loosely mounted phone it snapped to
     * the wrong road and made drift worse. Enable it once the calibration has
     * converged -- the gate in runMatch() enforces that too.
     */
    var useMapMatch = false

    // DR track accumulated during the outage, fed to the matcher
    private val trkLat = ArrayList<Double>(4096)
    private val trkLon = ArrayList<Double>(4096)
    private val trkHdg = ArrayList<Double>(4096)
    private val trkEl = ArrayList<Double>(4096)
    private var lastMatch = -1.0

    /** Snapped position, valid only while [matchOk]. */
    var mmLat = 0.0; var mmLon = 0.0; private set
    var matchOk = false; private set
    var matchInfo = ""; private set

    private val lax = DoubleArray(BUF); private val lay = DoubleArray(BUF)
    private val laz = DoubleArray(BUF); private val gxB = DoubleArray(BUF)
    private val gyB = DoubleArray(BUF); private val gzB = DoubleArray(BUF)
    private var head = 0; private var filled = 0

    /**
     * The engine's clock is the SENSOR clock, taken from SensorHub samples.
     * It must not be mixed with System.nanoTime() from the UI: those are
     * different clocks with different origins, and an outage anchored in one
     * while samples arrive in the other makes `elapsed` meaningless -- which
     * silently corrupts the speed model's input, the fusion's tau and the
     * heading head's anchor age all at once.
     */
    private var lastT = -1.0
    val sensorTime get() = lastT

    // live GNSS state
    var gpsLat = 0.0; var gpsLon = 0.0; var gpsSpeed = 0.0
    var gpsBearing = 0.0; var haveGps = false; private set

    // dead-reckoned state
    var drLat = 0.0; var drLon = 0.0; private set
    var drHeading = 0.0; private set
    var drSpeed = 0.0; private set
    var holdLat = 0.0; var holdLon = 0.0; private set

    var outage = false; private set
    var outageStart = 0.0; private set
    var elapsed = 0.0; private set
    var anchorSpeed = 0.0; private set
    private var anchorHeading = 0.0
    var useModel = true

    private var lastPredict = -1.0
    /** False when the last model output was rejected as implausible. */
    var modelOk = true; private set
    private var lastMagHeading: Double? = null
    private var dhGyro = 0.0          // heading change since the anchor, gyro only
    private var lastCorrect = -1.0
    /** Heading from integration + compass only, before the learned correction. */
    private var intHeading = 0.0
    /**
     * Latest learned correction, radians. The head predicts the TOTAL heading
     * error accumulated since the anchor (true - dh_gyro), so it is applied as an
     * offset on top of [intHeading] -- never added into it. Adding it every 0.5 s
     * re-applied the whole error twice a second and spun the heading, which made
     * every dead-reckoned marker circle on the spot in ML_GYRO mode.
     */
    private var mlCorr = 0.0

    private fun wrapPi(x: Double): Double {
        var v = x
        while (v > Math.PI) v -= 2 * Math.PI
        while (v < -Math.PI) v += 2 * Math.PI
        return v
    }

    fun onGps(lat: Double, lon: Double, speed: Double, bearingDeg: Double,
              @Suppress("UNUSED_PARAMETER") tIgnored: Double = 0.0) {
        val t = lastT                       // sensor clock, never the UI's
        gpsLat = lat; gpsLon = lon; gpsSpeed = speed
        gpsBearing = Math.toRadians(bearingDeg)
        haveGps = true
        if (speed > 2.0) calib.addFix(t, gpsBearing)
        if (!outage) {                                  // track GPS exactly
            drLat = lat; drLon = lon; holdLat = lat; holdLon = lon
            refLat = lat; refLon = lon
            drHeading = gpsBearing; intHeading = gpsBearing; drSpeed = speed
        }
    }

    fun startOutage(@Suppress("UNUSED_PARAMETER") tIgnored: Double = 0.0) {
        // anchored on the sensor clock, the same one onSample advances
        outage = true; outageStart = lastT; elapsed = 0.0
        // anchor the compass to the last known true bearing; nothing from GNSS
        // is used after this point
        val mh = lastMagHeading
        magOk = compass.reliable && mh != null
        if (mh != null) magOffset = wrapPi(gpsBearing - mh)
        trkLat.clear(); trkLon.clear(); trkHdg.clear(); trkEl.clear()
        lastMatch = -1.0; matchOk = false; matchInfo = ""
        dhGyro = 0.0; lastCorrect = -1.0; lastPredict = -1.0; modelOk = true
        mlCorr = 0.0; intHeading = gpsBearing
        mmLat = gpsLat; mmLon = gpsLon
        anchorSpeed = gpsSpeed
        anchorHeading = gpsBearing
        drLat = gpsLat; drLon = gpsLon; drHeading = gpsBearing; drSpeed = gpsSpeed
        holdLat = gpsLat; holdLon = gpsLon
        refLat = gpsLat; refLon = gpsLon
    }

    fun stopOutage() {
        outage = false; elapsed = 0.0; matchOk = false; matchInfo = ""
        trkLat.clear(); trkLon.clear(); trkHdg.clear(); trkEl.clear()
    }

    fun onSample(s: SensorHub.Sample) {
        val dt = if (lastT < 0) 1.0 / FS else (s.t - lastT)
        lastT = s.t
        if (dt <= 0 || dt > 1.0) return

        lax[head] = s.lax; lay[head] = s.lay; laz[head] = s.laz
        gxB[head] = s.gx; gyB[head] = s.gy; gzB[head] = s.gz
        head = (head + 1) % BUF
        if (filled < BUF) filled++

        calib.addGyro(s.gx, s.gy, s.gz, dt)
        compass.add(s.mag[0], s.mag[1], s.mag[2])
        lastMagHeading = compass.heading(s.mag[0], s.mag[1], s.mag[2], s.grav)
        if (!outage) return

        elapsed = s.t - outageStart
        val yawRate = calib.yawRate(s.gx, s.gy, s.gz)
        intHeading += yawRate * dt
        dhGyro += yawRate * dt
        // complementary fusion: the gyro supplies the short-term shape, the
        // compass the absolute reference. tau shrinks as the outage runs on, so
        // the longer the blackout the more the drifting gyro yields.
        val mh = lastMagHeading
        val wantsCompass = headingMode == Heading.MAG || headingMode == Heading.FUSED
        if (wantsCompass && mh != null && magOk) {
            if (headingMode == Heading.MAG) {
                // compass alone: nothing is integrated, so nothing drifts
                intHeading = mh + magOffset
            } else {
                val tau = compass.tauFor(elapsed)
                val alpha = dt / (tau + dt)
                intHeading += alpha * wrapPi(mh + magOffset - intHeading)
            }
        }
        // learned correction, applied at 2 Hz on top of whatever produced the
        // heading above. It predicts the residual against the fused estimate, so
        // it is a nudge, not a replacement.
        // ML_GYRO only. The head is fitted to the GYRO's residual; a head trained
        // on the fused residual scored -7.1% leave-one-out against +21.3% for
        // this one, because the compass has already removed what was learnable.
        if (headingMode == Heading.ML_GYRO &&
            filled >= BUF && (lastCorrect < 0 || s.t - lastCorrect >= 0.5)) {
            lastCorrect = s.t
            val f = buildFeatures()
            if (f != null) {
                val dhFused = wrapPi(intHeading - anchorHeading)
                val c = headModel.correct(f, anchorSpeed, elapsed, dhGyro, dhFused, yawRate)
                // same clip as the replay server (+-0.5 rad); non-finite is ignored
                if (c.isFinite()) mlCorr = c.coerceIn(-0.5, 0.5)
            }
        }
        drHeading = if (headingMode == Heading.ML_GYRO) intHeading + mlCorr else intHeading

        // the model runs at 2 Hz; between updates the last speed is held
        if (filled >= BUF && (lastPredict < 0 || s.t - lastPredict >= 0.5)) {
            lastPredict = s.t
            val f = buildFeatures()
            if (f != null) {
                val p = model.predict(f, anchorSpeed, elapsed)
                // A prediction is only accepted if it is finite and physically
                // plausible. The model clamps at zero, so a wildly negative
                // residual would otherwise pin the speed at 0 and freeze the
                // marker in place -- silently, and looking like a dead pointer
                // rather than a bad number.
                modelOk = p.isFinite() && p < 60.0 &&
                          Math.abs(p - anchorSpeed) < 25.0
                drSpeed = if (modelOk) p else anchorSpeed
            }
        }
        val v = if (useModel) drSpeed else anchorSpeed

        step(v, drHeading, dt) { la, lo -> drLat = la; drLon = lo }
        // hold-speed track, for side-by-side comparison
        stepFrom(holdLat, holdLon, anchorSpeed, drHeading, dt) { la, lo ->
            holdLat = la; holdLon = lo }
        // reference track: live GPS bearing with the model's speed. During a
        // SIMULATED outage GNSS is still arriving, so this shows how much of the
        // remaining error is speed rather than heading.
        stepFrom(refLat, refLon, v, gpsBearing, dt) { la, lo ->
            refLat = la; refLon = lo }

        trkLat.add(drLat); trkLon.add(drLon); trkHdg.add(drHeading); trkEl.add(elapsed)
        // Re-match every 2 s over the whole outage so far. Viterbi is run from
        // scratch each time rather than incrementally: the point of decoding the
        // full window is that later evidence can overturn an early wrong road,
        // which an incremental version would have already committed to.
        if (useMapMatch && s.t - lastMatch >= 2.0 && trkLat.size >= 30) {
            lastMatch = s.t
            runMatch()
        }
    }

    private fun runMatch() {
        val net = roads ?: run { matchInfo = "no road data"; return }
        // a wrong heading makes the matcher snap confidently to a wrong road,
        // so refuse to match at all until the gyro->yaw fit has converged
        if (!calib.ready || calib.r2 < 0.8) {
            matchOk = false
            matchInfo = "heading not calibrated (R2 %.2f)".format(calib.r2)
            return
        }
        if (!net.covers(trkLat[0], trkLon[0])) {
            matchOk = false; matchInfo = "outside map extract"; return
        }
        val r = MapMatcher.match(net, trkLat.toDoubleArray(), trkLon.toDoubleArray(),
                                 trkHdg.toDoubleArray(), trkEl.toDoubleArray())
        if (r == null) { matchOk = false; matchInfo = "no candidates"; return }
        matchOk = r.trustworthy
        mmLat = r.lat[r.lat.size - 1]; mmLon = r.lon[r.lon.size - 1]
        matchInfo = if (matchOk)
            "snap %.0fm  margin %.1f".format(r.snapEndM, r.margin)
        else
            "rejected (snap %.0fm  margin %.1f)".format(r.snapEndM, r.margin)
    }

    /** Distance from the snapped position to live GNSS, metres. */
    fun matchDriftMetres(): Double {
        if (!haveGps || !matchOk) return 0.0
        val dLat = Math.toRadians(mmLat - gpsLat)
        val dLon = Math.toRadians(mmLon - gpsLon) * cos(Math.toRadians(gpsLat))
        return Math.hypot(dLat * R_EARTH, dLon * R_EARTH)
    }

    private inline fun step(v: Double, h: Double, dt: Double, out: (Double, Double) -> Unit) =
        stepFrom(drLat, drLon, v, h, dt, out)

    private inline fun stepFrom(la: Double, lo: Double, v: Double, h: Double,
                                dt: Double, out: (Double, Double) -> Unit) {
        val dN = v * cos(h) * dt
        val dE = v * sin(h) * dt
        val nLat = la + Math.toDegrees(dN / R_EARTH)
        val nLon = lo + Math.toDegrees(dE / (R_EARTH * cos(Math.toRadians(la))))
        out(nLat, nLon)
    }

    /** 90 window statistics: 30 at each of 2 s, 5 s, 10 s, most recent first. */
    private fun buildFeatures(): DoubleArray? {
        val out = DoubleArray(SCALES.size * Features.PER_SCALE)
        var o = 0
        for (sc in SCALES) {
            val n = (sc * FS).toInt()
            if (n > filled) return null
            val a = DoubleArray(n); val b = DoubleArray(n); val c = DoubleArray(n)
            val d = DoubleArray(n); val e = DoubleArray(n); val f = DoubleArray(n)
            for (i in 0 until n) {
                val idx = ((head - n + i) % BUF + BUF) % BUF
                a[i] = lax[idx]; b[i] = lay[idx]; c[i] = laz[idx]
                // Gyro axes are REORDERED to match training. Recorder.kt writes
                // the device triad (x, y, z) into columns labelled Yaw, Pitch,
                // Roll, and the training pipeline then reads them as
                // (gx, gy, gz) = (roll, pitch, yaw) = device (z, y, x).
                // Twelve of the ninety features are per-axis, so feeding the
                // device order here silently corrupted them.
                d[i] = gzB[idx]; e[i] = gyB[idx]; f[i] = gxB[idx]
            }
            val w = Features.window(a, b, c, d, e, f, 1.0 / FS)
            System.arraycopy(w, 0, out, o, w.size); o += w.size
        }
        return out
    }

    /** Straight-line distance between the GPS and dead-reckoned markers. */
    fun driftMetres(): Double {
        if (!haveGps) return 0.0
        val dLat = Math.toRadians(drLat - gpsLat)
        val dLon = Math.toRadians(drLon - gpsLon) * cos(Math.toRadians(gpsLat))
        return Math.hypot(dLat * R_EARTH, dLon * R_EARTH)
    }

    fun holdDriftMetres(): Double {
        if (!haveGps) return 0.0
        val dLat = Math.toRadians(holdLat - gpsLat)
        val dLon = Math.toRadians(holdLon - gpsLon) * cos(Math.toRadians(gpsLat))
        return Math.hypot(dLat * R_EARTH, dLon * R_EARTH)
    }

    fun close() { tuned.close(); general.close(); headModel.close() }
}
