package com.teamua.idr

import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Tilt-compensated magnetic heading with online hard-iron correction.
 *
 * Why this exists. The gyro-integrated heading DRIFTS -- measured on a 9.4 km
 * single-anchor run it reached 93.5 deg of heading error and 96.2 % positional
 * drift. The compass does not drift: its error stays put however long the
 * blackout runs. Fusing the two gives the gyro's smoothness with the compass's
 * absolute reference, and on that same run it cut drift to 9.1 %.
 *
 * HARD-IRON CORRECTION IS NOT OPTIONAL. A fixed magnetic offset from the engine,
 * the frame or the phone's own magnets shifts the measured field off centre, and
 * uncorrected the heading error was 12-47 deg. Removing the offset cut it to
 * 3.7-21.5 deg -- a 2-3x improvement on every recording tested.
 *
 * The offset is the centre of the sphere the field traces as the phone rotates,
 * fitted by least squares on |m - c|^2 = r^2, which is linear in (c, r). It is
 * accumulated online from normal riding; no figure-of-eight waving is needed,
 * though the fit only becomes meaningful once the bike has turned through a
 * decent spread of headings.
 *
 * TRUST GATE: the field's magnitude should be near-constant as the phone turns.
 * When it is not, something nearby is distorting it and the heading is
 * worthless. |B| std was 4-5 uT on the mounted recordings where this worked and
 * 30.7 uT in a pocket where it did not, so [reliable] gates on that.
 */
class MagCompass {

    companion object {
        private const val MIN_SAMPLES = 200
        private const val BSTD_GOOD = 12.0      // uT; pocket measured 30.7, mount 4-5
        /**
         * Reject a reading whose field MAGNITUDE departs this far from the
         * running norm. Earth's field is near-constant in magnitude, so a spike
         * means something local is bending it -- a passing truck, a steel
         * bridge, rebar, overhead lines. Measured on a road test: |B| changed
         * 18.5 uT/s during spurious heading jumps versus 0.72 uT/s otherwise,
         * a 26x tell, and every one of those jumps was a false turn of up to
         * 90 deg. Rejecting them leaves the gyro to carry the heading through.
         */
        private const val B_TOL = 0.15          // fraction of the running mean
    }

    // running sums for the sphere fit: A^T A and A^T b, A = [2mx 2my 2mz 1]
    private val ata = Array(4) { DoubleArray(4) }
    private val atb = DoubleArray(4)
    private var n = 0

    // running mean/variance of |B|, for the trust gate
    private var bSum = 0.0
    private var bSq = 0.0

    private val c = DoubleArray(3)             // hard-iron offset
    var fitted = false; private set
    val samples get() = n

    /** Readings dropped as disturbed, for the UI. */
    var rejected = 0; private set

    private val bMean get() = if (n > 0) bSum / n else 0.0

    /** Std deviation of the field magnitude, uT. Low = clean, high = distorted. */
    val bStd: Double
        get() {
            if (n < 2) return 0.0
            val m = bSum / n
            return sqrt(maxOf(0.0, bSq / n - m * m))
        }

    val reliable get() = fitted && n >= MIN_SAMPLES && bStd < BSTD_GOOD

    fun add(mx: Float, my: Float, mz: Float) {
        val x = mx.toDouble(); val y = my.toDouble(); val z = mz.toDouble()
        if (x == 0.0 && y == 0.0 && z == 0.0) return
        val b = sqrt(x * x + y * y + z * z)
        if (b < 1.0 || b > 200.0) return       // implausible field, ignore
        // do not let a disturbed reading pollute the hard-iron fit either
        val m0 = bMean
        if (n > 50 && m0 > 1.0 && abs(b - m0) / m0 > B_TOL) { rejected++; return }
        bSum += b; bSq += b * b
        val row = doubleArrayOf(2 * x, 2 * y, 2 * z, 1.0)
        val rhs = x * x + y * y + z * z
        for (i in 0..3) {
            for (j in 0..3) ata[i][j] += row[i] * row[j]
            atb[i] += row[i] * rhs
        }
        n++
        if (n >= MIN_SAMPLES && n % 50 == 0) solve()
    }

    /** 4x4 solve by Gaussian elimination with partial pivoting; no matrix lib. */
    private fun solve() {
        val a = Array(4) { i -> DoubleArray(5) { j -> if (j < 4) ata[i][j] else atb[i] } }
        for (col in 0..3) {
            var piv = col
            for (r in col + 1..3) if (abs(a[r][col]) > abs(a[piv][col])) piv = r
            if (abs(a[piv][col]) < 1e-9) return          // singular: not enough spread yet
            val t = a[col]; a[col] = a[piv]; a[piv] = t
            for (r in 0..3) {
                if (r == col) continue
                val f = a[r][col] / a[col][col]
                for (k in col..4) a[r][k] -= f * a[col][k]
            }
        }
        for (i in 0..2) c[i] = a[i][4] / a[i][i]
        fitted = true
    }

    /**
     * Heading of the magnetic field projected into the horizontal plane, using
     * [grav] to define which way is down. Returns radians in an arbitrary but
     * CONSISTENT frame -- the offset to true bearing is anchored once from GNSS,
     * so only changes matter here.
     */
    fun heading(mx: Float, my: Float, mz: Float, grav: FloatArray): Double? {
        // Disturbed sample: report nothing rather than a wrong bearing. The
        // engine then coasts on the gyro, which is correct over the few seconds
        // a passing vehicle or bridge lasts.
        val b = sqrt((mx * mx + my * my + mz * mz).toDouble())
        val m0 = bMean
        if (n > 50 && m0 > 1.0 && abs(b - m0) / m0 > B_TOL) return null
        val gn = sqrt((grav[0] * grav[0] + grav[1] * grav[1] + grav[2] * grav[2]).toDouble())
        if (gn < 1e-6) return null
        val gx = grav[0] / gn; val gy = grav[1] / gn; val gz = grav[2] / gn
        val x = mx - c[0]; val y = my - c[1]; val z = mz - c[2]
        // remove the component along gravity, leaving the horizontal field
        val d = x * gx + y * gy + z * gz
        val hx = x - d * gx; val hy = y - d * gy; val hz = z - d * gz
        // an orthonormal horizontal basis built from gravity
        var e1x = 1.0 - gx * gx; var e1y = -gx * gy; var e1z = -gx * gz
        var nn = sqrt(e1x * e1x + e1y * e1y + e1z * e1z)
        if (nn < 1e-6) {                        // phone x nearly vertical: use y
            e1x = -gy * gx; e1y = 1.0 - gy * gy; e1z = -gy * gz
            nn = sqrt(e1x * e1x + e1y * e1y + e1z * e1z)
            if (nn < 1e-6) return null
        }
        e1x /= nn; e1y /= nn; e1z /= nn
        val e2x = gy * e1z - gz * e1y
        val e2y = gz * e1x - gx * e1z
        val e2z = gx * e1y - gy * e1x
        return atan2(hx * e2x + hy * e2y + hz * e2z, hx * e1x + hy * e1y + hz * e1z)
    }

    /**
     * Complementary-filter time constant, seconds, as a function of how long the
     * outage has run. Swept on four recordings: short windows preferred tau
     * about 10 s, while a full-trip single-anchor run preferred about 2 s. The
     * longer the blackout, the more the drifting gyro should yield to the
     * compass.
     */
    fun tauFor(elapsed: Double): Double = when {
        elapsed < 60.0 -> 10.0
        elapsed > 300.0 -> 2.0
        else -> 10.0 - (elapsed - 60.0) * 8.0 / 240.0
    }
}
