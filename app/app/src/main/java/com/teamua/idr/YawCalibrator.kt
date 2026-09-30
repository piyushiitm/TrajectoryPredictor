package com.teamua.idr

import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.sin

/**
 * Least-squares fit of the three gyro axes plus a bias to the heading change
 * observed between GPS fixes, exactly as idr_core.calibrate_yaw does.
 *
 * This is the "In-Vehicle Alignment & Calibration Engine". It matters because
 * gyro channel names cannot be trusted -- in IO-VNBD the channel labelled
 * "yaw" correlates 0.006 with true yaw rate while the one labelled "pitch"
 * correlates 0.95 -- and because projecting onto gravity fails whenever the
 * gyro triad is permuted relative to the accelerometer triad. Fitting recovers
 * the axis mapping, scale, sign and bias together, from data that is available
 * before the outage begins.
 *
 * Measured: a 300 s window reaches R^2 ~0.99 and holds heading to ~10 deg over
 * a 60 s blackout; a 60 s window only manages R^2 0.75 and 25 deg.
 */
class YawCalibrator(private val windowSec: Double = 180.0) {

    private data class Fix(val t: Double, val bearing: Double)
    private data class Row(val gx: Double, val gy: Double, val gz: Double,
                           val dt: Double, val dTheta: Double)

    private val fixes = ArrayList<Fix>()
    private val rows = ArrayList<Row>()

    // gyro integral since the previous fix
    private var sx = 0.0; private var sy = 0.0; private var sz = 0.0
    private var tAccum = 0.0

    /** coefficients [wx, wy, wz, bias]; identity-ish until the first fit */
    var coef = doubleArrayOf(0.0, 0.0, -1.0, 0.0); private set
    var r2 = 0.0; private set
    var nRows = 0; private set
    val ready: Boolean get() = nRows >= 8

    private fun wrap(a: Double): Double {
        var x = a
        while (x > Math.PI) x -= 2 * Math.PI
        while (x < -Math.PI) x += 2 * Math.PI
        return x
    }

    fun addGyro(gx: Double, gy: Double, gz: Double, dt: Double) {
        if (dt <= 0 || dt > 2.0) return
        sx += gx * dt; sy += gy * dt; sz += gz * dt; tAccum += dt
    }

    /** Call on every genuine GPS fix while the vehicle is moving. */
    fun addFix(t: Double, bearingRad: Double) {
        val prev = fixes.lastOrNull()
        fixes.add(Fix(t, bearingRad))
        if (prev != null && tAccum > 0.4 && tAccum < 15.0) {
            rows.add(Row(sx, sy, sz, tAccum, wrap(bearingRad - prev.bearing)))
        }
        sx = 0.0; sy = 0.0; sz = 0.0; tAccum = 0.0
        val cutoff = t - windowSec
        while (fixes.size > 2 && fixes.first().t < cutoff) fixes.removeAt(0)
        while (rows.size > 400) rows.removeAt(0)
        fit()
    }

    /** Normal equations for a 4-parameter fit -- no matrix library needed. */
    private fun fit() {
        val n = rows.size
        nRows = n
        if (n < 8) return
        val p = 4
        val ata = Array(p) { DoubleArray(p) }
        val atb = DoubleArray(p)
        var ybar = 0.0
        for (r in rows) ybar += r.dTheta
        ybar /= n
        for (r in rows) {
            val a = doubleArrayOf(r.gx, r.gy, r.gz, r.dt)
            for (i in 0 until p) {
                for (j in 0 until p) ata[i][j] += a[i] * a[j]
                atb[i] += a[i] * r.dTheta
            }
        }
        for (i in 0 until p) ata[i][i] += 1e-9          // ridge, for stability
        val x = solve(ata, atb) ?: return
        var ssRes = 0.0; var ssTot = 0.0
        for (r in rows) {
            val pred = r.gx * x[0] + r.gy * x[1] + r.gz * x[2] + r.dt * x[3]
            ssRes += (r.dTheta - pred) * (r.dTheta - pred)
            ssTot += (r.dTheta - ybar) * (r.dTheta - ybar)
        }
        coef = x
        r2 = if (ssTot > 1e-12) 1.0 - ssRes / ssTot else 0.0
    }

    private fun solve(a: Array<DoubleArray>, b: DoubleArray): DoubleArray? {
        val n = b.size
        val m = Array(n) { i -> DoubleArray(n + 1) { j -> if (j < n) a[i][j] else b[i] } }
        for (c in 0 until n) {
            var piv = c
            for (r in c + 1 until n) if (abs(m[r][c]) > abs(m[piv][c])) piv = r
            if (abs(m[piv][c]) < 1e-12) return null
            val tmp = m[c]; m[c] = m[piv]; m[piv] = tmp
            for (r in 0 until n) {
                if (r == c) continue
                val f = m[r][c] / m[c][c]
                for (k in c..n) m[r][k] -= f * m[c][k]
            }
        }
        // after Gauss-Jordan each row reads x_i * m[i][i] = m[i][n] (the augmented
        // column). Reading m[i][i + 1] instead returned 0 for every gyro weight and
        // left only the bias, so the calibrated yaw rate ignored the gyro entirely.
        return DoubleArray(n) { m[it][n] / m[it][it] }
    }

    /** Calibrated yaw rate, rad/s, compass sense. */
    fun yawRate(gx: Double, gy: Double, gz: Double): Double =
        gx * coef[0] + gy * coef[1] + gz * coef[2] + coef[3]
}
