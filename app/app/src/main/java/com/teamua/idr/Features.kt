package com.teamua.idr

import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.ln
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Port of preprocess_speed.window_features() -- 30 statistics per window.
 *
 * The ORDER must match feature_names() in the Python pipeline exactly, because
 * the normalisation vectors shipped in norm_stats.json are positional. Any
 * reordering silently produces garbage rather than an error.
 *
 * Every statistic is a magnitude, a spread or a rate of change, so the values
 * do not depend on how the phone is rotated in its mount.
 */
object Features {

    const val PER_SCALE = 30
    private const val N_BANDS = 5

    private fun percentile(sorted: DoubleArray, p: Double): Double {
        if (sorted.isEmpty()) return 0.0
        val idx = p / 100.0 * (sorted.size - 1)
        val lo = idx.toInt().coerceIn(0, sorted.size - 1)
        val hi = (lo + 1).coerceAtMost(sorted.size - 1)
        val f = idx - lo
        return sorted[lo] * (1 - f) + sorted[hi] * f
    }

    private fun mean(v: DoubleArray) = if (v.isEmpty()) 0.0 else v.sum() / v.size

    private fun std(v: DoubleArray): Double {
        if (v.size < 2) return 0.0
        val m = mean(v)
        var s = 0.0
        for (x in v) s += (x - m) * (x - m)
        return sqrt(s / v.size)
    }

    private fun variance(v: DoubleArray): Double {
        val s = std(v); return s * s
    }

    /** mean |diff| divided by dt, matching np.abs(np.diff(v)).mean() / dt */
    private fun absDeriv(v: DoubleArray, dt: Double): Double {
        if (v.size < 2) return 0.0
        var s = 0.0
        for (i in 1 until v.size) s += abs(v[i] - v[i - 1])
        return s / (v.size - 1) / dt
    }

    private fun diff(v: DoubleArray): DoubleArray {
        if (v.size < 2) return DoubleArray(0)
        return DoubleArray(v.size - 1) { v[it + 1] - v[it] }
    }

    /**
     * log1p of spectral power in N_BANDS equal slices up to Nyquist.
     * A direct DFT is used: windows are 20-100 samples, so O(N^2) is a few
     * thousand operations at 2 Hz -- an FFT would add code for no gain.
     */
    private fun bands(x: DoubleArray): DoubleArray {
        val out = DoubleArray(N_BANDS)
        val n = x.size
        if (n < 4) return out
        val m = mean(x)
        val half = n / 2
        val power = DoubleArray(half)
        for (k in 1..half) {          // skip DC, matching P[1:] in Python
            var re = 0.0; var im = 0.0
            val w = -2.0 * Math.PI * k / n
            for (t in 0 until n) {
                val v = x[t] - m
                re += v * cos(w * t); im += v * sin(w * t)
            }
            power[k - 1] = re * re + im * im
        }
        if (power.size < N_BANDS) return out
        // np.array_split: first (len % N) chunks get one extra element
        val base = power.size / N_BANDS
        val extra = power.size % N_BANDS
        var idx = 0
        for (b in 0 until N_BANDS) {
            val len = base + if (b < extra) 1 else 0
            var s = 0.0
            for (i in 0 until len) s += power[idx + i]
            idx += len
            out[b] = ln(1.0 + s)
        }
        return out
    }

    /**
     * 30 features for one window. Channel order matches the Python call
     * window_features(lax, lay, laz, gx, gy, gz, dt).
     */
    fun window(
        lax: DoubleArray, lay: DoubleArray, laz: DoubleArray,
        gx: DoubleArray, gy: DoubleArray, gz: DoubleArray, dt: Double
    ): DoubleArray {
        val n = lax.size
        val lin = DoubleArray(n) { sqrt(lax[it] * lax[it] + lay[it] * lay[it] + laz[it] * laz[it]) }
        val gyr = DoubleArray(n) { sqrt(gx[it] * gx[it] + gy[it] * gy[it] + gz[it] * gz[it]) }

        val f = ArrayList<Double>(PER_SCALE)

        // linacc_mean, then _spread -> std, p90, p10, range
        f.add(mean(lin))
        val ls = lin.clone(); ls.sort()
        val lp90 = percentile(ls, 90.0); val lp10 = percentile(ls, 10.0)
        f.add(std(lin)); f.add(lp90); f.add(lp10); f.add(lp90 - lp10)

        f.add(mean(gyr))
        val gs = gyr.clone(); gs.sort()
        val gp90 = percentile(gs, 90.0); val gp10 = percentile(gs, 10.0)
        f.add(std(gyr)); f.add(gp90); f.add(gp10); f.add(gp90 - gp10)

        for (v in arrayOf(lax, lay, laz, gx, gy, gz)) {
            f.add(std(v)); f.add(absDeriv(v, dt))
        }

        f.add(variance(diff(lin)))
        f.add(variance(diff(gyr)))
        var s = 0.0; for (x in laz) s += abs(x); f.add(s / n)

        for (b in bands(lin)) f.add(b)

        return f.toDoubleArray()
    }
}
