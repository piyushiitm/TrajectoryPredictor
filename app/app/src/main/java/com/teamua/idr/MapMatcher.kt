package com.teamua.idr

import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.min
import kotlin.math.sin

/**
 * HMM map matching (Newson & Krumm 2009), ported from `src/map_match.py`.
 *
 * The 60 s error budget splits almost evenly between cross-track (heading) and
 * along-track (speed) error. A road network constrains exactly the cross-track
 * half -- a vehicle is on a road, and roads run in specific directions.
 *
 *   States     candidate projections of each DR point onto nearby road points.
 *   Emission   Gaussian in the distance from the DR point to the candidate,
 *              with sigma growing as the outage runs on, so trust in the DR
 *              position decays instead of being assumed constant.
 *   Transition exponential in |distance between candidates - distance actually
 *              travelled|. This is what rejects a jump to a parallel road:
 *              reaching it would need more travel than the odometry reports.
 *   Decoding   Viterbi over the whole outage window, so a locally tempting
 *              wrong road is rejected when it cannot be reached consistently.
 *
 * Heading agreement is folded into the emission cost, which is what separates
 * the two carriageways of a divided road.
 *
 * Matching is BIMODAL: a correct snap removes cross-track error, a wrong snap
 * adds more than it removes. `Result.trustworthy` carries the gate, computed
 * only from quantities available at runtime -- no ground truth.
 */
object MapMatcher {

    class Result(
        val lat: DoubleArray,
        val lon: DoubleArray,
        val viterbiCost: Double,
        val margin: Double,
        val snapMeanM: Double,
        val snapEndM: Double,
        val density: Double,
    ) {
        /**
         * Reject a match that looks ambiguous or implausible. A large snap
         * distance means the matcher moved us further than the DR could
         * plausibly have drifted; a small margin means second-best was nearly
         * as good, i.e. parallel roads it could not tell apart.
         *
         * These thresholds were tightened after on-road testing: the original
         * margin > 1.0 / snap < 120 m accepted matches on a bike whose heading
         * was wrong, and a confident snap to the wrong road adds more error
         * than the cross-track error it removes.
         */
        val trustworthy: Boolean
            get() = margin > 3.0 && snapEndM < 60.0 && viterbiCost < 20.0
    }

    /**
     * @param elapsed seconds since the outage began, per sample
     * @param stride  decimation; the Viterbi runs on every [stride]-th sample
     */
    fun match(
        net: RoadNetwork,
        drLat: DoubleArray, drLon: DoubleArray, drHdg: DoubleArray,
        elapsed: DoubleArray,
        sigma0: Double = 8.0, sigmaRate: Double = 0.12,
        beta: Double = 12.0, hdgWeight: Double = 18.0,
        maxRadius: Double = 120.0, stride: Int = 10,
    ): Result? {
        val n = drLat.size
        if (n < 3) return null
        val xs = DoubleArray(n); val ys = DoubleArray(n)
        for (i in 0 until n) {
            val p = net.toXy(drLat[i], drLon[i]); xs[i] = p[0]; ys[i] = p[1]
        }
        val sel = (0 until n step stride).toList()
        if (sel.size < 3) return null
        val m = sel.size

        val cands = ArrayList<IntArray>(m)
        val emis = ArrayList<DoubleArray>(m)
        for (j in 0 until m) {
            val i = sel[j]
            val sig = min(sigma0 + sigmaRate * elapsed[i], maxRadius)
            val c = net.candidates(xs[i], ys[i], maxOf(sig * 2.5, 25.0))
            if (c.isEmpty()) return null
            val e = DoubleArray(c.size)
            for (q in c.indices) {
                val d = net.dist(c[q], xs[i], ys[i])
                // bearing disagreement; roads are bidirectional, so fold to [0, pi/2]
                val diff = net.bearing[c[q]] - drHdg[i]
                var dh = abs(atan2(sin(diff), cos(diff)))
                dh = min(dh, Math.PI - dh)
                e[q] = 0.5 * (d / sig) * (d / sig) + hdgWeight * (dh / Math.PI) * (dh / Math.PI)
            }
            cands.add(c); emis.add(e)
        }

        // distance actually travelled along the DR path between selected points
        val stepD = DoubleArray(m - 1)
        for (j in 0 until m - 1) {
            val a = sel[j]; val b = sel[j + 1]
            stepD[j] = hypot(xs[b] - xs[a], ys[b] - ys[a])
        }

        // Viterbi
        var cost = emis[0].copyOf()
        val back = ArrayList<IntArray>(m - 1)
        for (j in 1 until m) {
            val prev = cands[j - 1]; val cur = cands[j]
            val nc = DoubleArray(cur.size); val bp = IntArray(cur.size)
            for (q in cur.indices) {
                var best = Double.MAX_VALUE; var bi = 0
                for (p in prev.indices) {
                    val gap = hypot(net.x[cur[q]] - net.x[prev[p]].toDouble(),
                                    net.y[cur[q]] - net.y[prev[p]].toDouble())
                    val t = cost[p] + abs(gap - stepD[j - 1]) / beta
                    if (t < best) { best = t; bi = p }
                }
                nc[q] = best + emis[j][q]; bp[q] = bi
            }
            cost = nc; back.add(bp)
        }

        var bestIdx = 0
        for (q in cost.indices) if (cost[q] < cost[bestIdx]) bestIdx = q
        val sorted = cost.sorted()
        val margin = if (sorted.size > 1) sorted[1] - sorted[0] else 0.0

        val path = IntArray(m)
        path[m - 1] = bestIdx
        for (j in m - 2 downTo 0) path[j] = back[j][path[j + 1]]

        val mx = DoubleArray(m); val my = DoubleArray(m)
        var offSum = 0.0
        for (j in 0 until m) {
            val i = cands[j][path[j]]
            mx[j] = net.x[i].toDouble(); my[j] = net.y[i].toDouble()
            offSum += hypot(mx[j] - xs[sel[j]], my[j] - ys[sel[j]])
        }

        // interpolate the snapped track back onto every sample
        val lat = DoubleArray(n); val lon = DoubleArray(n)
        for (i in 0 until n) {
            val fx = interp(i.toDouble(), sel, mx)
            val fy = interp(i.toDouble(), sel, my)
            val ll = net.toLatLon(fx, fy); lat[i] = ll[0]; lon[i] = ll[1]
        }

        val snapEnd = hypot(mx[m - 1] - xs[sel[m - 1]], my[m - 1] - ys[sel[m - 1]])
        return Result(lat, lon, cost[bestIdx] / m, margin, offSum / m, snapEnd,
                      cands.sumOf { it.size }.toDouble() / m)
    }

    private fun interp(at: Double, xs: List<Int>, ys: DoubleArray): Double {
        if (at <= xs[0]) return ys[0]
        if (at >= xs[xs.size - 1]) return ys[ys.size - 1]
        var k = 0
        while (k < xs.size - 1 && xs[k + 1] < at) k++
        val x0 = xs[k].toDouble(); val x1 = xs[k + 1].toDouble()
        val f = if (x1 > x0) (at - x0) / (x1 - x0) else 0.0
        return ys[k] + f * (ys[k + 1] - ys[k])
    }
}
