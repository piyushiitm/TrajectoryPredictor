package com.teamua.idr

import android.content.Context
import java.io.DataInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min

/**
 * Drivable road geometry, pre-densified on the desktop side.
 *
 * The phone never parses OSM. `export_roads_asset.py` has already selected the
 * drivable ways, projected them into a local metric frame about (lat0, lon0),
 * resampled them to a fixed step and precomputed each segment's bearing, so
 * all that remains here is to read three float arrays and index them.
 *
 * The index is a uniform grid rather than a k-d tree: lookups are always
 * "everything within r metres", the points are near-uniformly spaced by
 * construction, and a grid needs no allocation per query.
 *
 * Asset layout (little-endian), matching the exporter:
 *   magic "IDRR" | int32 n | float64 lat0 | float64 lon0 | float32 step_m
 *   then n * (float32 x, float32 y, float32 bearing_rad)
 */
class RoadNetwork private constructor(
    val lat0: Double,
    val lon0: Double,
    val x: FloatArray,
    val y: FloatArray,
    val bearing: FloatArray,
) {
    companion object {
        const val R_EARTH = 6_371_000.0
        private const val CELL = 50.0f          // grid cell size, metres

        /** Returns null when the asset is absent, so matching degrades to off. */
        fun load(ctx: Context, name: String = "roads.bin"): RoadNetwork? {
            val bytes = try {
                ctx.assets.open(name).use { DataInputStream(it).readBytes() }
            } catch (e: Exception) {
                return null
            }
            if (bytes.size < 24) return null
            val bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
            val magic = ByteArray(4).also { bb.get(it) }
            if (String(magic) != "IDRR") return null
            val n = bb.int
            if (n <= 0 || bytes.size < 24 + 12L * n) return null
            val lat0 = bb.double; val lon0 = bb.double
            bb.float                                   // step_m, informational
            val x = FloatArray(n); val y = FloatArray(n); val b = FloatArray(n)
            for (i in 0 until n) { x[i] = bb.float; y[i] = bb.float; b[i] = bb.float }
            return RoadNetwork(lat0, lon0, x, y, b).also { it.buildIndex() }
        }
    }

    val size get() = x.size

    // uniform grid: cellStart[c] .. cellStart[c+1] indexes into cellItems
    private var minX = 0f; private var minY = 0f
    private var nx = 0; private var ny = 0
    private lateinit var cellStart: IntArray
    private lateinit var cellItems: IntArray

    private fun buildIndex() {
        minX = x.min(); minY = y.min()
        nx = max(1, (((x.max() - minX) / CELL).toInt() + 1))
        ny = max(1, (((y.max() - minY) / CELL).toInt() + 1))
        val counts = IntArray(nx * ny + 1)
        val cellOf = IntArray(size)
        for (i in 0 until size) {
            val c = cell(x[i], y[i]); cellOf[i] = c; counts[c + 1]++
        }
        for (c in 1 until counts.size) counts[c] += counts[c - 1]
        cellStart = counts
        val fill = cellStart.copyOf()
        cellItems = IntArray(size)
        for (i in 0 until size) cellItems[fill[cellOf[i]]++] = i
    }

    private fun cell(px: Float, py: Float): Int {
        val ix = min(nx - 1, max(0, ((px - minX) / CELL).toInt()))
        val iy = min(ny - 1, max(0, ((py - minY) / CELL).toInt()))
        return iy * nx + ix
    }

    /** Indices of up to [k] road points within [radius] metres of (px, py). */
    fun candidates(px: Double, py: Double, radius: Double, k: Int = 8): IntArray {
        val r = radius.toFloat()
        val ix0 = min(nx - 1, max(0, ((px - minX - r) / CELL).toInt()))
        val ix1 = min(nx - 1, max(0, ((px - minX + r) / CELL).toInt()))
        val iy0 = min(ny - 1, max(0, ((py - minY - r) / CELL).toInt()))
        val iy1 = min(ny - 1, max(0, ((py - minY + r) / CELL).toInt()))
        val hits = ArrayList<Int>(64); val dist = ArrayList<Double>(64)
        val r2 = radius * radius
        for (iy in iy0..iy1) for (ix in ix0..ix1) {
            val c = iy * nx + ix
            for (p in cellStart[c] until cellStart[c + 1]) {
                val i = cellItems[p]
                val dx = x[i] - px; val dy = y[i] - py
                val d2 = dx * dx + dy * dy
                if (d2 <= r2) { hits.add(i); dist.add(d2) }
            }
        }
        if (hits.isEmpty()) return nearest(px, py)
        if (hits.size <= k) return hits.toIntArray()
        val order = hits.indices.sortedBy { dist[it] }
        return IntArray(k) { hits[order[it]] }
    }

    /** Fallback when nothing is in range: the single closest point, scanned. */
    private fun nearest(px: Double, py: Double): IntArray {
        var best = -1; var bd = Double.MAX_VALUE
        for (i in 0 until size) {
            val dx = x[i] - px; val dy = y[i] - py
            val d = dx * dx + dy * dy
            if (d < bd) { bd = d; best = i }
        }
        return if (best < 0) IntArray(0) else intArrayOf(best)
    }

    fun toXy(lat: Double, lon: Double): DoubleArray = doubleArrayOf(
        Math.toRadians(lon - lon0) * cos(Math.toRadians(lat0)) * R_EARTH,
        Math.toRadians(lat - lat0) * R_EARTH,
    )

    fun toLatLon(px: Double, py: Double): DoubleArray = doubleArrayOf(
        lat0 + Math.toDegrees(py / R_EARTH),
        lon0 + Math.toDegrees(px / (R_EARTH * cos(Math.toRadians(lat0)))),
    )

    /** True when (lat, lon) falls inside the exported extract's footprint. */
    fun covers(lat: Double, lon: Double, marginM: Double = 500.0): Boolean {
        val p = toXy(lat, lon)
        return p[0] >= minX - marginM && p[0] <= x.max() + marginM &&
               p[1] >= minY - marginM && p[1] <= y.max() + marginM
    }

    fun dist(i: Int, px: Double, py: Double) = hypot(x[i] - px, y[i] - py)
}
