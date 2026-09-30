package com.teamua.idr

import android.content.Context
import android.os.Environment
import java.io.BufferedWriter
import java.io.File
import java.io.FileWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Writes a capture in the SAME 24-column layout as the existing trip files and
 * IO-VNBD's S-*.csv, so a recording made here drops straight into the training
 * pipeline with no conversion:
 *
 *     python src/check_recording.py  data/raw/tripN.csv
 *     python src/preprocess_speed2.py --input data/raw/tripN.csv --gps-truth ...
 *
 * Rows are written at the NATIVE sensor rate (typically 50-100 Hz), matching
 * how the existing trips were captured; the 10 Hz resampling that the model
 * needs happens later in preprocessing, not here. GPS columns carry the most
 * recent fix, which is what the loaders expect -- they detect genuine fixes by
 * looking for a CHANGE in latitude/longitude.
 *
 * NOTE on units: this column is labelled "GPS SPEED (Kmh)" and this writer
 * really does store km/h. IO-VNBD's own files store m/s under that same
 * heading, which caused three separate bugs in this project; the preprocessing
 * auto-detects per file, so both are handled, but ours are honestly labelled.
 */
class Recorder(private val ctx: Context) {

    private var writer: BufferedWriter? = null
    private var t0 = 0L
    var rows = 0; private set
    var file: File? = null; private set
    val active: Boolean get() = writer != null

    private val stamp = SimpleDateFormat("yyyy-MM-dd HH-mm-ss_SSS", Locale.US)

    private val header = listOf(
        "GPS LATITUDE (degrees)", " GPS LONGITUDE (degrees)", " GPS ALTITUDE (m)",
        " GPS SPEED (Kmh)", " GPS ACCURACY (m)", " GPS ORIENTATION (°)",
        "GPS SATELLITES IN RANGE", " TIME SINCE START (ms)",
        " DATE (YYYY-MO-DD HH-MI-SS_SSS)",
        " ACCELEROMETER X (m/s²) ", " ACCELEROMETER Y (m/s²)", " ACCELEROMETER Z (m/s²)",
        " GRAVITY X (m/s²)", " GRAVITY Y (m/s²)", " GRAVITY Z (m/s²)",
        " GYROSCOPE Yaw (rad/s)", " GYROSCOPE Pitch (rad/s)", " GYROSCOPE Roll (rad/s)",
        " MAGNETIC FIELD X (μT)", " MAGNETIC FIELD Y (μT)", " MAGNETIC FIELD Z (μT)",
        " ORIENTATION (Yaw) (°)", " ORIENTATION (Pitch) (°)", " ORIENTATION (Roll ) (°)"
    ).joinToString(",")

    fun start(): File {
        stop()
        val dir = ctx.getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS)
            ?: ctx.filesDir
        dir.mkdirs()
        val f = File(dir, "trip_${System.currentTimeMillis()}.csv")
        writer = BufferedWriter(FileWriter(f), 1 shl 16)
        writer!!.write(header); writer!!.newLine()
        t0 = System.currentTimeMillis(); rows = 0; file = f
        return f
    }

    fun stop(): File? {
        writer?.flush(); writer?.close(); writer = null
        return file
    }

    /**
     * One row at the native sensor rate.
     * @param gpsSpeedKmh most recent fix speed, km/h (0 if no fix yet)
     */
    fun row(
        lat: Double, lon: Double, alt: Double, gpsSpeedKmh: Double, acc: Double,
        bearingDeg: Double, sats: Int,
        a: FloatArray, g: FloatArray, w: FloatArray, mag: FloatArray, ori: FloatArray
    ) {
        val bw = writer ?: return
        val t = System.currentTimeMillis() - t0
        val sb = StringBuilder(220)
        sb.append(fmt(lat, 8)).append(',').append(fmt(lon, 8)).append(',')
          .append(fmt(alt, 2)).append(',').append(fmt(gpsSpeedKmh, 3)).append(',')
          .append(fmt(acc, 2)).append(',').append(fmt(bearingDeg, 2)).append(',')
          .append(sats).append(',').append(t).append(',')
          .append(stamp.format(Date())).append(',')
        for (v in floatArrayOf(a[0], a[1], a[2], g[0], g[1], g[2],
                               w[0], w[1], w[2], mag[0], mag[1], mag[2],
                               ori[0], ori[1], ori[2])) {
            sb.append(fmt(v.toDouble(), 6)).append(',')
        }
        sb.setLength(sb.length - 1)
        bw.write(sb.toString()); bw.newLine()
        rows++
    }

    private fun fmt(v: Double, dp: Int): String =
        if (v.isNaN() || v.isInfinite()) "0" else String.format(Locale.US, "%.${dp}f", v)
}
