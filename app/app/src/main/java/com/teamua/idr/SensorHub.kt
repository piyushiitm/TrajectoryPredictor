package com.teamua.idr

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager

/**
 * Collects accelerometer, gravity and gyroscope and republishes them on a
 * fixed 10 Hz grid.
 *
 * The whole pipeline -- window lengths, feature statistics, integration -- was
 * measured at 10 Hz, so the phone's native rate (typically 50-100 Hz, and
 * irregular) is resampled here rather than anywhere downstream. Samples inside
 * a 100 ms bin are averaged, which is also a mild anti-alias.
 */
class SensorHub(ctx: Context, private val onSample: (Sample) -> Unit) : SensorEventListener {

    data class Sample(
        val t: Double,                       // seconds since start
        val lax: Double, val lay: Double, val laz: Double,   // linear accel
        val gx: Double, val gy: Double, val gz: Double,      // gyro rad/s
        val grav: FloatArray,
        val mag: FloatArray                  // uT, for the compass
    )

    private val sm = ctx.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val accel = sm.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
    private val gravS = sm.getDefaultSensor(Sensor.TYPE_GRAVITY)
    private val gyro = sm.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
    private val magS = sm.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD)
    private val rotS = sm.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)

    /** Raw values at the NATIVE rate, for the recorder. Not resampled. */
    var onRaw: ((FloatArray, FloatArray, FloatArray, FloatArray, FloatArray) -> Unit)? = null

    private val a = FloatArray(3)
    private val g = FloatArray(3)
    private val w = FloatArray(3)
    private val mag = FloatArray(3)
    private val ori = FloatArray(3)          // yaw, pitch, roll in degrees
    private val rotM = FloatArray(9)
    private val rotV = FloatArray(5)
    private val orientRad = FloatArray(3)
    private var haveA = false; private var haveG = false; private var haveW = false

    private var t0 = 0L
    private var binIndex = -1L
    private var acc = DoubleArray(6)
    private var count = 0

    /** Peak |linear accel| and gravity-direction wobble, for the mount check. */
    var vibration = 0.0; private set
    var wobbleDeg = 0.0; private set
    private var lastGrav: FloatArray? = null

    fun start() {
        val rate = SensorManager.SENSOR_DELAY_GAME       // ~50 Hz
        accel?.let { sm.registerListener(this, it, rate) }
        gravS?.let { sm.registerListener(this, it, rate) }
        gyro?.let { sm.registerListener(this, it, rate) }
        magS?.let { sm.registerListener(this, it, rate) }
        rotS?.let { sm.registerListener(this, it, rate) }
    }

    fun stop() = sm.unregisterListener(this)

    override fun onAccuracyChanged(s: Sensor?, a: Int) {}

    override fun onSensorChanged(e: SensorEvent) {
        when (e.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> { System.arraycopy(e.values, 0, a, 0, 3); haveA = true }
            Sensor.TYPE_GRAVITY -> { System.arraycopy(e.values, 0, g, 0, 3); haveG = true }
            Sensor.TYPE_GYROSCOPE -> { System.arraycopy(e.values, 0, w, 0, 3); haveW = true }
            Sensor.TYPE_MAGNETIC_FIELD -> System.arraycopy(e.values, 0, mag, 0, 3)
            Sensor.TYPE_ROTATION_VECTOR -> {
                val n = minOf(e.values.size, rotV.size)
                System.arraycopy(e.values, 0, rotV, 0, n)
                SensorManager.getRotationMatrixFromVector(rotM, rotV.copyOf(n))
                SensorManager.getOrientation(rotM, orientRad)
                for (i in 0..2) ori[i] = Math.toDegrees(orientRad[i].toDouble()).toFloat()
            }
        }
        if (!(haveA && haveG && haveW)) return
        onRaw?.invoke(a, g, w, mag, ori)     // native rate, before resampling
        if (t0 == 0L) t0 = e.timestamp
        val t = (e.timestamp - t0) / 1e9
        val bin = (t * 10.0).toLong()

        if (binIndex < 0) binIndex = bin
        if (bin != binIndex) {
            if (count > 0) {
                val c = count.toDouble()
                onSample(Sample(binIndex / 10.0,
                    acc[0] / c, acc[1] / c, acc[2] / c,
                    acc[3] / c, acc[4] / c, acc[5] / c, g.copyOf(), mag.copyOf()))
            }
            binIndex = bin; acc = DoubleArray(6); count = 0
        }
        acc[0] += (a[0] - g[0]).toDouble(); acc[1] += (a[1] - g[1]).toDouble()
        acc[2] += (a[2] - g[2]).toDouble()
        acc[3] += w[0].toDouble(); acc[4] += w[1].toDouble(); acc[5] += w[2].toDouble()
        count++

        val lin = Math.sqrt(((a[0]-g[0])*(a[0]-g[0]) + (a[1]-g[1])*(a[1]-g[1]) +
                             (a[2]-g[2])*(a[2]-g[2])).toDouble())
        vibration = 0.995 * vibration + 0.005 * lin
        lastGrav?.let { p ->
            val n1 = Math.sqrt((p[0]*p[0]+p[1]*p[1]+p[2]*p[2]).toDouble())
            val n2 = Math.sqrt((g[0]*g[0]+g[1]*g[1]+g[2]*g[2]).toDouble())
            if (n1 > 1e-3 && n2 > 1e-3) {
                val dot = ((p[0]*g[0]+p[1]*g[1]+p[2]*g[2]) / (n1*n2)).coerceIn(-1.0, 1.0)
                val deg = Math.toDegrees(Math.acos(dot)) * 50.0   // per-sample -> per-second
                wobbleDeg = 0.99 * wobbleDeg + 0.01 * deg
            }
        }
        lastGrav = g.copyOf()
    }
}
