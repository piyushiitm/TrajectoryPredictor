package com.teamua.idr

import android.content.Context
import org.json.JSONObject
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import org.tensorflow.lite.Interpreter

/**
 * Learned correction to the integrated heading.
 *
 * This is the first learned heading component that survived honest validation.
 * Trained on 25 bike recordings and scored leave-one-recording-out -- never on a
 * held-out tail, which shares the ride and flatters everything -- it improved
 * heading by 17.2% with a tails-minus-LORO gap of only 4.0 points. On trip1, a
 * phone, mount and vehicle absent from every training pool, it cut heading error
 * from 24.73 to 20.75 degrees.
 *
 * For contrast, the compass-correction head trained the same way scored -48.5%
 * leave-one-out while looking positive on tails: it had memorised where specific
 * bridges and parked trucks were. That one is not shipped.
 *
 * INPUT ORDER IS POSITIONAL AND MUST MATCH gyroc_norm.json exactly:
 *   [0..89]  the same 90 window statistics the speed head uses
 *   [90]     v0, speed at the anchor
 *   [91]     elapsed seconds since the anchor
 *   [92]     dh_gyro, heading change so far from the calibrated gyro
 *   [93]     dh_fused, heading change so far from the gyro+compass fusion
 *   [94]     |dh_fused|
 *   [95]     the current yaw rate
 * A single misplaced value silently shifts every input after it.
 */
class HeadingModel(ctx: Context) {

    private val interpreter: Interpreter
    private val mean: DoubleArray
    private val std: DoubleArray
    val nInputs: Int
    private val input: ByteBuffer
    private val output = Array(1) { FloatArray(1) }

    init {
        val fd = ctx.assets.openFd("gyroc.tflite")
        val model = FileInputStream(fd.fileDescriptor).channel.map(
            FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
        interpreter = Interpreter(model, Interpreter.Options().apply { numThreads = 2 })
        val js = JSONObject(ctx.assets.open("gyroc_norm.json").bufferedReader().readText())
        val m = js.getJSONArray("mean")
        val s = js.getJSONArray("std")
        nInputs = m.length()
        mean = DoubleArray(nInputs) { m.getDouble(it) }
        std = DoubleArray(nInputs) { s.getDouble(it) }
        input = ByteBuffer.allocateDirect(4 * nInputs).order(ByteOrder.nativeOrder())
    }

    /**
     * @return a correction in radians to ADD to the integrated heading, or 0.0
     *         when the feature vector is the wrong length (safer than guessing).
     */
    fun correct(feats: DoubleArray, v0: Double, elapsed: Double,
                dhGyro: Double, dhFused: Double, yawRate: Double): Double {
        if (feats.size + 6 != nInputs) return 0.0
        input.rewind()
        for (i in 0 until nInputs) {
            val raw = when {
                i < feats.size -> feats[i]
                i == feats.size -> v0
                i == feats.size + 1 -> elapsed
                i == feats.size + 2 -> dhGyro
                i == feats.size + 3 -> dhFused
                i == feats.size + 4 -> Math.abs(dhFused)
                else -> yawRate
            }
            input.putFloat(((raw - mean[i]) / std[i]).toFloat())
        }
        input.rewind()
        interpreter.run(input, output)
        return output[0][0].toDouble()
    }

    fun close() = interpreter.close()
}
