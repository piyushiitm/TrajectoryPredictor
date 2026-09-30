package com.teamua.idr

import android.content.Context
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

/**
 * TFLite wrapper for the anchored speed head.
 *
 * The network predicts the RESIDUAL from the speed known when GNSS dropped,
 * not the absolute speed -- predicting absolute was measured at MAE 4.65 m/s
 * against 2.69 for the residual form. So v0 and the elapsed time are inputs,
 * and the output is added back to v0.
 *
 * 92 inputs: 30 window statistics at each of 2 s / 5 s / 10 s, then v0 and
 * elapsed. Order is positional and must match norm_stats.json.
 */
class SpeedModel(ctx: Context, asset: String = "speed_tuned") {

    private val interpreter: Interpreter
    private val mean: DoubleArray
    private val std: DoubleArray
    val nInputs: Int
    private val input: ByteBuffer
    private val output = Array(1) { FloatArray(1) }

    init {
        val fd = ctx.assets.openFd("$asset.tflite")
        val model = FileInputStream(fd.fileDescriptor).channel.map(
            FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
        interpreter = Interpreter(model, Interpreter.Options().apply { numThreads = 2 })

        val js = JSONObject(ctx.assets.open("${asset}_norm.json").bufferedReader().readText())
        val m = js.getJSONArray("mean"); val s = js.getJSONArray("std")
        nInputs = m.length()
        mean = DoubleArray(nInputs) { m.getDouble(it) }
        std = DoubleArray(nInputs) { s.getDouble(it) }
        input = ByteBuffer.allocateDirect(4 * nInputs).order(ByteOrder.nativeOrder())
    }

    /**
     * @param feats 90 window statistics (3 scales x 30)
     * @param v0 speed at the anchor, m/s
     * @param elapsed seconds since the anchor
     * @return predicted speed, clamped at zero
     */
    fun predict(feats: DoubleArray, v0: Double, elapsed: Double): Double {
        input.rewind()
        for (i in 0 until nInputs) {
            val raw = when {
                i < feats.size -> feats[i]
                i == feats.size -> v0
                else -> elapsed
            }
            input.putFloat(((raw - mean[i]) / std[i]).toFloat())
        }
        input.rewind()
        interpreter.run(input, output)
        return (v0 + output[0][0]).coerceAtLeast(0.0)
    }

    fun close() = interpreter.close()
}
