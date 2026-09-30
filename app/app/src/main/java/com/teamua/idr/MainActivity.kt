package com.teamua.idr

import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.os.Bundle
import android.os.Looper
import android.preference.PreferenceManager
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.google.android.gms.location.*
import com.teamua.idr.databinding.ActivityMainBinding
import org.osmdroid.config.Configuration
import org.osmdroid.tileprovider.tilesource.TileSourceFactory
import org.osmdroid.util.GeoPoint
import org.osmdroid.views.overlay.Marker
import org.osmdroid.views.overlay.Polyline
import java.util.Locale

/**
 * Live side-by-side comparison: where GNSS says the vehicle is, and where the
 * IMU dead reckoning thinks it is after GNSS was withheld.
 *
 * Press "Simulate GNSS outage" and the engine stops receiving position, speed
 * and bearing; the red marker then coasts on the IMU alone while the green one
 * keeps following the real fix. The gap between them IS the drift the
 * benchmark measures.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var b: ActivityMainBinding
    private lateinit var engine: IdrEngine
    private lateinit var sensors: SensorHub
    private lateinit var fused: FusedLocationProviderClient
    private lateinit var recorder: Recorder

    // latest fix, held so every recorded row carries the current GNSS state
    private var fLat = 0.0; private var fLon = 0.0; private var fAlt = 0.0
    private var fSpdKmh = 0.0; private var fAcc = 0.0; private var fBear = 0.0
    private var fSats = 0

    private lateinit var gpsMarker: Marker
    private lateinit var drMarker: Marker
    private lateinit var holdMarker: Marker
    private lateinit var mmMarker: Marker
    private lateinit var gyroMarker: Marker
    private val gpsTrack = Polyline()
    private val drTrack = Polyline()
    private val mmTrack = Polyline()
    private val gyroTrack = Polyline()

    private var t0 = 0L
    private var centred = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        Configuration.getInstance().load(this,
            PreferenceManager.getDefaultSharedPreferences(this))
        Configuration.getInstance().userAgentValue = packageName

        b = ActivityMainBinding.inflate(layoutInflater)
        setContentView(b.root)

        b.map.setTileSource(TileSourceFactory.MAPNIK)
        b.map.setMultiTouchControls(true)
        b.map.controller.setZoom(18.0)

        gpsTrack.outlinePaint.strokeWidth = 8f
        gpsTrack.outlinePaint.color = 0xFF2ECC71.toInt()
        drTrack.outlinePaint.strokeWidth = 8f
        drTrack.outlinePaint.color = 0xFFE74C3C.toInt()
        mmTrack.outlinePaint.strokeWidth = 8f
        mmTrack.outlinePaint.color = 0xFF16A085.toInt()      // teal, map-matched
        // purple: TRUE GPS bearing with the AI speed. The gap between purple and
        // green is speed error; the gap between red and purple is heading error.
        gyroTrack.outlinePaint.strokeWidth = 7f
        gyroTrack.outlinePaint.color = 0xFF9B59B6.toInt()
        b.map.overlays.add(gpsTrack); b.map.overlays.add(drTrack)
        b.map.overlays.add(mmTrack); b.map.overlays.add(gyroTrack)

        // while GNSS is healthy all three coincide, so they are drawn at
        // decreasing sizes to stay individually visible when stacked
        gpsMarker = mk(0xFF2ECC71.toInt(), "GNSS", 54)
        drMarker = mk(0xFFE74C3C.toInt(), "IMU dead reckoning", 36)
        holdMarker = mk(0xFF3498DB.toInt(), "hold-speed", 22)
        mmMarker = mk(0xFF16A085.toInt(), "map-matched", 44)
        gyroMarker = mk(0xFF9B59B6.toInt(), "GPS bearing + AI speed", 30)
        b.map.overlays.addAll(listOf(gpsMarker, drMarker, holdMarker, mmMarker,
                                     gyroMarker))

        engine = IdrEngine(this)
        recorder = Recorder(this)
        sensors = SensorHub(this) { s -> engine.onSample(s); render() }
        // native-rate callback writes the capture in IO-VNBD column order
        sensors.onRaw = { a, g, w, mag, ori ->
            if (recorder.active) recorder.row(fLat, fLon, fAlt, fSpdKmh, fAcc,
                fBear, fSats, a, g, w, mag, ori)
        }
        fused = LocationServices.getFusedLocationProviderClient(this)

        b.btnOutage.setOnClickListener {
            if (engine.outage) {
                engine.stopOutage()
                mmTrack.setPoints(emptyList()); gyroTrack.setPoints(emptyList())
                b.btnOutage.text = "Outage"
            } else {
                engine.startOutage(now())
                b.btnOutage.text = "Restore"
                drTrack.setPoints(emptyList())
            }
        }
        b.btnRecord.setOnClickListener {
            if (recorder.active) {
                val f = recorder.stop()
                b.btnRecord.text = "Record"
                android.widget.Toast.makeText(this,
                    "saved ${f?.name}  (${recorder.rows} rows)\n${f?.parent}",
                    android.widget.Toast.LENGTH_LONG).show()
            } else {
                val f = recorder.start()
                b.btnRecord.text = "Stop"
                android.widget.Toast.makeText(this, "recording ${f.name}",
                    android.widget.Toast.LENGTH_SHORT).show()
            }
        }
        b.btnHeading.setOnClickListener {
            // cycle GYRO -> MAG -> FUSED. At 10 s these are within half a point
            // of each other; the difference shows up on long outages.
            engine.headingMode = when (engine.headingMode) {
                IdrEngine.Heading.GYRO -> IdrEngine.Heading.MAG
                IdrEngine.Heading.MAG -> IdrEngine.Heading.FUSED
                IdrEngine.Heading.FUSED -> IdrEngine.Heading.ML_GYRO
                else -> IdrEngine.Heading.GYRO
            }
            b.btnHeading.text = "Hdg: " + engine.headingMode.name
        }

        // Map matching is not exposed: measured on 9 recordings it made things
        // WORSE at short outages (3.2 m of dead-reckoning error became 5.7 m,
        // because the DR is already finer than the road network's 8 m
        // granularity) and did nothing at long ones (63.2 m vs 62.7 m). The code
        // stays in the build; the button is more useful as the follow control.
        b.btnMap.text = "Follow: ON"
        b.btnMap.setOnClickListener {
            follow = !follow
            b.btnMap.text = if (follow) "Follow: ON" else "Follow: off"
            if (follow && engine.haveGps)
                b.map.controller.animateTo(GeoPoint(engine.gpsLat, engine.gpsLon))
        }
        // panning by hand releases the follow, so the map does not fight the rider
        b.map.setOnTouchListener { v, ev ->
            if (ev.action == android.view.MotionEvent.ACTION_MOVE && follow) {
                follow = false
                b.btnMap.text = "Follow: off"
            }
            v.performClick(); false
        }

        b.btnHead.setOnClickListener {
            // TUNED is this bike; GENERAL also saw the IO-VNBD cars and holds up
            // better on a vehicle the model has never met.
            engine.speedHead = if (engine.speedHead == IdrEngine.SpeedHead.TUNED)
                IdrEngine.SpeedHead.GENERAL else IdrEngine.SpeedHead.TUNED
            b.btnHead.text = if (engine.speedHead == IdrEngine.SpeedHead.TUNED)
                "Model: Tuned" else "Model: General"
        }

        b.btnSource.setOnClickListener {
            engine.useModel = !engine.useModel
            b.btnSource.text = if (engine.useModel) "AI speed" else "Hold speed"
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION)
            != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                arrayOf(Manifest.permission.ACCESS_FINE_LOCATION), 1)
        } else startLocation()
    }

    private fun mk(color: Int, title: String, size: Int = 36): Marker {
        val m = Marker(b.map)
        m.title = title
        m.setAnchor(Marker.ANCHOR_CENTER, Marker.ANCHOR_CENTER)
        m.icon = android.graphics.drawable.ShapeDrawable(
            android.graphics.drawable.shapes.OvalShape()).apply {
            intrinsicWidth = size; intrinsicHeight = size; paint.color = color
        }
        return m
    }

    private fun now(): Double {
        if (t0 == 0L) t0 = System.nanoTime()
        return (System.nanoTime() - t0) / 1e9
    }

    override fun onRequestPermissionsResult(rc: Int, p: Array<out String>, r: IntArray) {
        super.onRequestPermissionsResult(rc, p, r)
        if (r.isNotEmpty() && r[0] == PackageManager.PERMISSION_GRANTED) startLocation()
    }

    @SuppressLint("MissingPermission")
    private fun startLocation() {
        val req = LocationRequest.Builder(Priority.PRIORITY_HIGH_ACCURACY, 1000)
            .setMinUpdateIntervalMillis(1000).build()
        fused.requestLocationUpdates(req, object : LocationCallback() {
            override fun onLocationResult(res: LocationResult) {
                val l = res.lastLocation ?: return
                fLat = l.latitude; fLon = l.longitude; fAlt = l.altitude
                fSpdKmh = (if (l.hasSpeed()) l.speed.toDouble() else 0.0) * 3.6
                fAcc = l.accuracy.toDouble()
                fBear = if (l.hasBearing()) l.bearing.toDouble() else 0.0
                fSats = l.extras?.getInt("satellites", 0) ?: 0
                engine.onGps(l.latitude, l.longitude,
                    if (l.hasSpeed()) l.speed.toDouble() else 0.0,
                    if (l.hasBearing()) l.bearing.toDouble() else 0.0, now())
                if (!centred) {
                    b.map.controller.setCenter(GeoPoint(l.latitude, l.longitude))
                    centred = true
                }
                gpsTrack.addPoint(GeoPoint(l.latitude, l.longitude))
            }
        }, Looper.getMainLooper())
        sensors.start()
    }

    private var follow = true
    private var lastRender = 0L
    private fun render() {
        val ms = System.currentTimeMillis()
        if (ms - lastRender < 200) return          // 5 Hz is enough for the eye
        lastRender = ms
        if (!engine.haveGps) return

        gpsMarker.position = GeoPoint(engine.gpsLat, engine.gpsLon)
        // keep the map centred on GNSS unless the rider has panned away; a single
        // touch releases it, and Recentre re-arms it
        if (follow) b.map.controller.setCenter(GeoPoint(engine.gpsLat, engine.gpsLon))
        drMarker.position = GeoPoint(engine.drLat, engine.drLon)
        holdMarker.position = GeoPoint(engine.holdLat, engine.holdLon)
        if (engine.outage) {
            drTrack.addPoint(GeoPoint(engine.drLat, engine.drLon))
            gyroTrack.addPoint(GeoPoint(engine.refLat, engine.refLon))
        }
        gyroMarker.setVisible(engine.outage)
        if (engine.outage) gyroMarker.position = GeoPoint(engine.refLat, engine.refLon)
        // the snapped point only means anything while the gate accepts the match
        mmMarker.setVisible(engine.outage && engine.matchOk)
        if (engine.outage && engine.matchOk) {
            mmMarker.position = GeoPoint(engine.mmLat, engine.mmLon)
            mmTrack.addPoint(GeoPoint(engine.mmLat, engine.mmLon))
        }

        val drift = engine.driftMetres()
        val hold = engine.holdDriftMetres()
        b.stats.text = if (engine.outage) String.format(Locale.US,
            "OUTAGE  %5.1f s\n" +
            "drift    AI %6.1f m   hold %6.1f m\n" +
            "speed    AI %5.1f%s  GNSS %5.1f m/s\n" +
            "heading  %5.0f deg    cal R2 %.2f (n=%d)\n" +
            "map      %s  %s\n" +
            "compass  %s",
            engine.elapsed, drift, hold, engine.drSpeed,
            if (engine.modelOk) "" else "!", engine.gpsSpeed,
            Math.toDegrees(engine.drHeading), engine.calib.r2, engine.calib.nRows,
            if (engine.matchOk) String.format(Locale.US, "%6.1f m", engine.matchDriftMetres())
            else "  --  ", engine.matchInfo,
            if (engine.compass.reliable)
                String.format(Locale.US, "FUSED  |B| sd %.1f uT  tau %.0fs",
                    engine.compass.bStd, engine.compass.tauFor(engine.elapsed))
            else String.format(Locale.US, "off (|B| sd %.1f uT, n=%d)",
                    engine.compass.bStd, engine.compass.samples))
        else String.format(Locale.US,
            "GNSS ACTIVE — tracking\n" +
            "speed %5.1f m/s   bearing %5.0f deg\n" +
            "calibration R2 %.2f  (n=%d)  %s",
            engine.gpsSpeed, Math.toDegrees(engine.gpsBearing),
            engine.calib.r2, engine.calib.nRows,
            if (engine.calib.ready) "ready" else "collecting…") +
            String.format(Locale.US, "\ncompass %s  |B| sd %.1f uT  n=%d  rej=%d",
                if (engine.compass.reliable) "READY" else "calibrating",
                engine.compass.bStd, engine.compass.samples, engine.compass.rejected) +
            if (recorder.active) String.format(Locale.US,
                "\nREC %d rows", recorder.rows) else ""

        val wob = sensors.wobbleDeg; val vib = sensors.vibration
        val verdict = when {
            wob < 2.0 && vib < 0.8 -> "GOOD"
            wob < 3.5 && vib < 1.6 -> "marginal"
            else -> "POOR — damp the mount"
        }
        b.mount.text = String.format(Locale.US,
            "mount: wobble %.2f deg/s   vibration %.2f m/s2   %s", wob, vib, verdict)
        b.map.invalidate()
    }

    override fun onResume() { super.onResume(); b.map.onResume() }
    override fun onPause() { super.onPause(); b.map.onPause() }
    override fun onDestroy() {
        super.onDestroy(); recorder.stop(); sensors.stop(); engine.close()
    }
}
