"""Densify a road network into a flat binary the Android matcher can mmap.

The phone does not parse OSM. All the expensive work -- selecting drivable
ways, projecting to a local metric frame, resampling to a fixed step and
precomputing segment bearings -- happens here, so RoadNetwork.kt only has to
read three float arrays and build a grid index.

Layout (little-endian):
  magic "IDRR" | int32 n | float64 lat0 | float64 lon0 | float32 step_m
  then n * (float32 x, float32 y, float32 bearing_rad)
"""
import argparse, struct
from pathlib import Path
import numpy as np
from osm_pbf import roads_from_pbf
from map_match import RoadNetwork

ap = argparse.ArgumentParser()
ap.add_argument("--pbf", required=True)
ap.add_argument("--bbox", required=True, help="lat_min,lat_max,lon_min,lon_max")
ap.add_argument("--out", required=True)
ap.add_argument("--step", type=float, default=8.0)
a = ap.parse_args()

bb = [float(v) for v in a.bbox.split(",")]
ways = roads_from_pbf(a.pbf, *bb)
lat0 = 0.5 * (bb[0] + bb[1]); lon0 = 0.5 * (bb[2] + bb[3])
net = RoadNetwork(ways, lat0, lon0, step_m=a.step)
n = len(net.x)
print(f"  {len(ways)} ways -> {n} densified points at {a.step} m")

buf = bytearray(b"IDRR")
buf += struct.pack("<i", n) + struct.pack("<dd", lat0, lon0) + struct.pack("<f", a.step)
buf += np.column_stack([net.x, net.y, net.b]).astype("<f4").tobytes()
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_bytes(buf)
print(f"  wrote {a.out}  ({len(buf)/1024:.0f} KB)")
