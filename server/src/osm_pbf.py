"""
Extract a drivable road network from an offline Geofabrik .osm.pbf.

This replaced the Overpass path: the public mirrors either rate-limited us
(HTTP 429) or took 64 s for a trivial query. A regional extract is a one-time
~100-350 MB download, is immune to load, and is the more faithful reading of
the PS requirement for an OFFLINE map database.

Output is cached per bbox as JSON, in the same format `map_match.RoadNetwork`
expects, so the matcher does not care which source produced it.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import osmium

CACHE = Path(__file__).resolve().parent.parent / "data" / "osm"
DRIVABLE = {"motorway", "trunk", "primary", "secondary", "tertiary",
            "unclassified", "residential", "service", "living_street",
            "motorway_link", "trunk_link", "primary_link", "secondary_link",
            "tertiary_link", "road"}


class _Roads(osmium.SimpleHandler):
    def __init__(self, bbox):
        super().__init__()
        self.lat0, self.lon0, self.lat1, self.lon1 = bbox
        self.ways = []

    def way(self, w):
        hw = w.tags.get("highway")
        if hw not in DRIVABLE:
            return
        pts = []
        inside = False
        for n in w.nodes:
            try:
                la, lo = n.lat, n.lon
            except osmium.InvalidLocationError:
                return
            pts.append([la, lo])
            if self.lat0 <= la <= self.lat1 and self.lon0 <= lo <= self.lon1:
                inside = True
        if inside and len(pts) >= 2:
            self.ways.append(pts)


def roads_from_pbf(pbf, lat_min, lat_max, lon_min, lon_max, pad=0.01, verbose=True):
    bbox = (round(lat_min - pad, 4), round(lon_min - pad, 4),
            round(lat_max + pad, 4), round(lon_max + pad, 4))
    CACHE.mkdir(parents=True, exist_ok=True)
    key = hashlib.md5(("%.4f_%.4f_%.4f_%.4f" % bbox).encode()).hexdigest()[:12]
    cf = CACHE / f"roads_{key}.json"
    if cf.exists():
        ways = json.loads(cf.read_text())
        if verbose:
            print(f"  [osm] cache hit: {len(ways)} ways")
        return [np.array(w, float) for w in ways]

    h = _Roads((bbox[0], bbox[1], bbox[2], bbox[3]))
    # locations_on_ways resolves node coordinates without a separate node pass
    h.apply_file(str(pbf), locations=True)
    cf.write_text(json.dumps(h.ways))
    if verbose:
        print(f"  [osm] {Path(pbf).name}: {len(h.ways)} drivable ways in bbox -> {cf.name}")
    return [np.array(w, float) for w in h.ways]


if __name__ == "__main__":
    import sys
    P = Path(__file__).resolve().parent.parent / "data" / "osm" / "pbf"
    targets = {
        "bhopal":   (P / "central-zone.osm.pbf", 23.18, 23.26, 77.39, 77.52),
        "rajshahi": (P / "bangladesh.osm.pbf",   24.38, 24.41, 88.61, 88.65),
    }
    for name, (pbf, *bb) in targets.items():
        if len(sys.argv) > 1 and sys.argv[1] != name:
            continue
        if not pbf.exists():
            print(f"  {name}: {pbf.name} not downloaded yet"); continue
        w = roads_from_pbf(pbf, *bb)
        print(f"  {name}: {len(w)} ways, {sum(len(x) for x in w)} nodes")
