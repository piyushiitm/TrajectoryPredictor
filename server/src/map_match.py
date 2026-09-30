"""
HMM map matching (Newson & Krumm 2009) over a dead-reckoned trajectory.

Why this exists. The 60s error budget splits almost evenly: ~8.1% cross-track
(heading) and ~8.9% along-track (speed). Every model-side attempt to reduce
either has failed, but the cross-track half is exactly what a road network
constrains -- a vehicle is on a road, and roads run in specific directions. The
PS names this explicitly: "use the road layout as a constraint... snap the
drifting IMU path back onto the actual road grid."

Method
------
States   : candidate projections of each DR point onto nearby road segments.
Emission : Gaussian in the perpendicular distance from the DR point to the
           candidate, with sigma set by the expected DR error at that elapsed
           time (it grows through the outage, so the trust in the DR position
           shrinks accordingly).
Transition: exponential in |great-circle distance between candidates minus
           distance travelled along the DR path|. This is what rejects jumps
           to a parallel road: getting there would require travelling further
           than the odometry says.
Decoding : Viterbi over the whole outage window, so a locally tempting wrong
           road is rejected if it cannot be reached consistently later.

Heading agreement is folded into the emission term: a candidate segment whose
bearing disagrees with the DR heading is penalised, which is what disambiguates
the two carriageways of a divided road.
"""
import numpy as np
from scipy.spatial import cKDTree

R_EARTH = 6_371_000.0


def _to_xy(lat, lon, lat0, lon0):
    x = np.deg2rad(lon - lon0) * np.cos(np.deg2rad(lat0)) * R_EARTH
    y = np.deg2rad(lat - lat0) * R_EARTH
    return x, y


class RoadNetwork:
    """Road polylines resampled to a dense point set for candidate lookup."""

    def __init__(self, ways, lat0, lon0, step_m=8.0):
        self.lat0, self.lon0 = lat0, lon0
        px, py, pb = [], [], []
        for w in ways:
            if len(w) < 2:
                continue
            x, y = _to_xy(w[:, 0], w[:, 1], lat0, lon0)
            seg = np.hypot(np.diff(x), np.diff(y))
            for i in range(len(seg)):
                if seg[i] < 1e-6:
                    continue
                n = max(1, int(seg[i] // step_m))
                t = np.linspace(0, 1, n + 1)[:-1]
                px.append(x[i] + t * (x[i + 1] - x[i]))
                py.append(y[i] + t * (y[i + 1] - y[i]))
                # segment bearing, compass sense (0 = north, clockwise)
                b = np.arctan2(x[i + 1] - x[i], y[i + 1] - y[i])
                pb.append(np.full(n, b))
        if not px:
            raise ValueError("no usable road geometry")
        self.x = np.concatenate(px); self.y = np.concatenate(py)
        self.b = np.concatenate(pb)
        self.tree = cKDTree(np.column_stack([self.x, self.y]))

    def candidates(self, x, y, radius, k=8):
        idx = self.tree.query_ball_point([x, y], r=radius)
        if not idx:
            d, i = self.tree.query([x, y], k=1)
            return np.array([i])
        idx = np.array(idx)
        if len(idx) > k:                       # keep the k nearest
            d = np.hypot(self.x[idx] - x, self.y[idx] - y)
            idx = idx[np.argsort(d)[:k]]
        return idx


def match(net, dr_lat, dr_lon, dr_hdg, elapsed_s,
          sigma0=8.0, sigma_rate=0.12, beta=12.0, hdg_weight=18.0,
          max_radius=120.0, stride=10, diagnostics=False):
    """Snap a dead-reckoned track to the network. Returns (lat, lon) arrays.

    sigma0/sigma_rate set how fast trust in the DR position decays:
    sigma = sigma0 + sigma_rate * elapsed, i.e. the search widens as the
    dead reckoning drifts.
    """
    x, y = _to_xy(dr_lat, dr_lon, net.lat0, net.lon0)
    sel = np.arange(0, len(x), stride)
    if len(sel) < 3:
        return dr_lat, dr_lon
    xs, ys, hs, ts = x[sel], y[sel], dr_hdg[sel], elapsed_s[sel]

    # travelled distance along the DR path between successive selected points
    step_d = np.hypot(np.diff(xs), np.diff(ys))

    cands, emis = [], []
    for i in range(len(sel)):
        sig = min(sigma0 + sigma_rate * ts[i], max_radius)
        c = net.candidates(xs[i], ys[i], radius=max(sig * 2.5, 25.0))
        d = np.hypot(net.x[c] - xs[i], net.y[c] - ys[i])
        # bearing disagreement, folded in as an additional cost
        dh = np.abs(np.angle(np.exp(1j * (net.b[c] - hs[i]))))
        dh = np.minimum(dh, np.pi - dh)        # roads are bidirectional
        cands.append(c)
        emis.append(0.5 * (d / sig) ** 2 + hdg_weight * (dh / np.pi) ** 2)

    # Viterbi
    n = len(cands)
    cost = [np.asarray(emis[0], float)]
    back = []
    for i in range(1, n):
        prev, cur = cands[i - 1], cands[i]
        gap = np.hypot(net.x[cur][None, :] - net.x[prev][:, None],
                       net.y[cur][None, :] - net.y[prev][:, None])
        trans = np.abs(gap - step_d[i - 1]) / beta
        tot = cost[-1][:, None] + trans
        b = np.argmin(tot, 0)
        back.append(b)
        cost.append(tot[b, np.arange(len(cur))] + emis[i])

    path = np.empty(n, int)
    path[-1] = int(np.argmin(cost[-1]))
    for i in range(n - 2, -1, -1):
        path[i] = back[i][path[i + 1]]
    mx = np.array([net.x[cands[i][path[i]]] for i in range(n)])
    my = np.array([net.y[cands[i][path[i]]] for i in range(n)])

    # interpolate the snapped track back onto every sample
    fx = np.interp(np.arange(len(x)), sel, mx)
    fy = np.interp(np.arange(len(x)), sel, my)
    lat = net.lat0 + np.rad2deg(fy / R_EARTH)
    lon = net.lon0 + np.rad2deg(fx / (R_EARTH * np.cos(np.deg2rad(net.lat0))))
    if not diagnostics:
        return lat, lon

    # Signals a runtime gate can use to decide whether to TRUST this match.
    # Map matching is bimodal: a correct snap removes cross-track error, a
    # wrong snap adds more than it removes. None of these need ground truth.
    best = float(np.min(cost[-1]))
    srt = np.sort(cost[-1])
    margin = float(srt[1] - srt[0]) if len(srt) > 1 else 0.0
    off = np.hypot(mx - xs, my - ys)              # how far the snap moved us
    dens = float(np.mean([len(c) for c in cands]))  # candidate roads per point
    diag = {
        "viterbi_cost": best / max(n, 1),
        "viterbi_margin": margin,                  # 2nd-best minus best
        "n_states": dens,                          # local road density
        "snap_mean_m": float(off.mean()),
        "snap_max_m": float(off.max()),
        "snap_end_m": float(off[-1]),
        "path_len_m": float(np.sum(step_d)),
        "elapsed_s": float(elapsed_s[-1]),
    }
    return lat, lon, diag
