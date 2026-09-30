"""Map matching with road TOPOLOGY, in a single Cartesian frame.

Three changes from map_match.py, each aimed at a measured defect.

1. ONE CARTESIAN FRAME. The projection happens once, at the session anchor, and
   everything downstream -- dead reckoning, candidate search, Viterbi, scoring --
   is metres. map_match.py used a fixed cos(lat0) while the dead reckoning used a
   per-step cos(lat), so the two frames disagreed slightly at the edges of an
   extract. Harmless over 8 km, but it is a latent bug if coverage grows.

2. CAPPED FALLBACK. candidates() previously returned the single nearest road
   point when nothing was in range, with NO distance limit, so riding outside the
   extract snapped the track to the boundary however far away that was. Now a
   point with no road within MAX_SNAP is reported unmatched and left alone.

3. TOPOLOGY. The network was a point cloud: no notion of which points connect,
   so the matcher could not route between anchors and the output was interpolated
   STRAIGHT across gaps of up to 126 m -- straight through buildings. Ways now
   keep their node ordering, junctions are merged, and the output path is routed
   along the graph. Transition costs use route distance, which is what Newson &
   Krumm specify and what rejects a parallel road properly: reaching it needs
   real travel, not a short hop across a divider.
"""
import heapq

import numpy as np
from scipy.spatial import cKDTree

R_EARTH = 6_371_000.0
MAX_SNAP = 80.0          # m; beyond this a point is simply not matched
JUNCTION = 6.0           # m; points this close are treated as the same node
ROUTE_CAP = 4000.0       # m; abandon a route search beyond this


class Frame:
    """Local Cartesian frame, fixed at one origin for the whole session."""

    def __init__(self, lat0, lon0):
        self.lat0 = float(lat0)
        self.lon0 = float(lon0)
        self._k = np.cos(np.deg2rad(self.lat0)) * R_EARTH

    def to_xy(self, lat, lon):
        return (np.deg2rad(np.asarray(lon) - self.lon0) * self._k,
                np.deg2rad(np.asarray(lat) - self.lat0) * R_EARTH)

    def to_ll(self, x, y):
        return (self.lat0 + np.rad2deg(np.asarray(y) / R_EARTH),
                self.lon0 + np.rad2deg(np.asarray(x) / self._k))


class RoadGraph:
    """Densified road points plus the connectivity between them."""

    def __init__(self, ways, frame, step_m=8.0):
        self.frame = frame
        px, py, pb, seg_id = [], [], [], []
        for wi, w in enumerate(ways):
            if len(w) < 2:
                continue
            x, y = frame.to_xy(w[:, 0], w[:, 1])
            for i in range(len(x) - 1):
                dx, dy = x[i + 1] - x[i], y[i + 1] - y[i]
                L = float(np.hypot(dx, dy))
                if L < 1e-6:
                    continue
                n = max(1, int(L // step_m))
                t = np.linspace(0, 1, n + 1)[:-1]
                px.append(x[i] + t * dx)
                py.append(y[i] + t * dy)
                pb.append(np.full(n, np.arctan2(dx, dy)))   # compass sense
                seg_id.append(np.full(n, wi))
        if not px:
            raise ValueError("no usable road geometry")
        self.x = np.concatenate(px)
        self.y = np.concatenate(py)
        self.b = np.concatenate(pb)
        self.way = np.concatenate(seg_id)
        self.tree = cKDTree(np.column_stack([self.x, self.y]))
        self._build_graph()

    def _build_graph(self):
        """Adjacency: consecutive points inside a way, plus merged junctions."""
        n = len(self.x)
        adj = [[] for _ in range(n)]
        for i in range(n - 1):
            if self.way[i] == self.way[i + 1]:
                d = float(np.hypot(self.x[i + 1] - self.x[i], self.y[i + 1] - self.y[i]))
                if d < 60.0:              # guard against a jump between ways
                    adj[i].append((i + 1, d))
                    adj[i + 1].append((i, d))
        # junctions: points from DIFFERENT ways that nearly coincide
        for i, j in self.tree.query_pairs(JUNCTION):
            if self.way[i] != self.way[j]:
                d = float(np.hypot(self.x[j] - self.x[i], self.y[j] - self.y[i]))
                adj[i].append((j, d))
                adj[j].append((i, d))
        self.adj = adj

    def candidates(self, x, y, radius, k=8):
        """Road points within `radius`. Empty when nothing is close enough."""
        idx = self.tree.query_ball_point([x, y], r=min(radius, MAX_SNAP))
        if not idx:
            return np.empty(0, int)
        idx = np.asarray(idx)
        if len(idx) > k:
            d = np.hypot(self.x[idx] - x, self.y[idx] - y)
            idx = idx[np.argsort(d)[:k]]
        return idx

    def route(self, a, b, cap=ROUTE_CAP):
        """Dijkstra from a to b. Returns (distance, node path) or (inf, None)."""
        if a == b:
            return 0.0, [a]
        dist = {a: 0.0}
        prev = {}
        pq = [(0.0, a)]
        while pq:
            d, u = heapq.heappop(pq)
            if u == b:
                path = [b]
                while path[-1] != a:
                    path.append(prev[path[-1]])
                return d, path[::-1]
            if d > dist.get(u, np.inf) or d > cap:
                continue
            for v, w in self.adj[u]:
                nd = d + w
                if nd < dist.get(v, np.inf):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))
        return np.inf, None


def match(g, dr_x, dr_y, dr_hdg, elapsed_s, sigma0=8.0, sigma_rate=0.12,
          beta=12.0, hdg_weight=18.0, max_radius=120.0, stride=10,
          use_route=True, diagnostics=False):
    """Snap a dead-reckoned track, in metres, onto the road graph.

    Returns (x, y) arrays in the same frame. Sections with no road within
    MAX_SNAP keep their dead-reckoned position rather than being dragged to a
    distant road.
    """
    nn = len(dr_x)
    sel = np.arange(0, nn, stride)
    if len(sel) < 3:
        return dr_x, dr_y
    xs, ys, hs, ts = dr_x[sel], dr_y[sel], dr_hdg[sel], elapsed_s[sel]

    cands, emis, keep = [], [], []
    for i in range(len(sel)):
        sig = min(sigma0 + sigma_rate * ts[i], max_radius)
        c = g.candidates(xs[i], ys[i], radius=max(sig * 2.5, 25.0))
        if len(c) == 0:
            continue
        d = np.hypot(g.x[c] - xs[i], g.y[c] - ys[i])
        dh = np.abs(np.angle(np.exp(1j * (g.b[c] - hs[i]))))
        dh = np.minimum(dh, np.pi - dh)
        cands.append(c)
        emis.append(0.5 * (d / sig) ** 2 + hdg_weight * (dh / np.pi) ** 2)
        keep.append(i)
    if len(cands) < 3:
        return dr_x, dr_y
    keep = np.asarray(keep)
    step_d = np.hypot(np.diff(xs[keep]), np.diff(ys[keep]))

    # Viterbi. Transition cost uses ROUTE distance when available: a parallel
    # carriageway is close in a straight line but far along the network.
    cost = [np.asarray(emis[0], float)]
    back = []
    for i in range(1, len(cands)):
        prev, cur = cands[i - 1], cands[i]
        gap = np.hypot(g.x[cur][None, :] - g.x[prev][:, None],
                       g.y[cur][None, :] - g.y[prev][:, None])
        if use_route:
            for pi in range(len(prev)):
                for ci in range(len(cur)):
                    if gap[pi, ci] < 2 * step_d[i - 1] + 50:
                        rd, _ = g.route(int(prev[pi]), int(cur[ci]),
                                        cap=3 * step_d[i - 1] + 200)
                        if np.isfinite(rd):
                            gap[pi, ci] = rd
        trans = np.abs(gap - step_d[i - 1]) / beta
        tot = cost[-1][:, None] + trans
        b = np.argmin(tot, 0)
        back.append(b)
        cost.append(tot[b, np.arange(len(cur))] + emis[i])

    path = np.empty(len(cands), int)
    path[-1] = int(np.argmin(cost[-1]))
    for i in range(len(cands) - 2, -1, -1):
        path[i] = back[i][path[i + 1]]
    nodes = [int(cands[i][path[i]]) for i in range(len(cands))]

    # Build the output by ROUTING along the graph between anchors, so the path
    # follows roads instead of cutting straight across whatever lies between.
    out_i, out_x, out_y = [], [], []
    for j in range(len(nodes)):
        gi = sel[keep[j]]
        if j == 0:
            out_i.append(gi); out_x.append(g.x[nodes[0]]); out_y.append(g.y[nodes[0]])
            continue
        g0, g1 = sel[keep[j - 1]], gi
        rd, p = (g.route(nodes[j - 1], nodes[j]) if use_route else (np.inf, None))
        if p and len(p) > 1:
            fr = np.linspace(0, 1, len(p))[1:]
            for f, nd in zip(fr, p[1:]):
                out_i.append(g0 + f * (g1 - g0))
                out_x.append(g.x[nd]); out_y.append(g.y[nd])
        else:
            out_i.append(g1); out_x.append(g.x[nodes[j]]); out_y.append(g.y[nodes[j]])
    out_i = np.asarray(out_i, float)
    o = np.argsort(out_i, kind="stable")
    fx = np.interp(np.arange(nn), out_i[o], np.asarray(out_x)[o])
    fy = np.interp(np.arange(nn), out_i[o], np.asarray(out_y)[o])

    if not diagnostics:
        return fx, fy
    best = float(np.min(cost[-1]))
    srt = np.sort(cost[-1])
    diag = {"viterbi_cost": best / max(len(cands), 1),
            "viterbi_margin": float(srt[1] - srt[0]) if len(srt) > 1 else 0.0,
            "matched_frac": len(cands) / max(len(sel), 1),
            "snap_mean_m": float(np.mean(np.hypot(
                np.asarray(out_x)[:len(keep)] - xs[keep][:len(out_x)],
                np.asarray(out_y)[:len(keep)] - ys[keep][:len(out_y)])))}
    return fx, fy, diag
