"""Native probe geometry and stepping, extracted without changing operations.

The original research implementation is probes/blobs/agentenv/device.py.
Source hashes and extraction parity are recorded in handoff/repo_cleanup/.
"""
import numpy as np
import scipy.fft as sfft
from blobkit import genome as G

CTRL_TU = 5.0
MAX_STEP = 1.5
INJ_SIGMA = 2.0
DIL_BOUNDS = (0.5, 3.0)

def lattice_offsets(lattice, n_rings):
    """Node offsets (k,2) [unit spacing, (dy,dx)] for shells 0..n_rings-1
    (ring 0 = center). Canonical order: by ring, then angle.

    square  : Z^2, L1(diamond) shells        -> n_rings=3: 13 nodes
    squareC : Z^2, Linf(Chebyshev) shells    -> n_rings=3: 25 nodes
    hex     : A2 triangular packing shells   -> n_rings=3: 19 nodes
    tri     : honeycomb (3-coordinated), graph-distance shells
              -> n_rings=3: 1+3+6 = 10 nodes
    """
    R = n_rings - 1
    pts = []
    if lattice in ("square", "squareC"):
        for iy in range(-R, R + 1):
            for ix in range(-R, R + 1):
                r = (abs(iy) + abs(ix)) if lattice == "square" else max(abs(iy), abs(ix))
                if r <= R:
                    pts.append((r, float(iy), float(ix)))
    elif lattice == "hex":
        # A2: v1=(0,1), v2=(sqrt3/2, 1/2); shell = hex ring m
        v1 = np.array([0.0, 1.0])
        v2 = np.array([np.sqrt(3) / 2, 0.5])
        seen = set()
        for a in range(-2 * R, 2 * R + 1):
            for b in range(-2 * R, 2 * R + 1):
                m = max(abs(a), abs(b), abs(a + b))   # hex ring index
                if m <= R and (a, b) not in seen:
                    seen.add((a, b))
                    p = a * v1 + b * v2
                    pts.append((m, p[0], p[1]))
    elif lattice == "tri":
        # honeycomb: sites of a 3-coordinated lattice, shells by graph distance
        # build by BFS from center over honeycomb adjacency
        v1 = np.array([0.0, np.sqrt(3)])
        v2 = np.array([1.5, np.sqrt(3) / 2])
        basis = [np.array([0.0, 0.0]), np.array([1.0, 0.0])]
        sites = {}
        for a in range(-2 * R - 2, 2 * R + 3):
            for b in range(-2 * R - 2, 2 * R + 3):
                for s, off in enumerate(basis):
                    p = a * v1 + b * v2 + off
                    sites[(a, b, s)] = p
        # adjacency: sublattice 0 at (a,b) connects to sublattice 1 at
        # (a,b), (a,b-1), (a-1,b)  [within-cell + two neighbor cells]
        def nbrs(key):
            a, b, s = key
            if s == 0:
                return [(a, b, 1), (a, b - 1, 1), (a - 1, b, 1)]
            return [(a, b, 0), (a, b + 1, 0), (a + 1, b, 0)]
        dist = {(0, 0, 0): 0}
        frontier = [(0, 0, 0)]
        for d in range(1, R + 1):
            nxt = []
            for k in frontier:
                for nb in nbrs(k):
                    if nb in sites and nb not in dist:
                        dist[nb] = d
                        nxt.append(nb)
            frontier = nxt
        for k, d in dist.items():
            p = sites[k]
            pts.append((d, p[0], p[1]))
        # normalize: honeycomb NN distance is 1 (unit spacing) already
    else:
        raise ValueError(f"unknown lattice {lattice!r}")
    # canonical order: ring, then angle, then radius
    def keyf(t):
        r, y, x = t
        ang = np.arctan2(y, x) % (2 * np.pi)
        return (r, round(ang, 9), round(np.hypot(y, x), 9))
    pts.sort(key=keyf)
    return np.array([[y, x] for _, y, x in pts], float)


def rot2(theta, reflect=False):
    c, s = np.cos(theta), np.sin(theta)
    Rm = np.array([[c, -s], [s, c]])
    if reflect:
        Rm = Rm @ np.array([[1.0, 0.0], [0.0, -1.0]])
    return Rm


class ProbeDevice:
    """One rigid sensor array. All geometry secret; agent sees streams only."""

    def __init__(self, dev_id, lattice, n_rings, base_ds, center, L,
                 secret_rot, reflect, motion_theta, motion_reflect,
                 node_perm, dil_bounds=DIL_BOUNDS):
        self.dev_id = dev_id
        self.lattice = lattice
        self.n_rings = n_rings
        self.base_ds = float(base_ds)
        self.center = np.array(center, float)   # (y, x) world coords
        self.L = float(L)
        self.secret_rot = float(secret_rot)
        self.reflect = bool(reflect)
        self.Rm = rot2(secret_rot, reflect)     # lattice orientation
        self.Bm = rot2(motion_theta, motion_reflect)  # motion basis (cols)
        self.offs = lattice_offsets(lattice, n_rings)  # canonical (k,2)
        self.k = len(self.offs)
        self.node_perm = np.asarray(node_perm, int)    # stream slot -> node
        self.dilation = 1.0
        self.dil_bounds = dil_bounds

    # geometry (evaluator-side)
    def node_positions(self):
        """World coords (k,2) in CANONICAL node order."""
        world = self.center[None, :] + self.dilation * self.base_ds * \
            (self.offs @ self.Rm.T)
        return world % self.L

    # controls (called by WorldEnv with budget already checked)
    def apply_move(self, a):
        a = np.clip(np.asarray(a, float), -MAX_STEP, MAX_STEP)
        d = self.Bm @ a                          # secret basis
        self.center = (self.center + d) % self.L
        return float(np.abs(a).sum())            # cost in control units

    def apply_dilate(self, dgain):
        dgain = float(np.clip(dgain, -1.0, 1.0))
        old = self.dilation
        self.dilation = float(np.clip(self.dilation * np.exp(dgain),
                                      *self.dil_bounds))
        return abs(np.log(self.dilation / old))  # cost = |log change| actual

    def sample(self, fields, dx):
        """Bilinear sample all port fields at node positions.
        fields: (n_ports, N, N) view in PORT order. Returns (n_ports, k)
        in STREAM order (node_perm applied)."""
        pos = self.node_positions()              # canonical order
        vals = bilinear(fields, pos, dx)         # (n_ports, k)
        inv = np.empty_like(self.node_perm)
        inv[self.node_perm] = np.arange(self.k)
        # stream slot j carries node node_perm[j]
        return vals[:, self.node_perm]


def bilinear(fields, pos, dx):
    """Periodic bilinear sampling. fields (nf,N,N); pos (k,2) world (y,x).
    Grid cell centers at (i+0.5)*dx (program convention)."""
    N = fields.shape[-1]
    gy = pos[:, 0] / dx - 0.5
    gx = pos[:, 1] / dx - 0.5
    i0 = np.floor(gy).astype(int)
    j0 = np.floor(gx).astype(int)
    fy = (gy - i0)[None, :]
    fx = (gx - j0)[None, :]
    i0 %= N; j0 %= N
    i1 = (i0 + 1) % N
    j1 = (j0 + 1) % N
    f = fields.astype(np.float32, copy=False)
    v00 = f[:, i0, j0]; v01 = f[:, i0, j1]
    v10 = f[:, i1, j0]; v11 = f[:, i1, j1]
    return (v00 * (1 - fy) * (1 - fx) + v01 * (1 - fy) * fx
            + v10 * fy * (1 - fx) + v11 * fy * fx)


def step_chunk(S, n_steps, injections=None):
    """Advance sim state S by n_steps WITHOUT recording. Op order verbatim
    from blobkit.soup.sim_cpu.advance (parity-gated bitwise in tests).
    injections: list of dicts(field=int, y=float, x=float, amp=float)
    applied as a source term amp*gauss per TIME UNIT via genome.poke."""
    na, N = S["na"], S["N"]
    fdt, rng, workers = S["fdt"], S["rng"], S["workers"]
    F, E = S["F"], S["E"]
    lam, k1, u0f = S["lam"], S["k1"], S["u0f"]
    Wf, Kf, Wid = S["Wf"], S["Kf"], S["Wid"]
    thr_f, sc_f, inv_tau = S["thr_f"], S["sc_f"], S["inv_tau"]
    bilin, tanh_rows = S["bilin"], S["tanh_rows"]
    nsig, noise = S["nsig"], S["noise"]
    dt = S["dt"]
    g = S["g"]; dx = S["dx"]
    injections = injections or []

    for _ in range(n_steps):
        # source injections (perturbation; RNG stream untouched)
        for inj in injections:
            F = G.poke(F, g, inj["field"], inj["x"], inj["y"],
                       inj["amp"] * dt, INJ_SIGMA, dx)
        U = F[:na]; X = F[na:]
        Z = U - u0f
        R = np.empty_like(F)
        np.multiply(U, U, out=R[:na]); R[:na] *= -U
        R[:na] += lam * U; R[:na] += k1
        R[:na] -= np.tensordot(Kf, X, axes=(1, 0))
        for (i, c, c2, coef) in bilin:
            R[i] -= fdt(coef) * X[c] * X[c2]
        Rch = np.tensordot(Wid, Z, axes=(1, 0))
        for c in tanh_rows:
            acc = None
            for a in range(na):
                if Wf[c, a] != 0.0:
                    v = np.tanh(np.clip(Z[a] - thr_f[c], 0, None) / sc_f[c])
                    v *= Wf[c, a]
                    acc = v if acc is None else acc + v
            if acc is not None:
                Rch[c] = acc
        Rch -= X; Rch *= inv_tau
        R[na:] = Rch
        F = F + fdt(dt) * R
        if noise > 0:
            F[:na] += nsig * rng.standard_normal((na, N, N), dtype=fdt) \
                if fdt == np.float32 else \
                nsig * rng.standard_normal((na, N, N))
        F = sfft.irfft2(sfft.rfft2(F, workers=workers) * E, s=(N, N),
                        workers=workers)
        S["t_step"] += 1
        S["F"] = F
    return S
