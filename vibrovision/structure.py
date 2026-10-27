"""Output-only mechanical identification with explicit validity and uncertainty.

The mass-normalized stiffness D is invariant to positive diagonal ROI scales.
Row sums of the pixel operator represent mounts only with a shared spatial scale.
Graph connectivity, stiffness and damping are separate quantities. See
`docs/MATHEMATICS.md` for proofs, assumptions and empirical bootstrap limits.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.linalg import expm

from .simulate import FaultSpec, impact_response, severity_profile
from .mechanics import invariant_operator, graph_diagnostics, compare_operators


# ----------------------------------------------------------------------------- model + simulator
@dataclass
class StructFault:
    kind: str                     # "mount" (k_g of one part), "joint" (one coupling), "imbalance" (1x force)
    target: Tuple[str, ...]       # ("Rodamiento_B",) or ("Caja_Engranajes", "Rodamiento_A")
    factor: float = 0.3           # stiffness multiplier at full severity (mount/joint), force multiplier (imbalance)
    onset_s: float = 60.0
    full_s: float = 160.0


@dataclass
class StructureSpec:
    names: Tuple[str, ...] = ("Motor", "Caja_Engranajes", "Rodamiento_A", "Rodamiento_B")
    mass: Tuple[float, ...] = (4.0, 2.5, 1.0, 1.0)                 # relative (only M^-1 K matters)
    k_ground: Tuple[float, ...] = (2.0e4, 1.6e4, 1.0e4, 1.3e4)     # mounts (N/m per unit mass scale)
    couplings: Dict[Tuple[int, int], float] = field(default_factory=lambda: {(0, 1): 2.2e4, (1, 2): 1.0e4,
                                                                             (2, 3): 0.8e4})
    zeta: float = 0.03                                              # modal damping ratio
    axis_stiffness: Tuple[float, float] = (1.0, 1.5)                 # y axis stiffer than x
    response_rms_px: float = 0.20                                    # broadband response at healthy state
    tones: Tuple[Tuple[int, float, float], ...] = ((0, 24.6, 1.0), (0, 49.2, 0.4), (1, 36.9, 0.6))
    faults: Tuple[StructFault, ...] = ()
    bearing: Optional[Tuple[str, FaultSpec]] = None                  # local impacts (added in px, like the bench)

    @property
    def n(self) -> int:
        return len(self.names)

    def matrices(self, sev: Dict[int, float], axis: int = 0):
        """M, C, K at a given fault severity (sev: fault index -> [0, 1])."""
        n = self.n
        kg = np.array(self.k_ground, float)
        W = np.zeros((n, n))
        for (i, j), w in self.couplings.items():
            W[i, j] = W[j, i] = w
        for fi, f in enumerate(self.faults):
            s = sev.get(fi, 0.0)
            mult = 1.0 + (f.factor - 1.0) * s
            if f.kind == "mount":
                kg[self.names.index(f.target[0])] *= mult
            elif f.kind == "joint":
                i, j = self.names.index(f.target[0]), self.names.index(f.target[1])
                W[i, j] *= mult
                W[j, i] *= mult
        K = (np.diag(kg) + np.diag(W.sum(1)) - W) * self.axis_stiffness[axis]
        M = np.diag(self.mass)
        # constant modal damping: C = M^1/2 (2 zeta sqrt(D)) M^1/2,  D = M^-1/2 K M^-1/2
        mh = np.diag(np.sqrt(self.mass))
        mhi = np.diag(1 / np.sqrt(self.mass))
        ev, V = np.linalg.eigh(mhi @ K @ mhi)
        C = mh @ V @ np.diag(2 * self.zeta * np.sqrt(ev)) @ V.T @ mh
        return M, C, K, W, kg

    def truth(self, sev: Dict[int, float], axis: int = 0) -> Dict[str, object]:
        M, C, K, W, kg = self.matrices(sev, axis)
        Mi = np.linalg.inv(M)
        A = np.block([[np.zeros((self.n, self.n)), np.eye(self.n)], [-Mi @ K, -Mi @ C]])
        lam = np.linalg.eigvals(A)
        lam = lam[np.imag(lam) > 0]
        return {"MinvK": Mi @ K, "freq_hz": np.sort(np.abs(lam) / (2 * np.pi)),
                "stability_gap": float(np.min(-np.real(lam))), "mount": kg / np.array(self.mass) * self.axis_stiffness[axis],
                "W": W}


def _discretise(M, C, K, dt):
    n = M.shape[0]
    Mi = np.linalg.inv(M)
    Ac = np.block([[np.zeros((n, n)), np.eye(n)], [-Mi @ K, -Mi @ C]])
    Bc = np.vstack([np.zeros((n, n)), Mi])
    # exact ZOH via the augmented exponential
    E = expm(np.block([[Ac, Bc], [np.zeros((n, 3 * n))]]) * dt)
    return E[:2 * n, :2 * n], E[:2 * n, 2 * n:]


def severities(spec: StructureSpec, t: np.ndarray) -> Dict[int, np.ndarray]:
    out = {}
    for fi, f in enumerate(spec.faults):
        out[fi] = severity_profile(t, FaultSpec(onset_s=f.onset_s, full_s=f.full_s, growth=0.0))
    return out


def simulate_structure(spec: StructureSpec, fps: float, duration_s: float, seed: int = 7,
                       block_s: float = 1.0) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict]:
    """Ground-truth displacement (px) of every part, both axes. Quasi-static fault evolution
    (system matrices updated every block_s). Returns t, {name: (2, n)}, info."""
    rng = np.random.default_rng(seed)
    n_t = int(round(fps * duration_s))
    dt = 1.0 / fps
    t = np.arange(n_t) * dt
    sev = severities(spec, t)
    n = spec.n
    out = np.zeros((2, n, n_t))
    for ax in range(2):
        # broadband force: white, independent per part; scaled so the healthy response has the target RMS
        F = rng.standard_normal((n, n_t))
        tone_f = np.zeros((n, n_t))
        for part, f0, amp in spec.tones:
            gain = np.ones(n_t)
            for fi, f in enumerate(spec.faults):
                if f.kind == "imbalance" and spec.names.index(f.target[0]) == part and abs(f0 - spec.tones[0][1]) < 1e-6:
                    gain = 1.0 + (f.factor - 1.0) * sev[fi]
            ph = rng.uniform(0, 2 * np.pi)
            wander = 1.0 + 0.002 * np.cumsum(rng.standard_normal(n_t)) / np.sqrt(n_t)
            tone_f[part] += amp * gain * np.sin(2 * np.pi * np.cumsum(f0 * wander) * dt + ph)
        x = np.zeros(2 * n)
        Y = np.zeros((n, n_t))
        Yt = np.zeros((n, n_t))
        xt = np.zeros(2 * n)
        blk = max(1, int(round(block_s * fps)))
        # healthy scale factors (displacement per unit force), computed once
        Ad0, Bd0 = _discretise(*spec.matrices({}, ax)[:3], dt)
        for k0 in range(0, n_t, blk):
            s = {fi: float(v[k0]) for fi, v in sev.items()}
            Ad, Bd = _discretise(*spec.matrices(s, ax)[:3], dt) if spec.faults else (Ad0, Bd0)
            for k in range(k0, min(n_t, k0 + blk)):
                Y[:, k] = x[:n]
                Yt[:, k] = xt[:n]
                x = Ad @ x + Bd @ F[:, k]
                xt = Ad @ xt + Bd @ tone_f[:, k]
        # scale: broadband part to response_rms_px (healthy, first 30 s); tones keep their relative size
        ref = Y[:, : int(min(n_t, 30 * fps))]
        g = spec.response_rms_px / np.sqrt(np.mean(ref ** 2))
        if ax == 0:
            g0 = g
        tone_scale = g0 * 1.0
        out[ax] = g * Y + tone_scale * Yt
    truth = {nm: out[:, i, :] for i, nm in enumerate(spec.names)}
    if spec.bearing is not None:
        nm, fs_ = spec.bearing
        for ax, gain in ((0, 1.0), (1, 0.7)):
            truth[nm][ax] += gain * fs_.impact_rms_px * impact_response(t, fps, fs_, rng)
    return t, truth, {"severity": {k: v.tolist() for k, v in sev.items()}}


# ----------------------------------------------------------------------------- identification
def ssi_cov(y: np.ndarray, fs: float, order: int, lags: Optional[int] = None, covariance=None):
    """Covariance-driven stochastic subspace identification.

    y: (l, N) outputs. Returns discrete A (order x order), C (l x order) and continuous poles."""
    y = y - y.mean(axis=1, keepdims=True)
    l, N = y.shape
    s = lags or max(order // l + 2, int(np.ceil(1.5 * order / l)) + 2)
    # output covariances R_i = E[y_{k+i} y_k^T], i = 1..2s
    R = covariance if covariance is not None else [y[:, i:] @ y[:, :N - i].T / (N - i) for i in range(1, 2 * s + 1)]
    T = np.zeros((s * l, s * l))
    for r in range(s):
        for c in range(s):
            T[r * l:(r + 1) * l, c * l:(c + 1) * l] = R[s + r - c - 1]
    U, S, Vt = np.linalg.svd(T)
    O = U[:, :order] * np.sqrt(S[:order])
    C = O[:l]
    A = np.linalg.lstsq(O[:-l], O[l:], rcond=None)[0]
    mu, V = np.linalg.eig(A)
    lam = np.log(mu.astype(complex)) * fs
    return {"A": A, "C": C, "lam": lam, "shapes": C @ V, "sv": S[: order + 4]}


def _physical_mask(lam, fs, tone_hz: Sequence[float] = (), zeta_min=0.004, zeta_max=0.3, f_min=1.0):
    f = np.abs(lam) / (2 * np.pi)
    z = -np.real(lam) / np.maximum(np.abs(lam), 1e-12)
    ok = (np.imag(lam) > 0) & (z > zeta_min) & (z < zeta_max) & (f > f_min) & (f < 0.45 * fs)
    for th in tone_hz:                       # forced harmonics are (nearly) undamped poles
        ok &= ~((np.abs(f - th) < 0.6) & (z < 0.012))
    return ok


def mac(a: np.ndarray, b: np.ndarray) -> float:
    """Modal assurance criterion: 1 = same mode shape."""
    return float(np.abs(np.vdot(a, b)) ** 2 / (np.vdot(a, a).real * np.vdot(b, b).real + 1e-30))


def stable_modes(y: np.ndarray, fs: float, n_modes: int, tone_hz: Sequence[float] = (), base_order: int = None,
                 n_orders: int = 5, step: int = 4, min_count: int = 3, df: float = 0.015, mac_min: float = 0.9,
                 reference: Optional[Sequence[Tuple[float, np.ndarray]]] = None, covariance=None):
    """Stabilisation diagram: keep poles that persist (frequency, damping, shape) across model orders.

    Spurious poles (noise, split modes, tone leakage) move when the order changes; structural ones do not."""
    l = y.shape[0]
    base_order = base_order or 2 * l + 2 * len(tone_hz) + 4
    clusters: List[dict] = []
    for o in range(base_order, base_order + n_orders * step, step):
        r = ssi_cov(y, fs, o, covariance=covariance)
        for k in np.where(_physical_mask(r["lam"], fs, tone_hz))[0]:
            lam, sh = r["lam"][k], r["shapes"][:, k]
            sh = sh / np.linalg.norm(sh)
            f, z = abs(lam) / (2 * np.pi), -lam.real / abs(lam)
            best = None
            for c in clusters:
                if (abs(f - c["f"]) / c["f"] < df and abs(z - c["z"]) < 0.5 * c["z"] + 0.005
                        and mac(sh, c["shapes"][0]) > mac_min and o not in c["orders"]):
                    best = c
                    break
            if best is None:
                clusters.append({"f": f, "z": z, "lams": [lam], "shapes": [sh], "orders": [o]})
            else:
                best["lams"].append(lam)
                best["shapes"].append(sh)
                best["orders"].append(o)
                best["f"] = float(np.median([abs(v) / (2 * np.pi) for v in best["lams"]]))
                best["z"] = float(np.median([-v.real / abs(v) for v in best["lams"]]))
    return _choose(clusters, n_modes, min_count, reference, l)


def _rep(c):
    lam = np.median(np.real(c["lams"])) + 1j * np.median(np.imag(c["lams"]))
    sh = c["shapes"][int(np.argmin([abs(v - lam) for v in c["lams"]]))]
    return lam, sh


def _choose(clusters, n_modes, min_count, reference, l):
    """Pick n modes. Without a reference: the most persistent clusters. With a reference (baseline modes):
    mode tracking - assign each baseline mode the stable cluster minimising |df|/f + (1 - MAC), so an
    extra (unmodelled) mode cannot displace a structural one; unassigned clusters are reported."""
    from scipy.optimize import linear_sum_assignment
    stable = [c for c in clusters if len(c["orders"]) >= min_count]
    info = {"n_stable": len(stable)}
    if reference is None:
        stable.sort(key=lambda c: (-len(c["orders"]), np.mean([1 - mac(s, c["shapes"][0]) for s in c["shapes"]])))
        chosen = sorted(stable[:n_modes], key=lambda c: c["f"])
        extra = []
    else:
        pool = stable if len(stable) >= n_modes else [c for c in clusters if len(c["orders"]) >= 2]
        cost = np.full((len(reference), max(len(pool), 1)), 1e6)
        for i, (fr, shr) in enumerate(reference):
            for j, c in enumerate(pool):
                lam, sh = _rep(c)
                f, m = abs(lam) / (2 * np.pi), mac(sh, shr)
                if 0.5 < f / fr < 1.5 and m > 0.5:
                    cost[i, j] = abs(f - fr) / fr + (1 - m) - 0.02 * len(c["orders"])
        ri, ci = linear_sum_assignment(cost)
        ok = cost[ri, ci] < 1e5
        chosen = [pool[j] for j in ci[ok]]
        used = set(ci[ok].tolist())
        extra = [round(float(c["f"]), 2) for j, c in enumerate(pool) if j not in used and len(c["orders"]) >= min_count]
        chosen = sorted(chosen, key=lambda c: c["f"])
    reps = [_rep(c) for c in chosen]
    lam = np.array([r[0] for r in reps])
    shapes = np.stack([r[1] for r in reps], 1) if reps else np.zeros((l, 0))
    info.update(counts=[len(c["orders"]) for c in chosen], extra_modes_hz=extra)
    return lam, shapes, info


def physical_jacobian(lam_sel: np.ndarray, shapes_sel: np.ndarray):
    """Rebuild [[0, I], [-M^-1K, -M^-1C]] in output (pixel) coordinates from n modes.

    x = Psi q, x' = Psi Lambda q with the conjugate pairs included, so T = [Psi; Psi Lambda]."""
    lam = np.concatenate([lam_sel, np.conj(lam_sel)])
    Psi = np.concatenate([shapes_sel, np.conj(shapes_sel)], axis=1)
    T = np.vstack([Psi, Psi * lam[None, :]])
    A = np.real(T @ np.diag(lam) @ np.linalg.inv(T))
    n = shapes_sel.shape[0]
    consistency = float(np.linalg.norm(A[:n, :n]) + np.linalg.norm(A[:n, n:] - np.eye(n))) / np.sqrt(n)
    return A, -A[n:, :n], -A[n:, n:], consistency


def normalized_laplacian(W: np.ndarray):
    result = graph_diagnostics(W)
    return result["eigenvalues"], result["eigenvectors"]


def participation_ratio(v: np.ndarray) -> float:
    """(sum |v|^2)^2 / (n sum |v|^4): 1 = all parts move, 1/n = one part only."""
    p = np.abs(v) ** 2
    return float(p.sum() ** 2 / (len(v) * np.sum(p ** 2)))


def identify(disp: Dict[str, np.ndarray], fs: float, names: Sequence[str], axis: int = 0,
             tone_hz: Sequence[float] = (), order: Optional[int] = None,
             reference: Optional[Sequence[Tuple[float, np.ndarray]]] = None,
             bootstrap_reps: int = 0, common_scale: bool = False, seed: int = 0) -> Dict[str, object]:
    """Identify the structural Jacobian from multi-ROI displacements (one image axis).

    reference: baseline modes [(f_hz, shape)] (from a previous call's "modes") for mode tracking."""
    n = len(names)
    y = np.stack([disp[k][axis] for k in names])
    scales = y.std(axis=1)
    if not np.isfinite(y).all() or np.any(scales < 1e-12):
        return {"ok": False, "reason": "constant or invalid measured coordinate"}
    # Precondition SSI, then restore modal coordinates. This makes the numerical
    # identification equivariant to fixed positive per-channel magnification.
    normalized_reference = [(f, np.asarray(v) / scales) for f, v in reference] if reference is not None else None
    lam, sh, diag = stable_modes(y / scales[:, None], fs, n, tone_hz, base_order=order,
                                 reference=normalized_reference)
    sh = scales[:, None] * sh
    if len(lam) < n:
        return {"ok": False, "reason": f"only {len(lam)} stable structural modes for {n} parts", "n_modes": int(len(lam))}
    try:
        A, MinvK, MinvC, cons = physical_jacobian(lam, sh)
    except np.linalg.LinAlgError:
        return {"ok": False, "reason": "singular physical reconstruction"}
    physical = invariant_operator(MinvK)
    if not physical["valid"]:
        return {"ok": False, "reason": "; ".join(physical["reasons"]),
                "physical_diagnostics": {k: physical[k] for k in ("similarity_residual", "cycle_residual", "positive_offdiagonal_fraction")}}
    Wt = np.sqrt(np.clip(MinvK * MinvK.T, 0, None)) * (MinvK < 0) * (MinvK.T < 0)
    np.fill_diagonal(Wt, 0.0)
    graph = graph_diagnostics(Wt, names)
    ev, V = graph["eigenvalues"], graph["eigenvectors"]
    freqs = np.abs(lam) / (2 * np.pi)
    zetas = -np.real(lam) / np.abs(lam)
    slow = int(np.argmin(-np.real(lam)))
    D_ev = physical["eigenvalues"]
    mu = np.concatenate([lam, np.conj(lam)])
    Psi = np.concatenate([sh, np.conj(sh)], axis=1)
    omega_scale = max(float(np.median(np.abs(mu))), 1e-12)
    modal_condition = float(np.linalg.cond(np.vstack([Psi, Psi * mu[None, :] / omega_scale])))
    if modal_condition > 1e6:
        return {"ok": False, "reason": "ill-conditioned modal reconstruction", "modal_condition": modal_condition}
    uncertainty = operator_uncertainty(y, fs, lam, sh, physical["operator"], tone_hz,
                                       order=order, reps=bootstrap_reps, seed=seed) if bootstrap_reps else {
        "available": False, "reason": "bootstrap disabled", "replicates": 0}
    mount = MinvK.sum(1)
    return {
        "ok": True, "axis": axis, "names": list(names),
        "freq_hz": freqs.tolist(), "zeta": zetas.tolist(),
        "stability_gap": float(np.min(-np.real(lam))), "relax_time_s": float(1 / np.min(-np.real(lam))),
        "slow_mode_hz": float(freqs[slow]), "slow_mode_pr": participation_ratio(sh[:, slow]),
        "slow_mode_part": names[int(np.argmax(np.abs(sh[:, slow])))],
        "MinvK": MinvK.tolist(), "MinvC": MinvC.tolist(), "mount": mount.tolist(), "coupling": Wt.tolist(),
        "operator": physical["operator"].tolist(), "operator_eigenvalues": D_ev.tolist(),
        "physical_diagnostics": {k: physical[k] for k in ("similarity_residual", "cycle_residual", "positive_offdiagonal_fraction")},
        "modal_condition": modal_condition, "uncertainty": uncertainty,
        "mount_interpretable": bool(common_scale),
        "graph_connected": graph["connected"], "graph_components": graph["components"],
        "conductance": graph["conductance"], "conductance_cut": graph["cut"],
        "graph_gap": graph["gap"], "fiedler": graph["fiedler"].tolist(),
        "rayleigh_lower": float(np.min(mount)) if common_scale else None, "rayleigh_upper": float(np.min(np.diag(MinvK))),
        "omega_min_sq": float(D_ev[0]),
        "consistency": cons, "consistency_scope": "algebraic reconstruction identity; not a validation score", "stabilisation": diag,
        "modes": [(float(f), sh[:, k]) for k, f in enumerate(freqs)],
        "modes_pr": [participation_ratio(sh[:, k]) for k in range(n)],
    }


def operator_uncertainty(y, fs, lam, shapes, D, tone_hz=(), order=None, reps=16, seed=0):
    """Centered block bootstrap of lag-covariance contributions, never frame shuffling.

    Conditional empirical radius for this window/model. Coverage is not certified.
    Invalid replicates are counted and insufficient support produces abstention.
    """
    if reps < 8:
        return {"available": False, "reason": "at least 8 bootstrap replicates required", "replicates": reps}
    y = np.asarray(y, float)
    y = y - y.mean(1, keepdims=True)
    scales = y.std(axis=1)
    y = y / scales[:, None]
    n, N = y.shape
    base_order = order or 2 * n + 2 * len(tone_hz) + 4
    largest_order = base_order + 8
    s = max(largest_order // n + 2, int(np.ceil(1.5 * largest_order / n)) + 2)
    maxlag = 2 * s
    block = max(int(round(4 * fs)), 3 * maxlag)
    blocks = N // block
    if blocks < 6:
        return {"available": False, "reason": "fewer than six temporal blocks; collect longer footage", "replicates": reps}
    parts = []
    for i in range(blocks):
        v = y[:, i * block:(i + 1) * block]
        parts.append([v[:, lag:] @ v[:, :-lag].T / (block - lag) for lag in range(1, maxlag + 1)])
    parts = np.asarray(parts)
    center = parts.mean(0)
    original = np.asarray([y[:, lag:] @ y[:, :-lag].T / (N - lag) for lag in range(1, maxlag + 1)])
    reference = [(float(abs(p) / (2 * np.pi)), shapes[:, i] / scales) for i, p in enumerate(lam)]
    rng = np.random.default_rng(seed)
    operators, graph_laplacians = [], []
    for _ in range(reps):
        cov = original + parts[rng.integers(0, blocks, blocks)].mean(0) - center
        try:
            pl, sh, _ = stable_modes(y, fs, n, tone_hz, base_order=base_order, n_orders=3,
                                     min_count=2, reference=reference, covariance=cov)
            if len(pl) != n:
                continue
            _, B, _, _ = physical_jacobian(pl, scales[:, None] * sh)
            check = invariant_operator(B)
            if not check["valid"]:
                continue
            operators.append(check["operator"])
            graph_laplacians.append(graph_diagnostics(check["coupling"])["laplacian"])
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            continue
    valid = len(operators)
    result = {"available": False, "replicates": reps, "valid_replicates": valid,
              "blocks": blocks, "block_seconds": block / fs,
              "method": "centered multichannel block covariance bootstrap; empirical 95% radius",
              "coverage": "uncalibrated conditional estimate; model bias and nonstationarity excluded"}
    if valid < max(6, int(np.ceil(0.6 * reps))):
        result["reason"] = "too few physically compatible replicates; do not localize"
        return result
    operators = np.asarray(operators)
    errors = [np.linalg.norm(v - D, 2) for v in operators]
    qlo, qhi = np.quantile(operators, [0.025, 0.975], axis=0)
    graph0 = graph_diagnostics(invariant_operator(D)["coupling"])
    graph_radius = float(np.quantile([np.linalg.norm(L - graph0["laplacian"], 2) for L in graph_laplacians], 0.95))
    result.update(available=True, operator_radius=float(np.quantile(errors, 0.95)),
                  entry_lower=qlo.tolist(), entry_upper=qhi.tolist(),
                  graph_radius=graph_radius,
                  fiedler_sine_angle_bound=float(min(1., 2 * graph_radius / max(graph0["fiedler_separation"], 1e-15))))
    return result


def fiedler_cut(fiedler: Sequence[float], names: Sequence[str]) -> Tuple[List[str], List[str]]:
    v = np.asarray(fiedler)
    return [nm for nm, x in zip(names, v) if x >= 0], [nm for nm, x in zip(names, v) if x < 0]


def force_at(disp: Dict[str, np.ndarray], names: Sequence[str], fs: float, f0: float, MinvK, MinvC,
             axis: int = 0, search_hz: float = 0.3) -> Tuple[float, np.ndarray]:
    """Where is a tonal force applied? Invert the identified dynamics at one frequency:

        M^-1 f(w) = (-w^2 I + i w M^-1 C + M^-1 K) X(w)

    X(w) is the complex amplitude of each part's displacement at f0 (lock-in, f0 refined within +/- search_hz).
    Returns (f0_refined, |M^-1 f| per part): a force acting on one part shows up on that part only, even though
    the *response* to it spreads over the whole structure. Ill-conditioned for a line sitting on a lightly damped
    mode (D is then nearly singular and small modal errors dominate)."""
    y = np.stack([disp[k][axis] for k in names])
    y = y - y.mean(axis=1, keepdims=True)
    t = np.arange(y.shape[1]) / fs
    win = np.hanning(y.shape[1])
    best, X = f0, None
    for f in np.linspace(f0 - search_hz, f0 + search_hz, 31):
        Z = 2 * (y * win) @ np.exp(-2j * np.pi * f * t) / win.sum()
        if X is None or np.linalg.norm(Z) > np.linalg.norm(X):
            best, X = f, Z
    w = 2 * np.pi * best
    D = -w ** 2 * np.eye(len(names)) + 1j * w * np.asarray(MinvC) + np.asarray(MinvK)
    return float(best), np.abs(D @ X)


# ----------------------------------------------------------------------------- streaming watch (agent tool)
class StructuralWatch:
    """Baseline the identified Jacobian, then report relative changes of every mount (row sums of
    M^-1 K) and every coupling, plus the coupling graph's spectral gap and Fiedler cut.

    Thresholds are relative changes; defaults come from eval/run_structure.py (1.5 x the largest deviation on
    separate healthy seeds was 8 % for mounts and 7 % for couplings; rounded up to 10 %)."""

    def __init__(self, names: Sequence[str], fs: float, tone_hz: Sequence[float] = (), calib_n: int = 3,
                 mount_thr: float = 0.10, coupling_thr: float = 0.10, axes: Sequence[int] = (0, 1),
                 common_scale: bool = False, bootstrap_reps: int = 16, uncertainty_required: bool = True):
        self.names, self.fs, self.tone_hz = list(names), fs, list(tone_hz)
        self.calib_n, self.mount_thr, self.coupling_thr, self.axes = calib_n, mount_thr, coupling_thr, tuple(axes)
        self.calib: List[Tuple[np.ndarray, np.ndarray, float]] = []
        self.base = None
        self.common_scale, self.bootstrap_reps = bool(common_scale), int(bootstrap_reps)
        self.uncertainty_required = bool(uncertainty_required)
        self.operator_calib = []
        self.history: List[dict] = []

    def update(self, disp: Dict[str, np.ndarray], t: float) -> dict:
        ref = getattr(self, "ref", None)
        ids = {ax: identify(disp, self.fs, self.names, ax, self.tone_hz, reference=ref[ax] if ref else None,
                            bootstrap_reps=self.bootstrap_reps, common_scale=self.common_scale,
                            seed=int(round(t * self.fs)) + ax)
               for ax in self.axes}
        good = [ax for ax in self.axes if ids[ax]["ok"]]
        if not good or (self.base is None and len(good) < len(self.axes)):
            ev = {"t": t, "ok": False, "reason": next(r["reason"] for r in ids.values() if not r["ok"])}
            self.history.append(ev)
            return ev
        mo = {ax: np.array(ids[ax]["mount"]) for ax in good}
        if not self.common_scale:
            mo = {ax: np.diag(np.asarray(ids[ax]["operator"])) for ax in good}
        co = {ax: np.array(ids[ax]["coupling"]) for ax in good}
        gg = float(np.mean([ids[ax]["graph_gap"] for ax in good]))
        ev = {"t": t, "ok": True, "axes_used": good, "mount_abs": {ax: mo[ax].tolist() for ax in good},
              "jacobian": {ax: {"MinvK": ids[ax]["MinvK"], "MinvC": ids[ax]["MinvC"]} for ax in good}, "freq_hz": [ids[ax]["freq_hz"] for ax in good],
              "zeta": [ids[ax]["zeta"] for ax in good],
              "stability_gap": float(np.mean([ids[ax]["stability_gap"] for ax in good])), "graph_gap": gg,
              "fiedler_cut": ids[good[0]]["conductance_cut"],
              "conductance": ids[good[0]]["conductance"],
              "mount_interpretable": self.common_scale,
              "operator": {ax: ids[ax]["operator"] for ax in good},
              "uncertainty": {ax: ids[ax]["uncertainty"] for ax in good},
              "physical_diagnostics": {ax: ids[ax]["physical_diagnostics"] for ax in good},
              "extra_modes_hz": [ids[ax]["stabilisation"].get("extra_modes_hz", []) for ax in good]}
        if self.base is None:
            self.calib.append((mo, co, gg, {ax: ids[ax]["modes"] for ax in good}))
            self.operator_calib.append({ax: (np.asarray(ids[ax]["operator"]), ids[ax]["uncertainty"]) for ax in good})
            ev["phase"] = "calibrating"
            if len(self.calib) >= self.calib_n:
                self.base = ({ax: np.median([c[0][ax] for c in self.calib], 0) for ax in self.axes},
                             {ax: np.median([c[1][ax] for c in self.calib], 0) for ax in self.axes},
                             float(np.median([c[2] for c in self.calib])))
                self.ref = self.calib[-1][3]
                self.operator_base = {ax: np.mean([v[ax][0] for v in self.operator_calib], axis=0) for ax in self.axes}
                self.radius_base = {}
                for ax in self.axes:
                    valid = all(v[ax][1].get("available") for v in self.operator_calib)
                    self.radius_base[ax] = max(v[ax][1]["operator_radius"] + np.linalg.norm(v[ax][0] - self.operator_base[ax], 2)
                                               for v in self.operator_calib) if valid else None
            self.history.append(ev)
            return ev
        mo0, co0, gg0 = self.base
        decision_axes = [ax for ax in good if self.radius_base[ax] is not None and ids[ax]["uncertainty"].get("available")]
        if self.uncertainty_required and decision_axes:
            good = decision_axes
        ev["decision_axes"] = decision_axes
        n = len(self.names)
        mount_rel = np.mean([mo[ax] / mo0[ax] for ax in good], 0) - 1
        cmax = max(co0[ax].max() for ax in good)
        edges = [(i, j) for i in range(n) for j in range(i + 1, n) if all(co0[ax][i, j] > 0.05 * cmax for ax in good)]
        coup_rel = {f"{self.names[i]}|{self.names[j]}": float(np.mean([co[ax][i, j] / co0[ax][i, j] for ax in good]) - 1)
                    for i, j in edges}
        mz = np.abs(mount_rel) / self.mount_thr
        cz = {e: abs(v) / self.coupling_thr for e, v in coup_rel.items()}
        e_best = max(cz, key=cz.get) if cz else None
        where = None
        if e_best and cz[e_best] > max(1.0, mz.max()):
            where = {"kind": "joint", "between": e_best, "change": coup_rel[e_best]}
        elif mz.max() > 1.0:
            k = int(np.argmax(mz))
            where = {"kind": "mount" if self.common_scale else "local_stiffness",
                     "part": self.names[k], "change": float(mount_rel[k])}
        comparisons = {ax: compare_operators(self.operator_base[ax], ids[ax]["operator"], self.radius_base[ax],
                                             ids[ax]["uncertainty"].get("operator_radius")) for ax in good}
        supported = any(v["uncertainty_available"] for v in comparisons.values())
        resolved = any(v["resolved"] for v in comparisons.values())
        candidate = where
        decision = "nominal"
        if self.uncertainty_required and not supported:
            where, decision = None, "abstain"
        elif self.uncertainty_required and candidate and not resolved:
            where, decision = None, "collect_more_data"
        elif candidate:
            # Direct operator entries can localize even if a single eigenvector cannot.
            key = "between" if candidate["kind"] == "joint" else "part"
            indices = [self.names.index(v) for v in candidate[key].split("|")]
            i, j = indices if len(indices) == 2 else (indices[0], indices[0])
            entry_resolved = any(abs(np.asarray(ids[ax]["operator"])[i, j] - self.operator_base[ax][i, j]) >
                                 comparisons[ax].get("epsilon_combined", np.inf) for ax in good)
            if self.uncertainty_required and not entry_resolved:
                where, decision = None, "change_unlocalized"
            else:
                decision = "localized_change"
        ev.update(operator_comparison=comparisons, decision=decision, candidate_change=candidate,
                  uncertainty_scope="empirical conditional bootstrap, not certified coverage",
                  request="collect a longer stationary clip / improve texture" if decision in ("abstain", "collect_more_data") else None)
        ev.update(phase="watching", mount_rel=dict(zip(self.names, map(float, mount_rel))), coupling_rel=coup_rel,
                  graph_gap_baseline=gg0, change=where)
        self.history.append(ev)
        return ev
