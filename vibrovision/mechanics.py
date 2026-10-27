"""Mechanical invariants and diagnostics. See docs/MATHEMATICS.md for assumptions.

The functions never turn an algebraic identity into a claim of statistical confidence.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.linalg import eigh


def invariant_operator(B, tolerance=0.10):
    """Recover mass-normalized stiffness from B=S M^-1 K S^-1.

    Requires diagonal positive M/S and reciprocal spring couplings K_ij<=0.
    Small incompatible entries are projected out, but their residual is reported.
    `valid` is a model-compatibility gate, not a confidence level.
    """
    B = np.asarray(B, float)
    if B.ndim != 2 or B.shape[0] != B.shape[1] or not np.isfinite(B).all():
        raise ValueError("B must be a finite square matrix")
    n = len(B)
    off = B.copy()
    np.fill_diagonal(off, 0)
    scale = max(np.linalg.norm(B), 1e-15)
    positive = float(np.linalg.norm(np.maximum(off, 0)) / scale)
    reciprocal = (B < 0) & (B.T < 0)
    weights = np.sqrt(np.maximum(B * B.T, 0)) * reciprocal
    np.fill_diagonal(weights, 0)
    D = np.diag(np.diag(B)) - weights

    # Detailed balance: q_i B_ij = q_j B_ji, q_i = m_i / s_i^2.
    peak = max(float(weights.max()), 1e-15)
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)
             if weights[i, j] > 0.02 * peak]
    rows, rhs = [], []
    for i, j in edges:
        row = np.zeros(n)
        row[i], row[j] = 1, -1
        weight = np.sqrt(weights[i, j] / peak)
        rows.append(row * weight)
        rhs.append((np.log(abs(B[j, i])) - np.log(abs(B[i, j]))) * weight)
    logq = np.linalg.lstsq(np.asarray(rows).reshape(-1, n), rhs, rcond=None)[0] if rows else np.zeros(n)
    # Centering removes an arbitrary global scale. A large span is itself a warning.
    logq -= logq.mean()
    q = np.exp(np.clip(logq, -50, 50))
    similar = np.sqrt(q)[:, None] * B / np.sqrt(q)[None, :]
    residual = float(np.linalg.norm(similar - D) / max(np.linalg.norm(D), 1e-15))
    cycle_residual = float(np.linalg.norm(np.asarray(rows) @ logq - rhs) / max(np.sqrt(len(rows)), 1)) if rows else 0.0
    eig = np.linalg.eigvalsh(D)
    reasons = []
    if positive > tolerance:
        reasons.append("incompatible positive off-diagonal entries")
    if residual > tolerance or cycle_residual > 2 * tolerance:
        reasons.append("reciprocal couplings are not diagonally symmetrizable within tolerance")
    if eig[0] <= 0:
        reasons.append("non-positive stiffness operator")
    if np.ptp(logq) > 25:
        reasons.append("extreme coordinate scaling / mass conditioning")
    return {"operator": D, "coupling": weights, "eigenvalues": eig,
            "valid": not reasons, "reasons": reasons,
            "similarity_residual": residual, "cycle_residual": cycle_residual,
            "positive_offdiagonal_fraction": positive, "symmetrizer": q,
            "assumptions": "diagonal positive mass and image scales; reciprocal linear springs; full observed coordinates"}


def graph_diagnostics(W, names=None):
    """Normalized graph spectrum with isolated vertices assigned zero diagonal.

    Zero eigenvalues count components, including isolated vertices. Conductance
    is descriptive when the inferred edges are uncertain.
    """
    W = np.asarray(W, float)
    if W.ndim != 2 or W.shape[0] != W.shape[1] or not np.isfinite(W).all():
        raise ValueError("W must be finite and square")
    if np.min(W) < -1e-12 or not np.allclose(W, W.T):
        raise ValueError("W must be symmetric and nonnegative")
    n = len(W)
    if n == 0:
        raise ValueError("graph needs at least one vertex")
    W = W.copy()
    np.fill_diagonal(W, 0)
    d = W.sum(1)
    positive = d > 1e-12 * max(float(d.max()), 1)
    di = np.zeros(n)
    di[positive] = 1 / np.sqrt(d[positive])
    L = np.diag(positive.astype(float)) - di[:, None] * W * di[None, :]
    ev, V = np.linalg.eigh(L)
    components = []
    unseen = set(range(n))
    threshold = 1e-12 * max(float(W.max()), 1)
    while unseen:
        pending = [min(unseen)]
        component = set()
        while pending:
            i = pending.pop()
            if i in component:
                continue
            component.add(i)
            pending.extend(j for j in np.flatnonzero(W[i] > threshold) if j not in component)
        unseen -= component
        components.append(sorted(component))
    labels = list(names) if names is not None else [str(i) for i in range(n)]
    gap = float(max(0, ev[1])) if n > 1 else 0.0
    vector = V[:, 1] if n > 1 else np.ones(1)
    cut, conductance = None, None
    if len(components) > 1:
        a = components[0]
        cut = ([labels[i] for i in a], [labels[i] for i in range(n) if i not in a])
        conductance = 0.0
    elif n > 1:
        # Exact minimum for the small machines in the demo; sweep for larger ones.
        if n <= 12:
            candidates = [s for r in range(1, n) for s in combinations(range(n), r) if 0 in s]
        else:
            order = np.argsort(vector * di)
            candidates = [order[:r] for r in range(1, n)]
        best = np.inf
        for ids in candidates:
            ids = list(ids)
            rest = [i for i in range(n) if i not in ids]
            den = min(d[ids].sum(), d[rest].sum())
            h = W[np.ix_(ids, rest)].sum() / den if den > 0 else np.inf
            if h < best:
                best, cut = float(h), ([labels[i] for i in ids], [labels[i] for i in rest])
        conductance = best
    separation = min(ev[1] - ev[0], ev[2] - ev[1]) if n > 2 else (ev[1] - ev[0] if n > 1 else 0)
    return {"laplacian": L, "eigenvalues": ev, "eigenvectors": V, "gap": gap,
            "fiedler": vector, "fiedler_separation": float(max(0, separation)),
            "components": [[labels[i] for i in c] for c in components],
            "connected": len(components) == 1, "cut": cut, "conductance": conductance}


def modal_sensitivities(K, M, edges=None):
    """Simple-eigenvalue derivatives; physical parameters need calibrated M/K."""
    K, M = np.asarray(K, float), np.asarray(M, float)
    ev, V = eigh(K, M)
    n = len(K)
    edges = list(edges) if edges is not None else [(i, j) for i in range(n) for j in range(i + 1, n) if K[i, j] < 0]
    return {"omega_squared": ev, "modes": V, "mount": (V ** 2).T,
            "edges": edges, "joint": np.stack([(V[i] - V[j]) ** 2 for i, j in edges], axis=1) if edges else np.empty((n, 0)),
            "scope": "first order; simple modes; mass-normalized eigenvectors; fixed mass"}


def stiffness_loss(K, direction, loss):
    """Exact grounded compliance update; no inference of uncalibrated units."""
    K, b, loss = np.asarray(K, float), np.asarray(direction, float), float(loss)
    if loss < 0 or np.linalg.eigvalsh(K)[0] <= 0:
        raise ValueError("requires positive definite stiffness and nonnegative loss")
    inv = np.linalg.inv(K)
    v = inv @ b
    R = float(b @ v)
    margin = 1 - loss * R
    return {"effective_resistance": R, "margin": margin, "positive_definite": margin > 0,
            "inverse": inv + loss / margin * np.outer(v, v) if margin > 0 else None,
            "logdet_change": float(np.log(margin)) if margin > 0 else None}


def compare_operators(baseline, current, epsilon_baseline=None, epsilon_current=None):
    """Weyl/change and Davis--Kahan diagnostics conditional on supplied errors.

    Estimated bootstrap radii produce empirical diagnostics, not certified bounds.
    """
    D0, Dt = np.asarray(baseline, float), np.asarray(current, float)
    e0, V0 = np.linalg.eigh(D0)
    et, Vt = np.linalg.eigh(Dt)
    change = float(np.linalg.norm(Dt - D0, 2))
    out = {"operator_change": change, "relative_change": change / max(np.linalg.norm(D0, 2), 1e-15),
           "eigenvalue_changes": (et - e0).tolist(), "uncertainty_available": False,
           "resolved": False, "location_stable": False,
           "reason": "uncertainty unavailable; collect more data"}
    if epsilon_baseline is None or epsilon_current is None:
        return out
    radius = float(epsilon_baseline + epsilon_current)
    separation = np.array([min(abs(e0[i] - e0[j]) for j in range(len(e0)) if j != i)
                           for i in range(len(e0))]) if len(e0) > 1 else np.array([np.inf])
    angle_bounds = np.minimum(1., 2 * radius / np.maximum(separation, 1e-15))
    resolved = change > radius
    stable = bool(resolved and np.all(angle_bounds < 0.5))
    out.update(uncertainty_available=True, epsilon_combined=radius,
               eigenvalue_changes_resolved=(np.abs(et - e0) > radius).tolist(),
               mode_sine_angle_bound=angle_bounds.tolist(), resolved=resolved, location_stable=stable,
               reason=("change exceeds empirical operator uncertainty" if resolved else "change overlaps empirical uncertainty"),
               uncertainty_scope="conditional perturbation diagnostics using empirical bootstrap radii; not a certified coverage claim")
    return out
