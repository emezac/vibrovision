"""Tests for the mathematical core: image-Jacobian noise theory and structural-Jacobian identification."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vibrovision.pipeline import build_scenario  # noqa: E402
from vibrovision.simulate import generate_displacements, render_frames  # noqa: E402
from vibrovision.structure import (StructFault, StructuralWatch, StructureSpec, fiedler_cut, identify,  # noqa: E402
                                   normalized_laplacian, simulate_structure)
from vibrovision.theory import estimate_sensor_noise, ic_noise_model  # noqa: E402
from vibrovision.tracking import ICTracker  # noqa: E402

FS = 120.0


def test_sensor_noise_estimate_and_static_floor_match_theory():
    """sigma from frame differences, and the tracker noise on a static target, both as predicted."""
    sc = build_scenario("healthy", seed=5, cam_noise=4.0, duration_s=6.0)
    t, truth, shake = generate_displacements(sc)
    fr = list(render_frames(sc, truth, shake))
    sig = estimate_sensor_noise(fr[:240])["sigma"]
    assert abs(sig - np.sqrt(16 + 1 / 12)) / 4.0 < 0.05
    tr = ICTracker(fr[0], sc.reference_roi)
    d = np.array([tr.track(f)[:2] for f in fr[1:]]).T
    meas = np.sqrt(np.sum(d.var(axis=1)))
    pred = ic_noise_model(fr[0], sc.reference_roi, sig, FS)
    assert abs(meas / pred["ic_rms_px"] - 1) < 0.12
    assert pred["fisher_lambda_min"] > 0 and pred["crb_rms_px"] > 0


def test_ssi_recovers_mount_stiffness_from_displacement():
    sp = StructureSpec()
    t, truth, _ = simulate_structure(sp, FS, 60.0, seed=3)
    noisy = {k: v + 0.01 * np.random.default_rng(1).standard_normal(v.shape) for k, v in truth.items()}
    for ax in (0, 1):
        r = identify(noisy, FS, sp.names, ax, (24.6, 49.2, 36.9), common_scale=True)
        assert r["ok"]
        true = sp.truth({}, ax)
        np.testing.assert_allclose(r["mount"], true["mount"], rtol=0.08)
        np.testing.assert_allclose(r["freq_hz"], true["freq_hz"], rtol=0.02)
        # Rayleigh bound (2): min k_g/m <= omega_min^2 <= min (k_g + d)/m, on the identified matrix
        assert r["rayleigh_lower"] <= r["omega_min_sq"] * 1.02
        assert r["omega_min_sq"] <= r["rayleigh_upper"] * 1.02


def test_rayleigh_bound_holds_for_random_structures():
    rng = np.random.default_rng(0)
    for _ in range(50):
        n = 4
        kg = rng.uniform(1e3, 2e4, n)
        W = np.triu(rng.uniform(0, 2e4, (n, n)) * (rng.random((n, n)) < 0.6), 1)
        W = W + W.T
        m = rng.uniform(0.5, 4, n)
        K = np.diag(kg) + np.diag(W.sum(1)) - W
        D = np.diag(m ** -0.5) @ K @ np.diag(m ** -0.5)
        w2 = np.linalg.eigvalsh(D)[0]
        assert np.min(kg / m) <= w2 + 1e-6
        assert w2 <= np.min((kg + W.sum(1)) / m) + 1e-6


def test_fiedler_cut_finds_weak_joint():
    W = np.array([[0, 5, 0, 0], [5, 0, 0.3, 0], [0, 0.3, 0, 5], [0, 0, 5, 0]], float)
    ev, V = normalized_laplacian(W)
    a, b = fiedler_cut(V[:, 1], list("ABCD"))
    assert sorted([sorted(a), sorted(b)]) == [["A", "B"], ["C", "D"]]
    assert ev[1] < 0.1                       # Cheeger: weak cut <=> small spectral gap


def test_structural_watch_localises_loose_mount():
    sp = StructureSpec(faults=(StructFault("mount", ("Rodamiento_B",), 0.3, onset_s=60, full_s=100),))
    t, truth, _ = simulate_structure(sp, FS, 140.0, seed=2)
    w = StructuralWatch(sp.names, FS, (24.6, 49.2, 36.9), common_scale=True)
    ev = None
    for end in (40, 50, 60, 140):
        sl = slice(int((end - 40) * FS), int(end * FS))
        ev = w.update({k: v[:, sl] for k, v in truth.items()}, end)
    assert ev["change"] and ev["change"]["kind"] == "mount" and ev["change"]["part"] == "Rodamiento_B"
    assert ev["change"]["change"] < -0.4


def test_force_inversion_locates_tonal_force_source():
    """Imbalance on the motor: the response spreads over all parts, the inverted force does not."""
    from vibrovision.structure import force_at
    sp = StructureSpec(faults=(StructFault("imbalance", ("Motor",), 4.0, onset_s=0, full_s=1),))
    t, truth, _ = simulate_structure(sp, FS, 40.0, seed=4)
    r = identify(truth, FS, sp.names, 1, (24.6, 49.2, 36.9))
    assert r["ok"]
    f, F = force_at(truth, sp.names, FS, 24.6, r["MinvK"], r["MinvC"], axis=1)
    assert int(np.argmax(F)) == 0 and F[1:].max() < 0.35 * F[0]
