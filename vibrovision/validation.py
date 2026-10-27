"""Executable checks, with separate algebra, synthetic-video and real-video evidence."""
from __future__ import annotations

import platform
import time

import cv2
import numpy as np
import scipy
from scipy.linalg import eigh

from .mechanics import (invariant_operator, graph_diagnostics, modal_sensitivities,
                        stiffness_loss, compare_operators)
from .spectral import alpha_mixture, analyze_window, SpectralConfig
from .spectralmatrix import relative_spectral_evidence
from .structure import StructureSpec, simulate_structure, identify
from .simulate import default_scene, render_frames, colored_noise
from .tracking import track_sequence
from .theory import nuisance_fisher


def algebra_checks(seed=2026):
    rng = np.random.default_rng(seed)
    sp = StructureSpec()
    M, C, K, W, _ = sp.matrices({})
    B = np.linalg.solve(M, K)
    D = np.diag(np.diag(M) ** -0.5) @ K @ np.diag(np.diag(M) ** -0.5)
    errors = []
    for _ in range(100):
        scales = np.exp(rng.uniform(-2, 2, 4))
        result = invariant_operator(scales[:, None] * B / scales[None, :])
        assert result["valid"]
        errors.append(float(np.linalg.norm(result["operator"] - D) / np.linalg.norm(D)))
    assert max(errors) < 1e-12
    disconnected = graph_diagnostics(np.zeros((4, 4)))
    assert not disconnected["connected"] and disconnected["gap"] == 0
    modal = modal_sensitivities(K, M, [(1, 2)])
    b = np.array([0., 1., -1., 0.])
    finite = (eigh(K + 0.1 * np.outer(b, b), M, eigvals_only=True) - modal["omega_squared"]) / 0.1
    assert np.allclose(finite, modal["joint"][:, 0], rtol=1e-4)
    loss = stiffness_loss(K, b, 4000)
    inverse_error = float(np.linalg.norm(loss["inverse"] - np.linalg.inv(K - 4000 * np.outer(b, b))))
    assert inverse_error < 1e-12
    f = np.geomspace(2, 55, 100)
    slope = alpha_mixture(f ** -1.6, f ** -0.3, 1.6, 0.3, 0.1)
    assert np.all((slope > 0.3) & (slope < 1.6))
    no_change = compare_operators(D, D * 1.001, 0.01 * np.linalg.norm(D, 2), 0.01 * np.linalg.norm(D, 2))
    assert not no_change["resolved"]
    J, Z = rng.normal(size=(100, 2)), rng.normal(size=(100, 2))
    effective = nuisance_fisher(J, Z)
    assert np.linalg.eigvalsh(J.T @ J - effective).min() > -1e-10
    assert np.linalg.eigvalsh(effective).min() >= -1e-10
    x = rng.normal(size=(2, 1200))
    y = x + 0.2 * rng.normal(size=x.shape)
    evidence = relative_spectral_evidence(x, y, 120)
    gain = np.array([[0.3], [2.7]])
    changed_coordinates = relative_spectral_evidence(gain * x, gain * y, 120)
    assert abs(evidence["shape_distance"] - changed_coordinates["shape_distance"]) < 1e-8
    return {"passed": True, "roi_scale_trials": 100, "max_operator_relative_error": max(errors),
            "rank_one_inverse_absolute_error": inverse_error,
            "sensitivity_relative_tolerance": 1e-4,
            "disconnected_graph_gap": disconnected["gap"],
            "shared_coordinate_spectral_invariance_error": abs(evidence["shape_distance"] - changed_coordinates["shape_distance"]),
            "explanation": "Cambiar la escala de cada región no inventa daño en el operador. Las fórmulas de sensibilidad y flexibilidad coinciden con el cálculo directo."}


def synthetic_video_check(bootstrap_reps=24):
    start = time.perf_counter()
    sp = StructureSpec()
    _, truth, _ = simulate_structure(sp, 120., 40., seed=3)
    scene = default_scene(fault=False, duration_s=40., seed=3)
    boxes = {c.name: c.roi for c in scene.components}
    signals, _, _ = track_sequence(render_frames(scene, truth, np.zeros((2, 4800))), boxes)
    result = identify(signals, 120., sp.names, 0, (24.6, 49.2, 36.9),
                      bootstrap_reps=bootstrap_reps, common_scale=True, seed=2026)
    if not result["ok"]:
        return {"passed": False, "reason": result["reason"], "source": "fresh synthetic video through OpenCV"}
    M, _, K, _, _ = sp.matrices({})
    true_D = np.diag(np.diag(M) ** -0.5) @ K @ np.diag(np.diag(M) ** -0.5)
    error = float(np.linalg.norm(np.asarray(result["operator"]) - true_D, 2))
    relative = error / np.linalg.norm(true_D, 2)
    assert relative < 0.15, "synthetic operator recovery degraded"
    return {"passed": True, "source": "fresh synthetic video -> OpenCV -> SSI -> invariant operator",
            "operator_relative_error": float(relative), "operator_error": error,
            "operator": result["operator"], "true_operator": true_D.tolist(),
            "uncertainty": result["uncertainty"], "physical_diagnostics": result["physical_diagnostics"],
            "frequency_hz": result["freq_hz"], "wall_seconds": time.perf_counter() - start,
            "explanation": "En un video simulado conocemos la máquina original y podemos comparar la estimación con la verdad. Esto no equivale a validación industrial."}


def alpha_coverage_check(trials=200):
    covered = 0
    for i in range(trials):
        rng = np.random.default_rng(19000 + i)
        x = np.stack([colored_noise(1440, 1.6, 120., rng) for _ in range(2)])
        result = analyze_window(x, SpectralConfig())
        covered += int(result["ci_lo"] <= 1.6 <= result["ci_hi"])
    return {"trials": trials, "empirical_coverage": covered / trials,
            "nominal_coverage": 0.95, "scope": "exact stationary power law only; no tracker, codec or resonance",
            "explanation": "El intervalo se comprueba en señales ideales. Su cobertura real con compresión necesita una evaluación aparte."}


def run_checks(bootstrap_reps=24):
    return {"schema_version": "3.0", "software": {"python": platform.python_version(),
            "opencv": cv2.__version__, "numpy": np.__version__, "scipy": scipy.__version__},
            "algebra": algebra_checks(), "synthetic_video": synthetic_video_check(bootstrap_reps),
            "alpha_intervals": alpha_coverage_check(),
            "scope": "algebraic checks and one synthetic-video recovery; no field accuracy claim"}
