import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vibrovision.mechanics import invariant_operator, graph_diagnostics, compare_operators, modal_sensitivities, stiffness_loss
from vibrovision.structure import StructureSpec, identify, simulate_structure, StructuralWatch
from vibrovision.validation import algebra_checks
from vibrovision.spectralmatrix import relative_spectral_evidence
from vibrovision.theory import nuisance_fisher


def test_algebra_with_independent_coordinate_scales():
    assert algebra_checks()["passed"]


def test_nonreciprocal_model_is_not_certified_by_symmetrization():
    B = np.array([[4., -1., -1.], [-1., 4., -1.], [-8., -1., 4.]])
    assert not invariant_operator(B)["valid"]


def test_isolates_and_disconnected_components_are_not_connected():
    W = np.array([[0., 3., 0.], [3., 0., 0.], [0., 0., 0.]])
    g = graph_diagnostics(W, ["a", "b", "c"])
    assert not g["connected"] and len(g["components"]) == 2
    assert abs(g["gap"]) < 1e-10 and g["conductance"] == 0


def test_close_modes_do_not_get_a_stable_vector_label():
    base = np.diag([1., 1.00001, 5.])
    current = base + np.diag([0.02, -0.02, 0.])
    result = compare_operators(base, current, 0.001, 0.001)
    assert result["resolved"] and not result["location_stable"]
    assert not compare_operators(base, current)["resolved"]


def test_short_spectral_input_is_rejected():
    with pytest.raises(ValueError):
        relative_spectral_evidence(np.ones((2, 10)), np.ones((2, 10)), 30.)


def test_unknown_roi_scales_do_not_enable_mount_interpretation():
    sp = StructureSpec()
    _, truth, _ = simulate_structure(sp, 120., 40., seed=3)
    scaled = {n: v * s for (n, v), s in zip(truth.items(), [0.6, 1.7, 0.8, 2.1])}
    r = identify(scaled, 120., sp.names, 0, (24.6, 49.2, 36.9))
    assert r["ok"] and not r["mount_interpretable"] and r["rayleigh_lower"] is None
    M, _, K, _, _ = sp.matrices({})
    D = np.diag(np.diag(M) ** -0.5) @ K @ np.diag(np.diag(M) ** -0.5)
    assert np.linalg.norm(np.asarray(r["operator"]) - D) / np.linalg.norm(D) < 0.08


def test_unavailable_uncertainty_leads_to_abstention():
    sp = StructureSpec()
    _, truth, _ = simulate_structure(sp, 120., 70., seed=3)
    watch = StructuralWatch(sp.names, 120., (24.6, 49.2, 36.9), axes=(0,), bootstrap_reps=0)
    for end in (40, 50, 60, 70):
        event = watch.update({n: v[:, (end-40)*120:end*120] for n, v in truth.items()}, end)
    assert event["ok"] and event["change"] is None and event["decision"] == "abstain"


def test_eliminating_photometric_nuisance_cannot_add_information():
    rng = np.random.default_rng(1)
    J, Z = rng.normal(size=(20, 2)), rng.normal(size=(20, 2))
    covariance = np.eye(20) + 0.2 * np.ones((20, 20))
    F = nuisance_fisher(J, Z, covariance)
    known = J.T @ np.linalg.solve(covariance, J)
    assert np.linalg.eigvalsh(known - F).min() >= -1e-10


def test_confirmation_does_not_combine_different_locations():
    from types import SimpleNamespace
    from vibrovision.agent import InspectionAgent, AgentConfig, HumanGate
    agent = InspectionAgent.__new__(InspectionAgent)
    agent.cfg = AgentConfig(struct_confirm=2)
    agent.mon = SimpleNamespace(t=100.)
    agent.gate = HumanGate()
    agent.struct_hits, agent.struct_incident = 0, None
    agent._check_forcing = lambda event: None
    agent._log = lambda *args, **kwargs: None
    for part in ("A", "B", "B"):
        event = {"ok": True, "phase": "watching", "decision": "localized_change",
                 "change": {"kind": "mount", "part": part, "change": -0.3}}
        agent._call = lambda *args, **kwargs: (event, 0)
        agent._step_structure()
        if len(agent.gate.pending):
            assert part == "B" and agent.struct_hits == 2
        else:
            assert agent.struct_hits == 1
    assert len(agent.gate.pending) == 1
