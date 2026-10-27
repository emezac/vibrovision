import importlib
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("API_KEY", "k")
    monkeypatch.setenv("WORK_DIR", str(tmp_path))
    from fastapi.testclient import TestClient
    import service.app as app_mod
    importlib.reload(app_mod)
    return TestClient(app_mod.app)


def test_health_is_public(client):
    r = client.get("/healthz")
    assert r.status_code == 200 and r.json()["opencv"].startswith("5")


def test_api_key_required(client):
    assert client.get("/v1/scenarios").status_code == 401
    assert client.get("/v1/scenarios", headers={"X-API-Key": "k"}).status_code == 200


def test_cloudfront_origin_guard_still_requires_the_api_key(client, monkeypatch):
    import service.app as module
    monkeypatch.setattr(module, "ORIGIN_VERIFY_KEY", "origin-secret")
    assert client.get("/").status_code == 403
    assert client.get("/", headers={"X-Origin-Verify": "wrong"}).status_code == 403
    assert client.get("/", headers={"X-Origin-Verify": "origin-secret"}).status_code == 200
    assert client.get("/healthz").status_code == 200
    assert client.get("/v1/scenarios", headers={"X-Origin-Verify": "origin-secret"}).status_code == 401
    assert client.get("/v1/scenarios", headers={"X-Origin-Verify": "origin-secret", "X-API-Key": "k"}).status_code == 200


def test_health_reports_the_deployed_release(client, monkeypatch):
    import service.app as module
    monkeypatch.setattr(module, "RELEASE_SHA256", "a" * 64)
    assert client.get("/healthz").json()["release_sha256"] == "a" * 64


@pytest.mark.parametrize("query", ["key=secret&view=full", "%6bey=secret&view=full"])
def test_artifact_access_keys_are_redacted_from_request_logs(client, query):
    import logging
    import service.app as module
    record = logging.LogRecord("uvicorn.access", logging.INFO, "", 1, '%s - "%s %s HTTP/%s" %d',
                               ("client", "GET", "/v1/jobs/a/artifacts/report.json?" + query, "1.1", 200), None)
    assert module.RedactAccessKeys().filter(record)
    rendered = record.getMessage()
    assert "secret" not in rendered and "view=full" in rendered and "REDACTED" in rendered


def test_rejects_unknown_scenario_and_bad_upload(client):
    h = {"X-API-Key": "k"}
    assert client.post("/v1/jobs", data={"scenario": "nope"}, headers=h).status_code == 400
    r = client.post("/v1/jobs", files={"video": ("x.exe", b"abc")}, headers=h)
    assert r.status_code == 400


def test_unknown_incident_and_path_traversal(client):
    h = {"X-API-Key": "k"}
    assert client.post("/v1/incidents/nope/decision", json={"approve": True}, headers=h).status_code == 404
    assert client.get("/v1/jobs/nope/artifacts/..%2Fsecret", headers=h).status_code == 404


def test_rejects_invalid_profile_and_boxes(client):
    h = {"X-API-Key": "k"}
    assert client.post("/v1/jobs", data={"scenario": "healthy", "profile": "unknown"}, headers=h).status_code == 400
    assert client.post("/v1/jobs", data={"scenario": "healthy", "rois": "{"}, headers=h).status_code == 400
    assert client.post("/v1/jobs", data={"scenario": "healthy", "structural": "true"}, headers=h).status_code == 400


def test_example_comparison_uses_the_background_queue(client, monkeypatch):
    import service.app as module
    submitted = []
    monkeypatch.setattr(module.POOL, "submit", lambda *args: submitted.append(args))
    assert client.post("/v1/compare-example").status_code == 401
    response = client.post("/v1/compare-example", headers={"X-API-Key": "k"})
    assert response.status_code == 200 and submitted
    job_id = response.json()["job_id"]
    assert module.JOBS[job_id]["source"]["kind"] == "comparison"


def test_job_json_sanitizes_nan_and_infinite_evidence(client):
    import service.app as module
    module.JOBS["special"] = {"id": "special", "status": "done", "comparison": {"missing": float("nan"), "unstable": float("inf")}}
    os.makedirs(os.path.join(module.WORK_DIR, "special"), exist_ok=True)
    result = client.get("/v1/jobs/special", headers={"X-API-Key": "k"})
    assert result.status_code == 200
    assert result.json()["comparison"] == {"missing": None, "unstable": None}
