"""VibroVision inspection service (FastAPI).

Runs the OpenCV 5 pipeline + inspection agent on an uploaded video (or a
synthetic test scenario), exposes the agent trace, incidents with evidence,
and a human approval endpoint. Designed for a single AWS Graviton instance
(see infra/); logs are JSON and metrics use CloudWatch Embedded Metric Format.

Responsible operation:
* raw video is deleted right after processing; only ROI-level telemetry and
  evidence crops of the machine parts are kept (in memory, or S3 if configured)
* optional API key (env API_KEY) for every /v1 route; upload size limit
* the agent is advisory-only: escalation requires a human decision
"""
from __future__ import annotations

import json
import hmac
import logging
import os
import platform
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
import sys
import tempfile
import threading
import time
import uuid
from itertools import chain, islice
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vibrovision import __version__  # noqa: E402
from vibrovision.agent import AgentConfig, HumanGate  # noqa: E402
from vibrovision.monitor import REF  # noqa: E402
from vibrovision.pipeline import (DEFAULT_ASSETS, SCENARIOS, auto_regions, build_scenario,  # noqa: E402
                                  pick_reference, probe_reference, read_video, run_agent_session)
from vibrovision.simulate import generate_displacements, render_frames  # noqa: E402

API_KEY = os.environ.get("API_KEY", "")
ORIGIN_VERIFY_KEY = os.environ.get("ORIGIN_VERIFY_KEY", "")
RELEASE_SHA256 = os.environ.get("RELEASE_SHA256", "")
MAX_UPLOAD_MB = float(os.environ.get("MAX_UPLOAD_MB", "200"))
MAX_VIDEO_S = float(os.environ.get("MAX_VIDEO_S", "300"))
MAX_BUFFER_MB = float(os.environ.get("MAX_BUFFER_MB", "768"))
WORK_DIR = os.environ.get("WORK_DIR", os.path.join(tempfile.gettempdir(), "vibrovision"))
S3_BUCKET = os.environ.get("S3_BUCKET", "")
SNS_TOPIC_ARN = os.environ.get("SNS_TOPIC_ARN", "")
os.makedirs(WORK_DIR, exist_ok=True)

log = logging.getLogger("vibrovision")
logging.basicConfig(level=logging.INFO, format="%(message)s")


class RedactAccessKeys(logging.Filter):
    """Keep request diagnostics without sending artifact URL keys to CloudWatch."""
    def filter(self, record):
        if isinstance(record.args, tuple) and len(record.args) >= 3 and isinstance(record.args[2], str):
            path = record.args[2]
            try:
                url = urlsplit(path)
                query = parse_qsl(url.query, keep_blank_values=True)
                if any(name.lower() == "key" for name, _ in query):
                    safe = [(name, "[REDACTED]" if name.lower() == "key" else value) for name, value in query]
                    path = urlunsplit((url.scheme, url.netloc, url.path, urlencode(safe), ""))
            except ValueError:
                path = path.split("?", 1)[0]
            record.args = (*record.args[:2], path, *record.args[3:])
        return True


logging.getLogger("uvicorn.access").addFilter(RedactAccessKeys())


def jlog(event: str, **kw):
    log.info(json.dumps({"event": event, "ts": time.time(), **kw}, default=str))


def emf(metrics: Dict[str, float], **dims):
    """CloudWatch Embedded Metric Format: the awslogs driver turns this line into metrics."""
    names = [{"Name": k, "Unit": "None"} for k in metrics]
    doc = {"_aws": {"Timestamp": int(time.time() * 1000),
                    "CloudWatchMetrics": [{"Namespace": "VibroVision", "Dimensions": [list(dims)], "Metrics": names}]},
           **dims, **metrics}
    print(json.dumps(doc, default=float), flush=True)


app = FastAPI(title="VibroVision", version=__version__)


@app.middleware("http")
async def require_cloudfront_origin(request: Request, call_next):
    # CloudFront injects this header. Together with the origin-facing prefix
    # list, it prevents another distribution from bypassing our entry point.
    # Health remains accessible for the container's local readiness probe.
    if (ORIGIN_VERIFY_KEY and request.url.path != "/healthz"
            and not hmac.compare_digest(request.headers.get("x-origin-verify", "").encode(), ORIGIN_VERIFY_KEY.encode())):
        return JSONResponse(status_code=403, content={"detail": "use the HTTPS application endpoint"})
    return await call_next(request)


POOL = ThreadPoolExecutor(max_workers=int(os.environ.get("WORKERS", "1")))
JOBS: Dict[str, dict] = {}
INCIDENTS: Dict[str, str] = {}      # incident_id -> job_id
LOCK = threading.Lock()


def require_key(x_api_key: Optional[str] = Header(default=None), key: Optional[str] = Query(default=None)):
    """X-API-Key header (or ?key= for <img> tags) when API_KEY is configured."""
    if API_KEY and not any(hmac.compare_digest((value or "").encode(), API_KEY.encode()) for value in (x_api_key, key)):
        raise HTTPException(status_code=401, detail="missing or invalid X-API-Key")


# ----------------------------------------------------------------------------- job execution
def _healthmap(frame: np.ndarray, agent, path: str) -> None:
    img = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
    colors = {"CALIBRATING": (180, 180, 180), "NOMINAL": (60, 180, 60), "WATCH": (0, 200, 230),
              "SUSPECT": (0, 140, 255), "AWAITING_HUMAN": (40, 40, 230), "ESCALATED": (30, 30, 200)}
    for name, st in agent.rois.items():
        x, y, w, h = [3 * v for v in agent.mon.boxes[name]]
        c = colors.get(st.phase, (200, 200, 200))
        cv2.rectangle(img, (x, y), (x + w, y + h), c, 2)
        a = st.last.get("alpha", float("nan"))
        cv2.putText(img, f"{name} {st.phase}", (x, max(12, y - 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
        cv2.putText(img, f"alpha={a:.2f}", (x, max(24, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
    if REF in agent.mon.boxes:
        x, y, w, h = [3 * v for v in agent.mon.boxes[REF]]
        cv2.rectangle(img, (x, y), (x + w, y + h), (200, 120, 0), 1)
        cv2.putText(img, "static reference", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 120, 0), 1)
    cv2.imwrite(path, img)


def _run_job(job_id: str, source: dict) -> None:
    job = JOBS[job_id]
    jdir = os.path.join(WORK_DIR, job_id)
    os.makedirs(jdir, exist_ok=True)
    cfg = AgentConfig(evidence_dir=jdir)
    gate = HumanGate()               # queue mode: decisions arrive through the API
    job["gate"] = gate
    series: Dict[str, list] = {}
    t_start = time.time()

    def on_step(agent):
        job["agent"] = agent
        job["progress_s"] = round(agent.mon.t, 1)
        for n, st in agent.rois.items():
            if st.last:
                series.setdefault(n, []).append([round(agent.mon.t, 1), st.last.get("alpha"),
                                                 None if st.cusum is None else st.cusum.mu,
                                                 None if st.cusum is None else round(st.cusum.g, 3)])
        for inc in gate.pending:
            INCIDENTS.setdefault(inc, job_id)

    try:
        if source["kind"] == "comparison":
            from vibrovision.realvideo import compare_clips
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            paths = [os.path.join(root, "data", "real", name) for name in ("fan_nocoin.mp4", "fan_coin.mp4")]
            job["status"] = "running"
            job["comparison"] = compare_clips(paths, out_figure=os.path.join(jdir, "comparison.png"))
            job.update(status="done", wall_s=round(time.time() - t_start, 1))
            with open(os.path.join(jdir, "comparison.json"), "w") as fh:
                json.dump(job["comparison"], fh, allow_nan=False)
            _persist(job_id)
            return
        if source["kind"] == "synthetic":
            cfg.common_scale = True
            sc = build_scenario(source["scenario"], int(source.get("seed", 7)))
            if sc.structure is not None:
                cfg.structural = True          # coupled-structure bench: identify the machine Jacobian
            t, truth, shake = generate_displacements(sc)
            frames = render_frames(sc, truth, shake)
            first = next(render_frames(sc, truth, shake, 0, 1))
            fps = sc.fps
            rois = {c.name: c.roi for c in sc.components}
            regions = {c.name: c.region for c in sc.components}
            ref, assets = sc.reference_roi, DEFAULT_ASSETS
        else:
            fps, gen = read_video(source["path"], MAX_VIDEO_S)
            from vibrovision.realvideo import probe
            meta = probe(source["path"])
            dur = min(meta["duration_s"], MAX_VIDEO_S)
            if dur < 12:
                raise ValueError("single-clip inspection needs >=12 s; use paired comparison for short recordings")
            first_frames = list(islice(gen, max(2, int(3 * fps))))
            if not first_frames:
                raise ValueError("video has no readable frames")
            first = first_frames[0]
            regions = source.get("rois") or auto_regions(first, n=5)
            if source.get("reference"):
                ref = tuple(source["reference"])
            else:
                ref, ref_probe = pick_reference(first_frames, fps, regions)
                jlog("reference_probe", job=job_id, chosen=list(ref), rms=ref_probe)
            rois = {k: tuple(v) for k, v in regions.items()}
            regions = {k: _expand(v, first.shape, 16) for k, v in rois.items()}
            frames, assets = chain(first_frames, gen), {}
            prof = source.get("profile", "auto")
            if prof == "production" or (prof == "auto" and dur >= 1800):
                if dur < 660:
                    raise ValueError("production needs >=660 analyzed seconds (600 calibration + monitoring); raise MAX_VIDEO_S if appropriate")
                cfg = AgentConfig.production(evidence_dir=jdir)      # continuous-monitoring thresholds
            else:
                cfg.calib_s = min(cfg.calib_s, 0.6 * dur)
                cfg.window_s = min(cfg.window_s, max(4, dur / 3))
                cfg.max_window_s = min(cfg.max_window_s, max(cfg.window_s, dur / 2))
            cfg.structural = bool(source.get("structural"))
            cfg.common_scale = bool(source.get("common_scale"))
            buffer_mb = int((cfg.max_window_s + 2) * fps) * first.size / (1 << 20)
            if buffer_mb > MAX_BUFFER_MB:
                raise ValueError(f"frame buffer would need {buffer_mb:.0f} MB; use a smaller frame size or frame rate")
            job["profile"] = "production" if cfg.calib_s >= 600 else "demo"
            job["measurement_scope"] = "single clip, baseline learned inside this recording; not a healthy-vs-fault comparison"
        job.update(status="running", fps=fps)
        agent = run_agent_session(frames, fps, rois, regions, ref, cfg=cfg, gate=gate, assets=assets,
                                  on_step=on_step)
        job["agent"] = agent
        _healthmap(first, agent, os.path.join(jdir, "healthmap.png"))
        wall = time.time() - t_start
        job.update(status="done", wall_s=round(wall, 1), series=series)
        n_esc = sum(1 for e in agent.trace if e["tool"] == "request_human_approval")
        emf({"ProcessedSeconds": agent.mon.t, "WallSeconds": wall,
             "RoiFramesPerSecond": agent.mon.n * len(agent.mon.boxes) / max(agent.mon.track_seconds, 1e-9),
             "ToolCalls": len(agent.trace), "Escalations": n_esc}, Service="vibrovision", Arch=platform.machine())
        _persist(job_id)
    except Exception as e:  # noqa: BLE001
        job.update(status="error", error=str(e))
        jlog("job_error", job=job_id, error=str(e))
    finally:
        if source.get("kind") == "video":
            if "gen" in locals():
                gen.close()
            if os.path.exists(source["path"]):
                os.remove(source["path"])


def _expand(box, shape, m):
    x, y, w, h = box
    H, W = shape[:2]
    x0, y0 = max(0, x - m), max(0, y - m)
    return (x0, y0, min(W - x0, w + 2 * m), min(H - y0, h + 2 * m))


def _persist(job_id: str) -> None:
    if not S3_BUCKET:
        return
    try:
        import boto3
        s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION"))
        s3.put_object(Bucket=S3_BUCKET, Key=f"jobs/{job_id}/report.json",
                      Body=json.dumps(_job_view(job_id), default=str).encode())
        jdir = os.path.join(WORK_DIR, job_id)
        for f in os.listdir(jdir):
            s3.upload_file(os.path.join(jdir, f), S3_BUCKET, f"jobs/{job_id}/{f}")
    except Exception as e:  # noqa: BLE001
        jlog("s3_persist_failed", job=job_id, error=str(e))


def _job_view(job_id: str, full_trace: bool = False) -> dict:
    job = JOBS[job_id]
    out = {k: job.get(k) for k in ("id", "status", "source", "progress_s", "fps", "wall_s", "error", "created", "profile")}
    agent = job.get("agent")
    if job.get("comparison") is not None:
        out["comparison"] = job["comparison"]
        out["artifacts"] = sorted(os.listdir(os.path.join(WORK_DIR, job_id)))
    out["measurement_scope"] = job.get("measurement_scope")
    if agent is not None:
        out["summary"] = agent.summary()
        out["trace"] = [e for e in agent.trace if full_trace or e["level"] != "debug"]
        out["series"] = job.get("series")
        out["artifacts"] = sorted(os.listdir(os.path.join(WORK_DIR, job_id))) if os.path.isdir(
            os.path.join(WORK_DIR, job_id)) else []
    return _sanitize(out)


def _sanitize(value):
    if isinstance(value, dict):
        return {str(k): _sanitize(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_sanitize(v) for v in value]
    if isinstance(value, np.ndarray):
        return _sanitize(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _safe(o):
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# ----------------------------------------------------------------------------- routes
@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(os.path.dirname(__file__), "static", "index.html"), encoding="utf-8") as fh:
        return fh.read()


@app.get("/healthz")
def healthz():
    info = cv2.getBuildInformation()
    flags = [ln.strip() for ln in info.splitlines() if any(k in ln for k in ("Baseline", "Dispatched", "HAL", "Kleidi"))]
    return {"ok": True, "version": __version__, "opencv": cv2.__version__, "arch": platform.machine(),
            "cv_threads": cv2.getNumThreads(), "build_flags": flags, "jobs": len(JOBS),
            "release_sha256": RELEASE_SHA256}


@app.get("/v1/scenarios", dependencies=[Depends(require_key)])
def scenarios():
    return {"scenarios": list(SCENARIOS)}


@app.post("/v1/jobs", dependencies=[Depends(require_key)])
async def create_job(scenario: Optional[str] = Form(default=None), seed: int = Form(default=7),
                     profile: str = Form(default="auto"),
                     structural: bool = Form(default=False), common_scale: bool = Form(default=False),
                     rois: Optional[str] = Form(default=None), reference: Optional[str] = Form(default=None),
                     video: Optional[UploadFile] = File(default=None)):
    if profile not in ("auto", "demo", "production"):
        raise HTTPException(400, "unknown analysis profile")
    try:
        parsed_rois = json.loads(rois) if rois else None
        parsed_ref = json.loads(reference) if reference else None
        def valid_box(v):
            return isinstance(v, list) and len(v) == 4 and all(type(x) is int for x in v) and min(v[:2]) >= 0 and min(v[2:]) >= 8
        if parsed_rois is not None and (not isinstance(parsed_rois, dict) or not parsed_rois or
                                       not all(isinstance(k, str) and valid_box(v) for k, v in parsed_rois.items())):
            raise ValueError("ROIs need named integer [x,y,w,h] boxes")
        if parsed_ref is not None and not valid_box(parsed_ref):
            raise ValueError("invalid reference box")
        if structural and (not parsed_rois or len(parsed_rois) < 2):
            raise ValueError("structural analysis requires manually identified distinct parts (at least two ROIs)")
    except (ValueError, TypeError) as exc:
        raise HTTPException(400, str(exc)) from exc
    job_id = uuid.uuid4().hex[:12]
    if video is not None and video.filename:
        ext = os.path.splitext(video.filename)[1].lower()
        if ext not in (".mp4", ".mov", ".avi", ".mkv"):
            raise HTTPException(400, "video must be .mp4/.mov/.avi/.mkv")
        path = os.path.join(WORK_DIR, f"{job_id}{ext}")
        size = 0
        with open(path, "wb") as fh:
            while chunk := await video.read(1 << 20):
                size += len(chunk)
                if size > MAX_UPLOAD_MB * (1 << 20):
                    fh.close()
                    os.remove(path)
                    raise HTTPException(413, f"upload larger than {MAX_UPLOAD_MB} MB")
                fh.write(chunk)
        source = {"kind": "video", "path": path, "filename": video.filename, "profile": profile,
                  "rois": parsed_rois, "reference": parsed_ref, "structural": structural, "common_scale": common_scale}
    else:
        if scenario not in SCENARIOS:
            raise HTTPException(400, f"scenario must be one of {list(SCENARIOS)}")
        source = {"kind": "synthetic", "scenario": scenario, "seed": seed, "profile": "demo"}
    JOBS[job_id] = {"id": job_id, "status": "queued", "created": time.time(),
                    "source": {k: v for k, v in source.items() if k != "path"}}
    POOL.submit(_run_job, job_id, source)
    jlog("job_created", job=job_id, source=JOBS[job_id]["source"])
    return {"job_id": job_id}


@app.post("/v1/compare-example", dependencies=[Depends(require_key)])
def compare_example():
    """Fresh comparison of the two bundled fan clips; same implementation as Colab."""
    job_id = uuid.uuid4().hex[:12]
    source = {"kind": "comparison"}
    JOBS[job_id] = {"id": job_id, "status": "queued", "created": time.time(), "source": source}
    POOL.submit(_run_job, job_id, source)
    return {"job_id": job_id}


@app.get("/v1/jobs/{job_id}", dependencies=[Depends(require_key)])
def get_job(job_id: str, full_trace: bool = False):
    if job_id not in JOBS:
        raise HTTPException(404, "unknown job")
    return JSONResponse(_job_view(job_id, full_trace))


@app.get("/v1/jobs/{job_id}/artifacts/{name}", dependencies=[Depends(require_key)])
def artifact(job_id: str, name: str):
    if job_id not in JOBS or "/" in name or ".." in name:
        raise HTTPException(404, "not found")
    p = os.path.join(WORK_DIR, job_id, name)
    if not os.path.isfile(p):
        raise HTTPException(404, "not found")
    return FileResponse(p)


class Decision(BaseModel):
    approve: bool
    reviewer: str = "engineer"
    note: str = ""


@app.post("/v1/incidents/{incident_id}/decision", dependencies=[Depends(require_key)])
def decide(incident_id: str, d: Decision):
    job_id = INCIDENTS.get(incident_id)
    if job_id is None:
        raise HTTPException(404, "unknown incident")
    job = JOBS[job_id]
    with LOCK:
        if job["gate"].poll(incident_id) is not None:
            return {"ok": True, "already_decided": True}
        job["gate"].decide(incident_id, d.approve, d.reviewer, d.note)
    agent = job.get("agent")
    if agent is not None and job["status"] == "done":
        agent.apply_pending_decisions()
    jlog("human_decision", incident=incident_id, approve=d.approve, reviewer=d.reviewer)
    if d.approve and SNS_TOPIC_ARN:
        try:
            import boto3
            boto3.client("sns", region_name=os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")).publish(TopicArn=SNS_TOPIC_ARN, Subject=f"VibroVision {incident_id}",
                                        Message=json.dumps(job["gate"].pending[incident_id], default=str))
        except Exception as e:  # noqa: BLE001
            jlog("sns_failed", error=str(e))
    emf({"HumanDecisions": 1, "Approved": int(d.approve)}, Service="vibrovision")
    _persist(job_id)
    return {"ok": True}
