"""A release must remain usable by the container user after secure extraction."""
import importlib.util
from pathlib import Path
import subprocess


def test_release_directories_survive_restrictive_bootstrap_umask(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / "infra" / "deploy_aws.py"
    spec = importlib.util.spec_from_file_location("deployment_under_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / "source"
    source.mkdir()
    for name in ["vibrovision/__init__.py", "service/static/index.html",
                 "bench/example.py", "data/real/video.mp4", "requirements.txt", "infra/Dockerfile"]:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("example")
    folder = source / ".aws-deploy"
    folder.mkdir(mode=0o700)
    monkeypatch.setattr(module, "ROOT", source)
    deployment = object.__new__(module.Deployment)
    deployment.folder = folder
    release = deployment.package()
    target = tmp_path / "extracted"
    target.mkdir()
    subprocess.run(["tar", "-xzf", str(release), "--no-same-owner", "--same-permissions", "-C", str(target)],
                   check=True, umask=0o077)
    for path in target.rglob("*"):
        assert path.stat().st_mode & 0o777 == (0o755 if path.is_dir() else 0o644)
    assert (target / "service/static/index.html").read_text() == "example"
