#!/usr/bin/env python3
"""Plan, deploy and verify the local app using AWS CLI browser credentials.

No Git repository, local Docker installation or static AWS key is required.
Run `plan`, review its change sets, then run `execute` and `verify`.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Deployment:
    def __init__(self, args):
        self.args = args
        self.folder = ROOT / ".aws-deploy"
        self.folder.mkdir(mode=0o700, exist_ok=True)
        self.state_file = self.folder / f"{args.stack}.json"
        cli = shutil.which("aws") or str(Path.home() / ".local/bin/aws")
        self.base = [cli, "--profile", args.profile, "--region", args.region,
                     "--no-cli-pager", "--output", "json"]

    def aws(self, *args):
        result = subprocess.run(self.base + list(args), text=True, capture_output=True,
                                env={**os.environ, "AWS_PAGER": ""})
        if result.returncode:
            raise RuntimeError(result.stderr.strip())
        return json.loads(result.stdout) if result.stdout.strip() else {}

    def save(self, state):
        self.state_file.write_text(json.dumps(state, indent=2) + "\n")

    def load(self):
        state = json.loads(self.state_file.read_text())
        assert state["region"] == self.args.region and state["profile"] == self.args.profile
        identity = self.aws("sts", "get-caller-identity")
        assert identity["Account"] == state["account"], "AWS project changed since planning"
        return state

    def package(self):
        files = []
        for folder in ["vibrovision", "service", "bench", "data/real"]:
            files.extend(p for p in (ROOT / folder).rglob("*") if p.is_file()
                         and "__pycache__" not in p.parts and p.name != ".DS_Store")
        files += [ROOT / "requirements.txt", ROOT / "infra/Dockerfile"]
        # Explicit directory modes survive extraction under bootstrap's umask
        # 077. Implicit nested directories would otherwise become root-only
        # and remain unreadable by the container's unprivileged user.
        directories = {parent for file in files for parent in file.parents
                       if parent != ROOT and parent.is_relative_to(ROOT)}
        target = self.folder / "application.tar.gz"
        with target.open("wb") as out, gzip.GzipFile(fileobj=out, mode="wb", mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w") as tar:
                for file in sorted(directories) + sorted(files):
                    assert not file.is_symlink(), f"Unexpected source link: {file}"
                    info = tar.gettarinfo(str(file), str(file.relative_to(ROOT)))
                    info.uid = info.gid = info.mtime = 0
                    info.uname = info.gname = ""
                    info.mode = 0o755 if file.is_dir() else 0o644
                    tar.addfile(info, None if file.is_dir() else io.BytesIO(file.read_bytes()))
        return target

    def change_set(self, stack, template, parameters):
        name = "vibrovision-" + time.strftime("%Y%m%d-%H%M%S")
        params = [{"ParameterKey": key, "ParameterValue": value} for key, value in parameters.items()]
        result = self.aws("cloudformation", "create-change-set", "--stack-name", stack,
                          "--change-set-name", name, "--change-set-type", "CREATE",
                          "--template-body", "file://" + str(template),
                          "--parameters", json.dumps(params), "--capabilities", "CAPABILITY_IAM",
                          "--tags", "Key=Project,Value=VibroVision", "Key=Purpose,Value=Hackathon")
        arn = result["Id"]
        for _ in range(120):
            change = self.aws("cloudformation", "describe-change-set", "--change-set-name", arn)
            if change["Status"] in ("CREATE_COMPLETE", "FAILED"):
                break
            time.sleep(2)
        else:
            raise TimeoutError("Change-set planning did not finish; review it in CloudFormation")
        events = self.aws("cloudformation", "describe-events", "--change-set-name", arn)
        (self.folder / f"{stack}-validation.json").write_text(json.dumps(events, indent=2))
        if change["Status"] != "CREATE_COMPLETE":
            raise RuntimeError(change.get("StatusReason", "Change-set validation failed"))
        print(f"Validated plan: {stack} ({len(change.get('Changes', []))} resources)", flush=True)
        return arn

    def plan(self):
        if self.state_file.exists():
            raise RuntimeError("A deployment plan already exists. Use status or review it before planning again.")
        identity = self.aws("sts", "get-caller-identity")
        vpcs = self.aws("ec2", "describe-vpcs", "--filters", "Name=is-default,Values=true")["Vpcs"]
        if not vpcs:
            raise RuntimeError("No default VPC; select an existing public VPC/subnet before planning.")
        vpc = vpcs[0]["VpcId"]
        tables = self.aws("ec2", "describe-route-tables", "--filters", f"Name=vpc-id,Values={vpc}")["RouteTables"]
        subnets = self.aws("ec2", "describe-subnets", "--filters", f"Name=vpc-id,Values={vpc}")["Subnets"]
        choices = []
        for subnet in subnets:
            explicit = [t for t in tables if any(a.get("SubnetId") == subnet["SubnetId"] for a in t["Associations"])]
            effective = explicit or [t for t in tables if any(a.get("Main") for a in t["Associations"])]
            if any(r.get("DestinationCidrBlock") == "0.0.0.0/0" and r.get("GatewayId", "").startswith("igw-")
                   and r.get("State") == "active" for t in effective for r in t["Routes"]):
                choices.append(subnet)
        if not choices:
            raise RuntimeError("No subnet with a public internet-gateway route; no network was changed.")
        subnet = sorted(choices, key=lambda x: x["AvailabilityZone"])[0]["SubnetId"]
        prefixes = self.aws("ec2", "describe-managed-prefix-lists", "--filters",
                            "Name=prefix-list-name,Values=com.amazonaws.global.cloudfront.origin-facing")["PrefixLists"]
        if not prefixes:
            raise RuntimeError("CloudFront managed prefix list unavailable")
        archive = self.package()
        sha = digest(archive)
        release_stack = self.args.stack + "-releases"
        bucket = f"{release_stack}-{identity['Account']}-{self.args.region}"
        parameters = {"InstanceType": "t4g.small", "VpcId": vpc, "SubnetId": subnet,
                      "CloudFrontPrefixListId": prefixes[0]["PrefixListId"],
                      "ReleaseBucket": bucket, "ReleaseKey": f"releases/{sha}.tar.gz", "ReleaseSha256": sha}
        state = {"stack": self.args.stack, "profile": self.args.profile, "region": self.args.region,
                 "account": identity["Account"], "release_stack": release_stack,
                 "bucket": bucket, "parameters": parameters, "archive": str(archive),
                 "release_sha256": sha, "template_hashes": {
                     name: digest(ROOT / "infra" / name) for name in ["releases.yaml", "aws-demo.yaml"]}}
        state["release_change_set"] = self.change_set(release_stack, ROOT / "infra/releases.yaml", {})
        self.save(state)
        state["app_change_set"] = self.change_set(self.args.stack, ROOT / "infra/aws-demo.yaml", parameters)
        self.save(state)
        print("No application resources have been provisioned. Review the two change sets before execute.")
        print("Plan:", self.state_file)

    def wait_stack(self, stack):
        previous = None
        for _ in range(180):
            info = self.aws("cloudformation", "describe-stacks", "--stack-name", stack)["Stacks"][0]
            status = info["StackStatus"]
            if status != previous:
                print(stack, status, flush=True)
                previous = status
            if status == "CREATE_COMPLETE":
                return {o["OutputKey"]: o["OutputValue"] for o in info.get("Outputs", [])}
            if status not in ("CREATE_IN_PROGRESS", "REVIEW_IN_PROGRESS"):
                raise RuntimeError(f"{stack}: {status}; inspect CloudFormation describe-events")
            time.sleep(10)
        raise TimeoutError("Stack not ready; inspect CloudFormation and bootstrap logs")

    def execute(self):
        state = self.load()
        assert digest(state["archive"]) == state["release_sha256"], "Release changed after review"
        for name, sha in state["template_hashes"].items():
            assert digest(ROOT / "infra" / name) == sha, "Template changed after review"
        self.aws("cloudformation", "execute-change-set", "--change-set-name", state["release_change_set"])
        self.wait_stack(state["release_stack"])
        # Upload via s3api to avoid displaying a signed request or any credentials.
        self.aws("s3api", "put-object", "--bucket", state["bucket"],
                 "--key", state["parameters"]["ReleaseKey"], "--body", state["archive"],
                 "--server-side-encryption", "AES256", "--metadata", f"sha256={state['release_sha256']}")
        print("Private source release uploaded and fingerprinted.", flush=True)
        self.aws("cloudformation", "execute-change-set", "--change-set-name", state["app_change_set"])
        state["outputs"] = self.wait_stack(state["stack"])
        self.save(state)
        print("Ready for HTTPS verification:", state["outputs"]["Endpoint"], flush=True)

    def status(self):
        state = self.load()
        for stack in [state["release_stack"], state["stack"]]:
            info = self.aws("cloudformation", "describe-stacks", "--stack-name", stack)["Stacks"][0]
            print(stack, info["StackStatus"])
            if stack == state["stack"] and info.get("Outputs"):
                state["outputs"] = {o["OutputKey"]: o["OutputValue"] for o in info["Outputs"]}
        self.save(state)

    def verify(self):
        state = self.load()
        out = state["outputs"]
        endpoint = out["Endpoint"]
        secret = self.aws("secretsmanager", "get-secret-value", "--secret-id", out["AccessSecretArn"])
        key = json.loads(secret["SecretString"])["api_key"]
        access = self.folder / "vibrovision-access.json"
        fd = os.open(access, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as file:
            json.dump({"url": endpoint, "api_key": key}, file, indent=2)

        def request(path, *, auth=False, post=False):
            headers = {"X-API-Key": key} if auth else {}
            req = urllib.request.Request(endpoint.rstrip("/") + path, headers=headers,
                                         data=b"" if post else None)
            with urllib.request.urlopen(req, timeout=60) as response:
                data = response.read()
                return json.loads(data) if "application/json" in response.headers.get("Content-Type", "") else data.decode()

        health = request("/healthz")
        assert health["ok"] and health["arch"] == "aarch64" and health["opencv"].startswith("5.")
        assert health["release_sha256"] == state["release_sha256"]
        assert "VibroVision Inspector" in request("/")
        try:
            request("/v1/scenarios")
        except urllib.error.HTTPError as error:
            assert error.code == 401
        else:
            raise AssertionError("Unauthenticated API access was not rejected")
        assert request("/v1/scenarios", auth=True)["scenarios"]
        job = request("/v1/compare-example", auth=True, post=True)["job_id"]
        for _ in range(180):
            result = request("/v1/jobs/" + job, auth=True)
            if result["status"] in ("done", "error"):
                break
            time.sleep(2)
        assert result["status"] == "done", result.get("error", "Comparison timed out")
        comparison = result["comparison"]
        assert len(comparison["clips"]) == 2
        ratio = comparison["tone_amplitude_ratio"]
        assert math.isfinite(ratio) and ratio > 0
        for attempt in range(12):
            try:
                persisted = self.aws("s3api", "head-object", "--bucket", out["Bucket"],
                                     "--key", f"jobs/{job}/report.json")
                break
            except RuntimeError:
                if attempt == 11:
                    raise
                time.sleep(2)
        assert persisted["ServerSideEncryption"] == "AES256"
        public = self.aws("s3api", "get-public-access-block", "--bucket", out["Bucket"])
        assert all(public["PublicAccessBlockConfiguration"].values())
        report = {"endpoint": endpoint, "health": health, "unauthenticated_api": 401,
                  "real_comparison_job": job, "tone_amplitude_ratio": ratio,
                  "evidence_persisted": True, "evidence_bucket_private": True}
        (self.folder / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        print("Access key saved privately to:", access)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["plan", "execute", "status", "verify"])
    parser.add_argument("--profile", default="vibrovision")
    parser.add_argument("--region", default="us-east-2")
    parser.add_argument("--stack", default="vibrovision")
    args = parser.parse_args()
    if not (args.stack[:1].isalpha() and len(args.stack) <= 48
            and all(c.isalnum() or c == "-" for c in args.stack)):
        parser.error("Stack must start with a letter and contain at most 48 letters, digits or hyphens")
    getattr(Deployment(args), args.action)()


if __name__ == "__main__":
    main()
