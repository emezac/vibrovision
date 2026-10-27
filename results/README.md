# Evidence index

## Recorded AWS execution — 26 October 2026

- `aws_deployment_verification.json`: HTTPS, OpenCV 5.0.0 on aarch64, authentication, source-release fingerprint, encrypted private S3 persistence, and 60 passing local tests at deployment verification.
- `aws_initial_comparison.json`: the initial deployed comparison, 45.5 seconds wall time.
- `aws_demo_comparison.json` and `aws_demo_comparison.png`: the fresh comparison captured for the film, job `7459d1077a62`, 43.9 seconds wall time and 35.8999× tone-amplitude ratio.
- `aws_agent_trace.json`: completed `fault` synthetic scenario, job `7b61959df6ab`. Its trace includes watch → independent cross-validation → evidence generation → request for human approval. It does not demonstrate field fault sensitivity.
- `aws_demo_capture.json`: provenance of the two recorded demo jobs.

These records contain no API key. The private `.aws-deploy/` folder is excluded from this repository. Job state in the running app is in memory, so a saved job URL may stop working after a restart; use the comparison button to create a new job.

## Mathematical validation and real recordings

`math_validation_v3.json` records algebraic identities, randomized image-scale changes, a fresh synthetic-video-to-OpenCV-to-SSI recovery, an idealized alpha interval experiment, and a local fan comparison. `math_validation_v3.png` visualizes that execution. The local run's approximately 36.7× ratio and the recorded AWS run's 35.9× ratio belong to separate executions and are kept distinct.

`liss_bridge_validation.json`, `liss_bridge.png` and `verification_v3.json` describe the notebook repair and the supplied MIT/LISS montage. The two supplied filenames represent identical content. This is not a healthy/damaged pair and does not validate a full stiffness reconstruction. See `docs/MIT_VIDEOS.md`.

## Earlier evaluation records

Files such as `eval_summary.json`, `longrun_summary.json`, `structure_summary.json`, `camera_summary.json`, `theory_summary.json` and `bench_dev-container-x86.json` are retained as historical experimental records. They are not a new evaluation of every v3 behavior, an industrial reliability study, or an AWS/COOL performance benchmark. Intermediate real-video and agent records also retain the conditions of their original execution; use the named AWS files above for the deployed demonstration.

Serialized working arrays and verbose raw logs are excluded. The evaluation scripts remain in `eval/` and `bench/` so experiments can be rerun.
