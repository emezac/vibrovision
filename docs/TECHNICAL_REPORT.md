# VibroVision: technical report

## Problem and intended users

Video can help maintenance engineers, researchers and educators inspect several visible regions without attaching a sensor to every location. The challenge is deciding which conclusions the footage supports. Camera movement, image scale, excitation, texture, compression and sampling rate can affect an apparent vibration change.

VibroVision measures motion with OpenCV 5, evaluates the evidence and uses a rule-based inspection agent to choose a subsequent measurement or request review. Work-order approval remains human, and the application does not actuate machinery.

## Image analysis and agent loop

The implementation uses feature selection, optical flow and subpixel tracking, camera compensation, ROI quality checks and independent ECC cross-validation. The paired fan pathway estimates housing motion relative to the background and compares spectra and tone-band displacement. A single uploaded clip instead learns its own baseline and must not be confused with a healthy/damaged pair.

The agent can relocate a region, lengthen a window, cross-check a suspicious estimate, render evidence and request human approval. In the recorded AWS synthetic `fault` job, a watch at 88 seconds was followed by independent cross-validation at 110 seconds and evidence generation and a request for approval at 133 seconds. Those are analyzed-video times, not wall-clock latencies. The saved trace is `results/aws_agent_trace.json`.

## What the mathematics adds

The method considers spectral slope, joint spectral relationships, modal identification and a compatible mechanical operator. These answer different questions; one slope or connectivity value is not a universal damage indicator.

Under a reciprocal linear model with positive diagonal mass and image-scale matrices,

$$M\ddot{x}+C\dot{x}+Kx=f,\qquad y=Sx+\eta,$$

the identified pixel-coordinate stiffness operator is

$$B_y=SM^{-1}KS^{-1}.$$

Reciprocal off-diagonal products cancel positive per-region image scales. Defining

$$D_{ii}=(B_y)_{ii},\qquad D_{ij}=-\sqrt{(B_y)_{ij}(B_y)_{ji}}$$

gives $D=M^{-1/2}KM^{-1/2}$ under those assumptions. Its eigenvalues describe squared natural frequencies. This does not recover arbitrary masses, forces or physical stiffness from any video. Adequate observable modes, compatible coordinates and an appropriate excitation model are necessary.

The code checks physical signs, positivity, conditioning and compatibility before accepting an operator. Empirical multichannel block-bootstrap radii inform whether a change or localization is supported. These are conditional estimates without certified coverage; they exclude model bias and nonstationarity. Unknown excitation can prevent a unique causal structural diagnosis.

A joint spectral matrix preserves relationships between channels that an individual scalar summary discards. Its current comparison is descriptive, without a calibrated field alarm threshold. Alpha can increase or decrease depending on the spectral change, so the method does not assert that every fault lowers alpha.

The [full mathematical appendix](MATHEMATICS.md), in Spanish, contains derivations, modal sensitivity, finite stiffness loss, graph cuts, perturbation bounds and implementation conditions. The [video mathematics guide](VIDEO_MATHEMATICS.md) explains the animated examples and their exact numbers.

## Evaluation and principal results

| Experiment | Observed result | Interpretation |
|---|---|---|
| 100 randomized image-scale changes | Maximum relative discrepancy ≈9.1×10⁻¹⁷ | Numerical check of the model identity |
| Synthetic video → OpenCV → SSI → operator | 2.03% relative operator error | One controlled benchmark, not industrial accuracy |
| Alpha intervals on 200 ideal signals | 94.5% empirical coverage | Excludes tracker, codec and resonance effects |
| Recorded AWS fan comparison | 35.9× amplitude ratio; 43.9 s wall time | Difference between these two recordings |
| Automated tests | 60 passed | Math, tracking, invariants, API, video lab and packaging |

The recorded comparison is in `results/aws_demo_comparison.json`. The earlier local mathematical validation is in `results/math_validation_v3.json`, including its separate approximately 36.7× fan result. Software context and hashes remain attached to the corresponding execution.

The fan clips last approximately 15 seconds and observe one housing. Their alpha ranges overlap. They demonstrate a large imposed perturbation, not early wear, field sensitivity or a complete stiffness reconstruction. The structural verdict is **abstain**.

The supplied MIT/LISS files were identical copies of a montage. Only one original bridge segment is analyzed. Disagreement between tracking methods remains visible. This is not an independent healthy/damaged experiment. [Source and limits](MIT_VIDEOS.md).

## AWS deployment and reproducibility

The app runs OpenCV 5.0.0 in a Python 3.13 Docker image on EC2 t4g.small/aarch64 in us-east-2. CloudFront provides browser HTTPS. Private S3 holds encrypted evidence, Secrets Manager holds the key, and CloudWatch records logs and metrics. See [the architecture diagram](architecture.png) and the templates and controller in `infra/`.

The origin uses restricted HTTP from CloudFront to EC2; this is not an end-to-end encrypted production video pipeline. Analysis requires API authentication. An instance role and temporary CLI authentication avoid embedded AWS access keys. The [deployment guide](../DEPLOYMENT_AWS.md) documents operational details.

Pinned requirements, tests, evaluation scripts and the lightweight Drive-based notebook are included. Follow the root README to run or regenerate evidence. The notebook was verified locally on the supplied files; that does not claim a new authenticated cloud Colab execution.

The app installs stock pip OpenCV. COOL integration, Graviton speedup and an AWS cost benchmark are not claimed. Historical records are indexed in [results/README.md](../results/README.md).

## Responsible use and limitations

Use recordings made with permission. Uploaded videos are deleted after processing; retained evidence has a separate lifecycle. The app records recommendations and waits for human approval.

Web job state is held in memory, analysis is sequential, and the deployed frame buffer is limited to 512 MiB. A restart loses web job state, while persisted S3 evidence remains. Further validation needs longer multipart recordings, independent sensor references, varying loads and camera conditions, uncertainty calibration, and separate evaluation of detection, localization, abstention and false alarms.
