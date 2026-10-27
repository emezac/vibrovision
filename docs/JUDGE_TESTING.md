# Testing instructions for the judges

## Access

- **Live application:** https://d30icy4clt3p0c.cloudfront.net/
- **Source code:** https://github.com/emezac/vibrovision
- **Judge access key:** supplied separately with the private testing credentials. No AWS account or installation is required for the web demo.

Open the application in a desktop browser. Paste the access key into **API key (if the server requires one)**, then click outside that field. Wait for **Ready.** and a populated scenario selector. The key authorizes analysis; the public page alone does not.

Allow a few minutes for each run. The demo processes one job at a time; a queued job may be waiting for another analysis. The required tests use bundled data, so no file upload or ROI configuration is necessary.

## Test 1: compare the real fan recordings

1. Click **Compare the two real fan videos**.
2. Wait until the job status is **done**.
3. Inspect **Real video comparison**, the two recording cards, the **Tone amplitude ratio**, and the comparison figure.

**Expected result:** substantially higher tone-band motion in the recording with the coin. The recorded AWS reference run produced an amplitude ratio of approximately **35.9×**. Values and processing time may vary by execution. The app explains that these short clips cannot identify a damaged component or reconstruct the complete machine structure. That withheld diagnosis is intentional.

This test reruns OpenCV analysis of the bundled recordings; it does not just display a saved result.

## Test 2: inspect the agent's decisions and human review

1. Leave the upload field empty. Select **fault** under **Synthetic test scenario**, keep **Seed = 7**, and click **Start inspection**.
2. Follow the component cards, the alpha chart and **Agent trace (perception → decision → action)**.
3. After completion, inspect the trace entries for `watch`, `cross_validate`, `render_evidence` and `request_human_approval`. Inspect the incident and its evidence images under **Incidents (human sign-off)**.
4. In this newly created test job, enter `JudgeTest` in the reviewer field and click **Reject**. The incident should become **REJECTED** and show the reviewer name.

**Expected result:** measurements lead to additional checks, evidence generation and a request for a person to decide. The agent does not approve its own work order. This is a controlled synthetic scenario, not a claim of field fault-detection accuracy. Trace timestamps are analyzed-video time, not processing latency.

Optional: run **healthy**, with seed 7, as a separate synthetic comparison. A quiet result is not proof that every possible fault is absent.

## Mathematical verification and local reproduction

Python 3.12+ and FFmpeg are required for this optional path. It works without the hosted API key or AWS credentials.

```sh
git clone https://github.com/emezac/vibrovision.git
cd vibrovision
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q
python eval/run_math_validation.py --real --out results/judge_math_validation.json
uvicorn service.app:app --host 127.0.0.1 --port 8000
```

On Windows, activate the environment with `.venv\Scripts\activate` instead. Open `http://127.0.0.1:8000` and repeat the two web tests. API authentication is disabled locally by default unless `API_KEY` is set in your environment.

The reference test suite has **60 passing tests**. Mathematical validation writes fresh JSON and a figure to `results/judge_math_validation.json` and `results/judge_math_validation.png`; its final output should report `algebra: true` and `synthetic_video: true`. This covers scale invariance, modal sensitivity, a controlled video-to-operator benchmark and the two real clips. It is a separate execution from the recorded AWS results.

See the [English technical report](TECHNICAL_REPORT.md), [video mathematics guide](VIDEO_MATHEMATICS.md), and [full mathematical appendix](MATHEMATICS.md) for assumptions and interpretation. The [4:33 demonstration](../media/vibrovision_hackathon.mp4) illustrates joint spectra, scale invariance, natural modes and empirical uncertainty. The short real fan test demonstrates an amplitude change; the conditional structural mathematics is evaluated separately.

## Optional Google Colab path

Open the [project notebook](https://colab.research.google.com/github/emezac/vibrovision/blob/main/notebooks/vibrometria_visual_alpha_v2.ipynb). Copy the repository's two `data/real/` MP4 files to your own Drive folder **My Drive / Colab Notebooks / Vibrovision**, or change the configured folder. Set `VIDEO_SIN_MONEDA="fan_nocoin.mp4"` and `VIDEO_CON_MONEDA="fan_coin.mp4"`, keep `PROBAR_VENTILADOR=True`, and set `PROBAR_MIT=False`. Run the cells and authorize mounting your own Drive. The owner's Drive is not needed.

## Troubleshooting

- **401 / invalid key / empty scenario list:** re-enter the supplied key and click outside the field. A comparison or synthetic run needs authenticated access.
- **Queued:** wait for the current analysis to finish before submitting another run.
- **Old job link returns 404:** open the base application URL and create a new job. Hosted job history is in memory and resets if the service restarts.
- **Hosted service unavailable:** use the local instructions above. The public [recorded evidence](../results/README.md) and demonstration remain available for review.

Endpoint access was checked on 26 October 2026: `/healthz` returned `ok: true`, OpenCV `5.0.0`, application `3.0.0` and `aarch64`. Both recorded demonstration jobs were accessible. These checks do not claim that every optional experiment was rerun at that time.
