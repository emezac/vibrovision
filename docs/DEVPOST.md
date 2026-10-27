## Inspiration

A machine's movement carries information about how it is behaving. We wanted to explore how much of that information an ordinary camera could recover, making an initial vibration inspection more accessible to maintenance teams, researchers, and educators.

Our central question was: can a vision system measure a change, explain the evidence, and recognize when the recording is insufficient? Spectral theory gave us a way to study the rhythms in that movement and the relationships between visible parts.

## What it does

VibroVision turns video into motion measurements and an auditable inspection workflow. It uses OpenCV 5 to track displacement, account for camera movement, and evaluate measurement quality. Users can compare the included fan recordings, inspect an uploaded video, or run controlled synthetic scenarios.

A rule-based inspection agent uses the measurements to choose subsequent actions: reassess a region, extend the measurement window, cross-check an estimate, generate visual evidence, or request human review. A person must approve a work-order recommendation; the application does not control machinery.

The system also evaluates whether the available data supports a structural interpretation. It can report a change while withholding a damaged-part diagnosis.

Live application: https://d30icy4clt3p0c.cloudfront.net/ — analysis requires an access key supplied separately.

## How we built it

We built the analysis in Python with OpenCV 5, NumPy, SciPy, and Matplotlib, and exposed it through a FastAPI web application. OpenCV performs the image analysis, including feature selection, motion tracking, camera compensation, and independent measurement checks.

The mathematical pipeline combines power spectra, joint spectral relationships between channels, and conditional identification of a mechanical model. Under an explicit reciprocal linear model, we construct a mechanical operator that cancels fixed positive image-scale differences between regions. This helps prevent apparent size differences from being mistaken for mechanical changes. Model checks and empirical uncertainty estimates govern whether localization is supported.

Docker runs the application on an AWS Graviton EC2 instance in us-east-2. CloudFront provides the HTTPS entry point, Secrets Manager holds the access key, private S3 stores analysis evidence, and CloudWatch receives logs and metrics. CloudFormation defines the infrastructure.

A Google Colab notebook runs the same analysis code with videos read from Google Drive. Separate mathematical documentation explains the derivations, assumptions, and limitations.

## Challenges we ran into

The hardest problem was distinguishing machine motion from camera motion, drift, compression artifacts, and changes in excitation. Short recordings can reveal a strong difference while providing too little information to reconstruct a machine's structure.

We also had to separate quantities that answer different questions: spectral slope, natural frequencies, damping, and graph connectivity. None is a universal indicator of damage. Our uncertainty estimates remain empirical and conditional on the model.

Data provenance mattered too. Two supplied MIT showcase files were identical, and the montage included processed footage. We identified the duplicate and restricted analysis to an original segment rather than treating the files as independent experimental conditions.

Deployment exposed an additional issue: restrictive extraction permissions let the service pass its health check while its web page failed. We corrected the permissions and added a regression test for release packaging.

## Accomplishments that we're proud of

- Deployed a working OpenCV 5 application on AWS Graviton, with authenticated analysis and private, encrypted evidence storage.
- Captured a fresh AWS comparison showing a **35.9× tone-amplitude ratio** between the fan recordings without and with a coin. The recorded run completed in **43.9 seconds**. These are observations from that demonstration, not field sensitivity or speedup claims.
- Recovered the known mechanical operator with approximately **2.03% relative error** in a controlled synthetic-video benchmark, and checked the scale-invariance identity across 100 randomized rescaling trials.
- Recorded a synthetic inspection trace in which evidence changed the agent's behavior: monitoring, independent cross-validation, evidence generation, and a request for human approval.
- Passed **60 automated tests** and produced a reproducible notebook without embedded video files.

The real fan comparison explicitly abstains from identifying a damaged part: the recordings are short and observe one housing, rather than enough independent machine components.

## What we learned

A useful inspection system needs to explain what its measurements support. Image quality, sampling rate, camera motion, and physical assumptions all affect that conclusion.

Spectral analysis becomes more informative when we consider how regions move together and test whether a physical model is compatible with the observations. A clear abstention can be a useful outcome: it tells the inspector which additional evidence is needed.

We also learned to evaluate measurement quality, detection, localization, requests for review, and approved actions separately. A successful demonstration does not establish industrial reliability.

## What's next for Vibrovision

Our next priority is validation with longer recordings of multiple visible parts and independent reference measurements, such as accelerometers or a tachometer.

We want to test changing loads, illumination, compression, camera position, and closely spaced modes; calibrate uncertainty and false-alarm behavior; and evaluate detection, localization, and abstention separately.

On the application side, we plan to add a guided region-selection workflow, persistent job history, and clearer evidence review. The goal is to help an inspector understand what changed, how reliable the measurement is, and what to investigate next.
