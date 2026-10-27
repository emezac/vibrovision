# Demonstration video

[Download or play the expanded demonstration](vibrovision_hackathon.mp4) (4:33, 1920×1080, H.264/AAC, English narration in the application segment).

The first 49 seconds are the illustrated introduction. The companion shows the original fan recordings, recorded AWS interface, measured spectra, mathematical animations, a synthetic-test agent trace, and the deployed architecture. [English captions](vibrovision_hackathon_en.srt) accompany the narrated segment and are also embedded as a selectable subtitle track; the illustrated introduction already contains English explanations. [Chapter times](chapters.txt) locate each topic.

The expanded mathematical explanation covers joint spectra and phase (1:55), mass/damping/stiffness (2:14), fixed positive image-scale cancellation (2:31), the mechanical operator's eigenvalues and modes (2:53), and empirical uncertainty before localization (3:12). [The readable mathematical companion](../docs/VIDEO_MATHEMATICS.md) explains the exact examples, assumptions and limits.

The analytic animations are explicitly distinct from the fan measurements. [math-illustration.json](math-illustration.json) records the selected model and numerical checks against application functions, including 100 randomized scale trials. The uncertainty animation uses selected illustrative radii; it is not a fitted bootstrap experiment or a coverage guarantee.

The team can add a personal introduction of up to approximately 26 seconds while keeping the final edit below five minutes. No personal footage was supplied for this edit.

`verification.json` records the checked movie hashes, encoding, complete decoding, captions and source jobs. The standalone 3:44 companion referenced by that record remains a separate editing deliverable; this repository includes the combined video. Both movies were fully decoded without errors during verification.

The source fan clips are in `data/real/`. Numeric result callouts come from `results/aws_demo_comparison.json`; the spectral curves come from the earlier local analysis of the same recordings. The fan result remains a measured amplitude change with structural abstention. The agent trace is explicitly synthetic. No field sensitivity or COOL acceleration is claimed.

The cards, diagrams, transitions and score were generated for this project. The English narration uses the synthetic en-US-AriaNeural voice through edge-tts 7.3.1 and is not presented as a team member's voice. No InventorOS media or third-party background music is reused.
