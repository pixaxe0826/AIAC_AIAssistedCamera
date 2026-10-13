# AI Camera Coach

## Highlights

- Built an iOS camera, Mac/iPhone geometry exporter, and ML research pipeline.
- Best recorded pilot result: Sapiens2-0.4B partial last-block tuning test Local AUPRC **0.646**, versus spatial prior **0.116**.
- Evaluation uses group-based splits, group-level confidence intervals, and sealed snapshots.
- Failed results and their limitations are documented alongside positive signals.

An iPhone camera foundation and an offline research workflow for learning where photo-composition problems occur.

## The problem

The project aims to help with portrait framing while keeping the camera responsive. Its first research question was whether a model could identify the preferred photo within a shooting group. The research then moved toward localized supervision: a preference score identifies a winner, but does not identify the region that needs attention.

The current work connects a native camera foundation, explicit human problem-region annotations, and a model-assisted review workflow. Learned on-device coaching is **in progress**.

## Project components

| Component | Role |
|---|---|
| iOS camera | Native preview and capture, camera controls, and a separate Apple Vision analysis branch. |
| Mac Geometry Exporter | Original-image person, face, and pose observations, with a shared Debug-only iPhone parity runner. |
| ML research | Group-based evaluation, immutable ranking snapshots, localized annotation, Sapiens2 experiments, and reviewable AI Draft suggestions. |

## Methods

- Keep camera operations independent of inference; analyze the latest available frame and reject stale results.
- Evaluate ranking on held-out shooting groups using out-of-fold predictions and group-level uncertainty.
- Store typed problem regions in normalized original-image coordinates, independent of the displayed annotation grid.
- Compare frozen Sapiens2 features with partial last-block tuning; develop a separate POSE-only dense prediction workflow.
- Keep AI Draft suggestions separate from canonical human labels until explicit human approval.

## Recorded results

| Experiment | Recorded result | Interpretation |
|---|---|---|
| RGB Ranker v0 | OOF pair accuracy **51.99%**; group-macro accuracy **49.03%**, with a group-bootstrap 95% CI of **45.33–52.67%**. | Reliable composition-preference signal **not demonstrated**. |
| Geometry Ranker v1 | OOF group-macro accuracy **55.11%**, 95% CI **49.41–60.69%**; exploratory permutation p-value **0.0693**. | **Weak** trend; stable ranking generalization not demonstrated. |
| Sapiens2-0.4B localization pilot: 44 groups, single fixed group holdout | Test Local AUPRC: spatial prior **0.115790**, frozen **0.588734**, continued frozen **0.607465**, partial last-block tuning **0.645877**. | Learning signal on the evaluated labels, with remaining shortcuts and false positives; not yet ready for reliable labeling assistance. |

Current work targets **Sapiens2-1B POSE-only**; model qualification is **in progress**.

These experiments use different cohorts and objectives. Their metrics are not a common leaderboard. [The research log](docs/research-log.md) explains the failures, supervision, and limits of each result.

## Current status

| Area | Status |
|---|---|
| Camera foundation | The project owner reported successful camera-foundation acceptance. |
| Live Apple Vision perception | Implemented; complete runtime measurements and repeated multi-device validation are still pending. |
| Mac/iPhone still-image export | An 80-image parity sample passed the recorded checks and human visual review; full-dataset extraction was not performed in that task. |
| V3 localized annotator | Implemented, including typed regions and explicit human review; native browser-zoom acceptance has not yet been verified. |
| AI Draft review workflow | Implemented separation and approval safeguards; annotation quality and measured review-time savings remain to be demonstrated. |
| POSE-only model qualification, Core ML integration, and learned on-device coaching | Work is in progress. |

## Technology stack

| Layer | Technologies |
|---|---|
| Native camera and exporter | Swift, SwiftUI, AVFoundation, Apple Vision, ImageIO, Swift Package Manager. |
| Research | Python, PyTorch, TorchVision, NumPy, Pillow, Sapiens2. |
| Annotation and review | SQLite, Python HTTP services, JavaScript, HTML Canvas, CSS. |

## Read more

- [Architecture](docs/architecture.md): camera, exporter, ML, and human-review boundaries.
- [Research log](docs/research-log.md): the progression from ranking to problem localization and AI Draft review.
- [Evaluation](docs/evaluation.md): group splits, sealed snapshots, reproducibility, and hardware evidence.

This repository is a documentation showcase.
