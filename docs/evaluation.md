# Evaluation and Reproducibility

The showcase summarizes recorded experiments and implemented safeguards. It does not publish data, model weights, per-image outputs, or a runnable reproduction package. No new training or device testing was performed to prepare these documents.

## Split by the independent unit

Pairs from the same shooting group are correlated. The RGB and geometry ranking studies therefore split at group level, not by randomly distributing pair rows. Each pair contributes an OOF prediction only when its entire group is held out.

For snapshot and later localization workflows, exact-image duplicates and known parent or shooting links are connected into the same partition. Subject-safe evaluation requires trustworthy identity metadata; group-safe splitting alone does not establish it. The recorded geometry study left subject-safe generalization **NOT VERIFIED**.

Training-fold data alone determine imputation and normalization. Checkpoints, supported outputs, and suggestion thresholds use validation. Held-out test results do not justify reassigning groups or silently regenerating the split.

The RGB and geometry studies use different eligible cohorts. The Sapiens2 localization study uses a single fixed group holdout and a different objective. Comparisons must preserve these distinctions.

## Use uncertainty that matches the split

- Report pair accuracy and group-macro accuracy together. A large group can contribute many correlated pairs, so pair counts must not conceal group-level failures.
- Bootstrap shooting groups for the reported ranking confidence intervals, rather than treating pairs as independent observations.
- Preserve chance baselines and probability metrics. A small accuracy increase does not compensate for BCE worse than an uninformed predictor.
- Label post-hoc model or relation selection as exploratory. The Geometry Ranker permutation p-value is not a preregistered confirmatory result.
- Distinguish OOF evaluation from a single holdout. The localization pilot's additional seeds, cross-validation, and group-bootstrap CI were **NOT RUN**.

## Sealed snapshots

The implemented ranking snapshot pipeline captures completed human annotations, stable image identities, deterministic pair and ordered-partition manifests, fixed splits, provenance, and validation output. An inventory records payload hashes and sizes.

Publication checks the captured source revision for concurrent changes, validates the payload, and seals it before an atomic directory publication. Repeating the same publication verifies and reuses an identical snapshot; damaged content is rejected instead of overwritten. A consumer validates the seal, schemas, identities, split coverage, and deterministic manifest derivation, then rechecks original-image hashes.

The label-content digest and provenance-envelope digest have different purposes. Reformatting a source can alter the provenance envelope without changing completed label content. Recorded source revision time is kept separate from execution time.

This is application-level immutability and integrity checking. It is not an immutable backup of external photos, a digital signature, or a hardware durability guarantee. Originals must remain available and unchanged. Ranking-tier snapshots and V3 localized snapshots retain separate contracts; they are not interchangeable training inputs.

The historical ranking-snapshot input-consumer report records implementation and synthetic validation, with real training through that consumer **NOT RUN**. Snapshot tooling alone does not demonstrate a trained model or a usable current dataset.

## Supervision and AI Draft provenance

Explicit marked problem regions are positive supervision. Unmarked regions in a problem photo remain unknown; EXCLUDE and UNLABELED are not training labels. BEST supplies clean supervision only under the experiment's declared eligibility rules. Unordered ranking tiers do not define an order within a tier.

AI-generated suggestions remain outside the canonical human store until explicit review and approval. Confirmed model-assisted labels retain their origin. Draft generation, annotation editing, photo approval, and group completion are distinct operations. A draft is not ground truth merely because it was generated successfully.

## Reproducibility rules

Recorded experiments retain code revision, configuration, schema and preprocessing versions, data and split hashes, seeds, model identity, checkpoint selection, metrics, qualitative failure analysis, and known limitations. Official backbone identity and cached-feature identity are verified before reuse. Missing evidence remains unavailable or **NOT VERIFIED**.

Deterministic split assignment does not guarantee bitwise-identical CUDA training. The RGB pilot explicitly recorded that exact CUDA training repeatability was not guaranteed. Synthetic or tiny-set checks establish implementation behavior; they do not establish generalization or device acceptance.

## Hardware evidence from the localization pilot

The following historical training-stage measurements are from the recorded Sapiens2-0.4B pilot on a Tesla T4. Memory values are in GiB. These are not measurements of the later POSE-only model or of an iPhone.

| Stage | Peak Torch allocated | Peak Torch reserved | Min CUDA free | Peak NVML used | Peak host RAM used | Mean GPU utilization | Peak temperature | Approx. speed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen | 1.632 | 1.770 | 12.652 | 2.348 | 2.784 | 47.76% | 79 C | 31.21 steps/s |
| Frozen Continue | 1.632 | 1.770 | 12.652 | 2.348 | 2.803 | 46.82% | 83 C | 30.00 steps/s |
| Partial Last-1 | 2.003 | 2.139 | 12.283 | 2.717 | 2.822 | 68.71% | 82 C | 12.53 steps/s |

Torch allocator peaks are reset per stage; NVML values are sampled device-level maxima. Host RAM is system usage, not process RSS. Windows counters are unavailable in that Linux runtime. Approximate step rates include checkpoint and validation intervals and are not isolated forward-pass benchmarks.

The source report classified memory pressure as LOW for this workload, with no OOM and no sustained increase in live allocation at epoch endpoints. It did not establish that group size caused memory peaks, exclude host/shared paging directly, or isolate a storage bottleneck. Larger-model headroom, Windows residency, and later-model speed remain **NOT VERIFIED** by these measurements.

## Product and data boundaries

Camera acceptance, live Vision acceptance, exporter parity, research model quality, and human review benefits are separate gates. An 80-image exporter parity sample does not certify full-dataset extraction or detector accuracy. Live Vision retains its conditional runtime-evidence boundary.

Learned on-device coaching, production model integration, and qualified autonomous labeling are **in progress**.
