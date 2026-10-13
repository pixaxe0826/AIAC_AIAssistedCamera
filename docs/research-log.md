# Research Log

This sequence records changes in the research question, not a sequence of successful product releases. Ranking accuracy, supervised region localization, annotation-tool correctness, and on-device coaching are separate outcomes.

## 1. Ranker v0: can RGB predict a preferred photo?

An earlier crop-ranking pilot also failed to beat its constant-label baseline: test accuracy was **62.86%**, compared with **71.43%** for constant B, on only **7 held-out source images**. A subsequent annotation-semantics audit found that crop area, A/B position, and preference were structurally coupled. This crop-ranking result is separate from the same-scene photo experiment below.

An unchanged RGB-only composition scorer was evaluated on **10,897 pairs from 99 same-scene shooting groups**, using group-safe five-fold evaluation. Each pair appeared in exactly one held-out test fold; the aggregate results used complete out-of-fold (OOF) predictions.

| Metric | OOF result |
|---|---:|
| Pairwise accuracy | 51.99% |
| Group-macro accuracy | 49.03% |
| Group-bootstrap 95% CI for macro accuracy | 45.33–52.67% |
| Binary cross-entropy (BCE) | 1.002269 |
| Fair-random BCE | 0.693147 |
| Fold-training soft-prior BCE, fold mean | 0.693308 |

The interval includes chance, group-macro accuracy is below chance, and probability fit is worse than the constant baselines. A/B swap checks were consistent, so a structural ordering inconsistency did not explain the failure. Falling training loss alongside rising validation loss supported an overfitting concern.

**Decision: composition-preference signal NOT DEMONSTRATED.** The report recommended reviewing model and label formulation before expanding annotation at scale.

## 2. Geometry Ranker v1: does subject geometry help?

The geometry study evaluated person, face, pose, and derived geometric features. Imputation and normalization were fitted within each training fold, and feature joins checked original-image identity. The primary all-relations result used **5,968 pairs from 74 groups**.

| Metric | OOF result |
|---|---:|
| Pairwise accuracy | 53.28% |
| Group-macro accuracy | 55.11% |
| Group-bootstrap 95% CI | 49.41–60.69% |
| BCE | 0.701422 |
| Fair-random BCE | 0.693147 |
| Exploratory one-sided permutation p-value, 100 within-group permutations | 0.0693 |

The confidence interval includes chance, the permutation result is exploratory, and BCE does not beat the uninformed baseline. Group-level BEST top-1 retrieval matched the random ordering baseline.

**Decision: WEAK trend, not stable ranking generalization.** This geometry cohort differs from the RGB cohort, so the two tables are not a controlled estimate of an improvement from adding geometry.

A post-hoc BEST-versus-GOOD experiment reached **68.75% group-macro accuracy**, with a **58.17–78.69%** group-bootstrap interval, on **130 pairs from 52 groups**. It was selected after inspecting ablations and did not receive its own permutation test. The report treated it as a hypothesis for a bounded relation-semantics pilot, not confirmation.

## 3. Redefining supervision: where is the problem?

Earlier ranking labels expressed preferences between photos. The localized formulation instead records the region and type of an explicitly observed problem. A scalar winner/loser target and a localized problem annotation answer different questions; the earlier results do not prove why ranking failed.

The change in formulation introduced explicit BEST, PROBLEM, EXCLUDE, and UNLABELED states, localized problem regions, and a separate global-problem flag. Unmarked areas in a problem photo are unknown, not automatically clean negatives. Missing suggestions or removal of the last marked region do not imply BEST.

## 4. V3 annotator: preserve geometry and human intent

V3 is a local annotation tool, independent of model training. It stores typed problem regions in normalized original-image coordinates and derives the displayed tiles from those regions. Profile and grid changes do not overwrite canonical geometry.

The implementation includes persistent image identity, transactional SQLite saves, revisions, edit history, per-photo drawing tools, and typed regions that can overlap across problem types. Source photos remain unchanged. Group completion is explicit; a group can validly contain zero BEST photos, but must satisfy the documented problem and labeling rules.

The recorded implementation reports distinguish functional checks from acceptance. Native browser-zoom acceptance remained **NOT VERIFIED**. Correct annotation persistence is not evidence of better model generalization.

## 5. Sapiens2: supervised localization, then a POSE-only extension

### Recorded Sapiens2-0.4B localization pilot

The historical Sapiens2-0.4B pilot used **44 shooting groups**, split into **32 training, 6 validation, and 6 test groups**. There were **754 eligible images**. The bounded comparison used frozen spatial features, continued head training, and partial tuning of the last backbone block.

| System | Sapiens2-0.4B pilot: Test Local AUPRC |
|---|---:|
| Spatial prior | 0.115790 |
| Frozen | 0.588734 |
| Frozen Continue | 0.607465 |
| Partial Last-1 | 0.645877 |

The last-block result exceeded the equal-update continued-frozen result, but the report did not qualify an autonomous labeler:

- Human-marked regions are positives and eligible BEST tiles are negatives; unselected problem regions are masked as unknown. AUPRC on these labels does not establish discrimination between good and bad areas within a fully labeled problem photo.
- Review found person/torso shortcuts, broad responses over the body, and missed background-object problems.
- At threshold **0.5**, all **24 test BEST photos** produced local false positives in each of the three learned systems.
- The test set contains six groups; only five support the report's two-class per-group AP calculation. Group-bootstrap confidence intervals, additional seeds, and cross-validation were **NOT RUN**.
- The global task had only **4 positive and 24 negative** test examples, so its metrics had **INSUFFICIENT SUPPORT**.

**Decision: local learning signal DEMONSTRATED on the evaluated labels; overfit risk HIGH; AI-ASSIST LABELING: NOT READY.** This does not establish composition understanding, precise corrective advice, or on-device performance.

### Separate POSE-only workflow

The later documented extension targets Sapiens2-1B and a single-channel dense POSE head. It compares frozen and last-block-tuned RGB and RGB-plus-geometry candidates. Model selection and sparse-region policy use validation; an already exposed diagnostic holdout is not presented as a fresh unseen benchmark.

This extension is separate from the 0.4B pilot above. Those historical AUPRC values must not be attributed to the 1B POSE-only model. Qualification of the POSE-only model remains **in progress** in this showcase; no unsupported result is supplied.

## 6. AI Draft labeling: suggestions require human confirmation

AI Draft implements a review workflow with separate raw and working suggestions, edit/reject actions, revision checks, and durable approval receipts. Existing human content and exclusions take priority. Changed image identities, stale annotations, and completed or skipped groups block inappropriate draft application.

Only an explicit photo approval can write model-assisted, human-confirmed annotations. The generator cannot complete a group, and an empty suggestion is not a clean label. Regeneration cannot replace reviewed, edited, rejected, or approved drafts.

Functional workflow verification and model quality remain separate. Interaction timing is an estimate, not a measured reduction in review effort. Annotation quality, review-time savings, and autonomous readiness remain **in progress**.
