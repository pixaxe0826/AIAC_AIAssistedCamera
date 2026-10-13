# Group evaluation

Standalone extraction of the project's group-level ranking evaluation code. Runtime dependencies are limited to Python's standard library and NumPy; pytest is used for tests. The original arithmetic and seeded random generators are retained.

```sh
cd python/group-eval
python -m pip install -e '.[test]'
python -m pytest -q
```

The package exposes deterministic group-balanced folds, disjoint train/validation/test construction, OOF coverage and aggregation, RGB and geometry pair/group-macro metrics, and group-bootstrap percentile confidence intervals.

`PermutationData` contains only in-memory identities, labels, scene eligibility, and pair records. `permute_group_tiers` preserves each group's tier counts and regenerates cross-tier pairs. `permutation_summary` applies the original one-sided, plus-one correction to already-computed null-trial scores. Model refitting to produce those scores is outside this sample.

Fold construction accepts numeric group profiles with pair/image counts, relation counts, and A/B counts. Callers supply already-computed predictions to evaluation; the sample does not load files, photos, models, or checkpoints. Every test uses synthetic in-memory records.

The CI test checks that a deterministic bootstrap interval contains the point estimate for its hand-worked fixture. The original percentile algorithm is preserved; this is not a guarantee that every finite bootstrap sample contains its estimate.
