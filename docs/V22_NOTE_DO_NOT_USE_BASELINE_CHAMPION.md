# Important: v0.22 baseline `small_cnn` is NOT promoted

The first smoke runner emitted `VISION_REPRESENTATION_CANDIDATE` for `small_cnn` because it was the least-bad model under the validation ranking objective. The subsequent scientific review overrides that permissive mechanical label: validation weak macro-F1 was 0.07919 and outcome AUC was 0.49491; test weak macro-F1 was 0.08509 and outcome AUC was 0.48084.

Canonical scientific status: `NO_VISION_ENCODER_PROMOTED_BASELINE`.

Do not load this baseline into multimodal fusion, a trading policy, PAPER execution, or reinforcement learning. v0.22b is the first branch with explicit minimum acceptance gates.
