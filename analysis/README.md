# Context and representation analysis

Run the following commands from the code root. `DATASET_ROOT` points to the
runtime dataset created by `dataset_preparation.prepare_runtime`. Model-effect
analyses read actual verification outputs; preparation alone produces no scores.

## Context coverage and effects

```bash
python3 -m analysis.paper_audits coverage --dataset-root "$DATASET_ROOT" \
  --backbone llada --output-dir /path/to/coverage_llada
python3 -m analysis.paper_audits coverage --dataset-root "$DATASET_ROOT" \
  --backbone llama31 --output-dir /path/to/coverage_llama31
python3 -m analysis.paper_audits selector --dataset-root "$DATASET_ROOT" \
  --backbone llada --output-dir /path/to/selector_audit
python3 -m analysis.paper_audits groups --dataset-root "$DATASET_ROOT" \
  --output-dir /path/to/shared_groups
```

Coverage uses direct target-type dependencies, preceding same-file mathematical
commands and their actual final tokenized visibility. Missing dependency mappings
remain distinct from empty dependency sets. Selector audits retain all fixed
2,000 targets in the proof-success denominator; reference-proof dependencies are
used only for posthoc analysis, not to modify selector inputs.

Length analysis splits the 11,064 originally both-nonempty test targets into
four groups of 2,766 using complete pre-clipping LLaDA input length and sample ID
as a tie-breaker. Both backbones use the same target groups. Mixed-role groups
require explicit complete annotations supplied with `--mixed-annotations`.

For fixed-checkpoint context effects, add actual verifier-output files:

```bash
python3 -m analysis.paper_audits groups --dataset-root "$DATASET_ROOT" \
  --output-dir /path/to/context_effects \
  --predictions-env /path/to/env_verified.jsonl \
  --predictions-math /path/to/math_verified.jsonl \
  --predictions-full /path/to/full_verified.jsonl
```

Pairings hold checkpoint and decoding settings fixed. Differences use paired
source-file bootstrap intervals; missing attempts count as failures.

## Reference-conditioned linear CKA

```bash
python3 -m analysis.representation_analysis extract --help
python3 -m analysis.representation_analysis cka --help
```

Use the dataset's fixed 4,000 shared target IDs, sampled with seed 42. The central
25%-75% reference-proof window must be nonempty under both tokenizers, excluding
EOS. Runtime preparation binds those unchanged IDs to the actual token-ID files.

For LLaDA, uniformly mask `floor(L/2)` proof positions and deterministically
resample until the central window contains a masked token. Reuse the accepted
mask across all checkpoints and views. AR features use causal positions before
the corresponding proof tokens. EOS, padding, statement and context positions
are not pooled.

Average hidden states at blocks 8, 16, 24 and 32 over the specified proof
positions. Compute centered linear CKA on complete theorem-by-feature matrices,
not by averaging batchwise CKA. Run all four trained checkpoints with all four
input views for each backbone. Features retain target, mask, checkpoint and
tokenizer provenance. These are reference-conditioned forward probes, not
pre-generation or causal measurements.

## Tests

```bash
python3 -m unittest discover -s analysis -t .
```
