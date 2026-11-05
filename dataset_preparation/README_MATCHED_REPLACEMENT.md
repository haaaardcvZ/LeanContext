# Matched mathematical-context replacement

Construct three inputs for a fixed target: **Env** contains its original EC,
**Selected** adds its selected MC, and **Control** replaces MC with matched
declarations. The target and EC remain unchanged in all three conditions.

## Prepare inputs

Run from the code root after `dataset_preparation.prepare_runtime` has created
`DATASET_ROOT`. This directory must contain `benchmark/`, `command_mapping/`,
`source_snapshot/`, `tokenizers/`, and `private_posthoc_audit/kernel_inputs/`.
Construction uses the two local tokenizers and does not require model weights.

```bash
export DATASET_ROOT=/absolute/path/leancontext_runtime
export REPLACEMENT_ROOT=/absolute/path/matched_replacement
python3 -m pip install -r requirements-tokenization.txt
bash dataset_preparation/run_matched_replacement.sh "$DATASET_ROOT" "$REPLACEMENT_ROOT"
```

`REPLACEMENT_ROOT` must be a new directory outside `DATASET_ROOT`. The launcher
constructs the inputs, independently validates them, and finalizes their metadata.

## Matching method

1. Draw 2,500 originally both-nonempty test targets from sorted IDs using seed
   2026. Keep at most the first 2,000 jointly eligible targets in that draw order,
   without success-based screening or backfilling. The actual count may be lower.
2. Candidate replacements are complete mathematical declarations in the same
   source file, strictly before the target. Exclude the target and commands
   already present in its selected MC or EC.
3. Exclude recorded target-type and reference-proof dependencies using complete
   kernel evidence, including recorded same-command compiler auxiliaries.
   This exclusion uses the reference proof offline: replacement is not a
   proof-blind experiment. The target proof is never included in generator input.
4. Match each selected MC command to one distinct candidate with the exact
   declaration kind, the same conservative syntactic scope profile, and 90%-110%
   of its token length under the actual backbone tokenizer. A deterministic
   Hungarian assignment minimizes total absolute token-length deviation.
5. Require joined Control MC to remain within 90%-110% of Selected MC length and
   all three complete prompts to fit 1,024 tokens without truncation. MC slots,
   target text and EC are fixed. Both backbones share final target IDs, although
   their replacement command IDs may differ. One minimum-cost assignment is
   checked; rejection does not establish that every alternative assignment fails.

Scope matching is syntactic, not a new Lean elaboration certificate. Control
declarations are real preceding commands, but the assembled input is not newly
Lean-elaborated and is not guaranteed to be mathematically irrelevant.

## Outputs

```text
matched_replacement/frozen_test_manifest.json
matched_replacement/eligibility.jsonl
matched_replacement/pairs.jsonl
matched_replacement/validation_report.json
private_posthoc_audit/matched_replacement_dependency_exclusions.jsonl
tokenized/{llada,llama31}/matched_replacement/input_1024/pairs.jsonl
tokenized/{llada,llama31}/matched_replacement/input_1024/views/{Env,Selected,Control}/test.jsonl
```

## Fixed-checkpoint evaluation

Set `CHECKPOINT_ROOT`, `MODEL_CACHE` and `MATHLIB_ROOT` as for regular evaluation.
From `lean_proof_training`, inspect the plan, then replace `--plan` with
`--num-gpus 16 --execute`. Repeat with `--backbone llama31`.

```bash
python3 -m evaluation.run_interventions replacement \
  --backbone llada --extension-root "$REPLACEMENT_ROOT" \
  --output-root /absolute/path/replacement_eval_llada --grid --plan
```

The grid reuses fixed CMath-, CEnv- and CFull-trained SFT checkpoints across
Env/Selected/Control; it performs no retraining. Each target receives 32 sampled
attempts at temperature 0.6 and one separate greedy attempt at temperature 0.
Input and output each have a 1,024-token limit. Generated proofs replace the
original target proof and are checked in its unchanged original Lean environment.
Reports retain the actual common target denominator and include predictions,
verification records, per-target metrics and Pass@1/8/16/32. Compare Selected
against Control within each checkpoint, with Env as the no-MC reference.
