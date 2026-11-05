# Proof-blind context reconstruction

Compare original benchmark context with independently reconstructed context at
a fixed checkpoint. Both MC and EC can change; the theorem, tokenizer, test IDs
and generation limits stay fixed. This differs from selector ranking comparisons
that hold EC and the candidate pool fixed.

## Construct inputs

Run from the code root after `dataset_preparation.prepare_runtime` has created
`DATASET_ROOT`. Inputs include `selector/frozen_test_manifest.json`,
`command_mapping/targets.jsonl`, the complete source-command catalog and the two
local tokenizers. All 2,000 fixed selector test IDs are retained.

```bash
export DATASET_ROOT=/absolute/path/leancontext_runtime
export RECONSTRUCTION_ROOT=/absolute/path/context_reconstruction
python3 -m pip install -r requirements-tokenization.txt
python3 -m dataset_preparation.reconstruction \
  --dataset-root "$DATASET_ROOT" --output-root "$RECONSTRUCTION_ROOT" \
  --backbone both --plan
```

Replace `--plan` with `--execute` to construct inputs. The output directory must
be new and outside `DATASET_ROOT`. Construction needs no GPU or model weights.

## Selection and packing

1. Build a strictly preceding, same-file MC pool with source-named declarations.
   Direct theorem/lemma candidates require their source spelling to occur in the
   target statement; wrapped declarations follow the source eligibility rules.
   Current and later command bodies are excluded.
2. Rank candidates by Jaccard overlap between statement terms and declaration
   names/signatures; break ties by source distance and command ID. Declaration
   bodies may be supplied as complete context, but are not ranking features.
3. Reconstruct EC from active source scopes and relevant binding commands.
   Rebuild EC for every proposed MC addition using the target and proposed
   signatures/scopes, including required opener/closer cohorts and relevant
   variable, universe, include and omit commands.
4. Accept a complete MC addition only when its rebuilt EC and all four projected
   inputs fit 1,024 tokens under the actual tokenizer. Skip overlong additions
   and try later candidates; never truncate commands or the theorem statement.
5. Derive C0/CMath/CEnv/CFull from one selected Full union, without view-specific
   refill. Pack the original comparison by removing complete commands from the
   Full tail until all four projections fit, then project without refill.

Selection does not use the target reference proof, proof dependencies, model
outcomes or proof-length screening. EC reconstruction is conservative syntactic
analysis, not a minimal-environment or new Lean-elaboration certificate.
Unresolved scopes or over-budget mandatory EC produce explicit invalid inputs;
they do not remove targets from the fixed 2,000-target denominator.

## Outputs

```text
manifest.json
build_status.json
protocol.json
frozen_test_manifest.json
targets.jsonl
tokenized/{llada,llama31}/reconstruction/input_1024/pairs.jsonl
```

Each pair records original/reconstructed MC and EC, four prompt-only tokenized
views, candidate rankings, skipped candidates and validity. Target reference
proofs are not part of these generator inputs.

## Fixed-checkpoint evaluation

Set `CHECKPOINT_ROOT`, `MODEL_CACHE` and `MATHLIB_ROOT` as for regular evaluation.
From `lean_proof_training`, inspect the plan, then replace `--plan` with
`--num-gpus 16 --execute`. Repeat with `--backbone llama31`.

```bash
python3 -m evaluation.run_interventions reconstruction \
  --backbone llada --extension-root "$RECONSTRUCTION_ROOT" \
  --output-root /absolute/path/reconstruction_eval_llada --plan
```

The runner uses one fixed CFull-trained SFT checkpoint for original/reconstructed
inputs across all four views. It reuses identical C0 predictions rather than
sampling them twice. Valid inputs receive 32 samples at temperature 0.6 and one
separate greedy attempt; invalid inputs count as failed attempts. Input and
output are each limited to 1,024 tokens. Proofs are checked at the original target
location in the unchanged full Lean environment, not the reconstructed prompt
environment. Predictions, verification records and Pass@1/8/16/32 reports retain
all 2,000 targets; compare original and reconstructed outcomes within each view.
