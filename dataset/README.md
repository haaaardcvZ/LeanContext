# LeanContext Dataset

LeanContext is a Lean 4 proof-generation benchmark for studying the roles of
**mathematical context (MC)** and **environment context (EC)**. Each target is
paired with a statement, a reference proof, source metadata, and four aligned
context views. The dataset accompanies the [LeanContext framework](../README.md).

## Download

**[Download LeanContext dataset v1.0.0](https://github.com/haaaardcvZ/LeanContext/releases/download/dataset-v1.0.0/LeanContext-dataset-v1.0.0.zip)**

The full dataset is distributed as a GitHub Release asset. Extract the archive
to obtain `leancontext_dataset/`. This repository includes a readable
[example record](example.json); the full records and supporting resources are
in the download. See the [release metadata and SHA-256](release.json),
[statistics](statistics.json), and [validation summary](validation_summary.json).

## Dataset size

| Split | Targets | Targets with both MC and EC |
| --- | ---: | ---: |
| Train | 90,879 | 51,628 |
| Validation | 19,473 | 11,063 |
| Test | 19,474 | 11,064 |
| **Total** | **129,826** | **73,755** |

The dataset covers **4,469 Mathlib source files** and stores **519,304 view
records**: four views for every target. Proof families include Strict Term,
Ordinary Tactic, Hybrid, and Calc. The raw split counts above precede any
token-length filtering.

## Four context views

| View | Visible input |
| --- | --- |
| `C0` | Target statement only |
| `CMath` | Mathematical context followed by the target statement |
| `CEnv` | Environment context followed by the target statement |
| `CFull` | MC and EC merged in source order, followed by the target statement |

MC contains complete mathematical declarations. EC contains the selected
environment commands, such as namespaces, variables, notation, and scope
settings. All four views share the same target and reference proof. Prompts
separate command texts with blank lines and end with the target statement and
a newline.

The original benchmark context construction is **reference-assisted**. The
separate **proof-blind selector controls** use statement and source information
without consulting the target proof or generation results, with fixed cohorts
of 2,000 test targets and 200 validation targets. The four views study input
visibility; they are not four self-contained Lean projects.

## Record schema

`dataset.complete.with_four_views.json` is a standard JSON array. Each element
contains one target and its four saved views.

| Field | Contents |
| --- | --- |
| `sample_id`, `target_id`, `split` | Stable target identity and split membership |
| `statement`, `target` | Lean statement and reference proof |
| `proof_family`, `target_kind` | Proof style and declaration type |
| `mathematical_context`, `environment_context`, `full_context` | Context commands and their source metadata |
| `source_file`, `source_module`, `original_name` | Original Mathlib location and name |
| `target_command_uid`, `target_source_span`, `source_revision` | Command identity, source span, and revision information |
| `views.C0`, `views.CMath`, `views.CEnv`, `views.CFull` | Saved prompts, targets, context items, and command identifiers |

Context items carry command text, kind, source order, identifiers, and spans;
view-level context items also identify their `MC` or `EC` role. Reference-proof
verification metadata refers to the proof at its original source location in
the complete source environment.

## Use with the framework

Place `leancontext_dataset/` beside the repository, then run from the repository
root after installing `requirements-tokenize.txt`:

```bash
python3 -m pip install -r requirements-tokenize.txt
python3 -m dataset_preparation.prepare_runtime \
  --dataset-root ../leancontext_dataset \
  --output-root ../leancontext_runtime \
  --models llada llama31 --execute
```

The [runtime preparation tool](../dataset_preparation/prepare_runtime.py)
expects the extracted dataset directory, including the canonical JSON,
`selector/`, `command_mapping/`, `source_snapshot/`, and `tokenizers/` resources.
Use a new output directory. It exports the saved views and creates tokenized
runtime files; context is not selected again. Set `DATASET_ROOT` to the resulting
`leancontext_runtime/` directory when using the experiment configurations.

The preparation protocol uses a 1,024-token input limit and a separate limit
of 1,024 tokens for the complete proof plus one EOS token. It removes whole
context commands when packing inputs and excludes overlong complete
statements or proofs. The raw dataset retains the full statements and proofs.

## Included resources and licensing

The archive also includes source-command mappings, a Mathlib source snapshot,
fixed evaluation populations, selector inputs, tokenizer resources, analysis
subsets, and machine-readable statistics and manifests.

Mathlib-derived source content is distributed under **Apache License 2.0**;
see [LICENSE-Mathlib](LICENSE-Mathlib) and the archive's source attribution notices.
Bundled tokenizer assets retain their respective upstream provider terms.
**Built with Llama.** The archive includes the Llama 3.1 tokenizer's Community
License, attribution notice, and acceptable use policy.
