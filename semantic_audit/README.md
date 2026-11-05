# Lean dependency and ownership tools

`DependencyDump.lean` exports constants referenced by declaration types and
values. `OwnershipDump.lean` exports compiler-recorded ownership relations for
constructors, projections and auxiliary declarations.

Run from the built, pinned Mathlib project, with absolute output paths:

```bash
lake env lean --run /path/to/leancontext_code/semantic_audit/DependencyDump.lean \
  Mathlib /path/to/audit/declaration_dependencies.jsonl Mathlib
lake env lean --run /path/to/leancontext_code/semantic_audit/OwnershipDump.lean \
  /path/to/audit/declaration_ownership.jsonl
```

`build_source_name_candidates.py` and `audit_compiled_source_identity.py` accept
explicit audit, parser-output and source paths. `map_semantic_dependencies.py`
joins kernel declarations to original commands using source positions and
ownership evidence. `build_dependency_tables.py` aggregates the resulting
labels; `check_dependency_tables.py` independently checks those aggregates.
These mapping/table scripts read their prepared JSONL inputs from this directory.

The required mapping inputs are `semantic_inputs.jsonl`,
`semantic_source_commands.jsonl`, `source_canonical_name_candidates.jsonl`,
`declaration_dependencies.jsonl` and `declaration_ownership.jsonl`. Generating
these full audit inputs requires the matching parser outputs and compiled
Mathlib environment. Regular training and evaluation do not require rebuilding
the audit. Matched replacement and posthoc analysis read the scoped kernel
records already included in the companion dataset.

Reference-proof dependencies are for offline exclusion or posthoc analysis;
they must not enter proof-blind selection features or generator prompts. Source
identity checks and dependency labels are separate from Lean proof verification.
