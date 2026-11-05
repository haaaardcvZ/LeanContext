#!/usr/bin/env python3
"""Sample shared targets, extract reference-conditioned states, and compute CKA.

Implements the reference-conditioned representation analysis in Appendix E.
The shared target manifest fixes theorem order across checkpoints and views.

Commands (run from the repository root):
  python -m analysis.representation_analysis sample --dataset-root ../dataset \
      --output cka_4000_ids.json
  python -m analysis.representation_analysis extract --dataset-root ../dataset \
      --manifest cka_4000_ids.json --backbone llama31 --training-view CFull \
      --checkpoint /path/to/local/checkpoint --output states_llama_full.pt
  python -m analysis.representation_analysis cka --features states_llama_full.pt \
      --output cka_llama_full.json

Repeat extraction for C0/CMath/CEnv/CFull checkpoints and both backbones. Each
extraction evaluates all four inference views. No generation occurs: these are
reference-conditioned forward probes, not pre-generation or causal evidence.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
from pathlib import Path
import random
import sys

from .cka import central_proof_positions, centered_linear_cka, conditional_mask_positions

VIEWS = ("C0", "CMath", "CEnv", "CFull")
BLOCKS = (8, 16, 24, 32)
BACKBONES = ("llada", "llama31")
EOS = {"llada": 126081, "llama31": 128001}
MASK = 126336
MODEL_PINS = {
    "llada": ("GSAI-ML/LLaDA-8B-Base", "0f2787f2d87eac5eed8a087d5ecd24277e6255b2"),
    "llama31": ("meta-llama/Llama-3.1-8B", "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"),
}
ANALYSIS_POLICY = {
    "protocol": "leancontext-nonempty-central-cka-v2",
    "sampling_population": "shared_four_view_test_IDs_with_nonempty_central_proof_window_under_both_tokenizers",
    "proof_length_excludes_terminal_EOS": True,
    "central_window": "one_based_ceil_0.25L_through_floor_0.75L_inclusive",
    "mask_count": "floor_0.5_times_proof_token_length",
    "mask_draw": "uniform_fixed_size_subset_conditioned_on_at_least_one_central_masked_position",
    "retry_seed": "first_theorem_seed_then_SHA256_leancontext-central-mask-v2:theorem_seed:zero_based_attempt",
    "no_generation_outcomes_or_semantic_dependencies_used_for_membership": True,
    "no_zero_vectors_or_pooling_window_fallback": True,
}


def policy_sha256() -> str:
    return hashlib.sha256(json.dumps(ANALYSIS_POLICY, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def source_path(root: Path, backbone: str, view: str) -> Path:
    return root / "tokenized" / backbone / "training" / "input_1024" / "views" / view / "test.jsonl"


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def make_sample_manifest(answer_lengths_by_backbone: dict[str, dict[str, int]], *,
                         sources: dict, seed=42, size=4000, mask_seed=42) -> dict:
    """Build IDs from final-release membership + answer_length, without weights.

    Inputs are already validated answer lengths INCLUDING one EOS. Each map
    must contain only that backbone's common membership across all four views.
    Root packaging can call this helper while streaming the final archive rows;
    ``sources`` must contain hashes of those final release view files, not old
    workspace bytes. This helper never changes training/evaluation membership.
    """
    if set(answer_lengths_by_backbone) != set(BACKBONES):
        raise ValueError("Answer-length evidence is required for both backbones")
    common = set.intersection(*(set(answer_lengths_by_backbone[name]) for name in BACKBONES))
    empty = {name: set() for name in BACKBONES}
    for name in BACKBONES:
        for sid in common:
            answer_length = answer_lengths_by_backbone[name][sid]
            if isinstance(answer_length, bool) or not isinstance(answer_length, int) or answer_length < 2:
                raise ValueError("Validated answer_length must include a nonempty proof and one EOS")
            try:
                central_proof_positions(answer_length - 1)
            except ValueError:
                empty[name].add(sid)
    eligible = common - set.union(*(empty[name] for name in BACKBONES))
    if len(eligible) < size:
        raise ValueError(f"Only {len(eligible)} jointly central-window-eligible targets, requested {size}")
    selected = random.Random(seed).sample(sorted(eligible), size)
    return {
        "schema": "leancontext-reference-conditioned-cka-sample-v1",
        "policy": ANALYSIS_POLICY.copy(),
        "policy_sha256": policy_sha256(),
        "sampling_seed": seed,
        "mask_seed": mask_seed,
        "sample_count": size,
        "population": ANALYSIS_POLICY["sampling_population"],
        "shared_population_count": len(common),
        "central_empty_excluded_by_backbone": {name: len(empty[name]) for name in BACKBONES},
        "joint_central_empty_excluded_count": len(common - eligible),
        "shared_central_window_eligible_population_count": len(eligible),
        "training_or_evaluation_membership_changed": False,
        "eligibility_uses_reference_proof_token_length_only": True,
        "selection_uses_generation_outcomes": False,
        "selection_uses_semantic_dependencies": False,
        "sources": sources,
        "target_ids": selected,
    }


def validate_sample_manifest(manifest: dict, *, expected_size=4000) -> None:
    if manifest.get("policy") != ANALYSIS_POLICY or manifest.get("policy_sha256") != policy_sha256():
        raise ValueError("Incompatible CKA manifest: regenerate with the current central-window policy")
    ids = manifest.get("target_ids", [])
    if manifest.get("sample_count") != expected_size or len(ids) != expected_size or len(set(ids)) != expected_size:
        raise ValueError(f"Expected exactly {expected_size} distinct shared target IDs")
    if not isinstance(manifest.get("mask_seed"), int):
        raise ValueError("Manifest must freeze the shared masking seed")
    if manifest.get("training_or_evaluation_membership_changed") is not False:
        raise ValueError("CKA eligibility must not alter training/evaluation populations")


def sample_targets(args) -> None:
    root = Path(args.dataset_root)
    sources, lengths = {}, {}
    for backbone in BACKBONES:
        common, by_id = None, {}
        for view in VIEWS:
            path = source_path(root, backbone, view)
            rows = {}
            for row in iter_jsonl(path):
                sid = row["sample_id"]
                if sid in rows:
                    raise ValueError("Duplicate target in source tokenized views")
                _, proof = reference_ids(row, backbone)
                rows[sid] = len(proof) + 1
                if sid in by_id and by_id[sid] != rows[sid]:
                    raise ValueError("Reference-proof token length differs across inference views")
                by_id[sid] = rows[sid]
            common = set(rows) if common is None else common & set(rows)
            sources[f"{backbone}/{view}"] = {"sha256": sha256(path), "records": len(rows)}
        lengths[backbone] = {sid: by_id[sid] for sid in common}
    manifest = make_sample_manifest(lengths, sources=sources, seed=args.seed, size=args.size, mask_seed=args.mask_seed)
    validate_sample_manifest(manifest, expected_size=args.size)
    write_json(Path(args.output), manifest)


def theorem_mask_seed(sample_id: str, global_seed: int) -> int:
    value = f"{global_seed}:{sample_id}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")


def reference_ids(row: dict, backbone: str) -> tuple[list[int], list[int]]:
    ids = row["input_ids"]
    start = row["answer_span"]["start"]
    end = row["answer_span"]["end_exclusive"]
    if start != row["prompt_length"] or end != len(ids):
        raise ValueError("Answer span/prompt alignment is invalid")
    answer = ids[start:end]
    if not answer or answer[-1] != EOS[backbone] or answer.count(EOS[backbone]) != 1:
        raise ValueError("Expected complete proof plus exactly one terminal EOS")
    # Remove EOS before deciding proof positions and do not pool EOS.
    return ids[:start], answer[:-1]


def select_states(hidden, prompt_length: int, proof_length: int, backbone: str, mask_positions):
    positions = central_proof_positions(proof_length)
    if backbone == "llama31":
        # Zero-based proof offset p predicts y_(p+1) from the preceding position.
        absolute = [prompt_length + p - 1 for p in positions]
    else:
        masked = set(mask_positions)
        absolute = [prompt_length + p for p in positions if p in masked]
        if not absolute:
            raise ValueError(
                "Fixed corruption has no masked central-window tokens; "
                "no silent re-masking, target dropping, or different pooling is permitted"
            )
    if min(absolute) < 0 or max(absolute) >= hidden.shape[1]:
        raise ValueError("Pooling positions exceed the model sequence")
    return hidden[0, absolute, :].float().mean(dim=0).cpu()


def model_loading_config(*, backbone: str, training_view: str, dataset_root: Path,
                         checkpoint: str, dtype: str, base_config: dict | None = None,
                         trust_remote_code: bool | None = None) -> dict:
    """Plan loading without torch, network access, or a checkpoint read.

    Full-SFT outputs use best/state/model.safetensors plus best/tokenizer; LoRA
    outputs additionally use best/model/adapter_config.json. They are not plain
    Hugging Face model directories. The shared evaluator loader supports both.
    """
    if backbone not in BACKBONES or training_view not in VIEWS:
        raise ValueError("Unknown backbone/training view")
    architecture = "dllm" if backbone == "llada" else "ar"
    if base_config is None:
        repository, revision = MODEL_PINS[backbone]
        config = {
            "architecture": architecture,
            "model": {
                "name_or_path": repository,
                "revision": revision,
                "tokenizer_name_or_path": str(dataset_root / "tokenizers" / backbone),
                "checkpoint_type": "auto",
                "trust_remote_code": backbone == "llada",
                "use_fast_tokenizer": True,
            },
        }
    else:
        config = copy.deepcopy(base_config)
        if str(config.get("architecture", architecture)).lower() != architecture:
            raise ValueError("Base configuration architecture disagrees with the backbone")
        if not config.get("model", {}).get("name_or_path"):
            raise ValueError("Base configuration must name the original base model")
        config["architecture"] = architecture
    config["model"]["checkpoint"] = str(Path(checkpoint).expanduser().resolve())
    config["model"]["dtype"] = dtype
    if trust_remote_code is not None:
        config["model"]["trust_remote_code"] = trust_remote_code
    config["analysis_training_view"] = training_view
    return config


def resolve_block_modules(model):
    """Find actual Transformer blocks, never confuse final norm with block 32."""
    roots = [(model, "")]
    if callable(getattr(model, "get_base_model", None)):
        roots.append((model.get_base_model(), "get_base_model()."))
    for root, prefix in roots:
        for path in ("model.layers", "model.transformer.blocks", "transformer.blocks"):
            current = root
            for name in path.split("."):
                current = getattr(current, name, None)
                if current is None:
                    break
            if current is not None and len(current) >= max(BLOCKS):
                return current, prefix + path
    raise ValueError(
        "Unsupported checkpoint module layout: cannot identify actual Transformer blocks"
    )


def extract(args) -> None:
    import torch

    # Share the same base+full-state/PEFT checkpoint contract as proof generation.
    # This avoids trying to load an Accelerate best/state directory as an HF repo.
    project = Path(__file__).resolve().parents[1] / "lean_proof_training"
    if str(project) not in sys.path:
        sys.path.insert(0, str(project))
    from evaluation.io_utils import load_config
    from evaluation.model_loader import load_model_and_tokenizer
    from evaluation.current_inputs import encode

    root = Path(args.dataset_root)
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    validate_sample_manifest(manifest)
    if args.mask_seed != manifest["mask_seed"]:
        raise ValueError("Mask seed must match the immutable shared sample manifest")
    ids = manifest["target_ids"]
    if len(ids) != 4000 or len(set(ids)) != 4000:
        raise ValueError("The paper analysis requires exactly 4,000 distinct shared targets")
    wanted = set(ids)
    rows = {}
    for view in VIEWS:
        path = source_path(root, args.backbone, view)
        expected = manifest["sources"][f"{args.backbone}/{view}"]["sha256"]
        if sha256(path) != expected:
            raise ValueError(f"Source file changed after sampling: {path}")
        rows[view] = {r["sample_id"]: r for r in iter_jsonl(path) if r["sample_id"] in wanted}
        if set(rows[view]) != wanted:
            raise ValueError(f"Missing sampled targets under {view}")

    masks, mask_attempt_counts = {}, {}
    # Validate every selected theorem; never drop a target after sampling. The
    # conditional central-mask policy only changes the seed draw.
    for sample_id in ids:
        canonical_proof = None
        for view in VIEWS:
            _, proof = reference_ids(rows[view][sample_id], args.backbone)
            central = central_proof_positions(len(proof))
            if canonical_proof is None:
                canonical_proof = proof
            elif canonical_proof != proof:
                raise ValueError("Reference proof differs across inference views")
        if args.backbone == "llada":
            mask, attempts = conditional_mask_positions(len(canonical_proof), theorem_mask_seed(sample_id, args.mask_seed))
            masks[sample_id] = mask
            mask_attempt_counts[sample_id] = attempts

    base_config = load_config(Path(args.base_config)) if args.base_config else None
    loading = model_loading_config(
        backbone=args.backbone, training_view=args.training_view, dataset_root=root,
        checkpoint=args.checkpoint, dtype=args.dtype, base_config=base_config,
        trust_remote_code=args.trust_remote_code,
    )
    model, tokenizer, loading_metadata = load_model_and_tokenizer(
        loading, architecture=loading["architecture"], device=torch.device(args.device),
    )
    if tokenizer.eos_token_id != EOS[args.backbone]:
        raise ValueError("Checkpoint tokenizer EOS disagrees with saved tokenized proof data")
    for sample_id in ids:
        for view in VIEWS:
            row = rows[view][sample_id]
            prompt, _ = reference_ids(row, args.backbone)
            if encode(tokenizer, row["prompt"], args.backbone) != prompt:
                raise ValueError("Checkpoint tokenizer differs from the sampled prompt tokens")
    layer_count = getattr(model.config, "num_hidden_layers", getattr(model.config, "n_layers", 0))
    if layer_count < max(BLOCKS):
        raise ValueError("Checkpoint has fewer Transformer blocks than the paper observes")
    modules, block_path = resolve_block_modules(model)
    captured = {}
    pool_context = {}
    handles = []
    for block in BLOCKS:
        def hook(_module, _input, output, block=block):
            hidden = output[0] if isinstance(output, tuple) else output
            if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
                raise ValueError("Unsupported Transformer block output")
            captured[block] = select_states(hidden, **pool_context)
        handles.append(modules[block - 1].register_forward_hook(hook))
    pooled = {view: {block: [] for block in BLOCKS} for view in VIEWS}
    with torch.inference_mode():
        for sample_id in ids:
            for view in VIEWS:
                prompt, proof = reference_ids(rows[view][sample_id], args.backbone)
                model_proof = list(proof)
                mask = masks.get(sample_id, [])
                if args.backbone == "llada":
                    for index in mask:
                        model_proof[index] = MASK
                seq = torch.tensor([prompt + model_proof], dtype=torch.long, device=args.device)
                pool_context.clear()
                pool_context.update(prompt_length=len(prompt), proof_length=len(proof),
                                    backbone=args.backbone, mask_positions=mask)
                captured.clear()
                # No padding is introduced. LLaDA custom forwards need not expose
                # the Hugging Face attention_mask/output_hidden_states API.
                kwargs = {"attention_mask": torch.ones_like(seq)} if args.backbone == "llama31" else {}
                outputs = model(seq, **kwargs)
                if set(captured) != set(BLOCKS):
                    raise ValueError("Not every requested Transformer block executed")
                for block in BLOCKS:
                    pooled[view][block].append(captured[block])
                del outputs

    for handle in handles:
        handle.remove()

    for view in VIEWS:
        for block in BLOCKS:
            pooled[view][block] = torch.stack(pooled[view][block])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema": "leancontext-reference-conditioned-hidden-states-v1",
        "backbone": args.backbone,
        "training_view": args.training_view,
        "checkpoint": args.checkpoint,
        "model_loading": loading_metadata,
        "resolved_model_loading_config": loading,
        "target_ids": ids,
        "sample_manifest_sha256": sha256(manifest_path),
        "sampling_seed": manifest["sampling_seed"],
        "mask_seed": args.mask_seed,
        "mask_positions": masks,
        "mask_attempt_counts": mask_attempt_counts,
        "mask_pattern_sha256": hashlib.sha256(json.dumps(masks, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "analysis_policy": ANALYSIS_POLICY.copy(),
        "analysis_policy_sha256": policy_sha256(),
        "blocks": list(BLOCKS),
        "actual_block_module_path": block_path,
        "sample_count": len(ids),
        "reference_conditioned": True,
        "generated_proofs_used": False,
        "per_example_normalization": False,
        "dimensionality_reduction": False,
        "actual_model_forward_performed": True,
        "features": pooled,
    }, output)


def compute_cka(args) -> None:
    import torch

    # Only load this tool's own local tensor artifacts, never untrusted pickles.
    path = Path(args.features)
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if artifact.get("analysis_policy") != ANALYSIS_POLICY or artifact.get("analysis_policy_sha256") != policy_sha256():
        raise ValueError("Feature artifact does not match the current CKA policy")
    ids = artifact["target_ids"]
    if len(ids) != 4000 or len(set(ids)) != 4000:
        raise ValueError("A complete, aligned 4,000-theorem feature artifact is required")
    results = []
    for block in BLOCKS:
        for left, right in itertools.combinations(VIEWS, 2):
            x = artifact["features"][left][block].numpy()
            y = artifact["features"][right][block].numpy()
            if x.shape[0] != len(ids) or y.shape[0] != len(ids):
                raise ValueError("Feature rows do not align with the manifest")
            value = centered_linear_cka(x, y)
            results.append({"block": block, "left_view": left, "right_view": right,
                            "sample_count": len(ids), "cka": value, "one_minus_cka": 1-value})
    write_json(Path(args.output), {
        "backbone": artifact["backbone"], "training_view": artifact["training_view"],
        "feature_artifact_sha256": sha256(path),
        "sample_manifest_sha256": artifact["sample_manifest_sha256"],
        "sampling_seed": artifact["sampling_seed"],
        "reference_conditioned": True, "complete_matrix_cka": True,
        "comparison_count": len(results), "results": results,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sampling = sub.add_parser("sample")
    sampling.add_argument("--dataset-root", required=True)
    sampling.add_argument("--size", type=int, default=4000)
    sampling.add_argument("--seed", type=int, default=42)
    sampling.add_argument("--mask-seed", type=int, default=42)
    sampling.add_argument("--output", required=True)
    sampling.set_defaults(func=sample_targets)
    extraction = sub.add_parser("extract")
    extraction.add_argument("--dataset-root", required=True)
    extraction.add_argument("--manifest", required=True)
    extraction.add_argument("--backbone", choices=BACKBONES, required=True)
    extraction.add_argument("--training-view", choices=VIEWS, required=True)
    extraction.add_argument("--checkpoint", required=True)
    extraction.add_argument("--base-config", help="Optional existing evaluation YAML; --checkpoint overrides its checkpoint")
    extraction.add_argument("--output", required=True)
    extraction.add_argument("--device", default="cuda")
    extraction.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    extraction.add_argument("--mask-seed", type=int, default=42)
    extraction.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=None,
                            help="Override the base configuration; default true only for the pinned LLaDA backend")
    extraction.set_defaults(func=extract)
    cka_parser = sub.add_parser("cka")
    cka_parser.add_argument("--features", required=True)
    cka_parser.add_argument("--output", required=True)
    cka_parser.set_defaults(func=compute_cka)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
