"""Fixed-checkpoint crossing, selector, replacement, and reconstruction evaluations.

Default is a read-only JSON plan. --execute creates fresh result directories
and runs source indexing, gold smoke test, GPU generation and Lean checking.
Replacement uses a separately generated extension and its actual frozen IDs.
"""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any
from .io_utils import load_config, read_jsonl, write_json, write_jsonl
from .replacement_inputs import (CONDITIONS as REPLACEMENT_CONDITIONS,
                                 TRAIN_VIEWS as REPLACEMENT_TRAIN_VIEWS,
                                 extension_info, validate_replacement_artifacts)
from .reconstruction_inputs import (VARIANTS as RECONSTRUCTION_VARIANTS,
                                    extension_info as reconstruction_extension_info,
                                    validate_reconstruction_artifacts)
from .replacement_inputs import digest

VIEWS = ("C0", "CMath", "CEnv", "CFull")
POLICIES = ("baseline", "dependency_prioritized")


def checkpoint_for(base: dict[str, Any], train_view: str) -> str:
    if train_view not in VIEWS:
        raise ValueError("unknown training view")
    checkpoint = Path(str(base["model"]["checkpoint"]))
    if checkpoint.name != "best" or not checkpoint.parent.name.startswith("full_sft_"):
        raise ValueError("base configuration must specify a canonical full_sft_VIEW/best checkpoint")
    return str(checkpoint.parent.parent / f"full_sft_{train_view}" / "best")


def crossing_config(base: dict[str, Any], *, backbone: str, train_view: str,
                    inference_view: str, output_root: Path) -> dict[str, Any]:
    if backbone not in ("llada", "llama31") or inference_view not in VIEWS:
        raise ValueError("unknown backbone/inference view")
    config = copy.deepcopy(base)
    config.pop("_config_path",None)
    config["model"]["checkpoint"] = checkpoint_for(base,train_view)
    path = Path(str(config["data"]["tokenized_input_file"]))
    if path.parent.name not in VIEWS:
        raise ValueError("base tokenized input must be a current training VIEW/test.jsonl")
    config["data"]["tokenized_input_file"] = str(path.parent.parent / inference_view / path.name)
    config["generation"]["input_format"] = "training_view"
    config["generation"]["context_view"] = inference_view
    config["evaluation"].update(require_all_targets=True,expected_population_size=11064,max_samples=0)
    config["output"]["directory"] = str(output_root / "crossing" / backbone / f"train_{train_view}" / f"infer_{inference_view}")
    config["intervention"] = {"kind":"fixed_checkpoint_crossing","train_view":train_view,
                              "inference_view":inference_view,"population":11064,
                              "protocol_defaults":"controlled_fixed_checkpoint"}
    return config


def selector_config(base: dict[str, Any], *, backbone: str, train_view: str,
                    policy: str, pairs_file: Path, input_limit: int,
                    output_root: Path) -> dict[str, Any]:
    if policy not in POLICIES or input_limit not in (256,512,1024):
        raise ValueError("unknown selector policy/input budget")
    config=copy.deepcopy(base)
    config.pop("_config_path",None)
    config["model"]["checkpoint"] = checkpoint_for(base,train_view)
    config["data"]["tokenized_input_file"] = str(pairs_file)
    # Pair rows themselves define the immutable2000-ID population.
    config["data"]["allowed_ids_file"] = str(pairs_file)
    config["generation"].update(input_format="selector_pair",selector_policy=policy,
                                  context_view=f"selector_{policy}",input_limit=input_limit,
                                  max_new_tokens=1024,max_sequence_length=input_limit+1024)
    config["evaluation"].update(require_all_targets=True,expected_population_size=2000,max_samples=0)
    config["output"]["directory"] = str(output_root / "selector" / backbone / f"train_{train_view}" / f"input_{input_limit}" / policy)
    config["intervention"] = {"kind":"proof_blind_selector","train_view":train_view,
                              "selector_policy":policy,"population":2000,
                              "protocol_defaults":"controlled_fixed_checkpoint"}
    return config


def replacement_config(base: dict[str, Any], *, backbone: str, train_view: str,
                       condition: str, extension: dict[str, Any],
                       output_root: Path) -> dict[str, Any]:
    if backbone not in ("llada", "llama31") or train_view not in REPLACEMENT_TRAIN_VIEWS:
        raise ValueError("replacement uses llada/llama31 CMath, CEnv, or CFull fixed checkpoints")
    if condition not in REPLACEMENT_CONDITIONS:
        raise ValueError("unknown replacement input condition")
    config = copy.deepcopy(base)
    config.pop("_config_path", None)
    config["model"]["checkpoint"] = checkpoint_for(base, train_view)
    config["data"].update(tokenized_input_file=extension["pairs_file"],
                          replacement_raw_pairs_file=extension["raw_pairs_file"],
                          allowed_ids_file=extension["allowed_ids_file"])
    generation = config["generation"]
    generation.update(input_format="replacement_pair", replacement_condition=condition,
                      context_view=f"replacement_{condition}", input_limit=1024,
                      max_new_tokens=1024, max_sequence_length=2048)
    generation.setdefault("sampled", {}).update(num_candidates=32, temperature=0.6)
    generation.setdefault("greedy", {}).update(num_candidates=1, temperature=0.0)
    config["evaluation"].update(require_all_targets=True,
                                expected_population_size=extension["population"], max_samples=0)
    config["output"]["directory"] = str(output_root / "replacement" / backbone / f"train_{train_view}" / condition)
    config["intervention"] = {
        "kind": "matched_replacement_fixed_checkpoint", "backbone": backbone,
        "train_view": train_view, "input_condition": condition,
        **{key: extension[key] for key in ("extension_root", "frozen_manifest", "frozen_manifest_sha256",
                                           "population", "requested_population", "requested_size_reached")},
        "same_target_and_original_EC_across_conditions": True,
        "new_checkpoint_training_for_replacement": False,
        "offline_dependency_exclusion_is_reference_assisted": True,
        "target_reference_proof_in_generator_input": False,
        "synthetic_context_newly_Lean_elaborated": False,
        "verification_uses_unchanged_original_source_environment": True,
        "protocol_defaults": "controlled_fixed_checkpoint",
    }
    return config


def reconstruction_config(base: dict[str, Any], *, backbone: str, variant: str,
                          view: str, extension: dict[str, Any], output_root: Path) -> dict[str, Any]:
    if backbone not in ("llada", "llama31") or variant not in RECONSTRUCTION_VARIANTS or view not in VIEWS:
        raise ValueError("unknown D.4 reconstruction backbone/variant/view")
    config = copy.deepcopy(base)
    config.pop("_config_path", None)
    config["model"]["checkpoint"] = checkpoint_for(base, "CFull")
    config["data"].update(tokenized_input_file=extension["pairs_file"], allowed_ids_file=extension["allowed_ids_file"])
    generation = config["generation"]
    generation.update(input_format="reconstruction_pair", reconstruction_variant=variant,
                      reconstruction_view=view, context_view=f"reconstruction_{variant}_{view}",
                      input_limit=1024, max_new_tokens=1024, max_sequence_length=2048)
    generation.setdefault("sampled", {}).update(num_candidates=32, temperature=0.6)
    generation.setdefault("greedy", {}).update(num_candidates=1, temperature=0.0)
    config["evaluation"].update(require_all_targets=True, expected_population_size=2000, max_samples=0)
    config["output"]["directory"] = str(output_root / "reconstruction" / backbone / variant / view)
    config["intervention"] = {
        "kind": "D4_proof_blind_reconstruction", "backbone": backbone, "train_view": "CFull",
        "variant": variant, "view": view,
        **{key: extension[key] for key in ("extension_root", "manifest_sha256", "population")},
        "same_existing_selector_2000_IDs": True, "target_reference_proof_in_generator_input": False,
        "alternative_selection_uses_proof_or_target_type_dependencies": False,
        "original_selection_was_reference_assisted": True,
        "MC_and_EC_may_both_change": True, "is_D5_shared_pool_selector": False,
        "invalid_views_retained_as_failed_candidates": True,
        "new_checkpoint_training": False,
        "protocol_defaults": "proof_blind_context_reconstruction",
    }
    return config


def ensure_fresh_outputs(configs: list[dict[str, Any]]) -> None:
    paths=[Path(str(c["output"]["directory"])) for c in configs]
    if len(set(map(str,paths))) != len(paths):
        raise ValueError("planned runs collide in their output directory")
    existing=[str(p) for p in paths if p.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite any existing result directory: "+", ".join(existing))


def validate_population(config: dict[str, Any]) -> None:
    path=Path(str(config["data"]["allowed_ids_file"]))
    rows=list(read_jsonl(path))
    ids=[str(r["sample_id"]) for r in rows]
    expected=int(config["evaluation"]["expected_population_size"])
    if len(ids)!=expected or len(set(ids))!=expected:
        raise ValueError(f"declared population must contain exactly{expected}unique IDs")


def reuse_candidate_predictions(source: dict[str, Any], target: dict[str, Any], *, config_file: Path) -> None:
    """Reuse the exactly identical D.4 C0 run, never invent new candidates/results."""
    if source["intervention"]["kind"] != "D4_proof_blind_reconstruction" or target["intervention"]["kind"] != "D4_proof_blind_reconstruction":
        raise ValueError("automatic prediction reuse is restricted to identical D.4 C0 inputs")
    if source["generation"]["reconstruction_view"] != "C0" or target["generation"]["reconstruction_view"] != "C0":
        raise ValueError("prediction reuse requires the exactly shared C0 input")
    for key in ("model", "architecture", "data", "source", "verification", "evaluation", "seed"):
        if source.get(key) != target.get(key):
            raise ValueError("prediction reuse settings differ: " + key)
    left, right = copy.deepcopy(source["generation"]), copy.deepcopy(target["generation"])
    for value in (left, right):
        value.pop("reconstruction_variant", None)
        value.pop("context_view", None)
    if left != right:
        raise ValueError("prediction reuse decoding settings differ")
    source_dir, target_dir = Path(source["output"]["directory"]), Path(target["output"]["directory"])
    targets = list(read_jsonl(source_dir / "evaluation_targets.jsonl"))
    ids = [str(row["sample_id"]) for row in targets]
    allowed = [str(row["sample_id"]) for row in read_jsonl(target["data"]["allowed_ids_file"])]
    if len(ids) != len(set(ids)) or set(ids) != set(allowed):
        raise ValueError("source reused predictions do not cover the complete fixed population")
    prediction_file = source_dir / "predictions.jsonl"
    predictions = list(read_jsonl(prediction_file))
    keys = [(str(r["sample_id"]), str(r["mode"]), int(r["candidate_id"])) for r in predictions]
    expected = {(sid, "sampled", i) for sid in ids for i in range(32)} | {(sid, "greedy_temperature_0", 0) for sid in ids}
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("source reused predictions lack the exact 32+greedy candidate coverage")
    for row in predictions:
        row["context_view"] = target["generation"]["context_view"]
    write_jsonl(target_dir / "predictions.jsonl", predictions)
    write_jsonl(target_dir / "evaluation_targets.jsonl", targets)
    original_manifest = json.loads((source_dir / "generation_manifest.json").read_text(encoding="utf-8"))
    write_json(target_dir / "generation_manifest.json", {
        "config_file": str(config_file), "dataset_file": target["data"]["test_file"],
        "tokenized_input_file": target["data"]["tokenized_input_file"],
        "generation_target_filter_uses_reference_proof": False,
        "selected_samples": len(ids), "sampled_candidates_per_sample": 32,
        "deterministic_candidates_per_sample": 1,
        "model_generation_performed": False, "predictions_reused": True,
        "prediction_reuse_source": str(prediction_file), "prediction_reuse_source_sha256": digest(prediction_file),
        "source_generation_manifest": str(source_dir / "generation_manifest.json"),
        "model": original_manifest["model"], "generation": target["generation"],
        "intervention": target["intervention"], "predictions_file": str(target_dir / "predictions.jsonl"),
    })


def run_configs(configs: list[dict[str, Any]], *, num_gpus: int) -> None:
    import yaml
    if num_gpus<1:
        raise ValueError("num_gpus must be positive")
    ensure_fresh_outputs(configs)
    # Check every population before the first GPU process is dispatched.
    for config in configs:
        validate_population(config)
    checked_replacements = set()
    checked_reconstructions = set()
    for config in configs:
        if config.get("intervention", {}).get("kind") == "matched_replacement_fixed_checkpoint":
            key = (config["intervention"]["backbone"], config["intervention"]["frozen_manifest_sha256"])
            if key not in checked_replacements:
                validate_replacement_artifacts(config)
                checked_replacements.add(key)
        elif config.get("intervention", {}).get("kind") == "D4_proof_blind_reconstruction":
            key = (config["intervention"]["backbone"], config["intervention"]["manifest_sha256"])
            if key not in checked_reconstructions:
                validate_reconstruction_artifacts(config)
                checked_reconstructions.add(key)
    project=Path(__file__).resolve().parents[1]
    by_output = {str(config["output"]["directory"]): config for config in configs}
    completed_outputs = set()
    for config in configs:
        output=Path(str(config["output"]["directory"]))
        output.mkdir(parents=True,exist_ok=False)
        config_file=output/"intervention_config.yaml"
        config_file.write_text(yaml.safe_dump(config,sort_keys=False,allow_unicode=True),encoding="utf-8")
        commands=[
            [sys.executable,"-m","evaluation.source_index","--config",str(config_file)],
            [sys.executable,"-m","evaluation.gold_preflight","--config",str(config_file)]]
        for command in commands:
            subprocess.run(command,cwd=project,check=True)
        reuse_source = config.get("prediction_reuse_from")
        if reuse_source is not None:
            if reuse_source not in completed_outputs:
                raise ValueError("reuse source must be a previously completed run in this same plan")
            validate_reconstruction_artifacts(config)
            reuse_candidate_predictions(by_output[reuse_source], config, config_file=config_file)
        else:
            subprocess.run([sys.executable,"-m","torch.distributed.run","--standalone","--nproc_per_node",str(num_gpus),
                            "--module","evaluation.generate","--config",str(config_file)],cwd=project,check=True)
        subprocess.run([sys.executable,"-m","evaluation.verify","--config",str(config_file)],cwd=project,check=True)
        completed_outputs.add(str(output))


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="experiment",required=True)
    for name in ("crossing","selector","replacement","reconstruction"):
        child=sub.add_parser(name)
        child.add_argument("--backbone",choices=("llada","llama31"),required=True)
        child.add_argument("--base-config",type=Path)
        train_choices = REPLACEMENT_TRAIN_VIEWS if name == "replacement" else ("CFull",) if name == "reconstruction" else VIEWS
        child.add_argument("--train-view",choices=train_choices,default="CFull")
        child.add_argument("--output-root",type=Path,required=True)
        child.add_argument("--num-gpus",type=int,default=16)
        mode = child.add_mutually_exclusive_group()
        mode.add_argument("--execute",action="store_true",help="Actually execute; default only prints a plan")
        mode.add_argument("--plan",action="store_true",help="Print a read-only validated plan (default)")
    cross=sub.choices["crossing"]
    cross.add_argument("--inference-view",choices=VIEWS,default="CFull")
    cross.add_argument("--grid",action="store_true",help="Prepare all4train x4inference cells for this backbone")
    selector=sub.choices["selector"]
    selector.add_argument("--pairs-file",type=Path,required=True)
    selector.add_argument("--policy",choices=(*POLICIES,"both"),default="both")
    selector.add_argument("--input-limit",type=int,choices=(256,512,1024),default=1024)
    replacement = sub.choices["replacement"]
    replacement.add_argument("--extension-root", type=Path, required=True)
    replacement.add_argument("--condition", choices=(*REPLACEMENT_CONDITIONS, "all"), default="all")
    replacement.add_argument("--grid", action="store_true",
                             help="All 3 fixed training checkpoints x 3 replacement conditions")
    reconstruction = sub.choices["reconstruction"]
    reconstruction.add_argument("--extension-root", type=Path, required=True)
    reconstruction.add_argument("--variant", choices=(*RECONSTRUCTION_VARIANTS, "both"), default="both")
    reconstruction.add_argument("--view", choices=(*VIEWS, "all"), default="all")
    args=parser.parse_args()
    project=Path(__file__).resolve().parents[1]
    base=load_config(args.base_config or project/f"configs/evaluation/{args.backbone}_CFull.yaml")
    root=args.output_root.expanduser().resolve()
    if args.experiment=="crossing":
        cells=[(train,infer) for train in VIEWS for infer in VIEWS] if args.grid else [(args.train_view,args.inference_view)]
        configs=[crossing_config(base,backbone=args.backbone,train_view=train,inference_view=infer,output_root=root)
                 for train,infer in cells]
    elif args.experiment=="selector":
        policies=POLICIES if args.policy=="both" else (args.policy,)
        configs=[selector_config(base,backbone=args.backbone,train_view=args.train_view,policy=policy,
                                 pairs_file=args.pairs_file.resolve(),input_limit=args.input_limit,output_root=root)
                 for policy in policies]
    elif args.experiment == "replacement":
        extension = extension_info(args.extension_root.resolve(), args.backbone)
        train_views = REPLACEMENT_TRAIN_VIEWS if args.grid else (args.train_view,)
        conditions = REPLACEMENT_CONDITIONS if args.grid or args.condition == "all" else (args.condition,)
        configs = [replacement_config(base, backbone=args.backbone, train_view=train_view,
                                      condition=condition, extension=extension, output_root=root)
                   for train_view in train_views for condition in conditions]
        # The default plan is read-only, but still rejects missing or modified IDs
        # and incomparable inputs before showing an executable experiment.
        validate_replacement_artifacts(configs[0])
    else:
        extension = reconstruction_extension_info(args.extension_root.resolve(), args.backbone)
        variants = RECONSTRUCTION_VARIANTS if args.variant == "both" else (args.variant,)
        views = VIEWS if args.view == "all" else (args.view,)
        configs = [reconstruction_config(base, backbone=args.backbone, variant=variant, view=view,
                                         extension=extension, output_root=root)
                   for variant in variants for view in views]
        if args.variant == "both" and "C0" in views:
            source = next(c for c in configs if c["intervention"]["variant"] == "original" and c["intervention"]["view"] == "C0")
            target = next(c for c in configs if c["intervention"]["variant"] == "reconstructed" and c["intervention"]["view"] == "C0")
            target["prediction_reuse_from"] = source["output"]["directory"]
            target["intervention"]["identical_C0_predictions_reused"] = True
        validate_reconstruction_artifacts(configs[0])
    ensure_fresh_outputs(configs)
    if args.execute:
        run_configs(configs,num_gpus=args.num_gpus)
    else:
        print(json.dumps({"execution_performed":False,"planned_runs":configs},ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
