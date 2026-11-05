"""Pinned released-prover evaluation with native-tokenizer plain Lean inputs.

Default invocation prints a plan without loading a tokenizer/model or writing
data. --prepare-only builds prompt-only input rows on CPU. --execute then
runs the existing generation/Lean pipeline. No benchmark training is performed.
The controlled protocol uses plain Lean completion inputs, independent of
model-card chat, few-shot, or chain-of-thought templates.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import re
import subprocess
import sys
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from .current_inputs import VIEWS, encode, ordered, render
from .io_utils import load_config, read_jsonl, write_json, write_jsonl

MODEL_SPECS = {
    "deepseek_v15_sft": {
        "repository":"deepseek-ai/DeepSeek-Prover-V1.5-SFT",
        "revision":"e9a6e6fbb67620d4e9c4944bc51ff7c435af12da",
        "gated":False,
    },
    "deepseek_v15_rl": {
        "repository":"deepseek-ai/DeepSeek-Prover-V1.5-RL",
        "revision":"40a760138c68038d9869a436077c16b654b1cc72",
        "gated":False,
    },
    "deepseek_v2_7b": {
        "repository":"deepseek-ai/DeepSeek-Prover-V2-7B",
        "revision":"a8d9e14432b2e8dd9df2a4d4e70f1ba9bc8d9b7b",
        "gated":False,
        "non_cot_protocol":"plain_completion_with_the_released_checkpoint",
    },
    "goedel_sft": {
        "repository":"Goedel-LM/Goedel-Prover-SFT",
        "revision":"5b03a13d14265d438048ffd4620f28f637f85313",
        "gated":False,
    },
    "theoremllama": {
        "repository":"RickyDeSkywalker/TheoremLlama",
        "revision":"37ce41a80102504d2378ed2849cb22c7bd5e98b9",
        "gated":True,
        "official_code":"https://github.com/RickySkywalker/TheoremLlama",
        "access_note":"official HF repository is gated; accept its terms and provide authorized local/cache access; never substitute another tokenizer",
    },
}
PRIMARY_SOURCE_CHECK = "2026-10-13; official Hugging Face model cards/tree/commit pages; no weights fetched"
PROTOCOL = "leancontext-released-native-tokenizer-plain-completion-1024-v1"


def specification(alias: str) -> dict[str, Any]:
    if alias not in MODEL_SPECS:
        raise ValueError("unsupported released model; no approximate aliases/fallbacks")
    result=copy.deepcopy(MODEL_SPECS[alias])
    if re.fullmatch(r"[0-9a-f]{40}",result["revision"]) is None:
        raise ValueError("released model needs an immutable full commit revision")
    base="https://huggingface.co/"+result["repository"]
    result.update(model_alias=alias,official_model_card=base,official_revision_url=base+"/commit/"+result["revision"],
                  official_files=base+"/tree/"+result["revision"],primary_source_check=PRIMARY_SOURCE_CHECK)
    return result


def temperature_label(value: float) -> str:
    if value not in (.2,.6,1.0):
        raise ValueError("sampled temperature must be0.2,0.6or1.0; greedy0is a separate single attempt")
    return str(float(value)).replace(".","p")


def make_config(alias: str, *, dataset_root: Path, output_root: Path, model_cache: Path,
                 mathlib_root: Path, view: str, temperature: float=.6) -> dict[str, Any]:
    spec=specification(alias)
    if view not in VIEWS:
        raise ValueError("unknown context view")
    root=output_root/"released"/alias/f"temperature_{temperature_label(temperature)}"
    return {
        "seed":2026,"architecture":"ar",
        "model":{"name_or_path":spec["repository"],"revision":spec["revision"],
                 "tokenizer_name_or_path":str(root/"inputs/tokenizer"),"tokenizer_revision":spec["revision"],
                 "tokenizer_local_files_only":True,"checkpoint":None,"cache_dir":str(model_cache),
                 "dtype":"bfloat16","trust_remote_code":False,"use_fast_tokenizer":True,
                 "minimum_transformers_version":"4.46.3"},
        "data":{"test_file":str(dataset_root/"benchmark/test.jsonl"),
                "tokenized_input_file":str(root/f"inputs/views/{view}/test.jsonl"),
                "allowed_ids_file":str(dataset_root/"evaluation_populations/test_both_nonempty.jsonl")},
        "source":{"public_spans":True,"command_spans_root":str(dataset_root/"command_mapping/source_commands"),
                  "mathlib_root":str(mathlib_root)},
        "generation":{"context_view":view,"input_format":"training_view","tokenizer_mode":"native",
                      "input_limit":1024,"max_new_tokens":1024,"max_sequence_length":2048,"candidate_batch_size":1,
                      "sampled":{"num_candidates":32,"temperature":temperature,"top_p":.95,"top_k":0},
                      "greedy":{"num_candidates":1,"temperature":0},
                      "forbid_reasoning_markers":["<think>","</think>"] if alias=="deepseek_v2_7b" else []},
        "verification":{"lean_command":"lake env lean","preflight_import":"Mathlib","gold_preflight_samples":8,
                        "jobs":8,"timeout_seconds":60,"heartbeat_policy":"source_inherited","resume":True,
                        "keep_lean_files":False,"keep_failed_lean_files":True},
        "evaluation":{"max_samples":0,"require_all_targets":True,"expected_population_size":11064},
        "output":{"directory":str(root/view)},
        "released_evaluation":{**spec,"protocol":PROTOCOL,"benchmark_training":False,
                              "serializer":"source-ordered plain Lean commands + unchanged statement + newline; native tokenizer special-token postprocessor only",
                              "chat_template_or_extra_instruction_applied":False,
                              "model_card_recommended_prompt_reproduction":False,
                              "protocol_defaults":"controlled_plain_lean_completion",
                              "reasoning_token_accounting":"ALL generated tokens share max_new_tokens1024; no separate reasoning allowance or renewed proof budget",
                              "reasoning_marker_policy":"invalid candidate before proof cleanup when forbidden markers occur"},
    }


def plain_four_views(row: dict[str, Any], tokenizer: Any, *, input_limit: int=1024) -> dict[str, dict[str, Any]]:
    """Proof-blind exact packing; one CFull prefix, four projections, no refill."""
    if input_limit!=1024:
        raise ValueError("released primary evaluation is input1024only")
    statement=str(row["statement"])
    full=ordered(list(row["full_context"]))
    math={str(x["command_uid"]) for x in row["mathematical_context"]}
    env={str(x["command_uid"]) for x in row["environment_context"]}
    identities=[str(x["command_uid"]) for x in full]
    if len(set(identities))!=len(identities) or math&env or math|env!=set(identities):
        raise ValueError("context roles must form a unique disjoint Full union")
    bare=render(statement,[])
    bare_ids=encode(tokenizer,bare,"native")
    if len(bare_ids)>input_limit:
        return {view:{"sample_id":str(row["sample_id"]),"split":str(row["split"]),
                      "proof_family":str(row["proof_family"]),"view":view,"statement":statement,
                      "prompt":bare,"input_ids":bare_ids,"prompt_length":len(bare_ids),
                      "input_length":len(bare_ids),"input_limit":input_limit,"output_limit":1024,
                      "context_ids":[],"valid_input":False,
                      "invalid_input_reason":"complete_statement_exceeds_native_input_limit","protocol":PROTOCOL}
                for view in VIEWS}
    # Cache encodings of identical context projections across prefix iterations.
    cache:dict[tuple[str,...],tuple[str,list[int]]]={(): (bare,bare_ids)}
    prefix=list(full)
    while True:
        projected={"C0":[],"CMath":[x for x in prefix if x["command_uid"] in math],
                   "CEnv":[x for x in prefix if x["command_uid"] in env],"CFull":prefix}
        encoded={}
        for view,items in projected.items():
            key=tuple(str(x["command_uid"]) for x in items)
            if key not in cache:
                prompt=render(statement,items)
                cache[key]=(prompt,encode(tokenizer,prompt,"native"))
            encoded[view]=cache[key]
        if all(len(ids)<=input_limit for _,ids in encoded.values()):
            break
        prefix.pop()
    result={}
    for view,(prompt,ids) in encoded.items():
        if not ids or (tokenizer.eos_token_id is not None and ids[-1]==int(tokenizer.eos_token_id)):
            raise ValueError("native tokenizer added terminal input EOS or empty input; unsupported completion serializer, no fallback")
        result[view]={"sample_id":str(row["sample_id"]),"split":str(row["split"]),
                      "proof_family":str(row["proof_family"]),"view":view,"statement":statement,
                      "prompt":prompt,"input_ids":ids,"prompt_length":len(ids),"input_length":len(ids),
                      "input_limit":input_limit,"output_limit":1024,"context_ids":[x["command_id"] for x in projected[view]],
                      "context_uids":[x["command_uid"] for x in projected[view]],
                      "shared_full_prefix_command_uids":[x["command_uid"] for x in prefix],
                      "removed_command_uids":identities[len(prefix):],"valid_input":True,"protocol":PROTOCOL}
    return result


def tokenizer_identity(tokenizer: Any) -> dict[str, Any]:
    backend=getattr(tokenizer,"backend_tokenizer",None)
    return {"class":type(tokenizer).__name__,"bos_token_id":tokenizer.bos_token_id,
            "eos_token_id":tokenizer.eos_token_id,"pad_token_id":tokenizer.pad_token_id,
            "native_empty_input_special_ids":encode(tokenizer,"","native"),
            "backend_sha256":hashlib.sha256(backend.to_str().encode()).hexdigest() if backend else None,
            "transformers_version":importlib.metadata.version("transformers"),
            "tokenizers_version":importlib.metadata.version("tokenizers")}


def prepack_inputs(alias: str, config: dict[str, Any], *, allow_tokenizer_download: bool=False) -> dict[str, Any]:
    from transformers import AutoTokenizer
    spec=specification(alias)
    inputs=Path(config["data"]["tokenized_input_file"]).parents[2]
    if inputs.exists():
        raise FileExistsError(f"refusing existing released prompt directory: {inputs}")
    population_rows=list(read_jsonl(config["data"]["allowed_ids_file"]))
    target_ids=[str(x["sample_id"]) for x in population_rows]
    if len(target_ids)!=11064 or len(set(target_ids))!=11064:
        raise ValueError("released evaluation requires the complete original11064-ID population")
    allowed=set(target_ids)
    try:
        tokenizer=AutoTokenizer.from_pretrained(spec["repository"],revision=spec["revision"],
            cache_dir=config["model"]["cache_dir"],local_files_only=not allow_tokenizer_download,
            use_fast=True,trust_remote_code=False)
    except Exception as exc:
        raise RuntimeError("Cannot load the official pinned native tokenizer. No fallback allowed. "
                           "For TheoremLlama, accept its gated terms and authenticate/cache the snapshot. "
                           "No model weights were requested by this CPU prepack step.") from exc
    if tokenizer.eos_token_id is None:
        raise ValueError("native tokenizer must define EOS")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token=tokenizer.eos_token
    identity=tokenizer_identity(tokenizer)
    inputs.mkdir(parents=True,exist_ok=False)
    tokenizer.save_pretrained(inputs/"tokenizer")
    seen=set()
    counts=Counter()
    with ExitStack() as stack:
        handles={}
        for view in VIEWS:
            path=inputs/f"views/{view}/test.jsonl"
            path.parent.mkdir(parents=True,exist_ok=True)
            handles[view]=stack.enter_context(path.open("w",encoding="utf-8"))
        for source in read_jsonl(config["data"]["test_file"]):
            sid=str(source["sample_id"])
            if sid not in allowed:
                continue
            if sid in seen:
                raise ValueError("duplicate target in raw test benchmark")
            seen.add(sid)
            # Gold target/proof is deliberately not accessed or copied.
            problem={key:source[key] for key in ("sample_id","split","proof_family","statement",
                      "mathematical_context","environment_context","full_context")}
            packed=plain_four_views(problem,tokenizer)
            for view in VIEWS:
                row=packed[view]
                # Independent token-level equality check before saving any row.
                if encode(tokenizer,row["prompt"],"native")!=row["input_ids"]:
                    raise ValueError("native prompt/token IDs differ during prepack validation")
                handles[view].write(json.dumps(row,ensure_ascii=False)+"\n")
                counts[f"{view}_valid" if row["valid_input"] else f"{view}_invalid"]+=1
            if len(seen)%1000==0:
                print(json.dumps({"released_prepack_model":alias,"targets":len(seen),"total":11064}),flush=True)
    if seen!=allowed:
        raise ValueError("raw benchmark misses some fixed released-model evaluation targets")
    manifest={"protocol":PROTOCOL,"model":spec,"native_tokenizer":identity,"population":11064,
              "counts":dict(counts),"shared_whole_command_prefix_and_projection":True,
              "extra_instruction_or_chat_template":False,"reference_proof_used":False,
              "model_weights_requested_by_cpu_packer":False,"model_generation_or_Lean_verification_performed":False,
              "inputs_directory":str(inputs),"input_limit":1024,"output_limit":1024,
              "population_file":config["data"]["allowed_ids_file"]}
    write_json(inputs/"prepack_manifest.json",manifest)
    return manifest


def execute_configs(configs: list[dict[str, Any]], num_gpus: int) -> None:
    import yaml
    if num_gpus<1:
        raise ValueError("num_gpus must be positive")
    for config in configs:
        if Path(config["output"]["directory"]).exists():
            raise FileExistsError("refusing existing released-model run output")
    project=Path(__file__).resolve().parents[1]
    for config in configs:
        output=Path(config["output"]["directory"])
        output.mkdir(parents=True,exist_ok=False)
        path=output/"released_config.yaml"
        path.write_text(yaml.safe_dump(config,allow_unicode=True,sort_keys=False),encoding="utf-8")
        commands=[
            [sys.executable,"-m","evaluation.source_index","--config",str(path)],
            [sys.executable,"-m","evaluation.gold_preflight","--config",str(path)],
            [sys.executable,"-m","torch.distributed.run","--standalone","--nproc_per_node",str(num_gpus),
             "--module","evaluation.generate","--config",str(path)],
            [sys.executable,"-m","evaluation.verify","--config",str(path)]]
        for command in commands:
            subprocess.run(command,cwd=project,check=True)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",choices=tuple(MODEL_SPECS),required=True)
    parser.add_argument("--dataset-root",type=Path,required=True)
    parser.add_argument("--output-root",type=Path,required=True)
    parser.add_argument("--model-cache",type=Path,required=True)
    parser.add_argument("--mathlib-root",type=Path,required=True)
    parser.add_argument("--view",choices=(*VIEWS,"all"),default="all")
    parser.add_argument("--temperature",type=float,choices=(.2,.6,1.0),default=.6)
    parser.add_argument("--num-gpus",type=int,default=16)
    parser.add_argument("--prepare-only",action="store_true")
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--allow-tokenizer-download",action="store_true",help="Permit only tokenizer assets in CPU prepack; no weights")
    args=parser.parse_args()
    if args.prepare_only and args.execute:
        parser.error("choose --prepare-only or --execute")
    views=VIEWS if args.view=="all" else (args.view,)
    configs=[make_config(args.model,dataset_root=args.dataset_root.resolve(),output_root=args.output_root.resolve(),
                        model_cache=args.model_cache.resolve(),mathlib_root=args.mathlib_root.resolve(),
                        view=view,temperature=args.temperature) for view in views]
    for config in configs:
        if Path(config["output"]["directory"]).exists():
            raise FileExistsError("refusing existing released-model run output")
    if not args.prepare_only and not args.execute:
        print(json.dumps({"execution_performed":False,"tokenizer_loaded":False,"planned_configs":configs},ensure_ascii=False,indent=2))
        return
    manifest=prepack_inputs(args.model,configs[0],allow_tokenizer_download=args.allow_tokenizer_download)
    if args.execute:
        execute_configs(configs,args.num_gpus)
    else:
        import yaml
        input_root=Path(manifest["inputs_directory"])
        files=[]
        for config in configs:
            view=config["generation"]["context_view"]
            path=input_root/f"configs/{view}.yaml"
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(yaml.safe_dump(config,allow_unicode=True,sort_keys=False),encoding="utf-8")
            files.append(str(path))
        print(json.dumps({"CPU_prepack_complete":True,"generation_performed":False,
                          "runtime_configs":files,"manifest":manifest},ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
