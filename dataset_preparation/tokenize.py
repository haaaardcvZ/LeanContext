"""Exact, auditable tokenizer processing of prepared LeanContext datasets.

The module's packing helpers use only the standard library. Transformers is
imported lazily by the CLI; tokenizer loading is offline and revision-pinned.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
from typing import Any, Iterable


VIEWS = ("C0", "CMath", "CEnv", "CFull")
SPLITS = ("train", "validation", "test")
TRAINING_PROTOCOL = "leancontext-reconstructed-joint-four-view-v1"
SELECTOR_TEMPLATE = "leancontext-plain-source-order-scope-cohort-v2"


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{lineno}: expected an object")
            yield row


def write_row(handle: Any, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def command_id(command: dict[str, Any]) -> str:
    value = command.get("command_id", command.get("command_uid"))
    if value is None:
        raise ValueError("context command has no command_id or command_uid")
    return str(value)


def command_key(command: dict[str, Any]) -> str:
    return str(command.get("command_uid") or command_id(command))


def source_order(commands: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(item: dict[str, Any]) -> tuple[int, str]:
        order = item.get("source_order", item.get("position"))
        if order is None:
            raise ValueError(f"command {command_id(item)} has no source order")
        return int(order), command_key(item)

    return sorted(commands, key=key)


def unique_commands(commands: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for command in commands:
        key = command_key(command)
        if key not in seen:
            if not isinstance(command.get("text"), str) or not command["text"].strip():
                raise ValueError(f"command {command_id(command)} has no complete source text")
            seen.add(key)
            result.append(command)
    return result


def encode_prompt(tokenizer: Any, prompt: str, backbone: str) -> list[int]:
    if backbone not in {"dllm", "llama31"}:
        raise ValueError("backbone must be dllm or llama31")
    ids = list(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    if backbone == "llama31":
        bos = tokenizer.bos_token_id
        if bos is None:
            raise ValueError("Llama prompt encoding requires the actual tokenizer BOS")
        if not ids or ids[0] != int(bos):
            ids.insert(0, int(bos))
    return ids


def encode_answer(tokenizer: Any, proof: str) -> list[int]:
    if not isinstance(proof, str) or not proof.strip():
        raise ValueError("reference proof must be nonempty")
    if proof.startswith(("\n", "\r")):
        raise ValueError("reference proof has a leading newline; prompt owns the boundary newline")
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("actual tokenizer must define EOS")
    ids = list(tokenizer(proof, add_special_tokens=False)["input_ids"])
    if not ids:
        raise ValueError("reference proof tokenization is empty")
    if int(eos) in ids:
        raise ValueError("reference proof already contains an EOS token")
    return ids + [int(eos)]


def render_training_prompt(statement: str, commands: Iterable[dict[str, Any]]) -> str:
    if not isinstance(statement, str) or not statement.strip():
        raise ValueError("statement must be nonempty")
    parts = [item["text"] for item in source_order(commands)] + [statement]
    return "\n\n".join(parts).rstrip("\r\n") + "\n"


def prepare_training_views(
    tokenizer: Any,
    statement: str,
    reference_proof: str,
    contexts: dict[str, list[dict[str, Any]]],
    *,
    backbone: str,
    max_training_length: int = 4096,
) -> dict[str, Any]:
    """Trim one shared Full context, derive subsets without refilling, jointly reject.

    Prompt and proof are separately encoded, as in the existing training code.
    The proof's original text is preserved exactly. No padding/packing occurs.
    """
    if max_training_length <= 0:
        raise ValueError("max training length must be positive")
    if set(contexts) != set(VIEWS) or contexts["C0"]:
        raise ValueError("contexts must contain four views and C0 must be empty")
    full = source_order(unique_commands(contexts["CFull"]))
    full_keys = {command_key(item) for item in full}
    view_keys = {view: {command_key(item) for item in contexts[view]} for view in VIEWS}
    if view_keys["CMath"] & view_keys["CEnv"]:
        raise ValueError("Math and Env command sets overlap")
    if view_keys["CMath"] | view_keys["CEnv"] != full_keys:
        raise ValueError("Full must be the exact Math/Env union")
    answer_ids = encode_answer(tokenizer, reference_proof)
    bare_prompt = render_training_prompt(statement, [])
    bare_ids = encode_prompt(tokenizer, bare_prompt, backbone)
    bare_length = len(bare_ids) + len(answer_ids)
    if bare_length > max_training_length:
        return {
            "retained": False,
            "reason": "bare_statement_and_complete_proof_exceed_training_limit",
            "bare_sequence_length": bare_length,
            "prompt_length": len(bare_ids),
            "answer_length": len(answer_ids),
            "max_training_length": max_training_length,
        }
    selected = list(full)
    removed: list[dict[str, Any]] = []
    initial_lengths: dict[str, int] | None = None
    while True:
        rendered: dict[str, dict[str, Any]] = {}
        for view in VIEWS:
            visible = [item for item in selected if command_key(item) in view_keys[view]]
            prompt = render_training_prompt(statement, visible)
            prompt_ids = encode_prompt(tokenizer, prompt, backbone)
            rendered[view] = {
                "prompt": prompt,
                "statement": statement,
                "target": reference_proof,
                "reference_proof": reference_proof,
                "input_ids": prompt_ids + answer_ids,
                "prompt_length": len(prompt_ids),
                "answer_length": len(answer_ids),
                "target_length": len(answer_ids),
                "sequence_length": len(prompt_ids) + len(answer_ids),
                "answer_span": {"start": len(prompt_ids), "end_exclusive": len(prompt_ids) + len(answer_ids)},
                "answer_eos_count": 1,
                "answer_final_token_id": int(tokenizer.eos_token_id),
                "context_items": visible,
                "context_ids": [command_id(item) for item in visible],
                "context_uids": [command_key(item) for item in visible],
                "visible_command_ids": [command_id(item) for item in visible],
                "visible_command_uids": [command_key(item) for item in visible],
            }
        if initial_lengths is None:
            initial_lengths = {view: row["sequence_length"] for view, row in rendered.items()}
        if all(row["sequence_length"] <= max_training_length for row in rendered.values()):
            break
        if not selected:
            raise AssertionError("bare fit checked but empty shared Full context does not fit")
        removed.append(selected.pop())
    for view, row in rendered.items():
        row["removed_command_ids"] = [command_id(item) for item in source_order(removed)]
        row["removed_command_uids"] = [command_key(item) for item in source_order(removed)]
        row["view_removed_command_ids"] = [command_id(item) for item in source_order(removed) if command_key(item) in view_keys[view]]
        row["untrimmed_sequence_length"] = initial_lengths[view]
        row["max_training_length"] = max_training_length
    return {"retained": True, "views": rendered, "removed_command_ids": [command_id(item) for item in source_order(removed)]}


def render_selector_prompt(statement: str, ec: list[dict[str, Any]], mc: list[dict[str, Any]]) -> str:
    if not isinstance(statement, str) or not statement.strip():
        raise ValueError("statement must be nonempty")
    ec, mc = source_order(ec), source_order(mc)
    if {command_key(item) for item in ec} & {command_key(item) for item in mc}:
        raise ValueError("EC and MC overlap")
    # IDs and category labels belong to metadata, not to the generator input.
    # Use the exact same plain Lean serializer as the four training views.
    return render_training_prompt(statement, [*ec, *mc])


def dependency_coverage(dependencies: dict[str, Any], visible_ids: list[str], *, valid_input: bool) -> dict[str, Any]:
    status = str(dependencies.get("status", "uncertain"))
    dependency_ids = {str(value) for value in dependencies.get("command_ids", [])}
    if not valid_input:
        return {"status": "invalid_input", "coverage": None, "dependency_count": len(dependency_ids), "visible_dependency_count": None}
    if status not in {"complete_nonempty", "complete"}:
        return {"status": status, "coverage": None, "dependency_count": len(dependency_ids), "visible_dependency_count": None}
    if not dependency_ids:
        return {"status": "no_eligible_local_dependencies", "coverage": None, "dependency_count": 0, "visible_dependency_count": 0}
    count = len(dependency_ids & set(visible_ids))
    category = "zero" if count == 0 else "full" if count == len(dependency_ids) else "partial"
    return {"status": category, "coverage": count / len(dependency_ids), "dependency_count": len(dependency_ids), "visible_dependency_count": count}


def environment_scope_cohorts(environment: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Transitive closure of enclosing openers and paired block closers.

    This is syntactic preservation, not a certificate of Lean binding semantics.
    An accepted EC effect must never lose a required enclosing scope boundary.
    """
    index = {command_id(item): item for item in environment}
    if len(index) != len(environment):
        raise ValueError("shared EC contains duplicate command IDs")
    dependencies: dict[str, set[str]] = {}
    for cid, item in index.items():
        required = item.get("required_scope_command_ids", [])
        paired = item.get("scope_pair_command_ids", [])
        if not isinstance(required, list) or not isinstance(paired, list):
            raise ValueError(f"EC {cid}: malformed scope dependency records")
        refs = {str(value) for value in [*required, *paired]}
        if not refs <= set(index):
            raise ValueError(f"EC {cid}: required scope commands missing: {sorted(refs - set(index))}")
        role = item.get("scope_role")
        if role == "block_closer" and not paired:
            raise ValueError(f"EC {cid}: unpaired block closer")
        if role == "block_opener" and item.get("target_scope_active") is False and not paired:
            raise ValueError(f"EC {cid}: closed scope has no paired closer")
        dependencies[cid] = refs
    result = {}
    for cid in index:
        closure: set[str] = set()
        pending = [cid]
        while pending:
            current = pending.pop()
            if current in closure:
                continue
            closure.add(current)
            pending.extend(dependencies[current] - closure)
        result[cid] = source_order(index[value] for value in closure)
    return result


def pack_selector_pair(
    tokenizer: Any,
    statement: str,
    shared_environment: list[dict[str, Any]],
    baseline_ranking: list[dict[str, Any]],
    dependency_prioritized_ranking: list[dict[str, Any]],
    *,
    backbone: str,
    input_limit: int = 1024,
    output_limit: int = 1024,
    target_type_dependencies: dict[str, Any] | None = None,
    environment_scope_status: str | None = None,
) -> dict[str, Any]:
    """Freeze EC before MC packing, atomically fitting complete scope cohorts.

    Reconstructed, conservative whole-command protocol. Failed cohorts are
    skipped and later priorities are tried; both policies receive identical EC.
    """
    if input_limit <= 0 or output_limit <= 0:
        raise ValueError("input and output limits must be positive")
    deps = target_type_dependencies or {"status": "uncertain", "command_ids": []}
    base_prompt = render_selector_prompt(statement, [], [])
    base_ids = encode_prompt(tokenizer, base_prompt, backbone)
    scope_error = None
    try:
        if environment_scope_status not in {None, "syntactic_valid"}:
            raise ValueError(f"shared EC scope status is {environment_scope_status}")
        environment = unique_commands(shared_environment)
        cohorts = environment_scope_cohorts(environment)
    except ValueError as exc:
        scope_error = str(exc)
    if scope_error is not None or len(base_ids) > input_limit:
        reason = "shared_environment_scope_records_missing_or_invalid" if scope_error is not None else "target_and_fixed_template_exceed_input_limit"
        invalid = {
            "valid_input": False,
            "reason": reason,
            "scope_error": scope_error,
            "prompt": None,
            "input_ids": None,
            "input_length": None,
            "required_fixed_input_length": len(base_ids),
            "selected_command_ids": [],
            "visible_command_ids": [],
            "visible_mc_command_ids": [],
            "visible_ec_command_ids": [],
            "all_visible_command_ids": [],
            "budget_skipped_command_ids": [],
            "target_coverage": dependency_coverage(deps, [], valid_input=False),
        }
        return {"shared_environment": [], "shared_environment_ids": [], "environment_budget_skipped_command_ids": [], "baseline": dict(invalid), "dependency_prioritized": dict(invalid), "input_limit": input_limit, "output_limit": output_limit, "template_version": SELECTOR_TEMPLATE, "scope_packing_protocol": "transitive-required-scopes-and-paired-closers-atomic"}
    ec: list[dict[str, Any]] = []
    skipped_cohorts: list[dict[str, Any]] = []
    for item in environment:
        cohort = cohorts[command_id(item)]
        proposal = unique_commands([*ec, *cohort])
        if len(encode_prompt(tokenizer, render_selector_prompt(statement, proposal, []), backbone)) <= input_limit:
            ec = proposal
        else:
            skipped_cohorts.append({"priority_command_id": command_id(item), "cohort_command_ids": [command_id(member) for member in cohort]})
    ec = source_order(ec)
    selected_ec_ids = {command_id(item) for item in ec}
    result: dict[str, Any] = {
        "shared_environment": ec,
        "shared_environment_ids": [command_id(item) for item in ec],
        "environment_budget_skipped_command_ids": [command_id(item) for item in environment if command_id(item) not in selected_ec_ids],
        "environment_budget_skipped_cohorts": skipped_cohorts,
        "input_limit": input_limit,
        "output_limit": output_limit,
        "template_version": SELECTOR_TEMPLATE,
        "scope_packing_protocol": "transitive-required-scopes-and-paired-closers-atomic",
    }
    for policy, ranking in (("baseline", baseline_ranking), ("dependency_prioritized", dependency_prioritized_ranking)):
        mc: list[dict[str, Any]] = []
        skipped_mc: list[str] = []
        for item in unique_commands(ranking):
            proposal = [*mc, item]
            if len(encode_prompt(tokenizer, render_selector_prompt(statement, ec, proposal), backbone)) <= input_limit:
                mc.append(item)
            else:
                skipped_mc.append(command_id(item))
        mc = source_order(mc)
        prompt = render_selector_prompt(statement, ec, mc)
        ids = encode_prompt(tokenizer, prompt, backbone)
        visible = [command_id(item) for item in mc]
        result[policy] = {
            "valid_input": True,
            "reason": None,
            "prompt": prompt,
            "input_ids": ids,
            "input_length": len(ids),
            "prompt_length": len(ids),
            "selected_command_ids": visible,
            "visible_command_ids": visible,
            "visible_mc_command_ids": visible,
            "visible_ec_command_ids": [command_id(item) for item in ec],
            "all_visible_command_ids": [command_id(item) for item in source_order([*ec, *mc])],
            "visible_command_uids": [command_key(item) for item in mc],
            "selected_mathematical_context": mc,
            "budget_skipped_command_ids": skipped_mc,
            "target_coverage": dependency_coverage(deps, visible, valid_input=True),
        }
    result["paired_input_length_difference"] = result["dependency_prioritized"]["input_length"] - result["baseline"]["input_length"]
    return result


def load_actual_tokenizer(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    local = Path(args.tokenizer).expanduser()
    is_local = local.is_dir()
    revision = args.tokenizer_revision
    if not is_local and (not revision or not re.fullmatch(r"[0-9a-fA-F]{40}", revision)):
        raise ValueError("remote tokenizer IDs require an immutable full 40-hex commit --tokenizer-revision")
    try:
        import transformers
        from transformers import AutoTokenizer
        from transformers.utils.hub import cached_file
    except ImportError as exc:
        raise RuntimeError("install the model's Transformers/tokenizers environment on the processing server") from exc
    tokenizer = AutoTokenizer.from_pretrained(
        str(local) if is_local else args.tokenizer,
        revision=revision,
        local_files_only=True,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )
    names = {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json", "tokenizer.model", "vocab.json", "vocab.txt", "merges.txt", "config.json"}
    files: dict[str, Path] = {}
    if is_local:
        for path in local.iterdir():
            if path.is_file() and (path.name in names or path.name.startswith("tokenization_") and path.suffix == ".py"):
                files[path.name] = path
    else:
        for name in sorted(names):
            value = cached_file(args.tokenizer, name, revision=revision, local_files_only=True, _raise_exceptions_for_missing_entries=False, _raise_exceptions_for_connection_errors=False)
            if value:
                files[name] = Path(value)
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in sorted(files.items())}
    backend = getattr(tokenizer, "backend_tokenizer", None)
    backend_hash = hashlib.sha256(backend.to_str().encode()).hexdigest() if backend is not None else None
    if not hashes:
        raise ValueError("cannot locate tokenizer files for reproducible file hashes")
    metadata = {
        "tokenizer_source": str(local.resolve()) if is_local else args.tokenizer,
        "requested_revision": revision,
        "resolved_revision": tokenizer.init_kwargs.get("_commit_hash") or revision or "local-file-hashes",
        "tokenizer_file_sha256": hashes,
        "tokenizer_backend_sha256": backend_hash,
        "tokenizer_class": type(tokenizer).__name__,
        "transformers_version": transformers.__version__,
        "tokenizers_version": importlib.metadata.version("tokenizers"),
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "add_bos_token": args.backbone == "llama31",
        "local_files_only": True,
        "encoding": "separate-prompt-answer; add_special_tokens=False; backbone BOS; one answer EOS",
    }
    return tokenizer, metadata


def run_training(args: argparse.Namespace, tokenizer: Any, tokenizer_metadata: dict[str, Any]) -> dict[str, Any]:
    root, output = args.dataset_root, args.output_dir
    raw_root = root / "training" / "views"
    destination_view_root = (output / "views").resolve()
    resolved_raw_root = raw_root.resolve()
    if (destination_view_root == resolved_raw_root
            or resolved_raw_root in destination_view_root.parents
            or destination_view_root in resolved_raw_root.parents):
        raise ValueError("destination views must not overlap the raw training views")
    output.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, Any] = {"protocol": TRAINING_PROTOCOL, "max_training_length": args.max_training_length, "backbone": args.backbone, "sequence_packing": False, "padding_serialized": False, "tokenizer": tokenizer_metadata, "splits": {}}
    for split in SPLITS:
        with ExitStack() as stack:
            inputs = {view: iter(iter_jsonl(raw_root / view / f"{split}.jsonl")) for view in VIEWS}
            outputs = {}
            for view in VIEWS:
                path = output / "views" / view / f"{split}.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                outputs[view] = stack.enter_context(path.open("w", encoding="utf-8"))
            membership_path = output / "membership" / f"{split}.jsonl"
            rejection_path = output / "rejections" / f"{split}.jsonl"
            membership_path.parent.mkdir(parents=True, exist_ok=True)
            rejection_path.parent.mkdir(parents=True, exist_ok=True)
            membership = stack.enter_context(membership_path.open("w", encoding="utf-8"))
            rejections = stack.enter_context(rejection_path.open("w", encoding="utf-8"))
            stats = {"seen": 0, "retained": 0, "rejected": 0, "context_trimmed_targets": 0, "max_sequence_lengths": {view: 0 for view in VIEWS}}
            seen: set[str] = set()
            sentinel = object()
            while True:
                rows = {view: next(iterator, sentinel) for view, iterator in inputs.items()}
                if all(row is sentinel for row in rows.values()):
                    break
                if any(row is sentinel for row in rows.values()):
                    raise ValueError(f"{split}: raw four-view member counts differ")
                base = rows["C0"]
                sid = str(base["sample_id"])
                if sid in seen:
                    raise ValueError(f"{split}: duplicate sample_id {sid}")
                seen.add(sid)
                for view, row in rows.items():
                    if str(row["sample_id"]) != sid or row["statement"] != base["statement"] or row["target"] != base["target"] or row.get("target_id") != base.get("target_id"):
                        raise ValueError(f"{split}/{sid}: views differ in target membership, statement, or complete proof")
                    if row.get("split") != split or row.get("view") != view:
                        raise ValueError(f"{split}/{sid}: mismatched split/view label")
                stats["seen"] += 1
                result = prepare_training_views(tokenizer, base["statement"], base["target"], {view: row["context_items"] for view, row in rows.items()}, backbone=args.backbone, max_training_length=args.max_training_length)
                member = {"sample_id": sid, "target_id": base.get("target_id", sid), "split": split, "retained": result["retained"], "views": list(VIEWS)}
                if not result["retained"]:
                    stats["rejected"] += 1
                    rejected = {**member, **result}
                    write_row(rejections, rejected)
                    write_row(membership, rejected)
                    continue
                stats["retained"] += 1
                stats["context_trimmed_targets"] += bool(result["removed_command_ids"])
                write_row(membership, {**member, "reason": None})
                for view in VIEWS:
                    raw_context = rows[view]["context_items"]
                    row = {**rows[view], **result["views"][view], "raw_context_ids": rows[view].get("context_ids", [command_id(item) for item in raw_context]), "raw_context_uids": rows[view].get("context_uids", [command_key(item) for item in raw_context]), "sample_id": sid, "target_id": base.get("target_id", sid), "split": split, "view": view, "protocol": TRAINING_PROTOCOL, "tokenization_status": "complete_real_tokenizer"}
                    write_row(outputs[view], row)
                    stats["max_sequence_lengths"][view] = max(stats["max_sequence_lengths"][view], row["sequence_length"])
            metrics["splits"][split] = stats
    write_json(output / "tokenization_metrics.json", metrics)
    return metrics


class CommandCatalog:
    """Read only the source-file catalogs requested by selector pair records."""

    def __init__(self, dataset_root: Path):
        self.root = dataset_root / "command_mapping"
        self.cache: dict[str, dict[str, dict[str, Any]]] = {}
        self.flat: dict[str, dict[str, dict[str, Any]]] | None = None

    def for_source(self, source_file: str) -> dict[str, dict[str, Any]]:
        if source_file in self.cache:
            return self.cache[source_file]
        relative = Path(source_file)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("source_file must be a safe repository-relative path")
        path = self.root / "source_commands" / relative.with_suffix(".jsonl")
        if path.is_file():
            rows = iter_jsonl(path)
        else:
            flat_path = self.root / "source_commands.jsonl"
            if not flat_path.is_file():
                raise FileNotFoundError(f"missing source-command catalog: {path}")
            if self.flat is None:
                self.flat = {}
                for item in iter_jsonl(flat_path):
                    source = str(item["source_file"])
                    index = self.flat.setdefault(source, {})
                    index[command_id(item)] = item
                    index[command_key(item)] = item
            self.cache[source_file] = self.flat.get(source_file, {})
            return self.cache[source_file]
        index: dict[str, dict[str, Any]] = {}
        for item in rows:
            if item.get("source_file") != source_file:
                raise ValueError(f"{path}: source_file mismatch")
            index[command_id(item)] = item
            index[command_key(item)] = item
        self.cache[source_file] = index
        return index


def run_selector(args: argparse.Namespace, tokenizer: Any, tokenizer_metadata: dict[str, Any]) -> dict[str, Any]:
    input_path = Path(getattr(args, "selector_input_file", None) or args.dataset_root / "selector" / "pairs.jsonl")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "pairs.jsonl"
    if output_path.resolve() == input_path.resolve():
        raise ValueError("output must not overwrite raw selector pairs")
    catalog = CommandCatalog(args.dataset_root)
    metrics: dict[str, Any] = {"template_version": SELECTOR_TEMPLATE, "backbone": args.backbone, "input_limit": args.input_limit, "output_limit": args.output_limit, "tokenizer": tokenizer_metadata, "seen": 0, "both_valid": 0, "both_invalid": 0, "model_generation_performed": False, "coverage_categories": {"baseline": {}, "dependency_prioritized": {}}}
    seen: set[str] = set()
    with output_path.open("w", encoding="utf-8") as output:
        for row in iter_jsonl(input_path):
            sid = str(row.get("sample_id", row["target_id"]))
            if sid in seen:
                raise ValueError(f"duplicate selector target {sid}")
            seen.add(sid)
            commands = catalog.for_source(str(row["source_file"]))
            pool = {str(item["command_id"]) for item in row["candidate_pool"]}
            rankings: dict[str, list[dict[str, Any]]] = {}
            for field in ("baseline_ranking", "dependency_prioritized_ranking"):
                ids = [str(value) for value in row[field]]
                if len(ids) != len(set(ids)) or set(ids) != pool:
                    raise ValueError(f"{sid}: {field} is not a deduplicated complete candidate-pool ranking")
                try:
                    rankings[field] = [commands[value] for value in ids]
                except KeyError as exc:
                    raise ValueError(f"{sid}: ranked command missing from catalog: {exc}") from exc
            target_uid = str(row.get("target_command_uid", ""))
            target_cid = str(row.get("target_command_id", ""))
            for item in rankings["baseline_ranking"]:
                if command_key(item) == target_uid or command_id(item) == target_cid:
                    raise ValueError(f"{sid}: target command appears in candidate pool")
            statement = row.get("statement", row.get("target_statement"))
            environment = row["shared_environment"]
            environment_priority = row.get("shared_environment_priority")
            if environment_priority is not None:
                ec_index = {command_id(item): item for item in environment}
                priority_ids = [str(value) for value in environment_priority]
                if len(priority_ids) != len(set(priority_ids)) or set(priority_ids) != set(ec_index):
                    raise ValueError(f"{sid}: shared_environment_priority does not cover the common EC exactly")
                environment = [ec_index[value] for value in priority_ids]
            result = pack_selector_pair(tokenizer, statement, environment, rankings["baseline_ranking"], rankings["dependency_prioritized_ranking"], backbone=args.backbone, input_limit=args.input_limit, output_limit=args.output_limit, target_type_dependencies=row["target_type_dependencies"], environment_scope_status=row.get("EC_scope_stack_status", "missing_scope_status"))
            # Deliberately do not copy reference proofs or proof-derived evidence.
            packed = {key: row[key] for key in ("sample_id", "target_id", "split", "source_file", "target_command_id", "target_command_uid", "candidate_pool", "baseline_ranking", "dependency_prioritized_ranking", "target_type_dependencies", "shared_environment_priority", "protocol", "rule_revision", "rule_revision_sha256", "EC_identity_status", "EC_scope_stack_status", "EC_same_for_both_policies", "EC_budget_contract") if key in row}
            packed.update({"sample_id": sid, "statement": statement, "backbone": args.backbone, "raw_shared_environment_command_ids": [command_id(item) for item in row["shared_environment"]], "tokenization_status": "complete_real_tokenizer", "final_visibility_status": "actual_whole_command_tokenizer_packing", **result})
            write_row(output, packed)
            metrics["seen"] += 1
            metrics["both_valid" if result["baseline"]["valid_input"] else "both_invalid"] += 1
            for policy in ("baseline", "dependency_prioritized"):
                category = result[policy]["target_coverage"]["status"]
                counts = metrics["coverage_categories"][policy]
                counts[category] = counts.get(category, 0) + 1
    write_json(args.output_dir / "tokenization_metrics.json", metrics)
    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("training", "selector"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--dataset-root", type=Path, required=True)
        sub.add_argument("--tokenizer", required=True, help="actual local tokenizer directory or cached Hugging Face model ID")
        sub.add_argument("--tokenizer-revision", help="immutable full commit; required for remote IDs")
        sub.add_argument("--backbone", choices=("dllm", "llama31"), required=True)
        sub.add_argument("--output-dir", type=Path, required=True)
        sub.add_argument("--trust-remote-code", action="store_true")
        if command == "training":
            sub.add_argument("--max-training-length", type=int, default=4096, help="maximum combined training sequence length")
        else:
            sub.add_argument("--selector-input-file", type=Path,
                             help="Optional proof-free selector pairs; source catalogs still come from dataset-root")
            sub.add_argument("--input-limit", type=int, default=1024)
            sub.add_argument("--output-limit", type=int, default=1024)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "training":
        raise SystemExit(
            "The old combined-length training CLI is superseded. Use "
            "python -m dataset_preparation.tokenize_budgets with "
            "--input-budgets 1024 --output-limit 1024."
        )
    tokenizer, metadata = load_actual_tokenizer(args)
    result = run_selector(args, tokenizer, metadata)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
