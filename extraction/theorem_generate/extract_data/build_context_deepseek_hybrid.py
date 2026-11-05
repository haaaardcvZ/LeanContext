from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Iterable

from extract_data.context_pipeline import (
    add_common_args,
    build_candidates,
    build_output_record,
    case_key,
    filter_cases_by_source_file,
    iter_target_cases,
    load_done_keys,
    load_source_file_filters,
    make_llm_messages,
    normalize_keep_ids,
    command_text,
)
from extract_data.llm_client import ChatClient, ChatConfig


def make_deepseek_client(args: argparse.Namespace, *, model: str) -> ChatClient | None:
    if args.mode != "llm":
        return None
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("missing API key; set DEEPSEEK_API_KEY")
    return ChatClient(
        ChatConfig(
            provider="deepseek-hybrid",
            api_key=api_key,
            base_url=args.base_url.rstrip("/"),
            model=model,
            temperature=args.temperature,
            timeout=args.timeout,
            max_retries=args.max_retries,
        )
    )


def select_with_model(
    *,
    args: argparse.Namespace,
    client: ChatClient | None,
    case,
    candidates,
) -> tuple[list[str], str | None]:
    if args.mode == "candidates":
        return [], None
    if args.mode == "heuristic":
        return [str(rec.get("command_id")) for rec in candidates if rec.get("command_id")], (
            "heuristic keeps all generated candidates"
        )
    if client is None:
        raise ValueError("llm mode requires a ChatClient")
    obj = client.chat_json(
        make_llm_messages(
            case,
            candidates,
            max_chars_per_command=args.max_chars_per_command,
        )
    )
    reason = obj.get("reason")
    return normalize_keep_ids(obj, candidates, str(case.target.get("command_id"))), (
        reason if isinstance(reason, str) else None
    )


def run_hybrid(args: argparse.Namespace) -> None:
    input_root = Path(args.input_root).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    failed_path = Path(args.failed_output).resolve() if args.failed_output else output_path.with_suffix(
        output_path.suffix + ".failed.jsonl"
    )
    failed_path.parent.mkdir(parents=True, exist_ok=True)

    source_file_filters = load_source_file_filters(args)

    flash_client = make_deepseek_client(args, model=args.flash_model)
    pro_client = make_deepseek_client(args, model=args.pro_model)

    done = load_done_keys(output_path) if args.resume and args.mode != "candidates" else set()
    write_mode = "a" if args.resume and output_path.exists() and args.mode != "candidates" else "w"
    failed_write_mode = "a" if args.resume and failed_path.exists() else "w"

    def filtered_cases() -> Iterable[tuple[int, Any]]:
        yielded = 0
        for seen, case in enumerate(iter_target_cases(input_root), start=1):
            if source_file_filters and str(case.target.get("source_file")) not in source_file_filters:
                continue
            yielded += 1
            yield seen, case
            if args.limit and yielded >= args.limit:
                break

    def process_one(seen_idx: int, case: Any) -> dict[str, Any]:
        started = time.time()
        candidates = build_candidates(
            case,
            recursive_depth=args.recursive_depth,
            max_dependency_candidates=args.max_dependency_candidates,
        )
        use_pro = len(candidates) > args.pro_threshold
        model = args.pro_model if use_pro else args.flash_model
        provider = "deepseek-v4-pro" if use_pro else "deepseek-v4-flash"
        client = pro_client if use_pro else flash_client

        if args.mode == "candidates":
            rec = {
                "lean_file_name": case.target.get("source_file"),
                "source_module": case.target.get("source_module"),
                "source_jsonl": case.source_path.as_posix(),
                "target_command_id": case.target.get("command_id"),
                "target_kind": case.target.get("kind"),
                "theorem_name": case.target.get("name"),
                "target_start_line": case.target.get("start_line"),
                "target_end_line": case.target.get("end_line"),
                "target_text": command_text(case.target),
                "candidate_count": len(candidates),
                "planned_provider": provider,
                "planned_model": model,
                "candidates": [
                    {
                        "command_id": rec.get("command_id"),
                        "kind": rec.get("kind"),
                        "name": rec.get("name"),
                        "start_line": rec.get("start_line"),
                        "end_line": rec.get("end_line"),
                        "text": command_text(rec),
                    }
                    for rec in candidates
                ],
            }
        else:
            keep_ids, reason = select_with_model(
                args=args,
                client=client,
                case=case,
                candidates=candidates,
            )
            rec = build_output_record(
                case,
                candidates,
                keep_ids,
                provider=provider,
                model=model,
                llm_reason=reason,
                include_import_mathlib=args.import_mathlib,
                include_closing_ends=args.include_closing_ends,
                proof_end_marker=args.proof_end_marker,
            )
            rec["hybrid_threshold"] = args.pro_threshold

        return {
            "status": "ok",
            "seen_idx": seen_idx,
            "record": rec,
            "use_pro": use_pro,
            "elapsed_sec": round(time.time() - started, 3),
        }

    def process_one_safe(seen_idx: int, case: Any) -> dict[str, Any]:
        try:
            return process_one(seen_idx, case)
        except Exception as e:
            return {
                "status": "failed",
                "seen_idx": seen_idx,
                "source_file": case.target.get("source_file"),
                "target_command_id": case.target.get("command_id"),
                "target_kind": case.target.get("kind"),
                "theorem_name": case.target.get("name"),
                "error": repr(e),
            }

    written = 0
    skipped = 0
    flash_count = 0
    pro_count = 0
    failed = 0
    submitted = 0
    seen_cases = 0
    max_in_flight = args.max_in_flight or max(args.workers * 2, args.workers)

    def handle_result(result: dict[str, Any], out: Any, failed_out: Any) -> None:
        nonlocal written, failed, flash_count, pro_count
        if result["status"] == "ok":
            rec = result["record"]
            if result["use_pro"]:
                pro_count += 1
            else:
                flash_count += 1
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
            if args.flush_every and written % args.flush_every == 0:
                out.flush()
        else:
            failed_out.write(json.dumps(result, ensure_ascii=False) + "\n")
            failed += 1
            if args.flush_every and failed % args.flush_every == 0:
                failed_out.flush()

        done_count = written + failed + skipped
        if args.progress_every and done_count % args.progress_every == 0:
            print(
                f"[progress] done={done_count} written={written} failed={failed} "
                f"submitted={submitted} flash={flash_count} pro={pro_count} skipped={skipped}"
            )

    with output_path.open(write_mode, encoding="utf-8") as out, failed_path.open(
        failed_write_mode, encoding="utf-8"
    ) as failed_out:
        if args.workers <= 1:
            for seen_idx, case in filtered_cases():
                seen_cases += 1
                key = case_key(case)
                if key in done:
                    skipped += 1
                    continue
                submitted += 1
                handle_result(process_one_safe(seen_idx, case), out, failed_out)
        else:
            futures: set[Future[dict[str, Any]]] = set()
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                for seen_idx, case in filtered_cases():
                    seen_cases += 1
                    key = case_key(case)
                    if key in done:
                        skipped += 1
                        continue
                    futures.add(ex.submit(process_one_safe, seen_idx, case))
                    submitted += 1

                    if len(futures) >= max_in_flight:
                        done_futures, futures = wait(futures, return_when=FIRST_COMPLETED)
                        for fut in done_futures:
                            handle_result(fut.result(), out, failed_out)

                while futures:
                    done_futures, futures = wait(futures, return_when=FIRST_COMPLETED)
                    for fut in done_futures:
                        handle_result(fut.result(), out, failed_out)

    print(
        json.dumps(
            {
                "input_root": input_root.as_posix(),
                "output": output_path.as_posix(),
                "failed_output": failed_path.as_posix(),
                "mode": args.mode,
                "flash_model": args.flash_model,
                "pro_model": args.pro_model,
                "pro_threshold": args.pro_threshold,
                "workers": args.workers,
                "max_in_flight": max_in_flight,
                "source_file_filters": sorted(source_file_filters),
                "cases_seen": seen_cases,
                "submitted": submitted,
                "written": written,
                "skipped": skipped,
                "failed": failed,
                "flash_cases": flash_count,
                "pro_cases": pro_count,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build minimized theorem contexts with DeepSeek Flash for easy cases and Pro for hard cases."
    )
    add_common_args(parser)
    parser.add_argument("--base-url", default=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--flash-model", default=os.environ.get("DEEPSEEK_FLASH_MODEL", "deepseek-v4-flash"))
    parser.add_argument("--pro-model", default=os.environ.get("DEEPSEEK_PRO_MODEL", "deepseek-v4-pro"))
    parser.add_argument(
        "--pro-threshold",
        type=int,
        default=30,
        help="Use Pro when candidate_count is greater than this value.",
    )
    parser.add_argument("--workers", type=int, default=1, help="Parallel LLM/API workers.")
    parser.add_argument(
        "--max-in-flight",
        type=int,
        default=None,
        help="Maximum submitted but unfinished tasks. Defaults to workers * 2.",
    )
    parser.add_argument(
        "--failed-output",
        default=None,
        help="JSONL path for failed targets. Defaults to <output>.failed.jsonl.",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=25,
        help="Flush output files after this many written records.",
    )
    args = parser.parse_args()
    run_hybrid(args)


if __name__ == "__main__":
    main()
