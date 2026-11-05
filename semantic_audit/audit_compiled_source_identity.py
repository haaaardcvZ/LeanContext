"""Read-only source-byte and reference-proof identity audit for semantic labels."""
import argparse
import collections
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--audit-root', type=Path, required=True)
parser.add_argument('--parser-root', type=Path, required=True,
                    help='Command-span root containing Mathlib/**/*.jsonl')
parser.add_argument('--source-root', type=Path, required=True,
                    help='Pinned Mathlib source checkout')
args = parser.parse_args()
HERE = args.audit_root.resolve()
SOURCE = args.source_root.resolve()
EXTRACTION = Path(__file__).resolve().parents[1] / 'extraction/theorem_generate'
sys.path.insert(0, str(EXTRACTION))
from extract_data.context_pipeline import split_target_text

spec = importlib.util.spec_from_file_location('original_comment_processing', EXTRACTION / 'lean_parser/batch_extract_mathlib_command_spans.py')
normalizer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(normalizer)

def rows(path):
    with path.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

by_file = collections.defaultdict(dict)
for row in rows(HERE / 'semantic_inputs.jsonl'):
    by_file[row['source_file']][row['target_command_id']] = row
counts = collections.Counter()
split_counts = collections.defaultdict(collections.Counter)
failures = []
with (HERE / 'compiled_source_identity.jsonl').open('w') as out:
    for file, samples in sorted(by_file.items()):
        source_path = SOURCE / file
        if not source_path.exists():
            failures.append({'source_file': file, 'failure': 'source_missing'})
            continue
        data = source_path.read_bytes()
        counts['source_files'] += 1
        for command in rows(args.parser_root / Path(file).with_suffix('.jsonl')):
            actual = data[command['start_byte']:command['end_byte']].decode('utf-8')
            raw_match = actual == command['raw_text']
            effective = normalizer.normalize_effective_raw_text(actual)
            effective_match = effective == command['effective_raw_text']
            counts['source_commands'] += 1
            counts['source_raw_text_exact_match'] += raw_match
            counts['recomputed_effective_text_exact_match'] += effective_match
            if not raw_match or not effective_match:
                failures.append({'source_file': file, 'command_id': command['command_id'], 'source_raw_match': raw_match, 'effective_match': effective_match})
            row = samples.get(command['command_id'])
            if row is None:
                continue
            body = split_target_text(effective)['target_body_text'].strip()
            proof_match = body == row['reference_proof'].strip()
            result = {
                'sample_id': row['sample_id'], 'split': row['split'],
                'source_file': file, 'target_command_id': command['command_id'],
                'compiled_source_raw_text_exact_match': raw_match,
                'compiled_source_effective_text_exact_match': effective_match,
                'reference_proof_matches_recomputed_source_body': proof_match,
                'source_command_sha256': hashlib.sha256(actual.encode()).hexdigest(),
                'reference_proof_sha256': hashlib.sha256(row['reference_proof'].strip().encode()).hexdigest(),
            }
            out.write(json.dumps(result, ensure_ascii=False, separators=(',', ':')) + '\n')
            for counter in (counts, split_counts[row['split']]):
                counter['samples'] += 1
                counter['reference_proof_matches_recomputed_source_body'] += proof_match
                counter['target_source_raw_text_exact_match'] += raw_match
                counter['target_effective_text_exact_match'] += effective_match
            if not proof_match:
                failures.append({'sample_id': row['sample_id'], 'failure': 'proof_mismatch', 'source_body': body, 'reference_proof': row['reference_proof']})
manifest = {
    'schema_version': 'compiled-source-identity-v1',
    'source_root': str(SOURCE),
    'method': 'Compare original parser raw_text byte-for-byte with restored pinned Mathlib source spans; independently reapply original comment/blank-line normalization and header/body splitter, then compare reference proof exactly after outer stripping. No inference or model calls.',
    'counts': dict(counts),
    'split_counts': {k: dict(v) for k, v in split_counts.items()},
    'failures': failures,
}
(HERE / 'compiled_source_identity_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(manifest, ensure_ascii=False, indent=2))
