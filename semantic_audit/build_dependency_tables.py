"""Aggregate audited semantic dependencies at source-command granularity.

No generation, model inference, tokenizer or character count is involved.
"""
from pathlib import Path
from collections import Counter, defaultdict
import csv
import gzip
import json
import statistics

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parents[1] / 'outputs' / 'leancontext_dependencies'
SPLITS = ['train', 'validation', 'test']
EC_GROUPS = ['00', '10', '01', '11', 'uncertain']
EC_NAMES = {
    '00': 'Neither target nor selected MC expression',
    '10': 'Target expression only',
    '01': 'Selected MC expression only',
    '11': 'Both target and selected MC expressions',
    'uncertain': 'Uncertain expression-dependency label',
}

def rows(path):
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def csvout(name, data):
    if not data:
        return
    with open(OUT / 'tables' / name, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, list(data[0]))
        w.writeheader()
        w.writerows(data)

def ratio(a, b):
    return a / b if b else None

def run():
    (OUT / 'tables').mkdir(parents=True, exist_ok=True)
    (OUT / 'labels').mkdir(exist_ok=True)
    inputs = {r['sample_id']: r for r in rows(ROOT / 'semantic_inputs.jsonl')}
    mathematical = {(r['source_module'], r['command_id'])
                    for r in rows(ROOT / 'semantic_source_commands.jsonl')
                    if r['is_mathematical_command']}
    mc = {r['sample_id']: r for r in rows(ROOT / 'semantic_mc_labels.jsonl')}
    ecpath = ROOT / 'ec_semantic_test_final.jsonl'
    if not ecpath.exists():
        ecpath = ROOT / 'ec_semantic_results.jsonl'
    ec = {r['sample_id']: r for r in rows(ecpath)} if ecpath.exists() else {}
    assert set(inputs) == set(mc)
    assert set(ec) <= set(inputs)
    complete_ec_splits = {s for s in SPLITS if
        {sid for sid,r in ec.items() if r['split']==s} ==
        {sid for sid,r in inputs.items() if r['split']==s}}
    ec = {sid:r for sid,r in ec.items() if r['split'] in complete_ec_splits}
    aggregate = {}
    all_labels = []
    group_stats = []
    population_rows = []
    coverage_rows = []
    command_rows = []
    stage_rows = []
    joint_rows = []
    ec_edge_rows = []
    for split in SPLITS + ['all']:
        subset = [v for v in inputs.values() if split == 'all' or v['split'] == split]
        n = len(subset)
        ec_complete = split in complete_ec_splits or (split=='all' and len(complete_ec_splits)==3)
        categories = Counter()
        group_values = defaultdict(lambda: defaultdict(list))
        stage_commands = Counter()
        proof_stage_edges = Counter()
        type_stage_edges = Counter()
        coverage = Counter()
        fractions = []
        type_fractions = []
        complete = []
        joints = Counter()
        ec_edges = Counter()
        for inp in subset:
            sid = inp['sample_id']; dep = mc[sid]
            category = ('positive' if dep.get('d_m_value_minus_type') is True
                        else 'negative' if dep.get('d_m_value_minus_type') is False
                        else 'uncertain')
            categories['mc_' + category] += 1
            groups = ['mc_' + category]
            er = ec.get(sid) if ec_complete else None
            if er is not None:
                eg = er['ec_group']
                assert eg in EC_GROUPS
                categories['ec_' + eg] += 1
                groups.append('ec_' + eg)
                joints[(category, eg)] += 1
                target_edges = set(er.get('target_ec_command_ids', []))
                mc_edges = set(er.get('mc_ec_command_ids', []))
                ec_edges['observed_EC_to_target_command_edges'] += len(target_edges)
                ec_edges['observed_EC_to_MC_command_edges'] += len(mc_edges)
                ec_edges['observed_EC_to_either_command_edges'] += len(target_edges | mc_edges)
                ec_edges['targets_with_resolved_EC_to_target_origin'] += bool(target_edges)
                ec_edges['targets_with_resolved_EC_to_MC_origin'] += bool(mc_edges)
            mcount = len(inp['selected_mc_command_ids'])
            ecount = len(inp['selected_ec_command_ids'])
            for g in groups + ['population']:
                group_values[g]['mc'].append(mcount)
                group_values[g]['ec'].append(ecount)
                if dep.get('mc_semantic_status') == 'complete':
                    group_values[g]['proof_extra'].append(dep['proof_extra_constant_commands'])
            stages = {
                'candidate': set(inp['selector_candidate_command_ids']),
                'prefilter': set(inp['selected_command_ids_before_context_filter']),
                'selected': set(inp['selected_mc_command_ids']),
                'removed': set(inp['removed_context_command_ids']),
            }
            stage_commands['eligible'] += inp['eligible_preceding_mathematical_commands']
            for s, ids in stages.items():
                stage_commands[s] += sum((inp['source_module'], c) in mathematical for c in ids)
            coverage['filter_affected_targets'] += bool(stages['removed'])
            if dep.get('mc_semantic_status') == 'complete':
                complete.append(sid)
                coverage['complete_targets'] += 1
                ref = set(dep['proof_extra_dependency']['command_ids'])
                proof_stage_edges['eligible'] += len(ref)
                for s, ids in stages.items():
                    proof_stage_edges[s] += len(ref & ids)
                if ref:
                    coverage['complete_positive_targets'] += 1
                    retained = ref & stages['selected']
                    coverage['selected_any'] += bool(retained)
                    coverage['selected_all'] += ref <= stages['selected']
                    coverage['selected_none'] += not retained
                    fractions.append(len(retained) / len(ref))
                tp = dep['type_dependency']
                if not tp['unmapped_same_file_constants'] and not tp['nonpreceding_same_file_constants']:
                    coverage['complete_type_targets'] += 1
                    tref = set(tp['command_ids'])
                    type_stage_edges['eligible'] += len(tref)
                    for s, ids in stages.items():
                        type_stage_edges[s] += len(tref & ids)
                    if tref:
                        coverage['type_positive_targets'] += 1
                        coverage['type_selected_any'] += bool(tref & stages['selected'])
                        coverage['type_selected_all'] += tref <= stages['selected']
                        type_fractions.append(len(tref & stages['selected']) / len(tref))
            if split != 'all':
                label = {
                    'sample_id': sid, 'split': split, 'source_file': inp['source_file'],
                    'target_command_id': inp['target_command_id'], 'target_name': dep.get('target_name'),
                    'mc_group': category, 'mc_extraction_status': dep['mc_semantic_status'],
                    'selected_mc_commands': mcount, 'selected_ec_commands': ecount,
                    'eligible_preceding_mc_commands': inp['eligible_preceding_mathematical_commands'],
                    'proof_extra_dependency_commands': dep.get('proof_extra_constant_commands'),
                    'target_type_dependency_commands': dep.get('type_local_commands'),
                    'selected_proof_extra_commands': dep.get('selected_extra_commands'),
                    'selected_proof_extra_coverage': dep.get('selected_extra_coverage'),
                    'ec_group': er['ec_group'] if er else 'pending_analysis',
                    'ec_target_dependency': er.get('target_dep') if er else None,
                    'ec_mc_dependency': er.get('mc_dep') if er else None,
                    'observed_target_ec_commands': len(er.get('target_ec_command_ids', [])) if er else None,
                    'observed_mc_ec_commands': len(er.get('mc_ec_command_ids', [])) if er else None,
                }
                all_labels.append(label)
        assert sum(categories['mc_' + x] for x in ['positive', 'negative', 'uncertain']) == n
        assert not ec_complete or sum(categories['ec_' + x] for x in EC_GROUPS) == n
        assert proof_stage_edges['prefilter'] - proof_stage_edges['removed'] == proof_stage_edges['selected']
        population_rows.append({'split': split, 'both_nonempty_targets': n,
            'source_files': len({r['source_file'] for r in subset}),
            'mc_positive': categories['mc_positive'], 'mc_negative': categories['mc_negative'],
            'mc_unresolved': categories['mc_uncertain'],
            'complete_mc_mapping': coverage['complete_targets']})
        for g, vals in group_values.items():
            group_stats.append({'split': split, 'group': g, 'N': len(vals['mc']),
                'selected_MC_mean': statistics.mean(vals['mc']),
                'selected_MC_median': statistics.median(vals['mc']),
                'selected_MC_sum': sum(vals['mc']),
                'selected_EC_mean': statistics.mean(vals['ec']),
                'selected_EC_median': statistics.median(vals['ec']),
                'selected_EC_sum': sum(vals['ec']),
                'complete_mapping_N': len(vals['proof_extra']),
                'proof_extra_commands_mean_complete': statistics.mean(vals['proof_extra']) if vals['proof_extra'] else None})
        for g in ['mc_positive', 'mc_negative'] + ['ec_' + x for x in EC_GROUPS]:
            command_rows.append({'split': split, 'group': g,
                'N': categories[g] if (not g.startswith('ec_') or ec_complete) else 'TBD',
                'LLaDA_conditional_gain_pp': 'TBD', 'Llama_conditional_gain_pp': 'TBD'})
        for s in ['eligible', 'candidate', 'prefilter', 'selected', 'removed']:
            stage_rows.append({'split': split, 'stage': s,
                'all_population_MC_command_occurrences': stage_commands[s],
                'proof_extra_dependency_edges_complete': proof_stage_edges[s],
                'proof_extra_coverage_complete': ratio(proof_stage_edges[s], proof_stage_edges['eligible']),
                'target_type_dependency_edges_complete': type_stage_edges[s],
                'target_type_coverage_complete': ratio(type_stage_edges[s], type_stage_edges['eligible'])})
        if ec_complete:
            for mg in ['positive', 'negative', 'uncertain']:
                for eg in EC_GROUPS:
                    joint_rows.append({'split': split, 'mc_group': mg, 'ec_group': eg, 'N': joints[(mg, eg)]})
            ec_edge_rows.append({'split': split, 'N': n, **dict(ec_edges),
                'count_scope': 'resolved selected-EC source-command witnesses; not exhaustive semantic necessity'})
        coverage_rows.append({'split': split, **dict(coverage),
            'proof_extra_dependency_edges': proof_stage_edges['eligible'],
            'selected_proof_extra_dependency_edges': proof_stage_edges['selected'],
            'micro_proof_extra_coverage': ratio(proof_stage_edges['selected'], proof_stage_edges['eligible']),
            'macro_proof_extra_coverage': statistics.mean(fractions) if fractions else None,
            'all_retained_fraction': ratio(coverage['selected_all'], coverage['complete_positive_targets']),
            'any_retained_fraction': ratio(coverage['selected_any'], coverage['complete_positive_targets']),
            'type_dependency_edges': type_stage_edges['eligible'],
            'selected_type_dependency_edges': type_stage_edges['selected'],
            'micro_type_coverage': ratio(type_stage_edges['selected'], type_stage_edges['eligible']),
            'macro_type_coverage': statistics.mean(type_fractions) if type_fractions else None})
        aggregate[split] = {'N': n, 'ec_audit_complete': ec_complete, 'groups': dict(categories), 'coverage': coverage_rows[-1],
            'stage_commands': dict(stage_commands), 'proof_stage_edges': dict(proof_stage_edges),
            'type_stage_edges': dict(type_stage_edges)}
    csvout('table3_seven_groups.csv', command_rows)
    csvout('population_and_mc_coverage.csv', population_rows)
    csvout('command_counts_by_group.csv', group_stats)
    csvout('selection_stages.csv', stage_rows)
    csvout('selected_dependency_coverage.csv', coverage_rows)
    csvout('mc_ec_joint_counts.csv', joint_rows)
    csvout('observed_ec_dependency_commands.csv', ec_edge_rows)
    for split in SPLITS:
        data = [r for r in all_labels if r['split'] == split]
        with open(OUT / 'labels' / (split + '.csv'), 'w', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, list(data[0])); w.writeheader(); w.writerows(data)
    (OUT / 'summary.json').write_text(json.dumps(aggregate, ensure_ascii=False, indent=2))
    print(json.dumps(aggregate['test'], ensure_ascii=False, indent=2))

if __name__ == '__main__':
    run()
