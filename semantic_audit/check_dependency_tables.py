"""Independent semantic table validation and paired-population sensitivity."""
import csv,json,collections,statistics,math
from pathlib import Path
R=Path(__file__).resolve().parent; O=R.parents[1]/'outputs'/'leancontext_dependencies'
def jl(p):
 with open(p) as f:
  for line in f: yield json.loads(line)
def table(n): return list(csv.DictReader(open(O/'tables'/n,encoding='utf-8-sig')))
inputs={r['sample_id']:{'split':r['split'],'mc':set(r['selected_mc_command_ids']),'ec':set(r['selected_ec_command_ids'])} for r in jl(R/'semantic_inputs.jsonl')}
summary=json.load(open(R/'semantic_mc_summary.json'))
group=collections.defaultdict(lambda:collections.defaultdict(list)); pair=collections.defaultdict(collections.Counter); frac=collections.defaultdict(lambda:collections.defaultdict(list))
for d in jl(R/'semantic_mc_labels.jsonl'):
 i=inputs[d['sample_id']]; cat='mc_positive' if d['d_m_value_minus_type'] is True else 'mc_negative' if d['d_m_value_minus_type'] is False else 'mc_uncertain'
 for s in [i['split'],'all']:
  for g in [cat,'population']:
   group[(s,g)]['mc'].append(len(i['mc']));group[(s,g)]['ec'].append(len(i['ec']))
   if d['mc_semantic_status']=='complete': group[(s,g)]['dep'].append(d['proof_extra_constant_commands'])
  if d['mc_semantic_status']=='complete':
   t=d['type_dependency']
   if not t['unmapped_same_file_constants'] and not t['nonpreceding_same_file_constants']:
    pair[s]['common_complete_targets']+=1
    for key,ref in [('proof_extra',d['proof_extra_dependency']),('target_type',t)]:
     ids=set(ref['command_ids']); retained=ids&i['mc']
     pair[s][key+'_dependency_commands']+=len(ids);pair[s][key+'_selected_commands']+=len(retained)
     if ids:
      pair[s][key+'_positive_targets']+=1;frac[s][key].append(len(retained)/len(ids))
checks=[]
for r in table('selected_dependency_coverage.csv'):
 s=r['split']
 if s=='all':continue
 c=summary['coverage_complete_positive'][s]
 pairs={'complete_positive_targets':'targets','proof_extra_dependency_edges':'dependency_commands','selected_proof_extra_dependency_edges':'selected_dependency_commands','micro_proof_extra_coverage':'selected_micro_coverage','macro_proof_extra_coverage':'selected_mean_target_coverage','selected_any':'targets_selected_any','selected_all':'targets_selected_all'}
 for out,inside in pairs.items(): assert math.isclose(float(r[out]),c[inside],rel_tol=1e-12),(s,out,r[out],c[inside])
 assert int(r['complete_targets'])==summary['counts'][s]['complete']
 assert int(r['complete_type_targets'])==pair[s]['common_complete_targets']
 assert int(r['type_dependency_edges'])==pair[s]['target_type_dependency_commands']
 assert int(r['selected_type_dependency_edges'])==pair[s]['target_type_selected_commands']
checks.append('all split proof coverage equals frozen mapper; type coverage equals stated intersection population')
for r in table('selection_stages.csv'):
 s=r['split']; stage=r['stage']
 if s=='all': continue
 c=summary['coverage_complete_positive'][s]
 field={'eligible':'dependency_commands','candidate':'candidate_dependency_commands','prefilter':'prefilter_dependency_commands','selected':'selected_dependency_commands','removed':'removed_dependency_commands'}[stage]
 assert int(r['proof_extra_dependency_edges_complete'])==c[field],(s,stage)
 assert math.isclose(float(r['proof_extra_coverage_complete']),c[field]/c['dependency_commands'],rel_tol=1e-12)
 if stage=='selected':assert int(r['all_population_MC_command_occurrences'])==sum(group[(s,'population')]['mc'])
checks.append('all five proof selection stages and distinct per-target command denominators equal frozen mapper')
for r in table('command_counts_by_group.csv'):
 if r['group'].startswith('ec_'): continue
 v=group[(r['split'],r['group'])]
 assert int(r['N'])==len(v['mc'])
 for label,key in [('MC','mc'),('EC','ec')]:
  assert int(r[f'selected_{label}_sum'])==sum(v[key])
  assert math.isclose(float(r[f'selected_{label}_mean']),statistics.mean(v[key]),rel_tol=1e-12)
  assert math.isclose(float(r[f'selected_{label}_median']),statistics.median(v[key]),rel_tol=1e-12)
 assert int(r['complete_mapping_N'])==len(v['dep'])
 if v['dep']:assert math.isclose(float(r['proof_extra_commands_mean_complete']),statistics.mean(v['dep']),rel_tol=1e-12)
checks.append('MC group population counts, selected MC/EC source-command counts, means, medians and complete-subset dependency means independently verified')
result={}
for s,c in pair.items():
 v=dict(c)
 for k in ['proof_extra','target_type']:
  v[k+'_micro_coverage']=v[k+'_selected_commands']/v[k+'_dependency_commands']
  v[k+'_macro_positive_coverage']=statistics.mean(frac[s][k])
 v['type_minus_proof_extra_micro_pp']=100*(v['target_type_micro_coverage']-v['proof_extra_micro_coverage'])
 result[s]=v
out={'definition':'Both coverage measures use the identical population with complete proof-extra source attribution and complete target-type source attribution. Micro denominators count distinct source command edges per target. Macro denominators use only positive targets for that expression.','splits':result,'independent_table_checks':checks}
(R/'dependency_coverage_common_population.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
