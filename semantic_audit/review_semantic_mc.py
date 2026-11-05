"""Independent checks of source-mapped kernel-reference MC descriptors."""
import argparse, collections, hashlib, json
from pathlib import Path
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--audit-root', type=Path, required=True)
parser.add_argument('--parser-root', type=Path, required=True,
                    help='Command-span root containing Mathlib/**/*.jsonl')
args = parser.parse_args()
HERE=args.audit_root.resolve()
def rows(p):
 with p.open() as f:
  for l in f:
   if l.strip(): yield json.loads(l)
commands={ (r['source_module'],r['command_id']):r for r in rows(HERE/'semantic_source_commands.jsonl') }
inputs={r['sample_id']:r for r in rows(HERE/'semantic_inputs.jsonl')}
counts=collections.Counter(); split=collections.defaultdict(collections.Counter); issues=[]; pools=collections.defaultdict(list)
manual_path=HERE/'semantic_mc_manually_reviewed_ids.json'
manual_ids=set(json.loads(manual_path.read_text())['sample_ids']) if manual_path.exists() else set()
def check(ok,kind,row,detail=None):
 counts['checks']+=1
 if not ok: issues.append({'sample_id':row['sample_id'],'check':kind,'detail':detail})
for r in rows(HERE/'semantic_mc_labels.jsonl'):
 inp=inputs[r['sample_id']];mod=inp['source_module'];sc=split[r['split']]
 counts['rows']+=1;sc['rows']+=1
 check(r['source_file']==inp['source_file'] and r['target_command_id']==inp['target_command_id'],'source_identity_join',r)
 if r.get('target_name') is None:
  sc['unresolved_target']+=1
 else:
  check(set(r['value_minus_type_constants'])==set(r['value_constants'])-set(r['type_constants']),'constant_difference',r)
  for depkey,names in [('type_dependency',r['type_constants']),('value_dependency',r['value_constants']),('proof_extra_dependency',r['value_minus_type_constants'])]:
   dep=r[depkey]
   check(set(dep['command_ids'])==set(dep['constant_evidence']),'evidence_command_keys',r,depkey)
   for cid,evs in dep['constant_evidence'].items():
    cmd=commands[(mod,cid)]
    check(cmd['is_mathematical_command'],'command_is_mc',r,cid)
    check(cmd['start_byte']<inp['target_start_byte'] and cmd['end_byte']<=inp['target_start_byte'],'strictly_preceding',r,cid)
    for ev in evs:
     check(ev['constant'] in names,'constant_in_recorded_set',r,ev)
     counts['evidence_occurrences']+=1
     if ev['source_attribution'].startswith('generated_name_parent:'):counts['generated_parent_evidence_occurrences']+=1
  extra=set(r['proof_extra_dependency']['command_ids']);val=set(r['value_dependency']['command_ids']);typ=set(r['type_dependency']['command_ids'])
  check(r['proof_extra_command_commands']==len(val-typ),'command_difference',r)
  check(r['proof_extra_constant_commands']==len(extra),'constant_diff_command_count',r)
  for prefix,key in [('selected','selected_mc_command_ids'),('candidate','selector_candidate_command_ids'),('prefilter','selected_command_ids_before_context_filter'),('removed','removed_context_command_ids')]:
   check(r[prefix+'_extra_commands']==len(extra&set(inp[key])),'selection_stage_count',r,prefix)
  check(set(inp['selected_command_ids'])|set(inp['removed_context_command_ids'])==set(inp['selected_command_ids_before_context_filter']),'filter_partition',r)
  check(not(set(inp['selected_command_ids'])&set(inp['removed_context_command_ids'])),'filter_disjoint',r)
  bad=bool(r['proof_extra_dependency']['unmapped_same_file_constants'] or r['proof_extra_dependency']['nonpreceding_same_file_constants'])
  check(r['d_m_value_minus_type'] is not False or not bad,'no_unknown_negative',r)
  check(r['d_m_value_minus_type']==(True if extra else None if bad else False),'three_valued_label',r)
  sc['positive']+=r['d_m_value_minus_type'] is True
  sc['negative']+=r['d_m_value_minus_type'] is False
  sc['unknown_mapping']+=r['d_m_value_minus_type'] is None
  sc['complete_mapping']+=not bad
  sc['positive_with_incomplete_mapping']+=bad and bool(extra)
  sc['constant_positive_command_negative']+=bool(extra) and not(val-typ)
  sc['constant_vs_command_count_different']+=len(extra)!=len(val-typ)
  sc['constant_positive_command_negative_complete']+=not bad and bool(extra) and not(val-typ)
  sc['negative_has_self_auxiliary']+=r['d_m_value_minus_type'] is False and bool(r['proof_extra_dependency']['same_target_command_constants'])
  sc['has_other_local_non_mc_command']+=bool(r['proof_extra_dependency']['other_local_commands'])
 if r['split']!='test': continue
 groups=[r['mc_semantic_status']+'_'+str(r.get('d_m_value_minus_type'))]
 if r['sample_id'] in manual_ids:groups.append('source_manually_spotchecked')
 if r.get('proof_extra_constant_commands') and not r.get('proof_extra_command_commands'):groups.append('constant_vs_command_definition')
 if any(e['source_attribution'].startswith('generated_name_parent:') for es in r.get('proof_extra_dependency',{}).get('constant_evidence',{}).values() for e in es):groups.append('generated_parent')
 if r.get('removed_extra_commands'):groups.append('removed_dependencies')
 if r.get('selected_extra_commands'):groups.append('selected_dependencies')
 if r.get('self_auxiliary_extra_constants') and r.get('d_m_value_minus_type') is False:groups.append('negative_self_auxiliary')
 for g in groups:pools[g].append((hashlib.sha256(('semantic-review-v1:'+r['sample_id']).encode()).hexdigest(),r))
selected={}
for group,rs in sorted(pools.items()):
 for _,row in sorted(rs,key=lambda x:x[0])[:len(rs) if group=='source_manually_spotchecked' else 6]:
  ent=selected.setdefault(row['sample_id'],{'selection_strata':[],'annotation':row,'input':inputs[row['sample_id']]})
  ent['selection_strata'].append(group)
targets={s['annotation'].get('target_name') for s in selected.values()}
refs={e['constant'] for s in selected.values() for es in s['annotation'].get('proof_extra_dependency',{}).get('constant_evidence',{}).values() for e in es}
decls={r['name']:r for r in rows(HERE/'declaration_dependencies.jsonl') if r['name'] in targets|refs}
for s in selected.values():
 r=s['annotation'];target=decls.get(r.get('target_name'));s['kernel_target']=target
 if target:
  check(target['module']==r['source_module'],'kernel_target_module',r)
  check(set(target['type_constants'])==set(r.get('outer_type_constants',r['type_constants'])) and set(target['value_constants'])==set(r.get('outer_value_constants',r['value_constants'])),'kernel_target_actual_references',r)
  cmd=commands[(r['source_module'],r['target_command_id'])]
  pos=target.get('selection_range') or target.get('range')
  if pos:check((cmd['start_line'],cmd['start_col']) <= (pos['start_line'],pos['start_column']) < (cmd['end_line'],cmd['end_col']),'kernel_target_range',r)
 s['kernel_dependencies']={n:decls[n] for es in r.get('proof_extra_dependency',{}).get('constant_evidence',{}).values() for ev in es if (n:=ev['constant']) in decls}
 for cid,es in r.get('proof_extra_dependency',{}).get('constant_evidence',{}).items():
  command=commands[(r['source_module'],cid)]
  for ev in es:
   actual=decls[ev['constant']]
   check(actual['module']==r['source_module'],'kernel_witness_module',r,ev['constant'])
   if ev['source_attribution'] in ('kernel_selection_range','kernel_declaration_range'):
    pos=actual['selection_range'] if ev['source_attribution']=='kernel_selection_range' else actual['range']
    check(bool(pos) and (command['start_line'],command['start_col']) <= (pos['start_line'],pos['start_column']) < (command['end_line'],command['end_col']),'kernel_witness_source_range',r,ev['constant'])
 path=args.parser_root/Path(r['source_file']).with_suffix('.jsonl')
 needed={r['target_command_id']}|set(r.get('proof_extra_dependency',{}).get('command_ids',[]))
 s['source_command_texts']={c['command_id']:c['effective_raw_text'] for c in rows(path) if c['command_id'] in needed}
out={'review_type':'independent code/source audit; not human gold labels','counts':dict(counts),'split_counts':{k:dict(v) for k,v in split.items()},'issues':issues,'selected_samples':len(selected),'stratum_population':{k:len(v) for k,v in pools.items()}}
(HERE/'semantic_mc_independent_review_summary.json').write_text(json.dumps(out,indent=2)+'\n')
with (HERE/'semantic_mc_independent_review_samples.jsonl').open('w') as f:
 for s in selected.values():f.write(json.dumps(s,ensure_ascii=False)+'\n')
print(json.dumps(out,indent=2))
