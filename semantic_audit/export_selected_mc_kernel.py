import bisect, collections, json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def read(name):
    for line in open(ROOT/name):
        if line.strip(): yield json.loads(line)
selected=set()
for r in read('semantic_inputs.jsonl'):
    selected.update((r['source_module'],c) for c in r['selected_mc_command_ids'])
by_module=collections.defaultdict(list); commands={}
for c in read('semantic_source_commands.jsonl'):
    by_module[c['source_module']].append(c); commands[(c['source_module'],c['command_id'])]=c
starts={}
for m,cs in by_module.items():
    cs.sort(key=lambda c:(c['start_line'],c['start_col']))
    starts[m]=[(c['start_line'],c['start_col']) for c in cs]
result=collections.defaultdict(list)
for r in read('declaration_dependencies.jsonl'):
    m=r['module']; p=r['selection_range'] or r['range']
    if m not in starts or not p: continue
    pos=(p['start_line'],p['start_column']); idx=bisect.bisect_right(starts[m],pos)-1
    if idx<0:continue
    c=by_module[m][idx];key=(m,c['command_id'])
    if key not in selected or not (pos<(c['end_line'],c['end_col'])):continue
    result[key].append({k:r[k] for k in ['name','kind','has_value','type_constants','type_projections','range','selection_range']})
with open(ROOT/'selected_mc_kernel_declarations.jsonl','w') as f:
    for key in sorted(selected):
        c=commands[key]
        r={'source_module':key[0],'source_file':c['source_file'],'command_id':key[1],'parser_kind':c['parser_kind'],'effective_kind':c['effective_kind'],'parser_name':c['parser_name'],'start_line':c['start_line'],'end_line':c['end_line'],'declarations':result[key]}
        f.write(json.dumps(r,ensure_ascii=False)+'\n')
print('selected commands',len(selected),'with kernel range matches',sum(bool(result[k]) for k in selected))
