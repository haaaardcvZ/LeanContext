"""Recover source spelling candidates under original namespace scopes.

These are candidate names only. A caller must validate them against the Lean
environment and declaration range/type; private and anonymous names are omitted.
"""
import argparse, collections, json, re
from pathlib import Path
from mc_labels import declaration_names, mask_comments_strings
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--audit-root', type=Path, required=True,
                    help='Directory containing semantic_input_manifest.json')
parser.add_argument('--parser-root', type=Path, required=True,
                    help='Command-span root containing Mathlib/**/*.jsonl')
args = parser.parse_args()
HERE=args.audit_root.resolve()
def rows(p):
 with p.open() as f:
  for l in f:
   if l.strip():yield json.loads(l)
manifest=json.loads((HERE/'semantic_input_manifest.json').read_text())
counts=collections.Counter();issues=[]
def name_parts(name):
 return re.findall(r'«[^»]*»|[^.]+',name) if name else [None]
with (HERE/'source_canonical_name_candidates.jsonl').open('w') as out:
 for file in manifest['files']:
  stack=[];namespace='';scope_ok=True
  for c in rows(args.parser_root / Path(file['source_file']).with_suffix('.jsonl')):
   text=c['effective_raw_text'];kind=c['kind'];nm=c.get('name')
   wrapped_scope=re.search(r'\bin\s+(namespace|section)\s*(\S+)?\s*$',text)
   if kind not in ('namespace','section','end') and wrapped_scope:
    kind=wrapped_scope.group(1);nm=wrapped_scope.group(2)
   before=namespace
   if kind in ('namespace','section'):
    # Lean opens one scope per dotted component; e.g. namespace A.B may
    # later be closed by end B followed by end A, or one end A.B.
    for component in name_parts(nm):
     stack.append({'kind':kind,'name':component,'previous_namespace':namespace,'command_id':c['command_id']})
     if kind=='namespace':
      if not component:
       scope_ok=False;issues.append({'source_file':file['source_file'],'command_id':c['command_id'],'issue':'namespace_without_name'})
      elif component=='_root_':namespace=''
      else:namespace='.'.join(x for x in (namespace,component) if x)
   elif kind=='end':
    for component in reversed(name_parts(nm)):
     if not stack:
      scope_ok=False;issues.append({'source_file':file['source_file'],'command_id':c['command_id'],'issue':'end_without_scope','name':component})
     else:
      frame=stack.pop()
      if component and component!=frame['name']:
       scope_ok=False;issues.append({'source_file':file['source_file'],'command_id':c['command_id'],'issue':'end_name_mismatch','name':component,'frame':frame})
      namespace=frame['previous_namespace']
   else:
    parsed=declaration_names({'kind':kind,'text':text})
    if parsed['parse_status']=='unsupported':continue
    # A private source spelling is not its hidden kernel name. Keep the
    # spelling and private flag as evidence, without a public-name candidate.
    prefix=mask_comments_strings(text)
    is_private=bool(re.search(r'(?<![\w.])private\s+(?:(?:noncomputable|nonrec|protected|unsafe|partial)\s+)*(?:theorem|lemma|def|abbrev|opaque|instance|structure|class|inductive)\b',prefix))
    names=[]
    if scope_ok and not is_private:
     for name in parsed['names']:
      names.append(name[len('_root_.'):] if name.startswith('_root_.') else '.'.join(x for x in (namespace,name) if x))
     # Lean's environment name strings omit source escapes for ordinary
     # reserved-word components, such as CharP.\u00abexists\u00bb or \u00abProp\u00bb.isAtom.
     names=list(dict.fromkeys(n for name in names for n in [name,re.sub(r'«([^».]+)»',r'\1',name)]))
    row={'source_file':file['source_file'],'source_module':c['source_module'],'command_id':c['command_id'],'parser_kind':kind,'active_namespace':before,'explicit_declaration_names':parsed['names'],'canonical_name_candidates':names,'parse_status':parsed['parse_status'],'is_private':is_private,'scope_stack_valid':scope_ok,'basis':'complete_original_source_namespace_section_end_stack; candidate_only_requires_Lean_validation','start_line':c['start_line'],'end_line':c['end_line']}
    out.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
    counts['declaration_records']+=1;counts['with_candidates']+=bool(names);counts['private']+=is_private;counts[parsed['parse_status']]+=1
report={'counts':dict(counts),'issues':issues,'source_files':len(manifest['files'])}
(HERE/'source_canonical_name_candidate_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
