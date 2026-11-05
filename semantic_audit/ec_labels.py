"""Conservative, evidence-carrying static links in LeanContext records.

This is a lightweight lexical analysis, NOT Lean elaboration or dependency
verification. Public booleans mean that an enumerated static link was observed.
False means no link was witnessed by these rules, never semantic independence.
"""
from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache
from typing import Any

VERSION = "1.6.0"
IDENT_RE = re.compile(r"«[^»]*»|[^\W\d][\w'’!?]*(?:\.(?:«[^»]*»|[^\W\d][\w'’!?]*))*", re.UNICODE)
DECL_RE = re.compile(r"\b(theorem|lemma|axiom|def|abbrev|opaque|constant|structure|class|inductive|instance|alias|example)\b")
SYNTAX_RE = re.compile(r"\b(notation3?|infixl?|infixr|prefix|postfix|macro|syntax)\b")
KW = {"Type", "Sort", "Prop", "by", "fun", "forall", "let", "have", "show", "from", "where", "in", "outParam", "semiOutParam", "extends", "theorem", "lemma", "def", "instance"}
OPEN = {"(": ")", "[": "]", "{": "}", "⦃": "⦄"}
CLOSE = set(OPEN.values())


@lru_cache(maxsize=30000)
def mask_noncode(text: str, mask_strings: bool = True) -> str:
    """Blank nested comments and string literals without moving offsets."""
    out = list(text)
    i, n = 0, len(text)
    while i < n:
        if text.startswith("--", i):
            j = text.find("\n", i)
            if j < 0: j = n
            for k in range(i, j): out[k] = " "
            i = j
        elif text.startswith("/-", i):
            j, depth = i + 2, 1
            while j < n and depth:
                if text.startswith("/-", j): depth += 1; j += 2
                elif text.startswith("-/", j): depth -= 1; j += 2
                else: j += 1
            for k in range(i, j):
                if out[k] != "\n": out[k] = " "
            i = j
        elif mask_strings and text[i] == "'" and (i == 0 or not (text[i - 1].isalnum() or text[i - 1] in "_'’")):
            cm = re.match(r"'(?:\\(?:u\{[0-9a-fA-F]+\}|u[0-9a-fA-F]{4}|x[0-9a-fA-F]{2}|.)|[^'\\\n])'", text[i:])
            if cm:
                for k in range(i, i + len(cm.group())): out[k] = " "
                i += len(cm.group())
            else: i += 1
        elif mask_strings and text[i] == '"':
            j = i + 1
            while j < n:
                if text[j] == "\\": j += 2
                elif text[j] == '"': j += 1; break
                else: j += 1
            for k in range(i, min(j, n)):
                if out[k] != "\n": out[k] = " "
            i = j
        else:
            i += 1
    return "".join(out)


def ids(code: str):
    return list(IDENT_RE.finditer(code))


def clean_name(name: str) -> str:
    return name[1:-1] if name.startswith("«") and name.endswith("»") else name


def grouped_spans(code: str):
    """All balanced (), [], {}, and implicit-brace groups, innermost first."""
    stack, result = [], []
    for i, c in enumerate(code):
        if c in OPEN: stack.append((i, c))
        elif c in CLOSE:
            if stack and OPEN[stack[-1][1]] == c:
                start, opening = stack.pop(); result.append((start, i + 1, opening))
            else:
                stack.clear()
    return result


def top_level_colon(code: str):
    stack = []
    for i, c in enumerate(code):
        if c in OPEN: stack.append(c)
        elif c in CLOSE:
            if stack and OPEN[stack[-1]] == c: stack.pop()
        elif c == ":" and not stack and not code.startswith(":=", i): return i
    return None


def declaration_parts(text: str):
    """Return declaration kind/name plus signature/body with original offsets.

    A declaration boundary must follow start, a newline, or wrapper `in` after
    allowing attributes/modifiers. The implementation intentionally does not
    pretend to parse syntax quotations or arbitrary macros.
    """
    code = mask_noncode(text)
    # Attribute keywords such as @[class] are not declaration keywords.
    chars = list(code)
    attr_start = 0
    while True:
        attr_start = code.find("@[", attr_start)
        if attr_start < 0: break
        attr_end, depth = attr_start + 2, 1
        while attr_end < len(code) and depth:
            if code[attr_end] == "[": depth += 1
            elif code[attr_end] == "]": depth -= 1
            attr_end += 1
        for index in range(attr_start, attr_end):
            if chars[index] != "\n": chars[index] = " "
        attr_start = attr_end
    code = "".join(chars)
    m = DECL_RE.search(code)
    if not m:
        return {"kind": None, "name": None, "signature": code, "signature_start": 0,
                "body": "", "body_start": len(code), "parsed": False}
    kind = m.group(1)
    after = m.end()
    nm = IDENT_RE.match(code, len(code[:after]) + len(code[after:]) - len(code[after:].lstrip()))
    name = None
    if kind != "example" and nm and nm.group() not in KW:
        name = clean_name(nm.group()); after = nm.end()
    # Explicit declaration universe binders .{u,v} are retained for shadowing.
    stack = []
    cut, body_start = len(code), len(code)
    i = after
    while i < len(code):
        c = code[i]
        if c in OPEN: stack.append(c)
        elif c in CLOSE:
            if stack and OPEN[stack[-1]] == c: stack.pop()
        if not stack:
            if code.startswith(":=", i): cut, body_start = i, i + 2; break
            wm = re.match(r"\bwhere\b", code[i:])
            if wm: cut, body_start = i, i + len(wm.group()); break
            # Equation compiler body starts with a top-level pipe on a new line.
            if c == "|" and code[after:i].rsplit("\n", 1)[-1].strip() == "":
                cut, body_start = i, i; break
        i += 1
    return {"kind": kind, "name": name, "signature": code[after:cut],
            "signature_start": after, "body": code[body_start:], "body_start": body_start,
            "parsed": kind != "alias"}


def binder_fragment_names(fragment: str) -> set[str]:
    """Exclude binder names, not annotated types (including typed lambdas)."""
    result, outside = set(), []
    i = 0
    while i < len(fragment):
        ch = fragment[i]
        if ch == ":": break
        if ch in OPEN:
            stack = [OPEN[ch]]
            j = i + 1
            while j < len(fragment) and stack:
                if fragment[j] in OPEN: stack.append(OPEN[fragment[j]])
                elif fragment[j] == stack[-1]: stack.pop()
                j += 1
            inner = fragment[i + 1:j - 1] if not stack else fragment[i + 1:]
            if ch != "[" or ":" in inner: result.update(binder_fragment_names(inner))
            outside.append(" "); i = j
        else:
            outside.append(ch); i += 1
    result.update(clean_name(t.group()) for t in ids("".join(outside)) if t.group() != "_" and t.group() not in KW and "." not in t.group())
    return result


def explicit_bound_names(code: str) -> set[str]:
    """Conservatively exclude identifiers introduced anywhere in a fragment.

    This deliberately over-approximates shadowing. Local names are removed
    across the entire fragment rather than claiming fully parsed Lean scopes.
    It can miss real links, but avoids common false links from target binders.
    """
    result = set()
    for start, end, opening in grouped_spans(code):
        inside = code[start + 1:end - 1]
        colon = top_level_colon(inside)
        if colon is not None:
            lhs = inside[:colon].strip()
            # A binder lhs is names only, not an application or arbitrary term.
            terms = ids(lhs)
            remainder = IDENT_RE.sub("", lhs).strip()
            if terms and not remainder and all("." not in t.group() for t in terms):
                result.update(clean_name(t.group()) for t in terms if t.group() not in KW)
    # Unparenthesized quantified binders: `∀ x y, ...`, `fun x => ...`, `∃ x, ...`.
    for m in re.finditer(r"(?:∀|∃!|∃|\bforall\b|\bfun\b)\s*([^,⇒]*?)(?:=>|↦|,)", code):
        head = m.group(1)
        # Only initial untyped name sequence. Typed/grouped binders handled above.
        head = head.split(":", 1)[0]
        # Bounded quantifiers: `∀ x ∈ S, ...`; only names before the relation.
        head = re.split(r"[∈∉≤≥<>]", head, maxsplit=1)[0]
        # Untyped binder groups and destructuring patterns are conservatively
        # treated as binders throughout the fragment, including ⟨x, hx⟩.
        stripped = re.sub(r"[()\[\]{}⦃⦄⟨⟩,]", " ", head)
        if not IDENT_RE.sub("", stripped).strip():
            result.update(clean_name(t.group()) for t in ids(stripped) if t.group() not in KW)
    # Pattern binders in lambdas/match alternatives and tactic case splits.
    for m in re.finditer(r"(?:\bfun\s+|(?<![|])\|\s*)([^\n]*?)(?:=>|↦)", code):
        result.update(binder_fragment_names(m.group(1)))
    for m in re.finditer(r"\b(?:rcases|obtain|cases|cases'|rcases'|induction|induction')\b[^\n]*?\bwith\s+([^\n;]+)", code):
        result.update(binder_fragment_names(m.group(1)))
    for m in re.finditer(r"\b(?:letI?|have|suffices|obtain)\s+([\w'’]+)\s*(?::|:=)", code):
        result.add(m.group(1))
    # Tactic-introduced variables are not parsed; conservatively exclude simple names.
    for m in re.finditer(r"\b(?:intro|intros|rintro)\s+([^\n;]+)", code):
        result.update(binder_fragment_names(m.group(1)))
    um = re.match(r"\s*\.\{([^}]+)\}", code)
    if um: result.update(clean_name(t.group()) for t in ids(um.group(1)))
    return result


@lru_cache(maxsize=30000)
def parse_fragment(text: str, as_declaration: bool = True):
    part = declaration_parts(text) if as_declaration else {
        "kind": None, "name": None, "signature": mask_noncode(text), "signature_start": 0,
        "body": "", "body_start": len(text), "parsed": True}
    for field in ("signature", "body"):
        code = part[field]
        bound = explicit_bound_names(code)
        if field == "body": bound |= explicit_bound_names(part["signature"])
        part[field + "_bound_names"] = tuple(sorted(bound))
        part[field + "_identifiers"] = tuple(
            (clean_name(m.group()), m.start() + part[field + "_start"] + (1 if m.group().startswith("«") and m.group().endswith("»") else 0))
            for m in ids(code) if clean_name(m.group()) not in bound
            and not code[:m.start()].rstrip().endswith(".")
            and not (m.start() and code[m.start() - 1] == "?")
            and not code[m.end():].lstrip().startswith(":=")
            and not (not code[:m.start()].rsplit("\n", 1)[-1].strip() and code[m.end():].lstrip().startswith(":"))
        )
    return part


def parse_variable_command(text: str):
    code = mask_noncode(text)
    m = re.search(r"\bvariables?\b", code)
    if not m: return [], [], []
    payload = code[m.end():]
    # `variable ... in decl` is local to the wrapped command, never persistent.
    wrapped = bool(re.search(r"\bin\b", payload))
    bindings, adjustments, anonymous = [], [], []
    for start, end, opening in grouped_spans(payload):
        # Nested binder groups inside types are not section-variable commands.
        if any(s < start and e > end for s, e, _ in grouped_spans(payload)): continue
        inside = payload[start + 1:end - 1]
        colon = top_level_colon(inside)
        if colon is not None:
            lhs = inside[:colon].strip()
            terms = ids(lhs)
            if terms and not IDENT_RE.sub("", lhs).strip():
                for t in terms:
                    if "." not in t.group() and t.group() not in KW:
                        bindings.append({"symbol": clean_name(t.group()), "offset": m.end() + start + 1 + (len(inside[:colon]) - len(inside[:colon].lstrip())) + t.start() + (1 if t.group().startswith("«") and t.group().endswith("»") else 0),
                                         "binder_kind": opening, "wrapped": wrapped,
                                         "type_text": inside[colon + 1:].strip()})
        elif opening == "[":
            anonymous.append(inside.strip())
        elif not IDENT_RE.sub("", inside).strip():
            adjustments.extend(clean_name(t.group()) for t in ids(inside))
    return bindings, adjustments, anonymous


def syntax_literals(text: str):
    """Literal tokens in a syntax/notation LHS; quoted RHS is excluded."""
    code = mask_noncode(text, False)
    sm = SYNTAX_RE.search(code)
    if not sm: return []
    # The first => can occur in notation3 binder elaborators. This intentionally
    # retains only literals seen before it rather than guessing the full parser.
    lhs = code[sm.end():].split("=>", 1)[0]
    vals = []
    for m in re.finditer(r'"((?:\\.|[^"\\])*)"', lhs):
        val = m.group(1).replace('\\"', '"').replace('\\\\', '\\').strip()
        if val and val not in {"(", ")", "[", "]", "{", "}", ",", ":", ";", "|", "_", ".", "=", "+", "-", "*", "/", "<", ">", "≤", "≥", "∈", "∉", "∧", "∨", "¬", "∀", "∃", "→", "↔"}:
            vals.append((val, sm.end() + m.start()))
    return vals


def literal_occurs(literal: str, code: str):
    """Report exact literal spans, with identifier boundaries where applicable."""
    matches = []
    for m in re.finditer(re.escape(literal), code):
        a, b = m.span()
        if literal[0].isalnum() or literal[0] == "_":
            if a and (code[a - 1].isalnum() or code[a - 1] in "_'’"): continue
        if literal[-1].isalnum() or literal[-1] in "_'’":
            if b < len(code) and (code[b].isalnum() or code[b] in "_'’"): continue
        matches.append(a)
    return matches


def analyze_record(record: dict[str, Any]) -> dict[str, Any]:
    mc = record.get("mathematical_context", [])
    ec = record.get("environment_context", [])
    merged = sorted([(int(x["position"]), "mc", x) for x in mc] +
                    [(int(x["position"]), "ec", x) for x in ec])
    variables, universes, syntaxes = {}, {}, []
    scopes, namespace = [], []
    opened_scopes = set()
    evidence = {"mc_signature": [], "target_statement": [], "mc_body": [], "reference_proof": []}
    unsupported, adjustments, anonymous_instances, ec_mc_refs = [], [], [], []
    mc_decls = []
    counters = Counter()

    def warn(reason, position=None):
        entry = {"reason": reason}
        if position is not None: entry["ec_position"] = position
        if entry not in unsupported: unsupported.append(entry)

    def inspect(fragment, target_key, receiver_position=None, as_decl=True, field="signature"):
        parsed = parse_fragment(fragment, as_decl)
        if not parsed["parsed"]:
            warn("receiver_declaration_not_parsed", receiver_position)
            return parsed
        free = parsed[field + "_identifiers"]
        for symbol, offset in free:
            source = variables.get(symbol)
            mechanism = "ec_bound_identifier_occurrence"
            witness_symbol = symbol
            if source is None:
                source = universes.get(symbol); mechanism = "ec_universe_identifier_occurrence"
            if source is None and "." in symbol and not (offset and fragment[offset - 1] == "«"):
                prefix = symbol.split(".", 1)[0]
                if prefix not in parsed[field + "_bound_names"]:
                    source = variables.get(prefix)
                    if source is not None:
                        witness_symbol = prefix
                        mechanism = "ec_bound_receiver_prefix_occurrence"
            if source is not None:
                ev = {"mechanism": mechanism, "symbol": witness_symbol,
                    "ec_position": source["position"], "ec_offset": source["offset"],
                    "receiver_position": receiver_position, "receiver_offset": offset,
                    "scope_basis": "selected_command_order", "semantic_status": "unverified"}
                if mechanism == "ec_bound_receiver_prefix_occurrence": ev["receiver_identifier"] = symbol
                evidence[target_key].append(ev)
        fragment_code = parsed[field]
        for sy in syntaxes:
            current_namespace = ".".join(namespace)
            active = sy["scope"] is None or sy["scope"] in opened_scopes or current_namespace == sy["scope"] or current_namespace.startswith(sy["scope"] + ".")
            if not active: continue
            for off in literal_occurs(sy["literal"], fragment_code):
                evidence[target_key].append({"mechanism": "notation_or_syntax_literal_occurrence",
                    "symbol": sy["literal"], "ec_position": sy["position"], "ec_offset": sy["offset"],
                    "receiver_position": receiver_position, "receiver_offset": off + parsed[field + "_start"],
                    "scope_basis": "selected_command_order", "semantic_status": "unverified"})
        return parsed

    for pos, channel, item in merged:
        text = str(item.get("text", "")); kind = str(item.get("kind", "unknown"))
        code = mask_noncode(text)
        if channel == "mc":
            parsed = inspect(text, "mc_signature", pos)
            inspect(text, "mc_body", pos, field="body")
            if parsed.get("name"):
                mc_decls.append({"position": pos, "name": parsed["name"],
                    "qualified_candidate": ".".join(namespace + [parsed["name"]])})
            continue
        counters[kind] += 1
        if re.match(r"\s*(namespace|section)\b", code):
            opening = re.match(r"\s*(namespace|section)\b\s*([^\s]*)", code)
            name = opening.group(2)
            # `section ... in ...` is not treated as a persistent scope.
            if re.search(r"\bin\b", code): warn("wrapped_scope_command", pos); continue
            scopes.append((name, dict(variables), dict(universes), list(syntaxes), list(namespace), set(opened_scopes)))
            if opening.group(1) == "namespace":
                namespace.append(name); warn("namespace_resolution_not_elaborated", pos)
            continue
        if re.match(r"\s*end\b", code):
            if scopes:
                _, variables, universes, syntaxes, namespace, opened_scopes = scopes.pop()
            else: warn("unmatched_scope_end", pos)
            continue
        if re.search(r"\bvariables?\b", code) and kind == "variable":
            binds, adj, anon = parse_variable_command(text)
            for b in binds:
                if b["wrapped"]: warn("wrapped_variable_command", pos); continue
                variables[b["symbol"]] = {**b, "position": pos}
            adjustments.extend({"ec_position": pos, "symbol": a, "known_previous_binding": a in variables} for a in adj)
            anonymous_instances.extend({"ec_position": pos, "type_text": a} for a in anon)
            if anon: warn("anonymous_typeclass_instance_requires_elaboration", pos)
            if not (binds or adj or anon): warn("variable_command_not_parsed", pos)
            continue
        if kind == "universe" or re.match(r"\s*universes?\b", code):
            m = re.search(r"\buniverses?\b", code)
            for t in ids(code[m.end():]):
                universes[clean_name(t.group())] = {"position": pos, "offset": m.end() + t.start() + (1 if t.group().startswith("«") and t.group().endswith("»") else 0)}
            continue
        sm = SYNTAX_RE.search(code)
        if sm:
            literals = syntax_literals(text)
            scoped = bool(re.search(r"\bscoped\b", code[:sm.start()]))
            explicit_scope = re.search(r"\bscoped\s*\[\s*([^]\s]+)\s*\]", code[:sm.start()])
            scope = explicit_scope.group(1) if explicit_scope else (".".join(namespace) if scoped else None)
            if scoped and not scope:
                scope = "<unknown-scoped-namespace>"; warn("scoped_syntax_namespace_unknown", pos)
            for literal, offset in literals:
                syntaxes.append({"position": pos, "offset": offset, "literal": literal, "scope": scope})
            if not literals: warn("syntax_no_discriminative_literal", pos)
            warn("notation_syntax_resolution_not_elaborated", pos)
            continue
        if kind == "open_scoped" or re.match(r"\s*open\s+scoped\b", code):
            m = re.search(r"\bscoped\b", code)
            if m and not re.search(r"\bin\b", code[m.end():]):
                opened_scopes.update(clean_name(t.group()) for t in ids(code[m.end():]))
            warn("opened_scope_effect_requires_elaboration", pos)
            continue
        # Record reverse textual relations, e.g. EC attributes on MC declarations.
        if kind in {"attribute", "export", "include", "omit", "scoped"}:
            itokens = {clean_name(t.group()) for t in ids(code)}
            for decl in mc_decls:
                matches = sorted({n for n in (decl["name"], decl["qualified_candidate"]) if n in itokens or any(t.startswith(n + ".") for t in itokens)})
                if matches:
                    ec_mc_refs.append({"ec_position": pos, "mc_position": decl["position"],
                                       "mechanism": "ec_command_mentions_mc_name", "symbols": matches})
        if DECL_RE.search(code): warn("declaration_nested_in_environment_command", pos)
        warn({"open": "open_namespace_resolution_not_elaborated", "attribute": "attribute_effect_requires_elaboration",
              "include": "include_effect_requires_elaboration", "omit": "omit_effect_requires_elaboration",
              "set_option": "option_effect_requires_elaboration", "export": "export_resolution_not_elaborated"}.get(kind, "environment_command_not_elaborated"), pos)

    target = str(record.get("statement", ""))
    inspect(target, "target_statement")
    # The reference proof inherits explicit binders of the target declaration.
    target_bound = set(parse_fragment(target)["signature_bound_names"])
    saved_variables, saved_universes = variables, universes
    variables = {k: v for k, v in variables.items() if k not in target_bound}
    universes = {k: v for k, v in universes.items() if k not in target_bound}
    inspect(str(record.get("target", "")), "reference_proof", as_decl=False)
    variables, universes = saved_variables, saved_universes

    # Deduplicate evidence of identical (source, receiver, mechanism, symbol),
    # preserving the first exact receiver offset as an auditable witness.
    for k, vals in evidence.items():
        unique = {}
        for e in vals:
            key = (e["ec_position"], e["receiver_position"], e["mechanism"], e["symbol"])
            unique.setdefault(key, e)
        evidence[k] = list(unique.values())
    mechanisms = {k: dict(Counter(e["mechanism"] for e in v)) for k, v in evidence.items()}
    return {
        "schema_version": VERSION,
        "method": "conservative_static_lexical_witnesses_not_lean_verification",
        "ec_to_mc_signature_observed": bool(evidence["mc_signature"]),
        "ec_to_target_observed": bool(evidence["target_statement"]),
        "ec_to_mc_body_observed": bool(evidence["mc_body"]),
        "ec_to_reference_proof_observed": bool(evidence["reference_proof"]),
        "ec_to_mc_signature_status": "observed_static_link" if evidence["mc_signature"] else "no_observed_static_link",
        "ec_to_target_status": "observed_static_link" if evidence["target_statement"] else "no_observed_static_link",
        "semantic_dependency_status": "unverified",
        "unknown_or_unsupported": bool(unsupported),
        "unsupported_reasons": unsupported,
        "evidence": evidence,
        "mechanism_counts": mechanisms,
        "ec_mentions_mc_declaration": ec_mc_refs,
        "variable_visibility_adjustments": adjustments,
        "anonymous_instance_binders": anonymous_instances,
        "scope_basis": "selected_command_order_not_source_scope_recovery",
        "ec_kind_counts": dict(counters),
    }


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source")
    ap.add_argument("output")
    ns = ap.parse_args()
    with open(ns.output, "w") as out:
        for line in open(ns.source):
            r = json.loads(line)
            if r.get("mathematical_context") and r.get("environment_context"):
                out.write(json.dumps({"sample_id": r["sample_id"], "split": r.get("split"), **analyze_record(r)}, ensure_ascii=False) + "\n")
