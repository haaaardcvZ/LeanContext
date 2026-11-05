import Lean
import Lean.Util.FoldConsts
import Lean.Util.ForEachExpr

open Lean

/- Dump actual kernel expressions from a pinned, compiled environment.
   No model generation or theorem rewriting is performed. Every constant and
   every projection occurrence is recorded separately for type and value.
   Run: lake env lean --run /absolute/DependencyDump.lean Mathlib output.jsonl Mathlib
   The final prefix restricts declaration modules, not dependencies. -/

def declKind : ConstantInfo → String
  | .axiomInfo _ => "axiom"
  | .defnInfo _ => "definition"
  | .thmInfo _ => "theorem"
  | .opaqueInfo _ => "opaque"
  | .quotInfo _ => "quotient"
  | .inductInfo _ => "inductive"
  | .ctorInfo _ => "constructor"
  | .recInfo _ => "recursor"

def actualValue? : ConstantInfo → Option Expr
  | .opaqueInfo val => some val.value
  | val => val.value?

def rangeJson (r : DeclarationRange) : Json := Json.mkObj [
  ("start_line", toJson r.pos.line), ("start_column", toJson r.pos.column),
  ("end_line", toJson r.endPos.line), ("end_column", toJson r.endPos.column),
  ("start_utf16", toJson r.charUtf16), ("end_utf16", toJson r.endCharUtf16)]

def projectionRefs (env : Environment) (expr : Expr) : IO (Array Json) := do
  let found ← IO.mkRef ({} : NameSet)
  let rows ← IO.mkRef (#[] : Array Json)
  expr.forEach fun e => do
    if let .proj structName index _ := e then
      let key := Name.num structName index
      unless (← found.get).contains key do
        found.modify (·.insert key)
        let projName := (getStructureInfo? env structName).bind (·.getProjFn? index)
        rows.modify (·.push <| Json.mkObj [
          ("structure", toJson structName), ("index", toJson index),
          ("projection_name", toJson projName)])
  rows.get

def constantRow (env : Environment) (n : Name) (info : ConstantInfo) (moduleName : Name) : IO Json := do
  let typeNames := info.type.getUsedConstants
  let typeSet := info.type.getUsedConstantsAsSet
  let value := actualValue? info
  let valueNames := value.map (·.getUsedConstants) |>.getD #[]
  let valueOnly := valueNames.filter fun c => !typeSet.contains c
  let ranges := declRangeExt.find? env n
  let typeProjections ← projectionRefs env info.type
  let valueProjections ← match value with
    | some e => projectionRefs env e
    | none => pure #[]
  return Json.mkObj [
    ("name", toJson n), ("module", toJson moduleName), ("kind", toJson (declKind info)),
    ("range", ranges.map (rangeJson ∘ DeclarationRanges.range) |>.getD Json.null),
    ("selection_range", ranges.map (rangeJson ∘ DeclarationRanges.selectionRange) |>.getD Json.null),
    ("has_value", toJson value.isSome),
    ("type_constants", toJson typeNames), ("value_constants", toJson valueNames),
    ("value_minus_type_constants", toJson valueOnly),
    ("type_projections", toJson typeProjections), ("value_projections", toJson valueProjections)]

unsafe def main (args : List String) : IO UInt32 := do
  let moduleText := args[0]?.getD "Mathlib"
  let output := args[1]?.getD "declaration_dependencies.jsonl"
  let modulePrefix := args[2]?.getD moduleText
  initSearchPath (← findSysroot)
  enableInitializersExecution
  let stderr ← IO.getStderr
  stderr.putStrLn s!"Importing compiled module {moduleText}..."
  let env ← importModules #[{module := moduleText.toName}] {} 0
  stderr.putStrLn "Import complete; reading kernel declaration values."
  let file ← IO.FS.Handle.mk output .write
  let mut count := 0
  for (n, info) in env.constants do
    let some idx := env.getModuleIdxFor? n | continue
    let some modName := env.header.moduleNames[idx.toNat]? | continue
    unless modulePrefix.isEmpty || modName.toString == modulePrefix || modName.toString.startsWith (modulePrefix ++ ".") do continue
    let row ← constantRow env n info modName
    file.putStrLn row.compress
    count := count + 1
    if count % 10000 == 0 then stderr.putStrLn s!"Exported {count} declarations."
  file.flush
  stderr.putStrLn s!"Complete: {count} declarations to {output}"
  return 0
