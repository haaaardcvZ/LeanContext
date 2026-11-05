import Lean
open Lean

unsafe def main (args : List String) : IO UInt32 := do
  initSearchPath (← findSysroot)
  enableInitializersExecution
  let env ← importModules #[{module := `Mathlib}] {} 0
  let output : String := args[0]?.getD "declaration_ownership.jsonl"
  let file ← IO.FS.Handle.mk output .write
  let mut count := 0
  for (n, info) in env.constants do
    let some idx := env.getModuleIdxFor? n | continue
    let some modName := env.header.moduleNames[idx.toNat]? | continue
    unless modName.toString.startsWith "Mathlib." do continue
    let (owners, basis) := match info with
      | .ctorInfo v => (#[v.induct], "kernel_constructor_inductive")
      | .recInfo v => (v.all.toArray, "kernel_recursor_inductives")
      | _ => match env.getProjectionStructureName? n with
        | some owner => (#[owner], "registered_structure_projection")
        | none => if isAuxRecursor env n || isNoConfusion env n then
                    (#[n.getPrefix], "registered_aux_recursor_or_no_confusion")
                  else (#[], "none")
    if !owners.isEmpty then
      file.putStrLn <| (Json.mkObj [("name", toJson n), ("module", toJson modName),
        ("owners", toJson owners), ("basis", toJson basis)]).compress
      count := count + 1
  file.flush
  (← IO.getStderr).putStrLn s!"Exported {count} compiler-certified ownership relations."
  return 0
