import Lean
import Mathlib
import Lean.Elab.Import
import Lean.Elab.Frontend

open Lean
open Lean.Parser
open Lean.Elab

def jsonEscape (s : String) : String :=
  let rec loop (cs : List Char) (acc : String) : String :=
    match cs with
    | [] => acc
    | c :: rest =>
      let piece :=
        match c with
        | '"'  => "\\\""
        | '\\' => "\\\\"
        | '\n' => "\\n"
        | '\r' => "\\r"
        | '\t' => "\\t"
        | c    => String.singleton c
      loop rest (acc ++ piece)
  loop s.toList ""

def jsonStr (s : String) : String :=
  "\"" ++ jsonEscape s ++ "\""

def posToNat (p : String.Pos) : Nat :=
  p.byteIdx

def substringAt (s : String) (start stop : String.Pos) : String :=
  s.extract start stop

def isWs (c : Char) : Bool :=
  c == ' ' || c == '\n' || c == '\t' || c == '\r'

def isTokenDelimiter (c : Char) : Bool :=
  c == '(' || c == ')' ||
  c == '{' || c == '}' ||
  c == '[' || c == ']' ||
  c == ':' || c == '=' ||
  c == ',' || c == ';' ||
  c == '|'

partial def skipWsChars : List Char → List Char
  | [] => []
  | c :: rest =>
      if isWs c then skipWsChars rest else c :: rest

def rstripString (s : String) : String :=
  String.mk ((skipWsChars s.toList.reverse).reverse)

partial def skipStringChars : List Char → List Char
  | [] => []
  | '\\' :: _ :: rest => skipStringChars rest
  | '"' :: rest => rest
  | _ :: rest => skipStringChars rest

partial def skipBalancedLoop (openCh closeCh : Char) : Nat → List Char → List Char
  | _, [] => []
  | depth, c :: rest =>
      if c == '"' then
        skipBalancedLoop openCh closeCh depth (skipStringChars rest)
      else if c == openCh then
        skipBalancedLoop openCh closeCh (depth + 1) rest
      else if c == closeCh then
        if depth == 1 then
          rest
        else
          skipBalancedLoop openCh closeCh (depth - 1) rest
      else
        skipBalancedLoop openCh closeCh depth rest

def skipBalanced (openCh closeCh : Char) : List Char → List Char
  | [] => []
  | c :: rest =>
      if c == openCh then
        skipBalancedLoop openCh closeCh 1 rest
      else
        c :: rest

partial def skipLineCommentChars : List Char → List Char
  | [] => []
  | '\n' :: rest => rest
  | _ :: rest => skipLineCommentChars rest

partial def skipBlockCommentLoop : Nat → List Char → List Char
  | _, [] => []
  | depth, '/' :: '-' :: rest =>
      skipBlockCommentLoop (depth + 1) rest
  | depth, '-' :: '/' :: rest =>
      if depth == 1 then
        rest
      else
        skipBlockCommentLoop (depth - 1) rest
  | depth, _ :: rest =>
      skipBlockCommentLoop depth rest

partial def skipLeadingCommentsAndWs : List Char → List Char
  | cs =>
      let cs := skipWsChars cs
      match cs with
      | '-' :: '-' :: rest => skipLeadingCommentsAndWs (skipLineCommentChars rest)
      | '/' :: '-' :: rest => skipLeadingCommentsAndWs (skipBlockCommentLoop 1 rest)
      | _ => cs

partial def skipLeadingAttributes : List Char → List Char
  | cs =>
      let cs := skipLeadingCommentsAndWs cs
      match cs with
      | '@' :: '[' :: rest =>
          skipLeadingAttributes (skipBalanced '[' ']' ('[' :: rest))
      | _ => cs

partial def readTokenAux : List Char → List Char → String × List Char
  | [], acc => (String.mk acc.reverse, [])
  | c :: rest, acc =>
      if isWs c || isTokenDelimiter c then
        (String.mk acc.reverse, c :: rest)
      else
        readTokenAux rest (c :: acc)

def readToken (cs : List Char) : String × List Char :=
  let cs := skipWsChars cs
  match cs with
  | [] => ("", [])
  | c :: rest =>
      if isTokenDelimiter c then
        (String.singleton c, rest)
      else
        readTokenAux cs []

def isModifier (tok : String) : Bool :=
  tok == "private" ||
  tok == "protected" ||
  tok == "noncomputable" ||
  tok == "unsafe" ||
  tok == "partial" ||
  tok == "nonrec"

partial def skipModifiers (cs : List Char) : List Char :=
  let cs0 := skipWsChars cs
  let (tok, rest) := readToken cs0
  if isModifier tok then
    skipModifiers rest
  else
    cs0

def isBadNameToken (tok : String) : Bool :=
  tok == "" ||
  tok == ":" ||
  tok == ":=" ||
  tok == "=" ||
  tok == "|" ||
  tok == "(" ||
  tok == "{" ||
  tok == "[" ||
  tok == "where" ||
  tok == "extends" ||
  tok == "with"

def cleanName? (tok : String) : Option String :=
  if isBadNameToken tok then none else some tok

def skipInstanceOptions (cs : List Char) : List Char :=
  let cs := skipWsChars cs
  match cs with
  | '(' :: rest => skipBalanced '(' ')' ('(' :: rest)
  | _ => cs

def parseAttributeTarget? (cs : List Char) : Option String :=
  let cs := skipWsChars cs
  let cs :=
    match cs with
    | '[' :: rest => skipBalanced '[' ']' ('[' :: rest)
    | _ => cs
  let (tok, _) := readToken cs
  cleanName? tok

def parseKindNameFromRaw (raw : String) (syntaxKind : String) : String × Option String :=
  let effectiveRaw := rstripString raw
  let cs0 := effectiveRaw.toList
  let cs1 := skipLeadingAttributes cs0
  let cs2 := skipModifiers cs1
  let (tok, rest) := readToken cs2

  if tok == "theorem" then
    let (nm, _) := readToken rest
    ("theorem", cleanName? nm)

  else if tok == "lemma" then
    let (nm, _) := readToken rest
    ("lemma", cleanName? nm)

  else if tok == "example" then
    ("example", none)

  else if tok == "def" then
    let (nm, _) := readToken rest
    ("def", cleanName? nm)

  else if tok == "abbrev" then
    let (nm, _) := readToken rest
    ("abbrev", cleanName? nm)

  else if tok == "class" then
    let (nm, _) := readToken rest
    ("class", cleanName? nm)

  else if tok == "structure" then
    let (nm, _) := readToken rest
    ("structure", cleanName? nm)

  else if tok == "inductive" then
    let (nm, _) := readToken rest
    ("inductive", cleanName? nm)

  else if tok == "axiom" then
    let (nm, _) := readToken rest
    ("axiom", cleanName? nm)

  else if tok == "constant" then
    let (nm, _) := readToken rest
    ("constant", cleanName? nm)

  else if tok == "opaque" then
    let (nm, _) := readToken rest
    ("opaque", cleanName? nm)

  else if tok == "instance" then
    let rest := skipInstanceOptions rest
    let (nm, _) := readToken rest
    ("instance", cleanName? nm)

  else if tok == "namespace" then
    let (nm, _) := readToken rest
    ("namespace", cleanName? nm)

  else if tok == "section" then
    let (nm, _) := readToken rest
    ("section", cleanName? nm)

  else if tok == "end" then
    let (nm, _) := readToken rest
    ("end", cleanName? nm)

  else if tok == "open" then
    let (tok2, rest2) := readToken rest
    if tok2 == "scoped" then
      let (nm, _) := readToken rest2
      ("open_scoped", cleanName? nm)
    else
      ("open", cleanName? tok2)

  else if tok == "universe" then
    let (nm, _) := readToken rest
    ("universe", cleanName? nm)

  else if tok == "variable" then
    ("variable", none)

  else if tok == "variables" then
    ("variable", none)

  else if tok == "include" then
    let (nm, _) := readToken rest
    ("include", cleanName? nm)

  else if tok == "omit" then
    let (nm, _) := readToken rest
    ("omit", cleanName? nm)

  else if tok == "set_option" then
    let (nm, _) := readToken rest
    ("set_option", cleanName? nm)

  else if tok == "attribute" then
    ("attribute", parseAttributeTarget? rest)

  else if tok == "local" then
    let (tok2, rest2) := readToken rest
    if tok2 == "attribute" then
      ("local_attribute", parseAttributeTarget? rest2)
    else if tok2 == "instance" then
      let (nm, _) := readToken rest2
      ("local_instance", cleanName? nm)
    else if tok2 == "notation" then
      ("local_notation", none)
    else
      ("local", cleanName? tok2)

  else if tok == "scoped" then
    let (tok2, rest2) := readToken rest
    if tok2 == "attribute" then
      ("scoped_attribute", parseAttributeTarget? rest2)
    else if tok2 == "notation" then
      ("scoped_notation", none)
    else
      ("scoped", cleanName? tok2)

  else if tok == "notation" then
    ("notation", none)

  else if tok == "infix" || tok == "infixl" || tok == "infixr" || tok == "prefix" || tok == "postfix" then
    (tok, none)

  else if tok == "syntax" then
    ("syntax", none)

  else if tok == "macro" then
    ("macro", none)

  else if tok == "macro_rules" then
    ("macro_rules", none)

  else if tok == "alias" then
    let (nm, _) := readToken rest
    ("alias", cleanName? nm)

  else if tok == "export" then
    let (nm, _) := readToken rest
    ("export", cleanName? nm)

  else if tok == "library_note" then
    ("library_note", none)

  else if tok == "assert_not_exists" then
    let (nm, _) := readToken rest
    ("assert_not_exists", cleanName? nm)

  else if tok == "initialize_simps_projections" then
    let (nm, _) := readToken rest
    ("initialize_simps_projections", cleanName? nm)

  else
    (syntaxKind, none)

def lineColOfPos (inputCtx : InputContext) (p : String.Pos) : Nat × Nat :=
  let pos := inputCtx.fileMap.toPosition p
  (pos.line, pos.column)

def pad6 (n : Nat) : String :=
  let s := toString n
  let rec loop (k : Nat) (acc : String) :=
    match k with
    | 0 => acc
    | k + 1 => loop k ("0" ++ acc)
  if s.length >= 6 then s else loop (6 - s.length) s

def jsonCommand
    (inputCtx : InputContext)
    (contents : String)
    (idx : Nat)
    (stx : Syntax)
    (startPos : String.Pos)
    (endPos : String.Pos) : String :=
  let (sl, sc) := lineColOfPos inputCtx startPos
  let (el, ec) := lineColOfPos inputCtx endPos
  let raw := substringAt contents startPos endPos
  let effectiveRaw := rstripString raw
  let syntaxKind := toString stx.getKind
  let parsed := parseKindNameFromRaw effectiveRaw syntaxKind
  let cls := parsed.fst
  let name? := parsed.snd

  "{"
  ++ "\"command_id\":" ++ jsonStr ("cmd_" ++ pad6 idx) ++ ","
  ++ "\"kind\":" ++ jsonStr cls ++ ","
  ++ "\"syntax_kind\":" ++ jsonStr syntaxKind ++ ","
  ++ "\"name\":" ++ (match name? with | some n => jsonStr n | none => "null") ++ ","
  ++ "\"start_byte\":" ++ toString (posToNat startPos) ++ ","
  ++ "\"end_byte\":" ++ toString (posToNat endPos) ++ ","
  ++ "\"start_line\":" ++ toString sl ++ ","
  ++ "\"start_col\":" ++ toString sc ++ ","
  ++ "\"end_line\":" ++ toString el ++ ","
  ++ "\"end_col\":" ++ toString ec ++ ","
  ++ "\"raw_text\":" ++ jsonStr raw ++ ","
  ++ "\"effective_raw_text\":" ++ jsonStr effectiveRaw
  ++ "}"

partial def parseCommandsLoop
    (inputCtx : InputContext)
    (contents : String)
    (env : Environment)
    (state0 : ModuleParserState)
    (messages0 : MessageLog)
    (idx0 : Nat) : IO Unit := do
  let rec loop (state : ModuleParserState) (messages : MessageLog) (idx : Nat) : IO Unit := do
    let startPos := state.pos
    let (stx, state', messages') := parseCommand inputCtx { env := env, options := {} } state messages
    let endPos := state'.pos
    if isTerminalCommand stx then
      return ()
    else
      IO.println <| jsonCommand inputCtx contents idx stx startPos endPos
      loop state' messages' (idx + 1)

  loop state0 messages0 idx0

def syntaxSpan? (stx : Syntax) : Option (String.Pos × String.Pos) :=
  match stx.getPos?, stx.getTailPos? with
  | some startPos, some endPos => some (startPos, endPos)
  | _, _ => none

def jsonCommandFromSyntax
    (inputCtx : InputContext)
    (contents : String)
    (idx : Nat)
    (stx : Syntax) : Option String :=
  match syntaxSpan? stx with
  | some (startPos, endPos) =>
      if isTerminalCommand stx then
        none
      else
        some <| jsonCommand inputCtx contents idx stx startPos endPos
  | none => none

def processCommandsWithFrontend
    (inputCtx : InputContext)
    (contents : String)
    (env : Environment)
    (stateAfterHeader : ModuleParserState)
    (messages : MessageLog) : IO Unit := do
  let commandState := Lean.Elab.Command.mkState env messages {}
  let frontendState ← Lean.Elab.IO.processCommands inputCtx stateAfterHeader commandState
  let mut idx := 1
  for stx in frontendState.commands do
    match jsonCommandFromSyntax inputCtx contents idx stx with
    | some line =>
        IO.println line
        idx := idx + 1
    | none => pure ()

def main (args : List String) : IO UInt32 := do
  let fileName :=
    match args with
    | f :: _ => f
    | [] => ""

  if fileName == "" then
    IO.eprintln "usage: lake env lean --run ExtractCommandSpans.lean <LeanFile.lean>"
    return (1 : UInt32)
  else
    let contents ← IO.FS.readFile fileName
    let inputCtx := mkInputContext contents fileName

    let (_header, stateAfterHeader, messages) ← parseHeader inputCtx

    let (importsFromHeader, _importHeaderEnd, _importMessages) ← Lean.Elab.parseImports contents (some fileName)
    let imports :=
      if importsFromHeader.isEmpty then
        #[{ module := `Mathlib }]
      else
        importsFromHeader
    let env ← importModules imports {}

    processCommandsWithFrontend inputCtx contents env stateAfterHeader messages
    return (0 : UInt32)
