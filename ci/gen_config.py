#!/usr/bin/env python3
"""
gen_config.py

Generates configuration macros for:
  - Verilog header (-f verilog)
  - C/C++ header (-f cpp)
  - compiler flags (-f cflags)

Key features:
  - Reads defaults from TOML (ordered, sectioned).
  - Applies overrides from --cflags and trailing -D... args.
  - Supports [[enum]] blocks in TOML to declare enum-typed parameters.
  - Supports [[builtin]] blocks in TOML to declare builtin variables used only in expr evaluation.
  - Supports [[param]] blocks in TOML to declare parameter variables used only in expr evaluation / unresolved RHS.
  - builtin/param variables are NOT emitted to outputs.
  - Supports "expr:" values in TOML, with $NAME references.
  - Unresolved header mode (default for cpp/verilog): emits preprocessor-friendly
    definitions that can be overridden from -D flags.
  - Resolved mode (-r/--resolved): fully evaluates expressions and emits
    resolved constants (always enabled for cflags).
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import shlex
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

try:
  import tomllib  # Python 3.11+
except ModuleNotFoundError:
  try:
    import tomli as tomllib  # type: ignore
  except ModuleNotFoundError as e:
    raise SystemExit("Missing TOML parser: use Python>=3.11 (tomllib) or install tomli") from e

# -----------------------------
# Basics
# -----------------------------

_PUBLIC_SCOPE_RE = re.compile(r"^[A-Z0-9_]+$")
_DOLLAR_IDENT_RE = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")
_DOLLAR_BRACE_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_BACKTICK_EXPR_RE = re.compile(r"^`(.+)`$")
_INT_RE = re.compile(r"^[+-]?\d+$")
_HEX_TOKEN_RE = re.compile(r"^\s*0[xX]([0-9A-Fa-f_]+)\s*$")
_HEX_ASSIGN_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(0[xX][0-9A-Fa-f_]+)\s*(?:#.*)?$")


def _has_public_scope(name: str) -> bool:
  return _PUBLIC_SCOPE_RE.fullmatch(name) is not None


def _preprocess_expr(expr: str) -> str:
  expr = _DOLLAR_BRACE_RE.sub(r"\1", expr)
  expr = _DOLLAR_IDENT_RE.sub(r"\1", expr)
  return expr


def _is_expr_string(v: Any) -> bool:
  if not isinstance(v, str):
    return False
  s = v.strip()
  return s.startswith("expr:") or (_BACKTICK_EXPR_RE.fullmatch(s) is not None)


def _extract_expr(v: str) -> Optional[str]:
  s = v.strip()
  if s.startswith("expr:"):
    return s[len("expr:"):].strip()
  m = _BACKTICK_EXPR_RE.fullmatch(s)
  if m:
    return m.group(1).strip()
  return None


def _scalar(s: str) -> Any:
  ss = s.strip()
  if ss.lower() in ("true", "false"):
    return (ss.lower() == "true")
  if _INT_RE.fullmatch(ss):
    return int(ss, 10)
  if (ss.startswith('"') and ss.endswith('"')) or (ss.startswith("'") and ss.endswith("'")):
    return ss[1:-1]
  return ss


def _truthy(v: Any) -> bool:
  if isinstance(v, bool):
    return v
  if isinstance(v, int):
    return v != 0
  if isinstance(v, str):
    s = v.strip().lower()
    if s in ("0", "false", "no", "off", ""):
      return False
    if s in ("1", "true", "yes", "on"):
      return True
    return True
  return bool(v)


def _cpp_quote(s: str) -> str:
  return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


# -----------------------------
# Hex metadata (TOML source scan)
# -----------------------------

@dataclass(frozen=True)
class HexMeta:
  digits: str
  width: int  # exact nibble width in bits = 4 * len(digits)


def _scan_hex_literals(toml_path: str) -> Dict[str, HexMeta]:
  out: Dict[str, HexMeta] = {}
  in_enum = False
  in_builtin = False
  in_param = False
  with open(toml_path, "r", encoding="utf-8") as f:
    for raw_line in f:
      line = raw_line.rstrip("\n")
      s = line.strip()
      if not s:
        continue
      if s.startswith("[[") and s.endswith("]]"):
        in_enum = (s == "[[enum]]")
        in_builtin = (s == "[[builtin]]")
        in_param = (s == "[[param]]")
        continue
      if s.startswith("[") and s.endswith("]"):
        in_enum = False
        in_builtin = False
        in_param = False
        continue
      if in_enum or in_builtin or in_param:
        continue
      if "=" in line:
        rhs = line.split("=", 1)[1].lstrip()
        if rhs.startswith('"') or rhs.startswith("'"):
          continue
      m = _HEX_ASSIGN_RE.match(line)
      if not m:
        continue
      key = m.group(1)
      lit = m.group(2)
      digits = lit[2:].replace("_", "")
      width = 4 * len(digits)
      out[key] = HexMeta(digits=digits.upper(), width=width)
  return out


def _format_int_literal(dkind: str, key: str, value: int, hex_meta: Dict[str, HexMeta]) -> str:
  hm = hex_meta.get(key)
  if hm is None:
    return str(value)
  if dkind == "sv":  # verilog-style literal
    return f"{hm.width}'h{hm.digits}"
  return "0x" + hm.digits


def _format_int_from_expr_source(dkind: str, value_node: ast.AST, value: int, expr_src: str) -> str:
  seg = ast.get_source_segment(expr_src, value_node) or ""
  m = _HEX_TOKEN_RE.match(seg)
  if not m:
    return str(value)
  digits = m.group(1).replace("_", "").upper()
  width = 4 * len(digits)
  if dkind == "sv":
    return f"{width}'h{digits}"
  return "0x" + digits


# -----------------------------
# TOML layout (ordered + sections)
# -----------------------------

@dataclass
class Layout:
  sections: List[Tuple[Optional[str], List[str]]]
  ordered_keys: List[str]
  key_to_section: Dict[str, Optional[str]]


def _load_toml(path: str) -> Dict[str, Any]:
  with open(path, "rb") as f:
    data = tomllib.load(f)
  if not isinstance(data, dict):
    raise ValueError("TOML root must be a table")
  return data


def _flatten_with_layout(toml_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Layout]:
  flat: Dict[str, Any] = {}
  sections: List[Tuple[Optional[str], List[str]]] = []
  ordered: List[str] = []
  k2s: Dict[str, Optional[str]] = {}

  root_keys: List[str] = []
  for top_k, top_v in toml_data.items():
    if top_k in ("enum", "builtin", "param"):
      continue
    if isinstance(top_v, dict):
      sec_keys: List[str] = []
      for k, v in top_v.items():
        flat[k] = v
        sec_keys.append(k)
        ordered.append(k)
        k2s[k] = top_k
      sections.append((top_k, sec_keys))
    else:
      flat[top_k] = top_v
      root_keys.append(top_k)
      ordered.append(top_k)
      k2s[top_k] = None

  if root_keys:
    sections.insert(0, (None, root_keys))

  return flat, Layout(sections=sections, ordered_keys=ordered, key_to_section=k2s)


# -----------------------------
# Enums (declared in TOML with [[enum]])
# -----------------------------

@dataclass(frozen=True)
class EnumSpec:
  values: List[Any]


def _load_enums(toml_data: Dict[str, Any]) -> Dict[str, EnumSpec]:
  enums = toml_data.get("enum", None)
  if enums is None:
    return {}
  if not isinstance(enums, list):
    raise ValueError("TOML 'enum' must be an array of tables (use [[enum]])")

  out: Dict[str, EnumSpec] = {}
  for i, item in enumerate(enums):
    if not isinstance(item, dict) or len(item) == 0:
      raise ValueError(f"enum[{i}] must be a non-empty table like: [[enum]] XLEN=[32,64]")
    for name, values in item.items():
      if not isinstance(values, list) or len(values) == 0:
        raise ValueError(f"enum[{i}].{name} must be a non-empty list")
      if name in out:
        raise ValueError(f"Duplicate enum name: {name}")
      norm: List[Any] = []
      for v in values:
        if isinstance(v, str):
          norm.append(_scalar(v))
        else:
          norm.append(v)
      out[name] = EnumSpec(values=norm)
  return out


# -----------------------------
# Builtins / Params (declared in TOML with [[builtin]] / [[param]])
# -----------------------------

@dataclass(frozen=True)
class VarSpec:
  typ: str  # "bool" | "int" | "string"


def _load_var_table(toml_data: Dict[str, Any], key: str) -> Dict[str, VarSpec]:
  arr = toml_data.get(key, None)
  if arr is None:
    return {}
  if not isinstance(arr, list):
    raise ValueError(f"TOML '{key}' must be an array of tables (use [[{key}]])")

  out: Dict[str, VarSpec] = {}
  for i, item in enumerate(arr):
    if not isinstance(item, dict) or len(item) == 0:
      raise ValueError(f"Each [[{key}]] must be a non-empty table, e.g. [[{key}]] FOO=\"bool\"")
    for name, typ in item.items():
      if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError(f"Invalid {key} name: {name!r}")
      if name in out:
        raise ValueError(f"Duplicate {key} name: {name}")
      if not isinstance(typ, str):
        raise ValueError(
          f"{key} type for {name} must be a string (TOML requires quotes), e.g. {name}=\"int\""
        )
      t = typ.strip().lower()
      if t not in ("bool", "int", "string"):
        raise ValueError(f"Unsupported {key} type for {name}: {typ!r} (use \"bool\"|\"int\"|\"string\")")
      out[name] = VarSpec(typ=t)
  return out


def _var_default(spec: VarSpec) -> Any:
  if spec.typ == "bool":
    return False
  if spec.typ == "int":
    return 0
  return ""


# -----------------------------
# -D parsing + overrides
# -----------------------------

@dataclass
class Define:
  name: str
  value: Optional[str]


def _parse_defines(tokens: List[str]) -> List[Define]:
  out: List[Define] = []
  for tok in tokens:
    if not tok.startswith("-D"):
      continue
    item = tok[2:]
    if not item:
      continue
    if "=" in item:
      n, v = item.split("=", 1)
      out.append(Define(name=n.strip(), value=v.strip()))
    else:
      out.append(Define(name=item.strip(), value=None))
  return out


def _parse_defines_from_cflags(cflags: str) -> List[Define]:
  if not cflags:
    return []
  return _parse_defines(shlex.split(cflags))


def _apply_overrides(defs: List[Define], enums: Dict[str, EnumSpec]) -> Tuple[Dict[str, Any], Set[str]]:
  overrides: Dict[str, Any] = {}
  explicit: Set[str] = set()

  def set_enum(name: str, val: Any) -> None:
    spec = enums[name]
    if val not in spec.values:
      raise ValueError(f"Invalid value for {name}: {val} (allowed: {spec.values})")
    overrides[name] = val
    explicit.add(name)

  for d in defs:
    if d.name.endswith("_DISABLE"):
      base = d.name[:-8]
      en = f"{base}_ENABLE"
      dis = True if d.value is None else _truthy(_scalar(d.value))
      if dis:
        overrides[en] = False
        explicit.add(en)
      continue

    if d.name in enums:
      if d.value is None:
        raise ValueError(f"Enum override requires value: -D{d.name}=<value> or -D{d.name}_<value>")
      set_enum(d.name, _scalar(d.value))
      continue

    if "_" in d.name and d.value is None:
      base, suffix = d.name.rsplit("_", 1)
      if base in enums:
        set_enum(base, _scalar(suffix))
        overrides[d.name] = True
        explicit.add(d.name)
        continue

    if d.value is None:
      overrides[d.name] = True
      explicit.add(d.name)
    else:
      overrides[d.name] = _scalar(d.value)
      explicit.add(d.name)

  return overrides, explicit


# -----------------------------
# Dialect helpers
# -----------------------------

@dataclass(frozen=True)
class Dialect:
  kind: str  # "c" or "sv"

  def ifndef(self) -> str:
    return "#ifndef" if self.kind == "c" else "`ifndef"

  def ifdef(self) -> str:
    return "#ifdef" if self.kind == "c" else "`ifdef"

  def else_(self) -> str:
    return "#else" if self.kind == "c" else "`else"

  def endif(self) -> str:
    return "#endif" if self.kind == "c" else "`endif"

  def define(self) -> str:
    return "#define" if self.kind == "c" else "`define"

  def undef(self) -> str:
    return "#undef" if self.kind == "c" else "`undef"

  def ref(self, name: str) -> str:
    return name if self.kind == "c" else f"`{name}"


def _make_guard_macro(output_path: Optional[str], fallback: str) -> str:
  if not output_path:
    return fallback
  base = os.path.basename(output_path)
  base = re.sub(r"[^A-Za-z0-9]", "_", base).upper()
  if not base:
    return fallback
  if base[0].isdigit():
    base = "_" + base
  return base


# -----------------------------
# Builtin helpers in unresolved mode
# -----------------------------

class HelperSpec:
  fn: str
  macro: str

  def emit(self, d: Dialect) -> List[str]:
    raise NotImplementedError

  def translate_call(self, d: Dialect, args: List[str]) -> str:
    raise NotImplementedError


def _guarded_macro(d: Dialect, name: str, body: str) -> List[str]:
  return [
    f"{d.ifndef()} {name}",
    f"{d.define()} {name}{body}",
    f"{d.endif()}",
    "",
  ]


def _fold_binary(macro_call: str, args: List[str]) -> str:
  if len(args) == 0:
    return "0"
  if len(args) == 1:
    return args[0]
  s = f"{macro_call}({args[0]}, {args[1]})"
  for x in args[2:]:
    s = f"{macro_call}({s}, {x})"
  return s


class MinHelper(HelperSpec):
  fn = "min"
  macro = "__MIN"

  def emit(self, d: Dialect) -> List[str]:
    return _guarded_macro(d, self.macro, "(a,b) (((a) < (b)) ? (a) : (b))")

  def translate_call(self, d: Dialect, args: List[str]) -> str:
    return _fold_binary(d.ref(self.macro), args)


class MaxHelper(HelperSpec):
  fn = "max"
  macro = "__MAX"

  def emit(self, d: Dialect) -> List[str]:
    return _guarded_macro(d, self.macro, "(a,b) (((a) > (b)) ? (a) : (b))")

  def translate_call(self, d: Dialect, args: List[str]) -> str:
    return _fold_binary(d.ref(self.macro), args)


class UpHelper(HelperSpec):
  fn = "up"
  macro = "__UP"

  def emit(self, d: Dialect) -> List[str]:
    return _guarded_macro(d, self.macro, "(x) (((x) != 0) ? (x) : 1)")

  def translate_call(self, d: Dialect, args: List[str]) -> str:
    if len(args) != 1:
      return "0"
    return f"{d.ref(self.macro)}({args[0]})"


class ClampHelper(HelperSpec):
  fn = "clamp"
  macro = "__CLAMP"

  def emit(self, d: Dialect) -> List[str]:
    return _guarded_macro(
      d,
      self.macro,
      "(x,lo,hi) (((x) > (hi)) ? (hi) : (((x) < (lo)) ? (lo) : (x)))",
    )

  def translate_call(self, d: Dialect, args: List[str]) -> str:
    if len(args) != 3:
      return "0"
    return f"{d.ref(self.macro)}({args[0]}, {args[1]}, {args[2]})"


HELPERS: Dict[str, HelperSpec] = {
  "min": MinHelper(),
  "max": MaxHelper(),
  "up": UpHelper(),
  "clamp": ClampHelper(),
}


def _helpers_used(expr: str) -> Set[str]:
  expr2 = _preprocess_expr(expr)
  tree = ast.parse(expr2, mode="eval")
  used: Set[str] = set()
  for n in ast.walk(tree):
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
      if n.func.id in HELPERS:
        used.add(n.func.id)
  return used


# -----------------------------
# Unresolved macro expression translator
# -----------------------------

class MacroExprTranslator:
  def __init__(self, d: Dialect, params: Set[str], expr_src: Optional[str] = None) -> None:
    self.d = d
    self.params = params
    self.expr_src = expr_src

  def translate(self, expr: str, current_key: Optional[str], enums: Dict[str, EnumSpec]) -> str:
    expr2 = _preprocess_expr(expr)
    tree = ast.parse(expr2, mode="eval")
    self.expr_src = expr2
    return self._emit(tree.body, current_key=current_key, enums=enums)

  def _name_ref(self, name: str) -> str:
    if self.d.kind == "sv" and name in self.params:
      return name
    return self.d.ref(name)

  def _emit(self, node: ast.AST, current_key: Optional[str], enums: Dict[str, EnumSpec]) -> str:
    if isinstance(node, ast.Constant):
      if isinstance(node.value, bool):
        return "1" if node.value else "0"
      if node.value is None:
        return "0"
      if isinstance(node.value, int):
        if self.expr_src is not None:
          return _format_int_from_expr_source(self.d.kind, node, node.value, self.expr_src)
        return str(node.value)
      if isinstance(node.value, str):
        if current_key is not None and current_key in enums:
          if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", node.value):
            return node.value
        return _cpp_quote(node.value)
      return str(node.value)

    if isinstance(node, ast.Name):
      return self._name_ref(node.id)

    if isinstance(node, ast.BoolOp):
      join = " && " if isinstance(node.op, ast.And) else " || "
      parts = [self._emit(v, current_key, enums) for v in node.values]
      return "(" + join.join(f"({p})" for p in parts) + ")"

    if isinstance(node, ast.UnaryOp):
      if isinstance(node.op, ast.USub):
        return f"(-({self._emit(node.operand, current_key, enums)}))"
      if isinstance(node.op, ast.Not):
        return f"(!({self._emit(node.operand, current_key, enums)}))"
      return f"({self._emit(node.operand, current_key, enums)})"

    if isinstance(node, ast.BinOp):
      a = self._emit(node.left, current_key, enums)
      b = self._emit(node.right, current_key, enums)
      op = node.op
      if isinstance(op, (ast.Div, ast.FloorDiv)):
        return f"(({a}) / ({b}))"
      if isinstance(op, ast.Mult):
        return f"(({a}) * ({b}))"
      if isinstance(op, ast.Add):
        return f"(({a}) + ({b}))"
      if isinstance(op, ast.Sub):
        return f"(({a}) - ({b}))"
      if isinstance(op, ast.LShift):
        return f"(({a}) << ({b}))"
      if isinstance(op, ast.RShift):
        return f"(({a}) >> ({b}))"
      if isinstance(op, ast.BitOr):
        return f"(({a}) | ({b}))"
      if isinstance(op, ast.BitAnd):
        return f"(({a}) & ({b}))"
      if isinstance(op, ast.BitXor):
        return f"(({a}) ^ ({b}))"
      if isinstance(op, ast.Mod):
        return f"(({a}) % ({b}))"
      raise ValueError(f"Unsupported BinOp: {type(op).__name__}")

    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
      fn = node.func.id
      args = [self._emit(a, current_key, enums) for a in node.args]
      h = HELPERS.get(fn)
      if h is not None:
        return h.translate_call(self.d, args)
      if fn == "int" and len(args) == 1:
        return f"({args[0]})"
      if fn == "bool" and len(args) == 1:
        return f"(({args[0]}) ? 1 : 0)"
      raise ValueError(f"Unsupported function '{fn}' in unresolved macro expr")

    if isinstance(node, ast.IfExp):
      raise ValueError("Conditional expressions must be emitted via `ifdef trees, not as RHS")

    raise ValueError(f"Unsupported AST node in unresolved macro expr: {type(node).__name__}")


# -----------------------------
# Unresolved conditional emission
# -----------------------------

def _emit_test_tree(lines: List[str], d: Dialect, test: ast.AST, emit_true, emit_false) -> None:
  if isinstance(test, ast.Name):
    lines.append(f"{d.ifdef()} {test.id}")
    emit_true()
    lines.append(d.else_())
    emit_false()
    lines.append(d.endif())
    return

  if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not) and isinstance(test.operand, ast.Name):
    lines.append(f"{d.ifndef()} {test.operand.id}")
    emit_true()
    lines.append(d.else_())
    emit_false()
    lines.append(d.endif())
    return

  if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And):
    vals = list(test.values)

    def emit_and(idx: int) -> None:
      if idx >= len(vals):
        emit_true()
        return
      t = vals[idx]

      def t_true():
        emit_and(idx + 1)

      def t_false():
        emit_false()

      _emit_test_tree(lines, d, t, t_true, t_false)

    emit_and(0)
    return

  if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.Or):
    vals = list(test.values)

    def emit_or(idx: int) -> None:
      if idx >= len(vals):
        emit_false()
        return
      t = vals[idx]

      def t_true():
        emit_true()

      def t_false():
        emit_or(idx + 1)

      _emit_test_tree(lines, d, t, t_true, t_false)

    emit_or(0)
    return

  raise ValueError("Unresolved boolean/conditional test must use only $FLAG / not $FLAG / ($A and $B) / ($A or $B)")


def _is_pure_flag_test(node: ast.AST) -> bool:
  if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not) and isinstance(node.operand, ast.Name):
    return True
  if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
    def _leaf_ok(n: ast.AST) -> bool:
      if isinstance(n, ast.Name):
        return True
      if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not) and isinstance(n.operand, ast.Name):
        return True
      if isinstance(n, ast.BoolOp) and isinstance(n.op, (ast.And, ast.Or)):
        return all(_leaf_ok(v) for v in n.values)
      return False
    return _leaf_ok(node)
  return False


def _emit_unresolved_define_value(lines: List[str], d: Dialect, key: str, value_node: ast.AST,
                                 enums: Dict[str, EnumSpec], expr_src: str, params: Set[str]) -> None:
  if key in enums:
    if isinstance(value_node, ast.Constant) and isinstance(value_node.value, str):
      v = value_node.value
      if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", v):
        raise ValueError(f"Enum {key} value must be identifier-like, got: {v!r}")
      lines.append(f"{d.define()} {key} {v}")
      lines.append(f"{d.define()} {key}_{v}")
      return
    if isinstance(value_node, ast.Constant) and isinstance(value_node.value, int):
      v = value_node.value
      lines.append(f"{d.define()} {key} {v}")
      lines.append(f"{d.define()} {key}_{v}")
      return
    raise ValueError(f"Enum {key} unresolved branch must yield a string/int literal")

  if isinstance(value_node, ast.Constant) and isinstance(value_node.value, int):
    rhs = _format_int_from_expr_source(d.kind, value_node, value_node.value, expr_src)
    lines.append(f"{d.define()} {key} {rhs}".rstrip())
    return

  tr = MacroExprTranslator(d, params=params, expr_src=expr_src)
  rhs = tr._emit(value_node, current_key=None, enums=enums)
  lines.append(f"{d.define()} {key} {rhs}".rstrip())


def _emit_unresolved_conditional(lines: List[str], d: Dialect, key: str, node: ast.IfExp,
                                enums: Dict[str, EnumSpec], expr_src: str, params: Set[str]) -> None:
  def emit_true():
    if isinstance(node.body, ast.IfExp):
      _emit_unresolved_conditional(lines, d, key, node.body, enums, expr_src, params)
    else:
      _emit_unresolved_define_value(lines, d, key, node.body, enums, expr_src, params)

  def emit_false():
    if isinstance(node.orelse, ast.IfExp):
      _emit_unresolved_conditional(lines, d, key, node.orelse, enums, expr_src, params)
    else:
      _emit_unresolved_define_value(lines, d, key, node.orelse, enums, expr_src, params)

  _emit_test_tree(lines, d, node.test, emit_true, emit_false)


def _emit_unresolved_boolean_define(lines: List[str], d: Dialect, key: str, test: ast.AST, add_blank: bool = True) -> None:
  lines.append(f"{d.ifndef()} {key}")

  def t_true():
    lines.append(f"{d.define()} {key}")

  def t_false():
    return

  _emit_test_tree(lines, d, test, t_true, t_false)
  lines.append(d.endif())
  if add_blank:
    lines.append("")


def _emit_unresolved_key(lines: List[str], d: Dialect, key: str, raw: Any,
                         enums: Dict[str, EnumSpec], hex_meta: Dict[str, HexMeta],
                         params: Set[str]) -> None:
  if not _has_public_scope(key):
    return

  # enum handling unchanged...
  if key in enums:
    for v in enums[key].values:
      flag = f"{key}_{v}"
      lines.append(f"{d.ifdef()} {flag}")
      lines.append(f"{d.undef()} {key}")
      lines.append(f"{d.define()} {key} {v}")
      lines.append(d.endif())
    lines.append("")

    for v in enums[key].values:
      lines.append(f"{d.ifndef()} {key}_{v}")
    lines.append(f"{d.ifndef()} {key}")

    if _is_expr_string(raw):
      expr = _extract_expr(str(raw))
      if expr is None:
        raise ValueError(f"Bad expr syntax for {key}: {raw}")
      expr2 = _preprocess_expr(expr)
      tree = ast.parse(expr2, mode="eval")
      if isinstance(tree.body, ast.IfExp):
        _emit_unresolved_conditional(lines, d, key, tree.body, enums, expr2, params)
      elif isinstance(tree.body, ast.Constant) and isinstance(tree.body.value, (str, int)):
        _emit_unresolved_define_value(lines, d, key, tree.body, enums, expr2, params)
      else:
        raise ValueError(f"Enum {key} default expr must be conditional or string/int literal")
    else:
      v0 = raw
      if isinstance(v0, str):
        v0 = _scalar(v0)
      if v0 not in enums[key].values:
        raise ValueError(f"Enum {key} default {v0!r} not in allowed {enums[key].values}")
      lines.append(f"{d.define()} {key} {v0}")
      lines.append(f"{d.define()} {key}_{v0}")

    lines.append(d.endif())
    for _ in enums[key].values:
      lines.append(d.endif())
    lines.append("")
    return

  disable_guard = None
  if key.endswith("_ENABLE"):
    disable_guard = key[:-7] + "_DISABLE"

  # bool default (non-expr)
  if key.endswith("_ENABLE") and not _is_expr_string(raw):
    v = raw
    if isinstance(v, str):
      v = _scalar(v)
    if _truthy(v):
      # keep existing behavior: default true => guarded by DISABLE
      lines.append(f"{d.ifndef()} {disable_guard}")
      lines.append(f"{d.define()} {key}")
      lines.append(d.endif())
      lines.append("")
    return

  if _is_expr_string(raw):
    expr = _extract_expr(str(raw))
    if expr is None:
      raise ValueError(f"Bad expr syntax for {key}: {raw}")
    expr2 = _preprocess_expr(expr)
    tree = ast.parse(expr2, mode="eval")

    # SPECIAL CASE: X_ENABLE = "expr: $FLAG"
    if key.endswith("_ENABLE") and isinstance(tree.body, ast.Name):
      flag = tree.body.id
      # BUG FIX: wrap auto-enable with `ifndef X_DISABLE
      lines.append(f"{d.ifndef()} {disable_guard}")
      lines.append(f"{d.ifndef()} {key}")
      lines.append(f"{d.ifdef()} {flag}")
      lines.append(f"{d.define()} {key}")
      lines.append(d.endif())
      lines.append(d.endif())
      lines.append(d.endif())
      lines.append("")
      return

    # SPECIAL CASE: KEY = "expr: $OTHER" (non-enable)
    if isinstance(tree.body, ast.Name):
      nm = tree.body.id
      rhs = nm if (d.kind == "sv" and nm in params) else d.ref(nm)
      lines.append(f"{d.ifndef()} {key}")
      lines.append(f"{d.define()} {key} {rhs}")
      lines.append(d.endif())
      lines.append("")
      return

    # PURE FLAG BOOLEAN EXPR => emit nested ifdefs
    if _is_pure_flag_test(tree.body):
      # BUG FIX: if this is X_ENABLE, guard auto-define under X_DISABLE
      if key.endswith("_ENABLE"):
        lines.append(f"{d.ifndef()} {disable_guard}")
        _emit_unresolved_boolean_define(lines, d, key, tree.body, add_blank=False)
        lines.append(d.endif())
        lines.append("")
      else:
        _emit_unresolved_boolean_define(lines, d, key, tree.body, add_blank=True)
      return

    # Conditional expression => `ifdef tree
    if isinstance(tree.body, ast.IfExp):
      if key.endswith("_ENABLE"):
        lines.append(f"{d.ifndef()} {disable_guard}")
      lines.append(f"{d.ifndef()} {key}")
      _emit_unresolved_conditional(lines, d, key, tree.body, enums, expr2, params)
      lines.append(d.endif())
      if key.endswith("_ENABLE"):
        lines.append(d.endif())
      lines.append("")
      return

    # Generic expression
    tr = MacroExprTranslator(d, params=params, expr_src=expr2)
    rhs = tr._emit(tree.body, current_key=key if key in enums else None, enums=enums)
    if key.endswith("_ENABLE"):
      lines.append(f"{d.ifndef()} {disable_guard}")
    lines.append(f"{d.ifndef()} {key}")
    lines.append(f"{d.define()} {key} {rhs}".rstrip())
    lines.append(d.endif())
    if key.endswith("_ENABLE"):
      lines.append(d.endif())
    lines.append("")
    return

  # non-expr literal
  v = _scalar(raw) if isinstance(raw, str) else raw
  if isinstance(v, bool):
    if v:
      lines.append(f"{d.ifndef()} {key}")
      lines.append(f"{d.define()} {key}")
      lines.append(d.endif())
      lines.append("")
    return

  lines.append(f"{d.ifndef()} {key}")
  if isinstance(v, int):
    lit = _format_int_literal(d.kind, key, v, hex_meta)
    lines.append(f"{d.define()} {key} {lit}")
  elif isinstance(v, str):
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", v):
      lines.append(f"{d.define()} {key} {v}")
    else:
      lines.append(f"{d.define()} {key} {_cpp_quote(v)}")
  else:
    lines.append(f"{d.define()} {key} {v}")
  lines.append(d.endif())
  lines.append("")


def emit_unresolved_header(toml_defs: Dict[str, Any], layout: Layout, enums: Dict[str, EnumSpec],
                           fmt: str, output_path: Optional[str], hex_meta: Dict[str, HexMeta],
                           params: Set[str]) -> str:
  d = Dialect("c" if fmt == "cpp" else "sv")
  tool = os.path.basename(sys.argv[0])

  used: Set[str] = set()
  for k in layout.ordered_keys:
    if not _has_public_scope(k):
      continue
    raw = toml_defs.get(k)
    if raw is None or not _is_expr_string(raw):
      continue
    expr = _extract_expr(str(raw)) or ""
    used |= _helpers_used(expr)

  guard = _make_guard_macro(output_path, "GEN_CONFIG")
  lines: List[str] = []
  lines.append(f"// Auto-generated by {tool}")
  lines.append("")
  lines.append(f"{d.ifndef()} {guard}")
  lines.append(f"{d.define()} {guard}")
  lines.append("")

  for fn in sorted(used):
    h = HELPERS.get(fn)
    if h is not None:
      lines.extend(h.emit(d))

  for sec, keys in layout.sections:
    sec_lines: List[str] = []
    for k in keys:
      if k not in toml_defs:
        continue
      if not _has_public_scope(k):
        continue
      _emit_unresolved_key(sec_lines, d, k, toml_defs[k], enums, hex_meta, params)

    while sec_lines and sec_lines[-1] == "":
      sec_lines.pop()
    if not sec_lines:
      continue

    lines.append("")
    if sec is not None:
      lines.append(f"// --- {sec} ---")
    lines.extend(sec_lines)

  lines.append("")
  lines.append(f"{d.endif()}  // {guard}")
  return "\n".join(lines).rstrip() + "\n"


# -----------------------------
# Resolved evaluation
# -----------------------------

def _py_up(x: Any) -> Any:
  return x if x != 0 else 1


def _py_clamp(x: Any, lo: Any, hi: Any) -> Any:
  return hi if x > hi else (lo if x < lo else x)


class Resolver:
  def __init__(self, base: Dict[str, Any], overrides: Dict[str, Any],
               enums: Dict[str, EnumSpec], builtins: Dict[str, VarSpec], params: Dict[str, VarSpec]) -> None:
    self.base = base
    self.overrides = overrides
    self.enums = enums
    self.builtins = builtins
    self.params = params
    self.cache: Dict[str, Any] = {}
    self.visiting: Set[str] = set()

  def resolve(self, key: str) -> Any:
    if key in self.cache:
      return self.cache[key]
    if key in self.visiting:
      raise ValueError(f"Cycle detected while resolving '{key}'")

    if key in self.builtins:
      v = _var_default(self.builtins[key])
      self.cache[key] = v
      return v

    if key in self.params:
      v = _var_default(self.params[key])
      self.cache[key] = v
      return v

    if key in self.overrides:
      self.cache[key] = self.overrides[key]
      return self.cache[key]

    if "_" in key:
      base, suffix = key.rsplit("_", 1)
      if base in self.enums:
        cur = self.resolve(base)
        val = _scalar(suffix)
        self.cache[key] = (cur == val)
        return self.cache[key]

    if key not in self.base:
      raise ValueError(f"Undefined key '{key}'")

    raw = self.base[key]
    v = self._resolve_value(raw)

    if key in self.enums:
      spec = self.enums[key]
      if v not in spec.values:
        raise ValueError(f"Invalid value for {key}: {v} (allowed: {spec.values})")

    if key.endswith("_ENABLE"):
      v = _truthy(v)

    self.cache[key] = v
    return v

  def _resolve_value(self, raw: Any) -> Any:
    if _is_expr_string(raw):
      expr = _extract_expr(str(raw))
      if expr is None:
        raise ValueError(f"Bad expr syntax: {raw}")
      return self._eval(expr)
    if isinstance(raw, str):
      return _scalar(raw)
    return raw

  def _eval(self, expr: str) -> Any:
    expr2 = _preprocess_expr(expr)
    scope = EvalScope(self)
    try:
      return eval(expr2, {"__builtins__": {}}, scope)  # noqa: S307
    except Exception as e:
      raise ValueError(f"Failed to eval expr '{expr}': {e}") from e


class EvalScope(dict):
  def __init__(self, r: Resolver) -> None:
    super().__init__()
    self.r = r

  def __missing__(self, k: str) -> Any:
    if k == "min":
      return min
    if k == "max":
      return max
    if k == "up":
      return _py_up
    if k == "clamp":
      return _py_clamp
    if k == "int":
      return int
    if k == "bool":
      return bool
    if k in ("True", "true"):
      return True
    if k in ("False", "false"):
      return False
    if k == "None":
      return None
    return self.r.resolve(k)


def emit_resolved_header(cfg: Dict[str, Any], layout: Layout, enums: Dict[str, EnumSpec],
                         fmt: str, output_path: Optional[str], hex_meta: Dict[str, HexMeta]) -> str:
  d = Dialect("c" if fmt == "cpp" else "sv")
  tool = os.path.basename(sys.argv[0])
  guard = _make_guard_macro(output_path, "GEN_CONFIG")

  lines: List[str] = []
  lines.append(f"// Auto-generated by {tool}")
  lines.append("")
  lines.append(f"{d.ifndef()} {guard}")
  lines.append(f"{d.define()} {guard}")
  lines.append("")

  def emit_define(sec_lines: List[str], k: str, v: Any) -> None:
    if not _has_public_scope(k):
      return
    if isinstance(v, bool):
      if v:
        sec_lines.append(f"{d.define()} {k}")
      return
    if isinstance(v, int):
      sec_lines.append(f"{d.define()} {k} {_format_int_literal(d.kind, k, v, hex_meta)}")
      return
    if isinstance(v, str):
      if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", v):
        sec_lines.append(f"{d.define()} {k} {v}")
      else:
        sec_lines.append(f"{d.define()} {k} {_cpp_quote(v)}")
      return
    sec_lines.append(f"{d.define()} {k} {v}")

  for sec, keys in layout.sections:
    sec_lines: List[str] = []
    for k in keys:
      if k not in cfg:
        continue
      emit_define(sec_lines, k, cfg[k])
      if k in enums:
        sec_lines.append(f"{d.define()} {k}_{cfg[k]}")

    if not sec_lines:
      continue

    lines.append("")
    if sec is not None:
      lines.append(f"// --- {sec} ---")
    lines.extend(sec_lines)

  lines.append("")
  lines.append(f"{d.endif()}  // {guard}")
  return "\n".join(lines).rstrip() + "\n"


def emit_cflags(cfg: Dict[str, Any], layout: Layout, enums: Dict[str, EnumSpec],
               hex_meta: Dict[str, HexMeta]) -> str:
  toks: List[str] = []
  for k in enums.keys():
    v = cfg[k]
    toks.append(f"-D{k}={v}")
    toks.append(f"-D{k}_{v}")

  for sec, keys in layout.sections:
    for k in keys:
      if not _has_public_scope(k):
        continue
      if k not in cfg:
        continue
      v = cfg[k]
      if isinstance(v, bool):
        if v:
          toks.append(f"-D{k}")
      elif isinstance(v, int):
        lit = _format_int_literal("c", k, v, hex_meta)
        toks.append(f"-D{k}={lit}")
      elif isinstance(v, str):
        if v != "":
          toks.append(f"-D{k}={v}")
      else:
        toks.append(f"-D{k}={v}")
  return " ".join(toks) + "\n"


# -----------------------------
# Main
# -----------------------------

def main(argv: List[str]) -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--config", "-c", required=True, help="path to VX_config.toml")
  ap.add_argument("--cflags", default="", help="existing flags containing -D overrides")
  ap.add_argument("--format", "-f", choices=["cflags", "cpp", "verilog"], default="cflags")
  ap.add_argument("--output", "-o", default=None)
  ap.add_argument("--resolved", "-r", action="store_true",
                  help="Emit resolved constants for cpp/verilog (cflags is always resolved)")
  args, unknown = ap.parse_known_args(argv)

  hex_meta = _scan_hex_literals(args.config)

  toml_data = _load_toml(args.config)
  enums = _load_enums(toml_data)
  builtins = _load_var_table(toml_data, "builtin")
  params_tbl = _load_var_table(toml_data, "param")
  params_set = set(params_tbl.keys())

  toml_defs, layout = _flatten_with_layout(toml_data)

  defs: List[Define] = []
  defs += _parse_defines_from_cflags(args.cflags)
  defs += _parse_defines(unknown)
  overrides, _explicit = _apply_overrides(defs, enums)
  # - A name declared under [[param]] / [[builtin]] is a read-only symbol used only for expression evaluation.
  ro = params_set | set(builtins.keys())
  assigned = set(toml_defs.keys()) & ro
  if assigned:
    names = ", ".join(sorted(assigned))
    raise ValueError(
      f"Read-only symbol(s) assigned in config tables: {names}. "
      "Names declared in [[param]]/[[builtin]] must not appear as regular config keys."
    )

  illegal = set(overrides.keys()) & ro
  if illegal:
    names = ", ".join(sorted(illegal))
    raise ValueError(
      f"Illegal -D override(s) for read-only symbol(s): {names}. "
      "Remove these -D flags; [[param]]/[[builtin]] values are read-only."
    )

  resolved = True if args.format == "cflags" else args.resolved

  if resolved:
    r = Resolver(base=toml_defs, overrides=overrides, enums=enums, builtins=builtins, params=params_tbl)
    cfg: Dict[str, Any] = {}
    for k in layout.ordered_keys:
      cfg[k] = r.resolve(k)
    for k in enums.keys():
      if k not in cfg:
        cfg[k] = r.resolve(k)

    if args.format == "cflags":
      out = emit_cflags(cfg, layout, enums, hex_meta)
    elif args.format == "cpp":
      out = emit_resolved_header(cfg, layout, enums, "cpp", args.output, hex_meta)
    else:
      out = emit_resolved_header(cfg, layout, enums, "verilog", args.output, hex_meta)
  else:
    if args.format == "cflags":
      raise ValueError("Internal: cflags output is always resolved")
    out = emit_unresolved_header(toml_defs, layout, enums, args.format, args.output, hex_meta, params_set)

  if args.output:
    with open(args.output, "w", encoding="utf-8") as f:
      f.write(out)
  else:
    sys.stdout.write(out)
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv[1:]))
