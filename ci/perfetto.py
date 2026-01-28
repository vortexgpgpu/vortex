#!/usr/bin/env python3
"""
vortex_perfetto.py

Convert Vortex GPU RTL or SimX log traces into a Perfetto-compatible trace
(Chrome Trace JSON).

Design goals (per Vortex project requirements)
- One unified output schema regardless of log type (-t/--type).
- Two separate parsers (RTL vs SimX) that emit the same intermediate events.
- Streaming JSON writer (low memory) with optional gzip compression.
- Optional value capture controlled by a single type-agnostic switch:
    --values {none,dest,all}
  By default, only destination values are captured, and dest_value is attached
  ONLY to the commit stage marker.

Usage examples
  # SimX (default), compressed:
  python3 vortex_perfetto.py run_simx.log -c -o vortex.perfetto.json.gz

  # RTL sim:
  python3 vortex_perfetto.py run.log -t rtlsim -c -o vortex.perfetto.json.gz

  # Limit export to a cycle window:
  python3 vortex_perfetto.py run.log -t rtlsim --cycle-min 10000 --cycle-max 20000 -c -o win.json.gz

Open in Perfetto UI
  - Web UI: https://ui.perfetto.dev (load the .json or .json.gz)
  - VS Code: install a Perfetto trace viewer extension and open the .json/.json.gz.

Notes
  - Cache payloads are intentionally NOT captured (too costly). Only hit/miss/fill/writeback
    and lightweight req/rsp events are recorded when present.
  - RTL tmask bit ordering is normalized to match SimX convention (bit0 == thread0).
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import re
import sys
from itertools import chain
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

# -----------------------------
# Intermediate Representation
# -----------------------------

STAGES_CANON = {"schedule", "fetch", "decode", "ibuffer", "dispatch", "operands", "execute", "commit"}

@dataclass
class InstructionStageEvent:
  ts: int
  path: str
  stage: str
  uuid: int
  wid: Optional[int] = None
  pc: Optional[str] = None
  ex: Optional[str] = None
  op: Optional[str] = None
  wb: Optional[int] = None
  tmask: Optional[str] = None
  sop: int = 1
  eop: int = 1
  parent_uuid: Optional[int] = None
  # Canonical reg names (x/f/v) when known
  rd: Optional[str] = None
  rs1: Optional[str] = None
  rs2: Optional[str] = None
  rs3: Optional[str] = None
  # Values (emitted by Perfetto emitter according to --values)
  src_values: Optional[List[Dict[str, str]]] = None  # [{"reg": "...", "value": "..."}]
  dest_value: Optional[Dict[str, str]] = None        # {"reg": "...", "value": "..."}  (commit only)

@dataclass
class WarpStateEvent:
  ts: int
  path: str
  wid: int
  active: Optional[int] = None
  stalled: Optional[int] = None
  tmask: Optional[str] = None

@dataclass
class CacheEvent:
  ts: int
  path: str
  level: str  # icache/dcache/l2/l3/mem/unknown
  kind: str   # hit/miss/fill/writeback/req/rsp/replay/mshr/...
  uuid: Optional[int] = None
  wid: Optional[int] = None
  pc: Optional[str] = None
  tmask: Optional[str] = None
  addr: Optional[str] = None
  tag: Optional[str] = None

@dataclass
class RawInstantEvent:
  ts: int
  path: str
  name: str
  uuid: Optional[int] = None
  args: Dict[str, Any] = field(default_factory=dict)

# -----------------------------
# Shared parsing utilities
# -----------------------------

CONFIGS_RE = re.compile(r"\bCONFIGS:\s*(.*)$")
VXDRV_START_RE = re.compile(r"^\s*\[VXDRV\]\s+START:")

PARENT_HASH_RE = re.compile(r"\bparent=#(\d+)\b")
UUID_SUFFIX_RE = re.compile(r"\(#\s*(\d+)\s*\)\s*$")

# RTL "tick:" prefix
RTL_TICK_RE = re.compile(r"^\s*(\d+)\s*:\s*(.*)$")
# SimX TRACE/DEBUG prefixes
SIMX_TRACE_RE = re.compile(r"^\s*TRACE\s+(?P<cycle>\d+)\s*:\s*(?P<body>.*)$")
SIMX_DEBUG_RE = re.compile(r"^\s*DEBUG\s+(?P<body>.*)$")

# body format: "<path> <action>: <args...>"
BODY_ACTION_ARGS_RE = re.compile(r"^(?P<path>\S+)\s+(?P<action>[^:]+)\s*:\s*(?P<args>.*)$")
BODY_ACTION_ONLY_RE = re.compile(r"^(?P<path>\S+)\s+(?P<action>.*)$")

REG_NAME_RE = re.compile(r"^[xfv]\d+$", re.IGNORECASE)

def parse_configs_line(line: str) -> Optional[Dict[str, str]]:
  m = CONFIGS_RE.search(line)
  if not m:
    return None
  s = m.group(1).strip()
  parts = [p.strip() for p in s.split(",") if p.strip()]
  out: Dict[str, str] = {}
  for p in parts:
    if "=" in p:
      k, v = p.split("=", 1)
      out[k.strip()] = v.strip()
  return out

def split_top_level_commas(s: str) -> List[str]:
  """
  Split a string by commas, but ignore commas inside {...} or (...) or [...]
  (needed for register vectors and payload-like fields).
  """
  out: List[str] = []
  cur: List[str] = []
  depth_curly = 0
  depth_paren = 0
  depth_brack = 0
  i = 0
  while i < len(s):
    c = s[i]
    if c == "{":
      depth_curly += 1
    elif c == "}":
      depth_curly = max(0, depth_curly - 1)
    elif c == "(":
      depth_paren += 1
    elif c == ")":
      depth_paren = max(0, depth_paren - 1)
    elif c == "[":
      depth_brack += 1
    elif c == "]":
      depth_brack = max(0, depth_brack - 1)

    if c == "," and depth_curly == 0 and depth_paren == 0 and depth_brack == 0:
      tok = "".join(cur).strip()
      if tok:
        out.append(tok)
      cur = []
    else:
      cur.append(c)
    i += 1
  tok = "".join(cur).strip()
  if tok:
    out.append(tok)
  return out

def parse_kv_brace_aware(args_str: str) -> Dict[str, str]:
  """
  Parse "k=v, k2=v2, ..." with comma separation at top level only.
  Values may contain braces with commas.
  """
  out: Dict[str, str] = {}
  for tok in split_top_level_commas(args_str):
    if "=" not in tok:
      continue
    k, v = tok.split("=", 1)
    out[k.strip()] = v.strip()
  return out

def safe_int(x: Optional[str]) -> Optional[int]:
  if x is None:
    return None
  try:
    return int(x, 0)
  except Exception:
    try:
      return int(x)
    except Exception:
      return None

def parse_transition(v: str) -> str:
  v = v.strip()
  if "->" in v:
    return v.split("->", 1)[1].strip()
  return v

def normalize_tmask(bitstr: str, num_threads: Optional[int], log_type: str) -> str:
  bitstr = bitstr.strip()
  if num_threads is not None and len(bitstr) < num_threads and all(c in "01" for c in bitstr):
    bitstr = bitstr.rjust(num_threads, "0")
  # RTL uses reversed bit ordering vs SimX
  if log_type == "rtlsim" and all(c in "01" for c in bitstr):
    bitstr = bitstr[::-1]
  return bitstr

def decode_rtl_reg_index(reg_index: str) -> Optional[str]:
  """
  Decode RTL numeric register encoding:
    rtype = ireg // 32
    rvalue = ireg % 32
    rtype == 0 -> x<rvalue>
    rtype == 1 -> f<rvalue>
    rtype == 2 -> v<rvalue>
  """
  try:
    ireg = int(reg_index, 10)
  except Exception:
    return None
  rtype = ireg // 32
  rvalue = ireg % 32
  if rtype == 2:
    return f"v{rvalue}"
  if rtype == 1:
    return f"f{rvalue}"
  return f"x{rvalue}"

def infer_cache_level(path: str) -> str:
  p = path.lower()
  if "icache" in p:
    return "icache"
  if "dcache" in p:
    return "dcache"
  if "l2cache" in p:
    return "l2"
  if "l3cache" in p:
    return "l3"
  if p.startswith("mem") or "dram" in p or p.endswith("-mem") or "-mem" in p:
    return "mem"
  return "unknown"

def is_cache_event(path: str, action: str) -> bool:
  a = action.strip().lower()
  if a == "init":
    return False
  cachey = any(k in a for k in ("hit", "miss", "fill", "writeback", "wb", "mshr", "replay", "req", "rsp"))
  pathy = any(k in path.lower() for k in ("icache", "dcache", "l2cache", "l3cache", "mem", "dram"))
  a_cachey = any(k in a for k in ("icache", "dcache", "l2cache", "l3cache", "mem"))
  return (cachey and pathy) or a_cachey

def extract_body_fields(body: str) -> Tuple[str, str, str]:
  m = BODY_ACTION_ARGS_RE.match(body)
  if m:
    return m.group("path"), m.group("action").strip(), m.group("args").strip()
  m2 = BODY_ACTION_ONLY_RE.match(body)
  if m2:
    return m2.group("path"), m2.group("action").strip(), ""
  return "", "", body.strip()

def strip_payload_heavy_fields(args_str: str) -> str:
  """
  Drop payload-heavy fields to keep JSON small.
  """
  args_str = re.sub(r"\bdata=\{.*\}", "data={...}", args_str)
  args_str = re.sub(r"\ba_row=\{.*\}", "a_row={...}", args_str)
  args_str = re.sub(r"\bb_col=\{.*\}", "b_col={...}", args_str)
  return args_str

# -----------------------------
# Parser base and parsers
# -----------------------------

class ParserBase:
  def __init__(self, *, values_mode: str, cycle_min: Optional[int], cycle_max: Optional[int]):
    self.values_mode = values_mode  # none|dest|all
    self.cycle_min = cycle_min
    self.cycle_max = cycle_max
    self.num_threads: Optional[int] = None

  def _keep(self, ts: int) -> bool:
    if self.cycle_min is not None and ts < self.cycle_min:
      return False
    if self.cycle_max is not None and ts > self.cycle_max:
      return False
    return True

  def parse_lines(self, lines: Iterable[str]) -> Iterator[Any]:
    raise NotImplementedError

@dataclass
class InstrRecord:
  rd: Optional[str] = None
  rs1: Optional[str] = None
  rs2: Optional[str] = None
  rs3: Optional[str] = None
  parent_uuid: Optional[int] = None
  src_values: List[Dict[str, str]] = field(default_factory=list)
  dest_value: Optional[Dict[str, str]] = None
  op: Optional[str] = None

class RtlParser(ParserBase):
  def __init__(self, *, values_mode: str, cycle_min: Optional[int], cycle_max: Optional[int], wait_for_vxdrv_start: bool = True):
    super().__init__(values_mode=values_mode, cycle_min=cycle_min, cycle_max=cycle_max)
    self.wait_for_vxdrv_start = wait_for_vxdrv_start
    self._started = not wait_for_vxdrv_start
    self._instr: Dict[int, InstrRecord] = {}

  def parse_lines(self, lines: Iterable[str]) -> Iterator[Any]:
    for line in lines:
      line = line.rstrip("\n")
      if not line:
        continue

      cfg = parse_configs_line(line)
      if cfg is not None:
        if "num_threads" in cfg:
          self.num_threads = safe_int(cfg["num_threads"])
        continue

      if self.wait_for_vxdrv_start and not self._started:
        if VXDRV_START_RE.match(line):
          self._started = True
        continue

      if line.lstrip().startswith("[VXDRV]"):
        continue

      m_tick = RTL_TICK_RE.match(line)
      if not m_tick:
        continue

      ts = int(m_tick.group(1))
      if not self._keep(ts):
        continue

      body = m_tick.group(2).strip()
      uuid: Optional[int] = None
      m_uuid = UUID_SUFFIX_RE.search(body)
      if m_uuid:
        uuid = int(m_uuid.group(1))
        body = UUID_SUFFIX_RE.sub("", body).rstrip()

      path, action, args_str = extract_body_fields(body)
      if not path or not action:
        continue

      if action.strip() == "warp-state":
        kv = parse_kv_brace_aware(args_str)
        wid = safe_int(kv.get("wid"))
        if wid is None:
          continue
        active = kv.get("active")
        stalled = kv.get("stalled")
        tmask = kv.get("tmask")
        if active is not None:
          active = parse_transition(active)
        if stalled is not None:
          stalled = parse_transition(stalled)
        if tmask is not None:
          tmask = normalize_tmask(parse_transition(tmask), self.num_threads, "rtlsim")
        yield WarpStateEvent(
          ts=ts,
          path=path,
          wid=wid,
          active=safe_int(active) if active is not None else None,
          stalled=safe_int(stalled) if stalled is not None else None,
          tmask=tmask
        )
        continue

      if is_cache_event(path, action):
        args_clean = strip_payload_heavy_fields(args_str)
        kv = parse_kv_brace_aware(args_clean)
        tmask = kv.get("tmask")
        if tmask is not None:
          tmask = normalize_tmask(tmask, self.num_threads, "rtlsim")
        yield CacheEvent(
          ts=ts,
          path=path,
          level=infer_cache_level(path),
          kind=action.strip(),
          uuid=uuid,
          wid=safe_int(kv.get("wid")),
          pc=kv.get("PC") or kv.get("pc"),
          tmask=tmask,
          addr=kv.get("addr"),
          tag=kv.get("tag"),
        )
        continue

      if uuid is None:
        continue

      kv = parse_kv_brace_aware(args_str)
      wid = safe_int(kv.get("wid"))
      pc = kv.get("PC") or kv.get("pc")
      ex = kv.get("ex")
      op = kv.get("op")

      tmask = kv.get("tmask")
      if tmask is not None:
        if self.num_threads is None and all(c in "01" for c in tmask.strip()):
          self.num_threads = len(tmask.strip())
        tmask = normalize_tmask(tmask, self.num_threads, "rtlsim")

      parent_uuid = None
      mp = PARENT_HASH_RE.search(args_str)
      if mp:
        parent_uuid = int(mp.group(1))

      rd = decode_rtl_reg_index(kv["rd"]) if "rd" in kv else None
      rs1 = decode_rtl_reg_index(kv["rs1"]) if "rs1" in kv else None
      rs2 = decode_rtl_reg_index(kv["rs2"]) if "rs2" in kv else None
      rs3 = decode_rtl_reg_index(kv["rs3"]) if "rs3" in kv else None

      rec = self._instr.get(uuid)
      if rec is None:
        rec = InstrRecord()
        self._instr[uuid] = rec
      if parent_uuid is not None:
        rec.parent_uuid = parent_uuid
      if op is not None:
        rec.op = op
      if rd is not None:
        rec.rd = rd
      if rs1 is not None:
        rec.rs1 = rs1
      if rs2 is not None:
        rec.rs2 = rs2
      if rs3 is not None:
        rec.rs3 = rs3

      if self.values_mode == "all":
        for k_src, reg_name in (("rs1_data", rec.rs1), ("rs2_data", rec.rs2), ("rs3_data", rec.rs3)):
          if k_src in kv and reg_name is not None:
            rec.src_values.append({"reg": reg_name, "value": kv[k_src]})

      if self.values_mode in ("dest", "all") and action.strip() == "commit":
        if "data" in kv and rec.rd is not None:
          rec.dest_value = {"reg": rec.rd, "value": kv["data"]}

      stage = self._map_stage(path, action)
      if stage is None:
        yield RawInstantEvent(ts=ts, path=path, name=action.strip(), uuid=uuid, args={"raw": args_str})
        continue

      sop = safe_int(kv.get("sop")) or 1
      eop = safe_int(kv.get("eop")) or 1

      yield InstructionStageEvent(
        ts=ts,
        path=path,
        stage=stage,
        uuid=uuid,
        wid=wid,
        pc=pc,
        ex=ex,
        op=rec.op or op,
        wb=safe_int(kv.get("wb")),
        tmask=tmask,
        sop=sop,
        eop=eop,
        parent_uuid=rec.parent_uuid,
        rd=rec.rd,
        rs1=rec.rs1,
        rs2=rec.rs2,
        rs3=rec.rs3,
        src_values=rec.src_values if rec.src_values else None,
        dest_value=rec.dest_value if rec.dest_value else None,
      )

      if stage == "commit":
        self._instr.pop(uuid, None)

  def _map_stage(self, path: str, action: str) -> Optional[str]:
    a = action.strip()
    p = path
    if p.endswith("-scheduler") and a == "dispatch":
      return "schedule"
    if "-fetch" in p and a in ("req", "rsp", "fetch"):
      return "fetch"
    if p.endswith("-decode") and a == "decode":
      return "decode"
    if "ibuffer" in p and a in ("ibuffer", "decode"):
      return "ibuffer"
    if p.endswith("-dispatcher") and a == "dispatch":
      return "dispatch"
    if "-execute" in p or "-issue" in p:
      return "execute"
    if p.endswith("-commit") and a == "commit":
      return "commit"
    if a == "commit":
      return "commit"
    return None

class SimxParser(ParserBase):
  def __init__(self, *, values_mode: str, cycle_min: Optional[int], cycle_max: Optional[int], wait_for_vxdrv_start: bool = True):
    super().__init__(values_mode=values_mode, cycle_min=cycle_min, cycle_max=cycle_max)
    self.wait_for_vxdrv_start = wait_for_vxdrv_start
    self._started = not wait_for_vxdrv_start
    self._instr: Dict[int, InstrRecord] = {}
    self._open_debug_uuid: Optional[int] = None

  def parse_lines(self, lines: Iterable[str]) -> Iterator[Any]:
    for line in lines:
      line = line.rstrip("\n")
      if not line:
        continue

      cfg = parse_configs_line(line)
      if cfg is not None:
        if "num_threads" in cfg:
          self.num_threads = safe_int(cfg["num_threads"])
        continue

      if self.wait_for_vxdrv_start and not self._started:
        if VXDRV_START_RE.match(line):
          self._started = True
        continue

      m_trace = SIMX_TRACE_RE.match(line)
      if m_trace:
        ts = int(m_trace.group("cycle"))
        if not self._keep(ts):
          continue

        body = m_trace.group("body").strip()
        uuid: Optional[int] = None
        m_uuid = UUID_SUFFIX_RE.search(body)
        if m_uuid:
          uuid = int(m_uuid.group(1))
          body = UUID_SUFFIX_RE.sub("", body).rstrip()

        path, action, args_str = extract_body_fields(body)
        if not path or not action:
          continue

        if is_cache_event(path, action):
          args_clean = strip_payload_heavy_fields(args_str)
          kv = parse_kv_brace_aware(args_clean)
          tmask = kv.get("tmask")
          if tmask is not None:
            if self.num_threads is None and all(c in "01" for c in tmask.strip()):
              self.num_threads = len(tmask.strip())
            tmask = normalize_tmask(tmask, self.num_threads, "simx")
          yield CacheEvent(
            ts=ts,
            path=path,
            level=infer_cache_level(path),
            kind=action.strip(),
            uuid=uuid,
            wid=safe_int(kv.get("wid")),
            pc=kv.get("PC") or kv.get("pc"),
            tmask=tmask,
            addr=kv.get("addr"),
            tag=kv.get("tag"),
          )
          continue

        if uuid is None:
          continue

        kv = parse_kv_brace_aware(args_str)
        wid = safe_int(kv.get("wid"))
        pc = kv.get("PC") or kv.get("pc")
        ex = kv.get("ex")
        op = kv.get("op")

        tmask = kv.get("tmask")
        if tmask is not None:
          if self.num_threads is None and all(c in "01" for c in tmask.strip()):
            self.num_threads = len(tmask.strip())
          tmask = normalize_tmask(tmask, self.num_threads, "simx")

        sop = safe_int(kv.get("sop")) or 1
        eop = safe_int(kv.get("eop")) or 1

        parent_uuid = None
        mp = PARENT_HASH_RE.search(args_str)
        if mp:
          parent_uuid = int(mp.group(1))

        rec = self._instr.get(uuid)
        if rec is None:
          rec = InstrRecord()
          self._instr[uuid] = rec
        if parent_uuid is not None:
          rec.parent_uuid = parent_uuid
        if op is not None:
          rec.op = op

        stage = self._map_stage(path, action)
        if stage is None:
          yield RawInstantEvent(ts=ts, path=path, name=action.strip(), uuid=uuid, args={"raw": args_str})
          continue

        yield InstructionStageEvent(
          ts=ts,
          path=path,
          stage=stage,
          uuid=uuid,
          wid=wid,
          pc=pc,
          ex=ex,
          op=rec.op or op,
          wb=safe_int(kv.get("wb")),
          tmask=tmask,
          sop=sop,
          eop=eop,
          parent_uuid=rec.parent_uuid,
          rd=rec.rd,
          rs1=rec.rs1,
          rs2=rec.rs2,
          rs3=rec.rs3,
          src_values=rec.src_values if rec.src_values else None,
          dest_value=rec.dest_value if rec.dest_value else None,
        )

        if stage == "commit":
          self._instr.pop(uuid, None)
        continue

      m_dbg = SIMX_DEBUG_RE.match(line)
      if m_dbg:
        body = m_dbg.group("body").strip()
        uuid: Optional[int] = None
        m_uuid = UUID_SUFFIX_RE.search(body)
        if m_uuid:
          uuid = int(m_uuid.group(1))
          body = UUID_SUFFIX_RE.sub("", body).rstrip()
        else:
          uuid = self._open_debug_uuid
        if uuid is None:
          continue
        self._parse_debug(body, uuid)
        continue

  def _map_stage(self, path: str, action: str) -> Optional[str]:
    a = action.strip().lower()
    if a in STAGES_CANON:
      return a
    return None

  def _parse_debug(self, body: str, uuid: int) -> None:
    rec = self._instr.get(uuid)
    if rec is None:
      rec = InstrRecord()
      self._instr[uuid] = rec

    if body.startswith("Instr:"):
      self._open_debug_uuid = uuid
      s = body[len("Instr:"):].strip()

      mp = PARENT_HASH_RE.search(s)
      if mp:
        rec.parent_uuid = int(mp.group(1))

      mt = re.search(r"\btmask=([01]+)\b", s)
      if mt and self.num_threads is None:
        self.num_threads = len(mt.group(1))

      key_match = re.search(r"\b(cid|wid|tmask|PC|parent)=\b", s)
      asm_part = s[:key_match.start()].strip() if key_match else s.strip()

      if asm_part:
        parts = asm_part.split(None, 1)
        mnemonic = parts[0]
        rec.op = mnemonic

        operands = parts[1] if len(parts) > 1 else ""
        ops = [o.strip() for o in operands.split(",") if o.strip()]
        reg_ops = [o for o in ops if REG_NAME_RE.match(o)]
        if len(reg_ops) >= 1:
          rec.rd = reg_ops[0]
        if len(reg_ops) >= 2:
          rec.rs1 = reg_ops[1]
        if len(reg_ops) >= 3:
          rec.rs2 = reg_ops[2]
        if len(reg_ops) >= 4:
          rec.rs3 = reg_ops[3]
      return

    if body.startswith("Src") and "Reg:" in body:
      if self.values_mode != "all":
        return
      _, rhs = body.split("Reg:", 1)
      rhs = rhs.strip()
      if "=" not in rhs:
        return
      reg, val = rhs.split("=", 1)
      rec.src_values.append({"reg": reg.strip(), "value": val.strip()})
      return

    if body.startswith("Dest Reg:"):
      if self.values_mode not in ("dest", "all"):
        return
      rhs = body[len("Dest Reg:"):].strip()
      if "=" not in rhs:
        return
      reg, val = rhs.split("=", 1)
      reg = reg.strip()
      val = val.strip()
      rec.dest_value = {"reg": reg, "value": val}
      rec.rd = rec.rd or reg
      return

# -----------------------------
# Perfetto / Chrome Trace writer
# -----------------------------

class TraceWriter:
  def __init__(self, out_f: io.TextIOBase):
    self.out_f = out_f
    self.first = True
    self.out_f.write("[")

  def emit(self, evt: Dict[str, Any]) -> None:
    s = json.dumps(evt, separators=(",", ":"), ensure_ascii=False)
    if self.first:
      self.first = False
      self.out_f.write(s)
    else:
      self.out_f.write("," + s)

  def close(self) -> None:
    self.out_f.write("]\n")
    self.out_f.flush()

@dataclass
class TrackAllocator:
  pid: int
  writer: TraceWriter
  _next_tid: int = 100
  _tid_by_key: Dict[str, int] = field(default_factory=dict)
  _named: set = field(default_factory=set)

  def tid(self, key: str, name: str) -> int:
    if key in self._tid_by_key:
      return self._tid_by_key[key]
    tid = self._next_tid
    self._next_tid += 1
    self._tid_by_key[key] = tid
    self._emit_thread_name(tid, name)
    return tid

  def _emit_thread_name(self, tid: int, name: str) -> None:
    if tid in self._named:
      return
    self._named.add(tid)
    self.writer.emit({
      "ph": "M",
      "pid": self.pid,
      "tid": tid,
      "name": "thread_name",
      "args": {"name": name}
    })

def extract_hierarchy(path: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
  mc = re.search(r"(cluster\d+)", path)
  ms = re.search(r"(socket\d+)", path)
  mcore = re.search(r"(core\d+)", path)
  return (mc.group(1) if mc else None,
          ms.group(1) if ms else None,
          mcore.group(1) if mcore else None)

def track_name_for(path: str, wid: Optional[int], kind: str) -> Tuple[str, str]:
  cluster, socket, core = extract_hierarchy(path)
  base = "-".join([x for x in (cluster, socket, core) if x]) or "global"
  if kind == "warp" and wid is not None:
    return f"{base}/warp{wid}", f"{base}: warp{wid}"
  if kind == "warpstate" and wid is not None:
    return f"{base}/warp{wid}/state", f"{base}: warp{wid} state"
  if kind == "mem":
    lvl = infer_cache_level(path)
    return f"{base}/{lvl}", f"{base}: {lvl}"
  return f"{base}/{path}", path

@dataclass
class EmitterConfig:
  values_mode: str
  parent_flow: bool
  freq_mhz: Optional[float] = None
  cycle_ns: Optional[float] = None

def ts_to_us(ts: int, cfg: EmitterConfig) -> float:
  if cfg.cycle_ns is not None:
    return (cfg.cycle_ns * ts) / 1000.0
  if cfg.freq_mhz is not None and cfg.freq_mhz > 0:
    return ts / cfg.freq_mhz
  return float(ts)

class PerfettoEmitter:
  def __init__(self, writer: TraceWriter, cfg: EmitterConfig):
    self.writer = writer
    self.cfg = cfg
    self.pid = 1
    self.tracks = TrackAllocator(pid=self.pid, writer=writer)
    self._open_inst: Dict[int, Tuple[float, int]] = {}
    self._last_ts_us: float = 0.0

    self.writer.emit({"ph": "M", "pid": self.pid, "name": "process_name", "args": {"name": "Vortex GPU 1"}})
    self.writer.emit({"ph": "M", "pid": self.pid, "name": "process_sort_index", "args": {"sort_index": 1}})

  def finalize(self) -> None:
    end_ts = self._last_ts_us
    for uuid, (beg, tid) in list(self._open_inst.items()):
      self._emit_async_end(end_ts, tid, "inst", "vortex.inst", uuid, {"uuid": uuid, "incomplete": 1})
      self._open_inst.pop(uuid, None)

  def emit_event(self, ev: Any) -> None:
    if isinstance(ev, InstructionStageEvent):
      self._emit_instruction_stage(ev)
      return
    if isinstance(ev, WarpStateEvent):
      self._emit_warp_state(ev)
      return
    if isinstance(ev, CacheEvent):
      self._emit_cache(ev)
      return
    if isinstance(ev, RawInstantEvent):
      self._emit_raw(ev)
      return

  def _emit_instruction_stage(self, ev: InstructionStageEvent) -> None:
    ts_us = ts_to_us(ev.ts, self.cfg)
    self._last_ts_us = max(self._last_ts_us, ts_us)

    kind = "warp" if ev.wid is not None else "module"
    tkey, tname = track_name_for(ev.path, ev.wid, kind)
    tid = self.tracks.tid("inst:" + tkey, tname)

    if ev.uuid not in self._open_inst:
      self._open_inst[ev.uuid] = (ts_us, tid)
      self._emit_async_begin(ts_us, tid, "inst", "vortex.inst", ev.uuid, {"uuid": ev.uuid})

    args: Dict[str, Any] = {"uuid": ev.uuid, "path": ev.path, "sop": ev.sop, "eop": ev.eop}
    if ev.parent_uuid is not None:
      args["parent_uuid"] = ev.parent_uuid
    if ev.wid is not None:
      args["wid"] = ev.wid
    if ev.pc is not None:
      args["PC"] = ev.pc
    if ev.ex is not None:
      args["ex"] = ev.ex
    if ev.op is not None:
      args["op"] = ev.op
    if ev.wb is not None:
      args["wb"] = ev.wb
    if ev.tmask is not None:
      args["tmask"] = ev.tmask
    if ev.rd is not None:
      args["rd"] = ev.rd
    if ev.rs1 is not None:
      args["rs1"] = ev.rs1
    if ev.rs2 is not None:
      args["rs2"] = ev.rs2
    if ev.rs3 is not None:
      args["rs3"] = ev.rs3

    if self.cfg.values_mode == "all" and ev.stage in ("dispatch", "operands") and ev.src_values:
      args["src_values"] = ev.src_values

    if ev.stage == "commit":
      if self.cfg.values_mode in ("dest", "all") and ev.dest_value:
        # required: only commit marker has dest_value
        args["dest_value"] = ev.dest_value
      self._emit_async_end(ts_us, tid, "inst", "vortex.inst", ev.uuid, {"uuid": ev.uuid})
      self._open_inst.pop(ev.uuid, None)

    self._emit_instant(ts_us, tid, ev.stage, "vortex.stage", args)

    if self.cfg.parent_flow and ev.parent_uuid is not None and ev.parent_uuid != ev.uuid:
      flow_id = (ev.parent_uuid * 1315423911 + ev.uuid) & 0x7fffffff
      self.writer.emit({"ph": "s", "pid": self.pid, "tid": tid, "ts": ts_us, "name": "parent", "id": flow_id})
      self.writer.emit({"ph": "f", "pid": self.pid, "tid": tid, "ts": ts_us, "name": "parent", "id": flow_id})

  def _emit_warp_state(self, ev: WarpStateEvent) -> None:
    ts_us = ts_to_us(ev.ts, self.cfg)
    self._last_ts_us = max(self._last_ts_us, ts_us)
    tkey, tname = track_name_for(ev.path, ev.wid, "warpstate")
    tid = self.tracks.tid("wstate:" + tkey, tname)
    if ev.active is not None:
      self._emit_counter(ts_us, tid, "active", ev.active)
    if ev.stalled is not None:
      self._emit_counter(ts_us, tid, "stalled", ev.stalled)
    if ev.tmask is not None and all(c in "01" for c in ev.tmask):
      self._emit_counter(ts_us, tid, "active_threads", ev.tmask.count("1"))

  def _emit_cache(self, ev: CacheEvent) -> None:
    ts_us = ts_to_us(ev.ts, self.cfg)
    self._last_ts_us = max(self._last_ts_us, ts_us)
    tkey, tname = track_name_for(ev.path, None, "mem")
    tid = self.tracks.tid("mem:" + tkey, tname)
    args: Dict[str, Any] = {"path": ev.path, "level": ev.level, "kind": ev.kind}
    if ev.uuid is not None:
      args["uuid"] = ev.uuid
    if ev.wid is not None:
      args["wid"] = ev.wid
    if ev.pc is not None:
      args["PC"] = ev.pc
    if ev.tmask is not None:
      args["tmask"] = ev.tmask
    if ev.addr is not None:
      args["addr"] = ev.addr
    if ev.tag is not None:
      args["tag"] = ev.tag
    self._emit_instant(ts_us, tid, f"{ev.level}:{ev.kind}", "vortex.mem", args)

  def _emit_raw(self, ev: RawInstantEvent) -> None:
    ts_us = ts_to_us(ev.ts, self.cfg)
    self._last_ts_us = max(self._last_ts_us, ts_us)
    tkey, tname = track_name_for(ev.path, None, "module")
    tid = self.tracks.tid("raw:" + tkey, tname)
    args = dict(ev.args)
    if ev.uuid is not None:
      args["uuid"] = ev.uuid
    self._emit_instant(ts_us, tid, ev.name, "vortex.raw", args)

  def _emit_instant(self, ts_us: float, tid: int, name: str, cat: str, args: Dict[str, Any]) -> None:
    self.writer.emit({"ph": "i", "pid": self.pid, "tid": tid, "ts": ts_us, "s": "t", "name": name, "cat": cat, "args": args})

  def _emit_counter(self, ts_us: float, tid: int, name: str, value: Any) -> None:
    self.writer.emit({"ph": "C", "pid": self.pid, "tid": tid, "ts": ts_us, "name": name, "args": {"value": value}})

  def _emit_async_begin(self, ts_us: float, tid: int, name: str, cat: str, async_id: Any, args: Dict[str, Any]) -> None:
    self.writer.emit({"ph": "b", "pid": self.pid, "tid": tid, "ts": ts_us, "name": name, "cat": cat, "id": async_id, "args": args})

  def _emit_async_end(self, ts_us: float, tid: int, name: str, cat: str, async_id: Any, args: Dict[str, Any]) -> None:
    self.writer.emit({"ph": "e", "pid": self.pid, "tid": tid, "ts": ts_us, "name": name, "cat": cat, "id": async_id, "args": args})

# -----------------------------
# main
# -----------------------------

def parse_args() -> argparse.Namespace:
  ap = argparse.ArgumentParser(description="Convert Vortex RTL/SimX traces to Perfetto (Chrome Trace JSON).")
  ap.add_argument("input", help="Input log file.")
  ap.add_argument("-o", "--output", default=None, help="Output trace file (.json or .json.gz).")
  ap.add_argument("-c", "--compress", action="store_true", help="Gzip-compress output (.json.gz).")
  ap.add_argument("-t", "--type", default="auto", choices=["auto", "rtlsim", "simx"],
                  help="Log type: auto-detect (default), rtlsim, or simx.")
  ap.add_argument("--values", default="dest", choices=["none", "dest", "all"],
                  help="Capture values: none, dest (dest only at commit), all (src at dispatch/operands + dest at commit). Default: dest")
  ap.add_argument("--freq-mhz", type=float, default=None, help="Device frequency in MHz (maps cycles/ticks to microseconds).")
  ap.add_argument("--cycle-ns", type=float, default=None, help="Cycle period in ns (maps cycles/ticks to microseconds). Overrides --freq-mhz.")
  ap.add_argument("--cycle-min", type=int, default=None, help="Only include events with time >= this.")
  ap.add_argument("--cycle-max", type=int, default=None, help="Only include events with time <= this.")
  ap.add_argument("--no-vxdrv-start", action="store_true", help="RTL only: do not wait for '[VXDRV] START:' to begin parsing.")
  ap.add_argument("--parent-flow", action="store_true", help="Emit flow arrows for parent->uop when parent=#... is present. Default: off.")
  return ap.parse_args()

def _detect_log_type(peek_lines: List[str]) -> str:
  """Best-effort log type detection from the first handful of lines."""
  simx_score = 0
  rtl_score = 0

  for ln in peek_lines:
    if SIMX_TRACE_RE.match(ln) or SIMX_DEBUG_RE.match(ln):
      simx_score += 3
    # RTL logs: lines like "  37: cluster0-..." and driver markers.
    if RTL_TICK_RE.match(ln):
      rtl_score += 1
    if "[VXDRV]" in ln or "CONFIGS:" in ln:
      rtl_score += 2
    # Extra hint: module-path style is common in RTL logs.
    if "-scheduler" in ln or "-decode" in ln or "-commit" in ln:
      rtl_score += 1

  if rtl_score == 0 and simx_score == 0:
    # Unknown; keep the old default of simx.
    return "simx"
  return "rtlsim" if rtl_score >= simx_score else "simx"

def open_output(path: Path, compress: bool) -> io.TextIOBase:
  if compress or str(path).endswith(".gz"):
    return gzip.open(path, "wt", encoding="utf-8")
  return open(path, "w", encoding="utf-8")

def make_parser(args: argparse.Namespace, detected_type: Optional[str] = None) -> ParserBase:
  ptype = args.type
  if ptype == "auto":
    ptype = detected_type or "simx"

  if ptype == "rtlsim":
    return RtlParser(values_mode=args.values, cycle_min=args.cycle_min, cycle_max=args.cycle_max,
                     wait_for_vxdrv_start=not args.no_vxdrv_start)
  return SimxParser(values_mode=args.values, cycle_min=args.cycle_min, cycle_max=args.cycle_max,
                     wait_for_vxdrv_start=not args.no_vxdrv_start)

def run_export(parser: ParserBase, emitter: PerfettoEmitter, lines: Iterable[str]) -> int:
  n = 0
  for ev in parser.parse_lines(lines):
    emitter.emit_event(ev)
    n += 1
  emitter.finalize()
  return n

def main() -> int:
  args = parse_args()
  in_path = Path(args.input)
  if not in_path.exists():
    print(f"error: input not found: {in_path}", file=sys.stderr)
    return 2

  out_path = Path(args.output) if args.output else None
  if out_path is None:
    suffix = ".perfetto.json.gz" if args.compress else ".perfetto.json"
    out_path = in_path.with_suffix(in_path.suffix + suffix)

  out_f = open_output(out_path, args.compress)
  try:
    writer = TraceWriter(out_f)
    cfg = EmitterConfig(values_mode=args.values, parent_flow=args.parent_flow, freq_mhz=args.freq_mhz, cycle_ns=args.cycle_ns)
    emitter = PerfettoEmitter(writer, cfg)
    # Auto-detect log type from the head of the file (so we don't accidentally use the wrong file).
    with open(in_path, "r", encoding="utf-8", errors="ignore") as f:
      peek: List[str] = []
      for _ in range(2000):
        ln = f.readline()
        if not ln:
          break
        peek.append(ln)
      detected = _detect_log_type(peek)
      parser = make_parser(args, detected_type=detected)
      n_events = run_export(parser, emitter, chain(peek, f))

    if n_events == 0 and args.type in ("auto", "simx", "rtlsim"):
      print(
        "warning: produced 0 events; if this is an RTL log, try: -t rtlsim (or omit -t for auto-detect)",
        file=sys.stderr,
      )
    writer.close()
  finally:
    out_f.close()

  print(str(out_path))
  return 0

if __name__ == "__main__":
  raise SystemExit(main())
