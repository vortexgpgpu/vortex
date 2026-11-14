#!/usr/bin/env python3
# Copyright © 2019-2023
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RAM fold: block RAM for asynchronous reads, as a patch of the sources.

A block RAM reads synchronously. A VX_dp_ram of block-RAM size that reads
combinationally (OUT_REG=0) becomes one when its read address is computed one
cycle early: the RAM then registers that next address itself (OUT_REG=1,
RDW_MODE "W"), which reads exactly what the combinational read did on every
cycle (docs/hardware_library.md, read forms).

The next address is the address logic evaluated on the next values of the
registers that feed it, wherever in the hierarchy they are. The pass elaborates
the design with slang, the front end CIRCT uses, follows the address back to
those registers, and patches the sources it was given:

  - every signal on that cone gets a twin <name>__nx carrying its next value,
    driven by a copy of the signal's own driver with its operands renamed;
  - a register's twin is a combinational copy of its always_ff block that keeps
    only the assignments to that register (their right-hand sides still read
    current values, as non-blocking assignments do);
  - a port on the cone gets a twin port, connected at every instantiation;
  - an interface signal gets a twin member;
  - the RAM takes the twin address and the registered-address form, reading
    every cycle.

Edits only add code, except the RAM's own parameters and address connection.
The patched sources are elaborated again; a patch that adds an error is dropped
whole and every RAM keeps its original, combinational read. Nothing is folded across a
construct the pass does not handle: it reports the RAM and leaves it alone.
"""

import argparse
import difflib
import os
import re
import shlex
import sys

import pyslang
from pyslang import analysis, ast, driver

SFX = "__nx"
SK = ast.SymbolKind
EK = ast.ExpressionKind

RAM_MODULES = ("VX_dp_ram",)
CLOCK_GATE = "VX_clockgate"

CONST_KINDS = {SK.Parameter, SK.TypeParameter, SK.EnumValue, SK.Genvar, SK.Specparam}

PURE_SYSTEM_CALLS = {"$clog2", "$bits", "$signed", "$unsigned", "$countones", "$size",
                     "$high", "$low", "$left", "$right", "$onehot", "$onehot0", "$isunknown"}


class Unsupported(Exception):
    pass


def force_bram(depth, width):
    # `FORCE_BRAM in VX_platform.vh
    return depth >= 64 and depth * width >= 2048


def const_int(v):
    s = str(v)
    m = re.match(r"^(?:\d+)?'s?[dhbo]?([0-9a-fA-F_xXzZ]+)$", s)
    if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
        return int(s)
    if m:
        base = {"d": 10, "h": 16, "b": 2, "o": 8}
        b = re.search(r"'s?([dhbo])", s)
        return int(m.group(1).replace("_", ""), base[b.group(1)] if b else 10)
    if s.startswith('"'):
        return s
    raise ValueError(s)


class Sources:
    """File contents and text edits, by byte offset."""

    def __init__(self, sm):
        self.sm = sm
        self.data = {}

    def file(self, loc):
        return os.path.realpath(self.sm.getFullPath(loc.buffer))

    def bytes(self, path):
        if path not in self.data:
            with open(path, "rb") as f:
                self.data[path] = f.read()
        return self.data[path]

    def span(self, rng):
        path = self.file(rng.start)
        return path, rng.start.offset, rng.end.offset

    def text(self, rng):
        path, s, e = self.span(rng)
        return self.bytes(path)[s:e].decode()

    def indent(self, path, offset):
        b = self.bytes(path)
        line = b.rfind(b"\n", 0, offset) + 1
        m = re.match(rb"[ \t]*", b[line:])
        return m.group(0).decode()


class Edits:
    """Keyed insertions and replacements; the same key must carry the same edit."""

    def __init__(self):
        self.items = {}
        self.order = []

    def add(self, key, path, start, end, text):
        e = (path, start, end, text)
        if key in self.items:
            if self.items[key] != e:
                raise Unsupported("conflicting edits at %s:%d" % (os.path.basename(path), start))
            return
        self.items[key] = e
        self.order.append(key)

    def merge(self, other):
        for k in other.order:
            self.add(k, *other.items[k])

    def apply(self, srcs):
        out = {}
        by_file = {}
        for n, k in enumerate(self.order):
            path, s, e, t = self.items[k]
            by_file.setdefault(path, []).append((s, e, n, t))
        for path, lst in by_file.items():
            data = srcs.bytes(path)
            # replacements must not overlap each other or contain insertions
            reps = [(s, e) for s, e, _, _ in lst if e > s]
            for s, e in reps:
                for s2, e2, _, _ in lst:
                    if (s2, e2) != (s, e) and s < s2 < e or (e2 > s2 and s2 < e and s < e2 and (s2, e2) != (s, e)):
                        raise Unsupported("overlapping edits in %s at %d" % (os.path.basename(path), s))
            # apply back to front; at one offset, insertions keep their order
            lst.sort(key=lambda x: (x[0], x[2]))
            res = bytearray()
            pos = 0
            for s, e, _, t in lst:
                res += data[pos:s]
                res += t.encode()
                pos = max(pos, e) if e > s else max(pos, s)
            res += data[pos:]
            out[path] = bytes(res)
        return out


class Design:
    """An elaborated design and its indices."""

    def __init__(self, args, files):
        self.dr = driver.Driver()
        self.dr.addStandardArgs()
        # no error limit: slang stops its driver analysis once the limit is reached
        cmd = ["slang", "--allow-use-before-declare", "--allow-hierarchical-const",
               "--ignore-unknown-modules", "--disable-instance-caching", "--error-limit=0"] + args + files
        if not self.dr.parseCommandLine(shlex.join(cmd), driver.CommandLineOptions()):
            raise SystemExit("ram_fold: bad slang arguments")
        if not (self.dr.processOptions() and self.dr.parseAllSources()):
            raise SystemExit("ram_fold: cannot parse the sources")
        self.comp = self.dr.createCompilation()
        self.root = self.comp.getRoot()
        self.errors = self._errors()
        self.am = analysis.AnalysisManager()
        self.am.analyze(self.comp)
        self.srcs = Sources(self.dr.sourceManager)
        self.parent = {}    # instance path -> parent instance
        self.owner = {}     # value symbol path -> instance
        self.by_path = {}   # value symbol path -> symbol
        self.insts = {}     # instance path -> instance
        self.iface_maps = {}
        self.iface_users = {}   # interface instance path -> [(instance, port)]
        self.rams = []
        self.arrayed = set()   # instances that are elements of an instance array
        # A gated clock is the gate's input clock: every register on one keeps
        # its load enable, so it loads exactly when it would on the free clock.
        self.gated_clk = {}    # net path driven by a VX_clockgate -> its clk_in expression
        self.clk_alias = {}    # net path -> right-hand side of `assign net = value;`
        for top in self.root.topInstances:
            self.parent[top.hierarchicalPath] = None
            self.insts[top.hierarchicalPath] = top
            self._index(top.body, top)


    def _errors(self):
        de = pyslang.DiagnosticEngine(self.dr.sourceManager)
        cl = pyslang.TextDiagnosticClient()
        de.addClient(cl)
        errs = set()
        for d in self.comp.getAllDiagnostics():
            if d.isError():
                cl.clear()
                de.issue(d)
                errs.add(re.sub(r":\d+:\d+:", ":", cl.getString().split("\n")[0]))
        return errs

    def _index(self, scope, inst):
        for m in scope:
            k = m.kind
            if k == SK.Instance:
                self.parent[m.hierarchicalPath] = inst
                self.insts[m.hierarchicalPath] = m
                self._ifaces(m)
                if m.definition.name in RAM_MODULES:
                    self.rams.append(m)
                if m.definition.name == CLOCK_GATE:
                    self._index_clock_gate(m)
                self._index(m.body, m)
            elif k == SK.ContinuousAssign:
                a = m.assignment
                if a.kind == EK.Assignment and a.left.kind == EK.NamedValue:
                    self.clk_alias[a.left.symbol.hierarchicalPath] = a.right
            elif k in (SK.GenerateBlock, SK.GenerateBlockArray):
                if k == SK.GenerateBlock and m.isUninstantiated:
                    continue
                self._index(m, inst)
            elif k in (SK.Net, SK.Variable):
                self.owner[m.hierarchicalPath] = inst
                self.by_path[m.hierarchicalPath] = m
            elif k == SK.InstanceArray:
                self._index_array(m, inst)

    def _index_clock_gate(self, inst):
        conns = {c.port.name: c.expression for c in inst.portConnections if c.port.kind == SK.Port}
        out, src = conns.get("clk_out"), conns.get("clk_in")
        if out is not None and out.kind == EK.Assignment:
            out = out.left
        if out is not None and src is not None and out.kind == EK.NamedValue:
            self.gated_clk[out.symbol.hierarchicalPath] = src

    def _ifaces(self, inst):
        for p in inst.body.portList:
            if p.kind == SK.InterfacePort:
                c = inst.getPortConnection(p)
                if c is not None and c.ifaceConn is not None and c.ifaceConn[0] is not None:
                    self.iface_users.setdefault(c.ifaceConn[0].hierarchicalPath, []).append((inst, p))

    def _index_array(self, arr, inst):
        for e in arr:
            if e.kind == SK.InstanceArray:
                self._index_array(e, inst)
            elif e.kind == SK.Instance:
                self.parent[e.hierarchicalPath] = inst
                self.insts[e.hierarchicalPath] = e
                self.arrayed.add(e.hierarchicalPath)
                self._ifaces(e)
                self._index(e.body, e)

    def param(self, inst, name):
        for p in inst.body.parameters:
            if p.name == name:
                return p.value
        raise KeyError(name)

    def port(self, inst, name):
        for p in inst.body.portList:
            if p.name == name:
                return p
        raise KeyError(name)

    def port_of(self, inst, sym):
        for p in inst.body.portList:
            if p.kind == SK.Port and p.internalSymbol is not None and \
                    p.internalSymbol.hierarchicalPath == sym.hierarchicalPath:
                return p
        return None


def leaf_name_span(srcs, node, name):
    """The byte span of the last identifier of a value reference."""
    path, s, e = srcs.span(node.sourceRange)
    txt = srcs.bytes(path)[s:e].decode()
    m = None
    for m in re.finditer(r"\b%s\b" % re.escape(name), txt):
        pass
    if m is None:
        raise Unsupported("cannot locate '%s' in '%s'" % (name, txt))
    return path, s + m.start(), s + m.end()


def chain_of(node):
    """(root value expression, selector path) of a select/member chain."""
    path = []
    while True:
        k = node.kind
        if k == EK.ElementSelect:
            c = node.selector.constant
            path.append(("idx", str(c) if c is not None else None))
            node = node.value
        elif k == EK.RangeSelect:
            path.append(("rng", None))
            node = node.value
        elif k == EK.MemberAccess:
            path.append(("mem", node.member.name))
            node = node.value
        elif k == EK.Conversion and node.isImplicit if hasattr(node, "isImplicit") else False:
            node = node.operand
        else:
            break
    if node.kind in (EK.NamedValue, EK.HierarchicalValue):
        return node, tuple(reversed(path))
    return None, ()


def reroot(p, a, b):
    """p re-rooted from the hierarchy under a to the one under b, or None."""
    if p == a or p.startswith(a + ".") or p.startswith(a + "["):
        return b + p[len(a):]
    return None


def path_text(path):
    return "".join("[%s]" % (v if v is not None else "*") if k != "mem" else "." + v for k, v in path)


def compatible(a, b):
    for x, y in zip(a, b):
        if x[0] != y[0]:
            return True
        if x[0] == "mem" and x[1] != y[1]:
            return False
        if x[0] == "idx" and x[1] is not None and y[1] is not None and x[1] != y[1]:
            return False
    return True


class Fold:
    """Folds one RAM; its edits are merged only if it succeeds."""

    def __init__(self, design):
        self.d = design
        self.srcs = design.srcs
        self.edits = Edits()
        self.declared = set()
        self.done = set()

    # ---- canonical bodies -------------------------------------------------
    #
    # slang analyzes one body per set of identically parameterized instances, so
    # the drivers of a signal in any other instance are found on its counterpart
    # in that canonical body, and whatever those drivers reference is mapped back.

    def canon_path(self, path):
        """(path of the analyzed counterpart of path, mapping back)."""
        tr = []
        for _ in range(64):
            x = self.d.insts.get(path) or self.d.owner.get(path)
            moved = False
            while x is not None:
                cb = x.canonicalBody if not x.isInterface else None
                ci = cb.parentInstance if cb is not None else None
                if ci is not None and ci.hierarchicalPath != x.hierarchicalPath:
                    tr.append((ci.hierarchicalPath, x.hierarchicalPath))
                    path = reroot(path, x.hierarchicalPath, ci.hierarchicalPath)
                    moved = True
                    break
                x = self.d.parent.get(x.hierarchicalPath)
            if not moved:
                break
        return path, tuple(tr)

    def canon(self, sym):
        """(counterpart of sym that carries drivers, mapping back)."""
        path, tr = self.canon_path(sym.hierarchicalPath)
        csym = self.d.by_path.get(path)
        if csym is None:
            raise Unsupported("no canonical counterpart of '%s'" % sym.hierarchicalPath)
        return csym, tr

    def iface_drivers(self, sym, inst):
        """Drivers of an interface signal in the modules connected to it whose
        bodies slang analyzed only as another instance: [(driver, csym, tr)]."""
        rel = sym.hierarchicalPath[len(inst.hierarchicalPath):]
        found = []
        for user, port in self.d.iface_users.get(inst.hierarchicalPath, []):
            upath, tr = self.canon_path(user.hierarchicalPath)
            if not tr:
                continue    # analyzed as itself: its drivers are on sym already
            cuser = self.d.insts.get(upath)
            if cuser is None:
                continue
            cport = next((p for p in cuser.body.portList if p.name == port.name), None)
            c = cuser.getPortConnection(cport) if cport is not None else None
            if c is None or c.ifaceConn is None or c.ifaceConn[0] is None:
                continue
            csym = self.d.by_path.get(c.ifaceConn[0].hierarchicalPath + rel)
            if csym is None:
                continue
            for drv in self.d.am.getDrivers(csym):
                if reroot(drv.containingSymbol.hierarchicalPath, upath, upath) is not None:
                    found.append((drv, csym, tr))
        return found

    def back(self, path, tr):
        for cp, ap in reversed(tr):
            q = reroot(path, cp, ap)
            if q is None:
                # a signal of an interface the shared body reaches through a port
                for ic, ia in self.iface_map(cp, ap):
                    q = reroot(path, ic, ia)
                    if q is not None:
                        break
            if q is None:
                raise Unsupported("a shared instance body reads '%s' outside itself" % path)
            path = q
        return path

    def iface_map(self, cp, ap):
        """(interface instance path, its counterpart) for the interface ports of
        instance cp and its copy ap."""
        if (cp, ap) in self.d.iface_maps:
            return self.d.iface_maps[(cp, ap)]
        ci, ai = self.d.insts.get(cp), self.d.insts.get(ap)
        res = []
        if ci is not None and ai is not None:
            for cport, aport in zip(ci.body.portList, ai.body.portList):
                if cport.kind != SK.InterfacePort:
                    continue
                cc, ac = ci.getPortConnection(cport), ai.getPortConnection(aport)
                if cc is None or ac is None or cc.ifaceConn is None or ac.ifaceConn is None:
                    continue
                c0, a0 = cc.ifaceConn[0], ac.ifaceConn[0]
                if c0 is not None and a0 is not None:
                    res.append((c0.hierarchicalPath, a0.hierarchicalPath))
        self.d.iface_maps[(cp, ap)] = res
        return res

    def act(self, sym, tr):
        if not tr:
            return sym
        s = self.d.by_path.get(self.back(sym.hierarchicalPath, tr))
        if s is None:
            raise Unsupported("no instance counterpart of '%s'" % sym.hierarchicalPath)
        return s

    def act_inst(self, inst, tr):
        if not tr:
            return inst
        i = self.d.insts.get(self.back(inst.hierarchicalPath, tr))
        if i is None:
            raise Unsupported("no instance counterpart of '%s'" % inst.hierarchicalPath)
        return i

    def scope_key(self, sym, tr):
        p = sym.hierarchicalPath
        for cp, ap in reversed(tr):
            p = reroot(p, cp, ap) or p
        return p

    def clock_root(self, expr, tr=()):
        """Where a clock comes from: its net, followed up through input ports."""
        while True:
            while expr.kind == EK.Conversion:
                expr = expr.operand
            if expr.kind != EK.NamedValue:
                return None
            sym = self.act(self.value_symbol(expr), tr)
            tr = ()
            inst = self.d.owner.get(sym.hierarchicalPath)
            port = self.d.port_of(inst, sym) if inst is not None else None
            if port is None or port.direction != ast.ArgumentDirection.In or \
                    self.d.parent.get(inst.hierarchicalPath) is None:
                nxt = self.d.gated_clk.get(sym.hierarchicalPath)
                if nxt is None:
                    nxt = self.d.clk_alias.get(sym.hierarchicalPath)
                if nxt is None:
                    return sym.hierarchicalPath
                expr = nxt
                continue
            conn = inst.getPortConnection(port)
            if conn is None or conn.expression is None:
                return None
            expr = conn.expression

    # ---- references -------------------------------------------------------

    def value_symbol(self, node):
        sym = node.symbol
        if sym.kind == SK.ModportPort:
            sym = sym.internalSymbol
        return sym

    def leaves(self, expr, tr=()):
        """Non-constant value references in expr: (node, symbol, path)."""
        paths = {}
        found = []

        def visit(n):
            if not isinstance(n, ast.Expression):
                return ast.VisitAction.Advance
            k = n.kind
            if k in (EK.ElementSelect, EK.RangeSelect, EK.MemberAccess):
                root, p = chain_of(n)
                if root is not None:
                    key = root.sourceRange.start.offset
                    if key not in paths or len(p) > len(paths[key]):
                        paths[key] = p
            elif k in (EK.NamedValue, EK.HierarchicalValue):
                sym = self.value_symbol(n)
                if sym.kind in CONST_KINDS:
                    return ast.VisitAction.Advance
                if sym.kind not in (SK.Net, SK.Variable):
                    raise Unsupported("reference to %s '%s'" % (sym.kind.name, sym.name))
                found.append((n, self.act(sym, tr)))
            elif k == EK.Call:
                self.check_pure(n)
            elif k == EK.Assignment:
                raise Unsupported("assignment inside an expression")
            return ast.VisitAction.Advance

        expr.visit(visit)
        return [(n, s, paths.get(n.sourceRange.start.offset, ())) for n, s in found]

    def check_pure(self, call):
        sub = call.subroutine
        name = call.subroutineName
        if call.isSystemCall:
            if name not in PURE_SYSTEM_CALLS:
                raise Unsupported("system call %s" % name)
            return
        if sub.kind != SK.Subroutine:
            raise Unsupported("call to %s" % name)

        def visit(n):
            if isinstance(n, ast.Expression) and n.kind in (EK.NamedValue, EK.HierarchicalValue):
                s = n.symbol
                if s.kind in (SK.Net, SK.Variable) and not self.inside(s, sub):
                    raise Unsupported("function %s reads '%s'" % (name, s.name))
            return ast.VisitAction.Advance

        sub.visit(visit)

    @staticmethod
    def inside(sym, scope_sym):
        p = sym.lexicalPath if hasattr(sym, "lexicalPath") else ""
        return p.startswith(scope_sym.lexicalPath) if hasattr(scope_sym, "lexicalPath") else False

    def nx_expr(self, expr, extra=(), tr=()):
        """The source text of expr evaluated on next values."""
        path, s, e = self.srcs.span(expr.sourceRange)
        reps = []
        for node, sym, p in self.leaves(expr, tr):
            if extra and node.sourceRange.start.offset == s:
                p = p + tuple(extra)
            name = self.twin(sym, p)
            fp, ls, le = leaf_name_span(self.srcs, node, sym.name)
            if fp != path:
                raise Unsupported("expression spans files")
            reps.append((ls, le, name))
        return self.substitute(path, s, e, reps)

    def substitute(self, path, s, e, reps):
        data = self.srcs.bytes(path)
        out, pos = [], s
        for ls, le, t in sorted(set(reps)):
            out.append(data[pos:ls].decode())
            out.append(t)
            pos = le
        out.append(data[pos:e].decode())
        return "".join(out)

    def rename_root(self, lhs, tr=()):
        """The text of an lvalue with its root signal renamed to its twin."""
        root, _ = chain_of(lhs)
        if root is None:
            raise Unsupported("lvalue '%s'" % self.srcs.text(lhs.sourceRange))
        for n, _, _ in self.leaves(lhs, tr):
            if n.sourceRange.start.offset != root.sourceRange.start.offset:
                raise Unsupported("lvalue with a variable index '%s'" % self.srcs.text(lhs.sourceRange))
        sym = self.value_symbol(root)
        path, s, e = self.srcs.span(lhs.sourceRange)
        _, ls, le = leaf_name_span(self.srcs, root, sym.name)
        return self.substitute(path, s, e, [(ls, le, sym.name + SFX)]), sym

    # ---- twins ------------------------------------------------------------

    def twin(self, sym, path=()):
        key = sym.hierarchicalPath
        name = sym.name + SFX
        inst = self.d.owner.get(key)
        if inst is None:
            raise Unsupported("no owner for '%s'" % key)
        port = self.d.port_of(inst, sym) if inst.isModule or inst.isInterface else None
        if key not in self.declared:
            self.declared.add(key)
            self.declare(sym, inst, port)
        if port is not None and port.direction == ast.ArgumentDirection.In:
            if ("in", key, path) not in self.done:
                self.done.add(("in", key, path))
                self.input_port(sym, inst, port, path)
            return name
        if sym.kind == SK.Net and sym.initializer is not None:
            return name     # driven by its declaration, which declare() copied
        # every part read must be driven: a twin read without a driver would be
        # undriven logic that still elaborates
        csym, tr = self.canon(sym)
        drivers = [(drv, csym, tr) for drv in self.d.am.getDrivers(csym)]
        if inst.isInterface:
            drivers += self.iface_drivers(sym, inst)
        if not drivers:
            raise Unsupported("'%s' is undriven" % sym.hierarchicalPath)
        blocks = {(self.scope_key(drv.containingSymbol, t), drv.containingSymbol.syntax.sourceRange.start.offset)
                  for drv, c, t in drivers if drv.containingSymbol.kind == SK.ProceduralBlock}
        if len(blocks) > 1:
            raise Unsupported("'%s' is driven by %d procedural blocks" % (sym.hierarchicalPath, len(blocks)))
        if not any([self.driver(sym, inst, c, drv, path, t) for drv, c, t in drivers]):
            raise Unsupported("no driver of '%s'%s" % (sym.hierarchicalPath, path_text(path)))
        return name

    def declare(self, sym, inst, port):
        name = sym.name + SFX
        if port is not None:
            self.declare_port(port, name)
            return
        decl = sym.syntax
        while decl is not None and decl.kind not in (pyslang.syntax.SyntaxKind.DataDeclaration,
                                                     pyslang.syntax.SyntaxKind.NetDeclaration):
            decl = decl.parent
        if decl is None:
            raise Unsupported("no declaration for '%s'" % sym.name)
        path, ds, de = self.srcs.span(decl.sourceRange)
        _, vs, ve = self.srcs.span(sym.syntax.sourceRange)
        data = self.srcs.bytes(path)
        prefix = data[ds:self.srcs.span(decl.declarators[0].sourceRange)[1]].decode()
        dims = "".join(str(x) for x in sym.syntax.dimensions).strip()
        init = ""
        if sym.kind == SK.Net and sym.initializer is not None:
            init = " = " + self.nx_expr(sym.initializer)
        ind = self.srcs.indent(path, ds)
        text = "\n%s%s%s%s%s;" % (ind, prefix.strip() + " ", name, (" " + dims) if dims else "", init)
        self.edits.add(("decl", path, de, name), path, de, de, text)
        if inst.isInterface:
            self.modports(sym, inst, name)

    def declare_port(self, port, name):
        syn = port.syntax
        if syn is not None and syn.kind == pyslang.syntax.SyntaxKind.Declarator:
            syn = syn.parent
        if syn is None or syn.kind != pyslang.syntax.SyntaxKind.ImplicitAnsiPort:
            raise Unsupported("port '%s' is not an ANSI port" % port.name)
        header = syn
        while header is not None and not str(header.header).strip():
            header = self.prev_port(header)
        if header is None:
            raise Unsupported("port '%s' has no header" % port.name)
        dims = "".join(str(x) for x in syn.declarator.dimensions).strip()
        # appended after the last port, so an instantiation that connects by
        # position elsewhere keeps its ports
        ports = [p for p in syn.parent.ports if isinstance(p, pyslang.syntax.SyntaxNode)]
        path, ps, _ = self.srcs.span(syn.sourceRange)
        _, _, pe = self.srcs.span(ports[-1].sourceRange)
        ind = self.srcs.indent(path, ps)
        text = ",\n%s%s %s%s" % (ind, str(header.header).strip(), name, (" " + dims) if dims else "")
        self.edits.add(("port", path, pe, name), path, pe, pe, text)

    @staticmethod
    def prev_port(syn):
        lst = syn.parent.ports if hasattr(syn.parent, "ports") else syn.parent
        prev = None
        for item in lst:
            if item is syn or (hasattr(item, "sourceRange") and item.sourceRange.start.offset == syn.sourceRange.start.offset):
                return prev
            if isinstance(item, pyslang.syntax.SyntaxNode) and item.kind == pyslang.syntax.SyntaxKind.ImplicitAnsiPort:
                prev = item
        return None

    def modports(self, sym, inst, name):
        for m in inst.body:
            if m.kind != SK.Modport:
                continue
            for p in m:
                if p.name == sym.name:
                    path, s, e = self.srcs.span(p.syntax.sourceRange)
                    self.edits.add(("modport", path, e, name), path, e, e, ", " + name)

    def input_port(self, sym, inst, port, path):
        parent = self.d.parent.get(inst.hierarchicalPath)
        if parent is None:
            raise Unsupported("'%s' is a top-level input" % sym.hierarchicalPath)
        conn = inst.getPortConnection(port)
        if conn is None or conn.expression is None:
            raise Unsupported("input '%s' of %s is unconnected" % (port.name, inst.hierarchicalPath))
        text = self.nx_expr(conn.expression, path)
        self.connect(inst, port.name + SFX, text)

    def connect(self, inst, port_name, text):
        if inst.hierarchicalPath in self.d.arrayed:
            raise Unsupported("instance array %s on the address cone" % inst.hierarchicalPath)
        syn = inst.syntax
        conns = [c for c in syn.connections if isinstance(c, pyslang.syntax.SyntaxNode)]
        if not conns or any(c.kind != pyslang.syntax.SyntaxKind.NamedPortConnection for c in conns):
            raise Unsupported("instance %s does not connect its ports by name" % inst.name)
        path, s, e = self.srcs.span(conns[-1].sourceRange)
        ind = self.srcs.indent(path, s)
        self.edits.add(("conn", path, e, port_name), path, e, e,
                       ",\n%s.%s (%s)" % (ind, port_name, text))

    def driver(self, sym, inst, csym, drv, path, tr):
        cs = drv.containingSymbol
        k = cs.kind
        if k == SK.ContinuousAssign:
            lhs, rhs = cs.assignment.left, cs.assignment.right
            if lhs.kind == EK.Concatenation:
                return self.concat_assign(sym, csym, cs, lhs, rhs, path, tr)
            root, lpath = chain_of(lhs)
            if root is None:
                raise Unsupported("assign to '%s'" % self.srcs.text(lhs.sourceRange))
            if not compatible(lpath, path):
                return False
            key = ("assign", self.scope_key(cs, tr), cs.syntax.sourceRange.start.offset,
                   self.srcs.file(cs.syntax.sourceRange.start))
            if key in self.done:
                return True
            self.done.add(key)
            ltxt, _ = self.rename_root(lhs, tr)
            rtxt = self.nx_expr(rhs, (), tr)
            stmt = cs.syntax.parent
            while stmt is not None and stmt.kind != pyslang.syntax.SyntaxKind.ContinuousAssign:
                stmt = stmt.parent
            fpath, ss, se = self.srcs.span(stmt.sourceRange)
            ind = self.srcs.indent(fpath, ss)
            self.edits.add(("assign", fpath, se, ltxt), fpath, se, se,
                           "\n%sassign %s = %s;" % (ind, ltxt, rtxt))
            return True
        elif k == SK.ProceduralBlock:
            self.register(sym, csym, cs, tr)
            return True
        elif k == SK.Instance:
            return self.child_output(sym, inst, self.act_inst(cs, tr), path)
        else:
            raise Unsupported("'%s' is driven by a %s" % (sym.name, k.name))

    def concat_assign(self, sym, csym, cs, lhs, rhs, path, tr):
        """assign {.., sym, ..} = e: the twin of e, sliced."""
        ops = list(lhs.operands)
        mine = []
        for i, op in enumerate(ops):
            root, p = chain_of(op)
            if root is None:
                raise Unsupported("assign to '%s'" % self.srcs.text(lhs.sourceRange))
            if self.value_symbol(root).hierarchicalPath == csym.hierarchicalPath and compatible(p, path):
                mine.append(i)
        if not mine:
            return False
        key = ("cassign", self.scope_key(cs, tr), cs.syntax.sourceRange.start.offset,
               self.srcs.file(cs.syntax.sourceRange.start), sym.hierarchicalPath)
        if key in self.done:
            return True
        self.done.add(key)
        stmt = cs.syntax.parent
        while stmt is not None and stmt.kind != pyslang.syntax.SyntaxKind.ContinuousAssign:
            stmt = stmt.parent
        fpath, ss, se = self.srcs.span(stmt.sourceRange)
        ind = self.srcs.indent(fpath, ss)
        whole = self.srcs.text(lhs.sourceRange)
        tname = "%s_cat%d%s" % (sym.name, cs.syntax.sourceRange.start.offset, SFX)
        rtxt = self.nx_expr(rhs, (), tr)
        self.edits.add(("catw", fpath, se, tname), fpath, se, se,
                       "\n%swire [$bits(%s)-1:0] %s = %s;" % (ind, whole, tname, rtxt))
        for i in mine:
            later = [self.srcs.text(o.sourceRange) for o in ops[i + 1:]]
            off = "$bits({%s})" % ", ".join(later) if later else "0"
            ltxt, _ = self.rename_root(ops[i], tr)
            self.edits.add(("cats", fpath, se, ltxt), fpath, se, se,
                           "\n%sassign %s = %s[%s +: $bits(%s)];" % (ind, ltxt, tname, off, self.srcs.text(ops[i].sourceRange)))
        return True

    # ---- registers --------------------------------------------------------

    def register(self, sym, csym, block, tr):
        key = ("reg", self.scope_key(block, tr), block.syntax.sourceRange.start.offset,
               self.srcs.file(block.syntax.sourceRange.start), sym.name)
        if key in self.done:
            return
        self.done.add(key)
        pk = block.procedureKind
        if pk not in (ast.ProceduralBlockKind.AlwaysFF, ast.ProceduralBlockKind.Always):
            raise Unsupported("'%s' is driven by an %s block" % (sym.name, pk.name))
        timed = block.body
        if timed.kind != ast.StatementKind.Timed:
            raise Unsupported("'%s' is driven by an untimed block" % sym.name)
        tc = self.srcs.text(timed.timing.sourceRange)
        if not re.fullmatch(r"(@\s*\(\s*)?posedge\s+\w+(\s*\))?", tc.strip()):
            raise Unsupported("'%s' is clocked by '%s'" % (sym.name, tc))
        ev = getattr(timed.timing, "expr", None)
        clk = self.clock_root(ev, tr) if ev is not None else None
        if clk is None or clk != self.clock:
            raise Unsupported("'%s' is clocked by '%s', not the RAM's clock" % (sym.hierarchicalPath, clk or tc))
        body = timed.stmt
        path, bs, be = self.srcs.span(body.sourceRange)
        reps = []
        self.keep_only(csym, body, reps)
        btxt = self.substitute(path, bs, be, reps)
        name = sym.name + SFX
        bpath, s, e = self.srcs.span(block.syntax.sourceRange)
        ind = self.srcs.indent(bpath, s)
        body_ind = "\n".join(("    " + l) if l.strip() else l for l in btxt.split("\n"))
        text = "\n%salways @(*) begin\n%s    %s = %s;\n%s    %s\n%send" % (ind, ind, name, sym.name, ind, body_ind.lstrip(), ind)
        self.edits.add(key[2:], bpath, e, e, text)

    def keep_only(self, sym, st, reps):
        """Rewrite a clocked statement to compute sym's twin and nothing else."""
        k = st.kind
        SKd = ast.StatementKind
        if k == SKd.Block:
            self.keep_only(sym, st.body, reps)
        elif k == SKd.List:
            for s in st.list:
                self.keep_only(sym, s, reps)
        elif k == SKd.Conditional:
            self.keep_only(sym, st.ifTrue, reps)
            if st.ifFalse is not None:
                self.keep_only(sym, st.ifFalse, reps)
        elif k == SKd.Case:
            for it in st.items:
                self.keep_only(sym, it.stmt, reps)
            if st.defaultCase is not None:
                self.keep_only(sym, st.defaultCase, reps)
        elif k == SKd.ForLoop:
            self.keep_only(sym, st.body, reps)
        elif k in (SKd.Empty, SKd.VariableDeclaration):
            pass
        elif k == SKd.ExpressionStatement:
            e = st.expr
            path, s, en = self.srcs.span(st.sourceRange)
            if e.kind == EK.Assignment:
                root, _ = chain_of(e.left)
                rsym = self.value_symbol(root) if root is not None else None
                if e.isNonBlocking:
                    if rsym is None and any(x.hierarchicalPath == sym.hierarchicalPath
                                            for _, x, _ in self.leaves(e.left)):
                        raise Unsupported("'%s' is assigned as part of '%s'"
                                          % (sym.name, self.srcs.text(e.left.sourceRange)))
                    if rsym is not None and rsym.hierarchicalPath == sym.hierarchicalPath:
                        _, ls, le = leaf_name_span(self.srcs, root, sym.name)
                        reps.append((ls, le, sym.name + SFX))
                        op = e.syntax.operatorToken.range
                        reps.append((op.start.offset, op.end.offset, "="))
                    else:
                        reps.append((s, en, ";"))
                    return
                raise Unsupported("blocking assignment in the block driving '%s'" % sym.name)
            if e.kind == EK.Call and e.isSystemCall:
                reps.append((s, en, ";"))
                return
            raise Unsupported("statement '%s' in the block driving '%s'" % (self.srcs.text(st.sourceRange)[:40], sym.name))
        else:
            raise Unsupported("%s statement in the block driving '%s'" % (k.name, sym.name))

    # ---- child outputs ----------------------------------------------------

    def child_output(self, sym, inst, child, path):
        covered = False
        for conn in child.portConnections:
            port = conn.port
            if port.kind != SK.Port or port.direction != ast.ArgumentDirection.Out or conn.expression is None:
                continue
            lv = conn.expression
            if lv.kind == EK.Assignment:
                lv = lv.left
            refs = [(n, s, p) for n, s, p in self.leaves(lv) if s.hierarchicalPath == sym.hierarchicalPath]
            if not refs:
                continue
            key = ("out", child.hierarchicalPath, port.name, sym.hierarchicalPath, path)
            if key in self.done:
                covered = True
                continue
            self.done.add(key)
            self.twin(port.internalSymbol)
            pname = port.name + SFX
            if lv.kind == EK.Concatenation:
                covered = self.concat_output(sym, child, pname, lv, path) or covered
            else:
                root, p = chain_of(lv)
                if root is None or not compatible(p, path):
                    continue
                ltxt, _ = self.rename_root(lv)
                self.connect(child, pname, ltxt)
                covered = True
        return covered

    def concat_output(self, sym, child, pname, lv, path):
        ops = list(lv.operands)
        fname = "%s_%s" % (child.name, pname)
        stmt = child.syntax.parent
        fpath, ss, se = self.srcs.span(stmt.sourceRange)
        ind = self.srcs.indent(fpath, ss)
        whole = self.srcs.text(lv.sourceRange)
        self.edits.add(("fwire", fpath, ss, fname), fpath, ss, ss,
                       "wire [$bits(%s)-1:0] %s;\n%s" % (whole, fname, ind))
        self.connect(child, pname, fname)
        covered = False
        for i, op in enumerate(ops):
            root, p = chain_of(op)
            if root is None or self.value_symbol(root).hierarchicalPath != sym.hierarchicalPath:
                continue
            if not compatible(p, path):
                continue
            later = [self.srcs.text(o.sourceRange) for o in ops[i + 1:]]
            off = "$bits({%s})" % ", ".join(later) if later else "0"
            ltxt, _ = self.rename_root(op)
            self.edits.add(("fslice", fpath, se, ltxt), fpath, se, se,
                           "\n%sassign %s = %s[%s +: $bits(%s)];" % (ind, ltxt, fname, off, self.srcs.text(op.sourceRange)))
            covered = True
        return covered

    # ---- the RAM ----------------------------------------------------------

    def ram(self, ram, check):
        parent = self.d.parent[ram.hierarchicalPath]
        syn = ram.syntax
        stmt = syn.parent
        raddr = ram.getPortConnection(self.d.port(ram, "raddr"))
        if raddr is None or raddr.expression is None:
            raise Unsupported("raddr unconnected")
        clk = ram.getPortConnection(self.d.port(ram, "clk"))
        self.clock = self.clock_root(clk.expression) if clk is not None and clk.expression is not None else None
        if self.clock is None:
            raise Unsupported("cannot follow the RAM's clock")
        nx = self.nx_expr(raddr.expression)
        # parameters
        params = stmt.parameters
        if params is None:
            raise Unsupported("RAM without a parameter list")
        named = {}
        for p in params.parameters:
            if not isinstance(p, pyslang.syntax.SyntaxNode):
                continue
            if p.kind != pyslang.syntax.SyntaxKind.NamedParamAssignment:
                raise Unsupported("RAM parameters not assigned by name")
            named[p.name.valueText] = p
        for pname, val in (("OUT_REG", "1"), ("RDW_MODE", '"W"')):
            if pname in named:
                rng = named[pname].expr.sourceRange
                path, s, e = self.srcs.span(rng)
                self.edits.add(("rparam", path, s, pname), path, s, e, val)
            else:
                path, s, e = self.srcs.span(params.openParen.range)
                self.edits.add(("rparam+", path, e, pname), path, e, e, "\n        .%s (%s)," % (pname, val))
        path, s, e = self.srcs.span(raddr.expression.sourceRange)
        self.edits.add(("raddr", path, s), path, s, e, nx)
        read = ram.getPortConnection(self.d.port(ram, "read"))
        if read is not None and read.expression is not None:
            path, s, e = self.srcs.span(read.expression.sourceRange)
            self.edits.add(("read", path, s), path, s, e, "1'b1")
        else:
            self.connect(ram, "read", "1'b1")
        if check:
            self.check_assert(ram, raddr.expression, nx)
        return nx

    def check_assert(self, ram, orig, nx):
        clk = ram.getPortConnection(self.d.port(ram, "clk")).expression
        rst = ram.getPortConnection(self.d.port(ram, "reset")).expression
        stmt = ram.syntax.parent
        path, s, e = self.srcs.span(stmt.sourceRange)
        ind = self.srcs.indent(path, s)
        n = ram.name
        o = self.srcs.text(orig.sourceRange)
        c, r = self.srcs.text(clk.sourceRange), self.srcs.text(rst.sourceRange)
        text = ("\n%sreg [$bits(%s)-1:0] %s_fold_raddr_r;\n%sreg %s_fold_ok_r = 0;\n"
                "%salways @(posedge %s) begin\n%s    %s_fold_raddr_r <= %s;\n%s    %s_fold_ok_r <= !%s;\n"
                "%s    if (%s_fold_ok_r && !%s && %s_fold_raddr_r !== %s)\n"
                "%s        $error(\"ram_fold: %%m.%s next address %%h, address %%h\", %s_fold_raddr_r, %s);\n%send"
                % (ind, o, n, ind, n, ind, c, ind, n, nx, ind, n, r, ind, n, r, n, o, ind, n, n, o, ind))
        self.edits.add(("check", path, e, n), path, e, e, text)


def run(args):
    files = list(args.files)
    design = Design(args.slang, files)
    report = []
    edits = Edits()
    targets = {}
    for ram in design.rams:
        try:
            d = const_int(design.param(ram, "SIZE"))
            w = const_int(design.param(ram, "DATAW"))
            oreg = const_int(design.param(ram, "OUT_REG"))
            lut = const_int(design.param(ram, "LUTRAM"))
        except (KeyError, ValueError) as ex:
            report.append("skip %s: parameters (%s)" % (ram.hierarchicalPath, ex))
            continue
        if oreg != 0 or not args.all and (lut != 0 or not force_bram(d, w)):
            continue
        targets[ram.hierarchicalPath] = (ram, d, w)
    # A fold edits the RAM's instantiation statement, which every instance of its
    # module shares: all RAMs it makes fold together or none does.
    groups = {}
    for p, (ram, d, w) in sorted(targets.items()):
        syn = ram.syntax.parent.sourceRange
        groups.setdefault((design.srcs.file(syn.start), syn.start.offset), [])
    for ram in design.rams:
        syn = ram.syntax.parent.sourceRange
        key = (design.srcs.file(syn.start), syn.start.offset)
        if key in groups:
            groups[key].append(ram)
    folded = []
    for key in sorted(groups):
        rams = groups[key]
        f = Fold(design)
        cur = rams[0]
        try:
            for cur in rams:
                if const_int(design.param(cur, "OUT_REG")) != 0:
                    raise Unsupported("its instantiation also makes a RAM that registers its read")
            nxs = []
            for cur in rams:
                nxs.append(f.ram(cur, args.check))
            trial = Edits()
            trial.merge(edits)
            trial.merge(f.edits)
            trial.apply(design.srcs)
            edits = trial
            for ram, nx in zip(rams, nxs):
                folded.append(ram.hierarchicalPath)
                report.append("fold %s (%sx%s): raddr <- %s" % (
                    ram.hierarchicalPath, design.param(ram, "SIZE"), design.param(ram, "DATAW"), nx))
        except Exception as ex:
            if not isinstance(ex, Unsupported):
                ex = "internal error: %s: %s" % (type(ex).__name__, ex)
            for ram in rams:
                why = str(ex) if ram is cur else "%s cannot fold: %s" % (cur.hierarchicalPath, ex)
                report.append("keep %s (%sx%s): %s" % (
                    ram.hierarchicalPath, design.param(ram, "SIZE"), design.param(ram, "DATAW"), why))
    patched = edits.apply(design.srcs) if folded else {}
    return design, report, patched, folded


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--top", action="append", required=True, help="top module (slang --top)")
    ap.add_argument("--report", help="write the fold report here")
    ap.add_argument("--diff", help="write the patch as a unified diff here")
    ap.add_argument("--check", action="store_true",
                    help="add a simulation check that each folded RAM's next address is its address a cycle later")
    ap.add_argument("--all", action="store_true",
                    help="fold every combinational VX_dp_ram read, whatever its size (testing the pass)")
    ap.add_argument("--dry-run", action="store_true", help="report only, do not patch")
    ap.add_argument("files", nargs="*")
    args, slang = ap.parse_known_args()
    args.slang = [x for t in args.top for x in ("--top", t)] + [a for a in slang if not a.endswith((".sv", ".v"))]
    args.files = [a for a in args.files] + [a for a in slang if a.endswith((".sv", ".v"))]

    design, report, patched, folded = run(args)

    diff = []
    if patched and not args.dry_run:
        originals = {p: design.srcs.bytes(p) for p in patched}
        for p, b in patched.items():
            with open(p, "wb") as f:
                f.write(b)
        new = Design(args.slang, args.files)
        added = new.errors - design.errors
        if added:
            for p, b in originals.items():
                with open(p, "wb") as f:
                    f.write(b)
            report.append("REVERTED: the patched sources do not elaborate:")
            report += ["  " + e for e in sorted(added)]
            folded = []
        else:
            still = [r.hierarchicalPath for r in new.rams
                     if r.hierarchicalPath in folded and const_int(new.param(r, "OUT_REG")) != 1]
            if still:
                raise SystemExit("ram_fold: internal error, not folded after patching: %s" % still)
            for p, b in patched.items():
                diff += difflib.unified_diff(originals[p].decode().splitlines(True),
                                             b.decode().splitlines(True), fromfile=p, tofile=p)
    if args.diff:
        with open(args.diff, "w") as f:
            f.writelines(diff)

    text = "\n".join(report) + ("\n" if report else "")
    if args.report:
        with open(args.report, "w") as f:
            f.write(text)
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
