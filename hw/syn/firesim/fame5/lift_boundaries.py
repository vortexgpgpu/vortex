#!/usr/bin/env python3
"""Interface boundary lifting: make interface-port units FAME-5 fold classes.

sv2v inlines every module that has SystemVerilog interface ports, so units such
as VX_issue_slice or VX_cluster never reach the FIRRTL as modules and cannot be
threaded. This pass gives each requested unit a boundary made of plain wires,
without touching the RTL: it rewrites a copy of the (preprocessed) sources.

For each unit U:
  * every elaborated instance must have the same parameter values (parameters
    listed as droppable -- debug labels such as ISSUE_ID -- are left at their
    defaults);
  * a parameter-free wrapper U__lift is generated: each interface port becomes
    one plain port per signal (per array element, packed), with the direction
    taken from the port's modport or, for a bare interface port, from where
    the signal is driven (inside the unit -> output); the wrapper instantiates
    the interfaces with their elaborated parameters and U with its elaborated
    parameter values;
  * every instantiation site of U is replaced by glue wires/assigns between
    the parent's interfaces and an instance of U__lift.
Behaviour is unchanged: the glue is pure wiring.

Parameters that are functional instance IDs (e.g. VX_cluster's CLUSTER_ID)
are promoted to 32-bit input ports (port=P): the parameter becomes a port of U,
every child parameter override whose expression uses it is promoted the same
way down the hierarchy, and every affected instantiation passes the value as a
port connection instead (the old default where it was not overridden). Uses in
constant contexts (widths, generate conditions) make the re-elaboration fail,
so an invalid promotion cannot go through silently.

usage: lift_boundaries.py --srcs <srcs.f> --top <top> --out <dir>
                          --unit U[:drop=P1,P2][;port=P3] [--unit ...]
"""
import argparse
import collections
import os
import re
import shutil
import sys

import pyslang
from pyslang import ast, analysis

SIG_KINDS = (ast.SymbolKind.Variable, ast.SymbolKind.Net)


def read_srcs(path):
    defines, incdirs, files = [], [], []
    for line in open(path):
        line = line.strip()
        if line.startswith("+define+"):
            defines.append(line[len("+define+"):])
        elif line.startswith("+incdir+"):
            incdirs.append(line[len("+incdir+"):])
        elif line:
            files.append(line)
    return defines, incdirs, files


def q(a):
    return "'" + a.replace("'", "'\\''") + "'" if any(c in a for c in " '\"") else a


def elaborate(defines, incdirs, files, top):
    d = pyslang.driver.Driver()
    d.addStandardArgs()
    # The Vortex RTL references some nets before their declaration (VX_tcu_core),
    # which Vivado and Verilator accept; slang makes it an error unless allowed.
    args = ["slang", "--top", top, "--allow-hierarchical-const", "--allow-use-before-declare",
            "--ignore-unknown-modules"]
    args += [f"-D{x}" for x in defines] + [f"-I{x}" for x in incdirs] + files
    if not (d.parseCommandLine(" ".join(q(a) for a in args), pyslang.driver.CommandLineOptions())
            and d.processOptions() and d.parseAllSources()):
        sys.exit("lift: slang could not parse the sources")
    comp = d.createCompilation()
    used = set()

    def visit(sym):
        if sym.kind == ast.SymbolKind.Instance:
            used.add(sym.definition.name)
        return ast.VisitAction.Advance
    for t in comp.getRoot().topInstances:
        t.visit(visit)
    sm = comp.sourceManager

    def unused_only(fname):
        # A file that only defines modules outside the elaborated hierarchy (as
        # sv2v --top ignores them): errors there cannot affect the design.
        names = re.findall(r"^\s*(module|interface|package)\s+(\w+)", open(fname).read(), re.M)
        return bool(names) and all(k == "module" and n not in used for k, n in names)
    errs = [x for x in comp.getAllDiagnostics() if x.isError()
            and not unused_only(str(sm.getFileName(x.location)))]
    if errs:
        x = errs[0]
        sys.exit(f"lift: {len(errs)} elaboration errors (first: {x.code} at "
                 f"{sm.getFileName(x.location)}:{sm.getLineNumber(x.location)})")
    am = analysis.AnalysisManager()
    am.analyze(comp)
    return d, comp, am


def vec(w):
    return "" if w == 1 else f"[{w - 1}:0] "


def lit(v):
    s = str(v)
    return s


class Unit:
    def __init__(self, spec):
        name, _, opts = spec.partition(":")
        self.name = name
        self.drop = set()
        self.ports = []  # parameters promoted to input ports (functional instance IDs)
        for o in filter(None, opts.split(";")):
            k, _, v = o.partition("=")
            if k == "drop":
                self.drop |= set(filter(None, v.split(",")))
            elif k == "port":
                self.ports += [x for x in v.split(",") if x]
        self.insts = []


def params_of(inst, drop=()):
    """Overridable value and type parameters of an elaborated instance, as SV text."""
    out = collections.OrderedDict()
    for m in inst.body:
        if m.name in drop:
            continue
        if m.kind == ast.SymbolKind.Parameter and not m.isLocalParam:
            out[m.name] = lit(m.value)
        elif m.kind == ast.SymbolKind.TypeParameter and not m.isLocalParam:
            t = str(m.targetType.type)
            if not re.match(r"^[A-Za-z_][\w$]*(::[A-Za-z_][\w$]*)*$", t) and not re.match(r"^(logic|bit)( ?\[\d+:\d+\])*$", t):
                sys.exit(f"lift: type parameter {inst.hierarchicalPath}.{m.name} = '{t}' has no name to pass")
            out[m.name] = t
    return out


def iface_elements(conn):
    target, modport = conn.ifaceConn
    if target.kind == ast.SymbolKind.InstanceArray:
        return list(target.elements), modport
    return [target], modport


def signal_dirs(unit_inst, elem, modport_name, am):
    """[(signal, 'input'|'output', width)] from the unit's point of view."""
    out = []
    mp = None
    if modport_name:
        mp = next((m for m in elem.body if m.kind == ast.SymbolKind.Modport and m.name == modport_name), None)
    mp_dirs = {}
    if mp is not None:
        for p in mp:
            mp_dirs[p.name] = "input" if str(p.direction).endswith("In") else "output"
    upath = unit_inst.hierarchicalPath
    for m in elem.body:
        if m.kind not in SIG_KINDS:
            continue
        if m.type.isUnpackedArray:
            sys.exit(f"lift: unpacked interface signal {elem.hierarchicalPath}.{m.name} not supported")
        if mp is not None:
            if m.name not in mp_dirs:
                continue
            d = mp_dirs[m.name]
        else:
            drv = am.getDrivers(m)
            inside = any(dv.containingSymbol.hierarchicalPath == upath
                         or dv.containingSymbol.hierarchicalPath.startswith(upath + ".")
                         or dv.containingSymbol.hierarchicalPath.startswith(upath + "[") for dv in drv)
            d = "output" if inside else "input"
        out.append((m.name, d, m.type.bitWidth))
    return out


def port_range(port):
    """Declared interface-array range as text ('[0:3]'), or '' for a single interface."""
    r = port.declaredRange
    if not r:
        return "", 1
    rr = r[0]
    lo, hi = rr.left, rr.right
    return f"[{lo}:{hi}]", abs(hi - lo) + 1


def build_wrapper(u, rep, am):
    ports, body, inner_conns = [], [], []
    shape = []  # per port: ('plain', name, dir, w) or ('iface', name, rng, n, [(sig, dir, w)])
    for c in rep.portConnections:
        p = c.port
        if p.kind == ast.SymbolKind.InterfacePort:
            elems, mp = iface_elements(c)
            rng, n = port_range(p)
            sigs = signal_dirs(rep, elems[0], p.modport, am)
            idef = p.interfaceDef.name
            iparams = params_of(elems[0])
            pa = ""
            if iparams:
                pa = " #(" + ", ".join(f".{k}({v})" for k, v in iparams.items()) + ")"
            body.append(f"    {idef}{pa} {p.name} {rng}();".replace(" ();", "();"))
            for s, d, w in sigs:
                pn = f"{p.name}__{s}"
                ports.append(f"    {d} wire {('[' + str(n - 1) + ':0]') if rng else ''}{vec(w)}{pn}")
                for k in range(n):
                    ref = f"{p.name}[{k}].{s}" if rng else f"{p.name}.{s}"
                    pk = f"{pn}[{k}]" if rng else pn
                    body.append(f"    assign {ref} = {pk};" if d == "input" else f"    assign {pk} = {ref};")
            inner_conns.append(f"        .{p.name} ({p.name})")
            shape.append(("iface", p.name, rng, n, sigs))
        else:
            d = str(p.direction).split(".")[-1].lower()
            d = {"in": "input", "out": "output", "inout": "inout"}.get(d, d)
            if p.type.isUnpackedArray:
                sys.exit(f"lift: unpacked port {u.name}.{p.name} not supported")
            w = p.type.bitWidth
            ports.append(f"    {d} wire {vec(w)}{p.name}")
            inner_conns.append(f"        .{p.name} ({p.name})")
            shape.append(("plain", p.name, d, w))
    for pp in u.ports:
        ports.insert(0, f"    input wire [31:0] {pp}")
        inner_conns.insert(0, f"        .{pp} ({pp})")
        shape.insert(0, ("promoted", pp))
    prm = params_of(rep, u.drop | set(u.ports))
    pa = (" #(\n" + ",\n".join(f"        .{k} ({v})" for k, v in prm.items()) + "\n    )") if prm else ""
    text = (f"// Generated by lift_boundaries.py: {u.name} with plain ports (FAME-5 fold boundary).\n"
            f"module {u.name}__lift (\n" + ",\n".join(ports) + "\n);\n" + "\n".join(body) + "\n\n"
            f"    {u.name}{pa} inner (\n" + ",\n".join(inner_conns) + "\n    );\nendmodule\n")
    return text, shape


SLICE = re.compile(r"^(.*)\[(.+)\+:(.+)\]$")


def elem_expr(e, k, is_array):
    e = e.strip()
    if not is_array:
        return e
    m = SLICE.match(e)
    if m:
        return f"{m.group(1)}[({m.group(2).strip()}) + {k}]"
    if re.match(r"^[A-Za-z_]\w*$", e):
        return f"{e}[{k}]"
    if k == 0:
        return e  # a single element bound to a one-element array port
    sys.exit(f"lift: cannot index connection expression '{e}'")


def rewrite_site(u, site_syntax, shape, uid):
    """Replacement text for one instantiation statement."""
    inst = site_syntax.instances[0]
    iname = str(inst.decl.name).strip()
    conns = {}
    for c in inst.connections:
        if not hasattr(c, "name"):
            continue  # separator token
        nm = str(c.name).strip()
        ex = str(c.expr).strip() if c.expr is not None else ""
        conns[nm] = ex
    overrides = site_overrides(site_syntax)
    pre = f"__lift{uid}"
    decl, glue, pc = [], [], []
    for kind, *rest in shape:
        if kind == "promoted":
            pp = rest[0]
            if pp not in overrides:
                sys.exit(f"lift: {u.name} instance {iname}: promoted parameter {pp} is not overridden at the site")
            pc.append(f"        .{pp} ({overrides[pp]})")
            continue
        if kind == "plain":
            name, d, w = rest
            pc.append(f"        .{name} ({conns.get(name, '')})")
            continue
        name, rng, n, sigs = rest
        e = conns.get(name)
        if e is None:
            sys.exit(f"lift: {u.name} instance {iname}: interface port {name} not connected by name")
        for s, d, w in sigs:
            wn = f"{pre}_{name}__{s}"
            decl.append(f"wire {('[' + str(n - 1) + ':0]') if rng else ''}{vec(w)}{wn};")
            for k in range(n):
                ek = elem_expr(e, k, bool(rng))
                wk = f"{wn}[{k}]" if rng else wn
                glue.append(f"assign {wk} = {ek}.{s};" if d == "input" else f"assign {ek}.{s} = {wk};")
            pc.append(f"        .{name}__{s} ({wn})")
    return ("// lift_boundaries: " + u.name + " -> " + u.name + "__lift\n        "
            + "\n        ".join(decl + glue) + "\n        "
            + f"{u.name}__lift {iname} (\n" + ",\n".join(pc) + "\n        );")


def site_overrides(site):
    """{param: expr text} of a HierarchyInstantiationSyntax's #(...) named overrides."""
    out = {}
    pva = site.parameters
    if pva is None:
        return out
    for it in pva.parameters:
        if hasattr(it, "name") and hasattr(it, "expr"):
            out[str(it.name).strip()] = str(it.expr).strip() if it.expr is not None else ""
    return out


def _remove_item(text_len_getter, node, sm):
    """(start, end) covering a separated-list item and one adjacent comma."""
    return node.sourceRange.start.offset, node.sourceRange.end.offset


class Promoter:
    """Parameter-to-port promotion across definitions (see module docstring)."""

    def __init__(self, comp, sm, lifted_sites):
        self.comp, self.sm = comp, sm
        self.lifted_sites = lifted_sites  # site keys rewritten by lifting: skip
        self.done = {}                   # (def, param) -> default text
        self.edits = collections.defaultdict(list)
        self.site_conv = collections.defaultdict(set)  # site key -> {param} converted
        self.insts = collections.defaultdict(list)
        for t in comp.getRoot().topInstances:
            t.visit(self._collect)

    def _collect(self, sym):
        if sym.kind == ast.SymbolKind.Instance:
            self.insts[sym.definition.name].append(sym)
        return ast.VisitAction.Advance

    def key(self, node):
        r = node.sourceRange
        return (self.sm.getFileName(r.start), r.start.offset)

    def promote(self, defname, param):
        if (defname, param) in self.done:
            return
        rep = self.insts[defname][0]
        dsyn = rep.definition.syntax
        hdr = dsyn.header
        f = self.sm.getFileName(dsyn.sourceRange.start)
        text = open(f).read()
        default = None
        target = None
        for it in hdr.parameters.declarations:
            if not hasattr(it, "declarators"):
                continue
            names = [str(dd.name).strip() for dd in it.declarators if hasattr(dd, "name")]
            if param in names:
                if len(names) != 1:
                    sys.exit(f"lift: {defname}.{param} shares a declaration; split it first")
                dd = [dd for dd in it.declarators if hasattr(dd, "name")][0]
                default = str(dd.initializer.expr).strip() if dd.initializer is not None else "0"
                target = it
        if target is None:
            sys.exit(f"lift: no parameter {param} in {defname}")
        self.done[(defname, param)] = default
        a, b = target.sourceRange.start.offset, target.sourceRange.end.offset
        self.edits[f].append(_with_comma(text, a, b))
        op = hdr.ports.openParen.location.offset + 1
        self.edits[f].append((op, op, f"\n    input wire [31:0] {param},"))
        # children whose overrides use the promoted value
        for ch in rep.body:
            pass
        for cinst in self._children(rep):
            site = cinst.syntax.parent
            ov = site_overrides(site)
            for q, ex in ov.items():
                if re.search(rf"\b{re.escape(param)}\b", ex):
                    self.promote(cinst.definition.name, q)

    def _children(self, inst):
        out = []
        def v(sym):
            if sym is inst:
                return ast.VisitAction.Advance
            if sym.kind == ast.SymbolKind.Instance:
                out.append(sym)
                return ast.VisitAction.Skip
            return ast.VisitAction.Advance
        inst.visit(v)
        return out

    def convert_sites(self):
        """Every instantiation of a promoted (def, param): override -> port connection."""
        for (defname, param), default in self.done.items():
            for inst in self.insts[defname]:
                site = inst.syntax.parent
                k = self.key(site)
                if k in self.lifted_sites or param in self.site_conv[k]:
                    continue
                self.site_conv[k].add(param)
                f = k[0]
                text = open(f).read()
                ov_items = {}
                if site.parameters is not None:
                    for it in site.parameters.parameters:
                        if hasattr(it, "name"):
                            ov_items[str(it.name).strip()] = it
                if param in ov_items:
                    it = ov_items[param]
                    ex = str(it.expr).strip()
                    a, b = it.sourceRange.start.offset, it.sourceRange.end.offset
                    self.edits[f].append(_with_comma(text, a, b))
                else:
                    ex = default
                inst_syn = site.instances[0]
                op = inst_syn.openParen.location.offset + 1
                self.edits[f].append((op, op, f"\n            .{param} ({ex}),"))


def _with_comma(text, a, b):
    """Deletion range for a list item at [a, b) plus its separating comma."""
    j = b
    while j < len(text) and text[j] in " \t\r\n":
        j += 1
    if j < len(text) and text[j] == ",":
        return (a, j + 1, "")
    i = a - 1
    while i >= 0 and text[i] in " \t\r\n":
        i -= 1
    if i >= 0 and text[i] == ",":
        return (i, b, "")
    return (a, b, "")


def apply_edits(text, edits):
    """Apply (start, end, repl) edits; overlapping pure deletions are merged."""
    edits = sorted(edits, key=lambda e: (e[0], e[1]))
    merged = []
    for s, e, t in edits:
        if merged and s < merged[-1][1]:
            ps, pe, pt = merged[-1]
            if t or pt:
                sys.exit(f"lift: overlapping edits at {s}")
            merged[-1] = (ps, max(pe, e), "")
        else:
            merged.append((s, e, t))
    for s, e, t in reversed(merged):
        text = text[:s] + t + text[e:]
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--srcs", required=True)
    ap.add_argument("--top", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--unit", action="append", required=True)
    a = ap.parse_args()
    a.out = os.path.abspath(a.out)
    defines, incdirs, files = read_srcs(a.srcs)
    d, comp, am = elaborate(defines, incdirs, files, a.top)
    sm = comp.sourceManager
    units = {u.name: u for u in (Unit(s) for s in a.unit)}

    def visit(sym):
        if sym.kind == ast.SymbolKind.Instance and sym.definition.name in units:
            units[sym.definition.name].insts.append(sym)
        return ast.VisitAction.Advance
    # Only the elaborated hierarchy: slang also builds placeholder instances of
    # uninstantiated definitions, which must not count.
    for t in comp.getRoot().topInstances:
        t.visit(visit)

    edits = collections.defaultdict(list)  # file -> [(start, end, text)]
    wrappers = {}
    uid = 0
    lifted_sites = set()
    for u in units.values():
        if not u.insts:
            sys.exit(f"lift: no instance of {u.name}")
        skip = u.drop | set(u.ports)
        groups = collections.Counter(tuple(params_of(i, skip).items()) for i in u.insts)
        if len(groups) > 1:
            keys = list(params_of(u.insts[0], skip).keys())
            diff = [k for k in keys if len({dict(g)[k] for g in groups}) > 1]
            sys.exit(f"lift: instances of {u.name} differ in {diff}; drop them if they are labels")
        text, shape = build_wrapper(u, u.insts[0], am)
        wrappers[u.name] = text
        seen = set()
        for i in u.insts:
            site = i.syntax.parent  # HierarchyInstantiationSyntax
            rng = site.sourceRange
            key = (sm.getFileName(rng.start), rng.start.offset)
            if key in seen:
                continue
            seen.add(key)
            lifted_sites.add(key)
            if len(site.instances) != 1:
                sys.exit(f"lift: {u.name}: multi-instance statement at {key}")
            edits[key[0]].append((rng.start.offset, rng.end.offset, rewrite_site(u, site, shape, uid)))
            uid += 1
        print(f"lift: {u.name}: {len(u.insts)} instances, {len(seen)} sites, "
              f"{sum(1 for s in shape if s[0] == 'iface')} interface ports, "
              f"{sum(len(s[4]) for s in shape if s[0] == 'iface')} lifted signals")

    pr = Promoter(comp, sm, lifted_sites)
    for u in units.values():
        for pp in u.ports:
            pr.promote(u.name, pp)
    pr.convert_sites()
    for f, es in pr.edits.items():
        edits[f].extend(es)
    if pr.done:
        print("lift: promoted " + ", ".join(f"{d}.{p}" for d, p in pr.done))

    os.makedirs(os.path.join(a.out, "src"), exist_ok=True)
    out_files = []
    by_real = {os.path.realpath(str(k)): k for k in edits}
    applied, matched = 0, set()
    for f in files:
        dst = os.path.join(a.out, "src", os.path.basename(f))
        key = by_real.get(os.path.realpath(f))
        if key is not None:
            open(dst, "w").write(apply_edits(open(f).read(), edits[key]))
            applied += 1
            matched.add(key)
        else:
            shutil.copyfile(f, dst)
        out_files.append(dst)
    for name, text in wrappers.items():
        p = os.path.join(a.out, "src", f"{name}__lift.sv")
        open(p, "w").write(text)
        out_files.append(p)
    with open(os.path.join(a.out, "srcs.f"), "w") as o:
        for x in defines:
            o.write(f"+define+{x}\n")
        for x in incdirs:
            o.write(f"+incdir+{x}\n")
        o.write(f"+incdir+{os.path.join(a.out, 'src')}\n")
        for x in out_files:
            o.write(x + "\n")
    print(f"lift: wrote {len(out_files)} files ({applied} of {len(edits)} rewritten) to {a.out}")
    if applied != len(edits):
        sys.exit(f"lift: edits for unmatched files: {sorted(set(edits) - matched)}")


if __name__ == "__main__":
    main()
