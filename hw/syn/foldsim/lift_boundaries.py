#!/usr/bin/env python3
"""Interface boundary lifting: make interface-port units foldsim classes.

A class is threaded and wired into the token fabric port by port, so it must be
a module whose ports are plain signals; units such as VX_issue_slice or
VX_cluster have SystemVerilog interface ports instead. This pass gives each
requested unit a boundary made of plain wires, without touching the RTL: it
rewrites a copy of the (preprocessed) sources.

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

A unit whose instances differ in real parameters (VX_cache_bank: the L2, L1
data and L1 instruction caches' banks, which share one instantiation site in
VX_cache) is lifted with `sets`: one wrapper per parameter set, U__lift_<tag>,
each a class of its own. The tag is the shallowest instance-path segment that
all of the set's instances have and no other set's do (l2cache, dcache, ...),
else s<k> in hierarchy order. A site that serves several sets becomes a
generate-if over the site's own parameter overrides, one branch per set, each
instantiating that set's wrapper; a unit with one set is lifted as before.

Parameters that are functional instance IDs (e.g. VX_cluster's CLUSTER_ID)
are promoted to 32-bit input ports (port=P): the parameter becomes a port of U,
every child parameter override whose expression uses it is promoted the same
way down the hierarchy, and every affected instantiation passes the value as a
port connection instead (the old default where it was not overridden). Uses in
constant contexts (widths, generate conditions) make the re-elaboration fail,
so an invalid promotion cannot go through silently.

usage: lift_boundaries.py --srcs <srcs.f> --top <top> --out <dir>
                          --unit U[:drop=P1,P2][;port=P3][;sets] [--unit ...]
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
SK = pyslang.syntax.SyntaxKind


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


def elaborate(defines, incdirs, files, top, max_generate_steps=0):
    d = pyslang.driver.Driver()
    d.addStandardArgs()
    # The Vortex RTL references some nets before their declaration (VX_tcu_core),
    # which Vivado and Verilator accept; slang makes it an error unless allowed.
    args = ["slang", "--top", top, "--allow-hierarchical-const", "--allow-use-before-declare",
            "--ignore-unknown-modules"]
    # slang bounds the blocks a generate loop may make, counting those of the
    # loops around it (131072 by default, a guard against endless loops); a
    # large configuration's trees (VX_popcount over 16k inputs) go over it.
    if max_generate_steps:
        args += ["--max-generate-steps", str(max_generate_steps)]
    args += [f"-D{x}" for x in defines] + [f"-I{x}" for x in incdirs] + files
    if not (d.parseCommandLine(" ".join(q(a) for a in args), pyslang.driver.CommandLineOptions())
            and d.processOptions() and d.parseAllSources()):
        sys.exit("lift: slang could not parse the sources")
    comp = d.createCompilation()
    # Elaborates one canonical body per distinct instance (slang's instance
    # cache) and marks the others as copies of it.
    diags = comp.getAllDiagnostics()
    graph = definition_graph(comp)
    used = set(graph) | {c for cs in graph.values() for c in cs}
    sm = comp.sourceManager

    def unused_only(fname):
        # A file that only defines modules outside the elaborated hierarchy:
        # errors there cannot affect the design.
        names = re.findall(r"^\s*(module|interface|package)\s+(\w+)", open(fname).read(), re.M)
        return bool(names) and all(k == "module" and n not in used for k, n in names)
    errs = [x for x in diags if x.isError()
            and not unused_only(str(sm.getFileName(x.location)))]
    if errs:
        x = errs[0]
        sys.exit(f"lift: {len(errs)} elaboration errors (first: {x.code} at "
                 f"{sm.getFileName(x.location)}:{sm.getLineNumber(x.location)})")
    am = analysis.AnalysisManager()
    am.analyze(comp)
    return d, comp, am, graph


def definition_graph(comp):
    """{definition: {definitions it instantiates}} over the elaborated hierarchy.

    Walks canonical bodies only: an instance that slang's instance cache made a
    copy of another has the same children, and elaborating every copy's body
    costs memory per instance instead of per distinct body."""
    graph = collections.defaultdict(set)
    stack = list(comp.getRoot().topInstances)
    for t in stack:
        graph[t.definition.name]
    while stack:
        inst = stack.pop()
        name = inst.definition.name

        def visit(sym):
            if sym is inst:
                return ast.VisitAction.Advance
            if sym.kind == ast.SymbolKind.Instance:
                graph[name].add(sym.definition.name)
                if sym.canonicalBody is None:
                    stack.append(sym)
                return ast.VisitAction.Skip
            return ast.VisitAction.Advance
        inst.visit(visit)
    return graph


def find_instances(comp, graph, names):
    """{definition: [instances]} of the named definitions, in hierarchy order.

    Descends only into instances whose definition can contain one of them."""
    rev = collections.defaultdict(set)
    for p, cs in graph.items():
        for c in cs:
            rev[c].add(p)
    holders, todo = set(), list(names)
    while todo:
        for p in rev[todo.pop()]:
            if p not in holders:
                holders.add(p)
                todo.append(p)
    out = {n: [] for n in names}

    def visit(sym):
        if sym.kind == ast.SymbolKind.Instance:
            n = sym.definition.name
            if n in out:
                out[n].append(sym)
            return ast.VisitAction.Advance if n in holders else ast.VisitAction.Skip
        return ast.VisitAction.Advance
    for t in comp.getRoot().topInstances:
        t.visit(visit)
    return out


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
            elif k == "sets" and not v:
                self.sets = True
            else:
                sys.exit(f"lift: unknown option '{o}' for {name} (U[:drop=P,...][;port=P,...][;sets])")
        self.insts = []

    sets = False


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
            if unit_inst.canonicalBody is not None:
                sys.exit(f"lift: {upath} is a cached copy of another instance's body; "
                         f"its drivers are not analyzed")
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


def build_wrapper(u, rep, am, wname=None):
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
            f"module {wname or u.name + '__lift'} (\n" + ",\n".join(ports) + "\n);\n" + "\n".join(body) + "\n\n"
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


def site_connections(u, inst, shape):
    """{port: expression text} of an instance's connections, by name.

    `.p` is `.p (p)`, and `.*` connects each port not named otherwise to the
    signal of its name, as SystemVerilog defines them. Connections by
    position are refused: the rewrite reorders the ports."""
    iname = str(inst.decl.name).strip()
    conns, wildcard = {}, False
    for c in inst.connections:
        if isinstance(c, pyslang.parsing.Token):
            continue  # separator
        k = c.kind
        if k == SK.NamedPortConnection:
            nm = str(c.name).strip()
            if c.openParen.kind != pyslang.parsing.TokenKind.OpenParenthesis:
                conns[nm] = nm
            else:
                conns[nm] = str(c.expr).strip() if c.expr is not None else ""
        elif k == SK.WildcardPortConnection:
            wildcard = True
        elif k in (SK.OrderedPortConnection, SK.EmptyPortConnection):
            sys.exit(f"lift: {u.name} instance {iname}: ports connected by position; "
                     f"connect them by name")
        else:
            sys.exit(f"lift: {u.name} instance {iname}: unsupported port connection {k}")
    if wildcard:
        for kind, *rest in shape:
            if kind in ("plain", "iface") and rest[0] not in conns:
                conns[rest[0]] = rest[0]
    return conns


def rewrite_site(u, site_syntax, shape, uid, wname=None):
    """Replacement text for one instantiation statement."""
    wname = wname or u.name + "__lift"
    inst = site_syntax.instances[0]
    iname = str(inst.decl.name).strip()
    conns = site_connections(u, inst, shape)
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
    return ("// lift_boundaries: " + u.name + " -> " + wname + "\n        "
            + "\n        ".join(decl + glue) + "\n        "
            + f"{wname} {iname} (\n" + ",\n".join(pc) + "\n        );")


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

    def __init__(self, comp, sm, lifted_sites, graph):
        self.comp, self.sm, self.graph = comp, sm, graph
        self.lifted_sites = lifted_sites  # site keys rewritten by lifting: skip
        self.done = {}                   # (def, param) -> default text
        self.edits = collections.defaultdict(list)
        self.site_conv = collections.defaultdict(set)  # site key -> {param} converted
        self._insts = {}

    def insts(self, defname):
        if defname not in self._insts:
            self._insts[defname] = find_instances(self.comp, self.graph, [defname])[defname]
        return self._insts[defname]

    def key(self, node):
        r = node.sourceRange
        return (self.sm.getFileName(r.start), r.start.offset)

    def promote(self, defname, param):
        if (defname, param) in self.done:
            return
        rep = self.insts(defname)[0]
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
            for inst in self.insts(defname):
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
                if any(not isinstance(c, pyslang.parsing.Token)
                       and c.kind in (SK.OrderedPortConnection, SK.EmptyPortConnection)
                       for c in inst_syn.connections):
                    sys.exit(f"lift: {defname} instance {str(inst_syn.decl.name).strip()}: ports "
                             f"connected by position cannot take {param} as a port; "
                             f"connect them by name")
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


SV_INT = re.compile(r"^(?:\d+)?'(s?)([bodhBODH])([0-9a-fA-F_]+)$")


def num(v):
    """A parameter value's number, where its SV text is an integer literal
    (`1`, `1'b1`, `32'sd4` are the same number to `==`); else the text."""
    t = v.strip()
    if re.fullmatch(r"-?\d+", t):
        return int(t)
    m = SV_INT.match(t)
    if m:
        return int(m.group(3).replace("_", ""), {"b": 2, "o": 8, "d": 10, "h": 16}[m.group(2).lower()])
    return t


def set_key(params):
    return tuple((k, num(v)) for k, v in params.items())


def path_segments(inst):
    """The instance path's segment names, without array/generate indices."""
    return [re.sub(r"\[[^\]]*\]", "", x) for x in inst.hierarchicalPath.split(".")]


def set_tags(sets):
    """A name for each parameter set: the shallowest path segment that every
    instance of the set has and no instance of another set has, else s<k>."""
    segs = [[set(path_segments(i)) for i in insts] for _, insts in sets]
    tags = []
    for k, (_, insts) in enumerate(sets):
        common = set.intersection(*segs[k])
        others = set().union(*(x for j, ss in enumerate(segs) if j != k for x in ss))
        tag = next((g for g in path_segments(insts[0]) if g in common and g not in others), None)
        tag = re.sub(r"\W", "_", tag) if tag else f"s{k}"
        tags.append(tag)
    if len(set(tags)) != len(tags):
        tags = [f"s{k}" for k in range(len(sets))]
    return tags


def site_select(u, site, keys, iname):
    """For a site serving several parameter sets: per set, the generate
    condition that selects it, over the parameters the site overrides."""
    ov = site_overrides(site)
    vals = [dict(k) for k in keys]  # numbers where the values are integers
    cand = [p for p in vals[0] if p in ov and len({v[p] for v in vals}) > 1]
    chosen = []
    for p in cand:  # declaration order, until the values tell the sets apart
        chosen.append(p)
        if len({tuple(v[q] for q in chosen) for v in vals}) == len(vals):
            break
    else:
        sys.exit(f"lift: {u.name} instance {iname}: the parameter sets at one site differ only "
                 f"in parameters the site does not override")
    lit_of = lambda x: str(x) if isinstance(x, int) else x
    return [" && ".join(f"(({ov[p]}) == {lit_of(v[p])})" for p in chosen) for v in vals]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--srcs", required=True)
    ap.add_argument("--top", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--unit", action="append", required=True)
    ap.add_argument("--max-generate-steps", type=int, default=0,
                    help="slang's generate-loop step limit (0: slang's default)")
    a = ap.parse_args()
    a.out = os.path.abspath(a.out)
    defines, incdirs, files = read_srcs(a.srcs)
    d, comp, am, graph = elaborate(defines, incdirs, files, a.top, a.max_generate_steps)
    sm = comp.sourceManager
    units = {u.name: u for u in (Unit(s) for s in a.unit)}
    # Only the elaborated hierarchy: slang also builds placeholder instances of
    # uninstantiated definitions, which must not count.
    for name, insts in find_instances(comp, graph, list(units)).items():
        units[name].insts = insts

    edits = collections.defaultdict(list)  # file -> [(start, end, text)]
    wrappers = {}
    uid = 0
    lifted_sites = set()
    for u in units.values():
        if not u.insts:
            sys.exit(f"lift: no instance of {u.name}")
        skip = u.drop | set(u.ports)
        groups = collections.Counter(tuple(params_of(i, skip).items()) for i in u.insts)
        if u.sets:
            groups = collections.Counter(set_key(params_of(i, skip)) for i in u.insts)
        if len(groups) > 1 and not u.sets:
            keys = list(params_of(u.insts[0], skip).keys())
            diff = [k for k in keys if len({dict(g)[k] for g in groups}) > 1]
            sys.exit(f"lift: instances of {u.name} differ in {diff}; drop them if they are labels, "
                     f"or lift one class per parameter set (sets)")
        if len(groups) == 1:
            text, shape = build_wrapper(u, u.insts[0], am)
            wrappers[u.name + "__lift"] = text
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
            continue
        # One wrapper per parameter set, in hierarchy order.
        by_key = collections.OrderedDict()
        for i in u.insts:
            by_key.setdefault(set_key(params_of(i, skip)), []).append(i)
        sets = list(by_key.items())
        tags = set_tags(sets)
        shapes = {}
        for (key, insts), tag in zip(sets, tags):
            wname = f"{u.name}__lift_{tag}"
            text, shapes[key] = build_wrapper(u, insts[0], am, wname)
            wrappers[wname] = text
            print(f"lift: {u.name}: set {tag} -> {wname}: {len(insts)} instances "
                  f"(first {insts[0].hierarchicalPath})")
        wn = {key: f"{u.name}__lift_{tag}" for (key, _), tag in zip(sets, tags)}
        sites = collections.OrderedDict()  # site key -> (syntax, [set keys])
        for key, insts in sets:
            for i in insts:
                site = i.syntax.parent
                rng = site.sourceRange
                sk = (sm.getFileName(rng.start), rng.start.offset)
                ent = sites.setdefault(sk, (site, []))
                if key not in ent[1]:
                    ent[1].append(key)
        for sk, (site, keys) in sites.items():
            lifted_sites.add(sk)
            if len(site.instances) != 1:
                sys.exit(f"lift: {u.name}: multi-instance statement at {sk}")
            rng = site.sourceRange
            if len(keys) == 1:
                text = rewrite_site(u, site, shapes[keys[0]], uid, wn[keys[0]])
                uid += 1
            else:
                iname = str(site.instances[0].decl.name).strip()
                conds = site_select(u, site, keys, iname)
                parts = []
                for n, (key, cond) in enumerate(zip(keys, conds)):
                    head = "if" if n == 0 else "end else if"
                    if n == len(keys) - 1:
                        head, cond = "end else", None
                    body = rewrite_site(u, site, shapes[key], uid, wn[key])
                    uid += 1
                    blk = wn[key][len(u.name) + len("__lift_"):]
                    parts.append(f"{head}{' (' + cond + ')' if cond else ''} begin : g_lift_{blk}\n"
                                 f"        {body}\n        ")
                text = ("// lift_boundaries: " + u.name + ", one wrapper per parameter set\n        "
                        + "".join(parts) + "end")
            edits[sk[0]].append((rng.start.offset, rng.end.offset, text))
        print(f"lift: {u.name}: {len(u.insts)} instances, {len(sets)} parameter sets, "
              f"{len(sites)} sites")

    pr = Promoter(comp, sm, lifted_sites, graph)
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
    if applied != len(edits):
        sys.exit(f"lift: edits for unmatched files: {sorted(set(edits) - matched)}")
    for name, text in wrappers.items():
        p = os.path.join(a.out, "src", f"{name}.sv")
        open(p, "w").write(text)
        out_files.append(p)
    # Written last: the list is the build's mark of a finished lift.
    with open(os.path.join(a.out, "srcs.f"), "w") as o:
        for x in defines:
            o.write(f"+define+{x}\n")
        for x in incdirs:
            o.write(f"+incdir+{x}\n")
        o.write(f"+incdir+{os.path.join(a.out, 'src')}\n")
        for x in out_files:
            o.write(x + "\n")
    print(f"lift: wrote {len(out_files)} files ({applied} of {len(edits)} rewritten) to {a.out}")


if __name__ == "__main__":
    main()
