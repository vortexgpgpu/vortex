#!/usr/bin/env python3
"""Identifiers used before their declaration in the RTL (hw/rtl).

IEEE 1800 requires a name to be declared before it is used (6.10, 23.8).
Vivado follows it: an undeclared target of a continuous assignment or port
connection becomes a new implicit net in the scope that uses it (Synth
8-11241 "undeclared symbol", then 8-3848 "does not have driver" on the real
one), and a variable read before its declaration is Synth 8-6901. Verilator,
Yosys, sv2v and slang all bind such a name to the later declaration, so every
simulation and lint passes while the synthesized net is a constant: the V80
froze on exactly this (foldsim_mem's rr_peek_valid, assigned in generate
blocks 60 lines before its module-level wire; fixed in d6fbbfd55).

The check is syntactic, so no elaboration choice hides a block: every
generate branch is scanned. Each .sv file is preprocessed (pyslang) under a
few define sets, so the code behind the configuration's `ifdef`s is scanned
too. A reference is reported when the name is declared in its scope or an
enclosing one, but only after the reference. Forward references the standard
allows are not reported: calls of functions and tasks, and hierarchical
names through an instance.

    decl_before_use.py [--selftest] [--root VORTEX_HOME] [--build BUILD] [files...]

With no files, every .sv under <root>/hw/rtl. Run from a configured build
tree (its hw/ holds VX_types.vh), or pass --build. Exit 1 on any finding.
"""

import argparse
import glob
import os
import re
import subprocess
import sys

from pyslang import parsing as P
from pyslang import syntax as S
import pyslang

# Define sets the files are preprocessed under; findings are the union. Each
# is passed to gen_config.py, which resolves the rest of the configuration.
DEFINE_SETS = [
    "-DVX_CFG_XLEN=32 -DSYNTHESIS -DVIVADO -DNDEBUG",
    "-DVX_CFG_XLEN=64 -DSYNTHESIS -DVIVADO -DNDEBUG -DVX_CFG_L2_ENABLE -DVX_CFG_L3_ENABLE "
    "-DVX_CFG_VM_ENABLE -DVX_CFG_EXT_A_ENABLE -DVX_CFG_EXT_TCU_ENABLE -DVX_CFG_EXT_DXA_ENABLE "
    "-DVX_CFG_EXT_TEX_ENABLE -DVX_CFG_EXT_RASTER_ENABLE -DVX_CFG_EXT_OM_ENABLE "
    "-DVX_CFG_EXT_RTU_ENABLE -DVX_CFG_EXT_MUFU_ENABLE -DVX_CFG_EXT_ZFHMIN_ENABLE "
    "-DVX_CFG_TCU_SPARSE_ENABLE -DVX_CFG_TCU_MX_ENABLE -DVX_CFG_TCU_WGMMA_ENABLE",
    "-DVX_CFG_XLEN=64 -DSIMULATION -DSV_DPI -DVX_CFG_L2_ENABLE -DVX_CFG_VM_ENABLE "
    "-DVX_CFG_EXT_TCU_ENABLE -DVX_CFG_EXT_DXA_ENABLE -DVX_CFG_EXT_TEX_ENABLE "
    "-DVX_CFG_EXT_RASTER_ENABLE -DVX_CFG_EXT_OM_ENABLE -DVX_CFG_EXT_RTU_ENABLE "
    "-DVX_CFG_IMUL_DPI_ENABLE -DVX_CFG_IDIV_DPI_ENABLE",
]

K = S.SyntaxKind
# Nodes that open a name scope.
SCOPES = {K.ModuleDeclaration, K.InterfaceDeclaration, K.PackageDeclaration, K.ProgramDeclaration,
          K.GenerateBlock, K.LoopGenerate, K.FunctionDeclaration, K.TaskDeclaration,
          K.SequentialBlockStatement, K.ParallelBlockStatement, K.ClassDeclaration}


def _source_root():
    """VORTEX_HOME from the build tree's config.mk, else this script's repo."""
    try:
        with open("config.mk") as fh:
            for line in fh:
                m = re.match(r"\s*VORTEX_HOME\s*[?:]?=\s*(\S+)", line)
                if m:
                    return m.group(1)
    except OSError:
        pass
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Scope:
    def __init__(self, parent):
        self.parent = parent
        self.decls = {}   # name -> [(order, kind)]


def scan_tree(tree):
    """[(name, line, decl_line)] of the references used before their declaration."""
    order = [0]
    refs = []         # (name, order, scope, context, location)
    root = Scope(None)

    def declare(scope, tok, kind):
        scope.decls.setdefault(tok.valueText, []).append((order[0], kind, tok.location))

    def walk(node, scope, ctx):
        if not isinstance(node, S.SyntaxNode):
            order[0] += 1
            return
        kind = node.kind
        if kind in SCOPES:
            scope = Scope(scope)
        if kind == K.StructUnionMember or kind == K.ModportDeclaration:
            return                                   # members are not scope names
        if kind == K.Declarator:
            declare(scope, node.name, "data")
        elif kind == K.InstanceName:
            declare(scope, node.name, "inst")
        elif kind == K.TypedefDeclaration:
            declare(scope, node.name, "type")
        elif kind == K.LoopGenerate:
            declare(scope, node.identifier, "genvar")
        elif kind == K.GenvarDeclaration:
            for c in node.identifiers:
                if isinstance(c, S.SyntaxNode) and c.kind == K.IdentifierName:
                    declare(scope, c.identifier, "genvar")
            order[0] += 1
            return
        elif kind == K.FunctionPrototype:
            nm = node.name
            if isinstance(nm, S.SyntaxNode) and nm.kind == K.IdentifierName:
                # the subroutine is named in its parent's scope
                declare(scope.parent or scope, nm.identifier, "sub")
            for c in node:
                if c is not nm:
                    walk(c, scope, ctx)
            return
        elif kind in (K.IdentifierName, K.IdentifierSelectName):
            refs.append((node.identifier.valueText, order[0], scope, ctx, node.identifier.location))
        elif kind == K.InvocationExpression:
            walk(node.left, scope, "call")
            for c in node:
                if c is not node.left:
                    walk(c, scope, ctx)
            return
        elif kind == K.ScopedName:
            sep = node.separator.valueText if hasattr(node, "separator") else "."
            if sep == "::":
                return                               # package or class scope
            walk(node.left, scope, "hier")
            return                                   # the rest resolves through the left
        for c in node:
            walk(c, scope, ctx)

    walk(tree.root, root, "")

    sm = tree.sourceManager
    found = []
    for name, ref_order, scope, ctx, loc in refs:
        if not name:
            continue                                 # a macro the file's headers did not define
        decls, s = [], scope
        while s is not None:
            decls += s.decls.get(name, [])
            s = s.parent
        if not decls or any(o < ref_order for o, _, _ in decls):
            continue
        kinds = {k for _, k, _ in decls}
        if kinds & {"sub"} or (ctx in ("call",)) or ("inst" in kinds and ctx == "hier"):
            continue                                 # forward references the standard allows
        first = min(decls)[2]
        found.append((name, sm.getFileName(loc), sm.getLineNumber(loc),
                      sm.getFileName(first), sm.getLineNumber(first)))
    return found


# The checker's own cases: (source, names it must report).
SELFTEST = [
    # d6fbbfd55: an assignment in a generate block ahead of the module's wire
    ("module m #(parameter P = 1); if (P) begin : g assign v = 1'b1; end else begin : h assign v = 1'b0; end\n"
     "wire v; wire y = v; endmodule", ["v", "v"]),
    # efa613ecc: a register read before its declaration
    ("module m(input clk); wire d = c; reg c; always @(posedge clk) c <= ~c; endmodule", ["c"]),
    # a port sized by a body localparam
    ("module m(input wire [W-1:0] a); localparam W = 4; endmodule", ["W"]),
    # allowed: declared first; a function called before its body; a hierarchical name through
    # a later instance; a genvar; a package-scoped name; struct members
    ("module m; wire a; assign a = f(1'b0); function automatic logic f(logic x); return x; endfunction\n"
     "wire b = u.z; sub u(); for (genvar i = 0; i < 2; ++i) begin : g wire w = i[0]; end\n"
     "wire c = pkg::K; typedef struct packed { logic q; } s_t; s_t s; wire d = s.q; endmodule", []),
]


def selftest():
    bad = 0
    for src, want in SELFTEST:
        got = sorted(n for n, *_ in scan_tree(S.SyntaxTree.fromText(src)))
        if got != sorted(want):
            print("decl_before_use selftest: want {} got {} in:\n{}".format(sorted(want), got, src))
            bad += 1
    print("decl_before_use selftest: {} of {} cases pass".format(len(SELFTEST) - bad, len(SELFTEST)))
    return bad == 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--selftest", action="store_true", help="check the checker on its own cases first")
    ap.add_argument("--root", default=None, help="VORTEX_HOME (default: from config.mk)")
    ap.add_argument("--build", default=".", help="configured build tree (for hw/VX_types.vh)")
    ap.add_argument("files", nargs="*")
    a = ap.parse_args(argv)
    if a.selftest and not selftest():
        return 2
    root = os.path.abspath(a.root or _source_root())
    build = os.path.abspath(a.build)
    files = [os.path.abspath(f) for f in a.files] or sorted(glob.glob(os.path.join(root, "hw", "rtl", "**", "*.sv"), recursive=True))
    incdirs = sorted({os.path.dirname(f) for f in glob.glob(os.path.join(root, "hw", "rtl", "**", "*.*vh"),
                                                              recursive=True)}
                     | {os.path.join(root, "hw", "rtl"), os.path.join(root, "hw", "dpi")})
    work = os.path.join(build, "ci_decl_before_use")
    os.makedirs(work, exist_ok=True)
    findings = set()
    for i, defs in enumerate(DEFINE_SETS):
        cfg_dir = os.path.join(work, str(i))
        os.makedirs(cfg_dir, exist_ok=True)
        subprocess.run([sys.executable, os.path.join(root, "ci", "gen_config.py"),
                        "--config=" + os.path.join(root, "VX_config.toml"), "--cflags=" + defs,
                        "--format=verilog", "--output=" + os.path.join(cfg_dir, "VX_config.vh")], check=True)
        opts = P.PreprocessorOptions()
        opts.additionalIncludePaths = [cfg_dir, os.path.join(build, "hw")] + incdirs
        opts.predefines = [d[2:] for d in defs.split() if d.startswith("-D")]
        bag = pyslang.Bag([opts])
        for f in files:
            sm = pyslang.SourceManager()
            tree = S.SyntaxTree.fromFile(f, sm, bag)
            for hit in scan_tree(tree):
                findings.add(hit)
    for name, f, line, df, dline in sorted(findings, key=lambda h: (h[1], h[2], h[0])):
        rel = lambda p: os.path.relpath(p, root) if p.startswith(root) else p
        print("{}:{}: '{}' used before its declaration at {}:{}".format(rel(f), line, name, rel(df), dline))
    print("decl_before_use: {} finding(s) in {} file(s) x {} define sets".format(
        len(findings), len(files), len(DEFINE_SETS)))
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
