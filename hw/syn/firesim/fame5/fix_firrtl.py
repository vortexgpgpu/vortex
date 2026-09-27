#!/usr/bin/env python3
"""Repair Yosys write_firrtl's output so the FIRRTL compiler and Verilator take it.

usage: fix_firrtl.py <in.fir> <out.fir>

Each rewrite keeps the circuit's meaning; they work on expression trees parsed
in one linear pass (a converted Vortex has lines of over a million characters,
nested thousands deep).

1. Instance outputs driving split wires. Yosys emits
     cat(a, cat(b, c)) <= inst.port
   which FIRRTL does not allow; each piece gets its slice instead.

2. Shifts by a signed amount ($shift, B signed). Yosys emits
     mux(B < 0, bits(dshl(A, <clamped B>), 0, W), dshr(A, B))
   which is not FIRRTL (`<`), has bits() bounds reversed, and shifts left by B
   rather than -B. $shift is Y = A >> B, and Y = A << -B for B < 0, so it becomes
     mux(lt(asSInt(B), SInt<1>(0)),
         bits(dshl(pad(A, W), min(-B, W)), W-1, 0),
         bits(pad(dshr(A, B), W), W-1, 0))

3. Left shifts. Yosys writes bits(dshl(a, min(b, 2^19-1)), hi, lo): correct,
   but the intermediate is ~2^19 bits wide, beyond Verilator's constant limit.
   Only bits hi..lo are kept, so the clamp becomes the first 2^k-1 > hi.

4. Zero tests of signed values: neq(asSInt(x), 0) is rejected as a
   signed/unsigned comparison; x != 0 does not depend on the sign, so the
   asSInt goes (likewise for eq).

5. Deep expressions. The FIRRTL parser recurses once per nesting level, and
   Yosys writes cat()/mux() chains thousands deep for wide arbiters; any
   subexpression deeper than MAX_DEPTH is hoisted into a node.
"""
import re
import sys

sys.setrecursionlimit(100000)  # transforms recurse over trees thousands deep

MAX_DEPTH = 48
BIG = "UInt<19>(524287)"

TOKEN = re.compile(r"\s*(?:(?P<lit>[US]Int(?:<\d+>)?\([^()]*\))"
                   r"|(?P<call>[A-Za-z_]\w*)\("
                   r"|(?P<close>\))|(?P<comma>,)|(?P<lt><)"
                   r"|(?P<leaf>[^\s(),<]+))")
ZERO = re.compile(r'^UInt(<\d+>)?\(("h0+"|0)\)$')
DECL = re.compile(r"\s+(?:input|output|wire|reg) (\S+?)\s*: UInt<(\d+)>")
STMT = re.compile(r"(\s+)(node \S+ = |\S+ <= )(.*?)(\s*@\[.*)?$")


class Node:
    """An operation: op(args...). Leaves (references, literals, integers) are str."""
    __slots__ = ("op", "args")

    def __init__(self, op, args):
        self.op, self.args = op, args


def parse(text):
    """One linear pass over text; `B < 0` (only in rewrite 2) parses as Node('<')."""
    stack = [Node("", [])]
    items = [[]]  # tokens of the argument being read, per open call
    pos = 0
    while pos < len(text):
        m = TOKEN.match(text, pos)
        if not m or m.end() == pos:
            if text[pos:].strip() == "":
                break
            raise ValueError(f"cannot tokenize at: {text[pos:pos + 60]}")
        pos = m.end()
        kind = m.lastgroup
        if kind == "call":
            node = Node(m.group("call"), [])
            items[-1].append(node)
            stack.append(node)
            items.append([])
        elif kind in ("comma", "close"):
            arg = items[-1]
            if arg:
                stack[-1].args.append(arg[0] if len(arg) == 1 else Node("<", [arg[0], arg[2]]))
            items[-1] = []
            if kind == "close":
                stack.pop()
                items.pop()
        elif kind == "lt":
            items[-1].append("<")
        else:
            items[-1].append(m.group(kind))
    arg = items[0]
    return arg[0] if len(arg) == 1 else Node("<", [arg[0], arg[2]])


def emit(t):
    """Iterative, so an unhoisted tree of any depth can be printed."""
    out, work = [], [t]
    while work:
        x = work.pop()
        if isinstance(x, str):
            out.append(x)
        elif isinstance(x, Node):
            if x.op == "<":
                work += [x.args[1], " < ", x.args[0]]
                continue
            work.append(")")
            for i, a in enumerate(reversed(x.args)):
                work.append(a)
                if i < len(x.args) - 1:
                    work.append(", ")
            work.append(f"{x.op}(")
    return "".join(out)


def lit(w, v):
    return f"UInt<{w}>({v})"


counts = dict(shift=0, shl=0, zero=0, hoist=0, cat=0)


def rewrite(t):
    """Rewrites 2-4, bottom-up."""
    if isinstance(t, str):
        return t
    t.args = [rewrite(a) for a in t.args]
    a = t.args
    # 2. mux(B < 0, bits(dshl(A, S), 0, W), dshr(A, B))
    if t.op == "mux" and len(a) == 3 and isinstance(a[0], Node) and a[0].op == "<":
        b, w = a[0].args[0], int(a[1].args[2])
        A = a[1].args[0].args[0]
        c = w.bit_length()
        mag = Node("asUInt", [Node("neg", [Node("asSInt", [b])])])
        amt = Node("bits", [Node("mux", [Node("gt", [mag, lit(c, w)]), lit(c, w), mag]), str(c - 1), "0"])
        counts["shift"] += 1
        return Node("mux", [Node("lt", [Node("asSInt", [b]), "SInt<1>(0)"]),
                            Node("bits", [Node("dshl", [Node("pad", [A, str(w)]), amt]), str(w - 1), "0"]),
                            Node("bits", [Node("pad", [Node("dshr", [A, b]), str(w)]), str(w - 1), "0"])])
    # 3. bits(dshl(A, asUInt(mux(gt(b, BIG), BIG, bits(b, 18, 0)))), hi, lo)
    if (t.op == "bits" and len(a) == 3 and isinstance(a[0], Node) and a[0].op == "dshl"
            and isinstance(a[0].args[1], Node) and a[0].args[1].op == "asUInt"):
        mx = a[0].args[1].args[0]
        if (isinstance(mx, Node) and mx.op == "mux" and mx.args[1] == BIG
                and isinstance(mx.args[0], Node) and mx.args[0].op == "gt" and mx.args[0].args[1] == BIG):
            b = mx.args[0].args[0]
            k = (int(a[1]) + 1).bit_length()
            m = (1 << k) - 1
            a[0].args[1] = Node("asUInt", [Node("mux", [Node("gt", [b, lit(k, m)]), lit(k, m),
                                                       Node("bits", [Node("pad", [b, str(k)]), str(k - 1), "0"])])])
            counts["shl"] += 1
    # 4. (n)eq(asSInt(x), 0)
    if (t.op in ("eq", "neq") and len(a) == 2 and isinstance(a[0], Node) and a[0].op == "asSInt"
            and isinstance(a[1], str) and ZERO.match(a[1])):
        t.args[0] = a[0].args[0]
        counts["zero"] += 1
    return t


def hoist(t, nodes, indent):
    """5. Returns (tree, depth) with every deeper subexpression moved to a node."""
    if isinstance(t, str):
        return t, 0
    depth = 0
    for i, k in enumerate(t.args):
        k2, d = hoist(k, nodes, indent)
        if d >= MAX_DEPTH:
            name = f"_fx_{counts['hoist']}"
            counts["hoist"] += 1
            nodes.append(f"{indent}node {name} = {emit(k2)}\n")
            k2, d = name, 0
        t.args[i] = k2
        depth = max(depth, d)
    return t, depth + 1


def cat_leaves(t):
    """cat(x, cat(y, z)) -> [x, y, z] (most significant first)."""
    if isinstance(t, Node) and t.op == "cat":
        return [l for a in t.args for l in cat_leaves(a)]
    return [emit(t)]


def fix_cat_lhs(line, widths):
    m = re.match(r"(\s+)(cat\(.*\)) <= (\S+)(\s*@.*)?$", line.rstrip("\n"))
    if not m:
        return None
    indent, lhs, rhs, info = m.group(1), m.group(2), m.group(3), m.group(4) or ""
    out, lo = [], 0
    for leaf in reversed(cat_leaves(parse(lhs))):
        if leaf not in widths:
            sys.exit(f"fix_firrtl: no width for {leaf} in: {line.strip()[:120]}")
        w = widths[leaf]
        out.append(f"{indent}{leaf} <= bits({rhs}, {lo + w - 1}, {lo}){info}\n")
        lo += w
    counts["cat"] += 1
    return out


def needs_work(line):
    return (" < 0" in line or BIG in line or "eq(asSInt(" in line
            or line.count("(") > MAX_DEPTH)


widths = {}
with open(sys.argv[1]) as f, open(sys.argv[2], "w") as o:
    for line in f:
        if line.startswith("  module ") or line.startswith("  extmodule "):
            widths = {}
        d = DECL.match(line)
        if d:
            widths[d.group(1)] = int(d.group(2))
        if line.lstrip().startswith("cat("):
            fixed = fix_cat_lhs(line, widths)
            if fixed:
                o.writelines(fixed)
                continue
        if needs_work(line):
            m = STMT.match(line.rstrip("\n"))
            if m:
                indent, head, rhs, info = m.group(1), m.group(2), m.group(3), m.group(4) or ""
                nodes = []
                t, _ = hoist(rewrite(parse(rhs)), nodes, indent)
                o.writelines(nodes)
                o.write(f"{indent}{head}{emit(t)}{info}\n")
                continue
        o.write(line)
print(f"fix_firrtl: rewrote {counts['shift']} signed shifts, {counts['cat']} concatenated connects, "
      f"{counts['shl']} left shifts, {counts['zero']} signed zero tests, "
      f"hoisted {counts['hoist']} deep subexpressions")
