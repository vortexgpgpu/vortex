#!/usr/bin/env python3
"""Repair Yosys write_firrtl's output.

usage: fix_firrtl.py <in.fir> <out.fir>

1. Instance outputs driving split wires. Yosys emits
     cat(a, cat(b, c)) <= inst.port
   which FIRRTL does not allow; each piece gets its slice instead:
     c <= bits(inst.port, wc-1, 0); b <= bits(..., wc+wb-1, wc); ...

2. $shift cells with a signed amount, below.

5. Zero tests of signed values. Yosys's logic ops test each operand with
   neq(x, UInt<1>("h0")), even when x is asSInt(...), which FIRRTL rejects as
   a signed/unsigned comparison. x != 0 does not depend on the sign, so the
   asSInt is dropped (likewise for eq).

4. Deep expressions. The FIRRTL parser recurses once per nesting level, and
   Yosys writes long cat()/mux() chains (thousands deep for wide arbiters);
   any subexpression deeper than MAX_DEPTH is hoisted into a node.

3. Left shifts. Yosys writes bits(dshl(a, min(b, 2^19-1)), hi, 0): correct, but
   the intermediate is ~2^19 bits wide, beyond Verilator's constant limit. Only
   bits hi..0 are kept, so the clamp is lowered to the first 2^k-1 > hi.

Yosys emits  mux(B < 0, bits(dshl(A, <clamped B>), 0, W), dshr(A, B))
which is not FIRRTL (`<`), has bits() bounds reversed, and shifts left by B
rather than -B. $shift semantics are: Y = A >> B, and for B < 0, Y = A << -B.
This rewrites each occurrence to

  mux(lt(asSInt(B), SInt<1>(0)),
      bits(dshl(pad(A, W), <min(-B, W)>), W-1, 0),
      bits(pad(dshr(A, B), W), W-1, 0))

which needs no width but W, taken from the original bits(..., 0, W).
"""
import sys

sys.setrecursionlimit(100000)  # expressions nest thousands deep


def match_paren(s, i):
    """s[i] == '('; return the index of its matching ')'."""
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "(":
            depth += 1
        elif s[j] == ")":
            depth -= 1
            if depth == 0:
                return j
    raise ValueError("unbalanced")


def split_args(s):
    """Split a comma-separated argument list at top level."""
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    out.append(cur.strip())
    return out


def fix_line(line):
    count = 0
    start = 0
    while True:
        i = line.find("mux(", start)
        if i < 0:
            return line, count
        j = match_paren(line, i + 3)
        args = split_args(line[i + 4:j])
        if len(args) == 3 and args[0].endswith(" < 0"):
            b = args[0][: -len(" < 0")]
            left, right = args[1], args[2]
            # left: bits(dshl(A, S), 0, W)
            inner = split_args(left[len("bits("):-1])
            dshl = inner[0]
            w = int(inner[2])
            a = split_args(dshl[len("dshl("):-1])[0]
            assert right.startswith("dshr(")
            c = w.bit_length()
            mag = f"asUInt(neg(asSInt({b})))"
            amt = f"bits(mux(gt({mag}, UInt<{c}>({w})), UInt<{c}>({w}), {mag}), {c - 1}, 0)"
            new = (f"mux(lt(asSInt({b}), SInt<1>(0)), "
                   f"bits(dshl(pad({a}, {w}), {amt}), {w - 1}, 0), "
                   f"bits(pad(dshr({a}, {b}), {w}), {w - 1}, 0))")
            line = line[:i] + new + line[j + 1:]
            count += 1
            start = i + len(new)
        else:
            start = i + 4


import re

BIG = "UInt<19>(524287)"
MAX_DEPTH = 48
LITERAL = re.compile(r"^[US]Int<\d+>\(")


def parse(e):
    """Expression -> (text, children) tree; literals and references are leaves."""
    e = e.strip()
    m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\(", e)
    if m and not LITERAL.match(e) and match_paren(e, len(m.group(1))) == len(e) - 1:
        return (m.group(1), [parse(a) for a in split_args(e[len(m.group(1)) + 1:-1])])
    return (e, None)


def emit(t):
    op, kids = t
    return op if kids is None else f"{op}({', '.join(emit(k) for k in kids)})"


def hoist(t, nodes, ns, indent):
    """Rewrite t so no expression exceeds MAX_DEPTH; returns (tree, depth)."""
    op, kids = t
    if kids is None:
        return t, 0
    new, depth = [], 0
    for k in kids:
        k2, d = hoist(k, nodes, ns, indent)
        if d >= MAX_DEPTH:
            name = f"_fx_{ns[0]}"
            ns[0] += 1
            nodes.append(f"{indent}node {name} = {emit(k2)}\n")
            k2, d = (name, None), 0
        new.append(k2)
        depth = max(depth, d)
    return (op, new), depth + 1


ZERO = re.compile(r'^UInt(<\d+>)?\(("h0+"|0)\)$')


def fix_zero_tests(line):
    count, start = 0, 0
    while True:
        m = re.compile(r"\b(n?eq)\(asSInt\(").search(line, start)
        if not m:
            return line, count
        i = m.start()
        j = match_paren(line, m.end(1))
        args = split_args(line[m.end(1) + 1:j])
        if len(args) == 2 and ZERO.match(args[1]) and args[0].startswith("asSInt("):
            inner = args[0][len("asSInt("):-1]
            new = f"{m.group(1)}({inner}, {args[1]})"
            line = line[:i] + new + line[j + 1:]
            count += 1
            start = i + len(new)
        else:
            start = m.end()


def fix_deep(line, ns):
    m = re.match(r"(\s+)(node \S+ = |\S+ <= )(.*?)(\s*@\[.*)?$", line.rstrip("\n"))
    if not m:
        return None
    indent, head, rhs, info = m.group(1), m.group(2), m.group(3), m.group(4) or ""
    d = cur = 0
    for c in rhs:
        if c == "(":
            cur += 1
            d = max(d, cur)
        elif c == ")":
            cur -= 1
    if d <= MAX_DEPTH:
        return None
    nodes = []
    t, _ = hoist(parse(rhs), nodes, ns, indent)
    return nodes + [f"{indent}{head}{emit(t)}{info}\n"]


def fix_dshl(line):
    count, start = 0, 0
    while True:
        i = line.find("bits(dshl(", start)
        if i < 0:
            return line, count
        j = match_paren(line, i + 4)
        args = split_args(line[i + 5:j])
        if len(args) == 3 and BIG in args[0]:
            hi = int(args[1])
            d = args[0]
            dargs = split_args(d[len("dshl("):-1])
            amt = dargs[1]
            m = re.fullmatch(r"asUInt\(mux\(gt\((.*), UInt<19>\(524287\)\), UInt<19>\(524287\), bits\((.*), 18, 0\)\)\)", amt)
            if m and m.group(1) == m.group(2):
                b = m.group(1)
                k = (hi + 1).bit_length()
                mx = (1 << k) - 1
                amt = f"asUInt(mux(gt({b}, UInt<{k}>({mx})), UInt<{k}>({mx}), bits(pad({b}, {k}), {k - 1}, 0)))"
                new = f"bits(dshl({dargs[0]}, {amt}), {args[1]}, {args[2]})"
                line = line[:i] + new + line[j + 1:]
                count += 1
                start = i + len(new)
                continue
        start = i + 5


DECL = re.compile(r"\s+(?:input|output|wire|reg) (\S+?)\s*: UInt<(\d+)>")


def cat_leaves(expr):
    """cat(x, cat(y, z)) -> [x, y, z] (most significant first)."""
    expr = expr.strip()
    if not expr.startswith("cat("):
        return [expr]
    out = []
    for a in split_args(expr[4:match_paren(expr, 3)]):
        out += cat_leaves(a)
    return out


def fix_cat_lhs(line, widths):
    m = re.match(r"(\s+)(cat\(.*\)) <= (\S+)(\s*@.*)?$", line.rstrip("\n"))
    if not m:
        return None
    indent, lhs, rhs, info = m.group(1), m.group(2), m.group(3), m.group(4) or ""
    out, lo = [], 0
    for leaf in reversed(cat_leaves(lhs)):
        if leaf not in widths:
            sys.exit(f"fix_firrtl: no width for {leaf} in: {line.strip()[:120]}")
        w = widths[leaf]
        out.append(f"{indent}{leaf} <= bits({rhs}, {lo + w - 1}, {lo}){info}\n")
        lo += w
    return out


total = cats = shls = zeros = 0
hoisted = [0]
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
                cats += 1
                continue
        if " < 0" in line:
            line, n = fix_line(line)
            total += n
        if BIG in line:
            line, n = fix_dshl(line)
            shls += n
        if "asSInt(" in line and "eq(asSInt(" in line:
            line, n = fix_zero_tests(line)
            zeros += n
        if line.count("(") > MAX_DEPTH:
            deep = fix_deep(line, hoisted)
            if deep:
                o.writelines(deep)
                continue
        o.write(line)
print(f"fix_firrtl: rewrote {total} signed shifts, {cats} concatenated connects, {shls} left shifts, "
      f"{zeros} signed zero tests, hoisted {hoisted[0]} deep subexpressions")
