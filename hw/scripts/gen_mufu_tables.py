#!/usr/bin/env python3
# Copyright © 2019-2026
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

"""Generate the MUFU coefficient ROM.

Every table interval is approximated as P = A0 + A1*z + A2*z^2 in 30-bit fixed
point, z = xl/2^18 being the position inside the interval. Evaluation is the
bit-exact integer form shared by sw/common/mufu.cpp and hw/rtl/mufu:

    P = A0 + ((A1*xl) >> 18) + ((A2*((xl*xl) >> 18)) >> 18)

The ROM stores each coefficient without its S0/S1/S2 low bits (Ak = stored <<
Sk), so one entry is 72 bits and the 512 entries fill one 512x72 RAMB36.

Outputs (checked in, regenerate after any change here):
    sw/common/mufu_tables.h
    hw/rtl/mufu/VX_mufu_rom.sv   (no comments, by project convention for this unit)

Usage:
    gen_mufu_tables.py            write both outputs
    gen_mufu_tables.py --check    fail if the checked-in outputs are stale
    gen_mufu_tables.py --report   print per-table interpolation error
"""

import argparse
import math
import os
import sys

import numpy as np

FRAC = 30
XL_BITS = 18
# Coefficient widths seen by the datapath, and low bits dropped in the ROM.
A0_BITS = 32
A1_BITS = 27
A2_BITS = 20
S0, S1, S2 = 2, 2, 1
W0, W1, W2 = A0_BITS - 1 - S0, A1_BITS - 1 - S1, A2_BITS - S2
ROM_DEPTH = 512

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_H = os.path.join(ROOT, "sw", "common", "mufu_tables.h")
OUT_SV = os.path.join(ROOT, "hw", "rtl", "mufu", "VX_mufu_rom.sv")


# Intervals per table, sized for PTX .approx-level accuracy; --report prints the
# resulting worst-case errors.
N_RCP = 128         # 1/(1+t)
N_RSQRT = 64        # per exponent parity
N_EX2 = 64
N_LG2 = 64
N_TRIG = 64         # sin(pi/2 t); cos reads it at 1 - t
TANH_J0 = -5        # first tanh binade (below it tanh x = x)
TANH_BINADES = 8    # binades 2^-5 .. 2^2 (from 8 up tanh x = 1)
TANH_PER = 8        # intervals per tanh binade
TANH_PRESCALE = 4   # tanh entries hold 2^-(j+4)*tanh, keeping A1 in range


def _rcp(idx, z):
    return 1.0 / (1.0 + (idx + z) / N_RCP)


def _rsqrt(idx, z):
    t = ((idx % N_RSQRT) + z) / N_RSQRT
    s = (1.0 + t) * (2.0 if (idx // N_RSQRT) else 1.0)
    return 1.0 / np.sqrt(s)


def _ex2(idx, z):
    return np.exp2((idx + z) / N_EX2)


def _lg2(idx, z):
    return np.log2(1.0 + (idx + z) / N_LG2)


def _sin(idx, z):
    return np.sin(np.pi / 2.0 * (idx + z) / N_TRIG)


def _tanh(idx, z):
    j = (idx // TANH_PER) + TANH_J0
    t = ((idx % TANH_PER) + z) / TANH_PER
    return np.ldexp(np.tanh(np.ldexp(1.0 + t, j)), -j - TANH_PRESCALE)


def _mant_step(n):
    # xl takes the mantissa bits below the index, left-aligned in XL_BITS.
    low = 23 - (n.bit_length() - 1)
    return (1 << (XL_BITS - low)) if low < XL_BITS else 1, low > XL_BITS


# name, entries, f(idx, z), xl step (xl values are multiples of it),
# xl truncated (true argument lies anywhere in [xl, xl+1)), exact-left intervals
TABLES = [
    ("RCP",   N_RCP,       _rcp,   *_mant_step(N_RCP),   {0}),
    ("RSQRT", 2 * N_RSQRT, _rsqrt, *_mant_step(N_RSQRT), {0}),
    ("EX2",   N_EX2,       _ex2,   1, True,               {0}),
    ("LG2",   N_LG2,       _lg2,   *_mant_step(N_LG2),   {0}),
    ("SIN",   N_TRIG,      _sin,   1, True,               {0}),
    ("TANH",  TANH_BINADES * TANH_PER, _tanh, *_mant_step(TANH_PER), set()),
]


def eval_fixed(a0, a1, a2, xl):
    sq = (xl * xl) >> XL_BITS
    return a0 + ((a1 * xl) >> XL_BITS) + ((a2 * sq) >> XL_BITS)


def minimax(A, y, iters=40):
    # Lawson's iteration: reweighted least squares converging to the minimax fit.
    w = np.full(len(y), 1.0 / len(y))
    sol = None
    for _ in range(iters):
        sw = np.sqrt(w)
        sol, *_ = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)
        w = w * (np.abs(A @ sol - y) + 1e-12)
        w /= w.sum()
    return sol


def fit_interval(f, idx, step, trunc, exact_left):
    off = 0.5 if trunc else 0.0
    zs = (np.linspace(0, (1 << XL_BITS) - step, 2049) + off) / float(1 << XL_BITS)
    ys = f(idx, zs) * (1 << FRAC)

    if exact_left:
        a0 = int(round(float(f(idx, 0.0)) * (1 << FRAC)))
        a1, a2 = minimax(np.stack([zs, zs * zs], axis=1), ys - a0)
    else:
        a0, a1, a2 = minimax(np.stack([np.ones_like(zs), zs, zs * zs], axis=1), ys)
        a0 = int(round(a0))
    a1 = int(round(a1 / (1 << S1))) << S1
    a2 = int(round(a2 / (1 << S2))) << S2

    xl = np.arange(0, 1 << XL_BITS, step, dtype=np.int64)
    target = f(idx, (xl + off) / float(1 << XL_BITS)) * (1 << FRAC)
    if not exact_left:
        err = eval_fixed(a0, a1, a2, xl) - target
        a0 -= int(round((err.max() + err.min()) / 2.0))
    assert not exact_left or a0 % (1 << S0) == 0, (idx, a0)
    a0 = int(round(a0 / (1 << S0))) << S0
    err = eval_fixed(a0, a1, a2, xl) - target
    return a0, a1, a2, float(np.abs(err).max())


def build():
    entries = []
    report = []
    bases = []
    base = 0
    for name, n, f, step, trunc, exact in TABLES:
        bases.append((name, base, n))
        worst = 0.0
        wid = [0, 0, 0]
        for idx in range(n):
            a0, a1, a2, e = fit_interval(f, idx, step, trunc, idx in exact)
            q0, q1, q2 = a0 >> S0, a1 >> S1, a2 >> S2
            assert 0 <= q0 < (1 << W0), (name, idx, a0)
            assert -(1 << (W1 - 1)) <= q1 < (1 << (W1 - 1)), (name, idx, a1)
            assert -(1 << (W2 - 1)) <= q2 < (1 << (W2 - 1)), (name, idx, a2)
            entries.append((q0, q1, q2))
            worst = max(worst, e)
            wid = [max(wid[0], a0.bit_length()), max(wid[1], abs(a1).bit_length() + 1),
                   max(wid[2], abs(a2).bit_length() + 1)]
        report.append((name, worst, n, wid))
        base += n
    assert len(entries) <= ROM_DEPTH and W0 + W1 + W2 <= 72
    return entries, bases, report


def emit_h(entries, bases):
    out = []
    out.append("// Generated by hw/scripts/gen_mufu_tables.py. Do not edit.")
    out.append("#pragma once")
    out.append("")
    out.append("#include <cstdint>")
    out.append("")
    out.append("namespace vortex::mufu {")
    out.append("")
    out.append("// Stored coefficients; the datapath uses a0 << S0, a1 << S1, a2 << S2.")
    out.append("struct coeff_t { uint32_t a0; int32_t a1; int32_t a2; };")
    out.append("")
    out.append(f"constexpr uint32_t S0 = {S0};")
    out.append(f"constexpr uint32_t S1 = {S1};")
    out.append(f"constexpr uint32_t S2 = {S2};")
    out.append("")
    for name, base, n in bases:
        out.append(f"constexpr uint32_t TBL_{name}_BASE = {base};")
    out.append(f"constexpr uint32_t TBL_SIZE = {len(entries)};")
    out.append("")
    out.append("constexpr coeff_t kTable[TBL_SIZE] = {")
    for a0, a1, a2 in entries:
        out.append(f"  {{0x{a0:08x}u, {a1}, {a2}}},")
    out.append("};")
    out.append("")
    out.append("} // namespace vortex::mufu")
    out.append("")
    return "\n".join(out)


def emit_sv(entries, bases):
    aw = (ROM_DEPTH - 1).bit_length()
    iw = max(1, (max(n for _, _, n in bases) - 1).bit_length())
    tw = max(1, (len(bases) - 1).bit_length())
    w = W0 + W1 + W2
    out = []
    out.append("`include \"VX_platform.vh\"")
    out.append("")
    out.append("module VX_mufu_rom (")
    out.append("    input  wire clk,")
    out.append("    input  wire enable,")
    out.append(f"    input  wire [{tw - 1}:0] tbl,")
    out.append(f"    input  wire [{iw - 1}:0] idx,")
    out.append(f"    output wire [{A0_BITS - 1}:0] a0,")
    out.append(f"    output wire signed [{A1_BITS - 1}:0] a1,")
    out.append(f"    output wire signed [{A2_BITS - 1}:0] a2")
    out.append(");")
    out.append("")
    out.append(f"    reg [{aw - 1}:0] base;")
    out.append("")
    out.append("    always @(*) begin")
    out.append("        case (tbl)")
    for i, (_, base, _) in enumerate(bases):
        out.append(f"        {tw}'d{i}: base = {aw}'d{base};")
    out.append(f"        default: base = '0;")
    out.append("        endcase")
    out.append("    end")
    out.append("")
    out.append(f"    wire [{aw - 1}:0] addr = base + {aw}'(idx);")
    out.append("")
    out.append(f"    (* rom_style = \"block\" *) reg [{w - 1}:0] mem [0:{ROM_DEPTH - 1}];")
    out.append("")
    out.append("    initial begin")
    for i in range(ROM_DEPTH):
        q0, q1, q2 = entries[i] if i < len(entries) else (0, 0, 0)
        v = (q0 << (W1 + W2)) | ((q1 & ((1 << W1) - 1)) << W2) | (q2 & ((1 << W2) - 1))
        out.append(f"        mem[{i}] = {w}'h{v:0{(w + 3) // 4}x};")
    out.append("    end")
    out.append("")
    out.append(f"    reg [{w - 1}:0] rd_data, rd_data_r;")
    out.append("")
    out.append("    always @(posedge clk) begin")
    out.append("        if (enable) begin")
    out.append("            rd_data   <= mem[addr];")
    out.append("            rd_data_r <= rd_data;")
    out.append("        end")
    out.append("    end")
    out.append("")
    out.append(f"    assign a0 = {A0_BITS}'({{rd_data_r[{w - 1}:{W1 + W2}], {S0}'b0}});")
    out.append(f"    assign a1 = {A1_BITS}'($signed({{rd_data_r[{W1 + W2 - 1}:{W2}], {S1}'b0}}));")
    out.append(f"    assign a2 = {A2_BITS}'($signed({{rd_data_r[{W2 - 1}:0], {S2}'b0}}));")
    out.append("")
    out.append("endmodule")
    out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    entries, bases, report = build()
    h = emit_h(entries, bases)
    sv = emit_sv(entries, bases)

    if args.report:
        for name, worst, n, wid in report:
            print(f"{name:6s} {n:4d} entries  max interpolation error {worst:8.2f} units of 2^-{FRAC}"
                  f" (2^{math.log2(max(worst, 1e-9) / (1 << FRAC)):.2f})  bits a0/a1/a2 = {wid[0]}/{wid[1]}/{wid[2]}")
        print(f"total {len(entries)} entries")

    if args.check:
        stale = []
        for path, text in ((OUT_H, h), (OUT_SV, sv)):
            try:
                with open(path) as fp:
                    if fp.read() != text:
                        stale.append(path)
            except FileNotFoundError:
                stale.append(path)
        if stale:
            print("stale: " + ", ".join(stale), file=sys.stderr)
            return 1
        return 0

    if not args.report:
        os.makedirs(os.path.dirname(OUT_SV), exist_ok=True)
        with open(OUT_H, "w") as fp:
            fp.write(h)
        with open(OUT_SV, "w") as fp:
            fp.write(sv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
