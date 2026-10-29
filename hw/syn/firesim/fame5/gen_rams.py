#!/usr/bin/env python3
"""Replace VX_dp_ram/VX_sp_ram extmodules in a Yosys FIRRTL file with FIRRTL memories.

usage: gen_rams.py <in.fir> <out.fir>

Yosys keeps VX_dp_ram as a black box (its FIRRTL writer cannot emit clocked
read ports or byte enables) and writes one extmodule per instance, named
VX_dp_ram_<instance>, with that instance's parameters. Each is replaced here by
a module of the same name built on a FIRRTL `mem` whose data type is a vector
of WRENW lanes (so byte enables are write masks), reproducing VX_dp_ram's
simulation model exactly:

  OUT_REG=1, RDW_MODE=W  raddr registered on read, rdata = ram[raddr_r]
                         -> 1-cycle read of the next raddr_r, plus a bypass
                            of a same-cycle write to that address.
  OUT_REG=1, RDW_MODE=R  rdata_r <= ram[raddr] on read (value before write)
                         -> writes delayed one cycle through a buffer that
                            reads forward from; output held when not reading.
  OUT_REG=0, RDW_MODE=W  rdata = ram[raddr], combinational.
  OUT_REG=0, RDW_MODE=R  combinational, with the one-cycle old-data bypass.

The bypasses make the result independent of the memory's read-under-write
behaviour, which FAME-5's threaded memories do not preserve. Every OUT_REG=1
memory has read latency 1, so it stays block-RAM-mappable after threading.
"""
import re
import sys

DEFAULTS = dict(DATAW=1, SIZE=1, WRENW=1, OUT_REG=0, RDW_MODE="W", RESET_RAM=0, INIT_ENABLE=0)


def log2up(n):
    return max(1, (n - 1).bit_length())


def lanes(expr, n, w):
    return [f"bits({expr}, {i * w + w - 1}, {i * w})" for i in range(n)]


def cat(items):
    """cat(items[n-1], ..., items[0]) -- items[0] is the least significant."""
    out = items[0]
    for it in items[1:]:
        out = f"cat({it}, {out})"
    return out


def gen(name, prm, sp):
    p = dict(DEFAULTS)
    p.update(prm)
    for k in ("RESET_RAM", "INIT_ENABLE"):
        if int(p[k]):
            sys.exit(f"gen_rams: {name}: {k} is not supported")
    dw, size, nw = int(p["DATAW"]), int(p["SIZE"]), int(p["WRENW"])
    oreg, mode = int(p["OUT_REG"]), p["RDW_MODE"].strip('"')
    aw, lw = log2up(size), dw // nw
    assert lw * nw == dw
    L = []
    a = L.append
    a(f"  module {name}:")
    addrs = (("input", "addr", aw),) if sp else (("input", "waddr", aw), ("input", "raddr", aw))
    for d, n, w in (("input", "clk", None), ("input", "reset", 1), ("input", "read", 1), ("input", "write", 1),
                    ("input", "wren", nw), ("input", "wdata", dw)) + addrs + (("output", "rdata", dw),):
        a(f"    {d} {n}: {'Clock' if w is None else f'UInt<{w}>'}")
    a("")
    if sp:
        # VX_sp_ram: one address for both; its no-change mode is read-first
        # with the output kept when the cycle also writes.
        a("    node waddr = addr")
        a("    node raddr = addr")
        if mode == "N":
            assert oreg, f"{name}: RDW_MODE N needs OUT_REG"
            mode = "N"
    rden = "and(read, not(write))" if mode == "N" else "read"
    if mode == "N":
        mode = "R"
    readers = ["r", "r2"] if (oreg == 0 and mode == "R") else ["r"]
    a("    mem ram:")
    a(f"      data-type => UInt<{lw}>[{nw}]")
    a(f"      depth => {size}")
    a(f"      read-latency => {1 if oreg else 0}")
    a("      write-latency => 1")
    for r in readers:
        a(f"      reader => {r}")
    a("      writer => w")
    a("      read-under-write => undefined")
    a("")
    for r in readers:
        a(f"    ram.{r}.clk <= clk")
        a(f"    ram.{r}.en <= UInt<1>(1)")
    a("    ram.w.clk <= clk")
    mask = (lambda src: [f"bits({src}, {i}, {i})" for i in range(nw)]) if nw > 1 else (lambda src: ["UInt<1>(1)"])

    def write_port(en, addr, data, wren):
        a(f"    ram.w.en <= {en}")
        a(f"    ram.w.addr <= {addr}")
        for i, (d, m) in enumerate(zip(lanes(data, nw, lw), mask(wren))):
            a(f"    ram.w.data[{i}] <= {d}")
            a(f"    ram.w.mask[{i}] <= {m}")

    def merge(base, data, hit, wren):
        """base with the lanes of data enabled by (hit & wren) substituted."""
        m = mask(wren)
        out = []
        for i, (b, d) in enumerate(zip(lanes(base, nw, lw), lanes(data, nw, lw))):
            out.append(f"mux(and({hit}, {m[i]}), {d}, {b})")
        return cat(out)

    mem_out = lambda r: cat([f"ram.{r}.data[{i}]" for i in range(nw)])

    if oreg and mode == "W":
        a(f"    reg raddr_r: UInt<{aw}>, clk")
        a("    node next_addr = mux(read, raddr, raddr_r)")
        a("    raddr_r <= next_addr")
        a("    ram.r.addr <= next_addr")
        write_port("write", "waddr", "wdata", "wren")
        a("    reg hit_r: UInt<1>, clk")
        a("    hit_r <= and(write, eq(waddr, next_addr))")
        a(f"    reg wdata_r: UInt<{dw}>, clk")
        a("    wdata_r <= wdata")
        a(f"    reg wren_r: UInt<{nw}>, clk")
        a("    wren_r <= wren")
        a(f"    rdata <= {merge(mem_out('r'), 'wdata_r', 'hit_r', 'wren_r')}")
    elif oreg and mode == "R":
        a("    reg wb_v: UInt<1>, clk")
        a(f"    reg wb_addr: UInt<{aw}>, clk")
        a(f"    reg wb_data: UInt<{dw}>, clk")
        a(f"    reg wb_en: UInt<{nw}>, clk")
        a("    wb_v <= write")
        a("    wb_addr <= waddr")
        a("    wb_data <= wdata")
        a("    wb_en <= wren")
        write_port("wb_v", "wb_addr", "wb_data", "wb_en")
        a("    ram.r.addr <= raddr")
        a("    reg hit_r: UInt<1>, clk")
        a("    hit_r <= and(wb_v, eq(wb_addr, raddr))")
        a(f"    reg fwd_data: UInt<{dw}>, clk")
        a("    fwd_data <= wb_data")
        a(f"    reg fwd_en: UInt<{nw}>, clk")
        a("    fwd_en <= wb_en")
        a("    reg read_r: UInt<1>, clk")
        a(f"    read_r <= {rden}")
        a(f"    reg hold: UInt<{dw}>, clk")
        a(f"    node val = mux(read_r, {merge(mem_out('r'), 'fwd_data', 'hit_r', 'fwd_en')}, hold)")
        a("    hold <= val")
        a("    rdata <= val")
    elif not oreg and mode == "W":
        a("    ram.r.addr <= raddr")
        write_port("write", "waddr", "wdata", "wren")
        a(f"    rdata <= {mem_out('r')}")
    else:
        a("    ram.r.addr <= raddr")
        a("    ram.r2.addr <= waddr")
        write_port("write", "waddr", "wdata", "wren")
        a("    reg prev_write: UInt<1>, clk with: (reset => (reset, UInt<1>(0)))")
        a(f"    reg prev_data: UInt<{dw}>, clk with: (reset => (reset, UInt<{dw}>(0)))")
        a(f"    reg prev_waddr: UInt<{aw}>, clk with: (reset => (reset, UInt<{aw}>(0)))")
        a("    prev_write <= write")
        a(f"    prev_data <= {mem_out('r2')}")
        a("    prev_waddr <= waddr")
        a(f"    rdata <= mux(and(prev_write, eq(prev_waddr, raddr)), prev_data, {mem_out('r')})")
    a("")
    return L


src = open(sys.argv[1]).read().splitlines()
out, i, n = [], 0, 0
while i < len(src):
    m = re.match(r"  extmodule (VX_[ds]p_ram\S*?)\s*:", src[i])
    if not m:
        out.append(src[i])
        i += 1
        continue
    name, j, prm = m.group(1), i + 1, {}
    while j < len(src) and not re.match(r"  (ext)?module ", src[j]):
        pm = re.match(r"\s+parameter (\w+) = (.*?)\s*$", src[j])
        if pm:
            prm[pm.group(1)] = pm.group(2)
        j += 1
    out += gen(name, prm, name.startswith("VX_sp_ram"))
    n += 1
    i = j
open(sys.argv[2], "w").write("\n".join(out) + "\n")
print(f"gen_rams: {n} VX_dp_ram/VX_sp_ram instances")
