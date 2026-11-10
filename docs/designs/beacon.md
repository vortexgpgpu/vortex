# Beacon — logic depth and area, before Vivado

Beacon counts the logic levels of every register-to-register path
of a Vortex design the way Vivado counts them, and the LUTs, flip-flops,
distributed and block RAM and DSPs it maps to, in seconds and without
Vivado. It names the RTL units whose paths are too deep for the target clock,
gates their depth and the design's area on every RTL change (`logic_gate`),
and is the first step of RTL timing work: run Beacon, fix the deep units,
then confirm with one Vivado run.

| § | contents |
|---|---|
| [1](#1-overview) | Overview and how to run it |
| [2](#2-system-architecture) | System architecture: where each piece lives |
| [3](#3-compiler-architecture) | Compiler architecture: the two passes |
| [4](#4-the-model) | The model: levels per element, devices, area (§4.5) |
| [5](#5-outputs) | Outputs |
| [6](#6-integration) | Integration: the DUT flow, `logic_gate`, timing work |
| [7](#7-validation) | Validation against Vivado; the learned calibration (§7.1); area (§7.2) |
| [8](#8-diagnostics) | Diagnostics beyond levels |
| [9](#9-invariants-and-limitations) | Invariants and limitations |

---

## 1. Overview

It answers one question per register: **how many logic levels is its worst
input path?** At 300 MHz on the V80, failing end points have a median of
7–8 levels, and between half and four fifths of the registers at 12 or more
levels fail, depending on the gate (§7). So Beacon reports:

- **fix list:** units with end points at ≥ 12 levels;
- **watch list:** ≥ 8 levels.

It does not predict slack, because it does not see wires (§9).

One DUT of the synthesis catalog, from a configured build tree:

```bash
make -C hw/syn/xilinx/dut tcu-beacon
# build_tcu/beacon_paths.txt     every path at >= 8 levels, op by op, by unit
# build_tcu/beacon_units.json    deepest end point per RTL unit, area per unit
# build_tcu/beacon_area.txt      LUTs, FFs, memories and DSPs, by unit; each memory's placement
```

The whole catalog as a gate, and a baseline update (a human step):

```bash
ci/logic_gate.py -j 2
ci/logic_gate.py -b om --update-baseline
```

Any CIRCT netlist, with `ci/beacon.py` (subcommands `run`, `screen`,
`summarize`, `paths`, `validate`, `retime-split`, `ports`, `buffers`,
`calibrate`, `score`;
`foldsim-opt` from the `PATH` or `FOLDSIM_OPT`):

```bash
ci/beacon.py run design.mlir runs/full                     # map, then count levels and area
ci/beacon.py run runs/full/mapped.mlirbc runs/socket \
    top=VX_socket__lift black-boxes=VX_core__lift           # one unit of a mapped design
ci/beacon.py summarize runs/full --min-levels 12           # deep end points by RTL unit
ci/beacon.py paths runs/full --unit lsu_unit               # its paths, op by op (§5)
ci/beacon.py validate runs/socket ep.csv --native VX_socket__lift=dut
ci/beacon.py screen fold/netlist.mlir runs/fold top=foldsim_target   # a fold: module boundaries timed
```

`screen` maps with `foldsim-screen-map` instead of `foldsim-beacon-map`: its
LUT mapper times adders and comparators by carry block with their inputs'
arrival times, and a design that keeps several modules after inlining (a
fold's models, each mapped on its own) is mapped a second time with the
arrival times the first count found at the modules' ports
(`foldsim-screen-map{arrivals=}`, `--boundary-passes`, 2 by default), so a
late signal crossing a model boundary is timed as Vivado times the flattened
design. On the 4-SM fold the end point bits at 12 levels or more went from 66
to 1. It counts no area; `family=` is accepted for `target=`.

Or the passes directly:

```bash
foldsim-opt design.mlir --emit-bytecode -o mapped.mlirbc \
  --pass-pipeline='builtin.module(foldsim-beacon-map{target=virtexuplusHBM})'
foldsim-opt mapped.mlirbc -o /dev/null --pass-pipeline='builtin.module(
  foldsim-estimate-timing{top=VX_core_top target=virtexuplusHBM
    output=endpoints.csv instances=instances.csv report=report.txt
    top-paths=50 trace=lsu_unit})'
foldsim-opt design.mlir -o /dev/null --pass-pipeline='builtin.module(
  foldsim-beacon-map{target=virtexuplusHBM area},
  foldsim-estimate-area{top=VX_core_top target=virtexuplusHBM
    output=area.csv memories=area_memories.csv report=area.txt})'
```

| `foldsim-estimate-timing` option | meaning |
|---|---|
| `top=` | the top module (default: the one nothing instantiates) |
| `target=` | the Beacon target (§4.2): `versal` (V80, default), `virtexuplus`, `virtexuplusHBM` (U55C), `zynquplus`, or `asap7` (ASIC, §4.3); map with `foldsim-beacon-map` for the same target |
| `black-boxes=` | modules to treat as external, as a harness stubs them |
| `output=`, `instances=`, `report=` | the files of §5 |
| `min-levels=` | write only end points at or above this many levels |
| `top-paths=` | deepest paths to print op by op (default 50) |
| `trace=` | also print the deepest path of each end point whose name contains this |
| `paths=`, `path-levels=` | every path of at least `path-levels` (12) levels, one per pair of registers, op by op (§5) |
| `retime-levels=` | also estimate the depth after Vivado's retiming to this period (§4.4) |
| `ports=`, `port-modules=` | the depth at each port of the named modules' instances (§8) |

| `foldsim-estimate-area` option | meaning |
|---|---|
| `top=`, `target=`, `black-boxes=` | as for `foldsim-estimate-timing`; map with `foldsim-beacon-map{area}` for the same target |
| `output=` | the instance table of §5 (`area.csv`) |
| `memories=` | every memory, its shape and where it goes (`area_memories.csv`) |
| `report=` | the design's totals, the mapping's LUTs by op, the vendor IP |
| `lut-scale=` | device LUTs per mapped LUT (default: the target's, §4.5) |
| `modules=`, `state-threads=` | per module, its totals and its state's cost folded over that many threads (JSON; foldsim's fold-degree search prices copies with it) |

## 2. System architecture

![Beacon system](../assets/img/beacon_system.svg)

The engine is two compiler passes of **foldsim** (`foldsim-opt`), built on
CIRCT like the rest of foldsim, and shipped in the foldsim prebuilt. Vortex
never builds it. The integration is in Vortex:

| Piece | Where | Role |
|---|---|---|
| `foldsim-beacon-map`, `foldsim-estimate-timing`, `foldsim-estimate-area` | foldsim `lib/Transforms/Beacon.cpp`, `EstimateTiming.cpp`, `EstimateArea.cpp` | map to the target's cells; count levels; count area |
| `ci/beacon.py` | `ci/` | run Beacon on any netlist; read it by unit; compare with Vivado |
| `xilinx_cone_dump.tcl`, `xilinx_cone_probe.tcl`, `xilinx_wire_probe.tcl` | `hw/scripts` | Vivado probes for what levels cannot show (§8) |
| target `beacon` | `hw/syn/xilinx/dut/common.mk` | one DUT: sources → netlist → levels and area → summary |
| `circt_elaborate.sh -f mlir` | `hw/scripts` | the lowered HW/Comb/Seq netlist, no `firtool` |
| `gen_platform.py --emit=family` | `hw/scripts` | the family of `DEVICE`, from `hw/syn/xilinx/platforms` |
| `beacon_summary.py` | `hw/scripts` | `beacon_summary.csv`, per-unit `beacon_units.json`, `beacon_area.txt` |
| `Beacon` backend, `logic_gate.py` | `ci/synth_gate.py`, `ci/` | the gate, beside `fpga_gate` and `asic_gate` |
| `logic_gate.yml` | `.github/workflows` | runs it on pull requests that touch RTL |

Beacon reads the same CIRCT import that foldsim and the Yosys flow
use (`circt-verilog --ir-hw`), elaborated from the sources and defines
`gen_sources.sh` writes for Vivado, with the Xilinx floating-point IPs as
black boxes (`xilinx_ip_stubs.sv`).

## 3. Compiler architecture

![Beacon compiler](../assets/img/beacon_compiler.svg)

### 3.1 `foldsim-beacon-map`

Every pass loads the same **target** from foldsim's target table
(`foldsim/Beacon.h`): its cells for the mapping, its DSPs and delays for the
level count, its LUT scale for the area (§4.5). A target is named after its FPGA family or its PDK, so every
flow passes the name it already has: the DUT flow the family
`gen_platform.py --emit=family` gives for `DEVICE`, the ASIC flow `PDK`,
`ci/beacon.py run` its `target=`. The mapping is a pipeline of CIRCT's
synthesis passes plus three of foldsim's, per module:

1. **Inline small modules** (`hw-flatten-modules`, < 300 ops) so LUTs pack
   across buffers, arbiters and switches, as Vivado's flattened synthesis
   does. Larger modules stay instances; the estimator elaborates them.
2. **Canonicalize and balance** mux chains (`comb-balance-mux`).
3. **foldsim passes:**
   - `foldsim-lower-equality`: an equality compare is XORs and an OR tree,
     which pack into LUTs;
   - `foldsim-lower-constant-division`: a division by a constant is an
     extract (power of two) or a table (dividend of ≤ 10 bits), as
     synthesis writes it;
   - `foldsim-fold-zero-width`: zero-width values are constants, which
     CIRCT's lowering otherwise rejects.
4. **Lower to an and-inverter graph** (`convert-comb-to-synth`). Adders,
   subtracters, compares, multipliers, dividers and remainders stay
   word-level: the estimator models them as the carry chains, DSPs and
   tables Vivado builds, which a LUT mapping of their gates would not
   reproduce.
5. **Depth-optimize** the graph: timing-aware variadic lowering, maximum
   AND cover, SOP balancing, structural hashing.
6. **Map to the target's cells** (`synth-generic-lut-mapper`, timing
   strategy): 6-input LUTs on the FPGA families, 3-input cuts on ASAP7.

The full GPU maps in about a minute and 18–20 GB.

With `area`, the same lowering maps the design for `foldsim-estimate-area`
instead. A depth-oriented cover duplicates logic to save levels, so its
cells are no measure of area; the area mapping is what Vivado does off the
critical paths:

- every module stays an instance, so each one's resources stay its own
  (inlining moves the count by 1–4 %), and each module maps once however
  many instances it has;
- no mux balancing, no maximum AND cover and no SOP balancing;
- variadic logic lowered sharing common subsets (`reuse-subsets`), and the
  cut mapper's area strategy.

A DUT of the catalog maps for area in 1–16 s.

### 3.2 `foldsim-estimate-timing`

The estimator works on the elaborated design: one **context** per instance,
and one **node** per (context, value, bit). It never enumerates paths:

1. **Collect end points** by walking the instance tree from `top`.
2. **Resolve** each end point bit through wires, extracts, concatenations,
   replicates and instance ports to the op that drives it.
3. **Evaluate** with an iterative depth-first walk: a node's levels are the
   maximum over its inputs of the input's levels plus the edge's (§4), and
   it keeps its one worst predecessor. An input still being evaluated
   closes a combinational loop; that edge is left out and counted.
4. **Report**: each end point's start point is found by walking the worst
   predecessors back; the walk also yields the path's largest fanout and its
   module crossings.

Time and memory are linear in the design's bits: the full GPU runs in
16–21 s and 2.7 GB, a DUT of the catalog in seconds.

### 3.3 `foldsim-estimate-area`

The area estimator counts each module's own ops once (§4.5), then sums the
instance tree from `top`: a module's total is its own resources and its
instances' totals. It writes one row per instance, depth first, with the
instance's own resources and its subtree's, and one row per memory with the
placement chosen for it. It takes under a second on a DUT.

## 4. The model

### 4.1 Levels per element

| Element | Levels |
|---|---|
| LUT (`comb.truth_table`) | 1 |
| Adder or subtracter, input bit j to output bit i | 2 + ⌊i/8⌋ − ⌊j/8⌋: a LUT, then the 8-bit carry blocks; 1 if the output bit reads ≤ 6 input bits (one LUT6) |
| Order compare | as an adder, up to the highest bit not constant in both operands |
| Equality compare | lowered to LUTs by the mapping |
| Multiply, ≤ 10 variable operand bits | a table of constants: 1 + ⌈(bits − 6)/2⌉ |
| Multiply, wider | a DSP cascade over the significant bits (§4.2) |
| Divide or remainder | a table over the bits its operand cones depend on, if ≤ 10; else a ripple of one subtracter per quotient bit |
| Distributed-RAM read (`seq.firmem`, latency 0) | 1, plus 1 per doubling past 64 words (MUXF7, MUXF8, LUT), from the address |
| Other ops | 1 per op, listed in the report |

Significant bits exclude constant bits on top and copies of the bit below
(a sign extension). A table is a LUT6 for six inputs and a level of 4:1
multiplexers per two more.

### 4.2 Devices

Versal and the UltraScale+ families all have LUT6s, 8-bit carry blocks
(LOOKAHEAD8, CARRY8) and 64-word distributed RAMs; they differ in their DSPs.
The target is selected by `target=`, with the family names of
`hw/syn/xilinx/platforms`:

| `target=` | Device | DSP multiplier | fabric → registered product | product → fabric | per cascaded DSP |
|---|---|---|---|---|---|
| `versal` | V80 | DSP58, 27 × 24 | 1 | 6 | 3 |
| `virtexuplus` | U200, U250 | DSP48E2, 27 × 18 | 1 | 5 | 2 |
| `virtexuplusHBM` | U50, U55C, U280 | DSP48E2, 27 × 18 | 1 | 5 | 2 |
| `zynquplus` | Zynq UltraScale+ | DSP48E2, 27 × 18 | 1 | 5 | 2 |

A product takes ⌈a/A⌉·⌈b/B⌉ DSPs (the better orientation). The cell counts
are Vivado's: on Versal, M_DATA, ALUMUX, ALUADD and OUTPUT out of a DSP,
and ALUMUX, ALUADD and OUTPUT through each cascaded one. A register fed by
the product is the DSP's own (MREG or PREG), and so is a LUT that reads that
register back, its clock enable or feedback; any other load pays the output
stage. The fabric-to-product value is fitted on the V80 data. A name that
is not a target is an error that lists the targets.

### 4.3 ASAP7

For the ASIC flow (`hw/syn/yosys`, ASAP7 7.5-track RVT, TT corner)
Beacon counts **cell stages**, not LUT levels. `target=asap7`
maps the same balanced and-inverter graph to 3-input cuts, each one cell
(NAND, AOI, MAJ, XOR), and models the word-level operators
as an ASIC builds them:

| Element | Stages |
|---|---|
| cell (`comb.truth_table`, 3 inputs) or unmapped and-inverter gate | 1 |
| adder, input bit j to output bit i | 2 + ⌈log₂(i − j + 1)⌉: a prefix adder |
| order compare | 1 + ⌈log₂ span⌉ |
| multiply | partial products, a Dadda tree of full adders (2 stages each), a prefix adder |
| table of k inputs (small products and dividers) | k (a multiplexer tree) |
| memory of flops (no macro) | 1 + ⌈log₂ depth⌉: a decoder and a multiplexer tree |

SRAM macros (`VX_dp_ram_asic`, `VX_sp_ram_asic`) are black boxes, as in the
flow's synthesis: their pins end and start paths. The report converts the
deepest path to a period:

```
deepest path 45 stages: 58 + 167 + 45 x 24 + 6 = 1311 ps (763 MHz)
                        clk→Q  path  stages × cell  setup
```

The cell (24 ps at fanout 3 and its own slew), clock-to-output (58 ps) and
setup (6 ps) delays come from the Liberty, by `hw/syn/yosys/liberty_stage.py`;
the path term (167 ps: the inverter after a DFFHQN's QN, buffering, slew) is
fitted on OpenSTA (§7). Re-derive them for another corner or threshold
voltage.

### 4.4 Retiming (optional)

All of Vortex's Vivado runs retime. With `retime-levels=N`, the `retimed`
column estimates each end point's depth after one hop of balancing to a
period of N levels. The end register moves backward into the room its
output stage leaves, and the start register moves forward into the room its
input stage leaves. Neither crosses a distributed RAM or a DSP, and only a
register Vivado can retime (synchronous reset) moves. Top ports and
black-box pins count as registers with empty neighbour stages, as a gate
harness makes them. It ranks no better than plain levels, so `levels` stays
the ranking; `retimed ≥ 12` at `retime-levels=10` is a short list of fix
candidates retiming is unlikely to absorb.

### 4.5 Area

The area mapping (§3.1) first takes out of the netlist what Vivado keeps
out of LUTs:

- **Clock enables and synchronous resets** (`foldsim-register-enables`). A
  select between a register's next value and the register itself is the
  flip-flop's CE pin, and a select of a constant under it the R or S pin;
  neither takes a LUT (Vivado's `VX_pipe_register`s with reset and enable
  have none). The register becomes a `seq.compreg.ce`; its enable and reset
  are one gate per register, not per bit.
- **Register bits that never leave 0** (`foldsim-constant-registers`). A
  flip-flop powers up at 0, so a bit that only loads 0 or itself, and resets
  to 0 if at all, is the constant 0, and the register narrows to its other
  bits (`VX_cache_amo`'s write-back offset, of which only bit 5 varies, makes
  its shift a select of two amounts).
- **Bits a select passes through** (`foldsim-narrow-muxes`). A procedural
  write of part of a vector in a loop elaborates to a select per iteration
  of the whole vector, whose arms agree on every bit outside that part;
  lowered to and-inverter gates, each such bit stays a pair of gates and
  every iteration's condition reaches it. `VX_cache_amo`'s read-forward
  loop over the L2's 512-bit word (64 bytes of 4 write-back slots and the
  settling entry) is a chain of 320 whole-word selects, every one reaching
  every bit. Each select narrows to the runs where its arms differ.
- **Styled ROMs** (`foldsim-styled-roms`). A constant table on a wire with
  `rom_style` `block` or `distributed` becomes a memory of its shape, placed
  as below (`VX_mufu`'s table: one block RAM, as Vivado places it, instead
  of 12,321 mapped LUTs of logic).

Then, per op:

| Element | FPGA | ASAP7 |
|---|---|---|
| LUT (`comb.truth_table`) | 1 LUT | 1 cell |
| Adder or subtracter | a LUT per bit into the carry chain | a prefix adder: w · (2 + ⌈log₂ w⌉) cells |
| Order compare | a LUT per two bits | 2 cells per bit |
| Multiply by a constant | a LUT per bit (shifts and adders) | 3 cells per bit |
| Multiply, ≤ 10 variable operand bits | a table: a LUT per output bit (at most as many as the operands' variable bits), per 64 entries past six bits | the same in cells |
| Multiply, wider | DSPs: the fewer of ⌈a/A⌉·⌈b/B⌉ in either orientation over the significant bits (§4.2) | partial products and a Dadda tree: 3·a·b cells |
| Divide or remainder (output bits: at most the dividend's) | a table over ≤ 10 variable bits; a constant divisor, 2 LUTs per bit; else a subtracter per quotient bit, a LUT per result bit | the same, but 3 cells per subtracter bit |
| Register | a flip-flop per bit | a flip-flop per bit |
| Vendor IP (`xil_*`) | its logic LUTs, flip-flops and DSPs: for `xil_fma`, `xil_fdiv` and `xil_fsqrt` the medians of their routed instances in the U55C builds (616, 743, 412 LUTs; 1,064, 1,360, 786 flip-flops) | — |
| SRAM macro (black box) | — | a macro |

**Memories** go where Vivado infers them. A `ram_style` the RTL asks for wins
(`VX_dp_ram` writes `block` for `FORCE_BRAM` sizes and `distributed` for
`LUTRAM`; the RAM fold, `ram_fold.py`, registers the read of the RAMs it
folds, so `VX_dp_ram` writes `block` for them too). Otherwise a
read of latency 0 is distributed RAM, and a registered read of at least 32
words and 2 Kb is block RAM. The shapes are the same on every FPGA family:

| Placement | Cost |
|---|---|
| distributed RAM | 0.57 LUTs per bit up to 32 words (RAM32M16), 1.14 per 64 words beyond (RAM64M8), per read port, plus a multiplexer tree over the 64-word LUTs |
| block RAM | the fewest 36 Kb blocks of any shape (512 × 72 … 32K × 1) or 18 Kb halves; a copy per two ports, a read-write port one of them |
| UltraRAM (`ultra`) | 4096 × 72 each |
| `registers` | flip-flops and a read multiplexer |

Two shapes override the inference:

- A memory distributed RAM cannot hold is **flip-flops**: one read of
  latency 0 at each of its words, at constant addresses (an associative
  lookup, as the MSHR's `addr_table`, `word_table` and `sector_table` are),
  or more than seven read ports besides a read-write port. Its
  constant-address reads are wires (in a copy folded over threads, a
  multiplexer over the threads), every other read a multiplexer per bit,
  and each word a write enable on its flip-flops' CE pins. The L2 MSHR goes from 629 to 348 distributed RAM LUTs
  and from 167 to 567 flip-flops per bank, against Vivado's 352 and 579.
- A block RAM writes 8- or 9-bit bytes under a mask, so a memory masked in
  **lanes that are not whole bytes** (neither a multiple of 8 nor of 9 bits)
  is a block RAM per lane: the L2 tag store (64 × 136 in eight 17-bit
  lanes) is eight 18 Kb halves, the I-cache's (64 × 80 in four 20-bit lanes)
  four, as Vivado builds them.
- A memory without a write port is a **ROM**: in distributed form, logic
  (a table over the address per bit and read, nothing for one word), not
  distributed RAM, which is how Vivado reports it. A memory nothing reads
  costs nothing.

On ASAP7, a memory without a macro is flip-flops, a write decoder and a
read multiplexer tree.

**The LUT scale.** The cut mapper has none of Vivado's LUT combining or
wide multiplexers (MUXF7/F8), and maps module by module, without the logic
Vivado trims across module boundaries, so it covers a design with 2.6× the
logic LUTs Vivado leaves at the geometric mean, from 1.4× (`lmem`) to 7.5×
(`rtu`) (§7.2). The target's `lutScale`
converts the mapping's LUTs to Vivado's: 0.391 on the LUT6 families, the
geometric mean over the 14 routed U55C `fpga_gate` DUTs of Vivado's logic
LUTs, less what Beacon prices outside the mapping (vendor IP, memory
multiplexers), over Beacon's mapped LUTs (§7.2). The V80 shares that value
without a fit of its own, and `asap7` has none (1.0): its cell counts are
not yet compared with Yosys. Vendor IP and distributed RAM, and the
memories' read multiplexers, are not scaled; `lut-scale=1` gives the raw cover.

## 5. Outputs

| File | Contents |
|---|---|
| `endpoints.csv` | one row per end point bit: `levels,retimed,kind,start_kind,fanout,crossings,start_inst,start,end_inst,end` |
| `instances.csv` | the instance tree: `id,parent,instance,module` (the `*_inst` columns' ids) |
| `report.txt` | counts, combinational loops, ops without a model, the histogram of end points by levels, the `top-paths` deepest paths op by op, and the deepest path of each `trace=` match |
| `beacon_summary.csv` | the DUT flow's one row: max levels, end points ≥ 12, ≥ 8, all, and the area |
| `paths.jsonl` | `paths=`: one JSON line per pair of start and end registers whose deepest bit has at least `path-levels` levels: that bit's levels, the pair's bit count, start, end, and `ops`, the path from the end back to the start as `[levels, kind, fanout, instance id, name, source]` |
| `beacon_units.json` | the deepest end point of every unit at ≥ 8 levels, and the 10 deepest paths |
| `beacon_paths.txt` | the DUT flow's `paths.jsonl` at ≥ 8 levels, by unit (`ci/beacon.py paths` for any run) |
| `area.csv` | one row per instance, depth first: `depth,path,module`, its own `lut,lutram,ff,bram,uram,dsp,macros`, then its subtree's (`total_*`); `lut` is logic LUTs, scaled (§4.5) |
| `area_memories.csv` | one row per memory: its instance, name, depth, width, read, write and read-write ports, read latency, `ram_style`, placement and cost |
| `area.txt` | the design's totals, the mapping's LUTs by op before scaling, and the vendor IP |
| `beacon_area.txt` | `area.txt`, the units with ≥ 1 % of the LUTs (one copy's subtree, × copies), and every memory with its placement |

`beacon_summary.csv` also holds the design's `lut` (distributed RAM
included, as Vivado's utilization counts it), `ff`, `lutram`, `bram`,
`uram` and `dsp`, and `beacon_units.json` an `area` map of every unit's
resources (one copy's subtree) and copy count.

A path in `report.txt`, deepest first, one op per line: levels at the op,
its kind, the fanout of its bit, and its instance path and name:

```
#1 levels 20: .../mem_req_queue/pending_size/used_r.d[0] (VX_socket__lift)
    20 lut    fo 0     .../comb.truth_table[0]
    19 carry  fo 1     .../comb.add[0]
    ...
     0 reg    fo 59    .../req_pending/in_fire_r[13]
```

`report.txt`'s deepest paths are often one pair of registers, bit after bit.
The path list has **every** deep path once:

- one per pair of start and end registers, its deepest bit, with the
  number of bits it stands for;
- pairs whose names differ only in generate indices (`g_banks_0`,
  `g_banks_1`) and bits fold into one, as units fold them, so a unit the
  gate names is a heading of the list;
- units deepest first, and in each unit its paths deepest first;
- each op with the **source line** it came from. Mapped LUTs have no name
  of their own, so the file and line are what locate them in the RTL. The
  DUT flow elaborates `hw/rtl` itself (not preprocessed copies), so the
  lines are the repository's.

```
== cache/g_cache_cache/g_banks_*_bank/g_amo_amo: 24 levels, 23 paths

  24 levels, 1024 d bits in 4 copies
  from cache/g_cache.cache/g_banks_0.bank/g_amo.amo/g_commit.chain_match_r[0]
  to   cache/g_cache.cache/g_banks_0.bank/g_amo.amo/g_commit.wbq_data.d[6]
      24 lut    fo 0     VX_cache_amo.sv:211      .../g_amo.amo/comb.truth_table[0]
      22 lut    fo 1     VX_cache_amo.sv:700      .../g_amo.amo/comb.truth_table[0]
      ...
      10 carry  fo 839   VX_cache_amo.sv:689      .../g_amo.amo/comb.sub[1]
      ...
       0 reg    fo 2     VX_cache_amo.sv:773      .../g_commit.chain_match_r[0]
```

On the full GPU, the 3,283 pairs at ≥ 12 levels fold into 169 paths in
69 units.

## 6. Integration

### 6.1 The DUT flow

`hw/syn/xilinx/dut` builds every DUT of `catalog.mk` for Vivado; its
`beacon` target runs Beacon on the same DUT with the same sources, defines and
`DEVICE` (the U55C by default):

1. `gen_sources.sh` into `beacon/sources.f`, a filelist of `hw/rtl` with
   the defines (not preprocessed copies, so source lines are the
   repository's);
2. `circt_elaborate.sh -f mlir` into `beacon/netlist.mlir`, the FP IPs as
   black boxes, with every op's source location;
3. `foldsim-beacon-map`, then `foldsim-estimate-timing`; and
   `foldsim-beacon-map{area}`, then `foldsim-estimate-area`; all with
   `target=` the family `gen_platform.py --device=$(DEVICE) --emit=family`
   names;
4. `beacon_summary.py`: `beacon_summary.csv`, `beacon_units.json`,
   `beacon_paths.txt` and `beacon_area.txt` (§5).

The ASIC flow has the same target, with the same sources, defines and SRAM
black boxes as its synthesis, and `target=$(PDK)`:

```bash
make -C hw/syn/yosys/dut om TARGET=beacon
# om_VX_om_core_top/beacon/report.txt: the deepest paths and the period
```

### 6.2 `logic_gate`

The third synthesis gate, beside `fpga_gate` (Vivado, nightly) and
`asic_gate` (Yosys + OpenSTA, nightly); see
[continuous_integration.md §3.5](continuous_integration.md). It runs Beacon on
`fpga_gate`'s builds (`ci/testcases/logic_gate.yaml`; `top` as its Vortex
core, `top_vortex`, since `VX_cp_core` keeps LLHD reference ports) in about
four minutes, and checks each against `ci/baselines/synthesis/beacon/`:

- a **unit** that gets deeper fails, by name;
- a **new unit** at or above the fix threshold (12) fails;
- a unit that gets shallower, or leaves the watch list, is an improvement
  to lock in by re-recording;
- end point counts are recorded, not gated (they grow with lanes and warps);
- a **resource** of the design (`lut`, `ff`, `lutram`, `bram`, `uram`,
  `dsp`) more than 5 % above its baseline fails, `fpga_gate`'s LUT
  tolerance (catalog `thresholds` and `--metric-threshold` override it);
  5 % below is an improvement to lock in. A memory that moves between
  distributed and block RAM shows as both columns moving.

A failing build names its `beacon_paths.txt`, where the failing unit is a
heading with its paths op by op, its `beacon_area.txt`, and its
`beacon_risk.txt`, the registers the learned calibration (§7.1) predicts
slowest; CI uploads them with the build's logs. The calibration's Fmax and
count of likely failures are columns of the report (`p_fmax`, `risk`), not
gated: a wire-dominated path that levels rank low shows there first.

Depth is deterministic, so the tolerance is zero. Area is as
deterministic, but a feature adds some by design, so it has the Fmax gate's
tolerance. The baseline fingerprint
is the CIRCT and foldsim versions, `DEVICE` and `XLEN`: a new foldsim reads
as STALE, not as a regression. Baselines are golden data, re-recorded only
by a human (`--update-baseline`), as for the other gates.

### 6.3 RTL timing work: Beacon first

Beacon is the inner loop of any RTL change that can move timing. A DUT takes
minutes in Beacon and hours in Vivado or Yosys, so a deeper path is found and
fixed while the change is still open, and synthesis runs once, to confirm.

1. **Reference.** Beacon on the parent revision, once per base, in a build
   tree of its own:

   ```bash
   git worktree add ../ref HEAD
   ../ref/ci/logic_gate.py -b core -b lmem --build-dir $PWD/build32_beacon_ref
   ```

2. **Change.** The same builds on the working tree, each compared with the
   reference register by register:

   ```bash
   ci/logic_gate.py -b core -b lmem --against build32_beacon_ref
   ```

   A change behind a non-default knob is compared at that setting: give
   both runs the same `--configs="-D..."`, which appends the flags to every
   selected build's catalog configuration. The catalog baselines are not
   gated then, and none can be recorded.

   Besides the per-unit verdict against the golden baseline, each build
   writes `beacon_diff.txt` (`ci/beacon.py diff`). Its headline is the
   deepest register, the mean of the 50 deepest and the count at the watch
   threshold, before and after; then it lists the registers that got deeper,
   new deep registers, and the ones that got shallower, then the units whose
   deepest register moved. With area, the headline adds the LUTs, block RAMs
   and DSPs before and after, and a section lists each resource of the
   design and the units whose own area moved, all copies; a resource more
   than 5 % larger fails the run (`AREA`). The run fails (`DEEPER`) on a register 4 or more
   levels deeper, one that reaches the fix threshold (12) from below, or a new
   one at it. These limits come from 13 routed U55C builds: a register 4 or
   more levels deeper lost slack in Vivado in every case (170 of 170), while
   1 to 3 levels is within the placer's noise, so those are listed as
   "deeper by less" to watch, not failed. Of the headline numbers, the top-50
   mean follows Vivado's worst slack (Spearman 0.56) and the single deepest
   register does not (0.29). A new register below 12 is listed but passes,
   since a rename looks the same.

3. **Fix.** `ci/beacon.py paths <build>/beacon --unit <unit>` prints a deep
   unit's paths op by op. Edit, and repeat step 2 until nothing got deeper,
   or until each deeper path is one the change intends.

4. **Confirm.** Run synthesis once per change, not once per edit: the
   `fpga_gate` or `asic_gate` builds of the DUTs whose depth moved. Then feed
   the result back. `ci/beacon.py validate <build>/beacon ep.csv` compares the
   run with Vivado's end point table, and a disagreement there is a Beacon bug
   to fix in foldsim.

Beacon counts logic levels and cells; it does not see wire delay,
placement, SLR crossings or congestion. A change that moves those (a wide
fan-out, a floorplan) still needs synthesis before it lands. Compare a fix
against a native gate (`hw/syn/xilinx/dut` or the OOC gates), not the folded
design (§9).

## 7. Validation

Against Vivado, per RTL register or memory: Beacon's deepest bit
against Vivado's deepest failing path and worst slack, on seven V80 datasets
with no new Vivado runs (`ci/beacon.py validate`):

| Dataset | Vivado run | Spearman vs levels | Spearman vs slack | ≥ 12: precision / recall |
|---|---|---|---|---|
| full GPU, folded | 4.0 ns, routed, 270 k end points | 0.75 | 0.45 | — |
| socket | native gate, 4.0 ns, routed | 0.55 | 0.53 | 0.64 / 0.31 |
| D-cache | native, 4.0 ns, routed | 0.71 | 0.55 | 0.67 / 0.38 |
| top | native gate, 3.333 ns, placed | 0.53 | 0.39 | 0.19 / 0.53 |
| socket | native gate, 3.333 ns, placed | 0.50 | 0.33 | 0.52 / 0.18 |
| core | native gate, 3.333 ns, placed | 0.93 | 0.62 | 0.81 / 0.13 |
| compute slice | native gate, 3.333 ns, placed | 0.87 | 0.75 | 0.58 / 0.57 |

Beacon's depths run about 1.25× Vivado's on the native gates. The U55C
profile was checked against `fpga_gate`'s critical paths: it agrees with
the V80 profile wherever no DSP is on the path, and the remaining
disagreement is the same on both devices.

**ASAP7**, against OpenSTA on fresh syntheses of five DUTs (Yosys + ABC as
`hw/syn/yosys` builds them, flip-flops named after their registers,
`hw/syn/yosys/sta_endpoints.tcl` for the slack of each):

| DUT | Registers | Spearman, stages vs arrival | Worst path: OpenSTA → Beacon |
|---|---|---|---|
| om | 93 | 0.91 | 827 → 879 ps |
| cache | 526 | 0.82 | 1450 → 1311 ps |
| dxa | 169 | 0.56 | 1018 → 1215 ps |
| raster | 134 | 0.58 | 879 → 879 ps |
| tex | 1002 | 0.93 | 840 → 807 ps |

Per register, the fitted delay per stage is 17–25 ps, the library's 24.
The worst path is within a mean 8 % on these five, from which the path term
was fitted; on the six larger DUTs, the predicted Fmax is within a mean
13 % (−32 % to +3 %) of the asic_gate baselines, some of which predate the
current RTL. Beacon runs a DUT in 2–52 s, where Yosys and OpenSTA take
2.5–37 min.

### 7.1 Learned calibration (advisory)

Levels rank Vivado's near-critical registers at about half the ceiling set
by one Vivado run predicting another. `ci/beacon.py score` ranks them with a
calibration learned from routed U55C builds instead: gradient-boosted trees
on 38 per-register features, all from Beacon's own outputs (no Vivado input):

- **depth:** levels, mean levels over the bits, the share of bits within one
  level of the deepest;
- **width:** bits, and bits within one level of the deepest;
- **end and start kind:** D, RAM write data or control, RAM read; register,
  port, BRAM, IP;
- **fan-out:** the deepest start's, the cone's (`in_fanout`), the largest
  along the deepest path, nets of 64 or more loads on it;
- **span:** hierarchy depth, crossings, the bits under the lowest common
  instance of the register and its deepest start;
- **the deepest path's op mix** (`paths.jsonl`, at 8 levels or more): levels
  spent in LUT, carry, compare, LUTRAM, BRAM and DSP ops, distinct
  instances and source files, and the levels with a carry level costed at
  0.42 of a LUT level and a compare at 0.15.

The model is three tree ensembles: the worst slack (capped at 1 ns, the
`ep.csv` horizon), the probability of slack under 0.3 ns (near-critical) and
under 0 ns. A register never ranks safer for being deeper (a monotone
constraint on levels). `calibrate` trains them with scikit-learn and writes
them as plain JSON trees; `score` evaluates them in pure Python, in about
4 s for a core. The model trained on the 28 builds below ships as
[`hw/scripts/beacon_calibration_u55c.json`](../../hw/scripts/beacon_calibration_u55c.json);
it applies to the U55C, its slacks at 300 MHz.

```bash
ci/beacon.py score build_core/beacon hw/scripts/beacon_calibration_u55c.json \
    --top 30 --csv score.csv
ci/beacon.py calibrate cal.json --holdout \
    --build build_core/beacon build_core/ep.csv core \
    --build build_cache/beacon build_cache/ep.csv cache
```

Each `--build` is a DUT build's Beacon run (`path-levels=8`, as the DUT flow
writes it) and the end point table of the same sources. `--holdout` prints
the check below for the given builds.

**Validation.** 13 routed U55C builds at 3.333 ns: 8 core revisions, 4 lmem,
1 cache, 19 k registers. Each build is held out and scored by a model
trained on the builds outside its group. Revisions of one RTL lineage share
most of their registers, so they are held out together:

- core with the coretiming lineage (origin, it1, it2, lsuagu);
- core with the lmemwarp lineage (v1, v2, final, p2lsulmem);
- cache;
- lmem.

The stricter test holds out the whole DUT (core trained on cache and lmem
only).

| held out | near-critical AP: levels | levels + 0.35·log2(fan-out) | carry/compare cost | model, lineage out | model, DUT out | Spearman vs −slack: levels / model |
|---|---|---|---|---|---|---|
| core origin | 0.36 | 0.38 | 0.41 | 0.72 | 0.65 | 0.15 / 0.64 |
| core it1 | 0.40 | 0.42 | 0.42 | 0.75 | 0.65 | 0.18 / 0.63 |
| core it2 | 0.44 | 0.46 | 0.48 | 0.77 | 0.68 | 0.24 / 0.68 |
| core lsuagu | 0.68 | 0.70 | 0.70 | 0.80 | 0.71 | 0.33 / 0.54 |
| core lmemwarp v1 | 0.37 | 0.40 | 0.48 | 0.75 | 0.56 | 0.27 / 0.66 |
| core lmemwarp v2 | 0.44 | 0.50 | 0.54 | 0.82 | 0.61 | 0.35 / 0.73 |
| core lmemwarp | 0.37 | 0.41 | 0.52 | 0.81 | 0.59 | 0.32 / 0.64 |
| core p2lsulmem | 0.55 | 0.63 | 0.60 | 0.87 | 0.65 | 0.40 / 0.78 |
| cache | 0.47 | 0.57 | 0.52 | 0.65 | 0.65 | 0.50 / 0.56 |
| **mean** | **0.44** | **0.50** | **0.52** | **0.77** | **0.64** | **0.30 / 0.65** |

On the three lmem builds, with 4–5 near-critical registers each, the AP goes
from 0.02–0.03 to 0.63–0.68. One Vivado run predicting another of nearly
the same RTL (it1, it2) reaches 0.80. AP for failing registers goes from
0.21 to 0.44; that set is mostly placement noise at a WNS near −0.1 ns.

- **Width carries most of it.** Shuffling the bit counts costs 0.29 of AP.
  This is partly selection: a wide register shows the worst of many bits.
  Within registers of one deep bit, though, the AP still goes from 0.47 to
  0.79.
- **Span, the op mix and fan-out carry the rest** (0.03–0.08 each). The op
  mix matters most on the cache (0.27).
- **The DUT-out column is the honest number for a new DUT.** The lineage-out
  column is the one for a new revision of a DUT already trained on.
- **It does not track a whole design's slowdown.** The mean of the 50 lowest
  predicted slacks follows the change in WNS over 10 revision pairs worse
  than the top-50 mean of levels: Spearman 0.15 (lineage out) or 0.42 (DUT
  out), against 0.53. The lsuagu regression, −1.3 ns, moves it by 0.02 ns.

**Retrained on the gates' DUTs.** The shipped model adds the routed
`fpga_gate` builds of 13 DUTs (cache, core, dxa, gfx, lmem, om, raster, rtu,
tcu, tensor, tex, vm, and `top` as its Vortex core), 11 of them at 3.333 ns
and four at 4.0 ns, to the 13 builds above: 28 builds. Slacks at 4.0 ns are
moved to 3.333 ns (`xilinx_ep_dump.tcl`'s reference period). Each DUT is
held out (all its builds) and scored by a model trained on the rest:

| held out | near-critical AP: levels / model | Spearman vs −slack: levels / model | Fmax: Vivado / model |
|---|---|---|---|
| cache | 0.61 / 0.75 | 0.54 / 0.64 | 297.3 / 277.5 |
| core | 0.33 / 0.68 | 0.33 / 0.66 | 300.6 / 280.1 |
| dxa | 0.56 / 0.80 | 0.41 / 0.52 | 306.0 / 289.3 |
| gfx | 0.68 / 0.79 | 0.42 / 0.58 | 246.5 / 266.2 |
| lmem | 0.07 / 0.32 | 0.25 / 0.35 | 308.0 / 331.3 |
| rtu | 0.37 / 0.33 | 0.35 / 0.42 | 310.6 / 275.6 |
| tcu | 0.55 / 0.65 | −0.01 / 0.10 | 295.2 / 265.5 |
| tensor | 0.69 / 0.90 | 0.40 / 0.66 | 240.6 / 240.1 |
| tex | 0.23 / 0.24 | 0.25 / 0.18 | 298.2 / 282.8 |
| top (Vortex core) | 0.59 / 0.73 | 0.29 / 0.46 | 243.0 / 265.3 |
| vm | 0.78 / 0.88 | 0.47 / 0.61 | 251.6 / 243.5 |
| **mean** | **0.50 / 0.64** | **0.34 / 0.47** | **±7 %** |

(om and raster have no near-critical register.) Before this retrain, the
shipped model already put the registers that failed `tensor` and `top` at
250 MHz first: of 2884 and 2574 registers, the tensor socket arbiter's
output buffer ranks first by P(fail), and top's flush-arbiter output buffer
first and second by predicted slack. Both are wide enables driven across an
instance by a same-cycle lookup (an L2 tag compare, a TLB match), which
levels rank in the hundreds.

The design's Fmax, from its slowest predicted register, is within about 7 %
of Vivado's held out, misses top's failure, and on the core lineage moves
by nothing where Vivado moved by +4.5 % (it1) or −27 % (lsuagu, held out).

So the model is a ranking for where to look first, not a gate. `diff` and
`logic_gate` keep gating on levels; `logic_gate` records the model's Fmax
and the count of registers at P(fail) ≥ 0.5 (`p_fmax`, `risk`, not gated)
and writes each build's 40 lowest predicted slacks to `beacon_risk.txt`.

**More builds.** The DUT flow writes every routed build's end point table
(`ep.csv`, at 3.333 ns, through `hw/scripts/xilinx_ep_dump.tcl`), and
`fpga_gate` uploads it. A build's Beacon run is regenerated from its commit
(Beacon is deterministic), so each nightly `fpga_gate` adds a build of every
DUT to train on; `calibrate --holdout` is the measure of whether the Fmax
has become good enough to gate.

### 7.2 Area

Against the routed `fpga_gate` builds of the same sources (U55C, Vivado
2024.2.2, 1 November 2026):

Logic LUTs are Vivado's logic LUTs (without distributed RAM and shift
registers) against Beacon's at the fitted scale; *before* is the estimator
before the rules of §4.5 above, at its earlier scale of 0.337 and IP costs.
`top` leaves out the command processor (`cp_core`) on both sides: CIRCT
elaborates it with LLHD signals Beacon does not read.

| DUT | LUT ratio before | LUT (Beacon / Vivado) | FF | distributed RAM LUTs | block RAM | DSP |
|---|---|---|---|---|---|---|
| cache | 1.19 | 27,313 / 28,317 (0.96) | 34,922 / 25,280 (1.38) | 2,542 / 2,722 | 32 / 32 | 0 / 0 |
| core | 0.86 | 122,879 / 165,478 (0.74) | 158,903 / 146,008 (1.09) | 6,838 / 6,220 | 129 / 137 | 97 / 96 |
| dxa | 1.05 | 8,170 / 9,666 (0.85) | 9,215 / 7,132 (1.29) | 568 / 638 | 0 / 0 | 9 / 7 |
| gfx | 1.45 | 241,768 / 210,879 (1.15) | 379,837 / 236,308 (1.61) | 11,906 / 12,834 | 191.5 / 172.5 | 1,140 / 992 |
| lmem | 0.68 | 14,766 / 26,322 (0.56) | 14,361 / 13,623 (1.05) | 726 / 740 | 16 / 16 | 0 / 0 |
| mufu | 2.92 | 1,387 / 1,966 (0.71) | 1,806 / 1,105 (1.63) | 0 / 0 | 1 / 1 | 5 / 3 |
| om | 2.37 | 4,730 / 2,298 (2.06) | 4,456 / 3,496 (1.27) | 367 / 408 | 0 / 0 | 10 / 22 |
| raster | 1.00 | 5,044 / 7,981 (0.63) | 12,069 / 8,974 (1.34) | 496 / 606 | 8 / 8 | 14 / 14 |
| rtu | 2.01 | 99,325 / 63,721 (1.56) | 146,917 / 94,885 (1.55) | 1,904 / 1,840 | 16.5 / 18.5 | 140 / 140 |
| tcu | 1.04 | 121,388 / 115,727 (1.05) | 49,056 / 42,675 (1.15) | 915 / 1,360 | 0 / 0 | 769 / 768 |
| tensor | 1.12 | 220,251 / 225,309 (0.98) | 166,688 / 138,986 (1.20) | 8,962 / 7,842 | 116.5 / 133.5 | 267 / 265 |
| tex | 1.50 | 71,506 / 55,791 (1.28) | 96,949 / 34,582 (2.80) | 3,241 / 4,208 | 0 / 0 | 896 / 528 |
| top | 1.04 | 104,728 / 125,511 (0.83) | 157,573 / 144,522 (1.09) | 8,551 / 7,836 | 76 / 76 | 50 / 48 |
| vm | 1.55 | 109,070 / 116,552 (0.94) | 151,175 / 121,587 (1.24) | 10,276 / 10,366 | 168 / 172 | 0 / 0 |

- **LUTs**: the geometric mean error over the 14 DUTs goes from 41 % to
  31 %, the range from 0.68–2.92 to 0.56–2.06. The L2 of `vm` came to
  105,071 LUTs before (311,783 mapped) and comes to 47,555 now (121,625
  mapped), against Vivado's 54,150; each bank's `VX_cache_amo` to 14,733
  before and 3,608 now, against 5,001. The scale is one number per family,
  so a DUT whose logic Vivado trims or packs more or less than the average
  moves by that much; `logic_gate` compares a DUT with its own baseline, so
  the scale cancels there. The largest
  residues are logic Vivado removes across module boundaries, which Beacon,
  mapping module by module, keeps: `rtu`'s logic outside its 73
  floating-point cores is 19,536 LUTs and 18,088 flip-flops in Vivado against 53,970 LUTs
  (138,032 mapped) and 68,357 flip-flops in Beacon; `tex` keeps 2.8×
  Vivado's flip-flops; `om`'s `VX_om_dcr` and `VX_om_blend` come to 3,407
  LUTs against Vivado's 741. `lmem`, `raster` and `core` come out low.
- **Flip-flops** run 5–63 % high, and 2.8× on `tex`, for the same reason:
  Vivado drops the bits a parent never reads or ties to a constant (the
  cache DUT's stream buffers, 20,936 flip-flops in Beacon against 12,324), and moves pipeline registers into DSPs and block RAM
  output registers, none of which a module-by-module count sees.
- **Distributed RAM** is within 15 % on 10 of the 13 DUTs that have any
  (`raster` 0.82, `tex` 0.77, `tcu` 0.67).
- **Block RAM** is exact on 5 of the 10 DUTs that have any; the mask-lane
  rule matches Vivado on every masked tag store. The RTU scheduler's
  `window_store` (64 × 512 in eight 64-bit lanes) packs in 7.5 blocks, its
  lanes being whole bytes, where Vivado builds a 36 Kb block per lane, 8.
  `gfx` is 19 high: Beacon places four RTU scheduler memories without a
  `ram_style` (`fbuf_ram`, `ray_store`, `xfres_ram`, `trires_ram`, 23.5
  blocks) in block RAM by its size and latency rule, and Vivado does not;
  `core` and `tensor` are short by 8 and 17, not yet traced.
- **DSPs** are within two on 11 DUTs; `gfx` and `tex` are 148 and 368 high
  and `om` 12 low, where Vivado chooses between DSPs and LUTs for narrow
  products by a heuristic of its own.

The repository's Vivado baselines (`ci/baselines/synthesis/xilinx/`) date
from 5 October 2026 and predate later RTL changes, so they are not a
reference for this table.

## 8. Diagnostics beyond levels

**Port depths.** `port-modules=` names modules whose instances' ports
Beacon reports: the levels at which each input arrives, the levels behind
each output, and `through`, the bits of each output that a combinational
path from outside the instance reaches. A module that registers its
outgoing interface has `through` 0 on every output; `logic_gate` gates it
for the modules a build lists under `interfaces` (`BEACON_INTERFACES` in
the DUT flow). Small modules are inlined by the mapping, so keep them with
`run --keep`:

```bash
ci/beacon.py run --keep VX_elastic_buffer design.mlir runs/eb \
    port-modules=VX_elastic_buffer
ci/beacon.py buffers runs/eb      # the OUT_BUF advisor
ci/beacon.py ports runs/eb        # every kept module, by unit
```

- The **buffer advisor** lists the `VX_elastic_buffer`s that pass a deep
  ready chain through combinationally (`ready_in`, upstream, driven at
  least as deep as `ready_out` arrives from downstream), where a skid
  buffer (SIZE 2) or `OUT_BUF` would cut it, and those whose output is
  computed rather than registered, where `OUT_REG` would register it. It
  is evidence for an RTL review: the change also moves cycles, so it comes
  with the matching SimX timing update.
- For foldsim, the ports of the class modules (`VX_socket__lift`,
  `VX_core__lift`, ...) show the outputs computed from logic, and how deep,
  rather than registered: in the fold they cost host cycles on the class's
  links.

**Vivado probes.**

`hw/scripts` has three Vivado Tcl probes for questions
Beacon cannot answer, run on a checkpoint
(`vivado -mode batch -source <probe> -tclargs <dcp> ...`):

| Probe | Answers |
|---|---|
| `xilinx_cone_dump.tcl` | the worst path into a register, cell by cell, following retimed start registers back |
| `xilinx_cone_probe.tcl` | whether paths exist between two register sets, and how deep Vivado built them |
| `xilinx_wire_probe.tcl` | per path: length in tiles, bounding box and fanout beside logic and net delay |

## 9. Invariants and limitations

- **Depth, not slack.** On the placed 3.333 ns socket gate, wires are 83 %
  of the worst paths' delay; levels explain 1 % of net delay, and the real
  placement two thirds. No pre-placement wire term ranked slack better than
  levels on every held-out gate, so Beacon has none. Route-dominated
  failures with few levels are invisible to it: recall at ≥ 12 levels is
  0.13–0.57. The learned calibration (§7.1) ranks near-critical registers
  better from width, span and fan-out (held out: AP 0.64 against 0.50 over
  11 DUTs), and `logic_gate` reports it, but its Fmax is within about 7 %
  of Vivado's, so it is advisory.
- **Retiming and packing.** Vivado retimes and packs across what the RTL
  separates: a popcount, its adders and a compare become a LUT6 tree with a
  retimed register inside. Beacon's depths are RTL stage depths, about
  1.25× Vivado's.
- **The fold.** In foldsim's folded designs, the fold turns some registers
  into LUTRAM and cuts some chains, so its paths differ from the RTL's. The
  native gates are the reference for native RTL.
- **Black boxes.** Vendor IPs and `black-boxes=` modules end and start
  paths at their pins: their insides are not timed. Vendor IP's area is
  its measured cost per instance; a `black-boxes=` module's is left out.
- **Area is a count, not a utilization report.** The mapping is a cut
  mapper's cover scaled to Vivado's average (§4.5, §7.2), memories go by
  Vivado's inference rules, and what Vivado packs into DSP, block RAM and
  SRL cells Beacon counts as fabric. Logic that nothing observable reads is
  counted; Vivado removes it.
- **Devices.** The V80 and the U55C (Vivado), and ASAP7 RVT TT (the Yosys
  flow). The ASAP7 numbers are Beacon's period estimate, a sizing
  guide, not a sign-off: they come within about 10–30 % of OpenSTA.
- **Toolchain.** The catalog maps only with three CIRCT Synth fixes
  (zero-width mux, constant SOP cut, a failing longest-path query), which
  reach CI through the foldsim prebuilt.
