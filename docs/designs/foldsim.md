# Foldsim — threaded FPGA simulation of Vortex, on CIRCT

Foldsim runs Vortex on the Alveo V80 as a host-decoupled, cycle-exact
simulator in which every class of identical units — sockets, issue slices,
ALUs, FPUs, load-store units — is time-multiplexed through **one physical
copy** (FAME-5 model multithreading). A GPU many times the card's size fits
on it, and runs orders of magnitude faster than Verilator.

| § | contents |
|---|---|
| [1](#1-overview) | Overview and how to run it |
| [2](#2-the-compiler) | The compiler: netlists, threading, composition |
| [3](#3-the-simulator) | The simulator: threads, channels, the token fabric; timing options; domains and relays |
| [4](#4-the-kernel) | The kernel: memory and control endpoints |
| [5](#5-the-runtime) | The runtime and its register contract |
| [6](#6-validation) | Validation |
| [7](#7-measured-results) | Measured results |
| [8](#8-invariants-and-limitations) | Invariants and limitations |

---

## 1. Overview

![Foldsim architecture](../assets/img/foldsim_architecture.svg)

The whole flow is one IR from SystemVerilog to Verilog: `circt-verilog`
imports Vortex, Foldsim's passes cut it into models and compose the
simulator, `firtool` writes it out, and the card's native flow (IP-XACT
packaging, `slashkit link`) builds the image. Foldsim is a project of its own
that builds on CIRCT, as POCL builds on LLVM: the `foldsim` driver and
`foldsim-opt` come from the toolchain (`$(TOOLDIR)/foldsim`), CIRCT from
`$(TOOLDIR)/circt`. Nothing else is in the flow: no Yosys, no sv2v.

```bash
export FOLDSIM_CLASSES="VX_socket__lift VX_issue_slice__lift VX_alu_int__lift VX_alu_muldiv__lift VX_lsu_slice__lift+VX_lsu_scheduler__lift VX_fpu_std"
export FOLDSIM_LIFT="VX_socket:drop=INSTANCE_ID;port=SOCKET_ID VX_issue_slice:drop=ISSUE_ID VX_alu_int:port=BLOCK_IDX VX_alu_muldiv VX_lsu_slice:port=CORE_ID VX_lsu_scheduler"
CONFIGS="-DVX_CFG_NUM_CORES=2 -DVX_CFG_NUM_WARPS=4 -DVX_CFG_NUM_THREADS=4 -DVX_CFG_ISSUE_WIDTH=2"
CONFIGS="$CONFIGS" ./ci/blackbox.sh --driver=foldsim --app=sgemm                 # Verilator
make -C hw/syn/foldsim CONFIGS="$CONFIGS"                                         # the image
FPGA_BIN_DIR=$PWD/hw/syn/foldsim/build32_foldsim_hw/bin CONFIGS="$CONFIGS" \
  ./ci/blackbox.sh --driver=foldsim --target=hw --app=sgemm                       # the V80
```

`FOLDSIM_CLASSES` names the modules threaded as classes; a class may contain
others (the socket contains the issue slices). `FOLDSIM_LIFT` gives
interface-port units a plain-port boundary first, so they can be classes
(`hw/syn/foldsim/lift_boundaries.py`, a source-to-source pass on pyslang; the
RTL is not changed). `port=P` turns an instance-ID parameter into an input
port, and every parameter derived from it below, so that all instances of the
class elaborate to one module: the socket's `SOCKET_ID` carries every core's
`CORE_ID` with it. The tools are found through `FOLDSIM` (the driver) and
`CIRCT_BIN`, by default `hw/syn/common.mk`'s `FOLDSIM_PATH` and `CIRCT_PATH`.

A class `A+B` fuses two modules of the same parent (`foldsim-fuse-models`):
each `A` is paired with the one `B` it exchanges bits with, and each pair is
simulated as one model, so a path from one through the other and back takes
a thread one host cycle instead of a round trip through their channels. The
LSU slice and its scheduler are such a pair (the slice's request, the
scheduler's `ready` on it, the slice's `ready` on that): fused, tinygpu runs
at 3.00 host cycles per target cycle instead of 4.00 (V80), with the same
threads and copies and fewer LUTs.

The flow builds the configuration's FPU, as the native Vivado flows do: of
type DSP, the Xilinx floating-point cores. Their netlists are encrypted, so
their state cannot be threaded; each core declares a pipeline contract
(`hw/scripts/xilinx_ip_stubs.sv`: latency, clock enable, data ports, no other
state), and the compiler simulates each by one unthreaded copy of the core and
a per-thread store of its results (`foldsim --fold-pipelines`, see the
compiler's `foldsim-find-pipelines`). `hw/scripts/check_ip_contracts.py` checks
each contract's latency against the core's generation, the RTL and the
configuration; the in-process simulator runs the cores' models
(`sim/foldsim/xilinx_fp_models.sv`). `-DVX_CFG_FPU_TYPE_STD` in `CONFIGS`
selects the soft FPU. A RAM that asks for block RAM but reads asynchronously reads
synchronously once the target is flattened (CIRCT `seq-retime-firmem-reads`,
see the hardware library's "Block RAM and asynchronous reads");
`gen/memories.rpt` lists each such RAM and the decision.

---

## 2. The compiler

The `foldsim` driver drives four steps, all in one MLIR module of the HW,
Comb and Seq dialects; steps 2 to 4 run in `foldsim-opt`:

1. **Import and flatten.** `circt-verilog` elaborates the target
   (`foldsim_target`: Vortex with its AXI4 port, DCR, start and busy as plain
   ports) with `--detect-memories`; everything but the class modules is
   inlined.
2. **Cut netlists, innermost first** (`foldsim-extract-models{netlist}`). A
   module with class instances becomes `<module>_model` — its own logic, with
   ports for what it exchanges with the instances — and `<module>` becomes a
   netlist of the same ports, instantiating `<module>_model` beside the class
   instances, wired as before. Instances of the module elsewhere stay valid,
   so the socket's netlist is cut out of the top in turn. Bits of a model
   output that are always the same constant stay in the parent, and an input
   driven only by the parent's inputs and constants, or exactly by another
   model's output, is wired to those directly. Small logic (up to 64 ops)
   that drives the same input of every instance of a model from other models'
   outputs is moved into the model (`absorb-logic`), so that input comes from
   those models rather than through the parent's model.
3. **Clean up the models**: `hw-expand-array-writes` (a conditional write of
   one array element becomes a select per element), `seq-reg-const-prop`,
   `comb-share-adders` (`mux(c, a+b, a-b)` → one adder),
   `comb-low-bits-narrowing` (adders whose operands have known-zero low bits),
   canonicalization, and CSE after a topological sort (in a module body, CSE
   cannot merge a duplicate whose users come first). These are what a
   threaded model's area is sensitive to; ExportVerilog emits sign-extended
   multiplies as `$signed` so Vivado maps them to single DSPs.
4. **Compose** (`foldsim-compose`). The top netlist's nested netlists are
   inlined, every instance of a model becomes one thread of that model's
   single threaded copy (§3), and every wire between them a channel of the
   token fabric. The result is `foldsim_sim`: `hostClock`, `hostReset`, a
   channel per target port (`<port>_valid`, `_ready`, `_bits`) and a
   `profile_finish` bit per class.

The driver also writes `foldsim_sim.json` (the ports, and each class's thread
count) and `foldsim_sim.vh` (the class count, for the kernel).

---

## 3. The simulator

**Threads.** A model of N instances is one module that simulates them in turn
(`foldsim-fame5`'s threader): every register becomes an N-entry memory read at
the current thread, every memory N times deeper. A thread finishes its
target cycle when each input channel has its token and each output's token
was taken; its state is then written, on the host clock with that finish as
the enable — no gated clocks. Outputs that depend on no input fire eagerly,
the next cycle's value as soon as the thread finishes. The thread index
advances every host cycle and is a port of the module, so the fabric reads it
rather than keeping its own counter.

**Channels.** Every output of a model, and every input of the target, is a
*producer*; every input of a model, and every output of the target, a
*consumer* that reads bits of one or more producers, for each of its threads.
Each producer thread has two token slots and a count of tokens written; each
reader of it a count of tokens read. A producer is ready while every reader
of the thread has read all but one of its tokens; a consumer's token is valid
when each producer it reads has one for it. This is a two-entry channel
with no flow-through, and with fan-out, so the target's behaviour does not
depend on the host's schedule.

**Storage.** A producer read only by threaded consumers that read the same
bits of it for every thread keeps its slots in a memory addressed by the
reader's thread and slot (LUTRAM); any other, in registers. A target port
that is exactly one port of a model of one thread is wired to the shell
directly, since the shell's valid and ready depend on its own state alone.

### 3.1 Timing: the options the image flow uses

A threaded model's critical paths run from its thread index, through every
state memory's read and the target logic, into every state memory's write.
The image flow's default options (`FOLDSIM_FLAGS` in `hw/syn/foldsim`) cut
those paths without changing what the target computes; each keeps the
simulator cycle-exact (§6).

| option | what it does |
|---|---|
| `--region-loads 2048` | splits a model's state into regions of at most 2,048 loads, each with its own copy of the thread index and of the edge, written a host cycle after the target edge |
| `--registered-state` | reads the next thread's state a host cycle ahead into registers, forwarding the write landing in between; threads then go in turn (`--schedule`) |
| `--commit-select` | a model of one thread commits its state by selecting a registered candidate, not by writing it in the edge's cycle |
| `--state-stages 2` | writes the state two host cycles after its edge in models of three threads or more, so synthesis can retime into the logic before it |
| `--pipeline-multipliers` | with two stages, a state computed by a multiplier takes its first stage at the multiplier's inputs |
| `--staged-ram-writes` | with regions, memories with a registered read are written a host cycle late too, their reads forwarded |
| `--staged-tokens` | a channel writes each token into its slot a host cycle after it is taken, forwarded to its readers until then |
| `--fold-pipelines` | simulates a fixed-latency pipeline by one unthreaded copy and a per-thread store of its results (the Xilinx FP cores, §1) |

`--compact-staging` (staging placed at the state it stages; for large images,
s2w4i2's Verilog 34 → 16 MB) and `--channel-depth` are not in the default.

Out-of-context results for s2w4i2 (`make -C hw/syn/foldsim ooc`, xcv80,
4.0 ns unless noted; Vivado retiming moves a run by about ±20 MHz):

| options | FPU | MHz | LUTs | FFs |
|---|---|---:|---:|---:|
| none | soft | 186.8 | 130.1k | 26.8k |
| regions of 2,048 loads | soft | 214.3 | 136.9k | 83.6k |
| the default set but folding | soft | 250.0 | 211.8k | 174.7k |
| the default set | soft | 235.3 | 194.6k | 157.0k |
| the default set | Xilinx cores | 250.0 (262.7 at 3.33 ns) | 182.8k | 162.4k |

The soft FPU's own pipeline stages limit it: the same options with ready
scheduling instead of threads in turn reached 250.0 MHz, the difference being
which stages retiming balanced.

Small targets pay area for these options that they do not need;
`FOLDSIM_FLAGS=` (empty) builds the plain simulator.

### 3.2 Domains and relays

`--domains` emits the simulator as a tree of modules, one per class, nested as
the classes are, with every domain boundary a set of module ports;
`--relay <models>` gives the channels of the named models (or
`<model>:<model>` pairs) a relay stage, so a channel crossing an SLR costs a
host cycle of latency instead of a long wire. Both are off by default.
Measured on s2w4i2 (out of context): with the default options, domains and
relays reach the same 250 MHz at 4.0 ns; at 3.33 ns 273.6 against 262.7 MHz
with the Xilinx cores (equal slack after synthesis), but 225.7 against
238.9 MHz with the soft FPU; without the state-staging options
(regions and folding only) 231.8 against 226.2 MHz at 4.0 ns, for 1.4 % more
LUTs. Both differences are inside the retiming spread. A relay costs host
cycles where a loop runs through it (up to 22 % on img6b), so relays are
chosen per image from a Verilator run.

---

## 4. The kernel

`hw/rtl/afu/foldsim/foldsim_afu.sv` has the native V80 kernel's ports but for
the host-memory master: `s_axi_control` and one device-memory master
`m_axi_mem_<p>` per memory port of the target, each linked to the `MEM` HBM
aperture (the linker gives each its own way in). The target has as many
memory ports as its configuration has platform memory banks
(`VX_CFG_PLATFORM_MEMORY_NUM_BANKS` in `CONFIGS`: 1, 2, 4 or 8; one when
unset); the flow writes their number, the shift to a port's channel and their
AXI id width into `gen/foldsim_kernel.vh`, and the kernel lists its ports with
`FOLDSIM_REPEAT` (`foldsim_ports.vh`). It instantiates `foldsim_sim`, a
memory endpoint per port and the control endpoint, all models of the target's
surroundings that take one token per channel per target cycle
(`foldsim_token.sv`):

- **`foldsim_mem`** is one of the target's AXI4 ports, timed by one of two models
  (§4.1): a latency-bandwidth pipe — cycle for cycle
  `sim/common/axi_latency_pipe.h`: a fixed latency per transaction, bounded
  reads and writes in flight, one beat per cycle, responses in request
  order — or a DRAM channel's banks. A read goes to HBM (`base` + target address)
  the cycle the target makes it, a write the cycle it has both its AW and its
  W, so HBM only ever sees whole single-beat writes; a target cycle that
  should present a response memory has not returned yet does not finish until
  it has, and a write is acknowledged to the target only after HBM
  acknowledged it. A burst sets the memory error and goes nowhere; from the
  next cycle on the endpoint is ready for nothing, and HBM sees none of it.
- **`foldsim_ctrl`** is the AXI4-Lite register block and the endpoint of the
  target's reset, DCR, start and busy. The target runs free: the endpoint
  gives it a token every target cycle. Host commands are queued and each is
  applied for one target cycle (a reset for its count). Kernel reset holds the
  target in reset for 64 cycles.

With several ports, Vortex's AXI adapter puts the index of the GPU memory
port a read came from above its tag buffer's 4 index bits, so the target's
AXI ids are 4 + log2(ports) bits wide; the flow sets
`PLATFORM_MEMORY_ID_WIDTH` and the endpoints' id width to match.

### 4.1 The memory models

The kernel's own reset gives the pipe; a reset command chooses the model the
target has from that reset on (§5). The pipe is the default, and with it the
endpoint makes exactly the transactions and takes exactly the cycles it did
before the bank model existed.

**The bank model** (`foldsim_mem_bank.sv`) is cycle for cycle
`sim/common/dram_bank_model.h` (`DramBankAxi` over `DramBankModel`), the
reservation-based model of dram_sim's DRAM, configured with dram_sim's own
knobs (`VX_CFG_PLATFORM_MEMORY_{TYPE,DATA_RATE,LATENCY,QUEUE_SIZE}`,
`VX_CFG_PLATFORM_CLOCK_RATE`) and its data-sheet timings
(`sw/common/dram_timing.h`, converted to DRAM clocks with Ramulator's
rounding). The same model is a `DramSim` backend in the simulators
(`VORTEX_DRAM_MODEL=bank`) and rtlsim's AXI4 memory (`AXI_BUS=1
AXI_MEM_MODEL=bank`).

![the bank model](../assets/img/foldsim_mem_bank.svg)

- *Port.* A read holds one of 32 slots from acceptance until R takes it; its
  data, from HBM, waits in the slot. An AW and its W form a write. Reads and
  writes reach the channel in the order they formed, from the next target
  cycle; at most `QUEUE_SIZE` requests are in flight (0: the slots only).
- *Channel.* One request a request's bursts' worth of DRAM clocks enters the
  controller's read or write buffer (32 bursts each); a write is acknowledged
  there. The controller serves reads and drains writes in batches between
  dram_sim's watermarks (26 and 7 bursts).
- *Reservation.* The request at the head of the buffer being served is
  scheduled once: its PRE, ACT and column commands take the first DRAM clocks
  every JEDEC constraint allows (tRCD, tRP, tRAS, tRC, tRTP, tWR per bank;
  tCCD, tRRD, the read/write turnarounds per pseudo-channel and bank group;
  one command per clock; HBM's two-clock ACT), before or after commands
  reserved earlier. Rows stay open; a bank serves its requests in order. A
  batch of writes goes after every column reserved before it, the reads after
  it after the batch.
- *Completion.* A read is ready `LATENCY` + 1 target cycles after the target
  cycle whose DRAM clocks its data departs in (tCL + tBL after its last
  column); R and B take the oldest ready response, so reads complete out of
  order — legal, since Vortex never has two reads of one id outstanding.
- *Channels.* Each memory port is a channel of its own, with its own endpoint
  and engine, as dram_sim gives each memory bank its own channel. Ports
  interleave the target's memory by block, so a port's channel sees the
  block index without the port's bits (`PLATFORM_MEMORY_INTERLEAVE`); device
  memory takes the target's address on every port.
- *Refresh* (off unless `VORTEX_DRAM_REFRESH` asks for it; dram_sim models
  none). Every interval a refresh takes the DRAM clock's turn to reserve.
  Per bank (HBM REFsb, GDDR6 REFpb; the banks in turn, an interval of
  tREFI / banks): the bank is precharged if open, once its reserved commands
  allow, then refreshed once the precharge and its last activate allow, and
  may not activate for tRFCsb. All banks (an interval of tREFI): one
  precharge after every open bank's commands, one REF, and no bank activates
  for tRFC. Both commands take the row-command bus as a precharge does; a
  refresh that would reach past the horizon waits, and the requests with it.

In hardware the calendars are bitmaps over a ring of 512 DRAM clocks in
distributed RAM: per clock, whether it holds a command, the row bus, and per
pseudo-channel and bank group the clocks a read, a write or an ACT may not
take. Every command marks, once reserved, the clocks its constraints forbid
to later ones, so placing one is a first-free search over 64 clocks a host
cycle; a word of the ring is cleared, by a valid bit, as the DRAM clock
passes it, and a reservation reaches at most 448 clocks ahead (the model's
horizon). A refresh goes through the same search and commit as a request's
precharge; for all-bank refresh the channel keeps the latest time a
precharge and an activate are allowed over the banks opened since the last
one, so it never reads the whole bank table. A target cycle's DRAM clocks run in host cycles while the target
evaluates it — two host cycles a clock, about fifteen more for a
reservation — and the cycle finishes only once they have; the `memory
waits` counter includes the host cycles a finished cycle waits for them. tFAW is left to tRRD,
which covers it wherever four tRRD_S span a tFAW (HBM2, HBM3); the runtime
refuses a memory where it does not (GDDR6).

Both endpoints' token ends are staged (`foldsim_token`'s `STAGED`): `have`
and `done` are registers, so an endpoint's finish starts at its own state,
not at the simulator's valid and ready. A token counts a host cycle later,
which s2w4i2's eight host cycles a target cycle hide (its counts are
unchanged).

**Reset.** The control endpoint also gives the memory endpoint a token each
target cycle: the target's reset in that cycle. A reset command thereby
resets the target's whole memory port, not only the target. A target cycle in
reset takes no request, and finishes only once HBM has answered every
transaction it has from before the reset — reads returned, writes
acknowledged, nothing left offered to it — so the AXI4 port to HBM stays in
protocol. The cycle then drops the pipe: requests the target made and was not
answered, data not delivered to it, the lone half of a write, and the memory
error. No response from before a reset reaches the target after it; writes
HBM already took stay written, as they would behind a real memory
controller.

---

## 5. The runtime

`sw/runtime/foldsim/vortex.cpp` (`--driver=foldsim`) is a transport under the
runtime's software command-processor model: the
model's DCR writes and reads, start and busy become commands and status reads
of the control block, and its device-memory accesses DMA at the target's
addresses. `TARGET=hw` uses VRT — the kernel's registers through
`vrt::Kernel`, device memory through a QDMA queue — and `TARGET=avedsim`
the same kernel in Verilator (`sim/foldsim`), with a sparse device memory and
each register access costing the kernel cycles a PCIe access would.

The register map is `sw/common/foldsim_regs.h` (`hw/rtl/afu/foldsim/foldsim_regs.vh`
for the RTL):

| offset | register |
|---|---|
| `0x00`, `0x04` | ID `"FOLD"`, version |
| `0x08` | control: bit 0 the target runs |
| `0x0C` | status: busy (anything queued, running, or in flight), queue full, DCR response ready, memory error |
| `0x10`, `0x14`, `0x18` | command address, data, and the command: DCR write, DCR read, start, reset |
| `0x1C` | the last DCR read's response |
| `0x20`–`0x2C` | target cycles, kernel-clock cycles |
| `0x30`, `0x38` | kernel-clock cycles a target cycle waited for HBM; the number of classes |
| `0xC0`–`0xD4` | memory model: device base, the pipe's read and write latency, most reads and writes in flight |
| `0xD8`, `0xDC` | the memory model and refresh in effect (read only); the bank model's capacities: ports, engines and masters, slots, horizon (read only) |
| `0xE0`–`0xF8` | bank model: data rate and core clock, fixed latency, requests in flight and controller queues, geometry, write watermarks, refresh interval and busy time |
| `0x100 + 8c` | per class, threads finished |
| `0x800`–`0x844` | bank model: DRAM timings, a word each (tBL … tWTR_L, read-to-write extra) |
| `0xC00 + 0x40p` | memory port p's bank model: reads, writes, row hits, misses, conflicts, summed read latency, refreshes, target cycles it could take no read (64-bit each) |

`0x40`–`0xBF` is empty: on the V80 the PCIe block keeps its MSI-X table at
that offset of the BAR the first kernel is mapped at, and answers those
accesses itself (§8). The runtime refuses an image whose `VERSION` differs
from its own map.

`busy` covers queued commands, a start the target has not yet answered with
busy, and transactions the memory model has in flight, at the target or in
HBM, so a host that sees the target idle has every effect of what it asked
for.

The runtime resets the target on open, as a card's native driver resets its
device; that also clears a memory error an earlier process left, and the reset
command's address chooses the memory model and, with
`VORTEX_DRAM_REFRESH=bank|all`, the bank model's refresh. With
`FOLDSIM_MEM_MODEL=bank` the runtime first writes the bank model's registers
from the platform memory it was built for (its `CONFIGS`), every channel
alike, refusing what the image cannot hold and warning when a refresh keeps
a bank past the image's horizon; the counters print at close, per channel
and in all; otherwise it makes exactly the
register accesses it always has. The command
processor model's hooks return no status, so the runtime latches a device
error instead — a memory error, a failed register or memory access, or a wait
that timed out — and from then on every register access the runtime core
makes fails, and the launch, DCR access or copy that hit it fails with it. A
launch longer than `VORTEX_FOLDSIM_TIMEOUT_S` seconds is such an error (no
limit by default), as is a command the target does not take, a DCR read it
does not answer, or a reset it does not finish while its target cycles stop
advancing for that time (60 s when it is unset).

---

## 6. Validation

- **Token co-simulation** (`foldsim_sim` against the RTL it was compiled from,
  in Verilator): random target inputs, every channel stalled at random, every
  output token compared. s2w4i2: 3,000 target cycles, 51,000 tokens, 0
  differences.
- **The memory endpoint against `AxiLatencyPipe`**, in Verilator: random
  requests and device-memory latency, every target cycle's outputs and every
  read's data compared; five latency and limit settings, 0 differences.
- **The memory endpoint** (`hw/unittest/foldsim_mem`, in Verilator): an
  AXI4 master as the target, HBM with random ready and latency checking the
  device-side protocol, resets at random points with HBM busy, and each kind
  of burst; every target cycle's outputs compared with the C++ model's
  (`AxiLatencyPipe`, `DramBankAxi`), every response, id and read's data
  checked, none from before a reset delivered after it. Two pipe settings,
  twelve bank-model settings (HBM2 and HBM3 at the default, v100, a100 and
  h100 rates; small queues; row hits, random rows, one bank to the horizon,
  streams, read- and write-heavy) and five with refresh (per bank and all
  banks, at the standard intervals and faster), on the endpoint alone, with
  staged token ends, and as one of two ports (`foldsim_mem_ch2`): 0
  differences.
- **Refresh against its interval** (`tests/unittest/dram_bank_model
  refresh`): one refresh per interval, to the operation, at every memory and
  pattern; all-bank refresh costs a stream at least tRFC / tREFI of its
  bandwidth (9.3–12 %, against 9.0 %).
- **Applications** in the in-process simulator and on the V80, against rtlsim
  behind the AXI4 adapter on `AxiLatencyPipe` (`AXI_BUS=1`) with the same
  memory settings: retired instructions and cycles identical (§7).
- **Foldsim's own tests** (`check-foldsim` in the Foldsim repository): lit
  tests of every pass and of the driver, and small threaded simulators run in
  Verilator against their expected tokens — one and several threads,
  registered memory reads, register initial values, eager outputs, a reset of
  one cycle.

---

## 7. Measured results

Configuration s2w4i2 (2 cores, 4 warps, 4 threads, issue width 2), seven
classes: one rest-of-Vortex model, the socket (2 threads) and six unit
classes (4 threads each).

| | where | demo cycles | sgemm cycles | sgemm n=128 cycles | host / target cycle |
|---|---|---:|---:|---:|---:|
| Foldsim, threads in turn, in-process | Verilator | 8,702 | 415,454 | | 12.06 |
| Foldsim, ready scheduling, in-process | Verilator | 8,702 | 415,454 | | 8.00 |
| Foldsim, threads in turn | V80 | 8,702 | 415,454 | 3,497,994 | 8.00 or 12.00 |
| Foldsim, ready scheduling | V80 | 8,702 | 415,454 | 3,497,994 | 8.00 |
| rtlsim, AXI4 with `AxiLatencyPipe` | Verilator | 8,710 | 415,462 | | |
| rtlsim, same, launched after the caches' reset sweep | Verilator | 8,702 | 415,454 | | |

On the V80 the kernel closes at 149.8 MHz (run at 143 MHz) in 138k routed
LUTs, against 219k for the firesim backend's image of the same configuration.
sgemm n=128 takes 196 ms with ready scheduling: 50× SimX on the same host,
1.7× the firesim backend's image.

Instruction counts equal the firesim backend's and rtlsim's. Cycle counts are
1.5–6 % below the firesim backend's, whose memory model has the same
latency-bandwidth settings but adds channel cycles `AxiLatencyPipe` does not;
Foldsim agrees with `AxiLatencyPipe` (rtlsim) instead: every memory
transaction in the same target cycle. rtlsim's harness starts the kernel 8
cycles after reset, while each L1 bank still invalidates one set per cycle
(64 for the instruction cache), so its first fetch waits 8 cycles the
counters see; a host's launch comes thousands of cycles later. Given that
settle time, rtlsim's cycles are Foldsim's exactly.

### 7.1 The bank model

s2w4i2 in the in-process simulator (Verilator) with `FOLDSIM_MEM_MODEL=bank`,
against SimX on the same configuration (one memory bank, Ramulator behind
dram_sim) and rtlsim behind the AXI4 adapter on the same model
(`AXI_MEM_MODEL=bank`). Memories: the default (HBM2, 800 MT/s, 400 MHz core)
and the a100 and h100 profiles' (`VX_CFG_PLATFORM_MEMORY_*` from
`profiles/*.json`).

| app | memory | Foldsim, bank model | rtlsim, bank model | SimX, Ramulator | SimX, bank model |
|---|---|---:|---:|---:|---:|
| demo | default | 10,740 | 10,748 | 10,190 | 9,148 |
| sgemm | default | 399,337 | 399,345 | 401,434 | 401,421 |
| sgemm | a100 | 691,372 | | 690,935 | 690,780 |
| sgemm | h100 | 422,294 | | 425,547 | 425,541 |
| vecadd `-n16384` | default | 194,242 | | 193,267 | 192,901 |
| vecadd `-n16384` | a100 | 311,917 | | 312,529 | 312,587 |
| vecadd `-n16384` | h100 | 223,875 | | 224,921 | 224,921 |

Instruction counts are equal everywhere. rtlsim's cycles are Foldsim's plus
the 8-cycle launch settle it has on the pipe too (§7). Against SimX on
Ramulator Foldsim is within 0.8 % on sgemm and vecadd at all three memories;
the store-dominated demo (2,048 writes to 268 reads) is 5 % slower, and
SimX's own bank backend 10 % faster: the reservation model drains a write
stream differently from Ramulator's per-clock scheduler (the proposal's
mixed-traffic residual). The pipe, by contrast, has no memory to configure:
sgemm takes 415,454 cycles at every profile. A bank-model run costs host
cycles the pipe does not (memory waits): 7.3–9.5 host cycles per target cycle
against the pipe's 8.0.

### 7.2 tinygpu: Foldsim and the firesim backend

The same Vortex tree, configuration and V80 on both flows: 1 core, 2 warps,
2 threads, no L1 caches or local memory, STD FPU, one memory bank, kernel
clock 143 MHz. Every class has one instance, so Foldsim folds nothing here:
this compares the two simulators' fidelity and overheads, not threading.

| | Foldsim | firesim backend |
|---|---:|---:|
| kernel LUTs (synthesis) | 34,393 | 103,369 |
| of them distributed RAM | 1,822 | 51,943 |
| FFs / BRAM tiles / URAM / DSP | 30,198 / 4 / 0 / 8 | 34,376 / 5 / 8 / 8 |
| WNS at 143 MHz | +0.282 ns | +0.752 ns |
| image build (side by side, same host) | 56 min | 80 min |
| `demo` cycles, V80 (= Verilator) | 24,402 | 25,624 |
| `sgemm -n16` cycles, V80 (= Verilator) | 301,923 | 317,163 |
| `sgemm -n64` cycles, V80 | 13,264,035 | 13,937,835 |
| `sgemm -n64` kernel time, V80 | 278 ms | 178 ms |
| host cycles per target cycle, `sgemm -n64` | 3.00 ¹ | 1.84 (FMR) |

¹ With the plain flow (`FOLDSIM_FLAGS=`). The flow's default options (§3.1)
run tinygpu at 4.00 host cycles per target cycle (Verilator), cycles
unchanged: they buy Fmax for large targets with host cycles a small one pays.

Instruction counts are identical on both flows, in Verilator and on the V80.
Foldsim's cycles equal rtlsim's behind the AXI4 adapter on `AxiLatencyPipe`
with the same memory settings (`AXI_BUS=1`) exactly; the firesim backend's
are 5.0–5.1 % above, from its memory model's channel mechanics. With nothing
to fold, a Foldsim target cycle costs three host cycles (the token exchange
of each one-thread model), so the firesim backend runs tinygpu faster;
Foldsim's advantage is the GPU it fits, its area falling with every class it
folds (s2w4i2 above).

---

## 8. Invariants and limitations

1. **Valid and ready of an endpoint depend on its state alone.** The
   simulator wires target ports to the shell directly; a combinational path
   from a channel's ready to its valid would close a loop.
2. **Every model ticks on its first finish.** A threaded model's target clock
   is enabled from the first finish after reset, so token k of every channel
   is target cycle k.
3. **The memory endpoint takes single-beat transactions**, as Vortex issues
   them; a burst sets the status register's memory error, stops the
   endpoint, and only a reset clears it. Under the bank model, no two reads
   of one id may be outstanding (Vortex's AXI adapter never has them).
4. **Memory ports**: 1, 2, 4 or 8, one per platform memory bank, each a
   channel with its own engine (about 11k LUTs). A refresh that keeps its
   bank longer than the horizon (per-bank refresh at HBM3's clock: tRFCsb is
   524 clocks against 448) holds the requests behind one to that bank at the
   channel's ingress, which overstates refresh's cost; the runtime warns.
   With refresh on, a run's cycles depend on where the refresh schedule
   stands at launch, so on the host's timing before it: such runs equal
   another simulator's only from a common reset (the endpoint
   co-simulation), not across hosts. With refresh off they do not depend on
   host timing.
5. **Classes are named, not found**: `FOLDSIM_CLASSES` lists them.
6. **No kernel register in `0x40`–`0xBF`.** The V80 shell configures the
   PCIe block's MSI-X table internally, at offset `0x40` of the BAR whose
   offset 0 is the first kernel; reads there return the table's entries and
   writes stick in it, without reaching the kernel.
