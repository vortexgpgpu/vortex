# Implementing a Custom Accelerator ISA Extension in Vortex — Design Guide

**Scope:** how to add a fixed-function accelerator to Vortex and give it a
*good* SIMT-visible ISA. The hard part is almost never the accelerator's
datapath — it is the **interface**: how a warp passes arguments in and gets
results out, cheaply. This guide captures the patterns that make a custom
`CUSTOM*` opcode efficient and the mistakes that make one expensive. It applies
to any accelerator — a tensor core, a DMA/copy engine, a codec, a crypto unit,
a sort/scan unit, a traversal engine.

Read it *before* designing the kernel-facing intrinsics: the ISA surface is the
hardest part to change later.

This document is self-contained; all mechanisms are described and illustrated
here, with generic examples (`my_accel_*`) you map onto your unit.

---

## 1. The core problem: arguments don't fit in an instruction

A RISC-V instruction reads **at most 3 source registers** (`rs1`/`rs2`/`rs3`)
and writes **one** destination `rd`. A fixed-function accelerator usually needs
more than that per invocation — a DMA descriptor is a dozen fields, a matrix
fragment is 8–24 registers, a codec block is a buffer. The naive fix — a
per-(warp,lane) "special register file" the kernel fills one slot at a time with
`set` ops and reads back with `get` ops — is the **single worst mistake** in
this space (see §7). The whole craft is moving many arguments to the accelerator
*without* that overhead.

The key realization: **arguments live at different SIMT scopes, and each scope
has a different cheapest delivery mechanism.** Classify every argument first;
then pick the mechanism per scope.

### 1.1 Argument-scope taxonomy

| Scope | Meaning | Divergent? | Cheapest home |
|---|---|---|---|
| **per-thread** | one value per lane (the actual work item) | yes | register window, or custom-LD into SRAM |
| **per-warp** | uniform across the warp at the call site | usually no | lane-scatter pack into one register |
| **per-CTA** | shared by a thread block | no | shared memory / barrier-gated |
| **per-dispatch** | constant for the whole kernel launch | no | DCR (host-programmed) |

Putting a per-dispatch constant in a per-thread register, or a per-thread value
in a DCR, is always wrong. Match the mechanism to the scope.

---

## 2. The four argument-passing mechanisms

Vortex gives you four, in increasing per-call cost. Use the cheapest one that
fits each argument's scope. A real accelerator usually combines several: e.g. a
DMA engine uses a DCR-programmed descriptor (§2.1) *and* a lane-packed launch
operand (§2.2); a tensor core uses a register window for its fragments (§2.3)
*and* a custom-LD for a metadata table (§2.4).

### 2.1 DCR — per-dispatch config (host-programmed)

Dispatch-global state — enable bits, a mode/format selector, a callback entry
PC, a descriptor or buffer base the whole launch shares — goes in **Device
Control Registers**, written **host-side** by the runtime before the kernel
launches, not by the kernel.

```c
// Host side, once per launch. Program the accelerator's dispatch-global state.
vx_dcr_write(dev, MY_ACCEL_CONFIG,  pack_config(mode, fmt));
vx_dcr_write(dev, MY_ACCEL_BASE_LO, (uint32_t)(buffer_addr & 0xffffffff));
vx_dcr_write(dev, MY_ACCEL_BASE_HI, (uint32_t)(buffer_addr >> 32));
```

- Cost: **zero per-call** instructions; programmed once per launch.
- Wrap the `vx_dcr_write` calls in a runtime host header so the unit has a clean
  host API (the analog of a driver programming pipeline state).

**Rule:** if a value is identical for every thread in the launch, it is a DCR —
do **not** pass it per-call. But a value that varies *per invocation* (a pointer
the kernel chooses each call) is **not** per-dispatch; keep it in the
instruction.

### 2.2 Lane-scatter packing — the `vx_wgather` trick

This is the most useful and least obvious trick. A SIMT register is physically a
vector of `SIMD_WIDTH` lanes — at the common baseline `SIMD_WIDTH = 4`, one
register holds **4 × 32 bits = 128 bits across the warp**. For a *warp-uniform*
value you would normally replicate the same word into all lanes, wasting that
width. Instead, **pack several distinct warp-level scalars into the lanes of one
register**, and let the accelerator read lane *i* as argument *i*.

`vx_wgather(a, b, c, d)` performs exactly this scatter — it reads the four
scalars (from lane 0) and produces one register whose lanes hold them:

```c
// Pack a 4-word warp-uniform descriptor into the 4 lanes of ONE register.
//   lane 0 = arg0,  lane 1 = arg1,  lane 2 = arg2,  lane 3 = arg3
uint32_t desc = vx_wgather(arg0, arg1, arg2, arg3);

// One instruction hands all four to the accelerator; it reads the 4 lanes.
uint32_t handle = my_accel_launch(desc, /* per-thread operands... */);
```

Inside the accelerator's front-end the warp packet carries all four lanes of
`desc`, so reading "lane 0 = arg0, lane 1 = arg1, …" recovers the four arguments
from a **single register read** — turning four operands into one. This is how a
warp hands a small descriptor (e.g. `{src, dst, length, mode}`) to a launch/copy
engine in one instruction.

- Cost: **one** `vx_wgather` (pure register-domain, no memory). When the scalars
  are loop-invariant it hoists out of a loop → amortized ~0.
- Constraints:
  - Needs `SIMD_WIDTH ≥ number of packed scalars`. At the minimum width 4 you
    get exactly 4 slots; wider warps give more.
  - The values must be **warp-uniform** — the lane dimension is repurposed for
    *argument index*, not *thread*. If a value genuinely diverges per thread, it
    belongs in a per-thread mechanism (§2.3/§2.4), not here.
  - `vx_wgather` reads from lane 0, so the packed scalars must be valid in lane 0
    independent of the active mask — compute them **before** any divergence.

### 2.3 Register window — per-thread multi-register operands

The per-thread work item already lives in the register file as a contiguous
**register group**. The accelerator instruction reads that window directly — no
copy into a special file. Because a window exceeds the 3 read ports, the
instruction is a **macro-op** the sequencer expands (§3).

Lay the work item out as an N-register window and document the slot map. A
generic 8-word work item:

```
  8-register window  (base .. base+7):
   w0  w1  w2  w3  w4  w5  w6  w7
   └──────────── one per-thread work item ────────────┘
```

The instruction names the *base* register; hardware reads the whole window by
convention. The window simply *is* the values the compiler already had in
registers — there is no marshalling.

- Cost: `⌈window_size / read_ports⌉` issue uops; the window is the compiler's
  existing allocation.
- **Type-split (load-bearing).** Vortex has separate integer (`x`) and float
  (`f`) register files. Put float operands in an **FP** window and integer
  operands in **GP** registers, so the accelerator reads each from its natural
  file with **zero `fmv` conversions**. Float data forced into a GP window costs
  one `fmv.x.w` per word — reintroducing exactly the marshalling you were trying
  to avoid.
- **Register-group reservation.** A group of N must usually be N-aligned and
  fully caller-saved so the allocator can satisfy it at every call site. In
  RISC-V the only clean **8-aligned, all-caller-saved** range is **`f0–f7`**
  (the `ft0–ft7` temporaries): every 8-aligned integer range contains
  `zero/ra/sp/gp/tp` or callee-saved `s` registers, and the other FP 8-groups
  straddle callee-saved `fs` registers. Consequence: two accelerators that both
  want `f0–f7` cannot be live in the same kernel region at once — usually fine,
  since they are distinct workloads. Use the compiler backend's register-grouping
  support for the exact binding rather than hand-rolling register pinning.

### 2.4 Custom LD into accelerator SRAM — bypass the register file entirely

When the per-thread/per-warp data block already lives in memory, or is large,
don't route it through the register file at all. Define a **custom load**
instruction that reads from shared/device memory and writes **straight into the
accelerator's local SRAM**, with **no register (`rd`) writeback**.

```c
// Custom load: stream a per-lane block from memory into the accelerator's
// local SRAM. rs1 = base address (warp-broadcast); no register dest.
// Lane T reads base[T] and HW writes it into accel_sram[slot][mapped(T)].
my_accel_ld(slot, base_addr);   // data never enters the GPR file
```

A concrete shipping use of this pattern: a tensor unit loads a **table** (its
sparse-structure metadata) this way — the load reads one word per lane from
shared or device memory and writes each into a per-warp SRAM bank, the
destination slot selected by `rd`'s low bits and the format by `rs2`'s low bits.
The table never occupies a GPR; it goes memory → accelerator SRAM in one
custom-LD.

- Cost: one custom-LD per block (the sequencer may expand it into per-lane
  reads); **zero** register pressure for the payload.
- Use when: the argument is a sizeable per-thread/per-warp array, already in
  memory, that would otherwise burn many registers or a long marshalling
  sequence.

**Choosing 2.3 vs 2.4:** register window for the small, register-resident,
just-computed work item; custom-LD-into-SRAM for a memory-resident block (a
table, a descriptor array). They compose.

### 2.5 Encoding the instruction: R-type (R2) vs R4-type

Once you know the operands, choose the instruction *format* that carries them.
For a register-operand custom op the two candidates are:

- **R-type ("R2")** — `rd, rs1, rs2`: **2 source registers**, and a **7-bit
  `funct7`** field free for sub-op / format / a small immediate.
- **R4-type** — `rd, rs1, rs2, rs3`: **3 source registers**, but `funct7`
  shrinks to a **2-bit `funct2`** (the other 5 bits become the `rs3` index).

So the trade is exactly: **a third source register vs. 5 bits of sub-op/format
encoding space.**

| | R-type (R2) | R4-type |
|---|---|---|
| source registers | 2 (`rs1`, `rs2`) | 3 (`rs1`, `rs2`, `rs3`) |
| sub-op / format field | `funct7` = 7 bits (128 codes) | `funct2` = 2 bits (4 codes) |
| immediate room | `funct7` spare for a format/slot id | none — and `rd`-as-immediate is **not** expressible via the assembler's `.insn r4` |
| best when | config in `rs1` + window base in `rs2` + many sub-ops/formats | three genuinely-distinct register operands, few sub-ops |
| uop read width | unaffected (set by operand-collector ports) | unaffected |

**The decisive insight: the architectural format does *not* limit how wide an
operand a macro-op can read.** A macro-op's uops each read up to the operand
collector's port count (typically 3) *regardless* of whether the instruction is
R2 or R4 — the per-thread register window is addressed by a **base register +
group convention** (§2.3), not by spending an `rs3` field. So picking R2 costs
nothing on wide-operand throughput; it only means you address the window via
`rs2`'s base and keep `funct7` free.

**Recommendation — prefer R2 (R-type) for most accelerators.** With the patterns
in this guide the per-call operands collapse to: warp-uniform args lane-packed
into one register (`rs1`, §2.2) + the per-thread window addressed by one base
(`rs2`, §2.3) + a result/handle in `rd`. That fits R-type, and the 7-bit
`funct7` then encodes many sub-ops plus a format/slot selector — valuable
because `CUSTOM` opcode space is scarce (4 opcodes × 8 `funct3` rows). Reach for
**R4 only when you genuinely need three distinct register source operands** that
are neither lane-packable (not warp-uniform) nor a contiguous group (so a single
base won't address them) — e.g. an FMA-shaped op with three independent inputs.
If you do, budget for only 4 sub-ops in that `funct3` row and avoid relying on
an `rd` immediate.

(This is purely the *architectural encoding* choice. Internally the sequencer
can still expand an R-type macro-op into FMA-style 3-read micro-ops; the uop
datapath is independent of the instruction's encoded operand count.)

---

## 3. How a wide-operand instruction actually executes

A single issue collects ≤3 source registers and reserves one writeback. Any
instruction needing more is decoded as a **macro-op** and serviced by two
existing core mechanisms — you reuse them, you do not build new read ports.

- **Sequencer** — a per-warp micro-op *expander*. It takes the macro-op and
  emits a run of ordinary micro-ops, **one per cycle**, each reading up to 3
  registers. The macro-op stalls fetch until the run drains; the macro itself
  never commits — only the uops do. A *simple* instruction passes through
  unchanged.
- **Operand collector** — the *datapath* that physically reads each uop's
  registers from the narrow-ported, banked register file and assembles them for
  the functional unit. The limited port count is *why* the sequencer must spread
  a group read across several cycles.

So: **sequencer = "how many uops and which registers each"; operand collector =
"fetch those registers."** Decode marks the op a macro-op and the rest is free.

**Reading uop counts honestly.** The uop count is `⌈registers / read_ports⌉`,
**and** a uop reads from only one register file. A mixed GP+FP operand set
therefore splits into separate GP and FP uops. Worked example — a launch reading
1 GP descriptor word + an 8-word FP work item, with 3 read ports:

```
  uop 0  (GP):  descriptor        → allocate slot, return handle, latch pointer
  uop 1  (FP):  w0, w1, w2        → stream into slot
  uop 2  (FP):  w3, w4, w5        → stream into slot
  uop 3  (FP):  w6, w7            → stream into slot; arm the operation
                                    ──────────────────────────────────────
                                    = 4 uops  (1 GP + ⌈8/3⌉ FP), not 9, not 3
```

Each uop writes its data **straight into the accelerator's slot** via the latched
pointer (§6) — there is no staging buffer.

---

## 4. Returning results

- **Register writeback.** Mirror of §2.3: the accelerator writes a result window
  back, each value to its natural file (float → FP, integer → GP), over
  `⌈result_size / write_ports⌉` writeback uops.
- **Memory.** For large results, write a memory struct the kernel reads with
  ordinary loads.

Prefer register writeback for the few hot scalars the kernel needs immediately;
prefer memory for bulk. Split a mixed result by type — float outputs to FP
registers, integer outputs (and a status word in the `wait`'s `rd`) to GP — so
the kernel consumes each without conversion.

---

## 5. Async-by-design: decouple issue from completion

A long-latency accelerator (copy, traversal, multi-cycle compute) should **not**
block the warp for its whole duration. Make the launch return a **handle**, and
a separate `wait` op block on it:

```c
uint32_t h   = my_accel_launch(desc, work_item);  // returns immediately
// ... kernel does independent work here, overlapping the operation ...
uint32_t sts = my_accel_wait(h, &result);         // blocks via scoreboard
```

- The launch snapshots its inputs into the accelerator's own in-flight storage
  (its "slot"/"context"/"bank"), returns a small handle, and frees the warp.
- The accelerator runs asynchronously; the kernel overlaps independent work.
- `wait(handle)` blocks via the scoreboard (no spinning); back-pressure when the
  in-flight pool is full propagates to the issuing warp.

Only the **handle** (one register) persists across the long operation, and
latency hiding is free.

---

## 6. The efficiency principles (what "good" means)

Optimize three axes together — they mostly align:

1. **End-to-end latency.** Async (§5) takes the operation off the kernel's
   critical path; only the issue→dispatch hand-off is serial. Spreading a
   register window across register-file banks lets the operand collection read
   multiple words per cycle, shrinking that hand-off.
2. **Register-file storage.** Hold the work item in registers only during issue;
   after the launch dispatches, **only the handle survives**. Never reserve a
   dedicated per-(warp,lane) special register file — that storage is pure
   overhead (§7).
3. **Temporary SRAM.** The accelerator's in-flight slot is intrinsic — you must
   remember an async work item while it runs. Add **nothing else**.

**Stream-to-destination, never replicate.** The launch's first uop allocates the
slot and returns its index as the handle; the remaining uops write the work item
*directly into that slot*. There is **no intermediate staging buffer** and **no
second copy of the register file** — the source registers free once the slot
owns the data. The only transient micro-state is a write-pointer latch (the slot
index), not a buffer. The launch is, in effect, a *store of the work item into
the accelerator's storage*:

```
  registers / memory ──(uop reads)──► accelerator slot ──► (async run) ──► result
            (freed at dispatch)         (the one copy)
```

---

## 7. Anti-patterns (learned the hard way)

Each of these was shipped or drafted, and each is wrong:

- **Per-slot special-register marshalling.** A per-(warp,lane) "register file"
  the kernel fills one 32-bit slot at a time with `set` ops and reads back with
  `get` ops. A real example cost ~16 instructions to issue one work item (≈10
  sets + launch + wait + ≈4 gets) plus a dedicated ~116 B/lane SRAM. Replace with
  register windows (§2.3) + lane-packing (§2.2) + DCRs (§2.1). **This is the
  mistake this whole guide exists to prevent.**
- **Ignoring argument scope** — routing per-thread, per-warp, and per-dispatch
  values all through the same per-lane port. Partition first (§1.1).
- **Field-by-field operand delivery** when a register window or a custom-LD moves
  the whole block at once.
- **Intermediate staging buffers** to collect a wide operand before handing it
  off. Stream straight into the accelerator's slot/SRAM (§6).
- **Replicating register-file data into a second SRAM** with no lifetime benefit.
  The only legitimate second copy is the async in-flight slot, and the source
  registers free once it owns the data.
- **Float data in a GP register window** (forces an `fmv` per word). Type-split
  (§2.3).
- **Over-serial operand modeling** (1 register/uop when the collector has 3 read
  ports). Count `⌈n / ports⌉`, split by register file.
- **Putting a per-call pointer in a DCR** because "config goes in DCRs." A
  per-invocation binding is not per-dispatch; keep it in the instruction.

---

## 8. Modeling checklist (functional + timing)

- Classify the op in decode: functional-unit type, op type, source/dest register
  types (integer vs float), and mark wide-operand ops as macro-ops that stall
  fetch.
- Model timing honestly: every memory access is a real request; the sequencer's
  per-uop cycle accounting gives the macro-op honest latency. Never read memory
  behind the timing model's back.
- A bounded in-flight pool with real back-pressure to the issuing warp (no
  unbounded queues).
- Deterministic ordering (insertion-order containers, stable arbitration) so runs
  reproduce bit-for-bit.
- Share the operand-collector read-port arbitration; don't add a private port.

---

## 9. Decision checklist

For each argument the accelerator needs, in order:

1. Same for the whole launch? → **DCR** (§2.1), host-programmed.
2. Warp-uniform and few (≤ `SIMD_WIDTH`)? → **lane-scatter pack** with
   `vx_wgather` (§2.2).
3. Per-thread, register-resident, just computed? → **register window** (§2.3) —
   FP for floats, GP for integers.
4. A memory-resident per-thread/per-warp block? → **custom LD into SRAM** (§2.4).

For the launch as a whole: if it is long-latency, make it **async with a handle**
(§5). For results: register writeback for hot scalars, memory for bulk (§4).
Always stream inputs straight into the accelerator's slot — never a staging
buffer, never a replica of the register file (§6).
