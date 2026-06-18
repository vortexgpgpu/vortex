# Elastic Cache-Bank Pipeline (configurable AMAT) Proposal

## Summary

`VX_cache_bank` is hardwired to a 2-stage lookup/commit pipeline. That depth is
correct for a small, latency-critical L1, but it cannot close timing at 300 MHz
on a large last-level cache: the tag-array read, the way-resolving compare, and
the data-array access are forced into a single clock cycle, producing a
BRAM-to-BRAM critical path whose delay is dominated by routing and cannot be
retimed away.

This proposal refactors the bank into an **elastic pipeline** whose depth is a
per-cache parameter (`VX_CFG_<CACHE>_LATENCY`, default 2 = current behavior).
Larger caches raise the knob to insert register stages on the long paths,
trading a few cycles of hit latency — which a non-blocking, MSHR-backed cache
hides — for the Fmax needed to run the whole device at 300 MHz. We propose
`LATENCY = 4` for any L2/L3 larger than 64 KB.

This is the architecture real GPU L2/L3 caches use: deep, fully pipelined,
latency-tolerant behind a large miss-handling pool, rather than a shallow
single-cycle-lookup structure.

## Motivation

On the U55C at the 300 MHz platform clock (period 3.333 ns), a 2-core build with
the 1 MB 8-way L2 fails timing. Measured post-route WNS on the standalone
`Vortex` DUT (`xcu55c`, post-route, after the dirty-mask LUTRAM fix):

| Config | WNS @300 MHz | Implied Fmax | Worst path |
|--------|-------------:|-------------:|------------|
| L2 write-back | **-1.380 ns** | ~212 MHz | `cache_tags/tag_store` → `cache_data/.../data_store` (EN/WE) |
| L2 write-through | **-1.008 ns** | ~230 MHz | same structure |

Because this path sits in the L2, the *entire* device is capped at ~210–230 MHz.
Once integrated into the full XRT platform (HBM + PCIe + SLR crossings) the
slack erodes further. No amount of placement or logic restructuring closes a
single-cycle BRAM→BRAM dependency whose delay is ~78% routing — the cycle
boundary has to move.

## Current timing bottlenecks in `VX_cache_bank`

The bank runs a fixed two-stage pipe (`sel → S0 → S1`, two `VX_pipe_register`s).
Tag and data arrays are read at issue; the hit/way is resolved combinationally
in S0 and immediately drives the data array in the same cycle. The bottlenecks,
in order of severity:

1. **[PRIMARY] Tag-compare → data-array write-enable (S0).**
   `tag_store` (RAMB) clk-to-out → per-way tag compare (XNOR/AND tree) →
   `hit_any = |tag_matches` → `slice_write = fill || (write && hit_any &&
   word_en)` → `data_store` `ENARDEN`/`ENBWREN`/`WEA`. The failing endpoints are
   the data-array **enable/write-enable pins**, not the address pins. BRAM→BRAM,
   ~78% routing. This is the −1.38 ns path.

2. **Tag-compare → data-array address.** The way-folded array is addressed
   `data_addr = {hit_way, line_idx}`, so `hit_way` (from the same S0 compare)
   feeds the data BRAM address pins. Currently meets with thin slack; becomes
   the next wall the instant bottleneck (1) is broken.

3. **[ALREADY RESOLVED — prerequisite] Per-byte dirty mask (`byteen_store`).**
   Was the #1 path at −3.762 ns (xrt, 300 MHz). The mask needs 1-bit write
   granularity (`WRENW = LINE_SIZE`), which block RAM cannot do, so a
   `LUTRAM=1` instance was silently inferred as *shattered* BRAM. Fixed by making
   `VX_sp_ram`/`VX_dp_ram` honor `LUTRAM=1` via the portable `USE_FAST_BRAM`
   (`ram_style="distributed"`) attribute; the mask now maps to distributed RAM
   (LUTRAM 216 → 16,600, RAMB ≈ unchanged) and leaves the critical path. This
   refactor assumes that fix is in place.

4. **Replacement state (`cache_repl`) → data/tag.** FIFO/PLRU victim select and
   state update. At 250 MHz the worst path was `cache_repl` FIFO → `byteen_store`;
   the lookup/update feedback (`lookup_valid`/`repl_valid`) is a second-tier path
   that benefits from extra slack.

5. **MSHR probe/allocate (`cache_mshr`).** `probe_addr` is compared (CAM-style)
   against in-flight entries to produce `probe_pending_*`, which gates admission
   and AMO ordering. The compare fanout over `MSHR_SIZE` entries is a control path
   that tightens as MSHR grows.

6. **AMO read-modify-write (LLC, S1).** `read_word_st1` → AMO ALU
   (add/min/max/swap/compare) → writeback register → re-inject as a synthetic
   write. Only synthesized for the AMO-capable LLC bank, but it is a genuine S1
   compute path.

7. **Read-data → response / writeback formatting.** `read_data_st1` → `crsp`
   word select, and `evict_byteen`/`is_dirty` → `mem_req_queue`. Registered and
   comfortably met today, listed for completeness.

The elastic pipeline targets (1) and (2) directly (deferring the data access to a
later, register-fed stage) and relaxes (4)–(6) by giving each its own stage
budget instead of cramming lookup+commit into two cycles.

## Proposed design: elastic pipeline

### Single knob, distributed internally

Expose one parameter per cache, `LATENCY` (carried from
`VX_CFG_<CACHE>_LATENCY`), with `LATENCY = 2` reproducing today's behavior
bit-for-bit. The bank derives internal stage placement from it:

| Internal budget | Cuts | Implementation |
|-----------------|------|----------------|
| `TAG_RD_LAT` | sel→tag routing + tag BRAM clk-to-out | tag RAM output pipeline registers |
| **hit→data register** | bottleneck (1)/(2): compare → data EN/addr | one pipe stage (the key cut) |
| `DATA_RD_LAT` | data BRAM clk-to-out → way mux | data RAM output pipeline registers |
| response register | read-data → crsp/mreq | one pipe stage |

Extra RAM output registers retime into the BRAM/cascade and cost almost nothing
in fabric while buying most of the Fmax. The single new *logical* stage is the
hit→data register that moves the data access off the same cycle as the compare.

### Spine refactor (readability + elasticity)

Replace the ~40 parallel `_sel`/`_st0`/`_st1` wires and two hand-instantiated
pipe registers with:

1. **A packed payload struct** carrying all per-request control/data:
   ```systemverilog
   typedef struct packed {
       logic                          valid;
       logic [`CS_LINE_ADDR_WIDTH-1:0] addr;
       logic                          rw;
       logic [WORD_SIZE-1:0]          byteen;
       logic [`CS_WORD_WIDTH-1:0]     word;
       logic [`CS_WAY_SEL_WIDTH-1:0]  way;
       logic                          hit;
       // tag, idx, mshr_id, is_fill/flush/replay/dirty, amo, ...
   } pipe_t;
   ```

2. **A generate-loop register chain** of depth `LATENCY`:
   ```systemverilog
   pipe_t stg [0:LATENCY-1];          // stg[0] = arbitrated/selected request
   for (genvar i = 1; i < LATENCY; ++i) begin : g_pipe
       VX_pipe_register #(.DATAW($bits(pipe_t)), .RESETW(1)) reg_i (
           .clk, .reset, .enable(~pipe_stall),
           .data_in(stg[i-1]), .data_out(stg[i]));
   end
   ```
   Adding depth is a wider array — no `if (LATENCY==2) … else if (==3)` ladder.

3. **Control anchored to symbolic stage indices**, not literal `st0/st1`:
   ```systemverilog
   localparam HIT_ST  = TAG_RD_LAT;        // tag compare consumes stg[HIT_ST]
   localparam DATA_ST = HIT_ST + 1;        // data access uses *registered* way
   localparam RESP_ST = LATENCY - 1;       // crsp / mem-req fire here
   ```
   `cache_repl` lookup/update, `cache_mshr` allocate/finalize, tag write, and the
   response all key off these names, so the feedback loops stay one-request-per-
   cycle at any depth.

### Deferred whole-array access — no hazard logic required (implemented)

The implemented design is **simpler than the 1R1W split originally sketched**.
The data array stays a single-port `VX_sp_ram`; the *entire* access (read **and**
write, plus fill/flush) is deferred together by `PIPE_EX = LATENCY-2` register
stages. Two consequences:

- **The tag→data critical path is cut.** The data array is driven by *registered*
  `tag_matches` (and the registered way/line/word/byteen), so neither the write
  enable (bottleneck 1) nor the address (bottleneck 2) carries the combinational
  tag-compare result. Path becomes register→BRAM, intra-stage.
- **No store→load hazard, no forwarding, no stall scoreboard.** Because the
  array's read and write move to the *same* deferred stage, pipeline order is
  preserved: a younger same-line read always reaches the array *after* an older
  write, so store→load forwarding is automatic. (The 1R1W/forwarding scheme is
  unnecessary — keeping read+write co-located is strictly simpler and lower-risk.)

The tag array is left entirely at S0/S1, so its existing read-during-write
bypasses (`rdw_fill`/`rdw_write`) are unchanged.

### Decoupled pipeline — the MSHR must NOT be deferred (critical constraint)

`VX_cache_mshr` is **strongly coupled** to the bank pipeline: its coalescing
chain requires `allocate` (S0) and `finalize` (S1) to remain **exactly one cycle
apart**. The tail-find (`prev_idx`) only sees a predecessor's link once that
predecessor finalizes; deferring finalize makes 3+ coalesced same-line misses
(e.g. sequential icache fetches to one line) all link to the same predecessor,
orphaning intermediate entries → they never replay → **bank deadlock**. This was
confirmed empirically (a first "defer everything" attempt hung at both LATENCY=3
and 4).

So the implemented pipeline is **decoupled**:

- **S0 / S1 (fixed, 1 cycle apart):** tag compare, replacement victim-select,
  MSHR allocate **and finalize**, replacement update. Untouched.
- **stD = S0 + PIPE_EX:** the data-array access (read+write) — a pass-through
  register chain off S0 (`pipe_bubble_data`).
- **stC = S1 + PIPE_EX:** the core response and the memory request — a
  pass-through register chain off S1 (`pipe_bubble_commit`), aligned with the
  deferred data output `read_data_stC`.

`PIPE_EX=0` collapses stD→S0 and stC→S1, reproducing the classic 2-stage bank
bit-for-bit (verified: LATENCY=2 gives identical cycle counts).

### Memory-request queue sizing (constraint)

The mem-request push now fires `LATENCY` stages after admission, so the queue's
almost-full margin must reserve `LATENCY` slots (`PIPELINE_STAGES = LATENCY`).
This requires **`MREQ_SIZE > LATENCY`** (else `ALM_FULL ≤ 0` → permanent
almost-full → admission deadlock). Default small-cache `MREQ_SIZE = 4` is fine
for `LATENCY ≤ 3`; enabling `LATENCY = 4` on L2/L3 requires bumping their
`MREQ_SIZE` (see config section).

## Atomics (`AMO_ENABLE`) under elastic latency

`VX_cache_amo` is the most stage-coupled block in the bank and the part most
affected by changing depth, so it is called out separately. Today it reaches
*directly* into the fixed two-stage structure: it consumes lookup-stage signals
(`valid_st0`, `is_hit_st0`, `is_creq_st0`, `word_idx_st0`, `addr_st0`) and
commit-stage signals (`is_hit_st1`, `read_word_st1`, `do_write_st1`,
`byteen_st1`, `write_word_st1`, `addr_st1`, `mshr_id_st1`), performs the
read-modify-write, and re-injects the result as a synthetic writeback through the
admit path. Three mechanisms encode the assumption that commit is exactly one
cycle behind lookup.

**1. The RMW datapath becomes a stage budget, not a single-cycle path.**
The LLC atomic reads the line word at the data-output stage, runs the ALU
(add/min/max/swap/compare), and writes it back — bottleneck (6). At `LATENCY=2`
this is one S1 cycle. Under the elastic pipe it maps to the same symbolic stage
indices as the data path: read at the data-output stage, ALU in the following
stage, writeback at the commit stage. So deepening *relaxes* the AMO ALU path
(it gets its own stage) rather than complicating it — the engine must be
re-parameterized on `HIT_ST`/`DATA_ST`/`RESP_ST` instead of literal `st0`/`st1`.

**2. Same-line AMO chaining is the tightest interaction.** A chained atomic to a
line with an in-flight commit must observe the *previous* atomic's result. Today
`chain_stall` paces the follower by one cycle so the prior result reaches the
writeback register; `commit_busy` holds new admits while a single LLC commit is
outstanding. With depth `L`, the commit→visible round trip is `L-1` cycles, so
both pacing windows scale with `LATENCY`. The same-line stall scoreboard proposed
for general RAW hazards **covers AMO chains by construction** (a chained atomic
targets a line the scoreboard already marks in-flight); `chain_stall`/
`commit_busy` collapse into that one mechanism, sized to `L`, rather than a
separate hand-tuned 1-cycle pacer.

**3. Non-LLC forward / passthru-replay ordering is latency-agnostic.** A non-LLC
AMO forwards downstream, invalidates its local copy, and returns via a passthru
replay (`is_amo_fwd_*`, `is_amo_replay_st1`, `req_input_defer`). These are
event-ordered, not cycle-counted, so they carry over unchanged once they key off
the stage constants instead of `st0`/`st1`.

**LR/SC reservations** (`VX_CFG_AMO_RS_SIZE`) track line addresses, not pipeline
cycles, and are unaffected by depth beyond keeping the reservation-clear (any
intervening write to the line) anchored to the commit stage.

Net: `AMO_ENABLE` requires the engine's stage anchors to be re-expressed in terms
of the elastic stage constants and its chain pacing to be folded into the
depth-sized same-line scoreboard. At `LATENCY=2` the behavior is identical to
today (chain window = 1). The atomics regression (LR/SC, same-line AMO chains,
mixed AMO/load ordering) is part of the rtlsim sweep across `LATENCY` values.

## Proposed latency configuration

Add a per-cache knob in `VX_config.toml` (default 2):

```
VX_CFG_DCACHE_LATENCY = 2
VX_CFG_L2_LATENCY = "expr: 4 if $VX_CFG_L2_CACHE_SIZE > 65536 else 2"
VX_CFG_L3_LATENCY = "expr: 4 if $VX_CFG_L3_CACHE_SIZE > 65536 else 2"

# MREQ_SIZE must exceed LATENCY (margin); grow it with the deferral depth:
VX_CFG_L2_MREQ_SIZE = "expr: 4 + ($VX_CFG_L2_LATENCY - 2) + $VX_CFG_L2_WRITEBACK * ($VX_CFG_L2_MSHR_SIZE - 4)"
VX_CFG_L3_MREQ_SIZE = "expr: 4 + ($VX_CFG_L3_LATENCY - 2) + $VX_CFG_L3_WRITEBACK * ($VX_CFG_L3_MSHR_SIZE - 4)"
```

Rationale for the 64 KB threshold: below it the tag/data arrays fit in a few
BRAMs placed adjacently and the single-cycle path closes; above it (the 1 MB L2,
the 2 MB L3) the arrays span many BRAM columns and the cross-array route cannot
meet 3.333 ns.

The `MREQ_SIZE` expr adds `(LATENCY-2)` to the base so the almost-full margin
(`MREQ_SIZE - LATENCY`) stays constant as depth grows — `LATENCY=4` ⇒ base 6,
margin 2 (same as today's `LATENCY=2` margin). Without this, `LATENCY=4` with the
default `MREQ_SIZE=4` deadlocks (margin 0).

The bank parameter `LATENCY` is threaded from these macros through
`VX_cache`/`VX_cache_cluster` to each `VX_cache_bank` instance.

## How it resolves the timing violations

| Path | Today (2-stage) | Elastic (`LATENCY=4`) |
|------|-----------------|------------------------|
| (1) tag-compare → data EN/WE | single cycle, −1.38 ns | compare registered at `HIT_ST`; write driven by registers at `DATA_ST` — path is reg→reg, intra-stage |
| (2) hit_way → data address | single cycle, marginal | read addr still speculative but tag-read is itself registered (`TAG_RD_LAT`), so the source is a BRAM output reg, not a cross-array combinational chain |
| (4) repl, (5) mshr, (6) amo | share the 2 cycles | each gets its own stage slack |

The −1.38 ns path is replaced by register-to-register hops within a stage, each
comfortably under 3.333 ns. The tag and data BRAMs no longer have a same-cycle
dependency, so their placement is decoupled and the dominant routing term is
removed. Target: **WNS ≥ 0 at 300 MHz** for the 1 MB L2 in the full build.

## Area cost estimate

Per L2 bank (1 MB, 8-way, data array 16384 × 512 b), going `LATENCY` 2 → 4:

- **Flip-flops:** two extra payload stages. The wide field is the 512 b write
  word; with control (~70 b) the payload is ~590 b → ~1,180 FF/bank for the two
  added stages, plus the BRAM output pipeline regs (absorbed into the BRAM).
  Against the measured 117 k FF for the 2-core build, that is **~+1%**.
- **Block RAM:** unchanged. Data stays in the same BRAMs; the read/write split is
  BRAM-native dual-port.
- **LUTRAM / LUT:** the deferred-write mux + the same-line stall scoreboard add a
  few hundred LUTs per bank. (The 16,600 LUTRAM for the dirty mask is the
  separate, already-landed write-back cost, not attributable to this refactor.)

Net: **~+1% FF, ~0 BRAM, small LUT per large-cache bank** — cheap relative to a
+40% clock.

## AMAT impact

`LATENCY = 4` raises the L2 **hit** latency by 2 cycles. The bank stays fully
pipelined (one request/cycle throughput is unchanged), and the cache is
non-blocking (16-entry MSHR), so the added cycles overlap with in-flight misses.

Average-memory-access-time effect:

```
AMAT_overall ≈ t_L1 + m_L1 · (t_L2 + m_L2 · t_mem)
Δ(t_L2) = +2 cycles ⇒ ΔAMAT_overall = m_L1 · 2 cycles
```

For a typical L1 miss rate `m_L1 ≈ 0.10–0.20`, that is **+0.2–0.4 cycle** of
average access time — against a `t_mem` of hundreds of cycles, it is in the noise.

The decisive comparison is absolute wall-clock, because today the *whole device*
is stuck at the L2's Fmax:

| | 2-stage @ 212 MHz | 4-stage @ 300 MHz |
|---|---|---|
| L2 hit latency | 2 cyc = 9.4 ns | 4 cyc = 13.3 ns |
| Device clock | 212 MHz | **300 MHz (+42%)** |

A single L2 hit is ~3.9 ns slower, but every cycle everywhere else is 42%
faster, and that latency is hidden by the MSHR. Throughput-bound GPU workloads
win decisively.

## SimX model (cycle parity)

The elastic depth must be reflected in SimX or the SimX↔RTL cycle-parity target
drifts. No structural SimX work is needed: the bank model already carries a
configurable depth — `Cache::Config::latency` ("pipeline latency") sizes the
per-bank request pipe (`pipe_req_ = TFifo<bank_req_t>::Create("",
config.latency)` in `sim/simx/mem/cache.cpp`), so SimX already simulates a
`latency`-deep pipelined bank.

The gap is only that the value is **hardcoded** at construction instead of
sourced from config. This proposal targets the large caches, so only those are
rewired; the others keep their current literals and are out of scope:

| Cache | SimX site | Today | This proposal |
|-------|-----------|------:|---------------|
| **L2** | `sim/simx/cluster.cpp:82` | `2` | `VX_CFG_L2_LATENCY` (→ 4 when >64 KB) |
| **L3** | `sim/simx/processor.cpp:77` | `2` | `VX_CFG_L3_LATENCY` (→ 4 when >64 KB) |
| L1 D$/I$ | `sim/simx/socket.cpp:47,67` | `1` | unchanged (separate, pre-calibrated) |
| T$/O$/R$ | `sim/simx/cluster.cpp:196,266,320` | `2` | unchanged |

Because the `VX_CFG_L2_LATENCY`/`VX_CFG_L3_LATENCY` macros are emitted from the
same `VX_config.toml`, replacing those two literals with their macro makes the
**one config value drive both the RTL bank parameter and the SimX pipe depth**,
so they cannot diverge. The RTL bank's existing 2-cycle floor and SimX's L1
`latency=1` modeling are a pre-existing parity calibration this change does not
touch; the knob raises only L2/L3, where both sides read 2 today.

Two parity details to keep honest:
- **Same-line hazard stall.** The RTL adds a same-line in-flight stall at higher
  depth. SimX already accounts bank occupancy/contention (`bank_stalls`); the
  same-line RAW stall must be modeled in the SimX bank as well (a marked-line
  check on the `pipe_req_` occupancy) so the throughput effect matches, not just
  the latency. If same-line conflicts are rare for a workload the residual sits
  inside the <5% parity budget, but the mechanism should be present.
- **AMO chain pacing.** The SimX LLC atomic path must pace same-line chains over
  the same `LATENCY`-sized window (it collapses into the same marked-line check),
  matching the RTL `chain_stall`/`commit_busy` behavior at depth.

Parity is then re-confirmed by the existing SimX↔RTL trace-diff methodology at
each `LATENCY` value (default 2 must be unchanged from today).

## Validation plan / status

1. **[DONE]** `LATENCY = 2` bit-identical — rtlsim 2-core+L2 vecadd: `cycles=2164`,
   identical to the pre-refactor baseline.
2. **[DONE]** `LATENCY = 3` functional — vecadd `cycles=2239` (+3.5%, the one
   deferred stage, mostly MSHR-hidden) and sgemm (RAW-heavy reuse, exercises
   store→load across the deferral) both PASS.
3. **[pending]** `LATENCY = 4` once L2/L3 `MREQ_SIZE` is bumped (margin), plus the
   atomics-enabled sweep for the AMO path (`LATENCY ∈ {2,3,4}`).
4. **[pending]** SimX parity update (L2/L3 latency from `VX_CFG_*_LATENCY`) and
   trace-diff at each depth.
5. **[pending]** DUT synth of the 1 MB L2 bank at `LATENCY = 4`; confirm WNS ≥ 0
   @300 MHz and the new worst path is outside the cache.
6. **[pending]** Full 2-core `xrt` build at 300 MHz; on-card validation (#364).

## Risk / compatibility

- Correctness-sensitive (cache data path); gated on the rtlsim sweep above
  before any synthesis.
- L1 and all small caches default to `LATENCY = 2` and the 1-deep forward, so
  their behavior, latency, and area are unchanged.
- The spine refactor (struct + generate pipe + stage-indexed control) is a
  net readability improvement over the current parallel-wire style.
- Depends on the `VX_sp_ram`/`VX_dp_ram` `LUTRAM`/`USE_FAST_BRAM` fix (dirty-mask
  bottleneck #3) already being present.
