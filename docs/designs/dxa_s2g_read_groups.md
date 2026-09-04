# DXA S2G + READ-group architecture (clean first slice)

## Contract and non-goals

This branch adds one deliberately narrow contract:

```text
SMEM producer
    -> S2G copies source bytes into DXA-owned storage
    -> wait.read N transfers ownership of the selected SMEM stages
    -> ordinary byte-enabled global stores continue independently
```

It does **not** implement destination visibility. There is no `wait.full`,
STRSP, cache flush, cache invalidation, S2G multicast, or S2G non-row-major
layout in this branch. G2S continues to use its existing transactional
mbarrier path and does not allocate a group context.

All three feature switches remain opt-in. `VX_CFG_EXT_DXA_ENABLE` retains its
old G2S behavior; `VX_CFG_EXT_DXA_GROUP_ENABLE` adds commit/wait.read; and
`VX_CFG_EXT_DXA_S2G_ENABLE` adds S2G and requires the first two. When the group
switch is off, the tracker and every request/completion metadata bit disappear.

## Integration and data path

```text
one SM core

  warp instruction stream
       |
       | funct3: G2S / S2G / COMMIT / WAIT.READ N
       v
  +---------------- VX_dxa_unit ----------------+
  | S2G ISSUE: query+allocate exact op token     |
  | COMMIT: seal current open count              |
  | WAIT: snapshot sealed_tail, halt this warp   |
  +------------+--------------------+------------+
               |                    ^
   command     |                    | READ completion
   +{epoch,seq,op_id}                | {wid,epoch,seq,op_id}
               v                    |
       core request FIFO      VX_dxa_group_core_endpoint
               |                    |
               v                    v
         socket DXA request arb   one centralized tracker/core
               |
               v
       +-------------------- shared DXA core --------------------+
       | first-ready/RR worker dispatch                          |
       |                                                         |
       | worker i: one setup + one address generator             |
       |       +----------------+----------------+                |
       |       | G2S data half  | S2G data half |                |
       |       | GMEM read      | LMEM read     |                |
       |       | LMEM write     | payload pack  |                |
       |       | mbarrier done  | GMEM store    |                |
       |       +----------------+----------------+                |
       +----------------|----------------|------------------------+
                        |                |
                  local-memory bus   normal cache/NoC request
                                         {addr,data,byteen}
```

Setup and address generation are not duplicated. `VX_dxa_worker` owns one
`VX_dxa_setup` and one `VX_dxa_addr_gen`; direction selects the G2S or S2G data
half only after address generation. The setup stage may precompute the next
command while the active transfer drains, as before.

The group identity is **command metadata, not per-data-packet metadata**:

- The core-to-DXA command adds `{dir, epoch, group_seq, op_id}`.
- The worker latches those fields once with the high-level transfer.
- LMEM requests carry only the existing route/UUID tag.
- Global stores carry only normal memory fields `{address, data, byteen,
  attributes, memory tag}`; no warp/group/op id travels with every line.
- Exactly one READ completion per high-level S2G operation returns
  `{core route, wid, epoch, group_seq, op_id}`.

The current S2G memory endpoint is an ordinary write-only request channel: the
selected memory fabric must not generate a response for these writes. The RTL
endpoint therefore drives `rsp_ready=0` and asserts that `rsp_valid` is never
raised; SimX defensively drains any accidental write response. This is an
explicit interface invariant, not a destination-completion mechanism. A
future response-producing S2G backend must add a response sink before it is
enabled.

Thus a 4 KiB operation does not replicate group metadata on 64 separate cache
lines. It consumes one P-pool entry and sends one completion.

### Current worker serialization boundary

The clean first slice is intentionally conservative inside one S2G worker:

```text
address token 0
  -> LMEM request word 0 -> response/copy
  -> [LMEM request word 1 -> response/copy, if unaligned]
  -> one packed global-line store accepted
address token 1
  -> ...
```

There is one active global-line gather per worker and one LMEM word outstanding
for that gather. It is not a whole-engine serialization: multiple workers run
independently, G2S retains its existing multi-inflight read slots, setup of the
next transfer overlaps drain, and the core-level P pool admits multiple S2G
operations. READ for the high-level operation fires after its final LMEM
response has been copied, and can precede acceptance of its final global store
because the payload is then DXA-owned.

A later throughput-only change can add several line contexts per S2G worker.
Each line context would need `{valid, local-word progress, source buffer,
global address, byteen, last}`. That is orthogonal to the group tracker: it must
still return the same exact operation token once every line source is captured.

For deterministic SimX stress, the optional C++ build define
`VX_CFG_DXA_S2G_DEBUG_LATENCY=<cycles>` delays the first source request of each
high-level S2G transfer (with a small `{wid, group_seq}` skew).  Its default is
zero, it is not an RTL/ISA feature, and it changes neither the completion
definition nor destination-store ordering.  A large-tile regression can use
it to force two committed boundaries to coexist long enough to observe
`wait.read<1>`; normal builds leave it disabled. The integrated regression
exposes this as:

```text
make -C tests/regression/dxa_s2g_bulk_groups_ws run-simx-stress
```

Override `S2G_STRESS_LATENCY` for a shorter or longer delay.

## Centralized logical per-warp tracker

There is no sparse ActiveIssuerDirectory and no per-warp copy of the physical
pool. The implementation is one module per core with direct `wid` indexing:

```text
                     one core-local tracker

 S2G issue wid ----+----> shared P-entry operation pool
                   |      [index] valid/seen/done/generation/wid/epoch/seq
                   |                |
                   |                | completion uses {generation,index}
                   v                v (direct lookup; no CAM)
 per-warp state[wid] ------------------------------+
   epoch / poisoned                                |
   open_remaining                                  | decrement exact owner
   sealed_tail / read_head                         |
   wait_active / wait_snapshot / wait_N            |
   G-entry ring: [live, remaining] <---------------+
```

`wid` is part of ownership, but the physical implementation is centralized.
This matches the useful hardware granularity: arbitrary warps may issue, while
the costly resource is shared rather than duplicated P times per warp.

### Issue

For S2G only, `issue` finds a free physical context and returns:

```text
epoch     = state[wid].epoch
group_seq = state[wid].sealed_tail       // current open group
op_id     = {new slot generation, slot index}
```

It writes the owner fields in that context and increments
`state[wid].open_remaining`. If no context is free, the DXA instruction's
ready signal stays low. The instruction is not handshaken and nothing is
dropped. P is therefore both the metadata-pool size and the issue-credit
limit. A completion before commit decrements `open_remaining`, so this counter
means *currently live source reads*, not historical issues.

### Commit

Commit writes one ring boundary:

```text
ring[sealed_tail % G] = {live=1, remaining=open_remaining_after_completion}
open_remaining        = 0
sealed_tail            = sealed_tail + 1  (mod 256)
```

The combinational next-state ordering makes a same-cycle completion decrement
the open count before commit snapshots it. Completion and commit may coincide;
the integrated front end deliberately does not present a new ISSUE and COMMIT
handshake for the same warp in one cycle, so commit cannot overtake an older
issue. Empty groups are legal. A completed head boundary retires at a maximum
of one row per warp per cycle. If the G rows are occupied, commit backpressures
rather than overwriting a live boundary.

### Completion and exact-once checking

Completion decodes `op_id` directly; there is no associative search. It checks
generation, wid, epoch, and group sequence, then decrements either the open
counter (`seq == sealed_tail`) or the matching sealed row. `seen` and `done`
distinguish a duplicate from a never-valid token after `valid` is cleared.
Stale completions are dropped and counted (they cannot mutate a newer context).
Duplicate or invalid completions set the functional sticky bit and trip the
debug assertion in the integrated endpoint; optional per-class counters exist
under `PERF_ENABLE`.

Generation wrap is a bounded-transport assumption: a completion must not be
delayed through 16 reallocations of the same slot and simultaneously collide
in owner epoch and sequence. `VX_CFG_DXA_GROUP_CTX_GEN_BITS` is configurable
and should be raised if an implementation cannot prove that latency bound.

The normal RTL endpoint naturally separates a real S2G source completion from
allocation because a completion can only be generated after an LMEM response
returns. The tracker also has an explicit issue/completion bypass for a
zero-latency or cancelled endpoint: if the returned token exactly matches the
token being allocated in the same cycle, it is marked done without an
underflow. Any other same-cycle token remains subject to the ordinary
generation/owner checks.

### Wait N

Wait snapshots `sealed_tail` once. It is satisfied when:

```text
required = snapshot - read_head     // 8-bit modulo subtraction
required <= N
    OR
required > G                        // head already passed the snapshot
```

The second condition is essential after modulo wrap: an already-passed
snapshot looks like a large unsigned distance, while a live window can never
exceed G. Subsequent commits do not extend a parked wait. Only its issuing warp
is halted; CTA-wide ownership transfer remains the software pattern
`issuing-warp wait.read` followed by a CTA barrier.

### Warp reuse

At zero-tmask exit, the scheduler poisons the wid and holds that physical warp
slot out of CTA allocation. After all accepted contexts and boundaries drain,
the next CTA allocation advances the two-bit epoch and clears poison. A late
old-epoch completion cannot mutate the new CTA's counts. RTL and SimX both
model this lifetime rule. WSPAWN is not an owner-lifetime transition and does
not advance the epoch; once a wid has executed zero-tmask, only a subsequent CTA
dispatch reopens its S2G stream.

Reset is also a lifetime boundary. SimX resets the helper-owned tracker and
parked wait pointers through `SfuUnit::on_reset()`, advances the model epoch,
and drops queued completion beats. Thus a trace pointer or completion token
from a previous image cannot survive into the next dispatch. RTL reset is
specified as a coordinated transport flush: no pre-reset completion may remain
in the worker, endpoint, or tracker channel. The RTL epoch registers are reset
to zero under that assumption; if a platform cannot guarantee the flush, it
must retain a reset epoch/tombstone instead of reusing zero immediately.

## Parameter and state cost

Defaults are `W=4` warps/core, `P=16` operation contexts, `G=8` boundary rows
per warp, four generation bits, two epoch bits, and eight sequence bits.
Remaining counters are `ceil(log2(P+1)) = 5` bits.

Tracker functional storage at those defaults is:

| State | Formula | Default bits |
|---|---:|---:|
| operation pool | `P * (valid+seen+done + gen4 + wid2 + epoch2 + seq8)` | 304 |
| boundary rings | `W * G * (live + remaining5)` | 192 |
| per-warp owner/wait | `W * (tail8+head8+open5+epoch2+poison1+sticky2+wait_active1+wait_N5+snapshot8)` | 160 |
| **tracker total** |  | **656** |
| optional PERF counters | `W * 3 * 32 + 32` | 416 |
| scheduler exit hold | `W` | 4 |

The functional ring cost scales as `W*G`; the physical pool scales as P, not
`W*P`. The P-pool is compiled away when group tracking is disabled.

The S2G data half adds one source payload buffer per configured DXA worker. For
the default 64-byte global line and 16-byte bank-wide LMEM word, the unaligned
worst case is five words, or 640 payload bits. `VX_dxa_smem2cl` adds about 99
bits of address/offset/progress/control plus two always-present 32-bit debug
progress counters, for 739 bits; `VX_dxa_s2g_data` adds 35 completion/sent
state bits. Direction and `{epoch,seq,op_id}` add 19 bits to each live/staged
setup command (38 bits per worker) and 19 bits per queued high-level command.
These are flop-equivalent RTL fields; synthesis may implement or optimize the
payload buffer differently, and `NDEBUG` changes UUID width.

Default packet-width deltas are:

- high-level command: 19 bits = direction 1 + epoch 2 + sequence 8 + op id 8;
- READ completion payload: 20 bits = wid 2 + epoch 2 + sequence 8 + op id 8,
  plus the existing core-route header;
- LMEM and global data packets: zero group-metadata bits.

## Worker mapping and out-of-order completion

This branch does not implement fixed-home mapping. `VX_dxa_dispatch` retains
first-ready round-robin selection, so two operations from one warp may run on
different workers and READ completions may return out of order. Correctness
does not depend on completion order: `op_id` selects the exact context, its
owner sequence selects the exact counter, and `read_head` advances only across
a completed prefix of boundaries.

A compile-time `home = core_id % workers` policy is a possible determinism
option, not a correctness mechanism. It can reduce cross-worker reorder and
verification variability, but can also strand idle workers and create
head-of-line blocking when one core is hot. Even fixed-home still needs exact
tokens for duplicate/stale detection and multiple operations within one group.

## Tests

The directed RTL tracker test covers completion+commit in one cycle, empty
group retirement, one-row-per-cycle retirement, P-pool backpressure/no-drop,
generation reuse, deterministic two-issuer out-of-order completion, modulo
passed snapshots, poison/drain/epoch advance, and rejected late completion.

The primary integrated regression is `dxa_s2g_bulk_groups_ws` (SimX and
RTLSim). It uses eight logical warps, independent output/state issuer warps,
heterogeneous two-operation output groups, variable one/two/three-operation
state groups, two/three SMEM stage rings, and generation-coded host checks.
The older `dxa_store_read` test remains a small directed ownership smoke test
in the shared worktree; it is useful for quick diagnosis but is not the
primary evidence for this branch. Neither test claims destination-cache
visibility.

Focused commands used for this branch are recorded in the final commit report;
the regression must include both a feature-enabled build/run and a default
feature-off SimX build.

## Exact source-consumed milestone

The tracker counts one event per *architectural S2G operation*.  An operation
may be represented internally by several LMEM requests and several byte-masked
global stores, but it contributes exactly one `SOURCE_CONSUMED` event.  The
event is generated only after the final required LMEM response has been copied
into DXA-owned storage and no retry, replay, packetization, or downstream
backpressure can make the worker read the issuer's SMEM region again.  In
particular:

```text
LMEM read request(s)       -- not yet complete
LMEM response copied       -- source bytes are now stable in DXA
SOURCE_CONSUMED            -- tracker decrement (exactly once)
global store request(s)    -- may still be queued or in flight
destination/cache visible  -- deliberately outside this branch
```

This is why `wait_group_read<N>` is safe for reusing a producer stage while a
destination write is still being accepted.  It is not a completion or
visibility fence for the destination.

## State machine and ordering invariant

For each warp `w`, the direct-indexed context pool and boundary ring implement
the following state machine:

```text
                 first accepted issue
             +--------------------------+
             |                          v
       IDLE/open_valid=0          OPEN(seq=T, pending=k)
             ^       |                 |
             |       | commit          | each source completion
             |       v                 v
             +-- RETIRED <--- SEALED(seq=T, remaining=0)
                    ^              |
                    +-- FIFO head-+
```

An issue allocates a slot before it is sent to the DXA request queue.  Thus a
later `commit_group` cannot observe an issue that the tracker has not accepted.
Issue, commit, and wait use the same per-warp execute/SFU stream; the request
buffer may delay the data command, but it cannot reorder the tracker handshakes.
The commit boundary snapshots the post-completion count in the same cycle, so a
completion+commit race cannot leave an extra source event behind. The
integrated front end presents ISSUE and COMMIT on one ordered SFU stream and
deliberately does not accept both handshakes for one warp in one cycle.

The ring rules are intentionally asymmetric:

* An open group can accept more operations while older sealed groups drain.
* A new open group is backpressured when that warp's ring has no boundary row
  or its statically owned operation contexts are exhausted.
* A nonempty commit needs a boundary row.  An empty commit creates a zero-count
  sealed row and retires immediately when space exists; if every row is live,
  it is accepted as a trivially complete no-op without advancing the sequence.
* Completion is accepted even when it arrives out of order, but retirement
  advances only over a complete prefix at the FIFO head.

The completion token is `{wid, epoch, group_seq, op_id}`.  `op_id` contains a
physical context index and its generation.  The endpoint checks owner, epoch,
sequence, generation, live, and done bits before decrementing a count.  A
duplicate or stale token is consumed without mutating a newer context. The
RTL exposes duplicate/invalid status through the sticky bit and debug
assertions, while stale status is exposed through its stale-drop observer and
the SimX trace.

## Wait-read behavior and scheduler path

At decode/accept, a wait records `(snapshot=sealed_tail, N)` for that warp.
The snapshot excludes the currently open group.  It passes immediately when
the number of sealed groups at or before the snapshot is at most `N`; otherwise
the instruction is parked outside the SFU input queue:

```text
execute WAIT.READ N
       |
       +-- satisfied --> normal result/writeback; warp remains runnable
       |
       +-- unsatisfied -> parked_wait[w]; release SFU input; stall only w
                              |
                 head retires +--+-- condition true
                              v
                        unlock_mask[w]
                              |
                        scheduler resumes w
```

No software-visible barrier slot is consumed by this native path.  A CTA that
needs to hand a stage from producer warps to an issuer still uses a raw role
barrier (or an explicitly documented LSU/LMEM ordering fence) around the
producer writes.  `__syncthreads()` is not silently assumed to order an
independent asynchronous proxy.

## Reference transaction-barrier model

The native tracker is the performance implementation.  A debug/reference
model can map one logical group to one static transaction barrier slot by
holding a commit sentinel:

```text
open group:       expect_tx(1 sentinel); arrive()
each S2G issue:   expect_tx(1)
SOURCE_CONSUMED:  release one operation event
commit_group:     release sentinel
wait_read<N>:     wait/pop oldest committed barrier slots until depth <= N
```

Without the sentinel, an operation that finishes before `commit_group` would
make a barrier appear complete too early.  The sentinel preserves the crucial
“uncommitted groups are not visible to wait” rule.  The production path does
not allocate a software barrier and does not depend on this model.

## Producer/issuer regression topology

The integrated regression uses eight logical warps and two independent issuer
streams.  The role barriers are deliberately narrower than a CTA-wide
pipeline wrapper:

```text
warp 0,1  format/compute producers
    |             |
    +-- out_ready/free (raw barrier) --> warp 2: OUTPUT issuer
    |                                      Gout: [tile, scale]
    +-- state_ready/free -------------> warp 4: STATE issuer
                                           Gstate: [state0, state1?, meta?]
warp 3,5,6,7  auxiliary/control participants
```

Output uses two stages and `wait_group_read<1>`.  State uses three stages and
`wait_group_read<2>`, with one, two, or three operations per committed group.
The producer overwrites a stage immediately after the issuer's wait and free
handoff.  Generation-coded values and canaries make an early source event
observable rather than merely relying on timing.

The output stream's boundary timeline is:

```text
cycle/order:  fill S0 | issue G0(tile,scale) | commit G0
              fill S1 | issue G1(tile,scale) | commit G1
              wait.read<1> ------------------------------+
                                                         |
              reuse/fill S0 | issue G2 | commit G2        +-- G0 retired
              wait.read<1> ------------------------------+
              reuse/fill S1 | ...
              tail: wait.read<0> (source lifetime only)
```

The independent streams do not share a depth counter:

```text
OUTPUT warp 2:  O0 -> commit -> O1 -> commit -> wait<1> -> O2 ...
STATE  warp 4:  S0(a,b,c) -> commit -> S1(a) -> commit -> wait<2> -> ...
                   ^ completion of S0 cannot release O0 or block O1's tracker
```

## Trace and measurement points

With SimX debug tracing enabled, the DXA helper emits `GROUP_OPEN`,
`GROUP_ISSUE`, `SOURCE_CONSUMED`, `GROUP_COMMIT`, `GROUP_SOURCE_RETIRE`,
`WAIT_READ_PASS`, `WAIT_READ_STALL`, `WAIT_READ_WAKE`, and `GROUP_RING_FULL`.
Each record includes `core`, `wid`, `group_seq`, `slot/op_id`, generation,
pending count, committed depth, and `N` where applicable. RTL does not yet
serialize those strings onto a trace stream; its tracker observer ports expose
heads, tails, ring live bits, remaining counts, live contexts, and context
stalls for waveform/assertion inspection. The small optional counters count
committed groups, source events, maximum depth, and ring/context backpressure;
they are not a new global CSR interface.

## Future completion domains (reserved only)

The operation token is intentionally retained until the source context is
retired so a future implementation can deliver independent milestones:

```text
same operation token
       +--> SOURCE_CONSUMED tracker  --> wait_group_read<N>
       +--> DESTINATION_COMPLETE    --> future wait_group<N>
       +--> DESTINATION_VISIBLE     --> future cache/visibility primitive
```

This branch implements only the first arrow.  It does not add cache
invalidation, L1/L2 coherence rules, STRSP acknowledgement tracking, full
waits, multicast, or a claim that a destination is visible when a read wait
passes.

## Known first-slice throughput boundary

SimX and RTL share the tracker semantics, but the current S2G worker is a
correctness-first V1: one worker gathers one global line at a time and has one
LMEM word request outstanding for that line.  A socket can still run multiple
workers and multiple issuer groups concurrently.  Raising line-level
parallelism is a separate worker/data-buffer change; it must preserve the same
one-event-per-operation milestone and token checks.  Therefore this branch's
regression is a correctness/ownership proof, not a speedup claim.

The V1 watchdog coverage also remains the legacy G2S watchdog. S2G progress is
bounded by the LMEM/global request handshakes and covered by the directed
unit-test timeout; adding a dedicated S2G watchdog is a follow-up reliability
change, not part of this source-consumed contract.
