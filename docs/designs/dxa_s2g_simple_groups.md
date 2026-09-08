# S2G source groups: per-warp counters

This branch implements source-buffer reuse safety only. `wait_group_read<N>()`
does not establish destination completion or visibility. Cache acknowledgements,
L1 invalidation, and full waits are outside this design.

## State and identity

Each SM core has one controller with directly indexed per-warp state. There is
no active-issuer directory, operation-context allocator, associative lookup,
per-operation generation, or software group stack.

```
warp w
  head, tail       ring pointers, including a wrap bit
  open_valid       tail contains accepted but uncommitted issues
  pending[G]       source-unconsumed operation count in each group slot
  wait_active, N   a parked read wait
  closed           stop new issues while retiring the warp owner

issue token:       {core_id, wid, group_id}
source completion: {core_id, wid, group_id}
```

`G = VX_CFG_DXA_GROUP_DEPTH` defaults to 8; the physical `group_id` is the
low `log2(G)` bits of tail. The high pointer bit distinguishes full from empty.
`VX_CFG_DXA_GROUP_MAX_PENDING` defaults to 8, requiring a four-bit counter to
represent zero through eight. This bounds concurrent unfinished operations in
one group, not the historical number of instructions issued before commit.

For eight warps and eight slots, the pending array is only 8 x 8 x 4 = 256
state bits. Pointers and control flags are additional. This is a state-size
calculation, not an area or timing result; the implementation is not guaranteed
to infer a particular RAM primitive.

## One row from first issue to retirement

The first accepted issue reserves tail's row and sets `open_valid`. Later
issues increment the same row. Every accepted source completion decrements
that row, whether or not commit has occurred. No counter is copied on commit.

```
event                     pending[0]    open    head tail
issue output A                 1          1       0    0
issue statistics B             2          1       0    0
A source consumed              1          1       0    0
issue metadata C               2          1       0    0
commit                         2          0       0    1
C source consumed              1          0       0    1
B source consumed              0          0       0    1
retire head                    0          0       1    1
```

An open group that finishes early remains open until commit. A genuinely
empty commit is always a no-op. A group that had operations but has zero
remaining at commit is still a committed boundary. This distinction prevents
resource-dependent empty-commit behavior.

Completion and issue to the same row use `old + accepted_issue - completion`.
Completion and commit may occur together because sealing does not move the
counter. ISSUE and COMMIT use one ordered frontend and cannot both handshake
on that interface in the same cycle.

## Capacity and waits

Committed depth is the wrap-safe pointer difference `tail - head`. It counts
boundaries that have not retired in order, not just rows with nonzero pending.
The open row is excluded from a wait but included in storage capacity.

```
committed:  G0 pending=1 | G1 pending=0 | G2 pending=0
head ------^                                      ^ newest

wait_read<2> must still wait for G0.
Counting only nonzero rows would incorrectly allow it to pass.
```

One source-complete committed head retires per warp per cycle. A newly sealed
row first becomes eligible on the following cycle, matching the registered
commit boundary in the cycle model. A blocked
`wait_read<N>` is accepted into per-warp wait state; the shared execute path
is released. The scheduler unlocks that warp when depth <= N. Since a parked
warp cannot issue subsequent commits, no independent tail snapshot is needed.

Issue is backpressured when the open counter reaches its limit, or when a new
open group cannot reserve a free row. A completion returns counter capacity
even before commit. An existing open group has already reserved its row, so
commit does not need another allocation. Completion transport must continue
progressing while instruction issue is backpressured.

Capacity is per warp, but a blocked instruction can still cause head-of-line
blocking at a shared frontend. Static resource ownership is not a claim that
all other warps have a bypass path.

## Worker assignment and completion contract

Every core uses a fixed worker within its socket:

```
worker = floor(local_core_id * workers_per_socket / cores_per_socket)

4 cores, 2 workers:       core 0,1 -> worker 0
                         core 2,3 -> worker 1
```

The request queue dispatches its head only to that worker, even when another
worker is idle. A worker has one active architectural transfer and finishes
its local payload/source-event drain before reusing that transfer state.
Per-core request order is preserved. Different workers may progress independently.
More workers than cores is allowed but leaves some workers unused. A busy
head's assigned worker may stall requests for idle workers; eliminating this
head-of-line blocking would require separate ordered worker queues.

The group-counter mechanism itself does not require source completions to
arrive in group order: `{wid, group_id}` locates the correct pending count.
Fixed assignment reduces the first implementation's ordering surface; it is
not a mathematical prerequisite of counter-based groups.

The transport must deliver exactly one source event per accepted architectural
S2G operation. No per-operation ID means that a duplicate event cannot always
be distinguished from another valid operation in the same group. Assertions
check impossible targets and counter underflow, not arbitrary stale replay.
Before reusing a warp's owner state, the scheduler drains its old source work.
Reset must reset/flush the workers and completion transport together with the
tracker. These are explicit protocol requirements replacing generation tags.

## Payload tags are separate

```
SMEM -> LMEM responses {payload_slot, word} -> worker stable payload -> GMEM
                                               |
                             one SOURCE_CONSUMED {core, wid, group_id}
                                               |
                                  pending[wid][group_id]--
```

`payload_slot` and `word` belong to the worker's width-conversion buffers.
They locate data, not groups. They remain necessary for multiple LMEM reads
in flight even though architectural operation IDs have been removed.
SOURCE_CONSUMED means all required source bytes have been captured or already
forwarded, with no path capable of rereading them for this operation. It is
not the final read request, final destination request, or destination ACK.

## Software ownership

```
producer: write stage0 -> release ready0
issuer:   acquire ready0 -> S2G G0 -> commit
producer: write stage1 -> release ready1
issuer:   acquire ready1 -> S2G G1 -> commit
producer: write stage2 -> release ready2
issuer:   acquire ready2 -> S2G G2 -> commit -> wait_read<2>
issuer:   release free0
producer: acquire free0 -> overwrite stage0
```

Only the issuer's own group queue is affected by its wait. Ready/free raw
barriers explicitly propagate ownership to the producer warps. A second
issuer warp has independent pointers, counters, and waits.

The review retains the three ML scaffolds `dxa_tma_wgmma_s2g_pipeline`,
`dxa_tma_attention_s2g`, and `dxa_s2g_wgmma_epilogue`, plus directed tracker and
transport tests. Verification status and exact commands belong in the
validation report, not inferred from this architectural description.
