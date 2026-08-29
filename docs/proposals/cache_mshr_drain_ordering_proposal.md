# Cache same-line ordering across an MSHR fill-chain drain

## Symptom

`vulkan:deriv` failed on the opae leg only (`dFdy=15` at the lower-left
probe; instrs identical to the passing rtlsim run). CI cell:
vulkan/opaesim/32.

## Forensics

Per-uuid trace comparison (DEBUG=3, opae vs rtlsim, uuids correspond)
reduced the failure to one load: the fragment shader spills the
interpolated varying with `FSW sp+0x44` and reloads it three
instructions later with `LW sp+0x44`. On the failing lane the reload
returned the *previous* fragment's value while the other three lanes
saw the fresh store.

At the dcache bank (line `0xfffebf80`):

1. an older load's beat misses; a fill is requested (MSHR id=0);
2. the store's beat misses the same line and chains (id=3, prev=0);
3. the fill returns and the chain starts draining;
4. the younger reload's beat arrives the same cycle the chained store
   is dequeued, tags-hits the freshly filled line, and reads the data
   array one cycle before the store's replay writes it.

Two design decisions combine into the hazard:

- `VX_cache_mshr` excludes the entry being dequeued this cycle from the
  same-line match (so an allocate never links behind a slot invalidated
  the same cycle — that entry would be orphaned). The prober therefore
  "proceeds as a fresh hit/miss", which the comment calls safe because
  the fill has completed — true for the fill data, false for a chained
  WRITE that has not reached the data array.
- `VX_cache_bank` deliberately releases a request that hits while the
  line's chain is still draining (common with sectoring). A hit released
  ahead of an undrained chained write reads pre-store data; a write-hit
  released ahead of a chained read lets the older read observe younger
  data on replay.

The corruption is timing-dependent (which beats collide depends on
memory latencies), which is why the two verilated drivers corrupt
different pixels and only the opae pattern happened to land on a test
probe. It affects every configuration with a writeback dcache.

## Fix

- `VX_cache_mshr`: new probe output `allocate_pending_wr` — the probed
  line's chain contains a write entry.
- `VX_cache_bank`:
  - a tags-hit is demoted to a chained miss when the line's chain holds
    an undrained write (any request) or any undrained entry (write
    request). The request links onto the chain (pending, so no fill is
    issued) and replays in arrival order. Read-after-read keeps the
    fast release path. AMO requesters keep their own ordering machinery
    and are excluded.
  - `replay_link_hold`: when a same-line request is probing in the same
    cycle a replay would dequeue, the replay (and the fill path, so the
    dequeue pointer is not re-armed mid-chain) is held one cycle. The
    entry stays visible, the prober links behind it, and the drain
    resumes. The forward port needs no hold — it answers from the staged
    fill sector, which a younger array access cannot disturb.

Cost: deriv on opae moved 825263 → 826557 cycles (+0.16%, the hold
bubbles); rtlsim 824302 → 824171.

## Validation

- `vulkan:deriv` passes on opae and rtlsim after the fix (previously
  fail on opae, pass-by-luck on rtlsim — see below).
- graphics/compute regression and tight-tolerance parity legs re-run.

## Follow-up: the deriv image is still ~69% wrong on RTL (both drivers)

With the ordering fix in, a full-image readback shows ~2800 of 4096
pixels with a dead derivative channel on BOTH verilated drivers, in a
deterministic per-quad pattern, while SimX renders all 4096 pixels
correctly. The deriv test checks only three probe pixels and misses it
everywhere.

Traced cause: the derivative sign selection reads
`VX_CSR_CTA_THREAD_ID_X` (`&1`, `&2`) to find the lane's quad position.
The fragment shader runs as raster-launched fragment waves, whose
per-lane launch record is the stamp OVERLAY — `CTA_THREAD_ID_X` returns
stamp slice bits ({N, 0|1, 0, 0} patterns), which the RTL contract
declares undefined for fragment warps. SimX returns proper per-lane
thread ids, so the two layers diverge functionally.

Proposed follow-up (separate change): have the CSR return the lane's
quad-local identity for fragment warps (the lane index suffices — quads
are 4 adjacent lanes), which requires carrying the launch-kind bit in
`cta_warp_t`; and strengthen deriv to scan the full image rather than
three probes so this class of failure cannot pass by luck again.
