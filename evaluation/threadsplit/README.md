# ThreadSplit OpenCL evaluation

The measured implementation is the code, not the initial design proposal. The
RTL has a per-warp pending mask and a BRAM FIFO of `NUM_THREADS` `{mask, PC}`
entries, compiler yields, and subgroup-exit scheduling. It has no SCS watchdog
and no memory spill. The existing IPDOM stack still handles ordinary divergence.

## Reproduction

Run `run.py` from a configured build tree, using its absolute path. It creates
isolated `build32-eval-t{4,8,16,32}` trees, one per hardware width. `--prepare`
reconfigures and rebuilds kernel support and the driver before execution. The
script uses the generated OpenCL Makefiles to build applications and obtain
launch environments. It removes inherited `DEBUG`, `PERF`, and `SCOPE` settings;
`DEBUG=release` is not a valid debug-level setting in this tree.

```
python3 /home/nikhilrout97/vortex/evaluation/threadsplit/run.py \
  --threads 4 --backend rtlsim --prepare \
  --tooldir /home/nikhilrout97/vortex/build32-eval-tools \
  --output /home/nikhilrout97/vortex/evaluation/threadsplit/results
```

Repeat for 8, 16, and 32 threads. Different width trees can run concurrently;
do not run two measurements in the same tree at once. Completed rows are
resumed by benchmark/parameters/configuration. Use a new output directory after
changing code, toolchain, or measurement definitions. `--pilot` selects small
32-work-item correctness cases.

## Figures

`plot_microbenchmarks.py` uses Matplotlib only and reads the combined final CSV.
It writes paper-ready PDF and PNG files plus `microbenchmark-aggregates.csv`:

```
python3 evaluation/threadsplit/plot_microbenchmarks.py \
  --input evaluation/threadsplit/final-csvs/threadsplit-microbench.csv \
  --output /home/nikhilrout97/-ASPLOS-27-ThreadSplit-Copy-/figs
```

The synchronization plots aggregate parameter points by geometric mean for
normalized cycles and instructions, and arithmetic mean for SIMD utilization.
The control plot reports normalized cycles. Empty ITS rows render as an explicit
placeholder rather than a numerical bar; a future ITS result can therefore be
plotted without changing figure layout or configuration order.

The isolated evaluation toolchain uses the existing `scs_tools` components and
LLVM-Vortex 20.1.8, with the `llvm-yield-option.patch` applied to its source. The
patch adds an explicit yield-insertion switch and handles both polling-loop
branch polarities. Both IPDOM variants use `-vortex-scs-yield=0`; SCS uses `=1`.
The original `scs_tools` binaries are unchanged. The patch belongs in the LLVM
component before an eventual prebuilt-toolchain refresh; this local evaluation
does not publish toolchain changes.

## Workloads

All new kernels and their shared host are under `tests/opencl/threadsplit`.
`--benchmark` selects the named kernel specialization at OpenCL build time.
The hash insertion and lock-coupling patterns adapt the existing `lockht` and
`lclist` tests, using the same host inputs for natural and software variants.

* TL-CG/FG: protected increment, with one or multiple locks. Neighboring lanes
  have the same target (`(gid/2) % resources`).
* AM-CG/FG: protected maximum over deterministic candidate values, with a
  selected fraction of work-items participating (`gid < n * percent / 100`).
* HT: each work-item prepends its preallocated node to a bucket chain. Checks
  validate every node exactly once, correct bucket membership, no chain cycles,
  and all locks released; ordering within a bucket is deliberately unspecified.
* ATM: fixed-size transfers, holding the lower account lock before requesting
  the higher. Checks compare every balance to a sequential reference, not just
  the sum. Initial balances exceed the maximum possible debit.
* LCList: hand-over-hand traversal of a preallocated sorted key map. A selected
  fraction increments the matched key's payload; others search only. This is
  **not** the original insertion-only list or Synchrobench's insert/delete set.
  It varies traversal length and update fraction without reclamation or
  nondeterministic membership. Checks cover all payloads, search results, links,
  keys, and released locks.
* CP-DS: endpoint-locked distance projection on a perturbed 2D square grid.
  Coordinates use integer units (rest length 1024), with FP32 distance/projection
  arithmetic and rounded-toward-zero corrections. Horizontal and vertical edges
  are assigned cyclically. An atomic ticket records update order for host replay;
  this validation traffic is included equally in both versions. Checks also
  validate edge multiplicities and released locks. This isolates the distance
  solver; it is not a full T-shirt cloth simulation.
* BH-ST: parent-to-child publication of leaf offsets in a prebuilt heap-ordered
  octree, with a readiness flag per node. Child offsets use precomputed subtree
  leaf counts. Checks cover all offsets and publication flags. Tree construction,
  spatial input sorting, and force calculation are outside this kernel.
* Controls: existing OpenCL `atomicreduce`, `histogram`, `psort`, and `pathfinder`.
  Their original host/reference checks are retained.

`IPDOM_SW` serializes complete synchronization operations by local work-item ID
within each work-group. This source transform leaves the natural lock protocol
and all operation inputs intact, but only one lane reaches a blocking acquisition
at a time; it can therefore acquire, update, and release before its enclosing
branch reconverges. This is intentionally a conservative software baseline for
stack-based SIMT, rather than an attempt to emulate SCS with another scheduler.
BH-ST instead computes each prebuilt-tree offset by walking its parent chain,
because its producer/consumer publication dependency extends across work-groups.
Controls use their unmodified sources for both IPDOM variants.

## Measurement definitions

* Hardware: RV32, one core, eight warps, 4/8/16/32 lanes; issue width one,
  full-warp ALU/FPU/LSU width; 16 KiB four-way L1I and L1D; no L2/L3; A extension.
* Cycles and dynamic instructions are the runtime's `PERF: instrs=..., cycles=...`
  totals. A dynamic instruction is a warp instruction, not a lane instruction.
  These are device-run measurements, including dispatch/runtime instructions,
  not host wall time. `wall_seconds` is recorded separately for reproducibility.
* `issued_instructions` and `active_lane_instructions` count scheduler-output
  handshakes, once per warp instruction, including runtime. Utilization is
  `active_lane_instructions / (threads * issued_instructions)`.
* Yield count is the number of warp-level yield events, including no-op yields.
  Subgroup switches count pending installs and FIFO installs, including subgroup
  completion. Peak contexts include the running subgroup, pending subgroup, and
  FIFO entries, maximized over warps and cycles. This is not IPDOM nesting depth
  and not a count of memory spills. RTL watchdog switches are identically zero.
* SCS/IPDOM mode selection and logging are simulation instrumentation guarded by
  `THREADSPLIT_EVAL`; ordinary RTL builds retain the existing behavior. IPDOM
  disables parking, pending restoration, and yield-driven subgroup switching.
* Raw rows retain all four execution configurations, in plan order. ITS status
  and metrics remain empty. There are no invented ITS or deadlock measurements.
* Normalization joins on benchmark, exact workload parameters, warp width, and
  backend. Synchronization uses IPDOM_SW; controls use IPDOM_native. Only PASS
  rows with a PASS baseline receive normalized values.

## Nontermination classification

A wall-clock cutoff alone is not a proof of deadlock. Native synchronization
rows are classified DEADLOCK only after the host's WORKLOAD marker confirms
compilation completed and the following source-level dependency is applicable:

* TL/AM/HT: adjacent participating lanes contend for the same lock. The winner
  exits the acquisition loop, but strict IPDOM reconvergence postpones its
  unlock until the losing lane exits that same loop.
* ATM and LCList: the first account/head acquisition already has this dependency.
* CP-DS: adjacent edges share endpoints. A lane holding the endpoint needed by
  a sibling cannot leave the second-acquisition loop to release it while that
  sibling spins. The tested grids/work-group mapping expose this chain inside
  the first warp for every measured width.
* BH-ST: root and its children share the first warp. The root's readiness
  condition is satisfied; children wait for the publication that follows the
  root's loop exit, which IPDOM postpones until those children leave the loop.

The native cutoff is 20 seconds per case, with the observed duration in CSV.
Unexpected SCS/SW/control nontermination is TIMEOUT (default 180 seconds), and
build/runtime/check failures are ERROR. None of these rows receives numeric
performance values. Full logs are retained for every measured attempt.
