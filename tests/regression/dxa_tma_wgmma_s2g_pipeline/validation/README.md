# Remote validation evidence

Runs use `chengxuan@orcas2.cs.ucla.edu`, never `orcase2`, under
`/home/chengxuan/codex-runs/simple-groups-ws.P8caul`. They are SimX results,
not RTL timing measurements. L2 is enabled in both application and runtime.
No artificial source/DRAM latency was added.

## Reproduce

The source snapshots are `src` (legacy worker) and `src-pipe` (optional pipelined
worker). Builds are out-of-tree. `deps/softfloat` is an isolated TLS-enabled
SoftFloat build; the other third-party libraries are read-only reused dependencies.

```sh
mkdir build-pipe
cd build-pipe
../src-pipe/configure --xlen=32 --tooldir=/home/chengxuan/codex-toolchains/s2g-20260904
CONFIGS="-DVX_CFG_DXA_S2G_PIPELINED -DVX_CFG_DXA_S2G_PIPE_MULTI_READ" \
THIRD_PARTY_DIR=/home/chengxuan/codex-runs/simple-groups-ws.P8caul/deps \
make -C tests/regression/dxa_tma_wgmma_s2g_pipeline -j8 run-simx TMA_PIPE_K_TILES=4 TMA_PIPE_ITERS=16
```

Pass optional `CONFIGS` through the environment, not as a make command-line
variable: the latter overrides the test's appended DXA/L2/warp flags. The actual
`sw/runtime/simx_config.stamp` and application `config.stamp` were checked for
PIPELINED, PIPE_MULTI_READ, DXA, GROUP, S2G, L2, eight warps, and eight threads.

Run each mode in a separate process to avoid cumulative performance counters:

```sh
cd tests/regression/dxa_tma_wgmma_s2g_pipeline
LD_LIBRARY_PATH=../../../sw/runtime VORTEX_DRIVER=simx S2G_ONLY_MODE=0 S2G_DUMP_PERF=1 ./dxa_tma_wgmma_s2g_pipeline
```

Use the absolute `<build>/sw/runtime` path if invoking from another directory.
Repeat with modes 1, 2, and 3. The recorded runs used that absolute path.
For the four-stage case add `TMA_PIPE_ITERS=32 TMA_PIPE_OUT_STAGES=4
TMA_PIPE_STATE_STAGES=4`, retaining three input stages to fit 32 role-barrier IDs.

## Results and limitations

`remote_cycles.csv` contains exact device-cycle and retired-instruction counters
and remote log paths. All twelve rows passed complete host validation. At
K=32, 16 iterations, 3/3/3 stages, legacy-worker S2G took 70,293 cycles versus
76,381 for staged TMA/global-store and 314,227 for LSU/global-store. The optional
worker gave 70,309 cycles: this kernel does not demonstrate a useful additional
gain from enabling that worker. Serial S2G took 70,209 cycles; these data also do
not demonstrate a wait-N performance advantage. The 32-iteration, 3/4/4-stage
case is a separate workload, not an isolated stage-count comparison.

`pipe_k4_i16_sha256.txt` identifies the measured source and binary snapshot.
Later changes must not be assigned these historical measurements automatically.
The recovered attention and standalone epilogue directories are not covered by
these results. True functional-unit busy occupancy and data-bus utilization are
not available from the legacy frontend-issue/request-count timeline.

## Group trace

`groups_pipe_k1_i16.txt` was captured from a separate DEBUG=4 build with optional
worker enabled, K tile count 1, 16 iterations, and 3/3/3 stages. Only group,
source-consumed, wait, validation, and performance lines were retained.

- Issuer warp 3: 16 committed groups, each containing two operations.
- Issuer warp 4: 16 committed groups: eight with one operation, four with two,
  and four with three.
- 60 architectural SOURCE_CONSUMED events, including completion before commit.
- Each issuer executed thirteen `wait.read<2>` operations. The maximum committed
  source depth observed was only one, and none of these waits stalled.

Count GROUP_ISSUE events between commits to reconstruct historical membership.
The `ops=` field of GROUP_COMMIT is the *remaining pending* count, not the
historical group size. For example, warp 4's first group issues at cycles 7827,
7965, and 8031; two source completions precede the commit at cycle 8037, so its
commit prints `ops=1` despite having contained three operations.

These traces establish independent ownership and completion-before-commit, not
multiple committed groups concurrently outstanding. Tracker-directed tests, not
this GEMM workload, supply that latter stress coverage.

## Rejected experiment

`rejected_helper6_reduction.patch` moves all row reductions from compute warps
0/1 into state helper warp 6, allowing compute to proceed earlier. The fourth
output-free participant becomes helper 6 instead of helper 5 to protect its
output reads. This passed all four modes but regressed K=32, 16 iterations:

| Mode | Before | Helper-6 reduction |
|---|---:|---:|
| Pipelined S2G | 70,293 | 104,600 |
| Staged TMA/global | 76,381 | 105,628 |
| LSU/global | 314,227 | 326,394 |
| Serial S2G | 70,209 | 104,266 |

Remote logs are `build/offload-k4-i16-s3-mode{0,1,2,3}.log`. Reducing the number
of row-reduction lanes and adding a longer handoff likely outweighs the overlap
opportunity; this is an interpretation, not a measured critical-path breakdown.
The patch is retained as a rejected experiment, not applied to the kernel.
