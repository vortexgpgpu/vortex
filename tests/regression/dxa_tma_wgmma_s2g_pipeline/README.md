# Persistent TMA/WGMMA epilogue with two store queues

This compact GEMM + scale/shift + row-statistics regression uses raw DXA and
barrier APIs. It is not FlashAttention: the tensor-core computation is GEMM,
and the statistics are row sum/max plus optional mean checkpoints.

```
warp 2     loader:  A/B tile t+2 -> input_ready[s]
warps 0–1  compute: input_ready -> WGMMA tile t -> input_free[s]
                   accumulator -> scaled SMEM output + row statistics
                           |                          |
                      output_ready               state_ready
                           v                          v
WG {3,5}   warp 5 formats vector; warp 3 issues 2D tile + vector -> commit -> wait.read<2>
WG {4,6}   warp 6 formats metadata; warp 4 issues 3D sum/max + optional mean/metadata -> commit -> wait.read<2>
                           |                          |
                      output_free                state_free
                           +----> producer reuse <----+
```

The loader, compute, and two store roles each execute their own persistent
loop. There is no full-CTA barrier inside the steady state. Every source stage
has distinct ready/free raw barriers, with the two compute warps, its formatter,
and its issuer participating. A separate two-warp formatted barrier orders each
formatter before its own issuer. `vx_fence()` drains producer LSU writes before ready; WGMMA's
warp drain precedes input-free. S2G does not use a transaction barrier.

With three output stages, the store role issues G0, G1, G2, then executes
`wait_group_read<2>()` before publishing free for stage 0. The producers overwrite
that stage with an iteration-specific value. The state queue independently
contains 1, 2, or 3 operations per group:

- row sum/max: every iteration, 3D descriptor;
- mean checkpoint: even iterations, separate 2D descriptor;
- metadata: every fourth iteration, another 2D descriptor.

Output and state operations differ in source address, destination, shape, and
coordinates. Host checks include every tensor element, absent checkpoint slots,
destination guards, and guard regions after each output/state SMEM stage.
`wait.read<0>` only makes source reuse safe. Runtime kernel/DXA drain, not a
full-wait ISA operation, provides final host observation.

## Four comparable correctness modes

| `S2G_ONLY_MODE` | Input | Output | Source-lifetime wait |
|---|---|---|---|
| 0 | TMA | two S2G issuer queues | stage count − 1 |
| 1 | TMA | ordinary stores from the same staged tensors | LSU fence |
| 2 | LSU | ordinary stores from the same staged tensors | LSU fence |
| 3 | TMA | two S2G issuer queues | 0 after each commit |

The ordinary-store baseline is explicitly SMEM-staged because the fused
row-statistics calculation reads the assembled output. This is not evidence
that S2G always beats an optimal direct-register epilogue. There is no artificial
DXA latency knob in this test, and no strict speedup assertion.

From a configured out-of-tree build, with matching tool/dependency paths:

```sh
make -C tests/regression/dxa_tma_wgmma_s2g_pipeline -j8 run-simx TMA_PIPE_K_TILES=1 TMA_PIPE_ITERS=5
make -C tests/regression/dxa_tma_wgmma_s2g_pipeline -j8 run-simx TMA_PIPE_K_TILES=2 TMA_PIPE_ITERS=8
```

L2 is enabled by the test Makefile. `TMA_PIPE_IN_STAGES`,
`TMA_PIPE_OUT_STAGES`, and `TMA_PIPE_STATE_STAGES` independently control
real buffer allocation; output/state waits follow their stage counts.
Five or more iterations exercise reuse of all three default source stages.
The initial prototype used full-CTA synchronization several times per K tile
and iteration; that serial schedule is not retained here.

## Source inspiration and limits

The implementation is original Vortex code, combining patterns rather than
claiming a line-for-line port of a production kernel.

- CUTLASS SM90 epilogue, locally recorded snapshot
  `7107b05535f8977f5ecb9d01ee203205b1fd9bc4`:
  [`sm90_epilogue_tma_warpspecialized.hpp`](https://github.com/NVIDIA/cutlass/blob/7107b05535f8977f5ecb9d01ee203205b1fd9bc4/include/cutlass/epilogue/collective/sm90_epilogue_tma_warpspecialized.hpp),
  `tma_store_fn`, lines 734–760 of the local copy. Producer SMEM ordering,
  designated issuer, commit, and stage-count-based acquire inspired this handoff.
  The cached worktree's `.git` points to a stale absolute location; the hash is
  recorded provenance from the earlier audit, not newly verified Git metadata.
- FlashAttention checked-out commit `0251105a2fb19d2957484b7f023cd8c115286ced`:
  [`hopper/epilogue_fwd.hpp`](https://github.com/Dao-AILab/flash-attention/blob/0251105a2fb19d2957484b7f023cd8c115286ced/hopper/epilogue_fwd.hpp),
  lines 279–327: separate output and LSE objects, warp-group role selection,
  selected store warp, and explicit named-barrier handoff. That kernel uses
  ordinary stores for LSE and TMA for O; it does **not** establish that our exact
  two-S2G-issuer combination is used there. This test deliberately combines two
  independent output roles to stress Vortex's per-warp group ownership.

Trace frontend issue share is not functional-unit busy occupancy. Accepted
memory request counts times line size are not measured DRAM/L2 bus utilization.
Any performance report must label those distinctions and retain raw evidence.
