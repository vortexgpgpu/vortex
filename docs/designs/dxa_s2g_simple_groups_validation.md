# S2G simple-groups validation

This worktree implements source-consumed S2G groups only. It does not implement destination completion, cache visibility, invalidation, or full wait.

## Evidence

- SimX tracker: directed and randomized tests pass (411,895 checks).
- SimX S2G pipe: credits 1/4/8, slots 1/3/4/8, and word widths 4/16/256 pass.
- SimX worker: legacy 3272 cycles; pipelined four-credit model 1344 cycles with 132 out-of-order source responses and one source event per operation.
- RTL unit tests: tracker matrices, worker/pipe, scheduler, and fixed dispatch mapping pass.
- Full RTL `dxa_s2g_bulk_groups_ws` on orcas2, with L2 and S2G pipe/multi-read enabled, passes both SERIAL and PIPELINED modes with independent output/state issuer warps.

The representative kernel contains loader, WGMMA/epilogue, output, and state roles. Output groups contain two operations; state groups contain one, two, or three operations. The trace confirms independent issuer queues and heterogeneous group membership. It does not claim a universal speedup or calibrated full-GPU utilization: the current model's local-memory endpoint is not a cycle-accurate replacement for every RTL bank/arbitration path, and the measured wait depth is usually one.

## Reproduction

Use a fresh configured build (`configure --xlen=32 --tooldir=/home/chengxuan/tools`) and enable `VX_CFG_EXT_DXA_ENABLE`, `VX_CFG_EXT_DXA_GROUP_ENABLE`, `VX_CFG_EXT_DXA_S2G_ENABLE`, and `VX_CFG_L2_ENABLE`; add `VX_CFG_DXA_S2G_PIPELINED` and `VX_CFG_DXA_S2G_PIPE_MULTI_READ` for the pipe model. The exact remote logs are retained under `/home/chengxuan/codex-runs/` and copied to `/tmp/vortex-simple-rtl-20260908.AgAhwD/`.
