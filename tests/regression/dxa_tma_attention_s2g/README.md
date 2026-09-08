# TMA/WGMMA/S2G attention epilogue

Recovered reference source from `cfx-s2g-read-clean`; historical results do not
validate this branch. The kernel uses a positive linear weight proxy, not true
softmax/FlashAttention, and retains full-CTA barriers in its main loop. Use
`dxa_tma_wgmma_s2g_pipeline` for the persistent two-issuer/stage-2-wait example.

This regression is a compact FA3/FA4-like data path.  One CTA uses eight
warps: warp 2 loads Q, K, and V with TMA; warps 0 and 1 run WGMMA for QK; and
warp 3 owns the grouped S2G queue.  The epilogue normalizes each QK row,
multiplies it by a separately loaded V tile, and stages a 2-D output tile plus
one row-statistics vector in rotating shared-memory buffers.

The S2G path commits two heterogeneous operations per iteration (output and
statistics).  `wait_group_read<1>()` frees the older output stage while the
newer stage remains in flight.  Mode 1 writes the same result with ordinary
global stores after TMA loads; mode 2 replaces the Q/K/V TMA loads with
cooperative LSU loads and also uses ordinary global stores.  All three modes
run with L2 enabled and share the same host checks.

```sh
make -C tests/regression/dxa_tma_attention_s2g run-simx
make -C tests/regression/dxa_tma_attention_s2g run-simx-large
```

`ATT_K_TILES` and `ATT_ITERS` tune the K depth and number of output tiles.
The default is a 16x16 output with K=32; `run-simx-large` uses K=64 and eight
iterations.  This is a correctness and source-lifetime regression.  It does
not claim destination visibility or implement `wait_group` full completion.
