# WGMMA epilogue with grouped S2G

Recovered reference source from `cfx-s2g-read-clean`; historical results do not
validate this branch. Use `dxa_tma_wgmma_s2g_pipeline` for TMA input loading plus
two persistent store-issuer roles and actual three-stage buffer reuse.

This regression is a small, low-level model of a common tensor-kernel tail:

```text
WGMMA accumulators in registers
        -> ReLU/scale epilogue
        -> SMEM output stages
        -> grouped S2G stores to 2-D/3-D global tensors
```

Warps 0 and 1 produce the WGMMA tile. Warp 2 owns the output stream and puts a
tile plus a bias vector in each group. Warp 4 owns an independent statistics
stream and puts one 3-D tensor in each group. Warps 3 and 5--7 are resident
auxiliary/control warps, so the test has an eight-warp CTA while keeping the
two issuer streams explicit. Raw role barriers order the
producer writes and stage reuse; `wait_group_read<1>` and
`wait_group_read<2>` only release the corresponding issuer's SMEM stages.

The default geometry is a 16x16 output tile, `K=64`, and a 32x16x2 statistics
tensor. `run-simx-large` uses a deeper 24x15x3 tensor while staying within the
default 16-KiB local-memory limit. The output and statistics descriptors are
programmed independently, so the test exercises heterogeneous 2-D and 3-D
coordinates rather than a flat byte copy.

Run the focused checks from this directory:

```sh
make -B -j1 run-simx
make -B -j1 run-simx-large
make -j1 run-simx-stress S2G_STRESS_LATENCY=20000
make -j1 run-rtlsim
```

Both `SERIAL` and `PIPELINED` modes validate every output, bias, and statistics
element against the WGMMA reference and start each run with poisoned destination
buffers. This is a source-lifetime/correctness regression. L2 is explicitly
enabled (with the normal write-through L1 policy); the test does not claim
destination visibility or a throughput speedup.
