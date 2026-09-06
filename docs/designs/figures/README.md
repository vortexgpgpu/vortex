# Trace-derived steady-kernel figures

These three plots were generated from DEBUG=4 SimX traces produced on
`orcas2.cs.ucla.edu` (`chengxuan`, RTX A4000 host), with L2 enabled and
`WGMMA_NRC=16`, `ATT_K_TILES=4`, `ATT_ITERS=16`, and a real three-slot output
ring. Each trace was filtered only to pipeline-issue, L2 request, and DRAM
request records before plotting with `plot_util_timeline.py`.

The measured cycle counts are in `steady_kernel_metrics.csv`. S2G is 1.46x
faster than the LSU-load/global-store baseline, but remains 1.98% slower than
the current TMA-load/global-store path. The plots therefore document genuine
steady-state activity and the current bottleneck; they are not presented as a
fabricated S2G speedup over the already asynchronous TMA baseline.
