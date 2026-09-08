# Trace-derived steady-kernel figures

These are historical measurements from the predecessor implementation, not
validation of the simplified group tracker or the multi-read LMEM pipe.
The upper panels count frontend micro-op issues divided by total frontend
issue capacity; they do **not** measure execution-unit busy cycles or tensor
arithmetic utilization. Memory curves count accepted requests multiplied by
an assumed 64-byte line size: they are line-equivalent request rates, not
measured useful-byte or physical data-bus bandwidth. Partial stores and cache
hits prevent interpreting these curves as DRAM payload utilization.

The periodic phase separation is visible even after smoothing. These figures
do not establish an ideal load/compute/store steady state. New claims require
new traces from an identified source snapshot with matching model support.

These three plots were generated from DEBUG=4 SimX traces produced on
`orcas2.cs.ucla.edu` (`chengxuan`, RTX A4000 host), with L2 enabled and
`WGMMA_NRC=16`, `ATT_K_TILES=4`, `ATT_ITERS=16`, and a real three-slot output
ring. Each trace was filtered only to pipeline-issue, L2 request, and DRAM
request records before plotting with `tools/plot_s2g_timeline.py`. The dark
curves are an 11-bin moving average; faint curves retain the raw samples so
the smoothing does not hide burstiness.

The measured cycle counts are in `steady_kernel_metrics.csv`. For the fair
staged comparison, S2G is 1.066x faster than the TMA-load path whose output is
also staged in SMEM and then copied with LSU stores, and 1.534x faster than the
LSU-load/global-store baseline. The older direct-register global-store mode is
not used for this claim because it does not have the same cross-warp SMEM
ownership requirement.

Stage/wait-N sweep notes, including negative results, are in
`tuning_summary.csv`; all tested points retained correctness.
