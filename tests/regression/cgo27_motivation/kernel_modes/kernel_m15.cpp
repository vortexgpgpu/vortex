// GPU-side program for mode 15 -- DTCU_cluster, PIPELINED. One producer, the rest consume.
//
// Mode 8 splits the GEMM across cores and has every core spin on its own descriptor until
// the whole band is done. There is one MAC array, so that split buys no compute
// parallelism.
//
// There is only ONE engine here, so only ONE core is needed to feed it. Core 0 submits
// every slice. Warp 0 of core 0 is producer-only; N consumer warps on every core, including
// the remaining warps of core 0, partition each completed slice and run its epilogue while
// the engine is already working on the next slice.
//
// Why a consumer on another core can see this at all: dtensor_check() is an ordinary AMO on
// the descriptor's done word, not an engine instruction (vx_dtensor.h documents exactly
// this -- "completion has to be observable by cores that never issued the descriptor"). So
// no extra flag, no barrier, no cross-core protocol; the completion signal that already
// exists is the handshake.
//
// The cluster queue is NUM_CORES*2 deep, so all MOTI_PIPE_TILES slices are submitted up
// front and the engine never idles between them.
//
// D goes to L2, so no core's L1 holds it and a consumer's working set is its own. That is
// the contrast with mode 14, where the engine writes D through a dedicated port into the
// very L1 its consumer is reading from.

// ONE THREAD PER CORE POLLS, and the rest join through a barrier. dtensor_check() is an
// AMO, which by design bypasses the core's L1 and resolves at the LLC (that is what makes
// it visible to a core that never issued the descriptor). Letting every thread spin on it
// therefore aims 3 cores x 16 warps x 32 lanes of LLC traffic at one cache line, and it
// lands on the same port the engine is fetching its operands through: the first build did
// exactly that and the engine starved -- next_tile_load_stall=30,773 of 83,208 cycles,
// tma_mem_wait=26,558, for a GEMM that takes 21,780 cycles unpipelined.

#include "wmma_common.h"
#include "k_dtcu_desc.h"
#include "k_epi_rows.h"
#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include <vx_barrier.h>

__kernel __attribute__((aligned(256))) void moti_dtcu_cluster_pipe(kernel_arg_t* __UNIFORM__ arg) {
  const uint32_t core = (uint32_t)vx_core_id();
  const uint32_t warp = threadIdx.x / VX_CFG_NUM_THREADS;
  const uint32_t lane = threadIdx.x % VX_CFG_NUM_THREADS;
  const uint32_t nwarps = blockDim.x / VX_CFG_NUM_THREADS;
  const uint32_t cw_max = nwarps - 1;
  const uint32_t cw = (MOTI_PIPE_CONSUMER_WARPS < cw_max)
                    ? MOTI_PIPE_CONSUMER_WARPS : cw_max;
  if (cw == 0)
    return;
  const uint64_t base = arg->desc_addr;
  // Slices are whole 64-row engine tiles, so T can come out below MOTI_PIPE_TILES on a
  // short matrix. Every core computes it from the same inputs, so producer and consumers
  // agree without communicating.
  const uint32_t T = moti_pipe_slices(arg->M, DTCU_CLUSTER_TILE_M);

  // PRODUCER: warp 0 of core 0, with lane 0 issuing. The descriptor is one object per slice,
  // so
  // every lane would write the same bytes and then start the same address -- and a second
  // start on a live descriptor is a second GEMM, not a no-op.
  //
  // The other warps on core 0 remain consumers; excluding the whole core was not producer
  // specialisation, it discarded useful execution capacity.
  if (core == 0 && warp == 0) {
    if (lane != 0)
      return;
    for (uint32_t t = 0; t < T; ++t) {
      const moti_slice sl = moti_pipe_slice_of(0, arg->M, T, t, DTCU_CLUSTER_TILE_M);
      if (sl.nrow == 0)
        continue;                              // fewer rows than slices; nothing to submit
      auto d = reinterpret_cast<dtensor_desc_t*>(base) + t;
      const uint64_t da = base + (uint64_t)t * sizeof(dtensor_desc_t);
      moti_fill_desc(d, arg, sl, MOTI_CLUSTER_TILE_N);
      moti_publish_desc_verified(da, d);
      while (0 == dtensor_cluster_start(da))   // 0 = queue full, nothing was queued
        ;
    }
    return;
  }

  // CONSUMERS: N warps of every core, indexed by core so no two own the same row.
  // Slices are taken in order, so the wait is always on the oldest outstanding one.
  const uint32_t consumer_warp = (core == 0) ? (warp - 1) : warp;
  if (consumer_warp >= cw)
    return;
  const uint32_t nthreads = cw * VX_CFG_NUM_THREADS;
  (void)nthreads;   // unused when the warp-striped layout is selected
  const uint32_t cbase = core, ncore = VX_CFG_NUM_CORES;
#if MOTI_PIPE_STRIPE_WARP
  // Each warp takes its own rows and its 32 lanes split that row. Livelocks past a few
  // warps; retained so the sweep can quote it.
  const uint32_t worker = cbase * cw + consumer_warp, nworkers = ncore * cw;
  const uint32_t etid = lane, enthreads = VX_CFG_NUM_THREADS;
#else
  const uint32_t worker = cbase, nworkers = ncore;
  const uint32_t etid = consumer_warp * VX_CFG_NUM_THREADS + lane;
  const uint32_t enthreads = nthreads;
#endif
  vortex::barrier ready(0, cw);
  for (uint32_t t = 0; t < T; ++t) {
    const moti_slice sl = moti_pipe_slice_of(0, arg->M, T, t, DTCU_CLUSTER_TILE_M);
    if (sl.nrow == 0)
      continue;   // uniform across the block -- every warp takes the same branch, so the
                  // barrier below is still reached the same number of times by all of them
    const uint64_t da = base + (uint64_t)t * sizeof(dtensor_desc_t);
    if (consumer_warp == 0 && lane == 0)
      moti_wait_desc(da);
    ready.arrive_and_wait();
    moti_epi_rows(arg, sl.row0, sl.nrow, worker, nworkers, etid, enthreads);
  }
}


#include "k_epilogue.h"  // apps 6/9: whole-row/column pass after the pipeline launch
