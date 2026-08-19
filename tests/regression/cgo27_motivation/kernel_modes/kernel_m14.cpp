// GPU-side program for mode 14 -- DTCU_socket, PIPELINED.
//
// The socket counterpart of mode 15, and deliberately NOT the same shape of program,
// because the placement does not allow it. Every core dedicates warp 0 to submitting its
// row band to the engine in its socket; the remaining N warps consume that core's completed
// slices. With SOCKET_SIZE > 1, several producer warps feed the shared socket queue.
//
// Its engine writes D through a dedicated port into that socket's L1 -- which at
// SOCKET_SIZE=1 is this core's own L1, the same one the epilogue is reading from. Mode 15's
// D goes to L2 and leaves the consuming cores' L1s alone.
//
// Producer and consumers are disjoint warps. Producer lane 0 keeps the queue full while one
// consumer lane polls and the other consumer warps wait at their own barrier.

#include "wmma_common.h"
#include "k_dtcu_desc.h"
#include "k_epi_rows.h"
#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include <vx_barrier.h>

__kernel __attribute__((aligned(256))) void moti_dtcu_socket_pipe(kernel_arg_t* __UNIFORM__ arg) {
  const uint32_t core = (uint32_t)vx_core_id();
  const uint32_t warp = threadIdx.x / VX_CFG_NUM_THREADS;
  const uint32_t lane = threadIdx.x % VX_CFG_NUM_THREADS;
  const uint32_t nwarps = blockDim.x / VX_CFG_NUM_THREADS;
  const uint32_t cw_max = nwarps - 1;
  const uint32_t cw = (MOTI_PIPE_CONSUMER_WARPS < cw_max)
                    ? MOTI_PIPE_CONSUMER_WARPS : cw_max;
  if (cw == 0)
    return;

  // This core's band, then that band cut into slices. Cores in one socket enqueue to the
  // same engine, while every core retains ownership of the rows its consumers process.
  // The band is already M/NUM_CORES rows, so there is much less left to slice than mode 15,
  // and the slice floor is a whole 32-row engine tile. At M=128 the band is 32 rows -- one
  // tile -- so T comes out 1 and there is no slice-to-slice overlap. The dedicated
  // producer/consumer launch overhead still makes this distinct from mode 7.
  const moti_slice band = moti_slice_of(arg->M, VX_CFG_NUM_CORES, core);
  if (band.nrow == 0)
    return;
  const uint32_t T = moti_pipe_slices(band.nrow, DTCU_SOCKET_TILE_M);

  // Slots are [core * MOTI_PIPE_TILES + t]. Indexed by the CONSTANT, not by computed T,
  // so the stride matches what the host allocated whatever T comes out to.
  const uint64_t base = arg->desc_addr
                      + (uint64_t)core * MOTI_PIPE_TILES * sizeof(dtensor_desc_t);

  // Warp 0 is producer-only. Lane 0 is sufficient to fill and submit a descriptor; the
  // other lanes stay inactive, as with a conventional warp-specialised TMA producer.
  if (warp == 0) {
    if (lane == 0) {
      for (uint32_t t = 0; t < T; ++t) {
        const moti_slice s = moti_pipe_slice_of(
            band.row0, band.nrow, T, t, DTCU_SOCKET_TILE_M);
        if (s.nrow == 0)
          continue;
        auto d = reinterpret_cast<dtensor_desc_t*>(base) + t;
        const uint64_t da = base + (uint64_t)t * sizeof(dtensor_desc_t);
        moti_fill_desc(d, arg, s, MOTI_SOCKET_TILE_N);
        moti_publish_desc_verified(da, d);
        while (0 == dtensor_socket_start(da))
          ;
      }
    }
    return;
  }

  // Exactly N consumer warps participate. One lane polls at LLC scope; the rest join on
  // a barrier containing consumers only, so the producer never serializes behind the
  // epilogue.
  const uint32_t consumer_warp = warp - 1;
  if (consumer_warp >= cw)
    return;
  vortex::barrier ready(0, cw);
  for (uint32_t t = 0; t < T; ++t) {
    const moti_slice s = moti_pipe_slice_of(
        band.row0, band.nrow, T, t, DTCU_SOCKET_TILE_M);
    if (s.nrow == 0)
      continue;
    const uint64_t da = base + (uint64_t)t * sizeof(dtensor_desc_t);
    if (consumer_warp == 0 && lane == 0)
      moti_wait_desc(da);
    ready.arrive_and_wait();
    const uint32_t etid = consumer_warp * VX_CFG_NUM_THREADS + lane;
    moti_epi_rows(arg, s.row0, s.nrow, 0, 1, etid, cw * VX_CFG_NUM_THREADS);
  }
}


#include "k_epilogue.h"  // apps 6/9: whole-row/column pass after the pipeline launch
