// GPU-side program for mode 14 -- DTCU_socket, PIPELINED. Every core is its own producer.
//
// The socket counterpart of mode 15, and deliberately NOT the same shape of program,
// because the placement does not allow it. There are four engines and a core can only reach
// the one in its own socket, so there is no way to nominate a single producer and free the
// rest of the machine: every core must drive its own engine, and every core therefore
// consumes the slices it produced itself. Nobody is spare. That is the cost of putting the
// engine inside the socket, and it is what this mode exists to measure against 15.
//
// Its engine writes D through a dedicated port into that socket's L1 -- which at
// SOCKET_SIZE=1 is this core's own L1, the same one the epilogue is reading from. Mode 15's
// D goes to L2 and leaves the consuming cores' L1s alone.
//
// The socket queue is SOCKET_SIZE*2 = 2 deep, so this cannot submit every slice up front the
// way 15 does. It runs one ahead: submit t, then consume t-1 while t is in flight.

// One thread polls and the rest join through a barrier, for the reason kernel_m15.cpp
// documents: dtensor_check() is an AMO that resolves at the LLC, so a whole block spinning
// on it competes with the engine's own operand fetches.

#include "wmma_common.h"
#include "k_dtcu_desc.h"
#include "k_epi_rows.h"
#include <vx_spawn2.h>
#include <vx_intrinsics.h>

__kernel void moti_dtcu_socket_pipe(kernel_arg_t* __UNIFORM__ arg) {
  const uint32_t core = (uint32_t)vx_core_id();
  if ((core % VX_CFG_SOCKET_SIZE) != 0)
    return;                                     // not this socket's submitter
  const uint32_t sock  = core / VX_CFG_SOCKET_SIZE;
  const uint32_t nsock = VX_CFG_NUM_CORES / VX_CFG_SOCKET_SIZE;

  // This socket's band, then that band cut into slices. Two levels, because the outer split
  // is fixed by which engine a core can reach and the inner one is the pipeline depth.
  //
  // The band is already M/nsock rows, so there is much less left to slice than mode 15 has,
  // and the slice floor is a whole 32-row engine tile. At M=128 the band is 32 rows -- one
  // tile -- so T comes out 1 and this mode IS mode 7. That is not a limitation of the code;
  // it is what putting the engine inside the socket costs at that shape.
  const moti_slice band = moti_slice_of(arg->M, nsock, sock);
  if (band.nrow == 0)
    return;
  const uint32_t T = moti_pipe_slices(band.nrow, DTCU_SOCKET_TILE_M);

  // Slots are [sock * MOTI_PIPE_TILES + t]. Indexed by the CONSTANT, not by the computed T,
  // so the stride matches what the host allocated whatever T comes out to.
  const uint64_t base = arg->desc_addr
                      + (uint64_t)sock * MOTI_PIPE_TILES * sizeof(dtensor_desc_t);

  auto slice_of = [&](uint32_t t) {
    return moti_pipe_slice_of(band.row0, band.nrow, T, t, DTCU_SOCKET_TILE_M);
  };
  auto submit = [&](uint32_t t, moti_slice s) {
    auto d = reinterpret_cast<dtensor_desc_t*>(base) + t;
    const uint64_t da = base + (uint64_t)t * sizeof(dtensor_desc_t);
    moti_fill_desc(d, arg, s, MOTI_SOCKET_TILE_N);
    moti_publish_desc_verified(da, d);
    while (0 == dtensor_socket_start(da))
      ;
  };
  const uint32_t nwarps = blockDim.x / VX_CFG_NUM_THREADS;
  const uint32_t cw = (MOTI_PIPE_CONSUMER_WARPS < nwarps) ? MOTI_PIPE_CONSUMER_WARPS : nwarps;
  auto wait_and_consume = [&](uint32_t t, moti_slice s) {
    const uint64_t da = base + (uint64_t)t * sizeof(dtensor_desc_t);
    if (threadIdx.x == 0)
      moti_wait_desc(da);
    vx_barrier(0, nwarps);
    // This core's own warps are the only consumers of its own slices -- a core cannot reach
    // another socket's engine, so it cannot know when another socket's slice finished
    // without polling a descriptor it has no other business with. That is the asymmetry
    // against mode 15, where one producer frees every warp on the machine.
    const uint32_t warp = threadIdx.x / VX_CFG_NUM_THREADS;
    if (warp < cw)   // this core owns all of its own slices: worker 0 of 1
      moti_epi_rows(arg, s.row0, s.nrow, 0, 1, threadIdx.x, cw * VX_CFG_NUM_THREADS);
  };

  // One ahead: slice t is in flight while t-1 is being consumed. Only thread 0 may submit
  // (a second start on a live descriptor is a second GEMM), but ALL threads consume, so the
  // submit is guarded and the consume is not.
  uint32_t prev = T;                            // T = "nothing outstanding yet"
  moti_slice prev_s{};
  for (uint32_t t = 0; t < T; ++t) {
    const moti_slice s = slice_of(t);
    if (s.nrow == 0)
      continue;
    if (threadIdx.x == 0)
      submit(t, s);
    if (prev != T)
      wait_and_consume(prev, prev_s);
    prev = t; prev_s = s;
  }
  if (prev != T)
    wait_and_consume(prev, prev_s);
}
