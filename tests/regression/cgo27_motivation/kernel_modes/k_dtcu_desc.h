#ifndef _CGO27_K_DTCU_DESC_H_
#define _CGO27_K_DTCU_DESC_H_

// Descriptor construction for the two engine modes. The KERNEL builds its descriptor --
// the split follows vx_core_id(), which the host does not know at enqueue time, and
// every input is already in kernel_arg_t or is a build constant, so nothing had to be
// added to that struct.

#include "wmma_common.h"
#include <vx_dtensor.h>
#include <vx_intrinsics.h>

// Engine tile-N. The cluster engine accepts 16..128 and the harness has always used 32;
// the socket engine has exactly one legal value because its output must fit the socket's
// L1 (see the RFC 1.2). Compile-time, so shape_n_size folds to a constant.
#define MOTI_CLUSTER_TILE_N  32u
#define MOTI_SOCKET_TILE_N   ((uint32_t)DTCU_SOCKET_TILE_N_MAX)

// Row band for slice `s` of `slices`, over `M` rows. Ragged tails are legal -- the engine
// clamps a partial tile in hardware -- so the only thing to avoid is an EMPTY slice,
// which would make the engine abort on M == 0.
struct moti_slice { uint32_t row0, nrow; };
static inline moti_slice moti_slice_of(uint32_t M, uint32_t slices, uint32_t s) {
  const uint32_t rows = (M + slices - 1) / slices;
  const uint32_t row0 = s * rows;
  const uint32_t end  = (row0 + rows <= M) ? (row0 + rows) : M;
  return moti_slice{ row0, (end > row0) ? (end - row0) : 0u };
}

// Fill a descriptor for one row band. Only the ROW origin moves between slices: A and
// C/D are row-major so a band is contiguous, and B is shared by every slice untouched.
static inline void moti_fill_desc(dtensor_desc_t* d, kernel_arg_t* arg,
                                  moti_slice sl, uint32_t tile_n) {
  using it = typename vortex::tensor::ITYPE;
  using ot = typename vortex::tensor::OTYPE;
  const uint32_t N = arg->N, K = arg->K;

  d->ptrA = arg->A_addr + (uint64_t)sl.row0 * K * sizeof(typename it::dtype);
  d->ptrB = arg->B_addr;
  d->ptrC = arg->C_addr + (uint64_t)sl.row0 * N * sizeof(typename ot::dtype);
  d->ptrD = arg->D_addr + (uint64_t)sl.row0 * N * sizeof(typename ot::dtype);
  d->ldmA = K; d->ldmB = K; d->ldmC = N; d->ldmD = N;   // elements, not bytes
  d->M = (uint16_t)sl.nrow; d->N = (uint16_t)N; d->K = (uint16_t)K;
  d->fmt_s = (uint8_t)it::id; d->fmt_d = (uint8_t)ot::id;
  d->flags = 0x0;                                        // D = C + A*B, TMA overlap on
  d->shape_n_size = dtcu_shape_n_size(tile_n);
  d->shape_policy = 0;
  d->done = 0;   // the engine sets this; a consumer tells "finished" from "not started"
                 // by that transition alone, so it must be zero BEFORE the submit
}

// Make the descriptor visible to the engine, which reads it back out of memory.
//
// A fence is NOT enough. Core stores are write-through and fire-and-forget -- nothing
// acknowledges them (that is what the `strsp` opt-in exists for, RFC 1.6), so `fence`
// has no completion to wait on and the engine's descriptor read can pass the fill. That
// is not theoretical: with only the fence, mode 8's four slices produced 6,144 errors --
// exactly the 3/4 of the output belonging to the three descriptors the engine read as
// still-zero, each of which it retired instantly and flagged done.
//
// The AMO is what closes it. dtensor_check() is `amoor.w rd, x0` on the descriptor's
// `done` field, which takes the cache's AmoProbe path: it invalidates the local line and
// resolves at the last-level cache. Issuing it after the fill forces this core's stores
// to that same 64-byte line to have reached the LLC before the start instruction goes
// out. The returned value is discarded -- we only just wrote zero there.
static inline void moti_publish_desc(uint64_t desc_addr) {
  vx_fence();                        // orders the fill's stores ahead of the AMO
  (void)dtensor_check(desc_addr);    // drains them to the LLC the engine reads from
}

#endif // _CGO27_K_DTCU_DESC_H_
