#ifndef _CGO27_K_EPI_ROWS_H_
#define _CGO27_K_EPI_ROWS_H_

// The elementwise epilogue over a ROW RANGE, callable from inside another kernel.
//
// This is what the pipelined modes (14/15) run on slice t-1 while the engine is still
// producing slice t. k_epilogue.h's moti_epilogue is the same map as a standalone launch,
// which is what modes 7/8 use; both go through epi_apply_at, so the arithmetic is identical
// and only the scheduling differs.
//
// It lives in its own header rather than in k_epilogue.h so that a pipelined binary does
// not also carry the standalone kernels it will never call. That is not tidiness: adding
// two unused epilogue kernels to a binary moved mode 4 by +44.6 % (32,583 -> 47,118),
// because code it never executes still decides which icache set the code it does execute
// lands in. See common.h on MOTI_APP.

#include "wmma_common.h"
#include <vx_spawn2.h>

#if MOTI_APP_IS_ELEMENTWISE
// ROWS ARE STRIPED BY CORE; THE COLUMNS OF ONE ROW ARE SPLIT ACROSS THAT CORE'S THREADS.
//
// The obvious alternative -- give each WARP its own rows -- was built and it livelocks. A
// core's 16 warps then sit on 16 different rows at once, so the core has 16 rows' worth of
// distinct cache lines in flight instead of one row's. The cores are row 0 of a strict
// priority L2 arbiter (cluster.cpp:153) and the cluster engine is row 2, so past a
// threshold the consumers keep row 0 permanently non-empty, the engine is never granted a
// slot, its `done` never sets, and the consumers poll forever on a producer they have
// starved. It is not a deadlock -- everything is running -- and it is fully reproducible:
// 4 consumer warps at 256x256x64 finish, 8 do not.
//
// Keeping a core on one row at a time is what avoids that, and it is also simply faster:
// at 768x384x192 with GELU, 1,351,467 cycles against 2,511,251 for the warp-striped
// version at the largest setting that still terminates.
//
// `worker` of `nworkers` picks the rows (one consumer CORE each); `tid` of `nthreads`
// splits that row's N columns. nrow == 0 is legal and does nothing.
static inline void moti_epi_rows(kernel_arg_t* arg, uint32_t row0, uint32_t nrow,
                                 uint32_t worker, uint32_t nworkers,
                                 uint32_t tid, uint32_t nthreads) {
  const uint32_t N = arg->N;
  auto pD  = reinterpret_cast<float*>(arg->D_addr);
  auto aux = MOTI_AUX_PTR(arg->C_addr, arg->M, N);

  for (uint32_t r = worker; r < nrow; r += nworkers) {
    const uint32_t row = row0 + r;
    float* d = pD + (size_t)row * N;
    for (uint32_t j = tid; j < N; j += nthreads)
      d[j] = epi_apply_at(d[j], aux, row, j, N);
  }
}
#else
// No elementwise app in this binary: the consumers have nothing to fuse, and the pipelined
// modes become a control measuring what the extra descriptors and the waiting cost on their
// own. Apps 6 and 9 still get their reduction pass from the host, as every mode does.
static inline void moti_epi_rows(kernel_arg_t*, uint32_t, uint32_t, uint32_t, uint32_t,
                                 uint32_t, uint32_t) {}
#endif

#endif // _CGO27_K_EPI_ROWS_H_
