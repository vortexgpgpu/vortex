#ifndef _CGO27_K_HETERO_H_
#define _CGO27_K_HETERO_H_

// modes 9/10/11 — cores and descriptor engine(s) on the SAME GEMM at the same time.
//
// The output rows are split once, host-side: rows [0, m_tcu) are computed in-core by
// ordinary WMMA blocks, rows [m_tcu, M) by the engine(s) from descriptors the host has
// already sliced. `-p` sets the engine's share, so p=0 degenerates to mode 1 and p=100
// to mode 7/8 — those two endpoints are the sanity check that the split itself is not
// changing the arithmetic.
//
// One kernel entry serves all three because the only thing that differs is WHICH start
// instruction a submitting thread issues, and that is a branch, not a different program.
// The submit happens BEFORE the WMMA work and the wait AFTER it; that ordering is the
// whole point, since a submit that blocked would serialise the two units it is trying
// to overlap.
//
// Slice ownership is not first-come-first-served for the socket engines. A core can only
// reach its OWN socket's engine, so socket slice s must be submitted by a thread running
// on socket s — which is a property of where the block landed, not of its index. The
// cluster slice has no such constraint and goes to whoever claims it. `claimed[]` makes
// each slice go exactly once even when several blocks land on the same core.

#include "wmma_common.h"
#include <vx_spawn2.h>
#include <vx_dtensor.h>
#include <vx_intrinsics.h>

__kernel void moti_hetero(kernel_arg_t* __UNIFORM__ arg) {
  const uint32_t mode  = arg->mode;
  const uint32_t nsl   = arg->num_slices;
  const uint32_t ss    = arg->socket_size ? arg->socket_size : 1u;
  uint32_t* claimed    = reinterpret_cast<uint32_t*>(static_cast<uintptr_t>(arg->ctl_addr));

  // Descriptor this thread submitted, 0 if none. Device addresses are never 0 here, so
  // 0 is a safe sentinel.
  uint64_t mine = 0;

  // ---- 1. hand the engine(s) their work first, so it runs under the WMMA below ----
  if (vx_thread_id() == 0 && nsl != 0) {
    const uint32_t core = static_cast<uint32_t>(vx_core_id());
    // How many of the slices are socket-bound. Mode 10 has none; mode 11 keeps the last
    // slice for the cluster engine.
    const uint32_t n_sock = (mode == MOTI_MODE_HET_TCU_DCLUS) ? 0u
                          : (mode == MOTI_MODE_HET_ALL)       ? (nsl - 1u)
                                                              : nsl;
    if (n_sock != 0 && (core % ss) == 0) {
      const uint32_t s = core / ss;
      if (s < n_sock
          && __atomic_exchange_n(&claimed[s], 1u, __ATOMIC_ACQ_REL) == 0) {
        mine = arg->desc_addr + static_cast<uint64_t>(s) * sizeof(dtensor_desc_t);
        while (0 == dtensor_socket_start(mine))
          ;
      }
    }
    if (mine == 0
        && (mode == MOTI_MODE_HET_TCU_DCLUS || mode == MOTI_MODE_HET_ALL)) {
      const uint32_t s = nsl - 1u; // the cluster slice is always last
      if (__atomic_exchange_n(&claimed[s], 1u, __ATOMIC_ACQ_REL) == 0) {
        mine = arg->desc_addr + static_cast<uint64_t>(s) * sizeof(dtensor_desc_t);
        while (0 == dtensor_cluster_start(mine))
          ;
      }
    }
  }

  // ---- 2. this block's in-core tile, if it falls inside the TCU's row range ----
  // The grid can be taller than the TCU range: it is sized so every core receives a
  // block (otherwise a socket engine would never be handed its slice), and the surplus
  // blocks fall out here.
  const uint32_t tile_row = blockIdx.y * ctx::tileM;
  if (tile_row < arg->m_tcu) {
    const uint32_t N = arg->N, K = arg->K, app = arg->app;
    auto pA = reinterpret_cast<ctx::input_t*>(arg->A_addr);
    auto pB = reinterpret_cast<ctx::input_t*>(arg->B_addr);
    auto pC = reinterpret_cast<ctx::output_t*>(arg->C_addr);
    auto pD = reinterpret_cast<ctx::output_t*>(arg->D_addr);
    const uint32_t tile_col = blockIdx.x * ctx::tileN;

    ctx::fragment_a   fragA;
    ctx::fragment_b   fragB;
    ctx::fragment_acc fragD;
    wmma_seed_C(fragD, pC, tile_row, tile_col, N);

    for (uint32_t i = 0; i < K; i += ctx::tileK) {
      ctx::load_matrix_sync(fragA, pA + tile_row * K + i, K);                 // A row-major
      ctx::load_matrix_sync<vt::col_major>(fragB, pB + tile_col * K + i, K);  // B col-major
      ctx::mma_sync(fragD, fragA, fragB, fragD);
    }
    wmma_fuse_epilogue(fragD, app);
    wmma_store_D(pD, fragD, tile_row, tile_col, N);
  }

  // ---- 3. only now wait on the slice this thread submitted ----
  // Divergent by construction (one lane per block holds a slice). That is safe: the
  // spin depends on the engine, never on the other lanes, so reconvergence cannot
  // deadlock -- and by here the other lanes have no work left anyway.
  if (mine != 0) {
    while (0 == dtensor_check(mine))
      ;
  }
}

#endif // _CGO27_K_HETERO_H_
