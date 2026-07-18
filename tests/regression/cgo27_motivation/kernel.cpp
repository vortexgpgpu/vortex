#include "common.h"
#include <vx_spawn2.h>
#include <vx_tensor.h>
#include <vx_dtensor.h>
#include <vx_intrinsics.h>
#include <vx_dxa.h>
#include <vx_barrier.h>

namespace vt = vortex::tensor;
using ctx = vt::wmma_context<VX_CFG_NUM_THREADS, vt::ITYPE, vt::OTYPE>;

// Software fp16 -> fp32 (IEEE half to single). Used only by the SIMT path
// (mode 0), which lacks a hardware tensor unit and must convert operands in
// the scalar pipeline. Source: standard half->float bit expansion.
static inline float h2f(uint16_t h) {
  uint32_t s = (uint32_t)(h & 0x8000u) << 16;
  uint32_t e = (h >> 10) & 0x1fu;
  uint32_t m = h & 0x3ffu;
  uint32_t out;
  if (e == 0) {
    if (m == 0) {
      out = s;                                   // +/- 0
    } else {                                     // subnormal
      e = 127u - 15u + 1u;
      while ((m & 0x400u) == 0) { m <<= 1; e--; }
      m &= 0x3ffu;
      out = s | (e << 23) | (m << 13);
    }
  } else if (e == 0x1fu) {
    out = s | 0x7f800000u | (m << 13);           // inf / nan
  } else {
    out = s | ((e + (127u - 15u)) << 23) | (m << 13);
  }
  union { uint32_t u; float f; } cvt;
  cvt.u = out;
  return cvt.f;
}

// cgo27_motivation: one kernel binary, five modes selected by arg->mode. All
// compute the same GEMM D = C + A*B on the same input (A row-major, B col-major).
//
//   mode 0 : in-core SIMT  -- scalar MAC loop, one thread per output element.
//            Source: tests/regression/sgemm/kernel.cpp (scalar accumulate loop),
//            adapted to B col-major + software fp16->fp32.
//   mode 1 : in-core TCU   -- per-warp WMMA fragments.
//            Source: tests/regression/dtcu_compare/kernel.cpp (mode 0).
//   mode 2 : in-core TCU + DXA -- same WMMA, but A/B tiles are staged into smem
//            by the DXA engine instead of loaded directly from global.
//            Source: tests/regression/sgemm_tcu_wg_dxa/kernel.cpp
//            (vx_dxa_issue_2d_wg + vortex::barrier), retargeted from WGMMA to
//            the per-warp WMMA fragment path used in mode 1.
//   mode 3 : DTCU          -- single thread fires a descriptor, spins on poll.
//            Source: tests/regression/dtcu_compare/kernel.cpp (mode 1).
//   mode 4 : DTCU + DTCU_TMA -- identical descriptor path to mode 3. The SimX
//            DTCU always prefetches operands through its TMA engine, so this
//            path is byte- and cycle-identical to mode 3 unless the simulator is
//            rebuilt with a prefetch-suppression knob (see main.cpp note).
__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  const uint32_t N = arg->N;
  const uint32_t K = arg->K;

  if (arg->mode == 0) {
    // ---- in-core SIMT: one thread computes one output element ----
    auto pA = reinterpret_cast<const uint16_t*>(arg->A_addr); // fp16 storage
    auto pB = reinterpret_cast<const uint16_t*>(arg->B_addr);
    auto pC = reinterpret_cast<const float*>(arg->C_addr);
    auto pD = reinterpret_cast<float*>(arg->D_addr);

    uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row = blockIdx.y;

    float sum = pC[row * N + col];
    for (uint32_t k = 0; k < K; ++k) {
      sum += h2f(pA[row * K + k]) * h2f(pB[col * K + k]); // B col-major
    }
    pD[row * N + col] = sum;

  } else if (arg->mode == 1) {
    // ---- in-core TCU (WMMA), operands loaded directly from global ----
    auto pA = reinterpret_cast<ctx::input_t*>(arg->A_addr);
    auto pB = reinterpret_cast<ctx::input_t*>(arg->B_addr);
    auto pC = reinterpret_cast<ctx::output_t*>(arg->C_addr);
    auto pD = reinterpret_cast<ctx::output_t*>(arg->D_addr);

    ctx::fragment_a   fragA;
    ctx::fragment_b   fragB;
    ctx::fragment_acc fragD;

    uint32_t tile_row = blockIdx.y * ctx::tileM;
    uint32_t tile_col = blockIdx.x * ctx::tileN;

    ctx::load_matrix_sync(fragD, pC + tile_row * N + tile_col, N); // seed C

    for (uint32_t i = 0; i < K; i += ctx::tileK) {
      ctx::load_matrix_sync(fragA, pA + tile_row * K + i, K);            // A row-major
      ctx::load_matrix_sync<vt::col_major>(fragB, pB + tile_col * K + i, K); // B col-major
      ctx::mma_sync(fragD, fragA, fragB, fragD);
    }

    ctx::store_matrix_sync(pD + tile_row * N + tile_col, fragD, N);

  } else if (arg->mode == 2) {
    // ---- in-core TCU (WMMA), operands staged into smem by DXA ----
    auto pC = reinterpret_cast<ctx::output_t*>(arg->C_addr);
    auto pD = reinterpret_cast<ctx::output_t*>(arg->D_addr);

    uint32_t tile_row = blockIdx.y * ctx::tileM;
    uint32_t tile_col = blockIdx.x * ctx::tileN;

    auto smem   = reinterpret_cast<ctx::input_t*>(__local_mem());
    auto A_smem = smem;                                  // [tileM x tileK]
    auto B_smem = smem + ctx::tileM * ctx::tileK;         // [tileN x tileK]

    ctx::fragment_a   fragA;
    ctx::fragment_b   fragB;
    ctx::fragment_acc fragD;
    ctx::load_matrix_sync(fragD, pC + tile_row * N + tile_col, N); // seed C

    vortex::barrier bar(0);
    for (uint32_t i = 0; i < K; i += ctx::tileK) {
      // DXA fetch: A tile [tile_row.., i..] and B tile [tile_col.., i..] -> smem.
      bar.expect_tx(2);
      vx_dxa_issue_2d_wg(DESC_A, bar.id(), A_smem, i, tile_row);
      vx_dxa_issue_2d_wg(DESC_B, bar.id(), B_smem, i, tile_col);
      bar.arrive_and_wait();

      ctx::load_matrix_sync(fragA, A_smem, ctx::tileK);
      ctx::load_matrix_sync<vt::col_major>(fragB, B_smem, ctx::tileK);
      ctx::mma_sync(fragD, fragA, fragB, fragD);

      bar.arrive_and_wait(); // ensure WMMA done before next DXA overwrites smem
    }

    ctx::store_matrix_sync(pD + tile_row * N + tile_col, fragD, N);

  } else {
    // ---- DTCU (mode 3) and DTCU + DTCU_TMA (mode 4): identical descriptor path ----
    if (vx_thread_id() == 0) {
      dtensor_start(arg->desc_addr);
      while (0 == dtensor_poll()) {
        // busy-wait until the DTCU signals completion
      }
    }
  }
}
