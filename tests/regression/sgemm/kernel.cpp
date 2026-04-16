#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include <tensor_cfg.h>
#include "common.h"

namespace vt = vortex::tensor;

// Storage type: what's actually in memory (fp16 promoted to float on host)
using storage_itype_t = typename std::conditional<
    std::is_same<vt::ITYPE, vt::fp16>::value, float,
    typename vt::ITYPE::dtype
>::type;

using simt_otype_t = typename vt::OTYPE::dtype;

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  uint32_t M = arg->M;
  uint32_t N = arg->N;
  uint32_t K = arg->K;

  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row >= (int)M || col >= (int)N)
    return;

  auto C = reinterpret_cast<simt_otype_t*>(arg->C_addr);

  __rdcycle_time t0 = vx_rdcycle_sync_begin();

  simt_otype_t sum = 0;

  if constexpr (vt::ITYPE::bits == 4) {
    // int4: packed 2 nibbles per byte, need extraction + sign extension
    auto A = reinterpret_cast<const uint8_t*>(arg->A_addr);
    auto B = reinterpret_cast<const uint8_t*>(arg->B_addr);
    for (uint32_t e = 0; e < K; ++e) {
      uint32_t a_idx = row * K + e;
      uint8_t a_byte = A[a_idx / 2];
      int32_t a_val = (a_idx & 1) ? (a_byte >> 4) : (a_byte & 0x0F);
      if (a_val & 0x8) a_val |= (int32_t)0xFFFFFFF0; // sign extend

      uint32_t b_idx = e * N + col;
      uint8_t b_byte = B[b_idx / 2];
      int32_t b_val = (b_idx & 1) ? (b_byte >> 4) : (b_byte & 0x0F);
      if (b_val & 0x8) b_val |= (int32_t)0xFFFFFFF0;

      sum += a_val * b_val;
    }
  } else {
    // fp32, int8→int32: load as storage type, cast to otype for multiply
    auto A = reinterpret_cast<const storage_itype_t*>(arg->A_addr);
    auto B = reinterpret_cast<const storage_itype_t*>(arg->B_addr);
    for (uint32_t e = 0; e < K; ++e) {
      sum += (simt_otype_t)A[row * K + e] * (simt_otype_t)B[e * N + col];
    }
  }

  C[row * N + col] = sum;

  __rdcycle_time t1 = vx_rdcycle_sync_end();

  if (threadIdx.x == 0 && threadIdx.y == 0) {
    auto pCycles = reinterpret_cast<uint32_t*>(arg->cycles_addr);
    uint32_t block_id = blockIdx.y * gridDim.x + blockIdx.x;
    pCycles[block_id * 4 + 0] = t0.hi;
    pCycles[block_id * 4 + 1] = t0.lo;
    pCycles[block_id * 4 + 2] = t1.hi;
    pCycles[block_id * 4 + 3] = t1.lo;
  }
}
