#include "common.h"
#include <vx_spawn.h>
#include <vx_tensor.h>

namespace vt = vortex::tensor;
using ctx = vt::wmma_context<NUM_THREADS, vt::ITYPE, vt::OTYPE>;

// Decode fp16 bit pattern to value*100 (2 decimal places) using integer math
static inline int32_t fp16_to_x100(uint16_t h) {
  uint32_t e = (h >> 10) & 0x1F;
  uint32_t m = h & 0x3FF;
  if (e == 0) return 0;                // zero / subnormal → 0
  // val = 2^(e-15) * (1024+m) / 1024
  // val*100 = (1024+m)*100 * 2^(e-25)
  int32_t v = (int32_t)(1024 + m) * 100;
  int s = (int)e - 25;
  v = (s >= 0) ? (v << s) : (v >> (-s));
  return (h & 0x8000) ? -v : v;
}

void kernel_body(kernel_arg_t *__UNIFORM__ arg) {
  auto pA = reinterpret_cast<ctx::input_t *>(arg->A_addr);
  auto pB = reinterpret_cast<ctx::input_t *>(arg->B_addr);
  auto pC = reinterpret_cast<ctx::output_t *>(arg->C_addr);

  uint32_t M = arg->M;
  uint32_t N = arg->N;
  uint32_t K = arg->K;

  ctx::fragment_a   fragA;
  ctx::fragment_b   fragB;
  ctx::fragment_acc fragC;

  // calculate tile row & column based on block index
  uint32_t tile_row = blockIdx.y * ctx::tileM;
  uint32_t tile_col = blockIdx.x * ctx::tileN;

  // Initialize accumulator tile to zero
  ctx::fill_fragment(fragC, 0);

  for (int i = 0; i < (K)/2; i += (ctx::tileK)/2) {  
    auto pTileA = pA + tile_row * K + i;

    // Load A tile
    ctx::load_matrix_sync(fragA, pTileA, K);

    // Load B tile
    if constexpr (vt::ITYPE::bits < 8) {
      // For sub-byte matrix B must be in col-major format
      auto pTileB = pB + tile_col * K + i;
      ctx::load_matrix_sync<vt::col_major>(fragB, pTileB, K);
    } else {
      auto pTileB = pB + i * N + tile_col;
      ctx::load_matrix_sync(fragB, pTileB, N);
    }

    // if (vx_thread_id() == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
    //   for (uint32_t r = 0; r < 8; ++r) {
    //     uint32_t packed;
    //     asm volatile("fmv.x.w %0, %1" : "=r"(packed) : "f"(fragA.data[r]));
    //     int32_t lo = fp16_to_x100(packed & 0xFFFF);
    //     int32_t hi = fp16_to_x100((packed >> 16) & 0xFFFF);
    //     vx_printf("fragA[%d] | %d.%02d, %d.%02d\n", r,
    //               lo / 100, lo % 100, hi / 100, hi % 100);
    //   }
    //   for (uint32_t r = 0; r < 8; ++r) {
    //     uint32_t packed;
    //     asm volatile("fmv.x.w %0, %1" : "=r"(packed) : "f"(fragB.data[r]));
    //     int32_t lo = fp16_to_x100(packed & 0xFFFF);
    //     int32_t hi = fp16_to_x100((packed >> 16) & 0xFFFF);
    //     vx_printf("fragB[%d] | %d.%02d, %d.%02d\n", r,
    //               lo / 100, lo % 100, hi / 100, hi % 100);
    //   }
    // }

    // Matrix multiply-accumulate: c += a * b
    ctx::mma_sync(fragC, fragA, fragB, fragC);
  }

  // Store the computed C tile
  auto pTileC = pC + tile_row * N + tile_col;
  ctx::store_matrix_sync(pTileC, fragC, N);
}

int main() {
  auto arg = (kernel_arg_t *)csr_read(VX_CSR_MSCRATCH);
  return vx_spawn_threads(2, arg->grid_dim, arg->block_dim, (vx_kernel_func_cb)kernel_body, arg);
}
