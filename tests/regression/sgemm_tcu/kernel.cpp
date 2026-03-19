#include "common.h"
#include <vx_spawn.h>
#include <vx_tensor.h>
#include <vx_intrinsics.h>

namespace vt = vortex::tensor;
using ctx = vt::wmma_context<NUM_THREADS, vt::ITYPE, vt::OTYPE>;

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

  uint32_t start_cycles = csr_read(VX_CSR_MCYCLE);

  for (int i = 0; i < K; i += ctx::tileK) {
    auto pTileA = pA + tile_row * K + i;

    // Load A tile
    ctx::load_matrix_sync(fragA, pTileA, K);

    // Load B tile (col-major)
    auto pTileB = pB + tile_col * K + i;
    ctx::load_matrix_sync<vt::col_major>(fragB, pTileB, K);

    // Matrix multiply-accumulate: c += a * b
    ctx::mma_sync(fragC, fragA, fragB, fragC);
  }

  // Store the computed C tile
  auto pTileC = pC + tile_row * N + tile_col;
  ctx::store_matrix_sync(pTileC, fragC, N);

  uint32_t end_cycles = csr_read(VX_CSR_MCYCLE);

  // Write per-block cycle count
  auto pCycles = reinterpret_cast<uint32_t*>(arg->cycles_addr);
  uint32_t block_id = blockIdx.y * arg->grid_dim[0] + blockIdx.x;
  pCycles[block_id] = end_cycles - start_cycles;
}

int main() {
  auto arg = (kernel_arg_t *)csr_read(VX_CSR_MSCRATCH);
  return vx_spawn_threads(2, arg->grid_dim, arg->block_dim, (vx_kernel_func_cb)kernel_body, arg);
}
