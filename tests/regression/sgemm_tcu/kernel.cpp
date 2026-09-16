#include "common.h"
#include <vx_spawn2.h>
#include <vx_tensor.h>
#include <vx_intrinsics.h>

namespace vt = vortex::tensor;
using ctx = vt::wmma_context<VX_CFG_NUM_THREADS, vt::ITYPE, vt::OTYPE>;

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  const uint64_t instret_begin = vx_rdinstret_local();
  const __rdcycle_time cycle_begin = vx_rdcycle_sync_begin();
  const uint64_t body_start = ((uint64_t)cycle_begin.hi << 32) | cycle_begin.lo;

  auto pA = reinterpret_cast<ctx::input_t *>(arg->A_addr);
  auto pB = reinterpret_cast<ctx::input_t *>(arg->B_addr);
  auto pC = reinterpret_cast<ctx::output_t *>(arg->C_addr);

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

  for (uint32_t i = 0; i < K; i += ctx::tileK) {
    auto pTileA = pA + tile_row * K + i;
    auto pTileB = pB + tile_col * K + i;

    ctx::load_matrix_sync(fragA, pTileA, K);
    ctx::load_matrix_sync<vt::col_major>(fragB, pTileB, K);
    ctx::mma_sync(fragC, fragA, fragB, fragC);
  }

  // Store the computed C tile
  auto pTileC = pC + tile_row * N + tile_col;
  ctx::store_matrix_sync(pTileC, fragC, N);

  const __rdcycle_time cycle_end = vx_rdcycle_sync_end();
  const uint64_t body_end = ((uint64_t)cycle_end.hi << 32) | cycle_end.lo;
  const uint64_t instret_end = vx_rdinstret_local();
  const uint64_t total_cycles = vx_rdcycle_sync_diff(cycle_begin, cycle_end);
  const uint64_t total_instructions = instret_end - instret_begin;

  if (vx_thread_id() == 0) {
    auto cycles = reinterpret_cast<uint64_t *>(arg->cycles_addr);
    uint32_t block_id = blockIdx.y * gridDim.x + blockIdx.x;
    cycles[2 * block_id + 0] = body_start;
    cycles[2 * block_id + 1] = body_end;
  }

  if (blockIdx.x == 0 && blockIdx.y == 0 && vx_thread_id() == 0) {
    auto metrics = reinterpret_cast<uint64_t *>(arg->metrics_addr);
    metrics[0] = total_cycles;
    metrics[1] = total_instructions;
  }
}
