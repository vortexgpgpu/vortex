#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_print.h>
#include <vx_spawn.h>
#include "common.h"

// Pipeline demonstration kernel
// Executes N back-to-back TILE_GEMM operations
// This allows measuring cycle counts to verify pipelining is working
void kernel_body(int task_id, kernel_arg_t* arg) {
  (void)task_id;  // Unused
  
  // Get base addresses
  size_t A_base = arg->A_addr;
  size_t B_base = arg->B_addr;
  size_t C_base = arg->C_addr;
  uint32_t num_tiles = arg->num_tiles;
  
  // Each tile is 16x16 = 256 floats = 1KB
  constexpr size_t tile_stride = TILE_SIZE * TILE_SIZE * sizeof(float);
  
  // Tile register allocation:
  // T0: Weight A
  // T1: Input B
  // T2: Accumulator/Result C
  
  // Zero out accumulator initially by loading from C (will be zeros)
  vx_lt(2, C_base, 0);
  
  // =========================================================
  // Pipeline Test: Execute multiple TILE_GEMM back-to-back
  // 
  // For N=1: Total latency = SINGLE_INSTR_LATENCY = 53 cycles
  // For N=2: Total latency = 53 + PIPELINE_II = 53 + 16 = 69 cycles
  // For N=3: Total latency = 53 + 2*PIPELINE_II = 53 + 32 = 85 cycles
  // ...
  // For N:   Total latency = 53 + (N-1)*16 cycles
  //
  // Speedup vs non-pipelined = (N * 53) / (53 + (N-1)*16)
  // As N->infinity, speedup approaches 53/16 = 3.3x
  // =========================================================
  
  for (uint32_t i = 0; i < num_tiles; ++i) {
    // Load weight tile A[i]
    vx_lt(0, A_base + i * tile_stride, 0);
    
    // Load input tile B[i]
    vx_lt(1, B_base + i * tile_stride, 0);
    
    // Execute TILE_GEMM: C += A * B
    // This is where pipelining effects will be visible
    vx_tgemm(2, 0, 1);
  }
  
  // Store result
  vx_st(C_base, 0, 2);
}

int main() {
  kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
  
  // Single-threaded execution to clearly observe pipeline behavior
  kernel_body(0, arg);
  
  return 0;
}
