#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

// T Tile dimensions: 16x16 fp32 = 1KB per tile register
#define TILE_SIZE 16
#define T_TILE_BYTES (TILE_SIZE * TILE_SIZE * sizeof(float))  // 1KB

// VEGETA Engine Pipeline Configuration
// These values should match sparse_cfg.h::vegeta_engine_config_t
#define VEGETA_ALPHA 4          // Broadcast factor
#define VEGETA_BETA 2           // Reduction factor
#define VEGETA_NROWS 16         // TILE_K / BETA
#define VEGETA_NCOLS 4          // TOTAL_MACS / (NROWS * ALPHA * BETA)

// Pipeline stage latencies (cycles)
#define WL_LATENCY 16           // Weight Load
#define FF_LATENCY 16           // Feed First
#define FS_LATENCY 15           // Feed Second
#define DR_LATENCY 4            // Drain
#define REDUCE_LATENCY 2        // Reduction
#define SINGLE_INSTR_LATENCY (WL_LATENCY + FF_LATENCY + FS_LATENCY + DR_LATENCY + REDUCE_LATENCY)

// Pipeline initiation interval
#define PIPELINE_II FF_LATENCY  // = 16 cycles between instruction starts

// Tile load latency (1KB at 64B/cycle)
#define TILE_LOAD_LATENCY 16

// Test mode: number of back-to-back TILE_GEMM operations
typedef struct {
  uint64_t A_addr;      // Matrix A tiles (N_TILES * 1KB each)
  uint64_t B_addr;      // Matrix B tiles (N_TILES * 1KB each)
  uint64_t C_addr;      // Matrix C result tiles (N_TILES * 1KB each)
  uint32_t num_tiles;   // Number of TILE_GEMM operations
} kernel_arg_t;

#endif // _COMMON_H_
