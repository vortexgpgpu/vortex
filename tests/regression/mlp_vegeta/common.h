#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

#ifndef TYPE
#define TYPE float
#endif

// MLP Network Configuration (same as mlp_test, all multiples of 16)
#define INPUT_DIM   128
#define HIDDEN1_DIM 256
#define HIDDEN2_DIM 128
#define HIDDEN3_DIM 64
#define OUTPUT_DIM  16

// Number of layers
#define NUM_LAYERS  4

// VEGETA tile dimensions
#define TILE_SIZE 16
#define BATCH_SIZE 16  // VEGETA requires batch=16 for proper 16x16 tiles
#define T_TILE_BYTES (TILE_SIZE * TILE_SIZE * sizeof(TYPE))  // 1KB
#define U_TILE_BYTES (2 * T_TILE_BYTES)  // 2KB
#define V_TILE_BYTES (4 * T_TILE_BYTES)  // 4KB
#define M_TILE_BYTES (TILE_SIZE * TILE_SIZE / 2)  // 128 bytes metadata

// GEMM modes for VEGETA
typedef enum {
    MLP_MODE_TGEMM = 0,  // Dense × Dense (T × T)
    MLP_MODE_UGEMM = 1,  // Dense × 2:4 Sparse (T × U)
    MLP_MODE_VGEMM = 2   // Dense × 1:4 Sparse (T × V)
} mlp_mode_t;

// Layer configuration structure
typedef struct {
    uint32_t input_dim;
    uint32_t output_dim;
    uint64_t weights_addr;   // Weight matrix: input_dim x output_dim (row-major)
    uint64_t bias_addr;      // Bias vector: output_dim
    uint64_t metadata_addr;  // Sparsity metadata (for UGEMM/VGEMM modes)
} layer_config_t;

// Kernel arguments
typedef struct {
    uint32_t num_layers;
    uint32_t mode;              // MLP_MODE_TGEMM, MLP_MODE_UGEMM, or MLP_MODE_VGEMM
    uint64_t input_addr;        // Input data: [BATCH_SIZE x INPUT_DIM]
    uint64_t output_addr;       // Final output: [BATCH_SIZE x OUTPUT_DIM]
    uint64_t layer_configs_addr; // Pointer to layer configurations
    uint64_t buffer1_addr;      // Intermediate buffer 1
    uint64_t buffer2_addr;      // Intermediate buffer 2
} kernel_arg_t;

#endif
