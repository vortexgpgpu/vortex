#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

#ifndef TYPE
#define TYPE float
#endif

// MLP Network Configuration (all dimensions multiples of 16)
#define INPUT_DIM   128
#define HIDDEN1_DIM 256
#define HIDDEN2_DIM 128
#define HIDDEN3_DIM 64
#define OUTPUT_DIM  16

// Number of layers
#define NUM_LAYERS  4

// Layer configuration structure
typedef struct {
    uint32_t input_dim;
    uint32_t output_dim;
    uint64_t weights_addr;  // Weight matrix: output_dim x input_dim
    uint64_t bias_addr;     // Bias vector: output_dim
} layer_config_t;

// Kernel arguments
typedef struct {
    uint32_t num_layers;
    uint32_t batch_size;        // Number of input samples
    uint64_t input_addr;        // Input data
    uint64_t output_addr;       // Final output
    uint64_t layer_configs_addr; // Pointer to layer configurations
    // Intermediate buffers for layer outputs
    uint64_t buffer1_addr;      // For layer 1 output / layer 2 input
    uint64_t buffer2_addr;      // For layer 2 output / layer 3 input
} kernel_arg_t;

#endif
