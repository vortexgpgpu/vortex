#include <vx_spawn.h>
#include <vx_intrinsics.h>
#include <cmath>
#include "common.h"

// Aligned tile buffer for intermediate results 
static TYPE g_C_tile[TILE_SIZE * TILE_SIZE] __attribute__((aligned(64)));

// ReLU activation (in-place) for batched data
inline void apply_relu_batch(TYPE* data, uint32_t batch_size, uint32_t dim) {
    for (uint32_t b = 0; b < batch_size; ++b) {
        for (uint32_t i = 0; i < dim; ++i) {
            TYPE& v = data[b * dim + i];
            if (v < 0) v = 0;
        }
    }
}

// Softmax for each sample in batch
void apply_softmax_batch(TYPE* data, uint32_t batch_size, uint32_t dim) {
    for (uint32_t b = 0; b < batch_size; ++b) {
        TYPE* sample = data + b * dim;
        
        TYPE max_val = sample[0];
        for (uint32_t i = 1; i < dim; ++i) {
            if (sample[i] > max_val) max_val = sample[i];
        }
        
        TYPE sum_exp = 0.0f;
        for (uint32_t i = 0; i < dim; ++i) {
            sample[i] = expf(sample[i] - max_val);
            sum_exp += sample[i];
        }
        
        for (uint32_t i = 0; i < dim; ++i) {
            sample[i] /= sum_exp;
        }
    }
}

// Initialize C tile buffer to zeros
inline void clear_C_tile() {
    for (uint32_t i = 0; i < TILE_SIZE * TILE_SIZE; ++i) {
        g_C_tile[i] = 0.0f;
    }
}

// =============================================================================
// TGEMM: Dense × Dense (T × T → T)
// =============================================================================
void vegeta_tgemm_tile_accumulate(TYPE* A_tile, TYPE* B_tile, TYPE* C_tile) {
    vx_lt(0, (size_t)C_tile, 0);  // Load C to T0 (accumulator)
    vx_lt(1, (size_t)A_tile, 0);  // Load A to T1
    vx_lt(2, (size_t)B_tile, 0);  // Load B to T2
    vx_tgemm(0, 1, 2);             // T0 += T1 × T2
    vx_st((size_t)C_tile, 0, 0);  // Store result
}

void vegeta_layer_tgemm_batch(
    TYPE* input, TYPE* weights, TYPE* bias, TYPE* output,
    uint32_t in_dim, uint32_t out_dim
) {
    uint32_t in_tiles = in_dim / TILE_SIZE;
    uint32_t out_tiles = out_dim / TILE_SIZE;
    
    for (uint32_t out_t = 0; out_t < out_tiles; ++out_t) {
        clear_C_tile();
        
        for (uint32_t in_t = 0; in_t < in_tiles; ++in_t) {
            TYPE* A_tile = input + in_t * TILE_SIZE;
            TYPE* B_tile = weights + in_t * TILE_SIZE * out_dim + out_t * TILE_SIZE;
            vegeta_tgemm_tile_accumulate(A_tile, B_tile, g_C_tile);
        }
        
        for (uint32_t row = 0; row < BATCH_SIZE; ++row) {
            for (uint32_t col = 0; col < TILE_SIZE; ++col) {
                output[row * out_dim + out_t * TILE_SIZE + col] = 
                    g_C_tile[row * TILE_SIZE + col] + bias[out_t * TILE_SIZE + col];
            }
        }
    }
}

// =============================================================================
// UGEMM: Dense × 2:4 Sparse (T × U → T)
// For 2:4 sparsity: weights are compressed to half size, metadata indicates positions
// =============================================================================
void vegeta_ugemm_tile_accumulate(TYPE* A_tile, TYPE* B_tile, uint8_t* M_tile, TYPE* C_tile) {
    vx_lt(0, (size_t)C_tile, 0);  // Load C to T0 (accumulator)
    vx_lt(1, (size_t)A_tile, 0);  // Load A to T1
    vx_lm(1, (size_t)M_tile, 0);  // Load metadata to M1
    vx_lu(2, (size_t)B_tile, 0);  // Load B (2KB) to U2
    vx_ugemm(0, 1, 2);             // T0 += T1 × U2 (with M1 metadata)
    vx_st((size_t)C_tile, 0, 0);  // Store result
}

void vegeta_layer_ugemm_batch(
    TYPE* input, TYPE* weights, uint8_t* metadata, TYPE* bias, TYPE* output,
    uint32_t in_dim, uint32_t out_dim
) {
    uint32_t in_tiles = in_dim / TILE_SIZE;
    uint32_t out_tiles = out_dim / TILE_SIZE;
    
    for (uint32_t out_t = 0; out_t < out_tiles; ++out_t) {
        clear_C_tile();
        
        for (uint32_t in_t = 0; in_t < in_tiles; ++in_t) {
            TYPE* A_tile = input + in_t * TILE_SIZE;
            // Compressed weights: half the K dimension
            TYPE* B_tile = weights + (in_t * TILE_SIZE / 2) * out_dim + out_t * TILE_SIZE;
            // Metadata: 128 bytes per tile
            uint8_t* M_tile = metadata + (in_t * out_tiles + out_t) * M_TILE_BYTES;
            vegeta_ugemm_tile_accumulate(A_tile, B_tile, M_tile, g_C_tile);
        }
        
        for (uint32_t row = 0; row < BATCH_SIZE; ++row) {
            for (uint32_t col = 0; col < TILE_SIZE; ++col) {
                output[row * out_dim + out_t * TILE_SIZE + col] = 
                    g_C_tile[row * TILE_SIZE + col] + bias[out_t * TILE_SIZE + col];
            }
        }
    }
}

// =============================================================================
// VGEMM: Dense × 1:4 Sparse (T × V → T)
// For 1:4 sparsity: weights are compressed to quarter size, metadata indicates positions
// =============================================================================
void vegeta_vgemm_tile_accumulate(TYPE* A_tile, TYPE* B_tile, uint8_t* M_tile, TYPE* C_tile) {
    vx_lt(0, (size_t)C_tile, 0);  // Load C to T0 (accumulator)
    vx_lt(1, (size_t)A_tile, 0);  // Load A to T1
    vx_lm(1, (size_t)M_tile, 0);  // Load metadata to M1
    vx_lv(1, (size_t)B_tile, 0);  // Load B (4KB) to V1
    vx_vgemm(0, 1, 1);             // T0 += T1 × V1 (with M1 metadata)
    vx_st((size_t)C_tile, 0, 0);  // Store result
}

void vegeta_layer_vgemm_batch(
    TYPE* input, TYPE* weights, uint8_t* metadata, TYPE* bias, TYPE* output,
    uint32_t in_dim, uint32_t out_dim
) {
    uint32_t in_tiles = in_dim / TILE_SIZE;
    uint32_t out_tiles = out_dim / TILE_SIZE;
    
    for (uint32_t out_t = 0; out_t < out_tiles; ++out_t) {
        clear_C_tile();
        
        for (uint32_t in_t = 0; in_t < in_tiles; ++in_t) {
            TYPE* A_tile = input + in_t * TILE_SIZE;
            // Compressed weights: quarter the K dimension
            TYPE* B_tile = weights + (in_t * TILE_SIZE / 4) * out_dim + out_t * TILE_SIZE;
            // Metadata: 128 bytes per tile
            uint8_t* M_tile = metadata + (in_t * out_tiles + out_t) * M_TILE_BYTES;
            vegeta_vgemm_tile_accumulate(A_tile, B_tile, M_tile, g_C_tile);
        }
        
        for (uint32_t row = 0; row < BATCH_SIZE; ++row) {
            for (uint32_t col = 0; col < TILE_SIZE; ++col) {
                output[row * out_dim + out_t * TILE_SIZE + col] = 
                    g_C_tile[row * TILE_SIZE + col] + bias[out_t * TILE_SIZE + col];
            }
        }
    }
}

// =============================================================================
// Kernel Entry Point
// =============================================================================
void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
    auto input = reinterpret_cast<TYPE*>(arg->input_addr);
    auto output = reinterpret_cast<TYPE*>(arg->output_addr);
    auto layer_configs = reinterpret_cast<layer_config_t*>(arg->layer_configs_addr);
    auto buffer1 = reinterpret_cast<TYPE*>(arg->buffer1_addr);
    auto buffer2 = reinterpret_cast<TYPE*>(arg->buffer2_addr);
    uint32_t mode = arg->mode;
    
    TYPE* layer_input = input;
    TYPE* layer_output = buffer1;
    
    for (uint32_t layer = 0; layer < arg->num_layers; ++layer) {
        auto& config = layer_configs[layer];
        auto weights = reinterpret_cast<TYPE*>(config.weights_addr);
        auto bias = reinterpret_cast<TYPE*>(config.bias_addr);
        auto metadata = reinterpret_cast<uint8_t*>(config.metadata_addr);
        
        if (layer == arg->num_layers - 1) {
            layer_output = output;
        } else if (layer % 2 == 0) {
            layer_output = buffer1;
        } else {
            layer_output = buffer2;
        }
        
        // Execute layer based on mode
        if (mode == MLP_MODE_TGEMM) {
            vegeta_layer_tgemm_batch(layer_input, weights, bias, layer_output,
                                     config.input_dim, config.output_dim);
        } else if (mode == MLP_MODE_UGEMM) {
            vegeta_layer_ugemm_batch(layer_input, weights, metadata, bias, layer_output,
                                     config.input_dim, config.output_dim);
        } else if (mode == MLP_MODE_VGEMM) {
            vegeta_layer_vgemm_batch(layer_input, weights, metadata, bias, layer_output,
                                     config.input_dim, config.output_dim);
        }
        
        // Apply ReLU for hidden layers
        if (layer < arg->num_layers - 1) {
            apply_relu_batch(layer_output, BATCH_SIZE, config.output_dim);
        }
        
        layer_input = layer_output;
    }
    
    apply_softmax_batch(output, BATCH_SIZE, OUTPUT_DIM);
}

int main() {
    kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
    return vx_spawn_threads(1, nullptr, nullptr, (vx_kernel_func_cb)kernel_body, arg);
}
