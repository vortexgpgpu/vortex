#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

#ifndef NUM_THREADS
#define NUM_THREADS 4
#endif

#ifndef ITYPE
#define ITYPE fp32
#endif

#ifndef OTYPE
#define OTYPE fp32
#endif

#define INPUT_DIM   128
#define HIDDEN1_DIM 256
#define HIDDEN2_DIM 128
#define HIDDEN3_DIM  64
#define OUTPUT_DIM   16

#define NUM_LAYERS   4

#define BATCH_SIZE   16

typedef struct {
    uint32_t input_dim;
    uint32_t output_dim;
    uint64_t weights_addr; 
    uint64_t bias_addr;    
} layer_config_t;

typedef struct {
    uint32_t num_layers;
    uint32_t batch_size;
    uint32_t sparsity_degree; // 1 for 1:4, 2 for 2:4
    uint32_t _pad;            // explicit padding so uint64_t fields are 8-byte aligned
    uint64_t input_addr;      // [INPUT_DIM × batch_size], ITYPE
    uint64_t output_addr;     // [OUTPUT_DIM × batch_size], OTYPE
    uint64_t layer_configs_addr;
    uint64_t buffer1_addr;    
    uint64_t buffer2_addr;    
} kernel_arg_t;

#endif // _COMMON_H_
