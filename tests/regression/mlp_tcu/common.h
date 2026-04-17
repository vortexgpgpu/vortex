#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

#ifndef NUM_THREADS
#define NUM_THREADS 4
#endif

#ifndef ITYPE
#define ITYPE fp16
#endif

#ifndef OTYPE
#define OTYPE fp16
#endif

#define INPUT_DIM   128
#define HIDDEN1_DIM 256
#define HIDDEN2_DIM 128
#define HIDDEN3_DIM  64
#define OUTPUT_DIM   16

#define NUM_LAYERS  4

#define BATCH_SIZE  16

typedef struct {
    uint32_t input_dim;
    uint32_t output_dim;
    uint64_t weights_addr;  // [output_dim × input_dim] row-major, dtype = ITYPE
    uint64_t bias_addr;     // [output_dim], dtype = OTYPE
} layer_config_t;

typedef struct {
    uint32_t num_layers;
    uint32_t batch_size;        // must equal BATCH_SIZE; passed at runtime for flexibility
    uint64_t input_addr;        // [INPUT_DIM  × batch_size], dtype = ITYPE
    uint64_t output_addr;       // [OUTPUT_DIM × batch_size], dtype = OTYPE
    uint64_t layer_configs_addr;
    uint64_t buffer1_addr;      // [max_hidden × batch_size], dtype = OTYPE
    uint64_t buffer2_addr;      // [max_hidden × batch_size], dtype = OTYPE
    uint64_t buffer1_in_addr;   // [max_hidden × batch_size], dtype = ITYPE (downcast of buffer1)
    uint64_t buffer2_in_addr;   // [max_hidden × batch_size], dtype = ITYPE (downcast of buffer2)
} kernel_arg_t;

#endif // _COMMON_H_
