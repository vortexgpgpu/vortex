#ifndef _COMMON_H_
#define _COMMON_H_

#define WARP_SIZE 32
#define NUM_WARPS 4
#define BLOCK_SIZE (WARP_SIZE * NUM_WARPS)
#define NUM_BLOCKS 2
#define ARRAY_SIZE (BLOCK_SIZE * NUM_BLOCKS)

typedef struct {
    uint32_t N;
    uint64_t input_addr;
    uint64_t output_vals_addr;
    uint64_t output_indices_addr;
} kernel_arg_t;

#endif