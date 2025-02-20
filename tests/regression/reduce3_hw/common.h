#ifndef _COMMON_H_
#define _COMMON_H_

#define N_SIZE 256
#define HW_THREADS_PER_CORE 32
#define HW_CORES_PER_GRID 4

typedef struct {
    uint32_t N;
    uint64_t input_addr;
    uint64_t output_addr;
} kernel_arg_t;

#endif