#ifndef _COMMON_H_
#define _COMMON_H_

typedef struct {
    uint32_t N;
    uint64_t input_addr;
    uint64_t target_addr; 
    uint64_t loss_addr;
} kernel_arg_t;

#endif