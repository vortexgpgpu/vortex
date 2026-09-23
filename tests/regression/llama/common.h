#pragma once

struct matmul_kernel_args_t {
    uint32_t grid_dim[2];
    uint32_t block_dim[2];
    uint64_t A_addr;
    uint64_t B_addr;
    uint64_t C_addr;
    int M;
    int N;
    int K;
};


