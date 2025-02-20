#include <vx_spawn.h>
#include <vx_print.h>
#include "common.h"


void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
    float* input = (float*)arg->input_addr;
    float* output = (float*)arg->output_addr;
    int N = arg->N;

    // Enable threads
    vx_tile(0b10000000, HW_THREADS_PER_CORE);

    float* shared_sum = (float*)__local_mem(HW_THREADS_PER_CORE * sizeof(float));
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    float mySum = (i < N) ? input[i] : 0.0f;

    if (i + blockDim.x < N) {
        mySum += input[i + blockDim.x];
    }
    
    shared_sum[tid] = mySum;
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            mySum = mySum + shared_sum[tid + s];
            shared_sum[tid] = mySum;
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[blockIdx.x] = mySum;
    }

    vx_tile(0b10000000, HW_THREADS_PER_CORE);
}

int main() {
    kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
    uint32_t threads_per_core = HW_THREADS_PER_CORE;
    uint32_t cores_per_grid = HW_CORES_PER_GRID;
    return vx_spawn_threads(1, &cores_per_grid, &threads_per_core, (vx_kernel_func_cb)kernel_body, arg);
}