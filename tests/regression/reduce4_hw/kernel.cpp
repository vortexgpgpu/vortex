#include <vx_spawn.h>
#include <vx_print.h>
#include "common.h"
 
void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
    float* input = (float*)arg->input_addr;
    float* output = (float*)arg->output_addr;
    int N = arg->N;

    // Enable first warp threads
    vx_tile(0b10101010, HW_THREADS_PER_WARP);

    // Shared memory for partial sums
    float* shared_sum = (float*)__local_mem(HW_THREADS_PER_CORE * sizeof(float));
    int warp_id = threadIdx.x / HW_THREADS_PER_WARP;
    int lane_id = threadIdx.x % HW_THREADS_PER_WARP;
    
    // Calculate indices for this thread
    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    // Load and sum two elements per thread
    float mySum = 0.0f;
    if (i < N) {
        mySum = input[i];
    }
    if (i + blockDim.x < N) {
        mySum += input[i + blockDim.x];
    }
    
    // Store in shared memory
    shared_sum[threadIdx.x] = mySum;
    __syncthreads();
    
    // Reduce within block
    for (int s = blockDim.x / 2; s > HW_THREADS_PER_WARP; s >>= 1) {
        if (threadIdx.x < s) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    // Final warp reduction using shuffle
    if (threadIdx.x < HW_THREADS_PER_WARP) {
        mySum = shared_sum[threadIdx.x];
        if (blockDim.x >= HW_THREADS_PER_WARP * 2) {
            mySum += shared_sum[threadIdx.x + HW_THREADS_PER_WARP];
        }
        
        // Warp-level reduction using shuffle
        for (int offset = HW_THREADS_PER_WARP/2; offset > 0; offset /= 2) {
            mySum += vx_shfl_sync(1, 0xFF, mySum, offset, HW_THREADS_PER_WARP);
        }
        
        // First thread in warp writes result
        if (lane_id == 0) {
            output[blockIdx.x] = mySum;
        }
    }

    vx_tile(0b10000000, HW_THREADS_PER_WARP);
}

int main() {
    kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
    uint32_t threads_per_core = HW_THREADS_PER_CORE;
    uint32_t cores_per_grid = HW_CORES_PER_GRID;
    return vx_spawn_threads(1, &cores_per_grid, &threads_per_core, (vx_kernel_func_cb)kernel_body, arg);
}