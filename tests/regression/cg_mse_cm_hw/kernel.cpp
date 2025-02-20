#include <vx_spawn.h>
#include <vx_print.h>
#include <vx_intrinsics.h> 
#include "common.h"

#define HW_WARPS_PER_CORE 8
#define HW_THREADS_PER_WARP 32
#define HW_THREADS_PER_CORE HW_WARPS_PER_CORE*HW_THREADS_PER_WARP
#define SW_THREADS_PER_BLOCK HW_THREADS_PER_CORE


void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
    float* inp = (float*)arg->input_addr;
    float* y = (float*)arg->target_addr;
    float* loss = (float*)arg->loss_addr;
    int N = arg->N;

    vx_tile(0b10101010,8);

    float* shared_sum = (float*)__local_mem(HW_THREADS_PER_WARP * sizeof(float));
    int num_warps = blockDim.x / HW_THREADS_PER_WARP;
    int warp_id = threadIdx.x / HW_THREADS_PER_WARP;
    int lane_id = threadIdx.x % HW_THREADS_PER_WARP;
    float thread_sum = 0.0f;
    
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float diff = inp[i] - y[i];
        thread_sum += diff * diff;
    }
    
    float warp_sum = thread_sum;
    for (int offset = HW_THREADS_PER_WARP/2; offset > 0; offset /= 2)
        warp_sum += vx_shfl_sync(offset, 1, warp_sum, -1);
    shared_sum[warp_id] = warp_sum;
    __syncthreads();
    
    warp_sum = (lane_id < num_warps) ? shared_sum[lane_id] : 0.0f;

    float block_sum = warp_sum;
    for (int offset = HW_THREADS_PER_WARP/2; offset > 0; offset /= 2)
        block_sum += vx_shfl_sync(offset, 1, block_sum, -1);

    if (threadIdx.x == 0) {
        loss[0] = block_sum / N;
    }
    vx_tile(0b10000000,32);
}

int main() {
	kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
	return vx_spawn_threads(1, &arg->N, nullptr, (vx_kernel_func_cb)kernel_body, arg);
}
