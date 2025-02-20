#include <vx_spawn.h>
#include <vx_print.h>
#include "common.h"

#define HW_WARPS_PER_CORE 4
#define HW_THREADS_PER_WARP 4
#define HW_THREADS_PER_CORE (HW_WARPS_PER_CORE * HW_THREADS_PER_WARP)

void warp_reduce_max(float* val, int* index) {
    float cur_max_v = *val;
    int cur_max_idx = *index;
    
    // Perform warp reduction using vx_shfl_sync
    for (int i = 1; i < HW_THREADS_PER_WARP; i <<= 1) { 
        float v = (float)vx_shfl_sync(0, 0xf, (int)cur_max_v, i, -1);
        int idx = vx_shfl_sync(0, 0xf, cur_max_idx, i, -1);
        if (v > cur_max_v) {
            cur_max_v = v;
            cur_max_idx = idx;
        }
    }
    
    *val = cur_max_v;
    *index = cur_max_idx;
}

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
    float* input = (float*)arg->input_addr;
    float* output_vals = (float*)arg->output_vals_addr;
    int* output_indices = (int*)arg->output_indices_addr;
    
    // Enable warp vector mask
    // vx_tile(0b10101010, HW_THREADS_PER_WARP);
    
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    
    // Load input value and index
    float val = input[tid];
    int idx = tid;
    
    // Perform warp reduction
    warp_reduce_max(&val, &idx);
    
    // Only the first thread in each warp writes the result
    if (lane_id == 0) {
        output_vals[warp_id] = val;
        output_indices[warp_id] = idx;
    }
    
    // Disable warp vector mask
    // vx_tile(0b10000000, HW_THREADS_PER_WARP);
}

int main() {
    kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
    return vx_spawn_threads(NUM_BLOCKS, &arg->N, nullptr, (vx_kernel_func_cb)kernel_body, arg);
}