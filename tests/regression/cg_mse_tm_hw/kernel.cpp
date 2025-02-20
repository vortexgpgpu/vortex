#include <vx_spawn.h>
#include <vx_print.h>
#include <vx_intrinsics.h> 
#include "common.h"

#define SW_THREADS_PER_BLOCK 8
#define SW_THREADS_PER_WARP 32
#define CG_GROUP_SIZE SW_THREADS_PER_WARP

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
    float* inp = (float*)arg->input_addr;
    float* y = (float*)arg->target_addr;
    float* loss = (float*)arg->loss_addr;
    int N = arg->N;

    int cg_group_size = CG_GROUP_SIZE;
    
    float shared_sum[SW_THREADS_PER_BLOCK/CG_GROUP_SIZE];
    int num_warps_array[SW_THREADS_PER_BLOCK];
    int warp_id_array[SW_THREADS_PER_BLOCK];
    int lane_id_array[SW_THREADS_PER_BLOCK];
    float thread_sum_array[SW_THREADS_PER_BLOCK];
    
    // Parallel region #1
    for (int tid = 0; tid < SW_THREADS_PER_BLOCK; tid++) {
        num_warps_array[tid] = SW_THREADS_PER_BLOCK / SW_THREADS_PER_WARP;
        warp_id_array[tid] = tid / SW_THREADS_PER_WARP;
        lane_id_array[tid] = tid % SW_THREADS_PER_WARP;
        thread_sum_array[tid] = 0.0f;
        
        for (int i = tid; i < N; i += SW_THREADS_PER_BLOCK) {
            float diff = inp[i] - y[i];
            thread_sum_array[tid] += diff * diff;
        }
    }
    
    // Parallel region #2 (hierarchical collapsing)
    float cg_reduce_temp_array[SW_THREADS_PER_BLOCK / CG_GROUP_SIZE];
    for (int i = 0; i < SW_THREADS_PER_BLOCK / cg_group_size; i++) {
        float cg_reduce_temp = 0.0f;
        for (int j = 0; j < cg_group_size; j++) {
            int tid = i * cg_group_size + j;
            cg_reduce_temp += thread_sum_array[tid];
        }
        cg_reduce_temp_array[i] = cg_reduce_temp;
    }
    
    // Parallel region #3 (hierarchical collapsing)
    float warp_sum_array[SW_THREADS_PER_BLOCK];
    for (int i = 0; i < SW_THREADS_PER_BLOCK / cg_group_size; i++) {
        for (int j = 0; j < cg_group_size; j++) {
            int tid = i * cg_group_size + j;
            warp_sum_array[tid] = cg_reduce_temp_array[i];
            shared_sum[warp_id_array[tid]] = warp_sum_array[tid];
        }
    }
    
    // Parallel region #4 (hierarchical collapsing)
    float cg_reduce_temp2_array[SW_THREADS_PER_BLOCK / CG_GROUP_SIZE];
    for (int i = 0; i < SW_THREADS_PER_BLOCK / cg_group_size; i++) {
        float cg_reduce_temp2 = 0.0f;
        for (int j = 0; j < cg_group_size; j++) {
            int tid = i * cg_group_size + j;
            warp_sum_array[tid] = (lane_id_array[tid] < num_warps_array[tid]) ? 
                                 shared_sum[lane_id_array[tid]] : 0.0f;
            cg_reduce_temp2 += warp_sum_array[tid];
        }
        cg_reduce_temp2_array[i] = cg_reduce_temp2;
    }
    
    // Parallel region #5 (hierarchical collapsing)
    float block_sum_array[SW_THREADS_PER_BLOCK];
    for (int i = 0; i < SW_THREADS_PER_BLOCK / cg_group_size; i++) {
        for (int j = 0; j < cg_group_size; j++) {
            int tid = i * cg_group_size + j;
            block_sum_array[tid] = cg_reduce_temp2_array[i];
            
            if (tid == 0) {
                loss[0] = block_sum_array[tid] / N;
            }
        }
    }
}

int main() {
	kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
	return vx_spawn_threads(1, &arg->N, nullptr, (vx_kernel_func_cb)kernel_body, arg);
}
