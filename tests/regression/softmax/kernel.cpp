#include <vx_spawn.h>
#include "common.h"
#include "vx_print.h"

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
	auto src0_ptr = reinterpret_cast<TYPE*>(arg->src0_addr);
	auto src1_ptr = reinterpret_cast<TYPE*>(arg->src1_addr);
	auto dst_ptr  = reinterpret_cast<TYPE*>(arg->dst_addr);

    uint32_t num_cols = arg->num_cols;

    /*auto local_ptr = __local_mem(1024 * sizeof(TYPE));*/
    /*auto local_A = (TYPE*)local_ptr;*/

    uint32_t tid    = blockIdx.x;
    uint32_t offset = blockIdx.x * num_cols;

    // 1. Find the max value in src0_ptr 
    TYPE max = 0.0;
    for(uint32_t i = 0; i < num_cols; i++){

        auto curr = src0_ptr[offset + i];
        if(curr > max){
            max = curr;
        }
    }

    // 2. Subtract Max and Compute the Exp 
    TYPE sum = 0.0;
    for(uint32_t i = 0; i < num_cols; i++){
        auto subtract = src0_ptr[offset + i] - max; 
        auto computed = exp(subtract);

        src0_ptr[offset + i] = computed; 
        /*local_A[i] = computed;*/
        sum += computed;
    }

    // 3. Normalize  
    for(uint32_t i = 0; i < num_cols; i++){
        /*dst_ptr[offset + i] = local_A[i] / sum; */
        dst_ptr[offset + i] = src0_ptr[offset + i] / sum; 
    }
}

int main() {
	kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
	return vx_spawn_threads(1, &arg->num_rows, nullptr, (vx_kernel_func_cb)kernel_body, arg);
}
