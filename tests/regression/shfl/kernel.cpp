#include <vx_spawn.h>
#include "common.h"

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
	uint32_t count    = arg->task_size;
	int32_t* src0_ptr = (int32_t*)arg->src0_addr;
	int32_t* src1_ptr = (int32_t*)arg->src1_addr;
	int32_t* dst_ptr  = (int32_t*)arg->dst_addr;

	uint32_t offset = blockIdx.x * count;
	for (uint32_t i = 0; i < count; ++i) { 
		dst_ptr[offset+i] = src0_ptr[offset+i] + src1_ptr[offset+i];
	}
  
	vx_tmc(1);
	int val = 14;
	vx_store(val,3);                    
	vx_tmc(2);
	val = 5;
	vx_store(val,3);  
	vx_tmc(4);
	val =45;      
	vx_store(val,3);             
	vx_tmc(8);
	val = 120; 
	vx_store(val,3);                  
	vx_tmc(15); 
	vx_shfl();    
}       
 
int main() {
	kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
	return vx_spawn_threads(1, &arg->num_tasks, nullptr, (vx_kernel_func_cb)kernel_body, arg);
}
