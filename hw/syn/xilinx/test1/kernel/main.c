#include <stdint.h>
#include <vx_intrinsics.h>

#define KERNEL_ARG_DEV_MEM_ADDR 0x40

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  uint32_t count;
  uint32_t src_addr;
  uint32_t dst_addr;  
} kernel_arg_t;

void main() {
	kernel_arg_t* arg = (kernel_arg_t*)KERNEL_ARG_DEV_MEM_ADDR;
	uint32_t count   = arg->count;
	int32_t* src_ptr = (int32_t*)arg->src_addr;
	int32_t* dst_ptr = (int32_t*)arg->dst_addr;

	uint32_t offset  = vx_core_id() * count;
	
	for (uint32_t i = 0; i < count; ++i) {
		dst_ptr[offset + i] = src_ptr[offset + i];
	}
}

#ifdef __cplusplus
}
#endif