#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_spawn.h>
#include "common.h"


float4 texture2d(uint2 size, const unsigned char* texture, float2 texCoord) {
  int w = (int) (size.x * texCoord.x) % size.x;
  int h = size.y - ((int) (size.y * texCoord.y) % size.y) - 1;
  const unsigned char* color = texture + (h*size.x + w)*4;
  
  return (float4) {(float)*color / 255, (float)*(color+1) / 255, (float)*(color+2) / 255, (float)*(color+3) / 255};
}

void kernel_body(int task_id, kernel_arg_t* __UNIFORM__ arg) {
	auto size  = arg->size;
	auto image_ptr = reinterpret_cast<unsigned char*>(arg->image_addr);
	auto gl_Rasterization_ptr = reinterpret_cast<float4*>(arg->rasterization_addr);
	auto gl_FragColor_ptr = reinterpret_cast<float4*>(arg->fragColor_addr);

	// in out vars
	float4 texCoord = gl_Rasterization_ptr[task_id];
	// fragment operations
	gl_FragColor_ptr[task_id] = texture2d(size, image_ptr, {texCoord.x, texCoord.y});
}

int main() {
	kernel_arg_t* arg = (kernel_arg_t*)KERNEL_ARG_DEV_MEM_ADDR;
	vx_spawn_tasks(WIDTH*HEIGHT, (vx_spawn_tasks_cb)kernel_body, arg);
	return 0;
}
