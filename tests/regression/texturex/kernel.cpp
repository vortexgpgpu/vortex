#include "common.h"
#include <vx_intrinsics.h>
#include <bitmanip.h>
#include <vx_spawn.h>
#include <vx_print.h>

using namespace graphics;

float4 texture2d(uint2 size, const unsigned char* texture, float2 texCoord) {
  int w = (int) (size.x * texCoord.x) % size.x;
  int h = size.y - ((int) (size.y * texCoord.y) % size.y) - 1;
  const unsigned char* color = texture + (h*size.x + w)*4;
  
  return (float4) {(float)*color / 255, (float)*(color+1) / 255, (float)*(color+2) / 255, (float)*(color+3) / 255};
}

#ifdef SKYBOX
uint32_t lod;
#endif

void kernel_body(int task_id, kernel_arg_t* __UNIFORM__ arg) {
	#ifndef SKYBOX
	auto size  = arg->size;
	auto image_ptr = reinterpret_cast<unsigned char*>(arg->image_addr);
	#else
	auto sampler = arg->sampler;
  	auto image = arg->image;
	#endif
	auto gl_Rasterization_ptr = reinterpret_cast<float4*>(arg->rasterization_addr);
	auto gl_FragColor_ptr = reinterpret_cast<float4*>(arg->fragColor_addr);

	// in out vars
	float4 texCoord = gl_Rasterization_ptr[task_id];
	// fragment operations
	#ifndef SKYBOX
	gl_FragColor_ptr[task_id] = texture2d(size, image_ptr, {texCoord.x, texCoord.y});
	#else
	gl_FragColor_ptr[task_id].x = vx_tex(0, texCoord.x, texCoord.y, lod);
	#endif
}

int main() {
	kernel_arg_t* arg = (kernel_arg_t*)KERNEL_ARG_DEV_MEM_ADDR;
	#ifdef SKYBOX
	{
		auto tex_logdim    = arg->sampler.read(0, VX_DCR_TEX_LOGDIM);
		auto tex_logwidth  = tex_logdim & 0xffff;
  		auto tex_logheight = tex_logdim >> 16;
		auto width_ratio   = float(1 << tex_logwidth) / WIDTH;
		auto height_ratio  = float(1 << tex_logheight) / HEIGHT;
		auto minification  = std::max(width_ratio, height_ratio);
		auto j             = static_cast<fixed16_t>(std::max(minification, 1.0f));
		lod           = std::min<uint32_t>(log2floor(j.data()) - 16, VX_TEX_LOD_MAX);
	}
	#endif
	vx_spawn_tasks(WIDTH*HEIGHT, (vx_spawn_tasks_cb)kernel_body, arg);
	return 0;
}
