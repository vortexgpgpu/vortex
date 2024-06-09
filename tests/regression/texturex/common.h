#ifndef _COMMON_H_
#define _COMMON_H_

#define KERNEL_ARG_DEV_MEM_ADDR 0x7ffff000

#define WIDTH 700
#define HEIGHT 700

#ifndef TYPE
#define TYPE float
#endif

struct float4 { float x,y,z,w; };
struct uint2 { uint32_t x,y; };
struct float2 { float x,y; };

typedef struct {
  #ifndef SKYBOX
  uint2 size;
  uint64_t image_addr;
  #else 
  uint64_t image
  graphics::TexDCRS sampler;
  #endif
  uint64_t fragCoord_addr;
  uint64_t rasterization_addr;  
  uint64_t discard_addr;  
  uint64_t fragColor_addr;  
} kernel_arg_t;

#endif
