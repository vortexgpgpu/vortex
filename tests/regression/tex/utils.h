#include <cstdint>
#include <vector>
#include <iostream>
#include "surfacedesc.h"

#define ISPOW2(x)   (((x) != 0) && (0 == ((x) & ((x) - 1))))

inline uint32_t ilog2 (uint32_t value) {
  	return (uint32_t)(sizeof(uint32_t) * 8UL) - (uint32_t)__builtin_clzl((value << 1) - 1UL) - 1;
}

int LoadImage(const char *filename,
              ePixelFormat format,
              std::vector<uint8_t> &pixels,
              uint32_t *width,
              uint32_t *height);

int SaveImage(const char *filename,
              ePixelFormat format,
              const std::vector<uint8_t> &pixels,
              uint32_t width,
              uint32_t height);

int CopyBuffers(SurfaceDesc &dstDesc, 
                int32_t dstOffsetX,
                int32_t dstOffsetY, 
                uint32_t copyWidth, 
                uint32_t copyHeight,
                const SurfaceDesc &srcDesc, 
                int32_t srcOffsetX,
                int32_t srcOffsetY);

int ConvertImage(std::vector<uint8_t>& dst_pixels,
                 const std::vector<uint8_t>& src_pixels,
                 uint32_t width,
                 uint32_t height,
                 ePixelFormat src_format,
                 ePixelFormat dst_format);

void dump_image(const std::vector<uint8_t>& pixels, 
                uint32_t width, 
                uint32_t height, 
                uint32_t bpp);
