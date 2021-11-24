#include <cstdint>
#include <vector>
#include <iostream>
#include <bitmanip.h>
#include "surfacedesc.h"

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

int GenerateMipmaps(std::vector<uint8_t>& dst_pixels,
                    std::vector<uint32_t>& mip_offsets,
                    const std::vector<uint8_t>& src_pixels,
                    ePixelFormat format,
                    uint32_t src_width,
                    uint32_t src_height);

void dump_image(const std::vector<uint8_t>& pixels, 
                uint32_t width, 
                uint32_t height, 
                uint32_t bpp);