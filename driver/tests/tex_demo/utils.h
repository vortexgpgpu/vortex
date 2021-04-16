#include <cstdint>
#include <vector>
#include <iostream>
#include "blitter.h"

int LoadTGA(const char *filename, 
            std::vector<uint8_t> &pixels, 
            uint32_t *width,
            uint32_t *height);

int SaveTGA(const char *filename, 
            const std::vector<uint8_t> &pixels, 
            uint32_t width,
            uint32_t height, 
            uint32_t bpp);

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