#include <cstdint>
#include <vector>
#include <format.h>
#include <blitter.h>
#include <bitmanip.h>

int LoadImage(const char *filename,
              cocogfx::ePixelFormat format,
              std::vector<uint8_t> &pixels,
              uint32_t *width,
              uint32_t *height);

int SaveImage(const char *filename,
              cocogfx::ePixelFormat format,
              const std::vector<uint8_t> &pixels,
              uint32_t width,
              uint32_t height);

void dump_image(const std::vector<uint8_t>& pixels, 
                uint32_t width, 
                uint32_t height, 
                uint32_t bpp);