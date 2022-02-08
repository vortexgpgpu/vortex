#include <cstdint>
#include <vector>
#include <bitmanip.h>
#include <cocogfx/include/format.h>
#include <cocogfx/include/blitter.h>

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