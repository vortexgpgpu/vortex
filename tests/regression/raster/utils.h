#include <cstdint>
#include <vector>
#include <bitmanip.h>
#include <cocogfx/include/format.hpp>
#include <cocogfx/include/blitter.hpp>
#include <cocogfx/include/cgltrace.hpp>

uint32_t toVXFormat(cocogfx::ePixelFormat format);

uint32_t toVXCompare(cocogfx::CGLTrace::ecompare compare);

uint32_t toVXStencilOp(cocogfx::CGLTrace::eStencilOp op);

uint32_t toVXBlendFunc(cocogfx::CGLTrace::eBlendOp op);

uint32_t Binning(std::vector<uint8_t>& tilebuf, 
                 std::vector<uint8_t>& primbuf,
                 const std::unordered_map<uint32_t, cocogfx::CGLTrace::vertex_t>& vertices,
                 const std::vector<cocogfx::CGLTrace::primitive_t>& primitives,                
                 uint32_t width,
                 uint32_t height,
                 float near,
                 float far,
                 uint32_t tileSize);

int LoadImage(const char *filename,
              cocogfx::ePixelFormat format,
              std::vector<uint8_t> &pixels,
              uint32_t *width,
              uint32_t *height);

int SaveImage(const char *filename,
              cocogfx::ePixelFormat format,
              const uint8_t* pixels, 
              uint32_t width,
              uint32_t height,
              int32_t pitch);

void dump_image(const std::vector<uint8_t>& pixels, 
                uint32_t width, 
                uint32_t height, 
                uint32_t bpp);

int CompareImages(const char* filename1, 
                  const char* filename2, 
                  cocogfx::ePixelFormat format);