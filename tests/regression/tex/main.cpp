// Vortex2 KMU port of the skybox texture regression test:
//
// Host loads a PNG, generates a mipmap chain, configures TEX stage 0
// DCRs, then launches a 2D grid where each thread samples one output
// pixel via vx_tex(stage, u, v, lod). Output is saved to a PNG and
// optionally diffed against a reference image.
//
// Differences from the skybox v1 version:
//   - Drops the on-device software TextureSampler fallback (use_sw)
//     and the on-device DCRS shadow — those required the vortex v1
//     `int main()` + `vx_spawn_threads()` API. The v2 KMU launch only
//     dispatches `__kernel void kernel_main(...)` per thread.
//   - Per-pixel delta is precomputed on host and passed in kernel_arg.

#include <iostream>
#include <vector>
#include <unistd.h>
#include <string.h>
#include <chrono>
#include <cmath>
#include <assert.h>
#include <vortex.h>
#include <VX_types.h>
#include "common.h"
#include <bitmanip.h>
#include <gfxutil.h>
#include <cocogfx/include/blitter.hpp>
#include <cocogfx/include/imageutil.hpp>

using namespace cocogfx;
using namespace vortex;

#ifndef ASSETS_PATHS
#define ASSETS_PATHS ""
#endif

#ifndef VX_DCR_TEX_MIPOFF
#define VX_DCR_TEX_MIPOFF(lod) (VX_DCR_TEX_MIPOFF_BASE + (lod))
#endif

#define RT_CHECK(_expr)                                         \
   do {                                                         \
     int _ret = _expr;                                          \
     if (0 == _ret) break;                                      \
     printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);   \
     cleanup();                                                 \
     exit(-1);                                                  \
   } while (false)

const char* kernel_file    = "kernel.vxbin";
const char* input_file     = "palette64.png";
const char* output_file    = "output.png";
const char* reference_file = nullptr;
int         wrap           = VX_TEX_WRAP_CLAMP;
int         filter         = VX_TEX_FILTER_POINT;
float       scale          = 1.0f;
int         format         = VX_TEX_FORMAT_A8R8G8B8;
ePixelFormat eformat       = FORMAT_A8R8G8B8;

vx_device_h device      = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
vx_buffer_h dst_buffer  = nullptr;
vx_buffer_h src_buffer  = nullptr;

static void show_usage() {
   std::cout << "Vortex Texture Test (v2 KMU)." << std::endl;
   std::cout << "Usage: [-k: kernel] [-i image] [-o image] [-r reference] [-s scale] [-w wrap] [-f format] [-g filter] [-h: help]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "i:o:k:w:f:g:s:r:h?")) != -1) {
    switch (c) {
    case 'i': input_file = optarg; break;
    case 'o': output_file = optarg; break;
    case 'k': kernel_file = optarg; break;
    case 'w': wrap = atoi(optarg); break;
    case 'f': format = atoi(optarg); break;
    case 'g': filter = atoi(optarg); break;
    case 's': scale = atof(optarg); break;
    case 'r': reference_file = optarg; break;
    case 'h': case '?': show_usage(); exit(0);
    default:  show_usage(); exit(-1);
    }
  }
  if (output_file && strcmp(output_file, "null") == 0 && reference_file) {
    std::cout << "Error: output file is missing for reference validation" << std::endl;
    exit(1);
  }
}

void cleanup() {
  if (device) {
    if (src_buffer)  vx_mem_free(src_buffer);
    if (dst_buffer)  vx_mem_free(dst_buffer);
    if (krnl_buffer) vx_mem_free(krnl_buffer);
    if (args_buffer) vx_mem_free(args_buffer);
    vx_dev_close(device);
  }
}

int main(int argc, char *argv[]) {
  uint64_t src_addr;
  uint64_t dst_addr;
  std::vector<uint8_t> src_pixels;
  std::vector<uint32_t> mip_offsets;
  uint32_t src_width  = 0;
  uint32_t src_height = 0;

  parse_args(argc, argv);

  // ---- load PNG + generate mipmap chain --------------------------------
  {
    std::vector<uint8_t> staging;
    auto input_file_s = graphics::ResolveFilePath(input_file, ASSETS_PATHS);
    RT_CHECK(LoadImage(input_file_s.c_str(), eformat, staging, &src_width, &src_height));
    if (!ispow2(src_width) || !ispow2(src_height)) {
      std::cout << "Error: only power-of-two textures supported (got "
                << src_width << "x" << src_height << ")" << std::endl;
      return -1;
    }
    uint32_t src_bpp   = Format::GetInfo(eformat).BytePerPixel;
    uint32_t src_pitch = src_width * src_bpp;
    RT_CHECK(GenerateMipmaps(src_pixels, mip_offsets, staging.data(), eformat,
                             src_width, src_height, src_pitch));
  }

  uint32_t src_logwidth  = log2ceil(src_width);
  uint32_t src_logheight = log2ceil(src_height);
  uint32_t src_bufsize   = src_pixels.size();

  uint32_t dst_width   = (uint32_t)(src_width * scale);
  uint32_t dst_height  = (uint32_t)(src_height * scale);
  uint32_t dst_bpp     = 4;
  uint32_t dst_pitch   = dst_bpp * dst_width;
  uint32_t dst_bufsize = dst_pitch * dst_height;

  // ---- open device + sanity check --------------------------------------
  RT_CHECK(vx_dev_open(&device));

  uint64_t isa_flags;
  RT_CHECK(vx_dev_caps(device, VX_CAPS_ISA_FLAGS, &isa_flags));
  if ((isa_flags & VX_ISA_EXT_TEX) == 0) {
    std::cout << "tex extension not enabled (build with -DEXT_TEX_ENABLE)" << std::endl;
    cleanup();
    return -1;
  }

  uint64_t num_threads;
  RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_THREADS, &num_threads));

  std::cout << "src: " << src_width << "x" << src_height << " (" << src_bufsize << " bytes incl. mipmaps)" << std::endl;
  std::cout << "dst: " << dst_width << "x" << dst_height << " (" << dst_bufsize << " bytes)" << std::endl;

  // ---- upload kernel binary --------------------------------------------
  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));

  // ---- allocate device buffers -----------------------------------------
  RT_CHECK(vx_mem_alloc(device, src_bufsize, VX_MEM_READ, &src_buffer));
  RT_CHECK(vx_mem_address(src_buffer, &src_addr));
  RT_CHECK(vx_mem_alloc(device, dst_bufsize, VX_MEM_WRITE, &dst_buffer));
  RT_CHECK(vx_mem_address(dst_buffer, &dst_addr));
  RT_CHECK(vx_copy_to_dev(src_buffer, src_pixels.data(), 0, src_bufsize));

  // ---- pre-compute LOD + per-pixel delta -------------------------------
  // Match skybox kernel's lod selection: minification = max(width_ratio, height_ratio)
  // expressed as 16.16 fixed-point.
  uint64_t width_ratio_q16  = ((uint64_t)src_width  << 16) / dst_width;
  uint64_t height_ratio_q16 = ((uint64_t)src_height << 16) / dst_height;
  uint64_t minif_q16 = std::max(width_ratio_q16, height_ratio_q16);
  if (minif_q16 < (1ull << 16)) minif_q16 = 1ull << 16;
  // log2floor(minif_q16) - 16 = integer log2 of the minification ratio.
  int lod = 0;
  for (uint64_t v = minif_q16; v > (1ull << 16); v >>= 1) ++lod;
  if (lod > VX_TEX_LOD_MAX) lod = VX_TEX_LOD_MAX;
  uint32_t frac_q8 = (uint32_t)((minif_q16 - ((uint64_t)1 << (lod + 16))) >> (lod + 16 - 8));

  uint32_t deltaX = ((uint32_t)1 << VX_TEX_FXD_FRAC) / dst_width;
  uint32_t deltaY = ((uint32_t)1 << VX_TEX_FXD_FRAC) / dst_height;

  // ---- configure TEX DCRs ----------------------------------------------
  RT_CHECK(vx_dcr_write(device, VX_DCR_TEX_STAGE,  0));
  RT_CHECK(vx_dcr_write(device, VX_DCR_TEX_LOGDIM, (src_logheight << 16) | src_logwidth));
  RT_CHECK(vx_dcr_write(device, VX_DCR_TEX_FORMAT, format));
  RT_CHECK(vx_dcr_write(device, VX_DCR_TEX_WRAP,   (wrap << 16) | wrap));
  RT_CHECK(vx_dcr_write(device, VX_DCR_TEX_FILTER, (filter == VX_TEX_FILTER_BILINEAR) ? VX_TEX_FILTER_BILINEAR : VX_TEX_FILTER_POINT));
  RT_CHECK(vx_dcr_write(device, VX_DCR_TEX_ADDR,   src_addr / 64));
  for (uint32_t i = 0; i < mip_offsets.size() && i < (uint32_t)VX_TEX_LOD_MAX; ++i) {
    RT_CHECK(vx_dcr_write(device, VX_DCR_TEX_MIPOFF(i), mip_offsets[i]));
  }

  // ---- pack kernel arg + launch ----------------------------------------
  kernel_arg_t kernel_arg = {};
  kernel_arg.dst_addr      = dst_addr;
  kernel_arg.dst_width     = dst_width;
  kernel_arg.dst_height    = dst_height;
  kernel_arg.dst_pitch     = dst_pitch;
  kernel_arg.dst_stride    = (uint8_t)dst_bpp;
  kernel_arg.filter        = (uint8_t)filter;
  kernel_arg.use_trilinear = (filter == 2) ? 1 : 0;
  kernel_arg.deltaX        = deltaX;
  kernel_arg.deltaY        = deltaY;
  kernel_arg.lod           = (uint32_t)lod;
  kernel_arg.frac          = frac_q8;

  RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg), &args_buffer));

  // 2D launch: gx ranges [0, dst_width), gy ranges [0, dst_height).
  uint32_t block_x = std::min<uint32_t>((uint32_t)num_threads, dst_width);
  uint32_t block_y = 1;
  uint32_t grid_x  = (dst_width  + block_x - 1) / block_x;
  uint32_t grid_y  = (dst_height + block_y - 1) / block_y;
  uint32_t grid[2]  = { grid_x, grid_y };
  uint32_t block[2] = { block_x, block_y };
  std::cout << "launch grid=" << grid_x << "x" << grid_y
            << ", block=" << block_x << "x" << block_y
            << ", lod=" << lod << std::endl;

  auto t0 = std::chrono::high_resolution_clock::now();
  RT_CHECK(vx_start_g(device, krnl_buffer, args_buffer, 2, grid, block, 0));
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));
  auto t1 = std::chrono::high_resolution_clock::now();
  printf("Elapsed time: %.2f ms\n",
         std::chrono::duration<double, std::milli>(t1 - t0).count());

  // ---- save output PNG --------------------------------------------------
  if (output_file && strcmp(output_file, "null") != 0) {
    std::vector<uint8_t> dst_pixels(dst_bufsize);
    RT_CHECK(vx_copy_from_dev(dst_pixels.data(), dst_buffer, 0, dst_bufsize));
    RT_CHECK(SaveImage(output_file, FORMAT_A8R8G8B8, dst_pixels.data(),
                       dst_width, dst_height, dst_pitch));
  }

  cleanup();

  // ---- compare to reference --------------------------------------------
  if (reference_file) {
    auto reference_file_s = graphics::ResolveFilePath(reference_file, ASSETS_PATHS);
    auto errors = CompareImages(output_file, reference_file_s.c_str(), FORMAT_A8R8G8B8);
    if (0 == errors) {
      std::cout << "PASSED!" << std::endl;
    } else {
      std::cout << "FAILED: " << errors << " errors against reference" << std::endl;
      return errors;
    }
  } else {
    std::cout << "PASSED!" << std::endl;
  }

  return 0;
}
