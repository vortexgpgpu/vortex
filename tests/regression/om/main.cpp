// Vortex2 KMU port of the skybox output-merger regression test.

#include <iostream>
#include <vector>
#include <unistd.h>
#include <string.h>
#include <chrono>
#include <vortex.h>
#include <VX_types.h>
#include "common.h"
#include <gfxutil.h>
#include <cocogfx/include/fixed.hpp>
#include <cocogfx/include/imageutil.hpp>

using namespace cocogfx;

#ifndef ASSETS_PATHS
#define ASSETS_PATHS ""
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
const char* output_file    = "output.png";
const char* reference_file = nullptr;

uint32_t color = 0xffffffff;
uint32_t depth = TFixed<24>(0.5f).data();

bool blend_enable = false;
bool depth_enable = false;
bool backface     = false;

uint32_t clear_color = 0x00000000;
uint32_t clear_depth = TFixed<24>(0.5f).data();

uint32_t dst_width  = 128;
uint32_t dst_height = 128;

uint32_t cbuf_pitch, cbuf_size;
uint32_t zbuf_pitch, zbuf_size;
uint64_t cbuf_addr, zbuf_addr;

vx_device_h device       = nullptr;
vx_buffer_h krnl_buffer  = nullptr;
vx_buffer_h args_buffer  = nullptr;
vx_buffer_h depth_buffer = nullptr;
vx_buffer_h color_buffer = nullptr;

static void show_usage() {
   std::cout << "Vortex Render Output Test (v2 KMU)." << std::endl;
   std::cout << "Usage: [-c color] [-d depth] [-b blend] [-f face] [-k kernel] [-o image] [-r reference] [-w width] [-h height]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "o:r:k:w:h:c:bdf?")) != -1) {
    switch (c) {
    case 'o': output_file = optarg; break;
    case 'r': reference_file = optarg; break;
    case 'k': kernel_file = optarg; break;
    case 'w': dst_width = std::atoi(optarg); break;
    case 'h': dst_height = std::atoi(optarg); break;
    case 'f': backface = true; break;
    case 'c': color = std::atoi(optarg); break;
    case 'd': depth_enable = true; break;
    case 'b': blend_enable = true; break;
    case '?': show_usage(); exit(0);
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
    if (depth_buffer) vx_mem_free(depth_buffer);
    if (color_buffer) vx_mem_free(color_buffer);
    if (krnl_buffer)  vx_mem_free(krnl_buffer);
    if (args_buffer)  vx_mem_free(args_buffer);
    vx_dev_close(device);
  }
}

int main(int argc, char *argv[]) {
  parse_args(argc, argv);

  RT_CHECK(vx_dev_open(&device));

  uint64_t isa_flags;
  RT_CHECK(vx_dev_caps(device, VX_CAPS_ISA_FLAGS, &isa_flags));
  if ((isa_flags & VX_ISA_EXT_OM) == 0) {
    std::cout << "om extension not enabled (build with -DEXT_OM_ENABLE)" << std::endl;
    cleanup(); return -1;
  }

  uint64_t num_threads;
  RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_THREADS, &num_threads));

  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));

  // ---- allocate cbuf + zbuf --------------------------------------------
  zbuf_pitch = dst_width * 4;
  zbuf_size  = dst_height * zbuf_pitch;
  cbuf_pitch = dst_width * 4;
  cbuf_size  = dst_height * cbuf_pitch;

  RT_CHECK(vx_mem_alloc(device, zbuf_size, VX_MEM_READ_WRITE, &depth_buffer));
  RT_CHECK(vx_mem_address(depth_buffer, &zbuf_addr));
  RT_CHECK(vx_mem_alloc(device, cbuf_size, VX_MEM_READ_WRITE, &color_buffer));
  RT_CHECK(vx_mem_address(color_buffer, &cbuf_addr));

  // depth checkerboard prefill (matches skybox test).
  {
    std::vector<uint32_t> staging(dst_width * dst_height);
    for (uint32_t y = 0; y < dst_height; ++y) {
      for (uint32_t x = 0; x < dst_width; ++x) {
        staging[x + y * dst_width] = ((x & 1) == (y & 1))
            ? TFixed<24>(0.0f).data()
            : TFixed<24>(0.99f).data();
      }
    }
    RT_CHECK(vx_copy_to_dev(depth_buffer, staging.data(), 0, zbuf_size));
  }
  // clear cbuf to clear_color.
  {
    std::vector<uint32_t> staging(cbuf_size / 4, clear_color);
    RT_CHECK(vx_copy_to_dev(color_buffer, staging.data(), 0, cbuf_size));
  }

  // ---- configure OM DCRs -----------------------------------------------
  RT_CHECK(vx_dcr_write(device, VX_DCR_OM_CBUF_ADDR,      cbuf_addr / 64));
  RT_CHECK(vx_dcr_write(device, VX_DCR_OM_CBUF_PITCH,     cbuf_pitch));
  RT_CHECK(vx_dcr_write(device, VX_DCR_OM_CBUF_WRITEMASK, 0xf));
  RT_CHECK(vx_dcr_write(device, VX_DCR_OM_ZBUF_ADDR,      zbuf_addr / 64));
  RT_CHECK(vx_dcr_write(device, VX_DCR_OM_ZBUF_PITCH,     zbuf_pitch));

  if (depth_enable) {
    RT_CHECK(vx_dcr_write(device, VX_DCR_OM_DEPTH_FUNC,      VX_OM_DEPTH_FUNC_LESS));
    RT_CHECK(vx_dcr_write(device, VX_DCR_OM_DEPTH_WRITEMASK, 1));
  } else {
    RT_CHECK(vx_dcr_write(device, VX_DCR_OM_DEPTH_FUNC,      VX_OM_DEPTH_FUNC_ALWAYS));
    RT_CHECK(vx_dcr_write(device, VX_DCR_OM_DEPTH_WRITEMASK, 0));
  }

  RT_CHECK(vx_dcr_write(device, VX_DCR_OM_STENCIL_FUNC,
                        (VX_OM_DEPTH_FUNC_ALWAYS << 16) | VX_OM_DEPTH_FUNC_ALWAYS));
  RT_CHECK(vx_dcr_write(device, VX_DCR_OM_STENCIL_ZPASS,
                        (VX_OM_STENCIL_OP_KEEP << 16) | VX_OM_STENCIL_OP_KEEP));
  RT_CHECK(vx_dcr_write(device, VX_DCR_OM_STENCIL_ZFAIL,
                        (VX_OM_STENCIL_OP_KEEP << 16) | VX_OM_STENCIL_OP_KEEP));
  RT_CHECK(vx_dcr_write(device, VX_DCR_OM_STENCIL_FAIL,
                        (VX_OM_STENCIL_OP_KEEP << 16) | VX_OM_STENCIL_OP_KEEP));
  RT_CHECK(vx_dcr_write(device, VX_DCR_OM_STENCIL_REF,        0));
  RT_CHECK(vx_dcr_write(device, VX_DCR_OM_STENCIL_MASK,       VX_OM_STENCIL_MASK));
  RT_CHECK(vx_dcr_write(device, VX_DCR_OM_STENCIL_WRITEMASK,  0));

  RT_CHECK(vx_dcr_write(device, VX_DCR_OM_BLEND_MODE,
                        (VX_OM_BLEND_MODE_ADD << 16) | VX_OM_BLEND_MODE_ADD));
  if (blend_enable) {
    RT_CHECK(vx_dcr_write(device, VX_DCR_OM_BLEND_FUNC,
                            (VX_OM_BLEND_FUNC_ONE_MINUS_SRC_A << 24)   // dst_a
                          | (VX_OM_BLEND_FUNC_ONE_MINUS_SRC_A << 16)   // dst_rgb
                          | (VX_OM_BLEND_FUNC_ONE             <<  8)   // src_a
                          |  VX_OM_BLEND_FUNC_ONE));                   // src_rgb
  } else {
    RT_CHECK(vx_dcr_write(device, VX_DCR_OM_BLEND_FUNC,
                            (VX_OM_BLEND_FUNC_ZERO << 24)
                          | (VX_OM_BLEND_FUNC_ZERO << 16)
                          | (VX_OM_BLEND_FUNC_ONE  <<  8)
                          |  VX_OM_BLEND_FUNC_ONE));
  }
  RT_CHECK(vx_dcr_write(device, VX_DCR_OM_BLEND_CONST, 0));
  RT_CHECK(vx_dcr_write(device, VX_DCR_OM_LOGIC_OP,    VX_OM_LOGIC_OP_COPY));

  // ---- pack kernel arg + launch ----------------------------------------
  kernel_arg_t kernel_arg = {};
  kernel_arg.dst_width    = dst_width;
  kernel_arg.dst_height   = dst_height;
  kernel_arg.color        = color;
  kernel_arg.depth        = depth;
  kernel_arg.backface     = backface ? 1 : 0;
  kernel_arg.blend_enable = blend_enable ? 1 : 0;
  uint32_t r = (color >> 16) & 0xff;
  uint32_t g = (color >>  8) & 0xff;
  uint32_t b = (color      ) & 0xff;
  kernel_arg.a_scale_q16 = (255u << 16) / dst_height;
  kernel_arg.r_scale_q16 = (r    << 16) / dst_width;
  kernel_arg.g_scale_q16 = (g    << 16) / dst_height;
  kernel_arg.b_scale_q16 = (b    << 16) / (dst_width + dst_height);

  RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg), &args_buffer));

  uint32_t block_x = std::min<uint32_t>((uint32_t)num_threads, dst_width);
  uint32_t block_y = 1;
  uint32_t grid_x  = (dst_width  + block_x - 1) / block_x;
  uint32_t grid_y  = (dst_height + block_y - 1) / block_y;
  uint32_t grid[2]  = { grid_x, grid_y };
  uint32_t block[2] = { block_x, block_y };

  auto t0 = std::chrono::high_resolution_clock::now();
  RT_CHECK(vx_start_g(device, krnl_buffer, args_buffer, 2, grid, block, 0));
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));
  auto t1 = std::chrono::high_resolution_clock::now();
  printf("Elapsed time: %.2f ms\n",
         std::chrono::duration<double, std::milli>(t1 - t0).count());

  // ---- save output PNG --------------------------------------------------
  // (skybox test does a vertical flip when saving — mirror that for ref-image diff)
  if (output_file && strcmp(output_file, "null") != 0) {
    std::vector<uint8_t> dst_pixels(cbuf_size);
    RT_CHECK(vx_copy_from_dev(dst_pixels.data(), color_buffer, 0, cbuf_size));
    auto bits = dst_pixels.data() + ((dst_height - 1) * cbuf_pitch);
    RT_CHECK(SaveImage(output_file, FORMAT_A8R8G8B8, bits, dst_width, dst_height,
                       -(int)cbuf_pitch));
  }

  cleanup();

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
