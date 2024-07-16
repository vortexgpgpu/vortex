#include <iostream>
#include <vector>
#include <unistd.h>
#include <cstring>
#include <chrono>
#include <cmath>
#include <array>
#include <assert.h>
#include <vortex.h>
#include <graphics.h>
#include <gfxutil.h>
#include <VX_config.h>
#include <algorithm>
#include "common.h"
#include <cocogfx/include/imageutil.hpp>

using namespace cocogfx;

#ifndef ASSETS_PATHS
#define ASSETS_PATHS ""
#endif

#define RT_CHECK(_expr)                                         \
   do {                                                         \
     int _ret = _expr;                                          \
     if (0 == _ret)                                             \
       break;                                                   \
     printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);   \
	 cleanup();			                                              \
     exit(-1);                                                  \
   } while (false)

///////////////////////////////////////////////////////////////////////////////

const char* kernel_file = "kernel.vxbin";
const char* trace_file  = "triangle.cgltrace";
const char* output_file = "output.png";
const char* reference_file  = nullptr;

uint32_t clear_color = 0xff000000;

uint32_t dst_width  = 128;
uint32_t dst_height = 128;

uint32_t cbuf_stride;
uint32_t cbuf_pitch;
uint32_t cbuf_size;

uint64_t cbuf_addr;
uint64_t tilebuf_addr;
uint64_t primbuf_addr;

vx_device_h device      = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
vx_buffer_h color_buffer= nullptr;
vx_buffer_h tile_buffer = nullptr;
vx_buffer_h prim_buffer = nullptr;

bool use_sw = false;

kernel_arg_t kernel_arg = {};

uint32_t tileLogSize = RASTER_TILE_LOGSIZE;

static void show_usage() {
   std::cout << "Vortex rasterizer Test." << std::endl;
   std::cout << "Usage: [-t trace] [-o output] [-r reference] [-w width] [-h height] [-z no_hw] [-k tilelogsize]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "t:i:o:r:w:h:t:k:z?")) != -1) {
    switch (c) {
    case 't':
      trace_file = optarg;
      break;
    case 'o':
      output_file = optarg;
      break;
    case 'r':
      reference_file = optarg;
      break;
    case 'w':
      dst_width = std::atoi(optarg);
      break;
    case 'h':
      dst_height = std::atoi(optarg);
      break;
    case 'k':
      tileLogSize = std::atoi(optarg);
      break;
    case 'z':
      use_sw = true;
      break;
    case '?': {
      show_usage();
      exit(0);
    } break;
    default:
      show_usage();
      exit(-1);
    }
  }
  if (strcmp (output_file, "null") == 0 && reference_file) {
    std::cout << "Error: the output file is missing for reference validation!" << std::endl;
    exit(1);
  }
}

void cleanup() {
  vx_mem_free(color_buffer);
  vx_mem_free(tile_buffer);
  vx_mem_free(prim_buffer);
  vx_mem_free(krnl_buffer);
  vx_mem_free(args_buffer);
  vx_dev_close(device);
}

int render(const CGLTrace& trace) {
  // render each draw call
  for (auto& drawcall : trace.drawcalls) {
    std::vector<uint8_t> tilebuf;
    std::vector<uint8_t> primbuf;

    // Perform tile binning
    auto num_tiles = graphics::Binning(tilebuf, primbuf, drawcall.vertices, drawcall.primitives, dst_width, dst_height, drawcall.viewport.near, drawcall.viewport.far, tileLogSize);
    std::cout << "Binning allocated " << std::dec << num_tiles << " tiles with " << primbuf.size() << " total primitives." << std::endl;
    if (0 == num_tiles)
      continue;

    // allocate tile memory
    if (tile_buffer != nullptr) vx_mem_free(tile_buffer);
    if (prim_buffer != nullptr) vx_mem_free(prim_buffer);
    RT_CHECK(vx_mem_alloc(device, tilebuf.size(), VX_MEM_READ, &tile_buffer));
    RT_CHECK(vx_mem_address(tile_buffer, &tilebuf_addr));
    RT_CHECK(vx_mem_alloc(device, primbuf.size(), VX_MEM_READ, &prim_buffer));
    RT_CHECK(vx_mem_address(prim_buffer, &primbuf_addr));
    std::cout << "tile_buffer=0x" << std::hex << tilebuf_addr << std::dec << std::endl;
    std::cout << "prim_buffer=0x" << std::hex << primbuf_addr << std::dec << std::endl;

    // upload tiles buffer
    std::cout << "upload tile buffer" << std::endl;
    RT_CHECK(vx_copy_to_dev(tile_buffer, tilebuf.data(), 0, tilebuf.size()));

    // upload primitives buffer
    std::cout << "upload primitive buffer" << std::endl;
    RT_CHECK(vx_copy_to_dev(prim_buffer, primbuf.data(), 0, primbuf.size()));

    // upload kernel argument
    std::cout << "upload kernel argument" << std::endl;
    {
      kernel_arg.use_sw      = use_sw;
      kernel_arg.prim_addr   = primbuf_addr;
      kernel_arg.dst_width   = dst_width;
      kernel_arg.dst_height  = dst_height;
      kernel_arg.cbuf_addr   = cbuf_addr;
      kernel_arg.cbuf_stride = cbuf_stride;
      kernel_arg.cbuf_pitch  = cbuf_pitch;

      RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg_t), &args_buffer));
    }

    uint32_t primbuf_stride = sizeof(graphics::rast_prim_t);

    // configure raster units
    vx_dcr_write(device, VX_DCR_RASTER_TBUF_ADDR, tilebuf_addr / 64);  // block address
    vx_dcr_write(device, VX_DCR_RASTER_TILE_COUNT, num_tiles);
    vx_dcr_write(device, VX_DCR_RASTER_PBUF_ADDR, primbuf_addr / 64);  // block address
    vx_dcr_write(device, VX_DCR_RASTER_PBUF_STRIDE, primbuf_stride);
    vx_dcr_write(device, VX_DCR_RASTER_SCISSOR_X, (dst_width << 16) | 0);
    vx_dcr_write(device, VX_DCR_RASTER_SCISSOR_Y, (dst_height << 16) | 0);

    auto time_start = std::chrono::high_resolution_clock::now();

    // start device
    std::cout << "start device" << std::endl;
    RT_CHECK(vx_start(device, krnl_buffer, args_buffer));

    // wait for completion
    std::cout << "wait for completion" << std::endl;
    RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

    auto time_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
    printf("Elapsed time: %lg ms\n", elapsed);

  }

  // save output image
  if (strcmp(output_file, "null") != 0) {
    std::cout << "save output image" << std::endl;
    std::vector<uint8_t> dst_pixels(cbuf_size);
    RT_CHECK(vx_copy_from_dev(dst_pixels.data(), color_buffer, 0, cbuf_size));
    //DumpImage(dst_pixels, dst_width, dst_height, 4);
    auto bits = dst_pixels.data() + (dst_height-1) * cbuf_pitch;
    RT_CHECK(SaveImage(output_file, FORMAT_A8R8G8B8, bits, dst_width, dst_height, -cbuf_pitch));
  }

  return 0;
}

int main(int argc, char *argv[]) {
  // parse command arguments
  parse_args(argc, argv);

  // open device connection
  std::cout << "open device connection" << std::endl;
  RT_CHECK(vx_dev_open(&device));

  uint64_t isa_flags;
  RT_CHECK(vx_dev_caps(device, VX_CAPS_ISA_FLAGS, &isa_flags));
  bool has_ext = (isa_flags & VX_ISA_EXT_RASTER) != 0;
  if (!has_ext) {
    std::cout << "RASTER extensions not supported!" << std::endl;
    cleanup();
    return -1;
  }

  uint64_t num_cores, num_warps, num_threads;
  RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_CORES, &num_cores));
  RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_WARPS, &num_warps));
  RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_THREADS, &num_threads));

  uint32_t num_tasks = num_cores * num_warps * num_threads;

  std::cout << "number of tasks: " << std::dec << num_tasks << std::endl;

  CGLTrace trace;
  auto trace_file_s = graphics::ResolveFilePath(trace_file, ASSETS_PATHS);
  RT_CHECK(trace.load(trace_file_s.c_str()));

  // upload program
  std::cout << "upload program" << std::endl;
  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));

  cbuf_stride = 4;
  cbuf_pitch  = dst_width * cbuf_stride;
  cbuf_size   = dst_height * cbuf_pitch;

  // allocate device memory
  RT_CHECK(vx_mem_alloc(device, cbuf_size, VX_MEM_WRITE, &color_buffer));
  RT_CHECK(vx_mem_address(color_buffer, &cbuf_addr));
  std::cout << "color_buffer=0x" << std::hex << cbuf_addr << std::dec << std::endl;

  // clear destination buffer
  std::cout << "clear destination buffer" << std::endl;
  {
    std::vector<uint32_t> staging_buf(cbuf_size / 4, clear_color);
    RT_CHECK(vx_copy_to_dev(color_buffer, staging_buf.data(), 0, cbuf_size));
  }

  // update kernel arguments
  kernel_arg.num_tasks  = num_tasks;
  kernel_arg.dst_width  = dst_width;
  kernel_arg.dst_height = dst_height;

  kernel_arg.cbuf_stride= cbuf_stride;
  kernel_arg.cbuf_pitch = cbuf_pitch;
  kernel_arg.cbuf_addr  = cbuf_addr;

  // run tests
  std::cout << "render" << std::endl;
  RT_CHECK(render(trace));

  // cleanup
  std::cout << "cleanup" << std::endl;
  cleanup();

  if (strcmp (output_file, "") != 0 && reference_file) {
    auto reference_file_s = graphics::ResolveFilePath(reference_file, ASSETS_PATHS);
    auto errors = CompareImages(output_file, reference_file_s.c_str(), FORMAT_A8R8G8B8);
    if (0 == errors) {
      std::cout << "PASSED!" << std::endl;
    } else {
      std::cout << "FAILED! " << errors << " errors." << std::endl;
      return errors;
    }
  }

  return 0;
}