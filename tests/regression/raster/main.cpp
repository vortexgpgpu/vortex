#include <iostream>
#include <vector>
#include <unistd.h>
#include <cstring>
#include <chrono>
#include <cmath>
#include <array>
#include <assert.h>
#include <vortex2.h>
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
vx_queue_h  queue       = nullptr;
vx_module_h module_     = nullptr;
vx_kernel_h kernel      = nullptr;
vx_buffer_h color_buffer= nullptr;
vx_buffer_h tile_buffer = nullptr;
vx_buffer_h prim_buffer = nullptr;

bool use_sw = false;
uint64_t num_threads = 0;  // populated in main, read by render()

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
  if (color_buffer) vx_buffer_release(color_buffer);
  if (tile_buffer)  vx_buffer_release(tile_buffer);
  if (prim_buffer)  vx_buffer_release(prim_buffer);
  if (kernel)  vx_kernel_release(kernel);
  if (module_) vx_module_release(module_);
  if (queue)   vx_queue_release(queue);
  if (device)  vx_device_release(device);
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
    if (tile_buffer != nullptr) { vx_buffer_release(tile_buffer); tile_buffer = nullptr; }
    if (prim_buffer != nullptr) { vx_buffer_release(prim_buffer); prim_buffer = nullptr; }
    // tile_buffer / prim_buffer are bound to the raster unit (via
    // VX_DCR_RASTER_T/PBUF_ADDR) which bypasses the per-core MMU —
    // both need physical addresses.
    RT_CHECK(vx_buffer_create(device, tilebuf.size(), VX_MEM_READ | VX_MEM_PHYS, &tile_buffer));
    RT_CHECK(vx_buffer_address(tile_buffer, &tilebuf_addr));
    RT_CHECK(vx_buffer_create(device, primbuf.size(), VX_MEM_READ | VX_MEM_PHYS, &prim_buffer));
    RT_CHECK(vx_buffer_address(prim_buffer, &primbuf_addr));
    std::cout << "tile_buffer=0x" << std::hex << tilebuf_addr << std::dec << std::endl;
    std::cout << "prim_buffer=0x" << std::hex << primbuf_addr << std::dec << std::endl;

    // upload tiles buffer
    std::cout << "upload tile buffer" << std::endl;
    RT_CHECK(vx_enqueue_write(queue, tile_buffer, 0, tilebuf.data(), tilebuf.size(), 0, nullptr, nullptr));

    // upload primitives buffer
    std::cout << "upload primitive buffer" << std::endl;
    RT_CHECK(vx_enqueue_write(queue, prim_buffer, 0, primbuf.data(), primbuf.size(), 0, nullptr, nullptr));

    // prepare kernel argument
    std::cout << "prepare kernel argument" << std::endl;
    {
      kernel_arg.prim_addr   = primbuf_addr;
      kernel_arg.dst_width   = dst_width;
      kernel_arg.dst_height  = dst_height;
      kernel_arg.cbuf_addr   = cbuf_addr;
      kernel_arg.cbuf_stride = cbuf_stride;
      kernel_arg.cbuf_pitch  = cbuf_pitch;
    }

    uint32_t primbuf_stride = sizeof(graphics::rast_prim_t);

    // configure raster units
    vx_enqueue_dcr_write(queue, VX_DCR_RASTER_TBUF_ADDR, tilebuf_addr / 64, 0, nullptr, nullptr);  // block address
    vx_enqueue_dcr_write(queue, VX_DCR_RASTER_TILE_COUNT, num_tiles, 0, nullptr, nullptr);
    vx_enqueue_dcr_write(queue, VX_DCR_RASTER_PBUF_ADDR, primbuf_addr / 64, 0, nullptr, nullptr);  // block address
    vx_enqueue_dcr_write(queue, VX_DCR_RASTER_PBUF_STRIDE, primbuf_stride, 0, nullptr, nullptr);
    vx_enqueue_dcr_write(queue, VX_DCR_RASTER_SCISSOR_X, (dst_width << 16) | 0, 0, nullptr, nullptr);
    vx_enqueue_dcr_write(queue, VX_DCR_RASTER_SCISSOR_Y, (dst_height << 16) | 0, 0, nullptr, nullptr);

    auto time_start = std::chrono::high_resolution_clock::now();

    // start device — 1D launch, all threads in one block poll vx_rast()
    // until the cluster-shared raster unit drains its tile queue.
    vx_event_h launch_ev = nullptr;
    {
      uint32_t grid[1]  = { 1 };
      uint32_t block[1] = { (uint32_t)num_threads };
      std::cout << "start device (block=" << block[0] << ")" << std::endl;
      vx_launch_info_t li = {};
      li.struct_size  = sizeof(li);
      li.kernel       = kernel;
      li.args_host    = &kernel_arg;
      li.args_size    = sizeof(kernel_arg);
      li.ndim         = 1;
      li.grid_dim[0]  = grid[0];
      li.block_dim[0] = block[0];
      RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &launch_ev));
    }

    // wait for completion
    std::cout << "wait for completion" << std::endl;
    RT_CHECK(vx_event_wait_value(launch_ev, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(launch_ev);

    auto time_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
    printf("Elapsed time: %lg ms\n", elapsed);

  }

  // save output image
  if (strcmp(output_file, "null") != 0) {
    std::cout << "save output image" << std::endl;
    std::vector<uint8_t> dst_pixels(cbuf_size);
    {
      vx_event_h read_ev = nullptr;
      RT_CHECK(vx_enqueue_read(queue, dst_pixels.data(), color_buffer, 0, cbuf_size, 0, nullptr, &read_ev));
      RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
      vx_event_release(read_ev);
    }
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
  RT_CHECK(vx_device_open(0, &device));

  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));

  uint64_t isa_flags;
  RT_CHECK(vx_device_query(device, VX_CAPS_ISA_FLAGS, &isa_flags));
  bool has_ext = (isa_flags & VX_ISA_EXT_RASTER) != 0;
  if (!has_ext) {
    std::cout << "RASTER extensions not supported!" << std::endl;
    cleanup();
    return -1;
  }

  uint64_t num_cores, num_warps;
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_CORES, &num_cores));
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_WARPS, &num_warps));
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_THREADS, &num_threads));
  std::cout << "device: " << num_cores << " cores, " << num_warps
            << " warps, " << num_threads << " threads" << std::endl;

  CGLTrace trace;
  auto trace_file_s = graphics::ResolveFilePath(trace_file, ASSETS_PATHS);
  RT_CHECK(trace.load(trace_file_s.c_str()));

  // load kernel module
  std::cout << "load kernel module" << std::endl;
  RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
  RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

  cbuf_stride = 4;
  cbuf_pitch  = dst_width * cbuf_stride;
  cbuf_size   = dst_height * cbuf_pitch;

  // allocate device memory
  RT_CHECK(vx_buffer_create(device, cbuf_size, VX_MEM_WRITE, &color_buffer));
  RT_CHECK(vx_buffer_address(color_buffer, &cbuf_addr));
  std::cout << "color_buffer=0x" << std::hex << cbuf_addr << std::dec << std::endl;

  // clear destination buffer
  std::cout << "clear destination buffer" << std::endl;
  {
    std::vector<uint32_t> staging_buf(cbuf_size / 4, clear_color);
    vx_event_h ev = nullptr;
    RT_CHECK(vx_enqueue_write(queue, color_buffer, 0, staging_buf.data(), cbuf_size, 0, nullptr, &ev));
    RT_CHECK(vx_event_wait_value(ev, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(ev);
  }

  // update kernel arguments
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