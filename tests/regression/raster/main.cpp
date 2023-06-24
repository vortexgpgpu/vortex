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

const char* kernel_file = "kernel.bin";
const char* trace_file  = "triangle.cgltrace";
const char* output_file = "output.png";
const char* reference_file  = nullptr;

uint32_t clear_color = 0xff000000;

uint32_t dst_width  = 128;
uint32_t dst_height = 128;

uint32_t cbuf_stride;
uint32_t cbuf_pitch;
uint32_t cbuf_size;

vx_device_h device = nullptr;
vx_buffer_h staging_buf = nullptr;

uint64_t cbuf_addr    = 0;
uint64_t tilebuf_addr = 0;
uint64_t primbuf_addr = 0;

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
  if (staging_buf) {
    vx_buf_free(staging_buf);
  }
  if (device) {     
    if (cbuf_addr != 0) vx_mem_free(device, cbuf_addr);
    if (tilebuf_addr != 0) vx_mem_free(device, tilebuf_addr);
    if (primbuf_addr != 0) vx_mem_free(device, primbuf_addr);
    vx_dev_close(device);
  }
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
    if (tilebuf_addr != 0) vx_mem_free(device, tilebuf_addr); 
    if (primbuf_addr != 0) vx_mem_free(device, primbuf_addr); 
    RT_CHECK(vx_mem_alloc(device, tilebuf.size(), &tilebuf_addr));
    RT_CHECK(vx_mem_alloc(device, primbuf.size(), &primbuf_addr));
    std::cout << "tilebuf_addr=0x" << std::hex << tilebuf_addr << std::dec << std::endl;
    std::cout << "primbuf_addr=0x" << std::hex << primbuf_addr << std::dec << std::endl;

    uint32_t alloc_size = std::max({tilebuf.size(), primbuf.size(), sizeof(kernel_arg_t)});
    RT_CHECK(vx_buf_alloc(device, alloc_size, &staging_buf));
    
    // upload tiles buffer
    std::cout << "upload tile buffer" << std::endl;      
    {    
      auto buf_ptr = (uint8_t*)vx_host_ptr(staging_buf);
      memcpy(buf_ptr, tilebuf.data(), tilebuf.size());
      RT_CHECK(vx_copy_to_dev(staging_buf, tilebuf_addr, tilebuf.size(), 0));
    }

    // upload primitives buffer
    std::cout << "upload primitive buffer" << std::endl;      
    {    
      auto buf_ptr = (uint8_t*)vx_host_ptr(staging_buf);
      memcpy(buf_ptr, primbuf.data(), primbuf.size());
      RT_CHECK(vx_copy_to_dev(staging_buf, primbuf_addr, primbuf.size(), 0));
    }

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
    
      auto buf_ptr = (uint8_t*)vx_host_ptr(staging_buf);
      memcpy(buf_ptr, &kernel_arg, sizeof(kernel_arg_t));
      RT_CHECK(vx_copy_to_dev(staging_buf, KERNEL_ARG_DEV_MEM_ADDR, sizeof(kernel_arg_t), 0));
    }

    vx_buf_free(staging_buf);
    staging_buf = nullptr;

    uint32_t primbuf_stride = sizeof(graphics::rast_prim_t);

    // configure raster units
    vx_dcr_write(device, DCR_RASTER_TBUF_ADDR,   tilebuf_addr / 64);  // block address
    vx_dcr_write(device, DCR_RASTER_TILE_COUNT,  num_tiles);
    vx_dcr_write(device, DCR_RASTER_PBUF_ADDR,   primbuf_addr / 64);  // block address
    vx_dcr_write(device, DCR_RASTER_PBUF_STRIDE, primbuf_stride);    
    vx_dcr_write(device, DCR_RASTER_SCISSOR_X, (dst_width << 16) | 0);
    vx_dcr_write(device, DCR_RASTER_SCISSOR_Y, (dst_height << 16) | 0);

    auto time_start = std::chrono::high_resolution_clock::now();

    // start device
    std::cout << "start device" << std::endl;
    RT_CHECK(vx_start(device));

    // wait for completion
    std::cout << "wait for completion" << std::endl;
    RT_CHECK(vx_ready_wait(device, MAX_TIMEOUT));
    
    auto time_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
    printf("Elapsed time: %lg ms\n", elapsed);

  }

  // download destination buffer
  std::vector<uint8_t> dst_pixels(cbuf_size);
  {
    std::cout << "download destination buffer" << std::endl;
    RT_CHECK(vx_buf_alloc(device, cbuf_size, &staging_buf));
    RT_CHECK(vx_copy_from_dev(staging_buf, cbuf_addr, cbuf_size, 0));    
    auto buf_ptr = (uint8_t*)vx_host_ptr(staging_buf);
    memcpy(dst_pixels.data(), buf_ptr, cbuf_size);
    vx_buf_free(staging_buf);
    staging_buf = nullptr;
  }

  // save output image
  if (strcmp (output_file, "") != 0) {
    std::cout << "save output image" << std::endl;
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
  if (0 == (isa_flags & (VX_ISA_EXT_RASTER))) {
    std::cout << "RASTER extensions not supported!" << std::endl;
    cleanup();
    return -1;
  }

  uint64_t num_clusters, num_cores_per_cluster, num_warps, num_threads;
  RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_CLUSTERS, &num_clusters));
  RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_CORES, &num_cores_per_cluster));  
  RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_WARPS, &num_warps));
  RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_THREADS, &num_threads));
  auto num_cores = num_clusters * num_cores_per_cluster;

  uint32_t num_tasks = num_cores * num_warps * num_threads;

  std::cout << "number of tasks: " << std::dec << num_tasks << std::endl;

  CGLTrace trace;    
  RT_CHECK(trace.load(trace_file));

  // upload program
  std::cout << "upload program" << std::endl;  
  RT_CHECK(vx_upload_kernel_file(device, kernel_file));

  cbuf_stride = 4;
  cbuf_pitch  = dst_width * cbuf_stride;
  cbuf_size   = dst_width * cbuf_pitch;

  // allocate device memory  
  RT_CHECK(vx_mem_alloc(device, cbuf_size, &cbuf_addr));

  std::cout << "cbuf_addr=0x" << std::hex << cbuf_addr << std::dec << std::endl;

  // allocate staging buffer  
  {
    std::cout << "allocate staging buffer" << std::endl;    
    uint32_t alloc_size = cbuf_size;
    RT_CHECK(vx_buf_alloc(device, alloc_size, &staging_buf));
  }
  
  // clear destination buffer  
  {    
    std::cout << "clear destination buffer" << std::endl;      
    auto buf_ptr = (uint32_t*)vx_host_ptr(staging_buf);
    for (uint32_t i = 0; i < (cbuf_size/4); ++i) {
      buf_ptr[i] = clear_color;
    }    
    RT_CHECK(vx_copy_to_dev(staging_buf, cbuf_addr, cbuf_size, 0));  
  }  

  vx_buf_free(staging_buf);
  staging_buf = nullptr;

  // update kernel arguments
  kernel_arg.num_tasks     = num_tasks;
  kernel_arg.dst_width     = dst_width;
  kernel_arg.dst_height    = dst_height;

  kernel_arg.cbuf_stride   = cbuf_stride;
  kernel_arg.cbuf_pitch    = cbuf_pitch;    
  kernel_arg.cbuf_addr     = cbuf_addr;

  // run tests
  std::cout << "render" << std::endl;
  RT_CHECK(render(trace));

  // cleanup
  std::cout << "cleanup" << std::endl;  
  cleanup();  

  if (strcmp (output_file, "") != 0 && reference_file) {
    auto errors = CompareImages(output_file, reference_file, FORMAT_A8R8G8B8);
    if (0 == errors) {
      std::cout << "PASSED!" << std::endl;
    } else {
      std::cout << "FAILED! " << errors << " errors." << std::endl;
      return errors;
    }
  } 

  return 0;
}