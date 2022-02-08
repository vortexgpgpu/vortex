#include <iostream>
#include <vector>
#include <unistd.h>
#include <string.h>
#include <chrono>
#include <cmath>
#include <assert.h>
#include <vortex.h>
#include "common.h"
#include "utils.h"

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
const char* output_file = "output.png";
uint32_t dst_width = 64;
uint32_t dst_height = 64;
vx_device_h device = nullptr;
vx_buffer_h staging_buf = nullptr;
kernel_arg_t kernel_arg;

static void show_usage() {
   std::cout << "Vortex 3D Rendering Test." << std::endl;
   std::cout << "Usage: [-k: kernel] [-o image] [-u width] [-v height] [-h: help]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "k:o:u:v:h?")) != -1) {
    switch (c) {
    case 'o':
      output_file = optarg;
      break;
    case 'u':
      dst_width = std::atoi(optarg);
      break;
    case 'v':
      dst_height = std::atoi(optarg);
      break;
    case 'k':
      kernel_file = optarg;
      break;
    case 'h':
    case '?': {
      show_usage();
      exit(0);
    } break;
    default:
      show_usage();
      exit(-1);
    }
  }
}

void cleanup() {
  if (staging_buf) {
    vx_buf_free(staging_buf);
  }
  if (device) {
    vx_mem_free(device, kernel_arg.tiles_addr);
    vx_mem_free(device, kernel_arg.prims_addr);
    vx_mem_free(device, kernel_arg.dst_addr);
    vx_dev_close(device);
  }
}

int run_test(const kernel_arg_t& kernel_arg, 
             uint32_t buf_size, 
             uint32_t width, 
             uint32_t height,
             uint32_t bpp) {
  (void)bpp;
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

  // download destination buffer
  std::cout << "download destination buffer" << std::endl;
  RT_CHECK(vx_copy_from_dev(staging_buf, kernel_arg.dst_addr, buf_size, 0));

  std::vector<uint8_t> dst_pixels(buf_size);
  auto buf_ptr = (uint8_t*)vx_host_ptr(staging_buf);
  for (uint32_t i = 0; i < buf_size; ++i) {
    dst_pixels[i] = buf_ptr[i];
  } 

  // save output image
  std::cout << "save output image" << std::endl;  
  //dump_image(dst_pixels, width, height, bpp);  
  RT_CHECK(SaveImage(output_file, FORMAT_A8R8G8B8, dst_pixels, width, height));

  return 0;
}

void allocate_tiles() {
  // TODO
}

int main(int argc, char *argv[]) {  
  std::vector<tile_t> tiles;
  std::vector<prim_t> primitives;
  
  // parse command arguments
  parse_args(argc, argv);

  uint32_t dst_bpp     = 4;
  uint32_t dst_bufsize = dst_bpp * dst_width * dst_height;

  allocate_tiles();
  
  uint32_t tile_bufsize = tiles.size() * sizeof(tile_t);
  uint32_t prim_bufsize = primitives.size() * sizeof(prim_t);

  // open device connection
  std::cout << "open device connection" << std::endl;  
  RT_CHECK(vx_dev_open(&device));

  uint64_t isa_flags;
  RT_CHECK(vx_dev_caps(device, VX_CAPS_ISA_FLAGS, &isa_flags));
  if (0 == (isa_flags & (VX_ISA_EXT_RASTER | VX_ISA_EXT_ROP))) {
    std::cout << "raster or rop extensions not supported!" << std::endl;
    return -1;
  }

  uint64_t max_cores, max_warps, max_threads;
  RT_CHECK(vx_dev_caps(device, VX_CAPS_MAX_CORES, &max_cores));
  RT_CHECK(vx_dev_caps(device, VX_CAPS_MAX_WARPS, &max_warps));
  RT_CHECK(vx_dev_caps(device, VX_CAPS_MAX_THREADS, &max_threads));

  uint32_t num_tasks = max_cores * max_warps * max_threads;

  std::cout << "number of tasks: " << std::dec << num_tasks << std::endl;
  std::cout << "destination staging_buf: width=" << dst_width << ", heigth=" << dst_height << ", size=" << dst_bufsize << " bytes" << std::endl;

  // upload program
  std::cout << "upload program" << std::endl;  
  RT_CHECK(vx_upload_kernel_file(device, kernel_file));

  // allocate device memory
  std::cout << "allocate device memory" << std::endl;  
  uint64_t tile_addr, prim_addr, dst_addr;
  RT_CHECK(vx_mem_alloc(device, tile_bufsize, &tile_addr));
  RT_CHECK(vx_mem_alloc(device, prim_bufsize, &prim_addr));
  RT_CHECK(vx_mem_alloc(device, dst_bufsize, &dst_addr));

  std::cout << "tile_addr=0x" << std::hex << tile_addr << std::endl;
  std::cout << "prim_addr=0x" << std::hex << prim_addr << std::endl;
  std::cout << "dst_addr=0x" << std::hex << dst_addr << std::endl;

  // allocate staging shared memory  
  std::cout << "allocate shared memory" << std::endl;    
  uint32_t alloc_size = std::max<uint32_t>(sizeof(kernel_arg_t), 
                            std::max<uint32_t>(tile_bufsize,
                              std::max<uint32_t>(prim_bufsize, dst_bufsize)));
  RT_CHECK(vx_buf_alloc(device, alloc_size, &staging_buf));
  
  // upload kernel argument
  std::cout << "upload kernel argument" << std::endl;
  {
    kernel_arg.dst_width  = dst_width;
    kernel_arg.dst_height = dst_height;
    kernel_arg.dst_stride = dst_bpp;
    kernel_arg.dst_pitch  = dst_bpp * dst_width;    
    kernel_arg.dst_addr   = dst_addr;

    auto buf_ptr = (uint8_t*)vx_host_ptr(staging_buf);
    memcpy(buf_ptr, &kernel_arg, sizeof(kernel_arg_t));
    RT_CHECK(vx_copy_to_dev(staging_buf, KERNEL_ARG_DEV_MEM_ADDR, sizeof(kernel_arg_t), 0));
  }

  // upload tiles buffer
  std::cout << "upload tiles buffer" << std::endl;      
  {    
    auto buf_ptr = (tile_t*)vx_host_ptr(staging_buf);
    for (uint32_t i = 0; i < tiles.size(); ++i) {
      buf_ptr[i] = tiles.at(i);
    }      
    RT_CHECK(vx_copy_to_dev(staging_buf, kernel_arg.tiles_addr, tile_bufsize, 0));
  }

  // upload primitives buffer
  std::cout << "upload primitives buffer" << std::endl;      
  {    
    auto buf_ptr = (prim_t*)vx_host_ptr(staging_buf);
    for (uint32_t i = 0; i < primitives.size(); ++i) {
      buf_ptr[i] = primitives.at(i);
    }      
    RT_CHECK(vx_copy_to_dev(staging_buf, kernel_arg.prims_addr, prim_bufsize, 0));
  }

  // clear destination buffer
  std::cout << "clear destination buffer" << std::endl;      
  {    
    auto buf_ptr = (uint32_t*)vx_host_ptr(staging_buf);
    for (uint32_t i = 0; i < (dst_bufsize/4); ++i) {
      buf_ptr[i] = 0xdeadbeef;
    }    
    RT_CHECK(vx_copy_to_dev(staging_buf, kernel_arg.dst_addr, dst_bufsize, 0));  
  }

  // run tests
  std::cout << "run tests" << std::endl;
  RT_CHECK(run_test(kernel_arg, dst_bufsize, dst_width, dst_height, dst_bpp));

  // cleanup
  std::cout << "cleanup" << std::endl;  
  cleanup();

  std::cout << "PASSED!" << std::endl;

  return 0;
}