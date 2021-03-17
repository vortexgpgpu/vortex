#include <iostream>
#include <vector>
#include <unistd.h>
#include <string.h>
#include <assert.h>
#include <vortex.h>
#include "common.h"
#include "utils.h"

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
const char* input_file  = "sample.tga";
const char* output_file = "output.tga";
float scale = 1.0f;

vx_device_h device = nullptr;
vx_buffer_h buffer = nullptr;

static void show_usage() {
   std::cout << "Vortex Texture Test." << std::endl;
   std::cout << "Usage: [-k: kernel] [-i image] [-o image] [-s scale] [-h: help]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "i:o:k:h?")) != -1) {
    switch (c) {
    case 'i':
       input_file = optarg;
      break;
    case 'o':
       output_file = optarg;
      break;
    case 's':
      scale = std::stof(optarg, NULL);
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
  if (buffer) {
    vx_buf_release(buffer);
  }
  if (device) {
    vx_dev_close(device);
  }
}

int run_test(const kernel_arg_t& kernel_arg, uint32_t buf_size, uint32_t width, uint32_t height, uint32_t dst_bpp) {
  // start device
  std::cout << "start device" << std::endl;
  RT_CHECK(vx_start(device));

  // wait for completion
  std::cout << "wait for completion" << std::endl;
  RT_CHECK(vx_ready_wait(device, -1));

  // download destination buffer
  std::cout << "download destination buffer" << std::endl;
  RT_CHECK(vx_copy_from_dev(buffer, kernel_arg.dst_ptr, buf_size, 0));

  std::vector<uint8_t> dst_pixels(buf_size);
  auto buf_ptr = (int8_t*)vx_host_ptr(buffer);
  for (uint32_t i = 0; i < buf_size; ++i) {
    dst_pixels[i] = buf_ptr[i];
  } 

  // save output image
  std::cout << "save output image" << std::endl;  
  RT_CHECK(SaveTGA(output_file, dst_pixels, width, height, dst_bpp));

  return 0;
}

int main(int argc, char *argv[]) {
  kernel_arg_t kernel_arg;
  std::vector<uint8_t> src_pixels;
  uint32_t src_width;
  uint32_t src_height;
  uint32_t src_bpp;
 
  // parse command arguments
  parse_args(argc, argv);

  RT_CHECK(LoadTGA(input_file, src_pixels, &src_width, &src_height, &src_bpp));
  uint32_t src_bufsize = src_bpp * src_width * src_height;

  uint32_t dst_width   = (uint32_t)(src_width * scale);
  uint32_t dst_height  = (uint32_t)(src_height * scale);
  uint32_t dst_bpp     = 4;
  uint32_t dst_bufsize = dst_bpp * dst_width * dst_height;

  // open device connection
  std::cout << "open device connection" << std::endl;  
  RT_CHECK(vx_dev_open(&device));

  unsigned max_cores, max_warps, max_threads;
  RT_CHECK(vx_dev_caps(device, VX_CAPS_MAX_CORES, &max_cores));
  RT_CHECK(vx_dev_caps(device, VX_CAPS_MAX_WARPS, &max_warps));
  RT_CHECK(vx_dev_caps(device, VX_CAPS_MAX_THREADS, &max_threads));

  uint32_t num_tasks = max_cores * max_warps * max_threads;

  std::cout << "number of tasks: " << num_tasks << std::endl;
  std::cout << "source buffer: width=" << src_width << ", heigth=" << src_height << ", size=" << src_bufsize << " bytes" << std::endl;
  std::cout << "destination buffer: width=" << dst_width << ", heigth=" << dst_height << ", size=" << dst_bufsize << " bytes" << std::endl;

  // upload program
  std::cout << "upload program" << std::endl;  
  RT_CHECK(vx_upload_kernel_file(device, kernel_file));

  // allocate device memory
  std::cout << "allocate device memory" << std::endl;  
  size_t arg_addr, src_addr, dst_addr;
  RT_CHECK(vx_alloc_dev_mem(device, sizeof(kernel_arg_t), &arg_addr));
  RT_CHECK(vx_alloc_dev_mem(device, src_bufsize, &src_addr));
  RT_CHECK(vx_alloc_dev_mem(device, dst_bufsize, &dst_addr));

  assert(arg_addr == ALLOC_BASE_ADDR);

  std::cout << "arg_addr=" << std::hex << arg_addr << std::endl;
  std::cout << "src_addr=" << std::hex << src_addr << std::endl;
  std::cout << "dst_addr=" << std::hex << dst_addr << std::endl;

  // allocate staging shared memory  
  std::cout << "allocate shared memory" << std::endl;    
  uint32_t alloc_size = std::max<uint32_t>(sizeof(kernel_arg_t), std::max<uint32_t>(src_bufsize, dst_bufsize));
  RT_CHECK(vx_alloc_shared_mem(device, alloc_size, &buffer));
  
  // upload kernel argument
  std::cout << "upload kernel argument" << std::endl;
  {
    kernel_arg.num_tasks   = std::min<uint32_t>(num_tasks, dst_height);
    kernel_arg.src_width   = src_width;
    kernel_arg.src_height  = src_height;
    kernel_arg.src_pitch   = src_bpp * src_width * src_height;
    kernel_arg.dst_width   = dst_width;
    kernel_arg.dst_height  = dst_height;
    kernel_arg.dst_pitch   = dst_bpp * dst_width * dst_height;
    kernel_arg.src_ptr     = src_addr;
    kernel_arg.dst_ptr     = dst_addr;

    auto buf_ptr = (int*)vx_host_ptr(buffer);
    memcpy(buf_ptr, &kernel_arg, sizeof(kernel_arg_t));
    RT_CHECK(vx_copy_to_dev(buffer, arg_addr, sizeof(kernel_arg_t), 0));
  }

  // upload source buffer0
  std::cout << "upload source buffer0" << std::endl;      
  {    
    auto buf_ptr = (int8_t*)vx_host_ptr(buffer);
    for (uint32_t i = 0; i < src_bufsize; ++i) {
      buf_ptr[i] = src_pixels[i];
    }      
    RT_CHECK(vx_copy_to_dev(buffer, kernel_arg.src_ptr, src_bufsize, 0));
  }

  // clear destination buffer
  std::cout << "clear destination buffer" << std::endl;      
  {    
    auto buf_ptr = (int32_t*)vx_host_ptr(buffer);
    for (uint32_t i = 0; i < (dst_bufsize/4); ++i) {
      buf_ptr[i] = 0xdeadbeef;
    }    
    RT_CHECK(vx_copy_to_dev(buffer, kernel_arg.dst_ptr, dst_bufsize, 0));  
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