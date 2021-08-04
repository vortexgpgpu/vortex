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
const char* input_file  = "palette64.png";
const char* output_file = "output.png";
int wrap    = 0;
int filter  = 0;
float scale = 1.0f;
int format  = 0;
bool use_sw = false;
ePixelFormat eformat = FORMAT_A8R8G8B8;

vx_device_h device = nullptr;
vx_buffer_h buffer = nullptr;

static void show_usage() {
   std::cout << "Vortex Texture Test." << std::endl;
   std::cout << "Usage: [-k: kernel] [-i image] [-o image] [-s scale] [-w wrap] [-f format] [-g filter] [-z no_hw] [-h: help]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "zi:o:k:w:f:g:h?")) != -1) {
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
    case 'w':
      wrap = std::atoi(optarg);
      break;
    case 'z':
      use_sw = true;
      break;
    case 'f': {
      format  = std::atoi(optarg);
      switch (format) {
      case 0: eformat = FORMAT_A8R8G8B8; break;
      case 1: eformat = FORMAT_R5G6B5; break;
      case 2: eformat = FORMAT_R4G4B4A4; break;
      case 3: eformat = FORMAT_L8; break;
      case 4: eformat = FORMAT_A8; break;
      default:
        std::cout << "Error: invalid format: " << format << std::endl;
        exit(1);
      }      
    } break;
    case 'g':
      filter = std::atoi(optarg);
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

int run_test(const kernel_arg_t& kernel_arg, 
             uint32_t buf_size, 
             uint32_t width, 
             uint32_t height) {
  auto time_start = std::chrono::high_resolution_clock::now();

  // start device
  std::cout << "start device" << std::endl;
  RT_CHECK(vx_start(device));

  // wait for completion
  std::cout << "wait for completion" << std::endl;
  RT_CHECK(vx_ready_wait(device, -1));
  
  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  // download destination buffer
  std::cout << "download destination buffer" << std::endl;
  RT_CHECK(vx_copy_from_dev(buffer, kernel_arg.dst_ptr, buf_size, 0));

  std::vector<uint8_t> dst_pixels(buf_size);
  auto buf_ptr = (uint8_t*)vx_host_ptr(buffer);
  for (uint32_t i = 0; i < buf_size; ++i) {
    dst_pixels[i] = buf_ptr[i];
  } 

  // save output image
  std::cout << "save output image" << std::endl;  
  //dump_image(dst_pixels, width, height, bpp);
  RT_CHECK(SaveImage(output_file, FORMAT_A8R8G8B8, dst_pixels, width, height));

  return 0;
}

int main(int argc, char *argv[]) {
  kernel_arg_t kernel_arg;  
  std::vector<uint8_t> src_pixels;
  uint32_t src_width;
  uint32_t src_height;
  
  // parse command arguments
  parse_args(argc, argv);

  RT_CHECK(LoadImage(input_file, eformat, src_pixels, &src_width, &src_height));

  // check power of two support
  if (!ISPOW2(src_width) || !ISPOW2(src_height)) {
    std::cout << "Error: only power of two textures supported: width=" << src_width << ", heigth=" << src_height << std::endl;
    return -1;
  }

  uint32_t src_bpp = Format::GetInfo(eformat).BytePerPixel;
  
  //dump_image(src_pixels, src_width, src_height, src_bpp);

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

  std::cout << "number of tasks: " << std::dec << num_tasks << std::endl;
  std::cout << "source buffer: width=" << src_width << ", heigth=" << src_height << ", size=" << src_bufsize << " bytes" << std::endl;
  std::cout << "destination buffer: width=" << dst_width << ", heigth=" << dst_height << ", size=" << dst_bufsize << " bytes" << std::endl;

  // upload program
  std::cout << "upload program" << std::endl;  
  RT_CHECK(vx_upload_kernel_file(device, kernel_file));

  // allocate device memory
  std::cout << "allocate device memory" << std::endl;  
  size_t src_addr, dst_addr;
  RT_CHECK(vx_alloc_dev_mem(device, src_bufsize, &src_addr));
  RT_CHECK(vx_alloc_dev_mem(device, dst_bufsize, &dst_addr));

  std::cout << "src_addr=0x" << std::hex << src_addr << std::endl;
  std::cout << "dst_addr=0x" << std::hex << dst_addr << std::endl;

  // allocate staging shared memory  
  std::cout << "allocate shared memory" << std::endl;    
  uint32_t alloc_size = std::max<uint32_t>(sizeof(kernel_arg_t), std::max<uint32_t>(src_bufsize, dst_bufsize));
  RT_CHECK(vx_alloc_shared_mem(device, alloc_size, &buffer));
  
  // upload kernel argument
  std::cout << "upload kernel argument" << std::endl;
  {
    kernel_arg.num_tasks  = std::min<uint32_t>(num_tasks, dst_height);
    kernel_arg.format     = format;
    kernel_arg.filter     = filter;
    kernel_arg.wrap       = wrap;
    kernel_arg.use_sw     = use_sw;
    kernel_arg.lod        = 0x0;
    
    kernel_arg.src_logWidth  = (uint32_t)std::log2(src_width);
    kernel_arg.src_logHeight = (uint32_t)std::log2(src_height);
    kernel_arg.src_stride = src_bpp;
    kernel_arg.src_pitch  = src_bpp * src_width;
    kernel_arg.src_ptr    = src_addr;

    kernel_arg.dst_width  = dst_width;
    kernel_arg.dst_height = dst_height;
    kernel_arg.dst_stride = dst_bpp;
    kernel_arg.dst_pitch  = dst_bpp * dst_width;    
    kernel_arg.dst_ptr    = dst_addr;

    auto buf_ptr = (int*)vx_host_ptr(buffer);
    memcpy(buf_ptr, &kernel_arg, sizeof(kernel_arg_t));
    RT_CHECK(vx_copy_to_dev(buffer, KERNEL_ARG_DEV_MEM_ADDR, sizeof(kernel_arg_t), 0));
  }

  // upload source buffer
  std::cout << "upload source buffer" << std::endl;      
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
  RT_CHECK(run_test(kernel_arg, dst_bufsize, dst_width, dst_height));

  // cleanup
  std::cout << "cleanup" << std::endl;  
  cleanup();

  std::cout << "PASSED!" << std::endl;

  return 0;
}