#include <iostream>
#include <vector>
#include <unistd.h>
#include <string.h>
#include <chrono>
#include <cmath>
#include <assert.h>
#include <vortex.h>
#include "common.h"
#include <cocogfx/include/fixed.hpp>
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
const char* output_file = "output.png";
const char* reference_file  = nullptr;

uint32_t color = 0xffffffff;
uint32_t depth = TFixed<24>(0.5f).data();

bool blend_enable = false;
bool depth_enable = false;
bool backface     = false;

uint32_t clear_color = 0x00000000;
uint32_t clear_depth = TFixed<24>(0.5f).data();

uint32_t dst_width  = 128;
uint32_t dst_height = 128;

uint32_t zbuf_stride;
uint32_t zbuf_pitch;
uint32_t zbuf_size;

uint32_t cbuf_stride;
uint32_t cbuf_pitch;
uint32_t cbuf_size;

vx_device_h device = nullptr;
std::vector<uint8_t> staging_buf;
uint64_t zbuf_addr = 0;
uint64_t cbuf_addr = 0;
bool use_sw = false;

kernel_arg_t kernel_arg = {};

static void show_usage() {
   std::cout << "Vortex Render Output Test." << std::endl;
   std::cout << "Usage: [-c color] [-d depth] [-b blend] [-f face] [-k: kernel] [-o image] [-r reference] [-w width] [-h height] [-z no_hw]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "o:r:k:w:h:c:bdfz?")) != -1) {
    switch (c) {
    case 'o':
      output_file = optarg;
      break;
    case 'r':
      reference_file = optarg;
      break;
    case 'k':
      kernel_file = optarg;
      break;
    case 'w':
      dst_width = std::atoi(optarg);
      break;
    case 'h':
      dst_height = std::atoi(optarg);
      break;
    case 'f':
      backface = true;
      break;
    case 'c':
      color = std::atoi(optarg);
      break;
    case 'd':
      depth_enable = true;
      break;
    case 'b':
      blend_enable = true;
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
  if (device) {     
    if (zbuf_addr != 0) vx_mem_free(device, zbuf_addr);
    if (cbuf_addr != 0) vx_mem_free(device, cbuf_addr);
    vx_dev_close(device);
  }
}

int render(uint32_t num_tasks) {
  uint32_t alloc_size = sizeof(kernel_arg_t);
  staging_buf.resize(alloc_size);
  
  // upload kernel argument
  std::cout << "upload kernel argument" << std::endl;
  {
    kernel_arg.use_sw     = use_sw;
    kernel_arg.num_tasks  = num_tasks;
    kernel_arg.dst_width  = dst_width;
    kernel_arg.dst_height = dst_height;    
    kernel_arg.color      = color;
    kernel_arg.depth      = depth;
    kernel_arg.backface   = backface;
    kernel_arg.blend_enable = blend_enable;

    memcpy(staging_buf.data(), &kernel_arg, sizeof(kernel_arg_t));
    RT_CHECK(vx_copy_to_dev(device, KERNEL_ARG_DEV_MEM_ADDR, staging_buf.data(), sizeof(kernel_arg_t)));
  }

  // configure rop color buffer
  vx_dcr_write(device, VX_DCR_OM_CBUF_ADDR,  cbuf_addr / 64); // block address
  vx_dcr_write(device, VX_DCR_OM_CBUF_PITCH, cbuf_pitch);
  vx_dcr_write(device, VX_DCR_OM_CBUF_WRITEMASK, 0xf);

  // configure rop depth buffer to default
  vx_dcr_write(device, VX_DCR_OM_ZBUF_ADDR,  zbuf_addr / 64); // block address
  vx_dcr_write(device, VX_DCR_OM_ZBUF_PITCH, zbuf_pitch);   
  if (depth_enable) {
    vx_dcr_write(device, VX_DCR_OM_DEPTH_FUNC, VX_OM_DEPTH_FUNC_LESS);
    vx_dcr_write(device, VX_DCR_OM_DEPTH_WRITEMASK, 1);
  } else {
    vx_dcr_write(device, VX_DCR_OM_DEPTH_FUNC, VX_OM_DEPTH_FUNC_ALWAYS);
    vx_dcr_write(device, VX_DCR_OM_DEPTH_WRITEMASK, 0);
  }
  
  // configure rop stencil states to default
  vx_dcr_write(device, VX_DCR_OM_STENCIL_FUNC,  VX_OM_DEPTH_FUNC_ALWAYS);
  vx_dcr_write(device, VX_DCR_OM_STENCIL_ZPASS, VX_OM_STENCIL_OP_KEEP);
  vx_dcr_write(device, VX_DCR_OM_STENCIL_ZPASS, VX_OM_STENCIL_OP_KEEP);
  vx_dcr_write(device, VX_DCR_OM_STENCIL_FAIL,  VX_OM_STENCIL_OP_KEEP);
  vx_dcr_write(device, VX_DCR_OM_STENCIL_REF,   0);
  vx_dcr_write(device, VX_DCR_OM_STENCIL_MASK,  VX_OM_STENCIL_MASK);
  vx_dcr_write(device, VX_DCR_OM_STENCIL_WRITEMASK, 0);

  // configure rop blend states to default
  if (blend_enable) {
    vx_dcr_write(device, VX_DCR_OM_BLEND_MODE, (VX_OM_BLEND_MODE_ADD << 16)   // DST
                                              | (VX_OM_BLEND_MODE_ADD << 0));  // SRC
    vx_dcr_write(device, VX_DCR_OM_BLEND_FUNC, (VX_OM_BLEND_FUNC_ONE_MINUS_SRC_A << 24)  // DST_A
                                              | (VX_OM_BLEND_FUNC_ONE_MINUS_SRC_A << 16)  // DST_RGB 
                                              | (VX_OM_BLEND_FUNC_ONE << 8)    // SRC_A
                                              | (VX_OM_BLEND_FUNC_ONE << 0));  // SRC_RGB
  } else {
    vx_dcr_write(device, VX_DCR_OM_BLEND_MODE, (VX_OM_BLEND_MODE_ADD << 16)   // DST
                                              | (VX_OM_BLEND_MODE_ADD << 0));  // SRC
    vx_dcr_write(device, VX_DCR_OM_BLEND_FUNC, (VX_OM_BLEND_FUNC_ZERO << 24)  // DST_A
                                              | (VX_OM_BLEND_FUNC_ZERO << 16)  // DST_RGB 
                                              | (VX_OM_BLEND_FUNC_ONE << 8)    // SRC_A
                                              | (VX_OM_BLEND_FUNC_ONE << 0));  // SRC_RGB
  }
  
  auto time_start = std::chrono::high_resolution_clock::now();

  // start device
  std::cout << "start device" << std::endl;
  RT_CHECK(vx_start(device));

  // wait for completion
  std::cout << "wait for completion" << std::endl;
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));
  
  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  // save output image
  if (strcmp(output_file, "null") != 0) {
    std::cout << "save output image" << std::endl;
    std::vector<uint8_t> dst_pixels(cbuf_size);
    RT_CHECK(vx_copy_from_dev(device, dst_pixels.data(), cbuf_addr, cbuf_size));
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
  if (0 == (isa_flags & VX_ISA_EXT_OM)) {
    std::cout << "ROP extension not supported!" << std::endl;
    cleanup();
    return -1;
  }

  std::cout << "using color=" << std::hex << color << ", depth=" << depth << std::endl;

  uint64_t num_cores, num_warps, num_threads;
  RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_CORES, &num_cores));
  RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_WARPS, &num_warps));
  RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_THREADS, &num_threads));

  uint32_t num_tasks = num_cores * num_warps * num_threads;

  std::cout << "number of tasks: " << std::dec << num_tasks << std::endl;

  // upload program
  std::cout << "upload program" << std::endl;  
  RT_CHECK(vx_upload_kernel_file(device, kernel_file));

  zbuf_stride = 4;
  zbuf_pitch  = dst_width * zbuf_stride;
  zbuf_size   = dst_height * zbuf_pitch;

  cbuf_stride = 4;
  cbuf_pitch  = dst_width * cbuf_stride;
  cbuf_size   = dst_height * cbuf_pitch;

  // allocate device memory  
  RT_CHECK(vx_mem_alloc(device, zbuf_size, VX_MEM_TYPE_GLOBAL, &zbuf_addr));
  RT_CHECK(vx_mem_alloc(device, cbuf_size, VX_MEM_TYPE_GLOBAL, &cbuf_addr));

  std::cout << "zbuf_addr=0x" << std::hex << zbuf_addr << std::endl;
  std::cout << "cbuf_addr=0x" << std::hex << cbuf_addr << std::endl;

  // allocate staging buffer  
  std::cout << "allocate staging buffer" << std::endl;    
  uint32_t alloc_size = std::max(zbuf_size, cbuf_size);
  staging_buf.resize(alloc_size);
  
  // clear depth buffer
  std::cout << "clear depth buffer" << std::endl;      
  {    
    auto buf_ptr = (uint32_t*)staging_buf.data();
    for (uint32_t y = 0; y < dst_height; ++y) {
      for (uint32_t x = 0; x < dst_width; ++x) {
        buf_ptr[x + y * dst_width] = ((x & 0x1) == (y & 0x1)) ? TFixed<24>(0.0f).data() : TFixed<24>(0.99f).data();
      }
    }    
    RT_CHECK(vx_copy_to_dev(device, zbuf_addr, staging_buf.data(), zbuf_size));
  }

  // clear destination buffer
  std::cout << "clear destination buffer" << std::endl;      
  {    
    auto buf_ptr = (uint32_t*)staging_buf.data();
    for (uint32_t i = 0; i < (cbuf_size/4); ++i) {
      buf_ptr[i] = clear_color;
    }    
    RT_CHECK(vx_copy_to_dev(device, cbuf_addr, staging_buf.data(), cbuf_size));  
  }
  
  // run tests
  std::cout << "render" << std::endl;
  RT_CHECK(render(num_tasks));

  // cleanup
  std::cout << "cleanup" << std::endl;  
  cleanup();  

  if (reference_file) {
    auto errors = CompareImages(output_file, reference_file, FORMAT_A8R8G8B8);
    if (0 == errors) {
      std::cout << "PASSED!" << std::endl;
    } else {
      std::cout << "FAILED!" << std::endl;
      return errors;
    }
  }

  return 0;
}