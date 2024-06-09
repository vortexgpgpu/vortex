#include <iostream>
#include <unistd.h>
#include <string.h>
#include <vector>
#include <vortex.h>
#include "common.h"
#include "common.c"

#define FLOAT_ULP 6

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

template <typename Type>
class Comparator {};

template <>
class Comparator<int> {
public:
  static const char* type_str() {
    return "integer";
  }
  static int generate() { 
    return rand(); 
  }
  static bool compare(int a, int b, int index, int errors) { 
    if (a != b) {
      if (errors < 100) {
        printf("*** error: [%d] expected=%d, actual=%d\n", index, a, b);
      }
      return false;
    }
    return true;
  }  
};

template <>
class Comparator<float> {
private:
  union Float_t { float f; int i; };
public:
  static const char* type_str() {
    return "float";
  }
  static int generate() { 
    return static_cast<float>(rand()) / RAND_MAX;
  }
  static bool compare(float a, float b, int index, int errors) {     
    union fi_t { float f; int32_t i; };
    fi_t fa, fb;
    fa.f = a;
    fb.f = b;
    auto d = std::abs(fa.i - fb.i);
    if (d > FLOAT_ULP) {
      if (errors < 100) {
        printf("*** error: [%d] expected=%f, actual=%f\n", index, a, b);
      }
      return false;
    }
    return true;
  }  
};

#define TEX_DCR_WRITE(addr, value)  \
  vx_dcr_write(device, addr, value); \
  kernel_arg.sampler.write(addr, value)

const char* kernel_file = "kernel.bin";
uint32_t size = 16;

vx_device_h device = nullptr;
std::vector<TYPE> source_data;
std::vector<float4> coord_data;
std::vector<uint8_t> staging_buf;
kernel_arg_t kernel_arg = {};

static void show_usage() {
   std::cout << "Vortex Test." << std::endl;
   std::cout << "Usage: [-k: kernel] [-n words] [-h: help]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "n:k:h?")) != -1) {
    switch (c) {
    case 'n':
      size = atoi(optarg);
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
  if (device) {    
    vx_mem_free(device, kernel_arg.discard_addr);
    vx_mem_free(device, kernel_arg.fragCoord_addr);
    vx_mem_free(device, kernel_arg.image_addr);
    vx_mem_free(device, kernel_arg.rasterization_addr);
    vx_mem_free(device, kernel_arg.fragColor_addr);
    vx_dev_close(device);
  }
}

int run_test(const kernel_arg_t& kernel_arg,
             uint32_t buf_size, 
             uint32_t num_points) {              
  // start device
  std::cout << "start device" << std::endl;
  RT_CHECK(vx_start(device));

  // wait for completion
  std::cout << "wait for completion" << std::endl;
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  // download destination buffer
  std::cout << "download destination buffer" << std::endl;
  RT_CHECK(vx_copy_from_dev(device, staging_buf.data(), kernel_arg.fragColor_addr, buf_size));

  // verify result
  /*
  std::cout << "verify result" << std::endl;  
  {
    int errors = 0;
    auto buf_ptr = (TYPE*)staging_buf.data();
    for (uint32_t i = 0; i < num_points; ++i) {
      auto ref = source_data[2 * i + 0] + source_data[2 * i + 1];
      auto cur = buf_ptr[i];
      if (!Comparator<TYPE>::compare(cur, ref, i, errors)) {
        ++errors;
      }
    }
    if (errors != 0) {
      std::cout << "Found " << std::dec << errors << " errors!" << std::endl;
      std::cout << "FAILED!" << std::endl;
      return 1;  
    }
  }
  */
  return 0;
}

int main(int argc, char *argv[]) {  
  // parse command arguments
  parse_args(argc, argv);

  std::srand(50);

  // open device connection
  std::cout << "open device connection" << std::endl;  
  RT_CHECK(vx_dev_open(&device));

  uint64_t num_cores, num_warps, num_threads;
  RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_CORES, &num_cores));
  RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_WARPS, &num_warps));
  RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_THREADS, &num_threads));
  std::cout << "number of cores: " << num_cores << std::endl;
  std::cout << "number of warps: " << num_warps << std::endl;
  std::cout << "number of threads: " << num_threads << std::endl;

  uint32_t num_fragments = WIDTH*HEIGHT;  

  std::cout << "number of fragments: " << num_fragments << std::endl;  

  // upload program
  std::cout << "upload program" << std::endl;  
  RT_CHECK(vx_upload_kernel_file(device, kernel_file));

  // allocate device memory
  std::cout << "allocate device memory" << std::endl;
  RT_CHECK(vx_mem_alloc(device, num_fragments*sizeof(uint8_t[4]), VX_MEM_TYPE_GLOBAL, &kernel_arg.image_addr));
  RT_CHECK(vx_mem_alloc(device, num_fragments*sizeof(float[4]), VX_MEM_TYPE_GLOBAL, &kernel_arg.fragCoord_addr));
  RT_CHECK(vx_mem_alloc(device, num_fragments*sizeof(float[4]), VX_MEM_TYPE_GLOBAL, &kernel_arg.rasterization_addr));
  RT_CHECK(vx_mem_alloc(device, num_fragments*sizeof(uint8_t), VX_MEM_TYPE_GLOBAL, &kernel_arg.discard_addr));
  RT_CHECK(vx_mem_alloc(device, num_fragments*sizeof(float[4]), VX_MEM_TYPE_GLOBAL, &kernel_arg.fragColor_addr));

  kernel_arg.size = {1, 1}; // TODO: image size

  std::cout << "dev_image=0x" << std::hex << kernel_arg.image_addr << std::endl;
  std::cout << "dev_fragCoord=0x" << std::hex << kernel_arg.fragCoord_addr << std::endl;
  std::cout << "dev_rasterization=0x" << std::hex << kernel_arg.rasterization_addr << std::endl;
  std::cout << "dev_discard=0x" << std::hex << kernel_arg.discard_addr << std::endl;
  std::cout << "dev_fragColor=0x" << std::hex << kernel_arg.fragColor_addr << std::endl;
  
  // allocate staging buffer  
  std::cout << "allocate staging buffer" << std::endl;    
  uint32_t alloc_size = std::max<uint32_t>(num_fragments*sizeof(float[4]), sizeof(kernel_arg_t));
  staging_buf.resize(alloc_size);
  
  // upload kernel argument
  std::cout << "upload kernel argument" << std::endl;
  memcpy(staging_buf.data(), &kernel_arg, sizeof(kernel_arg_t));
  RT_CHECK(vx_copy_to_dev(device, KERNEL_ARG_DEV_MEM_ADDR, staging_buf.data(), sizeof(kernel_arg_t)));

  // generate source data
  PPMImage* image = readPPM("dog.ppm");
  uint8_t* data = (uint8_t*) malloc(image->x*image->y*sizeof(uint8_t[4]));
  for(uint32_t i=0; i<image->x*image->y; ++i) {
    data[i*4+0] = image->data[i].red;
    data[i*4+1] = image->data[i].green;
    data[i*4+2] = image->data[i].blue;
    data[i*4+3] = 0xFFu;
  }

  RT_CHECK(vx_copy_to_dev(device, kernel_arg.image_addr, data, image->x*image->y*sizeof(uint8_t[4])));

  coord_data.resize(num_fragments);
  for(int h=0; h<HEIGHT; ++h) {
    for(int w=0; w<WIDTH; ++w) {
      coord_data[h*WIDTH+w] = {
        .x = (float)w,
        .y = (float)h
      };
    }
  }

  // upload source buffer0
  RT_CHECK(vx_copy_to_dev(device, kernel_arg.rasterization_addr, coord_data.data(), num_fragments*sizeof(float4)));

  // clear destination buffer
  std::cout << "clear destination buffer" << std::endl;
  memset(staging_buf.data(), 0, num_fragments*sizeof(float4));
  RT_CHECK(vx_copy_to_dev(device, kernel_arg.fragColor_addr, staging_buf.data(), num_fragments*sizeof(float4)));  
  

  // configure texture units
  #ifdef SKYBOX
	TEX_DCR_WRITE(VX_DCR_TEX_STAGE,   0);
	TEX_DCR_WRITE(VX_DCR_TEX_LOGDIM,  (log2ceil(image->y) << 16) | log2ceil(image->x));	
	TEX_DCR_WRITE(VX_DCR_TEX_FORMAT,  VX_TEX_FORMAT_A8R8G8B8);
	TEX_DCR_WRITE(VX_DCR_TEX_WRAP,    (VX_TEX_WRAP_CLAMP << 16) | VX_TEX_WRAP_CLAMP);
	TEX_DCR_WRITE(VX_DCR_TEX_FILTER,  (VX_TEX_FILTER_POINT ? VX_TEX_FILTER_BILINEAR : VX_TEX_FILTER_POINT));
	TEX_DCR_WRITE(VX_DCR_TEX_ADDR,    kernel_arg.image_addr / 64); // block address
  #endif

  // run tests
  std::cout << "run tests" << std::endl;
  RT_CHECK(run_test(kernel_arg, num_fragments*sizeof(float4), num_fragments));

  // cleanup
  std::cout << "cleanup" << std::endl;  
  cleanup();

  std::cout << "PASSED!" << std::endl;

  return 0;
}
