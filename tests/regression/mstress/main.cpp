#include <iostream>
#include <unistd.h>
#include <string.h>
#include <vortex.h>
#include "common.h"
#include <assert.h>
#include <limits>
#include <math.h>
#include <vector>

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

union Float_t {    
    float f;
    int   i;
    struct {
        uint32_t man  : 23;
        uint32_t exp  : 8;
        uint32_t sign : 1;
    } parts;
};

inline float fround(float x, int32_t precision = 8) {
  auto power_of_10 = std::pow(10, precision);
  return std::round(x * power_of_10) / power_of_10;
}

inline bool almost_equal_eps(float a, float b, int ulp = 128) {
  auto eps = std::numeric_limits<float>::epsilon() * (std::max(fabs(a), fabs(b)) * ulp);
  auto d = fabs(a - b);
  if (d > eps) {
    std::cout << "*** almost_equal_eps: d=" << d << ", eps=" << eps << std::endl;
    return false;
  }
  return true;
}

inline bool almost_equal_ulp(float a, float b, int32_t ulp = 6) {
  Float_t fa{a}, fb{b};
  auto d = std::abs(fa.i - fb.i);
  if (d > ulp) {
    std::cout << "*** almost_equal_ulp: a=" << a << ", b=" << b << ", ulp=" << d << ", ia=" << std::hex << fa.i << ", ib=" << fb.i << std::endl;
    return false;
  }
  return true;
}

inline bool almost_equal(float a, float b) {
  if (a == b)
    return true;
  /*if (almost_equal_eps(a, b))
    return true;*/
  return almost_equal_ulp(a, b);
}

///////////////////////////////////////////////////////////////////////////////

const char* kernel_file = "kernel.bin";
uint32_t count = 0;

std::vector<float> test_data;
std::vector<uint32_t> addr_table;

vx_device_h device = nullptr;
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
      count = atoi(optarg);
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
    vx_mem_free(device, kernel_arg.src0_addr);
    vx_mem_free(device, kernel_arg.src1_addr);
    vx_mem_free(device, kernel_arg.dst_addr);
    vx_dev_close(device);
  }
}

void gen_input_data(uint32_t num_points) {
  test_data.resize(num_points);
  addr_table.resize(num_points + NUM_LOADS - 1);

  for (uint32_t i = 0; i < num_points; ++i) {
    float r = static_cast<float>(std::rand()) / RAND_MAX;
    test_data[i] = r;
  }

  for (uint32_t i = 0; i < addr_table.size(); ++i) {
    float r = static_cast<float>(std::rand()) / RAND_MAX;
    uint32_t index = static_cast<uint32_t>(r * num_points);
    assert(index < num_points);
    addr_table[i] = index;
  }
}

int run_test(const kernel_arg_t& kernel_arg,
             uint32_t dst_buf_size, 
             uint32_t num_points) {
  // start device
  std::cout << "start device" << std::endl;
  RT_CHECK(vx_start(device));

  // wait for completion
  std::cout << "wait for completion" << std::endl;
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  // download destination buffer
  std::cout << "download destination buffer" << std::endl;
  RT_CHECK(vx_copy_from_dev(device, staging_buf.data(), kernel_arg.dst_addr, dst_buf_size));

  // verify result
  std::cout << "verify result" << std::endl;  
  {
    int errors = 0;
    auto buf_ptr = (float*)staging_buf.data();
    
    for (uint32_t i = 0; i < num_points; ++i) {

      float ref = 0.0f;
      for (uint32_t j = 0; j < NUM_LOADS; ++j) {
        uint32_t addr = i + j;
        uint32_t index = addr_table.at(addr);
        float value = test_data.at(index);
        //printf("*** [%d] addr=%d, index=%d, value=%f\n", i, addr, index, value);
        ref *= value;
      }
      
      float cur = buf_ptr[i];
      if (!almost_equal(cur, ref)) {
        std::cout << "error at result #" << std::dec << i
                  << ": actual " << cur << ", expected " << ref << std::endl;
        ++errors;
      }
    }

    if (errors != 0) {
      std::cout << "Found " << std::dec << errors << " errors!" << std::endl;
      std::cout << "FAILED!" << std::endl;
      return 1;  
    }
  }

  return 0;
}

int main(int argc, char *argv[]) {  
  // parse command arguments
  parse_args(argc, argv);

  if (count == 0) {
    count = 1;
  }

  std::srand(50);

  // open device connection
  std::cout << "open device connection" << std::endl;  
  RT_CHECK(vx_dev_open(&device));

  uint64_t num_cores, num_warps, num_threads;
  RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_CORES, &num_cores));
  RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_WARPS, &num_warps));
  RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_THREADS, &num_threads));

  uint32_t num_tasks  = num_cores * num_warps * num_threads;
  uint32_t num_points = count * num_tasks;

  // generate input data
  gen_input_data(num_points);

  uint32_t addr_buf_size = addr_table.size() * sizeof(int32_t);
  uint32_t src_buf_size  = test_data.size() * sizeof(int32_t);  
  uint32_t dst_buf_size  = test_data.size() * sizeof(int32_t);

  std::cout << "number of points: " << num_points << std::endl;
  std::cout << "buffer size: " << dst_buf_size << " bytes" << std::endl;

  // upload program
  std::cout << "upload program" << std::endl;  
  RT_CHECK(vx_upload_kernel_file(device, kernel_file));

  // allocate device memory
  std::cout << "allocate device memory" << std::endl;
  RT_CHECK(vx_mem_alloc(device, addr_buf_size, VX_MEM_TYPE_GLOBAL, &kernel_arg.src0_addr));
  RT_CHECK(vx_mem_alloc(device, src_buf_size, VX_MEM_TYPE_GLOBAL, &kernel_arg.src1_addr));
  RT_CHECK(vx_mem_alloc(device, dst_buf_size, VX_MEM_TYPE_GLOBAL, &kernel_arg.dst_addr));

  kernel_arg.num_tasks = num_tasks;
  kernel_arg.stride = count;

  std::cout << "dev_addr=0x" << std::hex << kernel_arg.src0_addr << std::endl;
  std::cout << "dev_src=0x"  << std::hex << kernel_arg.src1_addr << std::endl;  
  std::cout << "dev_dst=0x"  << std::hex << kernel_arg.dst_addr << std::endl;
  
  // allocate staging buffer  
  std::cout << "allocate staging buffer" << std::endl;    
  uint32_t staging_buf_size = std::max<uint32_t>(src_buf_size, 
                                std::max<uint32_t>(addr_buf_size, 
                                  std::max<uint32_t>(dst_buf_size, 
                                    sizeof(kernel_arg_t))));
  staging_buf.resize(staging_buf_size);
  
  // upload kernel argument  
  std::cout << "upload kernel argument" << std::endl;
  memcpy(staging_buf.data(), &kernel_arg, sizeof(kernel_arg_t));
  RT_CHECK(vx_copy_to_dev(device, KERNEL_ARG_DEV_MEM_ADDR, staging_buf.data(), sizeof(kernel_arg_t)));
  
  // upload source buffer0
  {
    std::cout << "upload address buffer" << std::endl;
    auto buf_ptr = staging_buf.data();
    memcpy(buf_ptr, addr_table.data(), addr_table.size() * sizeof(int32_t));  
    RT_CHECK(vx_copy_to_dev(device, kernel_arg.src0_addr, staging_buf.data(), addr_buf_size));
  }

  // upload source buffer1
  {
    std::cout << "upload source buffer" << std::endl;
    auto buf_ptr = staging_buf.data();
    memcpy(buf_ptr, test_data.data(), test_data.size() * sizeof(int32_t));       
    RT_CHECK(vx_copy_to_dev(device, kernel_arg.src1_addr, staging_buf.data(), src_buf_size));
  }

  // clear destination buffer
  {
    std::cout << "clear destination buffer" << std::endl;      
    auto buf_ptr = (int32_t*)staging_buf.data();
    for (uint32_t i = 0; i < test_data.size(); ++i) {
      buf_ptr[i] = 0xdeadbeef;
    }  
    RT_CHECK(vx_copy_to_dev(device, kernel_arg.dst_addr, staging_buf.data(), dst_buf_size));  
  }

  // run tests
  std::cout << "run tests" << std::endl;
  RT_CHECK(run_test(kernel_arg, dst_buf_size, num_points));

  // cleanup
  std::cout << "cleanup" << std::endl;  
  cleanup();

  std::cout << "PASSED!" << std::endl;

  return 0;
}