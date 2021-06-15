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
vx_buffer_h staging_buf = nullptr;

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
  if (staging_buf) {
    vx_buf_release(staging_buf);
  }
  if (device) {
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
  RT_CHECK(vx_ready_wait(device, -1));

  // download destination buffer
  std::cout << "download destination buffer" << std::endl;
  RT_CHECK(vx_copy_from_dev(staging_buf, kernel_arg.dst_ptr, dst_buf_size, 0));

  // verify result
  std::cout << "verify result" << std::endl;  
  {
    int errors = 0;
    auto buf_ptr = (float*)vx_host_ptr(staging_buf);
    
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
  size_t value; 
  kernel_arg_t kernel_arg;
  
  // parse command arguments
  parse_args(argc, argv);

  if (count == 0) {
    count = 1;
  }

  std::srand(50);

  // open device connection
  std::cout << "open device connection" << std::endl;  
  RT_CHECK(vx_dev_open(&device));

  unsigned max_cores, max_warps, max_threads;
  RT_CHECK(vx_dev_caps(device, VX_CAPS_MAX_CORES, &max_cores));
  RT_CHECK(vx_dev_caps(device, VX_CAPS_MAX_WARPS, &max_warps));
  RT_CHECK(vx_dev_caps(device, VX_CAPS_MAX_THREADS, &max_threads));

  uint32_t num_tasks  = max_cores * max_warps * max_threads;
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

  RT_CHECK(vx_alloc_dev_mem(device, addr_buf_size, &value));
  kernel_arg.addr_ptr = value;
  RT_CHECK(vx_alloc_dev_mem(device, src_buf_size, &value));
  kernel_arg.src_ptr = value;
  RT_CHECK(vx_alloc_dev_mem(device, dst_buf_size, &value));
  kernel_arg.dst_ptr = value;

  kernel_arg.num_tasks = num_tasks;
  kernel_arg.stride = count;

  std::cout << "dev_addr=" << std::hex << kernel_arg.addr_ptr << std::endl;
  std::cout << "dev_src="  << std::hex << kernel_arg.src_ptr << std::endl;  
  std::cout << "dev_dst="  << std::hex << kernel_arg.dst_ptr << std::endl;
  
  // allocate shared memory  
  std::cout << "allocate shared memory" << std::endl;    
  uint32_t staging_buf_size = std::max<uint32_t>(src_buf_size, 
                                std::max<uint32_t>(addr_buf_size, 
                                  std::max<uint32_t>(dst_buf_size, 
                                    sizeof(kernel_arg_t))));
  RT_CHECK(vx_alloc_shared_mem(device, staging_buf_size, &staging_buf));
  
  // upload kernel argument
  std::cout << "upload kernel argument" << std::endl;
  {
    auto buf_ptr = (int*)vx_host_ptr(staging_buf);
    memcpy(buf_ptr, &kernel_arg, sizeof(kernel_arg_t));
    RT_CHECK(vx_copy_to_dev(staging_buf, KERNEL_ARG_DEV_MEM_ADDR, sizeof(kernel_arg_t), 0));
  }

  // upload source buffer0
  {
    auto buf_ptr = (int32_t*)vx_host_ptr(staging_buf);
    for (uint32_t i = 0; i < addr_table.size(); ++i) {
      buf_ptr[i] = addr_table.at(i);
    }
  }
  std::cout << "upload address buffer" << std::endl;      
  RT_CHECK(vx_copy_to_dev(staging_buf, kernel_arg.addr_ptr, addr_buf_size, 0));

  // upload source buffer1
  {
    auto buf_ptr = (int32_t*)vx_host_ptr(staging_buf);
    for (uint32_t i = 0; i < test_data.size(); ++i) {
      buf_ptr[i] = test_data.at(i);
    }
  }
  std::cout << "upload source buffer" << std::endl;      
  RT_CHECK(vx_copy_to_dev(staging_buf, kernel_arg.src_ptr, src_buf_size, 0));

  // clear destination buffer
  {
    auto buf_ptr = (int32_t*)vx_host_ptr(staging_buf);
    for (uint32_t i = 0; i < test_data.size(); ++i) {
      buf_ptr[i] = 0xdeadbeef;
    }
  }
  std::cout << "clear destination buffer" << std::endl;      
  RT_CHECK(vx_copy_to_dev(staging_buf, kernel_arg.dst_ptr, dst_buf_size, 0));  

  // run tests
  std::cout << "run tests" << std::endl;
  RT_CHECK(run_test(kernel_arg, dst_buf_size, num_points));

  // cleanup
  std::cout << "cleanup" << std::endl;  
  cleanup();

  std::cout << "PASSED!" << std::endl;

  return 0;
}