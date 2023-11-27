#include <iostream>
#include <unistd.h>
#include <string.h>
#include <vector>
#include <chrono>
#include <vortex.h>
#include "common.h"

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

static void matmul_cpu(TYPE* out, const TYPE* A, const TYPE* B, uint32_t width, uint32_t height) {
  for (uint32_t row = 0; row < height; ++row) {
    for (uint32_t col = 0; col < width; ++col) {
      TYPE sum(0);
      for (uint32_t e = 0; e < width; ++e) {
          sum += A[row * width + e] * B[e * width + col];
      }
      out[row * width + col] = sum;
    }
  }
}

const char* kernel_file = "kernel.bin";
uint32_t size = 32;

vx_device_h device = nullptr;
std::vector<uint8_t> staging_buf;
kernel_arg_t kernel_arg = {};

static void show_usage() {
   std::cout << "Vortex Test." << std::endl;
   std::cout << "Usage: [-k: kernel] [-n size] [-h: help]" << std::endl;
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
    vx_mem_free(device, kernel_arg.A_addr);
    vx_mem_free(device, kernel_arg.B_addr);
    vx_mem_free(device, kernel_arg.C_addr);
    vx_dev_close(device);
  }
}

int main(int argc, char *argv[]) {  
  // parse command arguments
  parse_args(argc, argv);

  std::srand(50);

  // open device connection
  std::cout << "open device connection" << std::endl;  
  RT_CHECK(vx_dev_open(&device));

  uint32_t num_points = size * size;
  uint32_t buf_size = num_points * sizeof(TYPE);

  std::cout << "data type: " << Comparator<TYPE>::type_str() << std::endl;
  std::cout << "matrix size: " << size << "x" << size << std::endl;
  std::cout << "buffer size: " << buf_size << " bytes" << std::endl;

  // upload program
  std::cout << "upload program" << std::endl;  
  RT_CHECK(vx_upload_kernel_file(device, kernel_file));

  // allocate device memory
  std::cout << "allocate device memory" << std::endl;
  RT_CHECK(vx_mem_alloc(device, buf_size, VX_MEM_TYPE_GLOBAL, &kernel_arg.A_addr));
  RT_CHECK(vx_mem_alloc(device, buf_size, VX_MEM_TYPE_GLOBAL, &kernel_arg.B_addr));
  RT_CHECK(vx_mem_alloc(device, buf_size, VX_MEM_TYPE_GLOBAL, &kernel_arg.C_addr));

  kernel_arg.num_tasks = num_points;
  kernel_arg.size = size;

  std::cout << "dev_src0=0x" << std::hex << kernel_arg.A_addr << std::endl;
  std::cout << "dev_src1=0x" << std::hex << kernel_arg.B_addr << std::endl;
  std::cout << "dev_dst=0x" << std::hex << kernel_arg.C_addr << std::endl;
  
  // allocate staging buffer  
  std::cout << "allocate staging buffer" << std::endl;    
  uint32_t alloc_size = std::max<uint32_t>(buf_size, sizeof(kernel_arg_t));
  staging_buf.resize(alloc_size);
  
  // upload kernel argument
  std::cout << "upload kernel argument" << std::endl;
  memcpy(staging_buf.data(), &kernel_arg, sizeof(kernel_arg_t));
  RT_CHECK(vx_copy_to_dev(device, KERNEL_ARG_DEV_MEM_ADDR, staging_buf.data(), sizeof(kernel_arg_t)));

  // generate source data
  std::vector<TYPE> src_A(num_points);
  std::vector<TYPE> src_B(num_points);
  std::vector<TYPE> refs(num_points);
  for (uint32_t i = 0; i < num_points; ++i) {
    auto a = static_cast<float>(std::rand()) / RAND_MAX;
    auto b = static_cast<float>(std::rand()) / RAND_MAX;
    src_A[i] = static_cast<TYPE>(a * size);
    src_B[i] = static_cast<TYPE>(b * size);
  }
  matmul_cpu(refs.data(), src_A.data(), src_B.data(), size, size);

  // upload source buffer0
  {
    std::cout << "upload source buffer0" << std::endl;
    auto buf_ptr = (TYPE*)staging_buf.data();
    for (uint32_t i = 0; i < num_points; ++i) {
      buf_ptr[i] = src_A[i];
    }
    RT_CHECK(vx_copy_to_dev(device, kernel_arg.A_addr, staging_buf.data(), buf_size));
  }

  // upload source buffer1
  {
    std::cout << "upload source buffer1" << std::endl;
    auto buf_ptr = (TYPE*)staging_buf.data();
    for (uint32_t i = 0; i < num_points; ++i) {
      buf_ptr[i] = src_B[i];
    }   
    RT_CHECK(vx_copy_to_dev(device, kernel_arg.B_addr, staging_buf.data(), buf_size));
  }

  // clear destination buffer
  std::cout << "clear destination buffer" << std::endl;
  memset(staging_buf.data(), 0, num_points * sizeof(TYPE));
  RT_CHECK(vx_copy_to_dev(device, kernel_arg.C_addr, staging_buf.data(), buf_size));  

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

  // download destination buffer
  std::cout << "download destination buffer" << std::endl;
  RT_CHECK(vx_copy_from_dev(device, staging_buf.data(), kernel_arg.C_addr, buf_size));

  // verify result
  std::cout << "verify result" << std::endl;  
  {
    int errors = 0;
    auto buf_ptr = (TYPE*)staging_buf.data();
    for (uint32_t i = 0; i < refs.size(); ++i) {
      auto ref = refs[i];
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

  // cleanup
  std::cout << "cleanup" << std::endl;  
  cleanup();

  std::cout << "PASSED!" << std::endl;

  return 0;
}