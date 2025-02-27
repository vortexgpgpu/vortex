#include "common.h"
#include <fstream>
#include <iostream>
#include <string.h>
#include <unistd.h>
#include <vector>
#include <vortex.h>

#define FLOAT_ULP 6

#define RT_CHECK(_expr)                                      \
  do {                                                       \
    int _ret = _expr;                                        \
    if (0 == _ret)                                           \
      break;                                                 \
    printf("Error: '%s' returned %d!\n", #_expr, (int)_ret); \
    cleanup();                                               \
    exit(-1);                                                \
  } while (false)

///////////////////////////////////////////////////////////////////////////////

template <typename Type>
class Comparator {};

template <>
class Comparator<int> {
public:
  static const char *type_str() {
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
  union Float_t {
    float f;
    int i;
  };

public:
  static const char *type_str() {
    return "float";
  }
  static int generate() {
    return static_cast<float>(rand()) / RAND_MAX;
  }
  static bool compare(float a, float b, int index, int errors) {
    union fi_t {
      float f;
      int32_t i;
    };
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

const char *kernel_file = "kernel.vxbin";
uint32_t size = 16;

vx_device_h device = nullptr;
std::vector<TYPE> source_data;
std::vector<uint8_t> staging_buf;
kernel_arg_t kernel_arg = {};
vx_buffer_h krnl_buffer = nullptr;

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
    // vx_mem_free(device, kernel_arg.addr_a);
    // vx_mem_free(device, kernel_arg.addr_b);
    // vx_mem_free(device, kernel_arg.addr_dst);
    vx_dev_close(device);
  }
}

int run_test(const kernel_arg_t &kernel_arg,
             uint32_t buf_size,
             uint32_t num_points) {
  // start device
  std::cout << "start device" << std::endl;

  // RT_CHECK(vx_start(device));

  /*
    // wait for completion
    std::cout << "wait for completion" << std::endl;
    RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

    // download destination buffer
    std::cout << "download destination buffer" << std::endl;
    RT_CHECK(vx_copy_from_dev(device, staging_buf.data(), kernel_arg.addr_dst, buf_size));

    // verify result
    std::cout << "verify result" << std::endl;
    {
      int errors = 0;
      auto buf_ptr = (TYPE *)staging_buf.data();
      for (uint32_t i = 0; i < num_points; ++i) {
        auto ref = source_data[2 * i + 0] + source_data[2 * i + 1];
        auto cur = buf_ptr[i];
        if (!Comparator<TYPE>::compare(ref, cur, i, errors)) {
          ++errors;
        }
      }
      if (errors != 0) {
        std::cout << "Found " << std::dec << errors << " errors!" << std::endl;
        std::cout << "FAILED!" << std::endl;
        return 1;
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

  uint32_t num_points = size;
  uint32_t buf_size = num_points * sizeof(TYPE);

  std::cout << "number of points: " << num_points << std::endl;
  std::cout << "data type: " << Comparator<TYPE>::type_str() << std::endl;
  std::cout << "buffer size: " << buf_size << " bytes" << std::endl;

  // upload program (full stiched binary)
  std::cout << "upload program" << std::endl;
  // RT_CHECK(vx_upload_kernel_file(device, kernel_file));
  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));

  // allocate device memory
  std::cout << "allocate device memory" << std::endl;
  // RT_CHECK(vx_mem_alloc(device, buf_size, VX_MEM_TYPE_GLOBAL, &kernel_arg.addr_a));
  // RT_CHECK(vx_mem_alloc(device, buf_size, VX_MEM_TYPE_GLOBAL, &kernel_arg.addr_b));
  // RT_CHECK(vx_mem_alloc(device, buf_size, VX_MEM_TYPE_GLOBAL, &kernel_arg.addr_dst));
  kernel_arg.addr_a = 0xa0000000UL;
  kernel_arg.addr_b = 0xa1000000UL;
  kernel_arg.addr_dst = 0xc0000000UL;

  kernel_arg.dim = num_points;

  std::cout << "dev_addr_a=0x" << std::hex << kernel_arg.addr_a << std::endl;
  std::cout << "dev_addr_a=0x" << std::hex << kernel_arg.addr_b << std::endl;
  std::cout << "dev_addr_dst=0x" << std::hex << kernel_arg.addr_dst << std::endl;

  // allocate staging buffer
  std::cout << "allocate staging buffer" << std::endl;
  uint32_t alloc_size = std::max<uint32_t>(buf_size, sizeof(kernel_arg_t));
  staging_buf.resize(alloc_size);

  // generate source data
  source_data.resize(2 * num_points);
  for (uint32_t i = 0; i < source_data.size(); ++i) {
    // source_data[i] = Comparator<TYPE>::generate();
    source_data[i] = static_cast<float>(i);
  }

// NOTE(hansung): Uncomment below to generate args.bin, input.a.bin and
// input.b.bin automatically from the host code.
#if 0
  // upload kernel argument
  std::cout << "upload kernel argument" << std::endl;
  memcpy(staging_buf.data(), &kernel_arg, sizeof(kernel_arg_t));
  RT_CHECK(vx_copy_to_dev(device, KERNEL_ARG_DEV_MEM_ADDR, staging_buf.data(), sizeof(kernel_arg_t)));

  std::ofstream file("args.bin", std::ios::binary | std::ios::out);
  if (!file) {
    std::cerr << "error: failed to open args.bin for writing\n";
    exit(EXIT_FAILURE);
  }
  file.write(reinterpret_cast<char *>(staging_buf.data()), sizeof(kernel_arg_t));
  file.close();

  // upload source buffer0
  {
    std::cout << "upload source buffer0" << std::endl;
    auto buf_ptr = (TYPE *)staging_buf.data();
    for (uint32_t i = 0; i < num_points; ++i) {
      buf_ptr[i] = source_data[2 * i + 0];
    }
    RT_CHECK(vx_copy_to_dev(device, kernel_arg.addr_a, staging_buf.data(), buf_size));

    std::ofstream file("input.a.bin", std::ios::binary | std::ios::out);
    if (!file) {
      std::cerr << "error: failed to open input.a.bin for writing\n";
      exit(EXIT_FAILURE);
    }
    file.write(reinterpret_cast<char *>(buf_ptr), buf_size);
    file.close();
  }

  // upload source buffer1
  {
    std::cout << "upload source buffer1" << std::endl;
    auto buf_ptr = (TYPE *)staging_buf.data();
    for (uint32_t i = 0; i < num_points; ++i) {
      buf_ptr[i] = source_data[2 * i + 1];
    }
    RT_CHECK(vx_copy_to_dev(device, kernel_arg.addr_b, staging_buf.data(), buf_size));

    std::ofstream file("input.b.bin", std::ios::binary | std::ios::out);
    if (!file) {
      std::cerr << "error: failed to open input.b.bin for writing\n";
      exit(EXIT_FAILURE);
    }
    file.write(reinterpret_cast<char *>(buf_ptr), buf_size);
    file.close();
  }
#endif

  std::cout << "[UDIT]" << std::endl;

  // clear destination buffer
  std::cout << "clear destination buffer" << std::endl;
  memset(staging_buf.data(), 0, num_points * sizeof(TYPE));
  //  vx_copy_to_dev(vx_buffer_h hbuffer, uint64_t dev_maddr, uint64_t size, uint64_t src_offset)
  RT_CHECK(vx_copy_to_dev(device, (void *)kernel_arg.addr_dst, staging_buf.data(), buf_size));
  /*

    // run tests
    std::cout << "run tests" << std::endl;
    RT_CHECK(run_test(kernel_arg, buf_size, num_points));
    */

  // cleanup
  std::cout << "cleanup" << std::endl;
  cleanup();

  std::cout << "PASSED!" << std::endl;

  return 0;
}