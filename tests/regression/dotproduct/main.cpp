#include <cmath>
#include <iostream>
#include <unistd.h>
#include <string.h>
#include <vector>
#include <vortex2.h>
#include "common.h"

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
  // Integer sums wrap modulo 2^32 on the device, so they are exact in any
  // order; the reference wraps the same way, in unsigned arithmetic.
  static bool verify(const std::vector<int>& a, const std::vector<int>& b,
                     const std::vector<int>& partials, uint32_t /*depth*/) {
    uint32_t ref = 0, cur = 0;
    for (size_t i = 0; i < a.size(); ++i) {
      ref += uint32_t(a[i]) * uint32_t(b[i]);
    }
    for (int p : partials) {
      cur += uint32_t(p);
    }
    if (cur != ref) {
      printf("*** error: expected=%d, actual=%d\n", int(ref), int(cur));
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
  static float generate() {
    return static_cast<float>(rand()) / RAND_MAX;
  }
  // The device rounds through at most `depth` float operations on the way from
  // each product to its block sum, so the forward error bound is
  // gamma_depth * sum|a*b| with gamma_k = k*u/(1 - k*u), u = 2^-24 (Higham,
  // summation in any order). The reference and the sum of the block partials are
  // computed in double, so the bound holds for every n.
  static bool verify(const std::vector<float>& a, const std::vector<float>& b,
                     const std::vector<float>& partials, uint32_t depth) {
    double ref = 0, abs_sum = 0, cur = 0;
    for (size_t i = 0; i < a.size(); ++i) {
      double p = double(a[i]) * double(b[i]);
      ref += p;
      abs_sum += std::abs(p);
    }
    for (float p : partials) {
      cur += p;
    }
    double ku = depth * std::ldexp(1.0, -24);
    double tol = ku / (1 - ku) * abs_sum;
    if (std::abs(cur - ref) > tol) {
      printf("*** error: expected=%f, actual=%f, tolerance=%g\n", ref, cur, tol);
      return false;
    }
    return true;
  }
};

const char* kernel_file = "kernel.vxbin";
uint32_t size = 16;

vx_device_h device = nullptr;
vx_buffer_h src0_buffer = nullptr;
vx_buffer_h src1_buffer = nullptr;
vx_buffer_h dst_buffer = nullptr;
vx_queue_h  queue = nullptr;
vx_module_h module_ = nullptr;
vx_kernel_h kernel = nullptr;
kernel_arg_t kernel_arg = {};

static void show_usage() {
   std::cout << "Vortex Test." << std::endl;
   std::cout << "Usage: [-k: kernel] [-n words] [-h: help]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "n:k:h")) != -1) {
    switch (c) {
    case 'n':
      size = atoi(optarg);
      break;
    case 'k':
      kernel_file = optarg;
      break;
    case 'h':
      show_usage();
      exit(0);
      break;
    default:
      show_usage();
      exit(-1);
    }
  }
}

void cleanup() {
  if (device) {
    if (src0_buffer) vx_buffer_release(src0_buffer);
    if (src1_buffer) vx_buffer_release(src1_buffer);
    if (dst_buffer)  vx_buffer_release(dst_buffer);
    if (kernel)  vx_kernel_release(kernel);
    if (module_) vx_module_release(module_);
    if (queue)   vx_queue_release(queue);
    vx_device_dump_perf(device, stdout);
    vx_device_release(device);
  }
}

int main(int argc, char *argv[]) {
  // parse command arguments
  parse_args(argc, argv);

  std::srand(50);

  // open device connection
  std::cout << "open device connection" << std::endl;
  RT_CHECK(vx_device_open(0, &device));

  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));

  uint32_t num_points = size;
  uint32_t buf_size = num_points * sizeof(TYPE);

  std::cout << "number of points: " << num_points << std::endl;
  std::cout << "data type: " << Comparator<TYPE>::type_str() << std::endl;
  std::cout << "buffer size: " << buf_size << " bytes" << std::endl;

  kernel_arg.num_points = num_points;

  uint32_t grid_dim[1], block_dim[1];
  RT_CHECK(vx_device_max_occupancy_grid(device, 1, &num_points, grid_dim, block_dim));
  uint32_t blocksPerGrid = grid_dim[0];
  uint32_t threadsPerBlock = block_dim[0];

  std::cout << "threads per block: " << threadsPerBlock << std::endl;
  std::cout << "blocks per grid: " << blocksPerGrid << std::endl;

  uint32_t dst_buf_size = blocksPerGrid * sizeof(TYPE);
  // allocate device memory
  std::cout << "allocate device memory" << std::endl;
  RT_CHECK(vx_buffer_create(device, buf_size, VX_MEM_READ, &src0_buffer));
  RT_CHECK(vx_buffer_address(src0_buffer, &kernel_arg.src0_addr));
  RT_CHECK(vx_buffer_create(device, buf_size, VX_MEM_READ, &src1_buffer));
  RT_CHECK(vx_buffer_address(src1_buffer, &kernel_arg.src1_addr));
  RT_CHECK(vx_buffer_create(device, dst_buf_size, VX_MEM_WRITE, &dst_buffer));
  RT_CHECK(vx_buffer_address(dst_buffer, &kernel_arg.dst_addr));

  std::cout << "dev_src0=0x" << std::hex << kernel_arg.src0_addr << std::endl;
  std::cout << "dev_src1=0x" << std::hex << kernel_arg.src1_addr << std::endl;
  std::cout << "dev_dst=0x" << std::hex << kernel_arg.dst_addr << std::endl;

  // allocate host buffers
  std::cout << "allocate host buffers" << std::endl;
  std::vector<TYPE> h_src0(num_points);
  std::vector<TYPE> h_src1(num_points);
  std::vector<TYPE> h_dst(blocksPerGrid);

  for (uint32_t i = 0; i < num_points; ++i) {
    h_src0[i] = Comparator<TYPE>::generate();
    h_src1[i] = Comparator<TYPE>::generate();
  }

  // upload source buffer0
  std::cout << "upload source buffer0" << std::endl;
  RT_CHECK(vx_enqueue_write(queue, src0_buffer, 0, h_src0.data(), buf_size, 0, nullptr, nullptr));

  // upload source buffer1
  std::cout << "upload source buffer1" << std::endl;
  RT_CHECK(vx_enqueue_write(queue, src1_buffer, 0, h_src1.data(), buf_size, 0, nullptr, nullptr));

  // load kernel module
  std::cout << "load kernel module" << std::endl;
  RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
  RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

  // launch kernel — args passed as a host blob (UVA), no args device buffer
  std::cout << "launch kernel" << std::endl;
  vx_event_h launch_ev = nullptr, read_ev = nullptr;
  {
    vx_launch_info_t li = {};
    li.struct_size  = sizeof(li);
    li.kernel       = kernel;
    li.args_host    = &kernel_arg;
    li.args_size    = sizeof(kernel_arg);
    li.ndim         = 1;
    li.grid_dim[0]  = grid_dim[0];
    li.block_dim[0] = block_dim[0];
    li.lmem_size    = threadsPerBlock * sizeof(TYPE);
    RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &launch_ev));
  }

  // download destination buffer — chained after the launch
  std::cout << "download destination buffer" << std::endl;
  RT_CHECK(vx_enqueue_read(queue, h_dst.data(), dst_buffer, 0, dst_buf_size, 1, &launch_ev, &read_ev));

  // wait for completion
  std::cout << "wait for completion" << std::endl;
  RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(read_ev);
  vx_event_release(launch_ev);

  // verify result
  std::cout << "verify result" << std::endl;

  // Per element: one product, ceil(n/stride) grid-stride accumulations, then
  // log2(threadsPerBlock) tree levels.
  uint32_t stride = threadsPerBlock * blocksPerGrid;
  uint32_t depth = 1 + (num_points + stride - 1) / stride;
  for (uint32_t t = 1; t < threadsPerBlock; t *= 2) {
    ++depth;
  }
  int errors = 0;
  if (!Comparator<TYPE>::verify(h_src0, h_src1, h_dst, depth)) {
    ++errors;
  }

  // cleanup
  std::cout << "cleanup" << std::endl;
  cleanup();

  if (errors != 0) {
    std::cout << "Found " << std::dec << errors << " errors!" << std::endl;
    std::cout << "FAILED!" << std::endl;
    return 1;
  }

  std::cout << "PASSED!" << std::endl;

  return 0;
}