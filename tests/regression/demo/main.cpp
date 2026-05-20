#include <iostream>
#include <unistd.h>
#include <string.h>
#include <vector>
#include <vortex2.h>
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
        printf("*** error: [%d] expected=%d, actual=%d\n", index, b, a);
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
  static float generate() {
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
        printf("*** error: [%d] expected=%f(0x%x), actual=%f(0x%x), ulp=%d\n", index, b, fb.i, a, fa.i, d);
      }
      return false;
    }
    return true;
  }
};

const char* kernel_file = "kernel.vxbin";
uint32_t count = 16;
uint32_t usr_block_x = 0;
uint32_t usr_block_y = 0;

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
   std::cout << "Usage: [-k: kernel] [-n words] [-x: block_dim_x] [-y: block_dim_y] [-h: help]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "n:k:x:y:h")) != -1) {
    switch (c) {
    case 'n':
      count = atoi(optarg);
      break;
    case 'k':
      kernel_file = optarg;
      break;
    case 'x':
      usr_block_x = atoi(optarg);
      break;
    case 'y':
      usr_block_y = atoi(optarg);
      break;
    case 'h':{
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
    if (src0_buffer) vx_buffer_release(src0_buffer);
    if (src1_buffer) vx_buffer_release(src1_buffer);
    if (dst_buffer)  vx_buffer_release(dst_buffer);
    if (kernel)  vx_kernel_release(kernel);
    if (module_) vx_module_release(module_);
    if (queue)   vx_queue_release(queue);
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

  uint64_t num_cores, num_warps, num_threads;
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_CORES, &num_cores));
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_WARPS, &num_warps));
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_THREADS, &num_threads));

  uint32_t total_threads = num_cores * num_warps * num_threads;

  // determine 2D launch dimensions
  uint32_t block_dim_x = (usr_block_x != 0) ? usr_block_x : num_threads;
  uint32_t block_dim_y = (usr_block_y != 0) ? usr_block_y : 1;
  uint32_t threads_per_block = block_dim_x * block_dim_y;
  uint32_t num_blocks = (total_threads + threads_per_block - 1) / threads_per_block;
  uint32_t num_tasks = num_blocks * threads_per_block;
  uint32_t dim_x = num_blocks * block_dim_x;
  uint32_t num_points = count * num_tasks;
  uint32_t buf_size = num_points * sizeof(TYPE);

  kernel_arg.num_tasks = num_tasks;
  kernel_arg.task_size = count;
  kernel_arg.dim_x = dim_x;

  std::cout << "data type: " << Comparator<TYPE>::type_str() << std::endl;
  std::cout << "number of points: " << num_points << std::endl;
  std::cout << "buffer size: " << buf_size << " bytes" << std::endl;

  // allocate device memory
  std::cout << "allocate device memory" << std::endl;
  RT_CHECK(vx_buffer_create(device, buf_size, VX_MEM_READ, &src0_buffer));
  RT_CHECK(vx_buffer_address(src0_buffer, &kernel_arg.src0_addr));
  RT_CHECK(vx_buffer_create(device, buf_size, VX_MEM_READ, &src1_buffer));
  RT_CHECK(vx_buffer_address(src1_buffer, &kernel_arg.src1_addr));
  RT_CHECK(vx_buffer_create(device, buf_size, VX_MEM_WRITE, &dst_buffer));
  RT_CHECK(vx_buffer_address(dst_buffer, &kernel_arg.dst_addr));

  std::cout << "dev_src0=0x" << std::hex << kernel_arg.src0_addr << std::endl;
  std::cout << "dev_src1=0x" << std::hex << kernel_arg.src1_addr << std::endl;
  std::cout << "dev_dst=0x" << std::hex << kernel_arg.dst_addr << std::endl;

  // allocate host buffers
  std::cout << "allocate host buffers" << std::endl;
  std::vector<TYPE> h_src0(num_points);
  std::vector<TYPE> h_src1(num_points);
  std::vector<TYPE> h_dst(num_points);

  // generate source data
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
    uint32_t grid[2] = {num_blocks, 1};
    uint32_t block[2] = {block_dim_x, block_dim_y};
    std::cout << "grid_dim=" << grid[0] << "x" << grid[1]
              << ", block_dim=" << block[0] << "x" << block[1]
              << std::endl;
    vx_launch_info_t li = {};
    li.struct_size  = sizeof(li);
    li.kernel       = kernel;
    li.args_host    = &kernel_arg;
    li.args_size    = sizeof(kernel_arg);
    li.ndim         = 2;
    li.grid_dim[0]  = grid[0];
    li.grid_dim[1]  = grid[1];
    li.block_dim[0] = block[0];
    li.block_dim[1] = block[1];
    RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &launch_ev));
  }

  // download destination buffer — chained after the launch
  std::cout << "download destination buffer" << std::endl;
  RT_CHECK(vx_enqueue_read(queue, h_dst.data(), dst_buffer, 0, buf_size, 1, &launch_ev, &read_ev));

  // wait for completion
  std::cout << "wait for completion" << std::endl;
  RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(read_ev);
  vx_event_release(launch_ev);

  // verify result
  std::cout << "verify result" << std::endl;
  int errors = 0;
  for (uint32_t i = 0; i < num_points; ++i) {
    auto ref = h_src0[i] + h_src1[i];
    auto cur = h_dst[i];
    if (!Comparator<TYPE>::compare(cur, ref, i, errors)) {
      ++errors;
    }
  }

  // cleanup
  std::cout << "cleanup" << std::endl;
  cleanup();

  if (errors != 0) {
    std::cout << "Found " << std::dec << errors << " errors!" << std::endl;
    std::cout << "FAILED!" << std::endl;
    return errors;
  }

  std::cout << "PASSED!" << std::endl;

  return 0;
}
