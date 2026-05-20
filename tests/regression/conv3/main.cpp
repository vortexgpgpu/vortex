#include <iostream>
#include <unistd.h>
#include <string.h>
#include <vector>
#include <chrono>
#include <vortex2.h>
#include <cmath>
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
        printf("*** error: [%d] expected=%f, actual=%f\n", index, b, a);
      }
      return false;
    }
    return true;
  }
};

static void convolution_cpu(TYPE *O, TYPE *I, TYPE *W, int32_t width, int32_t height) {
  int paddedWidth = width + 2;
  for (int32_t y = 0; y < height; ++y) {
    for (int32_t x = 0; x < width; ++x) {
      int paddedY = y + 1;
      int paddedX = x + 1;
      TYPE sum(0);
      for (int32_t ky = -1; ky <= 1; ++ky) {
        for (int32_t kx = -1; kx <= 1; ++kx) {
          int32_t iy = paddedY + ky;
          int32_t ix = paddedX + kx;
          TYPE value = I[iy * paddedWidth + ix];
          TYPE weight = W[(ky + 1) * 3 + (kx + 1)];
          sum += value * weight;
        }
      }
      O[y * width + x] = sum;
    }
  }
}

const char* kernel_file = "kernel.vxbin";
int size = 32;
bool use_lmem = false;

vx_device_h device = nullptr;
vx_buffer_h I_buffer = nullptr;
vx_buffer_h W_buffer = nullptr;
vx_buffer_h O_buffer = nullptr;
vx_queue_h  queue   = nullptr;
vx_module_h module_ = nullptr;
vx_kernel_h kernel  = nullptr;
kernel_arg_t kernel_arg = {};

static void show_usage() {
   std::cout << "Vortex Test." << std::endl;
   std::cout << "Usage: [-k kernel] [-l: local memory] [-n size] [-h|?: help]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "n:k:lh")) != -1) {
    switch (c) {
    case 'n':
      size = atoi(optarg);
      break;
    case 'l':
      use_lmem = true;
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
    if (I_buffer) vx_buffer_release(I_buffer);
    if (W_buffer) vx_buffer_release(W_buffer);
    if (O_buffer) vx_buffer_release(O_buffer);
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

  std::cout << "data type: " << Comparator<TYPE>::type_str() << std::endl;
  std::cout << "matrix size: " << size << "x" << size << std::endl;

  kernel_arg.width = size;
  kernel_arg.use_lmem = use_lmem;

  uint32_t o_points = size * size;
  uint32_t i_points = (size+2) * (size+2);
  uint32_t w_points = 3 * 3;

  // allocate device memory
  std::cout << "allocate device memory" << std::endl;
  size_t i_nbytes = i_points * sizeof(TYPE);
  size_t w_nbytes = w_points * sizeof(TYPE);
  size_t o_nbytes = o_points * sizeof(TYPE);
  RT_CHECK(vx_buffer_create(device, i_nbytes, VX_MEM_READ, &I_buffer));
  RT_CHECK(vx_buffer_address(I_buffer, &kernel_arg.I_addr));
  RT_CHECK(vx_buffer_create(device, w_nbytes, VX_MEM_READ, &W_buffer));
  RT_CHECK(vx_buffer_address(W_buffer, &kernel_arg.W_addr));
  RT_CHECK(vx_buffer_create(device, o_nbytes, VX_MEM_WRITE, &O_buffer));
  RT_CHECK(vx_buffer_address(O_buffer, &kernel_arg.O_addr));

  if (use_lmem) {
    uint64_t dev_local_mem_size;
    RT_CHECK(vx_device_query(device, VX_CAPS_LOCAL_MEM_SIZE, &dev_local_mem_size));
    if (w_nbytes > dev_local_mem_size) {
      std::cout << "Error: Not enough local memory: needed=" << w_nbytes << ", available=" << dev_local_mem_size << std::endl;
      cleanup();
      exit(1);
    }
  }

  std::cout << "dev_argI=0x" << std::hex << kernel_arg.I_addr << std::endl;
  std::cout << "dev_argW=0x" << std::hex << kernel_arg.W_addr << std::endl;
  std::cout << "dev_argO=0x" << std::hex << kernel_arg.O_addr << std::endl;

  // Generate input values
  std::vector<TYPE> h_I(i_points);
  std::vector<TYPE> h_W(w_points);
  std::vector<TYPE> h_O(o_points);
  for (int32_t y = -1; y < size+1; ++y) {
    for (int32_t x = -1; x < size+1; ++x) {
      if (x >= 0 && x < size && y >= 0 && y < size) {
        h_I[(y+1) * (size+2) + (x+1)] = static_cast<TYPE>(rand()) / RAND_MAX;
      } else {
        h_I[(y+1) * (size+2) + (x+1)] = 0;
      }
    }
  }
  for (uint32_t i = 0; i < w_points; ++i) {
    h_W[i] = static_cast<TYPE>(rand()) / RAND_MAX;
  }

  // upload input buffer
  {
    std::cout << "upload source buffer" << std::endl;
    RT_CHECK(vx_enqueue_write(queue, I_buffer, 0, h_I.data(), i_nbytes, 0, nullptr, nullptr));
  }

  // upload weight buffer
  {
    std::cout << "upload weight buffer" << std::endl;
    RT_CHECK(vx_enqueue_write(queue, W_buffer, 0, h_W.data(), w_nbytes, 0, nullptr, nullptr));
  }

  // load kernel module
  std::cout << "load kernel module" << std::endl;
  RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
  RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

  auto time_start = std::chrono::high_resolution_clock::now();

  // start device
  std::cout << "start device" << std::endl;
  vx_event_h launch_ev = nullptr, read_ev = nullptr;
  {
    uint64_t num_threads;
    RT_CHECK(vx_device_query(device, VX_CAPS_NUM_THREADS, &num_threads));
    uint32_t NT = (uint32_t)num_threads;
    uint32_t lmem_size = use_lmem ? (uint32_t)w_nbytes : 0;
    uint32_t grid_dim[2]  = {(size + NT - 1) / NT, (uint32_t)size};
    uint32_t block_dim[2] = {NT, 1};
    vx_launch_info_t li = {};
    li.struct_size  = sizeof(li);
    li.kernel       = kernel;
    li.args_host    = &kernel_arg;
    li.args_size    = sizeof(kernel_arg);
    li.ndim         = 2;
    li.grid_dim[0]  = grid_dim[0];
    li.grid_dim[1]  = grid_dim[1];
    li.block_dim[0] = block_dim[0];
    li.block_dim[1] = block_dim[1];
    li.lmem_size    = lmem_size;
    RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &launch_ev));
  }

  // download destination buffer
  std::cout << "download destination buffer" << std::endl;
  RT_CHECK(vx_enqueue_read(queue, h_O.data(), O_buffer, 0, o_nbytes, 1, &launch_ev, &read_ev));

  // wait for completion
  std::cout << "wait for completion" << std::endl;
  RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(read_ev);
  vx_event_release(launch_ev);

  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  // verify result
  std::cout << "verify result" << std::endl;
  int errors = 0;
  {
    std::vector<TYPE> h_ref(o_points);
    convolution_cpu(h_ref.data(), h_I.data(), h_W.data(), size, size);

    for (uint32_t i = 0; i < h_ref.size(); ++i) {
      auto ref = h_ref[i];
      auto cur = h_O[i];
      if (!Comparator<TYPE>::compare(cur, ref, i, errors)) {
        ++errors;
      }
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