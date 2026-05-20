#include <iostream>
#include <unistd.h>
#include <string.h>
#include <vortex2.h>
#include <chrono>
#include <vector>
#include "common.h"

#define NONCE  0xdeadbeef

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

const char* kernel_file = "kernel.vxbin";
int test = -1;
uint32_t count = 0;

vx_device_h device = nullptr;
vx_buffer_h src_buffer = nullptr;
vx_buffer_h dst_buffer = nullptr;
vx_queue_h  queue   = nullptr;
vx_module_h module_ = nullptr;
vx_kernel_h kernel  = nullptr;
kernel_arg_t kernel_arg = {};
uint64_t num_cores = 0, num_threads = 0;

static void show_usage() {
   std::cout << "Vortex Test." << std::endl;
   std::cout << "Usage: [-t testno][-k: kernel][-n words][-h: help]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "n:t:k:h")) != -1) {
    switch (c) {
    case 'n':
      count = atoi(optarg);
      break;
    case 't':
      test = atoi(optarg);
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
    if (src_buffer) vx_buffer_release(src_buffer);
    if (dst_buffer) vx_buffer_release(dst_buffer);
    if (kernel)  vx_kernel_release(kernel);
    if (module_) vx_module_release(module_);
    if (queue)   vx_queue_release(queue);
    vx_device_release(device);
  }
}

inline uint32_t shuffle(int i, uint32_t value) {
  return (value << i) | (value & ((1 << i)-1));;
}

int run_memcopy_test(const kernel_arg_t& kernel_arg) {
  uint32_t num_points = kernel_arg.count;
  uint32_t buf_size = num_points * sizeof(int32_t);

  std::vector<uint32_t> h_src(num_points);
  std::vector<uint32_t> h_dst(num_points);

  // update source buffer
  for (uint32_t i = 0; i < num_points; ++i) {
    h_src[i] = shuffle(i, NONCE);
  }

  auto time_start = std::chrono::high_resolution_clock::now();

  // upload source buffer
  std::cout << "write source buffer to device memory" << std::endl;
  auto t0 = std::chrono::high_resolution_clock::now();
  RT_CHECK(vx_enqueue_write(queue, dst_buffer, 0, h_src.data(), buf_size, 0, nullptr, nullptr));
  auto t1 = std::chrono::high_resolution_clock::now();

  // download destination buffer
  std::cout << "read destination buffer from device memory" << std::endl;
  auto t2 = std::chrono::high_resolution_clock::now();
  vx_event_h mc_read_ev = nullptr;
  RT_CHECK(vx_enqueue_read(queue, h_dst.data(), dst_buffer, 0, buf_size, 0, nullptr, &mc_read_ev));
  RT_CHECK(vx_event_wait_value(mc_read_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(mc_read_ev);
  auto t3 = std::chrono::high_resolution_clock::now();

  // verify result
  int errors = 0;
  std::cout << "verify result" << std::endl;
  for (uint32_t i = 0; i < num_points; ++i) {
    auto cur = h_dst[i];
    auto ref = shuffle(i, NONCE);
    if (cur != ref) {
      printf("*** error: [%d] expected=%d, actual=%d\n", i, ref, cur);
      ++errors;
    }
  }

  auto time_end = std::chrono::high_resolution_clock::now();

  double elapsed;
  elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
  printf("upload time: %lg ms\n", elapsed);
  elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();
  printf("download time: %lg ms\n", elapsed);
  elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Total elapsed time: %lg ms\n", elapsed);

  return errors;
}

int run_kernel_test(const kernel_arg_t& kernel_arg) {
  uint32_t num_points = kernel_arg.count;
  uint32_t buf_size = num_points * sizeof(int32_t);

  std::vector<uint32_t> h_src(num_points);
  std::vector<uint32_t> h_dst(num_points);

  // update source buffer
  for (uint32_t i = 0; i < num_points; ++i) {
    h_src[i] = shuffle(i, NONCE);
  }

  // load kernel module
  std::cout << "load kernel module" << std::endl;
  RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
  RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

  auto time_start = std::chrono::high_resolution_clock::now();

  // upload source buffer
  auto t0 = std::chrono::high_resolution_clock::now();
  RT_CHECK(vx_enqueue_write(queue, src_buffer, 0, h_src.data(), buf_size, 0, nullptr, nullptr));
  auto t1 = std::chrono::high_resolution_clock::now();

  // start execution — legacy vx_start maps to grid=num_cores, block=num_threads
  std::cout << "start execution" << std::endl;
  auto t2 = std::chrono::high_resolution_clock::now();
  vx_event_h launch_ev = nullptr, read_ev = nullptr;
  {
    vx_launch_info_t li = {};
    li.struct_size  = sizeof(li);
    li.kernel       = kernel;
    li.args_host    = &kernel_arg;
    li.args_size    = sizeof(kernel_arg);
    li.ndim         = 1;
    li.grid_dim[0]  = (uint32_t)num_cores;
    li.block_dim[0] = (uint32_t)num_threads;
    RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &launch_ev));
  }
  RT_CHECK(vx_event_wait_value(launch_ev, 1, VX_TIMEOUT_INFINITE));
  auto t3 = std::chrono::high_resolution_clock::now();

  // download destination buffer
  std::cout << "read destination buffer from device memory" << std::endl;
  auto t4 = std::chrono::high_resolution_clock::now();
  RT_CHECK(vx_enqueue_read(queue, h_dst.data(), dst_buffer, 0, buf_size, 1, &launch_ev, &read_ev));
  RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(read_ev);
  vx_event_release(launch_ev);
  auto t5 = std::chrono::high_resolution_clock::now();

  // verify result
  int errors = 0;
  std::cout << "verify result" << std::endl;
  for (uint32_t i = 0; i < num_points; ++i) {
    auto cur = h_dst[i];
    auto ref = shuffle(i, NONCE);
    if (cur != ref) {
      printf("*** error: [%d] expected=%d, actual=%d\n", i, ref, cur);
      ++errors;
    }
  }

  auto time_end = std::chrono::high_resolution_clock::now();

  double elapsed;
  elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
  printf("upload time: %lg ms\n", elapsed);
  elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();
  printf("execute time: %lg ms\n", elapsed);
  elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t5 - t4).count();
  printf("download time: %lg ms\n", elapsed);
  elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Total elapsed time: %lg ms\n", elapsed);

  return errors;
}

int main(int argc, char *argv[]) {
  // parse command arguments
  parse_args(argc, argv);

  if (count == 0) {
    count = 1;
  }

  // open device connection
  std::cout << "open device connection" << std::endl;
  RT_CHECK(vx_device_open(0, &device));

  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));

  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_CORES, &num_cores));
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_THREADS, &num_threads));

  uint32_t num_points = count * num_cores;
  uint32_t buf_size = num_points * sizeof(int32_t);

  std::cout << "number of points: " << num_points << std::endl;
  std::cout << "buffer size: " << buf_size << " bytes" << std::endl;

  // allocate device memory
  std::cout << "allocate device memory" << std::endl;
  RT_CHECK(vx_buffer_create(device, buf_size, VX_MEM_READ, &src_buffer));
  RT_CHECK(vx_buffer_address(src_buffer, &kernel_arg.src_addr));
  RT_CHECK(vx_buffer_create(device, buf_size, VX_MEM_WRITE, &dst_buffer));
  RT_CHECK(vx_buffer_address(dst_buffer, &kernel_arg.dst_addr));

  kernel_arg.count = count;

  std::cout << "dev_src=0x" << std::hex << kernel_arg.src_addr << std::endl;
  std::cout << "dev_dst=0x" << std::hex << kernel_arg.dst_addr << std::endl;

  int errors = 0;

  // run tests
  if (0 == test || -1 == test) {
    std::cout << "run memcopy test" << std::endl;
    errors = run_memcopy_test(kernel_arg);
  }

  if (1 == test || -1 == test) {
    std::cout << "run kernel test" << std::endl;
    errors = run_kernel_test(kernel_arg);
  }

  // cleanup
  std::cout << "cleanup" << std::endl;
  cleanup();

  if (errors != 0) {
    std::cout << "Found " << std::dec << errors << " errors!" << std::endl;
    std::cout << "FAILED!" << std::endl;
    return errors;
  }

  std::cout << "Test PASSED" << std::endl;

  return 0;
}