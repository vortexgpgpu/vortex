#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <vector>
#include <vortex2.h>
#include "common.h"

#define RT_CHECK(_expr)                                         \
   do {                                                         \
     int _ret = _expr;                                          \
     if (0 == _ret)                                             \
       break;                                                   \
     printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);   \
     cleanup();                                                 \
     exit(-1);                                                  \
   } while (false)

const char* kernel_file = "kernel.vxbin";

vx_device_h device = nullptr;
vx_buffer_h p0_buffer = nullptr;
vx_buffer_h p1_buffer = nullptr;
vx_queue_h  queue   = nullptr;
vx_module_h module_ = nullptr;
vx_kernel_h kernel  = nullptr;
kernel_arg_t kernel_arg = {};

static void show_usage() {
  std::cout << "Vortex Test." << std::endl;
  std::cout << "Usage: [-k: kernel] [-h: help]" << std::endl;
}

static void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "k:h")) != -1) {
    switch (c) {
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
    if (p0_buffer) vx_buffer_release(p0_buffer);
    if (p1_buffer) vx_buffer_release(p1_buffer);
    if (kernel)  vx_kernel_release(kernel);
    if (module_) vx_module_release(module_);
    if (queue)   vx_queue_release(queue);
    vx_device_dump_perf(device, stdout);
    vx_device_release(device);
  }
}

int main(int argc, char* argv[]) {
  parse_args(argc, argv);

  std::cout << "open device connection" << std::endl;
  RT_CHECK(vx_device_open(0, &device));

  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));

  uint64_t num_cores, num_warps, num_threads;
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_CORES, &num_cores));
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_WARPS, &num_warps));
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_THREADS, &num_threads));

  if (num_cores < 2) {
    std::cout << "Device does not have enough cores to run the test (need at least 2)" << std::endl;
    cleanup();
    return -1;
  }

  kernel_arg.num_cores  = static_cast<uint32_t>(num_cores);
  kernel_arg.num_groups = static_cast<uint32_t>(num_cores * num_warps);
  kernel_arg.group_size = static_cast<uint32_t>(num_threads);

  uint32_t buf_size = kernel_arg.num_cores * sizeof(uint32_t);

  std::cout << "num_cores=" << num_cores
            << ", num_warps=" << num_warps
            << ", num_threads=" << num_threads << std::endl;
  std::cout << "num_groups=" << kernel_arg.num_groups << std::endl;

  std::cout << "allocate device memory" << std::endl;
  RT_CHECK(vx_buffer_create(device, buf_size, VX_MEM_READ_WRITE, &p0_buffer));
  RT_CHECK(vx_buffer_address(p0_buffer, &kernel_arg.p0_addr));
  RT_CHECK(vx_buffer_create(device, buf_size, VX_MEM_READ_WRITE, &p1_buffer));
  RT_CHECK(vx_buffer_address(p1_buffer, &kernel_arg.p1_addr));

  // 0xff is not a legal 1-bit phase, so a slot the kernel never wrote is
  // distinguishable from a phase of 0.
  std::vector<uint32_t> h_p0(kernel_arg.num_cores, 0xff);
  std::vector<uint32_t> h_p1(kernel_arg.num_cores, 0xff);

  RT_CHECK(vx_enqueue_write(queue, p0_buffer, 0, h_p0.data(), buf_size, 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_write(queue, p1_buffer, 0, h_p1.data(), buf_size, 0, nullptr, nullptr));

  std::cout << "load kernel module" << std::endl;
  RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
  RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

  std::cout << "start device" << std::endl;
  vx_event_h launch_ev = nullptr;
  {
    uint32_t grid_dim[1]  = {kernel_arg.num_groups};
    uint32_t block_dim[1] = {kernel_arg.group_size};
    vx_launch_info_t li = {};
    li.struct_size  = sizeof(li);
    li.kernel       = kernel;
    li.args_host    = &kernel_arg;
    li.args_size    = sizeof(kernel_arg);
    li.ndim         = 1;
    li.grid_dim[0]  = grid_dim[0];
    li.block_dim[0] = block_dim[0];
    RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &launch_ev));
  }

  std::cout << "download results" << std::endl;
  vx_event_h read_ev0 = nullptr, read_ev1 = nullptr;
  RT_CHECK(vx_enqueue_read(queue, h_p0.data(), p0_buffer, 0, buf_size, 1, &launch_ev, &read_ev0));
  RT_CHECK(vx_enqueue_read(queue, h_p1.data(), p1_buffer, 0, buf_size, 1, &read_ev0, &read_ev1));

  std::cout << "wait for completion" << std::endl;
  RT_CHECK(vx_event_wait_value(read_ev1, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(read_ev1);
  vx_event_release(read_ev0);
  vx_event_release(launch_ev);

  // A global barrier is a per-CLUSTER object, so the phase is reported per
  // core: a per-core table is what makes the failure legible (the flip may
  // land on one core's slot by coincidence while every other core is stuck).
  int errors = 0;
  std::cout << "\ncore  phase@N  phase@N+1  advanced" << std::endl;
  for (uint32_t i = 0; i < kernel_arg.num_cores; ++i) {
    bool advanced = (h_p0[i] != h_p1[i]);
    std::cout << "  " << i
              << "        " << h_p0[i]
              << "          " << h_p1[i]
              << "        " << (advanced ? "yes" : "NO") << std::endl;
    if (h_p0[i] == 0xff || h_p1[i] == 0xff) {
      std::cout << "    core " << i << ": kernel never wrote a phase" << std::endl;
      ++errors;
    } else if (!advanced) {
      // vx_barrier.h: arrive() returns "phase (current generation number)".
      // A completed generation must therefore change it.
      std::cout << "    core " << i << ": PHASE error: a completed global-barrier"
                << " generation did not advance the phase (" << h_p0[i]
                << " -> " << h_p1[i] << ")" << std::endl;
      ++errors;
    }
  }
  std::cout << "phase errors: " << errors << std::endl;

  std::cout << "cleanup" << std::endl;
  cleanup();

  if (errors != 0) {
    std::cout << "Found " << errors << " errors!" << std::endl;
    std::cout << "FAILED!" << std::endl;
    return errors;
  }

  std::cout << "PASSED!" << std::endl;
  return 0;
}
