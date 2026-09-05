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
vx_buffer_h err_buffer = nullptr;
vx_buffer_h pre_buffer = nullptr;
vx_buffer_h post_buffer = nullptr;
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
    case 'k': kernel_file = optarg; break;
    case 'h': show_usage(); exit(0); break;
    default:  show_usage(); exit(-1);
    }
  }
}

void cleanup() {
  if (device) {
    if (err_buffer)  vx_buffer_release(err_buffer);
    if (pre_buffer)  vx_buffer_release(pre_buffer);
    if (post_buffer) vx_buffer_release(post_buffer);
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

  if (num_warps < 2) {
    std::cout << "Device does not have enough warps to run the test (need at least 2)" << std::endl;
    cleanup();
    return -1;
  }

  kernel_arg.num_cores = static_cast<uint32_t>(num_cores);
  kernel_arg.num_warps = static_cast<uint32_t>(num_warps);
  kernel_arg.rounds    = ROUNDS;

  // One block per core, sized to the whole core, so that a block's barrier
  // (vortex::barrier, default num_warps = get_num_sub_groups()) rendezvouses
  // every warp of that core -- which is what releases warp 0 and warp 1 on
  // the same cycle.
  uint32_t num_slots = kernel_arg.num_cores * kernel_arg.num_warps;
  uint32_t buf_size  = num_slots * sizeof(uint32_t);

  std::cout << "num_cores=" << num_cores
            << ", num_warps=" << num_warps
            << ", num_threads=" << num_threads
            << ", rounds=" << kernel_arg.rounds << std::endl;

  std::cout << "allocate device memory" << std::endl;
  RT_CHECK(vx_buffer_create(device, buf_size, VX_MEM_READ_WRITE, &err_buffer));
  RT_CHECK(vx_buffer_address(err_buffer, &kernel_arg.err_addr));
  RT_CHECK(vx_buffer_create(device, buf_size, VX_MEM_READ_WRITE, &pre_buffer));
  RT_CHECK(vx_buffer_address(pre_buffer, &kernel_arg.pre_addr));
  RT_CHECK(vx_buffer_create(device, buf_size, VX_MEM_READ_WRITE, &post_buffer));
  RT_CHECK(vx_buffer_address(post_buffer, &kernel_arg.post_addr));

  std::vector<uint32_t> h_err(num_slots, 0xffffffff);
  std::vector<uint32_t> h_pre(num_slots, 0xffffffff);
  std::vector<uint32_t> h_post(num_slots, 0xffffffff);

  RT_CHECK(vx_enqueue_write(queue, err_buffer, 0, h_err.data(), buf_size, 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_write(queue, pre_buffer, 0, h_pre.data(), buf_size, 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_write(queue, post_buffer, 0, h_post.data(), buf_size, 0, nullptr, nullptr));

  std::cout << "load kernel module" << std::endl;
  RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
  RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

  std::cout << "start device" << std::endl;
  vx_event_h launch_ev = nullptr;
  {
    vx_launch_info_t li = {};
    li.struct_size  = sizeof(li);
    li.kernel       = kernel;
    li.args_host    = &kernel_arg;
    li.args_size    = sizeof(kernel_arg);
    li.ndim         = 1;
    li.grid_dim[0]  = kernel_arg.num_cores;
    li.block_dim[0] = static_cast<uint32_t>(num_warps * num_threads);
    RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &launch_ev));
  }

  std::cout << "download results" << std::endl;
  vx_event_h ev0 = nullptr, ev1 = nullptr, ev2 = nullptr;
  RT_CHECK(vx_enqueue_read(queue, h_err.data(), err_buffer, 0, buf_size, 1, &launch_ev, &ev0));
  RT_CHECK(vx_enqueue_read(queue, h_pre.data(), pre_buffer, 0, buf_size, 1, &ev0, &ev1));
  RT_CHECK(vx_enqueue_read(queue, h_post.data(), post_buffer, 0, buf_size, 1, &ev1, &ev2));

  std::cout << "wait for completion" << std::endl;
  RT_CHECK(vx_event_wait_value(ev2, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(ev2);
  vx_event_release(ev1);
  vx_event_release(ev0);
  vx_event_release(launch_ev);

  // Only warps 0 and 1 probe; every warp writes its slot, so an unwritten
  // entry means the launch did not cover the core the way the test assumes.
  int errors = 0;
  std::cout << "\ncore  warp  missed-flips  first(pre,post)" << std::endl;
  for (uint32_t c = 0; c < kernel_arg.num_cores; ++c) {
    for (uint32_t w = 0; w < kernel_arg.num_warps; ++w) {
      uint32_t i = c * kernel_arg.num_warps + w;
      if (h_err[i] == 0xffffffff) {
        std::cout << "  " << c << "     " << w << "     <not written>" << std::endl;
        ++errors;
        continue;
      }
      if (w < 2) {
        std::cout << "  " << c << "     " << w << "        " << h_err[i]
                  << "            (" << h_pre[i] << "," << h_post[i] << ")" << std::endl;
      }
      if (h_err[i] != 0) {
        // A count-1 arrival completes its generation, so the next arrival on
        // that slot must observe the complemented phase.
        std::cout << "    core " << c << " warp " << w << ": SLOT PHASE error: "
                  << h_err[i] << " of " << kernel_arg.rounds
                  << " rounds saw a count-1 arrival fail to advance its own slot's"
                  << " phase (pre=" << h_pre[i] << " post=" << h_post[i] << ")" << std::endl;
        errors += static_cast<int>(h_err[i]);
      }
    }
  }
  std::cout << "slot phase errors: " << errors << std::endl;

  std::cout << "cleanup" << std::endl;
  cleanup();

  if (errors != 0) {
    std::cout << "Found " << errors << " errors!" << std::endl;
    std::cout << "FAILED!" << std::endl;
    return 1;
  }

  std::cout << "PASSED!" << std::endl;
  return 0;
}
