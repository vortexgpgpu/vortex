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
     cleanup();                                                 \
     exit(-1);                                                  \
   } while (false)

///////////////////////////////////////////////////////////////////////////////

const char* kernel_file = "kernel.vxbin";
uint32_t rounds = 8;
bool plain_guard = false;
uint32_t mode = FENCE_MP_PAIRS;

vx_device_h device = nullptr;
vx_buffer_h payload_buffer = nullptr;
vx_buffer_h guard_buffer = nullptr;
vx_buffer_h result_buffer = nullptr;
vx_buffer_h flood_buffer = nullptr;
vx_queue_h  queue = nullptr;
vx_module_h module_ = nullptr;
vx_kernel_h kernel = nullptr;
kernel_arg_t kernel_arg = {};

static void show_usage() {
   std::cout << "Vortex Test." << std::endl;
   std::cout << "Usage: [-k: kernel] [-r rounds] [-g: plain-store guard] [-m mode: 0=pairs, 1=lanes, 2=ports] [-h: help]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "r:gm:k:h")) != -1) {
    switch (c) {
    case 'r':
      rounds = atoi(optarg);
      break;
    case 'g':
      plain_guard = true;
      break;
    case 'm':
      mode = atoi(optarg);
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
    if (payload_buffer) vx_buffer_release(payload_buffer);
    if (guard_buffer)   vx_buffer_release(guard_buffer);
    if (result_buffer)  vx_buffer_release(result_buffer);
    if (flood_buffer)   vx_buffer_release(flood_buffer);
    if (kernel)  vx_kernel_release(kernel);
    if (module_) vx_module_release(module_);
    if (queue)   vx_queue_release(queue);
    vx_device_dump_perf(device, stdout);
    vx_device_release(device);
  }
}

int main(int argc, char *argv[]) {
  parse_args(argc, argv);
  if (rounds == 0) {
    rounds = 1;
  }

  std::cout << "open device connection" << std::endl;
  RT_CHECK(vx_device_open(0, &device));

  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));

  uint64_t num_cores, num_warps, num_threads;
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_CORES, &num_cores));
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_WARPS, &num_warps));
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_THREADS, &num_threads));

  if ((mode == FENCE_MP_LANES && num_cores < 2) || (mode == FENCE_MP_PORTS && num_cores < 3)) {
    std::cout << "Error: mode " << mode << " needs " << (mode == FENCE_MP_LANES ? 2 : 3) << " cores" << std::endl;
    cleanup();
    return -1;
  }

  // Lane mode runs a writer warp and a reader warp; port mode adds a warp on
  // every other core.
  uint32_t num_blocks = (mode == FENCE_MP_LANES) ? 2 : num_cores;
  uint32_t num_tasks = (mode == FENCE_MP_PAIRS) ? num_cores * num_warps * num_threads
                                                : num_blocks * num_threads;
  uint32_t payload_size = num_tasks * rounds * sizeof(uint32_t);
  uint32_t guard_size   = num_tasks * sizeof(uint32_t);
  uint32_t result_size  = 2 * num_tasks * sizeof(uint32_t);

  std::cout << "number of tasks: " << num_tasks << ", rounds: " << rounds
            << (plain_guard ? ", plain-store guard" : ", atomic guard")
            << ", mode: " << mode << std::endl;

  kernel_arg.num_tasks = num_tasks;
  kernel_arg.rounds = rounds;
  kernel_arg.plain_guard = plain_guard;
  kernel_arg.mode = mode;

  std::cout << "allocate device memory" << std::endl;
  RT_CHECK(vx_buffer_create(device, payload_size, VX_MEM_READ_WRITE, &payload_buffer));
  RT_CHECK(vx_buffer_address(payload_buffer, &kernel_arg.payload_addr));
  RT_CHECK(vx_buffer_create(device, guard_size, VX_MEM_READ_WRITE, &guard_buffer));
  RT_CHECK(vx_buffer_address(guard_buffer, &kernel_arg.guard_addr));
  RT_CHECK(vx_buffer_create(device, result_size, VX_MEM_WRITE, &result_buffer));
  RT_CHECK(vx_buffer_address(result_buffer, &kernel_arg.result_addr));
  // the write runs in the background, so its source outlives the launch
  std::vector<uint32_t> flood_zeros;
  if (mode != FENCE_MP_PAIRS) {
    uint64_t flood_words = (mode == FENCE_MP_LANES) ? LANE_BUFFER_WORDS : PORT_BUFFER_WORDS(num_blocks);
    RT_CHECK(vx_buffer_create(device, flood_words * sizeof(uint32_t), VX_MEM_READ_WRITE, &flood_buffer));
    RT_CHECK(vx_buffer_address(flood_buffer, &kernel_arg.flood_addr));
    flood_zeros.assign(flood_words, 0);
    RT_CHECK(vx_enqueue_write(queue, flood_buffer, 0, flood_zeros.data(), flood_words * sizeof(uint32_t), 0, nullptr, nullptr));
  }

  std::cout << "clear payload and guard words" << std::endl;
  std::vector<uint32_t> zeros(num_tasks * rounds, 0);
  RT_CHECK(vx_enqueue_write(queue, payload_buffer, 0, zeros.data(), payload_size, 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_write(queue, guard_buffer, 0, zeros.data(), guard_size, 0, nullptr, nullptr));

  std::cout << "load kernel module" << std::endl;
  RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
  RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

  std::cout << "launch kernel" << std::endl;
  vx_event_h launch_ev = nullptr, read_ev = nullptr;
  {
    uint32_t n = kernel_arg.num_tasks;
    uint32_t grid_dim[1], block_dim[1];
    if (mode != FENCE_MP_PAIRS) {
      grid_dim[0]  = num_blocks;
      block_dim[0] = num_threads;
    } else {
      RT_CHECK(vx_device_max_occupancy_grid(device, 1, &n, grid_dim, block_dim));
    }
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

  std::vector<uint32_t> h_result(2 * num_tasks);
  RT_CHECK(vx_enqueue_read(queue, h_result.data(), result_buffer, 0, result_size, 1, &launch_ev, &read_ev));

  std::cout << "wait for completion" << std::endl;
  RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(read_ev);
  vx_event_release(launch_ev);

  std::cout << "verify result" << std::endl;
  uint64_t errors = 0, checked = 0;
  if (mode == FENCE_MP_LANES) {
    // The reader saw the guard, then the payload.
    checked = h_result[0];
    if (h_result[0] != 0 && h_result[1] != LANE_PAYLOAD) {
      std::cout << "reader read 0x" << std::hex << h_result[1] << ", expected 0x"
                << LANE_PAYLOAD << std::dec << std::endl;
      errors = 1;
    }
  } else if (mode == FENCE_MP_PORTS) {
    // The reader saw the guard, then the payload.
    checked = h_result[0];
    if (h_result[0] != 0 && h_result[1] != PORT_PAYLOAD) {
      std::cout << "reader read 0x" << std::hex << h_result[1] << ", expected 0x"
                << PORT_PAYLOAD << std::dec << std::endl;
      errors = 1;
    }
  } else for (uint32_t i = 0; i < num_tasks; ++i) {
    errors  += h_result[2 * i + 0];
    checked += h_result[2 * i + 1];
    if (h_result[2 * i + 0] != 0 && errors <= 8) {
      std::cout << "task " << i << ": " << h_result[2 * i + 0]
                << " stale payload reads" << std::endl;
    }
  }
  std::cout << "released rounds observed: " << checked << ", stale payloads: " << errors << std::endl;

  std::cout << "cleanup" << std::endl;
  cleanup();

  if (errors != 0) {
    std::cout << "Found " << errors << " errors!" << std::endl;
    std::cout << "FAILED!" << std::endl;
    return 1;
  }
  if (checked == 0) {
    std::cout << "No task observed a released partner; nothing was checked." << std::endl;
    std::cout << "FAILED!" << std::endl;
    return 1;
  }

  std::cout << "PASSED!" << std::endl;
  return 0;
}
