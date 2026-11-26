#include <iostream>
#include <unistd.h>
#include <cstdlib>
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
uint32_t num_states = 256;
uint32_t num_perms = 2;

vx_device_h device = nullptr;
vx_buffer_h state_buf = nullptr;
vx_queue_h  queue   = nullptr;
vx_module_h module_ = nullptr;
vx_kernel_h kernel  = nullptr;
kernel_arg_t kernel_arg = {};

static void show_usage() {
  std::cout << "Usage: [-k kernel] [-n states] [-p permutations] [-h]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "k:n:p:h")) != -1) {
    switch (c) {
    case 'k': kernel_file = optarg; break;
    case 'n': num_states = atoi(optarg); break;
    case 'p': num_perms = atoi(optarg); break;
    case 'h': show_usage(); exit(0);
    default:  show_usage(); exit(-1);
    }
  }
}

void cleanup() {
  if (device) {
    if (state_buf) vx_buffer_release(state_buf);
    if (kernel)  vx_kernel_release(kernel);
    if (module_) vx_module_release(module_);
    if (queue)   vx_queue_release(queue);
    vx_device_dump_perf(device, stdout);
    vx_device_release(device);
  }
}

int main(int argc, char *argv[]) {
  parse_args(argc, argv);

  // Known answer: Keccak-f[1600] of the zero state.
  {
    uint64_t z[25] = {};
    keccak_f(z);
    if (z[0] != 0xF1258F7940E1DDE7ull) {
      std::cout << "Error: host Keccak-f[1600] reference is wrong" << std::endl;
      return -1;
    }
  }

  RT_CHECK(vx_device_open(0, &device));
  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));
  std::cout << "states=" << num_states << " permutations=" << num_perms << std::endl;

  uint32_t size = 25 * num_states * sizeof(uint64_t);
  kernel_arg.num_states = num_states;
  kernel_arg.num_perms = num_perms;
  RT_CHECK(vx_buffer_create(device, size, VX_MEM_READ_WRITE, &state_buf));
  RT_CHECK(vx_buffer_address(state_buf, &kernel_arg.state_addr));

  std::srand(50);
  std::vector<uint64_t> h_state(25 * num_states);
  for (auto& x : h_state)
    x = (uint64_t(rand()) << 33) ^ (uint64_t(rand()) << 11) ^ uint64_t(rand());
  RT_CHECK(vx_enqueue_write(queue, state_buf, 0, h_state.data(), size, 0, nullptr, nullptr));

  RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
  RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

  vx_event_h launch_ev = nullptr, read_ev = nullptr;
  {
    uint32_t grid_dim[1], block_dim[1];
    RT_CHECK(vx_device_max_occupancy_grid(device, 1, &num_states, grid_dim, block_dim));
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
  std::vector<uint64_t> d_state(25 * num_states);
  RT_CHECK(vx_enqueue_read(queue, d_state.data(), state_buf, 0, size, 1, &launch_ev, &read_ev));
  RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(read_ev);
  vx_event_release(launch_ev);

  int errors = 0;
  for (uint32_t s = 0; s < num_states; ++s) {
    uint64_t A[25];
    for (int i = 0; i < 25; ++i)
      A[i] = h_state[i * num_states + s];
    for (uint32_t p = 0; p < num_perms; ++p)
      keccak_f(A);
    for (int i = 0; i < 25; ++i) {
      uint64_t d = d_state[i * num_states + s];
      if (d != A[i] && errors++ < 10)
        printf("*** error: state[%u] lane %d expected=0x%016llx, actual=0x%016llx\n", s, i,
               (unsigned long long)A[i], (unsigned long long)d);
    }
  }

  cleanup();
  if (errors != 0) {
    std::cout << "Found " << errors << " errors!" << std::endl;
    std::cout << "FAILED!" << std::endl;
    return 1;
  }
  std::cout << "PASSED!" << std::endl;
  return 0;
}
