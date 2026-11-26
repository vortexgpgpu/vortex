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
uint32_t num_items = 256;
uint32_t num_iters = 4;

vx_device_h device = nullptr;
vx_buffer_h mod_buf = nullptr, x_buf = nullptr, y_buf = nullptr;
vx_queue_h  queue   = nullptr;
vx_module_h module_ = nullptr;
vx_kernel_h kernel  = nullptr;
kernel_arg_t kernel_arg = {};

static void show_usage() {
  std::cout << "Usage: [-k kernel] [-n items] [-i iterations] [-h]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "k:n:i:h")) != -1) {
    switch (c) {
    case 'k': kernel_file = optarg; break;
    case 'n': num_items = atoi(optarg); break;
    case 'i': num_iters = atoi(optarg); break;
    case 'h': show_usage(); exit(0);
    default:  show_usage(); exit(-1);
    }
  }
}

void cleanup() {
  if (device) {
    if (mod_buf) vx_buffer_release(mod_buf);
    if (x_buf)   vx_buffer_release(x_buf);
    if (y_buf)   vx_buffer_release(y_buf);
    if (kernel)  vx_kernel_release(kernel);
    if (module_) vx_module_release(module_);
    if (queue)   vx_queue_release(queue);
    vx_device_dump_perf(device, stdout);
    vx_device_release(device);
  }
}

static uint32_t rand32() {
  return (uint32_t(rand()) << 16) ^ uint32_t(rand());
}

int main(int argc, char *argv[]) {
  parse_args(argc, argv);
  std::srand(50);

  // odd modulus with its top bit set; operands below it
  std::vector<uint32_t> h_mod(LIMBS);
  for (auto& w : h_mod) w = rand32();
  h_mod[0] |= 1;
  h_mod[LIMBS - 1] |= 0x80000000u;
  uint32_t inv = h_mod[0];
  for (int k = 0; k < 5; ++k)
    inv *= 2 - h_mod[0] * inv;
  if (inv * h_mod[0] != 1) {
    std::cout << "Error: modulus inverse" << std::endl;
    return -1;
  }
  uint32_t n0inv = 0u - inv;

  std::vector<uint32_t> h_x(LIMBS * num_items), h_y(LIMBS * num_items);
  for (uint32_t i = 0; i < num_items; ++i) {
    for (int j = 0; j < LIMBS; ++j) {
      h_x[j * num_items + i] = rand32();
      h_y[j * num_items + i] = rand32();
    }
    h_x[(LIMBS - 1) * num_items + i] %= h_mod[LIMBS - 1];
    h_y[(LIMBS - 1) * num_items + i] %= h_mod[LIMBS - 1];
  }

  RT_CHECK(vx_device_open(0, &device));
  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));
  std::cout << "items=" << num_items << " limbs=" << LIMBS << " iterations=" << num_iters << std::endl;

  uint32_t mod_size = LIMBS * sizeof(uint32_t);
  uint32_t op_size = LIMBS * num_items * sizeof(uint32_t);
  kernel_arg.num_items = num_items;
  kernel_arg.num_iters = num_iters;
  kernel_arg.n0inv = n0inv;
  RT_CHECK(vx_buffer_create(device, mod_size, VX_MEM_READ, &mod_buf));
  RT_CHECK(vx_buffer_address(mod_buf, &kernel_arg.mod_addr));
  RT_CHECK(vx_buffer_create(device, op_size, VX_MEM_READ_WRITE, &x_buf));
  RT_CHECK(vx_buffer_address(x_buf, &kernel_arg.x_addr));
  RT_CHECK(vx_buffer_create(device, op_size, VX_MEM_READ, &y_buf));
  RT_CHECK(vx_buffer_address(y_buf, &kernel_arg.y_addr));
  RT_CHECK(vx_enqueue_write(queue, mod_buf, 0, h_mod.data(), mod_size, 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_write(queue, x_buf, 0, h_x.data(), op_size, 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_write(queue, y_buf, 0, h_y.data(), op_size, 0, nullptr, nullptr));

  RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
  RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

  vx_event_h launch_ev = nullptr, read_ev = nullptr;
  {
    uint32_t grid_dim[1], block_dim[1];
    RT_CHECK(vx_device_max_occupancy_grid(device, 1, &num_items, grid_dim, block_dim));
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
  std::vector<uint32_t> d_x(LIMBS * num_items);
  RT_CHECK(vx_enqueue_read(queue, d_x.data(), x_buf, 0, op_size, 1, &launch_ev, &read_ev));
  RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(read_ev);
  vx_event_release(launch_ev);

  int errors = 0;
  for (uint32_t i = 0; i < num_items; ++i) {
    uint32_t x[LIMBS];
    for (int j = 0; j < LIMBS; ++j)
      x[j] = h_x[j * num_items + i];
    for (uint32_t it = 0; it < num_iters; ++it)
      mont_mul(x, [&](int k) { return h_y[k * num_items + i]; }, h_mod.data(), n0inv);
    for (int j = 0; j < LIMBS; ++j) {
      uint32_t d = d_x[j * num_items + i];
      if (d != x[j] && errors++ < 10)
        printf("*** error: x[%u] limb %d expected=0x%08x, actual=0x%08x\n", i, j, x[j], d);
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
