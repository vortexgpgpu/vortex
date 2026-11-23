#include <iostream>
#include <unistd.h>
#include <cmath>
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
uint32_t num_rows = 256;

vx_device_h device = nullptr;
vx_buffer_h acc_buf = nullptr, bias_buf = nullptr, q_buf = nullptr, scale_buf = nullptr;
vx_queue_h  queue   = nullptr;
vx_module_h module_ = nullptr;
vx_kernel_h kernel  = nullptr;
kernel_arg_t kernel_arg = {};

static void show_usage() {
  std::cout << "Usage: [-k kernel] [-r rows] [-h]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "k:r:h")) != -1) {
    switch (c) {
    case 'k': kernel_file = optarg; break;
    case 'r': num_rows = atoi(optarg); break;
    case 'h': show_usage(); exit(0);
    default:  show_usage(); exit(-1);
    }
  }
}

void cleanup() {
  if (device) {
    if (acc_buf)   vx_buffer_release(acc_buf);
    if (bias_buf)  vx_buffer_release(bias_buf);
    if (q_buf)     vx_buffer_release(q_buf);
    if (scale_buf) vx_buffer_release(scale_buf);
    if (kernel)  vx_kernel_release(kernel);
    if (module_) vx_module_release(module_);
    if (queue)   vx_queue_release(queue);
    vx_device_dump_perf(device, stdout);
    vx_device_release(device);
  }
}

int main(int argc, char *argv[]) {
  parse_args(argc, argv);
  if (ROW % CHUNK != 0 || CHUNK % 4 != 0) {
    std::cout << "Error: CHUNK must divide ROW=" << ROW << " and be a multiple of 4" << std::endl;
    return -1;
  }
  std::srand(50);

  RT_CHECK(vx_device_open(0, &device));
  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));
  std::cout << "rows=" << num_rows << " row=" << ROW << " chunk=" << CHUNK << std::endl;

  uint32_t acc_size = num_rows * ROW * sizeof(float);
  uint32_t q_size = num_rows * (ROW / 4) * sizeof(uint32_t);
  kernel_arg.num_rows  = num_rows;
  kernel_arg.act_scale = 0.25f;

  RT_CHECK(vx_buffer_create(device, acc_size, VX_MEM_READ, &acc_buf));
  RT_CHECK(vx_buffer_address(acc_buf, &kernel_arg.acc_addr));
  RT_CHECK(vx_buffer_create(device, ROW * sizeof(float), VX_MEM_READ, &bias_buf));
  RT_CHECK(vx_buffer_address(bias_buf, &kernel_arg.bias_addr));
  RT_CHECK(vx_buffer_create(device, q_size, VX_MEM_WRITE, &q_buf));
  RT_CHECK(vx_buffer_address(q_buf, &kernel_arg.q_addr));
  RT_CHECK(vx_buffer_create(device, num_rows * sizeof(float), VX_MEM_WRITE, &scale_buf));
  RT_CHECK(vx_buffer_address(scale_buf, &kernel_arg.scale_addr));

  std::vector<float> h_acc(num_rows * ROW), h_bias(ROW);
  for (auto& x : h_acc)  x = 32.0f * (float(rand()) / RAND_MAX) - 16.0f;
  for (auto& x : h_bias) x = 4.0f * (float(rand()) / RAND_MAX) - 2.0f;
  RT_CHECK(vx_enqueue_write(queue, acc_buf, 0, h_acc.data(), acc_size, 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_write(queue, bias_buf, 0, h_bias.data(), ROW * sizeof(float), 0, nullptr, nullptr));

  RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
  RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

  vx_event_h launch_ev = nullptr, ev1 = nullptr, read_ev = nullptr;
  {
    uint32_t grid_dim[1], block_dim[1];
    RT_CHECK(vx_kernel_max_occupancy_grid(kernel, 1, &num_rows, 0, grid_dim, block_dim));
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
  std::vector<uint32_t> d_q(num_rows * (ROW / 4));
  std::vector<float> d_scale(num_rows);
  RT_CHECK(vx_enqueue_read(queue, d_q.data(), q_buf, 0, q_size, 1, &launch_ev, &ev1));
  RT_CHECK(vx_enqueue_read(queue, d_scale.data(), scale_buf, 0, num_rows * sizeof(float), 1, &ev1, &read_ev));
  RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(read_ev);
  vx_event_release(ev1);
  vx_event_release(launch_ev);

  // The reference follows the device's arithmetic; fused multiply-adds may
  // move a value across a rounding boundary, so int8 codes may differ by one.
  int errors = 0;
  for (uint32_t r = 0; r < num_rows; ++r) {
    std::vector<float> y(ROW);
    float amax = 1e-12f;
    for (uint32_t i = 0; i < ROW; ++i) {
      y[i] = rp_gelu((h_acc[r * ROW + i] + h_bias[i]) * kernel_arg.act_scale);
      amax = std::fmax(amax, std::fabs(y[i]));
    }
    float s = 127.0f / amax;
    if (std::fabs(d_scale[r] - amax / 127.0f) > 1e-4f * (amax / 127.0f) && errors++ < 10)
      printf("*** error: scale[%u] expected=%f, actual=%f\n", r, amax / 127.0f, d_scale[r]);
    for (uint32_t i = 0; i < ROW; ++i) {
      int ref = rp_quant(y[i], s);
      int cur = int8_t((d_q[r * (ROW / 4) + i / 4] >> (8 * (i % 4))) & 0xff);
      if (std::abs(cur - ref) > 1 && errors++ < 10)
        printf("*** error: q[%u][%u] expected=%d, actual=%d\n", r, i, ref, cur);
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
