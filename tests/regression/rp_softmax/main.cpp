#include <iostream>
#include <unistd.h>
#include <cmath>
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
uint32_t num_cols = 512;
uint32_t head_dim = HEAD;

vx_device_h device = nullptr;
vx_buffer_h s_buf = nullptr, p_buf = nullptr, o_buf = nullptr, ml_buf = nullptr;
vx_queue_h  queue   = nullptr;
vx_module_h module_ = nullptr;
vx_kernel_h kernel  = nullptr;
kernel_arg_t kernel_arg = {};

static void show_usage() {
  std::cout << "Usage: [-k kernel] [-r rows] [-n columns] [-h]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "k:r:n:h")) != -1) {
    switch (c) {
    case 'k': kernel_file = optarg; break;
    case 'r': num_rows = atoi(optarg); break;
    case 'n': num_cols = atoi(optarg); break;
    case 'h': show_usage(); exit(0);
    default:  show_usage(); exit(-1);
    }
  }
}

void cleanup() {
  if (device) {
    if (s_buf)  vx_buffer_release(s_buf);
    if (p_buf)  vx_buffer_release(p_buf);
    if (o_buf)  vx_buffer_release(o_buf);
    if (ml_buf) vx_buffer_release(ml_buf);
    if (kernel)  vx_kernel_release(kernel);
    if (module_) vx_module_release(module_);
    if (queue)   vx_queue_release(queue);
    vx_device_dump_perf(device, stdout);
    vx_device_release(device);
  }
}

static bool close_to(float a, float b) {
  return std::fabs(a - b) <= 1e-4f * std::fmax(1.0f, std::fabs(b));
}

int main(int argc, char *argv[]) {
  parse_args(argc, argv);
  if (num_cols % CHUNK != 0) {
    std::cout << "Error: columns must be a multiple of CHUNK=" << CHUNK << std::endl;
    return -1;
  }
  std::srand(50);

  RT_CHECK(vx_device_open(0, &device));
  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));

  std::cout << "rows=" << num_rows << " columns=" << num_cols << " head_dim=" << head_dim
            << " chunk=" << CHUNK << std::endl;

  uint32_t s_size  = num_rows * num_cols * sizeof(float);
  uint32_t o_size  = num_rows * head_dim * sizeof(float);
  uint32_t ml_size = num_rows * 2 * sizeof(float);

  kernel_arg.num_rows = num_rows;
  kernel_arg.num_cols = num_cols;
  kernel_arg.scale    = 0.125f;

  RT_CHECK(vx_buffer_create(device, s_size, VX_MEM_READ, &s_buf));
  RT_CHECK(vx_buffer_address(s_buf, &kernel_arg.s_addr));
  RT_CHECK(vx_buffer_create(device, s_size, VX_MEM_WRITE, &p_buf));
  RT_CHECK(vx_buffer_address(p_buf, &kernel_arg.p_addr));
  RT_CHECK(vx_buffer_create(device, o_size, VX_MEM_READ_WRITE, &o_buf));
  RT_CHECK(vx_buffer_address(o_buf, &kernel_arg.o_addr));
  RT_CHECK(vx_buffer_create(device, ml_size, VX_MEM_WRITE, &ml_buf));
  RT_CHECK(vx_buffer_address(ml_buf, &kernel_arg.ml_addr));

  std::vector<float> h_s(num_rows * num_cols), h_o(num_rows * head_dim);
  for (auto& x : h_s) x = 16.0f * (float(rand()) / RAND_MAX) - 8.0f;
  for (auto& x : h_o) x = float(rand()) / RAND_MAX;

  RT_CHECK(vx_enqueue_write(queue, s_buf, 0, h_s.data(), s_size, 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_write(queue, o_buf, 0, h_o.data(), o_size, 0, nullptr, nullptr));

  RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
  RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

  vx_event_h launch_ev = nullptr, read_ev = nullptr;
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

  std::vector<float> d_p(num_rows * num_cols), d_o(num_rows * head_dim), d_ml(num_rows * 2);
  vx_event_h ev1 = nullptr, ev2 = nullptr;
  RT_CHECK(vx_enqueue_read(queue, d_p.data(), p_buf, 0, s_size, 1, &launch_ev, &ev1));
  RT_CHECK(vx_enqueue_read(queue, d_o.data(), o_buf, 0, o_size, 1, &ev1, &ev2));
  RT_CHECK(vx_enqueue_read(queue, d_ml.data(), ml_buf, 0, ml_size, 1, &ev2, &read_ev));
  RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(read_ev);
  vx_event_release(ev2);
  vx_event_release(ev1);
  vx_event_release(launch_ev);

  int errors = 0;
  for (uint32_t r = 0; r < num_rows; ++r) {
    const float* s = &h_s[r * num_cols];
    std::vector<float> o(h_o.begin() + r * head_dim, h_o.begin() + (r + 1) * head_dim);
    float m = s[0] * kernel_arg.scale, l = 0.0f;
    for (uint32_t c = 0; c < num_cols; c += CHUNK) {
      float mt = s[c] * kernel_arg.scale;
      for (uint32_t i = 1; i < CHUNK; ++i) mt = rp_max(mt, s[c + i] * kernel_arg.scale);
      float mn = rp_max(m, mt);
      float corr = rp_exp(m - mn);
      float sum = 0.0f;
      for (uint32_t i = 0; i < CHUNK; ++i) {
        float pv = rp_exp(s[c + i] * kernel_arg.scale - mn);
        sum += pv;
        if (!close_to(d_p[r * num_cols + c + i], pv) && errors++ < 10)
          printf("*** error: p[%u][%u] expected=%f, actual=%f\n", r, c + i, pv, d_p[r * num_cols + c + i]);
      }
      l = l * corr + sum;
      for (auto& x : o) x *= corr;
      m = mn;
    }
    for (uint32_t k = 0; k < head_dim; ++k) {
      float ref = o[k] / l;
      if (!close_to(d_o[r * head_dim + k], ref) && errors++ < 10)
        printf("*** error: o[%u][%u] expected=%f, actual=%f\n", r, k, ref, d_o[r * head_dim + k]);
    }
    if ((!close_to(d_ml[r * 2], m) || !close_to(d_ml[r * 2 + 1], l)) && errors++ < 10)
      printf("*** error: m,l[%u] expected=%f,%f actual=%f,%f\n", r, m, l, d_ml[r * 2], d_ml[r * 2 + 1]);
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
