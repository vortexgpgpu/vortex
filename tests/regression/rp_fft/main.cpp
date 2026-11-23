#include <iostream>
#include <unistd.h>
#include <cmath>
#include <complex>
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
uint32_t num_ffts = 64;

vx_device_h device = nullptr;
vx_buffer_h data_buf = nullptr, tmp_buf = nullptr, tw_buf = nullptr;
vx_queue_h  queue   = nullptr;
vx_module_h module_ = nullptr;
vx_kernel_h kernel  = nullptr;
kernel_arg_t kernel_arg = {};

static void show_usage() {
  std::cout << "Usage: [-k kernel] [-b transforms] [-h]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "k:b:h")) != -1) {
    switch (c) {
    case 'k': kernel_file = optarg; break;
    case 'b': num_ffts = atoi(optarg); break;
    case 'h': show_usage(); exit(0);
    default:  show_usage(); exit(-1);
    }
  }
}

void cleanup() {
  if (device) {
    if (data_buf) vx_buffer_release(data_buf);
    if (tmp_buf)  vx_buffer_release(tmp_buf);
    if (tw_buf)   vx_buffer_release(tw_buf);
    if (kernel)  vx_kernel_release(kernel);
    if (module_) vx_module_release(module_);
    if (queue)   vx_queue_release(queue);
    vx_device_dump_perf(device, stdout);
    vx_device_release(device);
  }
}

static void ref_fft(std::vector<std::complex<double>>& a) {
  size_t n = a.size();
  if (n == 1)
    return;
  std::vector<std::complex<double>> e(n / 2), o(n / 2);
  for (size_t i = 0; i < n / 2; ++i) {
    e[i] = a[2 * i];
    o[i] = a[2 * i + 1];
  }
  ref_fft(e);
  ref_fft(o);
  for (size_t k = 0; k < n / 2; ++k) {
    auto t = std::polar(1.0, -2.0 * M_PI * double(k) / double(n)) * o[k];
    a[k] = e[k] + t;
    a[k + n / 2] = e[k] - t;
  }
}

int main(int argc, char *argv[]) {
  parse_args(argc, argv);
  uint32_t passes = 0;
  for (uint32_t n = 1; n < FFT_LEN; n *= RADIX)
    ++passes;
  if ((1u << (passes * __builtin_ctz(RADIX))) != FFT_LEN) {
    std::cout << "Error: FFT_LEN=" << FFT_LEN << " is not a power of RADIX=" << RADIX << std::endl;
    return -1;
  }
  std::srand(50);

  RT_CHECK(vx_device_open(0, &device));
  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));
  std::cout << "transforms=" << num_ffts << " length=" << FFT_LEN << " radix=" << RADIX
            << " passes=" << passes << std::endl;

  uint32_t data_size = num_ffts * FFT_LEN * 2 * sizeof(float);
  uint32_t tw_size = FFT_LEN * 2 * sizeof(float);
  kernel_arg.num_ffts = num_ffts;
  RT_CHECK(vx_buffer_create(device, data_size, VX_MEM_READ_WRITE, &data_buf));
  RT_CHECK(vx_buffer_address(data_buf, &kernel_arg.data_addr));
  RT_CHECK(vx_buffer_create(device, data_size, VX_MEM_READ_WRITE, &tmp_buf));
  RT_CHECK(vx_buffer_address(tmp_buf, &kernel_arg.tmp_addr));
  RT_CHECK(vx_buffer_create(device, tw_size, VX_MEM_READ, &tw_buf));
  RT_CHECK(vx_buffer_address(tw_buf, &kernel_arg.tw_addr));

  std::vector<float> h_data(num_ffts * FFT_LEN * 2), h_tw(FFT_LEN * 2);
  for (auto& x : h_data) x = 2.0f * (float(rand()) / RAND_MAX) - 1.0f;
  for (uint32_t k = 0; k < FFT_LEN; ++k) {
    h_tw[2 * k]     = float(std::cos(-2.0 * M_PI * k / FFT_LEN));
    h_tw[2 * k + 1] = float(std::sin(-2.0 * M_PI * k / FFT_LEN));
  }
  RT_CHECK(vx_enqueue_write(queue, data_buf, 0, h_data.data(), data_size, 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_write(queue, tw_buf, 0, h_tw.data(), tw_size, 0, nullptr, nullptr));

  RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
  RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

  vx_event_h launch_ev = nullptr, read_ev = nullptr;
  {
    uint32_t grid_dim[1], block_dim[1];
    RT_CHECK(vx_device_max_occupancy_grid(device, 1, &num_ffts, grid_dim, block_dim));
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
  // an odd number of passes leaves the result in the scratch buffer
  std::vector<float> d_out(num_ffts * FFT_LEN * 2);
  RT_CHECK(vx_enqueue_read(queue, d_out.data(), (passes & 1) ? tmp_buf : data_buf, 0, data_size,
                           1, &launch_ev, &read_ev));
  RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(read_ev);
  vx_event_release(launch_ev);

  int errors = 0;
  for (uint32_t f = 0; f < num_ffts; ++f) {
    std::vector<std::complex<double>> a(FFT_LEN);
    for (uint32_t i = 0; i < FFT_LEN; ++i)
      a[i] = {h_data[(i * num_ffts + f) * 2], h_data[(i * num_ffts + f) * 2 + 1]};
    ref_fft(a);
    double peak = 0;
    for (auto& v : a)
      peak = std::max(peak, std::abs(v));
    for (uint32_t i = 0; i < FFT_LEN; ++i) {
      std::complex<double> d(d_out[(i * num_ffts + f) * 2], d_out[(i * num_ffts + f) * 2 + 1]);
      if (std::abs(d - a[i]) > 1e-4 * peak && errors++ < 10)
        printf("*** error: fft[%u][%u] expected=(%f,%f), actual=(%f,%f)\n", f, i,
               a[i].real(), a[i].imag(), d.real(), d.imag());
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
