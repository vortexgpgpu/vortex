// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

// Host side shared by the flash_attn tests: arguments, input data, the CPU
// reference, the output check and the device session.

#ifndef FLASH_ATTN_H
#define FLASH_ATTN_H

#include <vortex2.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <vector>

#define CHECK(expr) do { \
    vx_result_t _r = (expr); \
    if (_r != VX_SUCCESS) { \
      std::fprintf(stderr, "FAIL %s:%d: '%s' returned %s\n", \
                   __FILE__, __LINE__, #expr, vx_result_string(_r)); \
      std::exit(1); \
    } \
  } while (0)

namespace flash_attn {

// Every size is used as given or rejected; 0 for r, c and t lets the test
// derive the tile from the device caps.
struct Args {
  const char* kernel_file = "kernel.vxbin";
  uint32_t seq_len  = 64;
  uint32_t head_dim = 4;
  uint32_t r = 0; // block_size_r: Q rows per CTA
  uint32_t c = 0; // block_size_c: K/V rows per block
  uint32_t t = 0; // head_dim_tile
};

inline void parse_args(int argc, char** argv, Args& args) {
  int c;
  while ((c = getopt(argc, argv, "N:D:r:c:t:k:h")) != -1) {
    switch (c) {
    case 'N': args.seq_len     = std::atoi(optarg); break;
    case 'D': args.head_dim    = std::atoi(optarg); break;
    case 'r': args.r           = std::atoi(optarg); break;
    case 'c': args.c           = std::atoi(optarg); break;
    case 't': args.t           = std::atoi(optarg); break;
    case 'k': args.kernel_file = optarg;            break;
    default:
      std::printf("Usage: [-N seq_len] [-D head_dim] [-r block_size_r] [-c block_size_c] "
                  "[-t head_dim_tile] [-k kernel] [-h]\n");
      std::exit(c == 'h' ? 0 : 1);
    }
  }
  if (args.seq_len == 0 || args.head_dim == 0) {
    std::printf("Error: seq_len (-N) and head_dim (-D) must be nonzero\n");
    std::exit(1);
  }
}

// Q, K and V in [0, 1), row-major [seq_len][head_dim].
inline std::vector<float> random_matrix(uint32_t rows, uint32_t cols) {
  std::vector<float> m(size_t(rows) * cols);
  for (auto& x : m) {
    x = static_cast<float>(std::rand()) / RAND_MAX;
  }
  return m;
}

// O = softmax(Q K^T) V in double precision. quantize_p maps each
// unnormalized probability to the value the device multiplies V with.
template <typename QuantizeP>
std::vector<float> reference(const std::vector<float>& Q, const std::vector<float>& K,
                             const std::vector<float>& V, uint32_t N, uint32_t D,
                             QuantizeP quantize_p) {
  std::vector<float> O(size_t(N) * D);
  std::vector<double> s(N);
  for (uint32_t i = 0; i < N; ++i) {
    double mx = -INFINITY;
    for (uint32_t j = 0; j < N; ++j) {
      double acc = 0;
      for (uint32_t k = 0; k < D; ++k) {
        acc += double(Q[i * D + k]) * K[j * D + k];
      }
      s[j] = acc;
      mx = std::max(mx, acc);
    }
    double l = 0;
    for (uint32_t j = 0; j < N; ++j) {
      s[j] = std::exp(s[j] - mx);
      l += s[j];
    }
    for (uint32_t k = 0; k < D; ++k) {
      double acc = 0;
      for (uint32_t j = 0; j < N; ++j) {
        acc += double(quantize_p(float(s[j]))) * V[j * D + k];
      }
      O[i * D + k] = float(acc / l);
    }
  }
  return O;
}

inline std::vector<float> reference(const std::vector<float>& Q, const std::vector<float>& K,
                                    const std::vector<float>& V, uint32_t N, uint32_t D) {
  return reference(Q, K, V, N, D, [](float p) { return p; });
}

// Counts the elements outside |out - ref| <= atol + rtol * |ref|.
inline int check_output(const std::vector<float>& ref, const std::vector<float>& out,
                        float atol, float rtol) {
  int errors = 0;
  float worst = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    float err = std::fabs(out[i] - ref[i]);
    worst = std::max(worst, err);
    if (!(err <= atol + rtol * std::fabs(ref[i]))) {
      if (errors < 16) {
        std::printf("*** error: [%zu] expected=%f, actual=%f\n", i, ref[i], out[i]);
      }
      ++errors;
    }
  }
  std::printf("worst absolute error: %.3g (atol=%g, rtol=%g)\n", worst, atol, rtol);
  return errors;
}

inline int report(int errors) {
  if (errors) {
    std::printf("Found %d errors!\nFAILED!\n", errors);
    return 1;
  }
  std::printf("PASSED!\n");
  return 0;
}

// One device, one queue, the buffers and one kernel launch.
class Session {
public:
  Session() {
    CHECK(vx_device_open(0, &dev_));
    vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
    CHECK(vx_queue_create(dev_, &qi, &queue_));
  }

  ~Session() {
    for (auto buf : buffers_) {
      vx_buffer_release(buf);
    }
    if (kernel_) {
      vx_kernel_release(kernel_);
    }
    if (module_) {
      vx_module_release(module_);
    }
    vx_queue_release(queue_);
    vx_device_dump_perf(dev_, stdout);
    vx_device_release(dev_);
  }

  uint64_t caps(uint32_t id) const {
    uint64_t value = 0;
    CHECK(vx_device_query(dev_, id, &value));
    return value;
  }

  // Returns the device address of a new read-only buffer holding data.
  uint64_t upload(const void* data, size_t bytes) {
    auto buf = this->create(bytes, VX_MEM_READ);
    CHECK(vx_enqueue_write(queue_, buf, 0, data, bytes, 0, nullptr, nullptr));
    return this->address(buf);
  }

  // Returns the device address of the output buffer that run() reads back.
  uint64_t output(size_t bytes) {
    out_buf_ = this->create(bytes, VX_MEM_WRITE);
    out_bytes_ = bytes;
    return this->address(out_buf_);
  }

  // block_dim = {lanes, warps}: X is the lane axis, so row reductions are
  // shuffles within a warp.
  void run(const char* kernel_file, const void* args, size_t args_size,
           uint32_t grid_x, uint32_t lanes, uint32_t warps, uint64_t lmem_size, void* out) {
    CHECK(vx_module_load_file(dev_, kernel_file, &module_));
    CHECK(vx_module_get_kernel(module_, "main", &kernel_));

    vx_launch_info_t li{};
    li.struct_size  = sizeof(li);
    li.kernel       = kernel_;
    li.args_host    = args;
    li.args_size    = args_size;
    li.ndim         = 2;
    li.grid_dim[0]  = grid_x;
    li.grid_dim[1]  = 1;
    li.block_dim[0] = lanes;
    li.block_dim[1] = warps;
    li.lmem_size    = lmem_size;

    auto t_start = std::chrono::high_resolution_clock::now();
    vx_event_h launch_ev = nullptr, read_ev = nullptr;
    CHECK(vx_enqueue_launch(queue_, &li, 0, nullptr, &launch_ev));
    CHECK(vx_enqueue_read(queue_, out, out_buf_, 0, out_bytes_, 1, &launch_ev, &read_ev));
    CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
    auto t_end = std::chrono::high_resolution_clock::now();
    std::printf("Elapsed: %ld ms\n",
        (long)std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count());
    vx_event_release(read_ev);
    vx_event_release(launch_ev);
  }

private:
  vx_buffer_h create(size_t bytes, uint32_t flags) {
    vx_buffer_h buf = nullptr;
    CHECK(vx_buffer_create(dev_, bytes, flags, &buf));
    buffers_.push_back(buf);
    return buf;
  }

  static uint64_t address(vx_buffer_h buf) {
    uint64_t addr = 0;
    CHECK(vx_buffer_address(buf, &addr));
    return addr;
  }

  vx_device_h dev_ = nullptr;
  vx_queue_h queue_ = nullptr;
  vx_module_h module_ = nullptr;
  vx_kernel_h kernel_ = nullptr;
  std::vector<vx_buffer_h> buffers_;
  vx_buffer_h out_buf_ = nullptr;
  size_t out_bytes_ = 0;
};

} // namespace flash_attn

#endif // FLASH_ATTN_H
