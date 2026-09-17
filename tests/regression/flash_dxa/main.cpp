// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

// flash attention dxa regression test.

#include <vortex2.h>
#include <dxa.h>
#include "common.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <unistd.h>
#include <vector>

#define FLOAT_ULP 8192

#define RT_CHECK(expr) do { \
    int _r = (expr); \
    if (_r != 0) { \
        std::fprintf(stderr, "FAIL %s:%d: '%s' returned %d\n", \
                     __FILE__, __LINE__, #expr, _r); \
        std::exit(1); \
    } \
} while (0)

namespace {
const char* kernel_file = "kernel.vxbin";
uint32_t N = 64;
uint32_t d = 4;
// 0 auto-derives from queried hardware caps
uint32_t r_override = 0;
uint32_t c_override = 0;
uint32_t d_override = 0;

static void parse_args(int argc, char **argv) {
  	int c;
	while ((c = getopt(argc, argv, "N:D:k:r:c:d:h")) != -1) {
		switch (c) {
			case 'N': N 		  = std::atoi(optarg); break;
			case 'D': d			  = std::atoi(optarg); break;
			case 'k': kernel_file = optarg; 	 	   break;
			case 'r': r_override  = std::atoi(optarg); break;
			case 'c': c_override  = std::atoi(optarg); break;
			case 'd': d_override  = std::atoi(optarg); break;
			default:
				std::cout << "Usage: [-k: kernel] [-N: sequence_length] [-D: head_dim] "
				             "[-r: block_size_r] [-c: block_size_c] [-d: head_dim_tile] [-h]" << std::endl;
				std::exit(c == 'h' ? 0 : -1);
		}
	}
}

template <typename Type>
class Comparator {};

template <>
class Comparator<float> {
public:
  static const char* type_str() {
    return "float";
  }
  static float generate() {
    return static_cast<float>(rand()) / RAND_MAX;
  }
  static bool compare(float a, float b, int index, int errors) {
    union fi_t { float f; int32_t i; };
    fi_t fa, fb;
    fa.f = a;
    fb.f = b;
    auto d = std::abs(fa.i - fb.i);
    if (d > FLOAT_ULP) {
      if (errors < 100) {
        printf("*** error: [%d] expected=%f, actual=%f\n", index, a, b);
      }
      return false;
    }
    return true;
  }
};

static void flash_attention_cpu(TYPE* out, const TYPE* Q, const TYPE* K, const TYPE* V, uint32_t N, uint32_t d) {
  std::vector<TYPE> scores(N);
  std::vector<TYPE> probs(N);
  for (uint32_t i = 0; i < N; ++i) {
    // Compute row of scores
    for (uint32_t j = 0; j < N; ++j) {
      TYPE sum = TYPE(0);
      for (uint32_t k = 0; k < d; ++k)
        sum += Q[i * d + k] * K[j * d + k];
      scores[j] = sum;
    }

    // Compute softmax of row
    auto max = scores[0];
    for (uint32_t j = 1; j < N; ++j)
      max = std::max(max, scores[j]);
    TYPE exp_sum = TYPE(0);
    for (uint32_t j = 0; j < N; ++j) {
      auto exp = std::exp(scores[j] - max);
      probs[j] = exp;
      exp_sum += exp;
    }
    for (uint32_t j = 0; j < N; ++j)
      probs[j] /= exp_sum;

    // Compute row of O
    for (uint32_t k = 0; k < d; ++k) {
      TYPE sum = TYPE(0);
      for (uint32_t j = 0; j < N; ++j)
        sum += probs[j] * V[j * d + k];
      out[i * d + k] = sum;
    }
  }
}
} // namespace

int main(int argc, char *argv[]) {
    parse_args(argc, argv);
    std::srand(50);

    std::cout << "flash vortex2: " << N << "x" << d << std::endl;

    if (N == 0 || d == 0) {
        printf("Error: sequence length %u and head dimension %u must both be nonzero\n", N, d);
        return -1;
    }

    vx_device_h dev = nullptr;
    RT_CHECK(vx_device_open(0, &dev));

    auto t_start = std::chrono::high_resolution_clock::now();

    vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
    vx_queue_h q = nullptr;
    RT_CHECK(vx_queue_create(dev, &qi, &q));

    uint64_t isa_flags = 0;
    RT_CHECK(vx_device_query(dev, VX_CAPS_ISA_FLAGS, &isa_flags));
    if ((isa_flags & VX_ISA_EXT_DXA) == 0) {
        std::cerr << "Error: DXA ISA extension is disabled in runtime CONFIGS." << std::endl;
        return -1;
    }

    uint64_t num_threads_q = 0, num_warps_q = 0, lmem_size = 0;
    RT_CHECK(vx_device_query(dev, VX_CAPS_NUM_THREADS,    &num_threads_q));
    RT_CHECK(vx_device_query(dev, VX_CAPS_NUM_WARPS,      &num_warps_q));
    RT_CHECK(vx_device_query(dev, VX_CAPS_LOCAL_MEM_SIZE, &lmem_size));
    uint32_t num_threads = static_cast<uint32_t>(num_threads_q);
    uint32_t num_warps   = static_cast<uint32_t>(num_warps_q);

    // block_size_r: rows (= warps) per CTA. Starts at num_warps/2 (or the
    // user override); if that can't fit even d_tile=1 given the LMEM
    // budget, falls back to progressively smaller block_size_r
    uint32_t r_start = r_override ? r_override
                     : std::max(1u, std::min(num_warps / 2, N));
    if (r_start > num_warps) {
        printf("Error: block_size_r %u exceeds NUM_WARPS %u\n", r_start, num_warps);
        return -1;
    }

    uint32_t block_size_r = 0, block_size_c = 0, d_tile = 0, occupancy = 0;
    uint64_t local_mem = 0;

    for (uint32_t r = r_start; r >= 1; r /= 2) {
        uint32_t c = c_override ? c_override : std::max(r, num_threads);
        uint32_t dt = 0, occ = 0;
        uint64_t lmem = 0;

        // The strided cooperative loops over block_size_c assign each lane
        // c/num_threads columns, so must divide evenly.
        if ((c % num_threads) == 0) {
            uint64_t fixed = sizeof(TYPE) * (2 * static_cast<uint64_t>(d) * r +
                                             static_cast<uint64_t>(r) * c);
            if (d_override) {
                uint64_t cand_lmem = fixed + sizeof(TYPE) * 4 *
                                     static_cast<uint64_t>(std::min(d_override, d)) * c;
                if (cand_lmem <= lmem_size) {
                    dt = std::min(d_override, d);
                    lmem = cand_lmem;
                    occ = static_cast<uint32_t>(std::max<uint64_t>(1, lmem_size / std::max<uint64_t>(1, lmem)));
                    occ = std::min(occ, num_warps / r);
                }
            } else {
                for (uint32_t target = std::max(1u, num_warps / r); target >= 1; --target) {
                    uint64_t budget = lmem_size / target;
                    if (fixed >= budget) {
                        if (target == 1) break;
                        continue;
                    }
                    uint64_t per_col = sizeof(TYPE) * 4 * static_cast<uint64_t>(c);
                    uint64_t max_tile = (budget - fixed) / per_col;
                    dt = static_cast<uint32_t>(std::min<uint64_t>(d, std::max<uint64_t>(1, max_tile)));
                    lmem = fixed + per_col * dt;
                    occ = target;
                    break;
                }
            }
        }

        if (dt > 0) {
            block_size_r = r;
            block_size_c = c;
            d_tile       = dt;
            local_mem    = lmem;
            occupancy    = occ;
            break;
        }
        if (r_override || r == 1) break;
    }

    if (d_tile == 0) {
        printf("Error: no block_size_r from %u down to 1 fits the LMEM budget "
               "(%llu bytes)\n", r_start, (unsigned long long)lmem_size);
        return -1;
    }

    uint32_t size = N * d;
    uint32_t buf_size = size * sizeof(TYPE);

    std::cout << "num_threads=" << num_threads << " num_warps=" << num_warps
               << " lmem_size=" << lmem_size << " bytes" << std::endl;
    std::cout << "block_size_r=" << block_size_r << " block_size_c=" << block_size_c
               << " head_dim_tile=" << d_tile << std::endl;
    std::cout << "local memory: " << local_mem << " bytes, occupancy=" << occupancy << std::endl;

    vx_buffer_h Q_buf = nullptr, K_buf = nullptr, V_buf = nullptr, O_buf = nullptr;
    RT_CHECK(vx_buffer_create(dev, buf_size, VX_MEM_READ,  &Q_buf));
    RT_CHECK(vx_buffer_create(dev, buf_size, VX_MEM_READ,  &K_buf));
    RT_CHECK(vx_buffer_create(dev, buf_size, VX_MEM_READ,  &V_buf));
    RT_CHECK(vx_buffer_create(dev, buf_size, VX_MEM_WRITE, &O_buf));

    vx_module_h mod = nullptr;
    vx_kernel_h kernel = nullptr;
    RT_CHECK(vx_module_load_file(dev, kernel_file, &mod));
    RT_CHECK(vx_module_get_kernel(mod, "main", &kernel));

    kernel_arg_t kernel_arg{};
    kernel_arg.seq_len       = N;
    kernel_arg.head_dim      = d;
    kernel_arg.head_dim_tile = d_tile;
    kernel_arg.block_size_r  = block_size_r;
    kernel_arg.block_size_c  = block_size_c;
    RT_CHECK(vx_buffer_address(Q_buf, &kernel_arg.Q_addr));
    RT_CHECK(vx_buffer_address(K_buf, &kernel_arg.K_addr));
    RT_CHECK(vx_buffer_address(V_buf, &kernel_arg.V_addr));
    RT_CHECK(vx_buffer_address(O_buf, &kernel_arg.O_addr));

    // DXA descriptor slots.
    // K/V are both [seq_len, head_dim] row-major: head_dim (dim0) is the
    // contiguous axis, seq_len (dim1) is reached via stride0_bytes.
    constexpr uint32_t kDescK = 0;
    constexpr uint32_t kDescV = 1;
    RT_CHECK(vortex::dxa::program_2d(dev, kDescK, kernel_arg.K_addr,
        /*size0=*/d, /*size1=*/N,
        /*stride0_bytes=*/d * sizeof(TYPE),
        /*tile0=*/d_tile, /*tile1=*/block_size_c,
        /*elem_bytes=*/sizeof(TYPE)));
    RT_CHECK(vortex::dxa::program_2d(dev, kDescV, kernel_arg.V_addr,
        /*size0=*/d, /*size1=*/N,
        /*stride0_bytes=*/d * sizeof(TYPE),
        /*tile0=*/d_tile, /*tile1=*/block_size_c,
        /*elem_bytes=*/sizeof(TYPE)));

    std::vector<TYPE> h_Q(size), h_K(size), h_V(size), h_O(size);
    for (uint32_t i = 0; i < size; ++i) {
        h_Q[i] = Comparator<TYPE>::generate();
        h_K[i] = Comparator<TYPE>::generate();
        h_V[i] = Comparator<TYPE>::generate();
    }

    RT_CHECK(vx_enqueue_write(q, Q_buf, 0, h_Q.data(), buf_size, 0, nullptr, nullptr));
    RT_CHECK(vx_enqueue_write(q, K_buf, 0, h_K.data(), buf_size, 0, nullptr, nullptr));
    RT_CHECK(vx_enqueue_write(q, V_buf, 0, h_V.data(), buf_size, 0, nullptr, nullptr));

    vx_launch_info_t li{};
    li.struct_size  = sizeof(li);
    li.kernel       = kernel;
    li.args_host    = &kernel_arg;
    li.args_size    = sizeof(kernel_arg);
    li.ndim         = 2;
    li.grid_dim[0]  = (N + block_size_r - 1) / block_size_r;
    li.grid_dim[1]  = 1;
    li.block_dim[0] = num_threads;
    li.block_dim[1] = block_size_r;
    li.lmem_size    = local_mem;

    vx_event_h launch_ev = nullptr, read_ev = nullptr;
    RT_CHECK(vx_enqueue_launch(q, &li, 0, nullptr, &launch_ev));
    RT_CHECK(vx_enqueue_read(q, h_O.data(), O_buf, 0, buf_size, 1, &launch_ev, &read_ev));
    RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));

    auto t_end = std::chrono::high_resolution_clock::now();
    std::printf("Elapsed: %ld ms\n",
        (long)std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count());

    vx_event_release(read_ev);
    vx_event_release(launch_ev);

    int errors = 0;
    std::vector<TYPE> h_ref(size);
    flash_attention_cpu(h_ref.data(), h_Q.data(), h_K.data(), h_V.data(), N, d);
    for (uint32_t i = 0; i < size; ++i) {
        if (!Comparator<TYPE>::compare(h_ref[i], h_O[i], i, errors)) {
            ++errors;
        }
    }

    vx_buffer_release(Q_buf);
    vx_buffer_release(K_buf);
    vx_buffer_release(V_buf);
    vx_buffer_release(O_buf);
    vx_kernel_release(kernel);
    vx_module_release(mod);
    vx_queue_release(q);
    vx_device_dump_perf(dev, stdout);
    vx_device_release(dev);

    if (errors) {
        std::cout << "Found " << errors << " errors!\nFAILED!" << std::endl;
        return errors;
    }
    std::cout << "PASSED!" << std::endl;
    return 0;
}
