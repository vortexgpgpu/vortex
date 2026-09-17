// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

// attention regression test.

#include <vortex2.h>
#include "common.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <unistd.h>
#include <vector>
#include <VX_types.h>

#define FLOAT_ULP 6

#define CHECK(expr) do { \
    vx_result_t _r = (expr); \
    if (_r != VX_SUCCESS) { \
        std::fprintf(stderr, "FAIL %s:%d: '%s' returned %s\n", \
                     __FILE__, __LINE__, #expr, vx_result_string(_r)); \
        std::exit(1); \
    } \
} while (0)

namespace {
const char* kernel_file = "kernel.vxbin";
uint32_t N = 64;
uint32_t d = 4;

static void parse_args(int argc, char **argv) {
  	int c;
	while ((c = getopt(argc, argv, "n:d:k:h")) != -1) {
		switch (c) {
			case 'n': N 		  = std::atoi(optarg); break;
			case 'd': d			  = std::atoi(optarg); break;
			case 'k': kernel_file = optarg; 	 	   break;
			default:
				std::cout << "Usage: [-k: kernel] [-n: sequence_length] [-d: head_dim] [-h]" << std::endl;
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

static void matmul_cpu(TYPE* out, const TYPE* A, const TYPE* B, uint32_t M, uint32_t N, uint32_t K) {
	for (uint32_t row = 0; row < M; ++row) {
		for (uint32_t col = 0; col < N; ++col) {
			TYPE sum(0);
			for (uint32_t e = 0; e < K; ++e) {
				sum += A[row * K + e] * B[e * N + col];
			}
			out[row * N + col] = sum;
		}
	}
}

static void softmax_cpu(TYPE* out, const TYPE* A, uint32_t M, uint32_t N) {
	for (uint32_t row = 0; row < M; ++row) {
		TYPE max_val = A[row * N];
		for (uint32_t col = 1; col < N; ++col) {
			max_val = std::max(max_val, A[row * N + col]);
		}
		TYPE exp_sum = 0;
		for (uint32_t col = 0; col < N; ++col) {
			out[row * N + col] = std::exp(A[row * N + col] - max_val);
			exp_sum += out[row * N + col];
		}
		for (uint32_t col = 0; col < N; ++col) {
			out[row * N + col] /= exp_sum;
		}
	}
}
} // namespace

int main(int argc, char *argv[]) {
  	parse_args(argc, argv);
  	std::srand(50);

	std::cout << "attention vortex2: " << N << "x" << d << std::endl;

    vx_device_h dev = nullptr;
    CHECK(vx_device_open(0, &dev));

	auto t_start = std::chrono::high_resolution_clock::now();

    vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
    vx_queue_h q = nullptr;
    CHECK(vx_queue_create(dev, &qi, &q));

    uint64_t num_cores = 0;
    CHECK(vx_device_query(dev, VX_CAPS_NUM_CORES, &num_cores));

	// Performance query helper that doesn't print output
    auto query_perf = [&](uint64_t* out_instrs, uint64_t* out_cycles) {
        uint64_t total_instrs = 0, max_cycles = 0;
        for (uint32_t c = 0; c < (uint32_t)num_cores; ++c) {
            uint64_t ci = 0, cc = 0;
            CHECK(vx_device_mpm_query(dev, VX_DCR_MPM_CLASS_BASE, VX_CSR_MINSTRET, c, &ci));
            CHECK(vx_device_mpm_query(dev, VX_DCR_MPM_CLASS_BASE, VX_CSR_MCYCLE,   c, &cc));
            total_instrs += ci;
            max_cycles = std::max(max_cycles, cc);
        }
        *out_instrs = total_instrs;
        *out_cycles = max_cycles;
    };

	uint32_t nd_size = N * d;
	uint32_t nd_buf_size = nd_size * sizeof(TYPE);
	uint32_t nn_size = N * N;
	uint32_t nn_buf_size = nn_size * sizeof(TYPE);

    vx_buffer_h Q_buf=nullptr, K_buf=nullptr, S_buf=nullptr, P_buf=nullptr, V_buf=nullptr, O_buf=nullptr;
    CHECK(vx_buffer_create(dev, nd_buf_size, VX_MEM_READ,  &Q_buf));
    CHECK(vx_buffer_create(dev, nd_buf_size, VX_MEM_READ,  &K_buf));
    CHECK(vx_buffer_create(dev, nn_buf_size, VX_MEM_READ_WRITE, &S_buf));
    CHECK(vx_buffer_create(dev, nn_buf_size, VX_MEM_READ_WRITE,  &P_buf));
    CHECK(vx_buffer_create(dev, nd_buf_size, VX_MEM_READ,  &V_buf));
    CHECK(vx_buffer_create(dev, nd_buf_size, VX_MEM_WRITE, &O_buf));

	vx_module_h mod = nullptr;
    vx_kernel_h k_qk = nullptr, k_softmax = nullptr, k_pv = nullptr;
    CHECK(vx_module_load_file(dev, kernel_file, &mod));
    CHECK(vx_module_get_kernel(mod, "kernel_qk", &k_qk));
	CHECK(vx_module_get_kernel(mod, "kernel_softmax", &k_softmax));
	CHECK(vx_module_get_kernel(mod, "kernel_pv", &k_pv));

    kernel_arg_t kernel_arg{};
    kernel_arg.N = N, kernel_arg.d = d;
    CHECK(vx_buffer_address(Q_buf, &kernel_arg.Q_addr));
    CHECK(vx_buffer_address(K_buf, &kernel_arg.K_addr));
    CHECK(vx_buffer_address(S_buf, &kernel_arg.S_addr));
	CHECK(vx_buffer_address(P_buf, &kernel_arg.P_addr));
    CHECK(vx_buffer_address(V_buf, &kernel_arg.V_addr));
    CHECK(vx_buffer_address(O_buf, &kernel_arg.O_addr));

    std::vector<TYPE> h_Q(nd_size), h_K(nd_size), h_S(nn_size), h_P(nn_size), h_V(nd_size), h_O(nd_size);
    for (uint32_t i = 0; i < nd_size; ++i) {
        h_Q[i] = Comparator<TYPE>::generate();
        h_K[i] = Comparator<TYPE>::generate();
		h_V[i] = Comparator<TYPE>::generate();
    }

	// -------------------------------------------------------------------------------------------------
	// S = QK^T

	std::cout << "Calculate S = QK^T:" << std::endl;

	auto t0 = std::chrono::high_resolution_clock::now();

	CHECK(vx_enqueue_write(q, Q_buf, 0, h_Q.data(), nd_buf_size, 0,nullptr,nullptr));
    CHECK(vx_enqueue_write(q, K_buf, 0, h_K.data(), nd_buf_size, 0,nullptr,nullptr));

	vx_launch_info_t li0{};
    li0.struct_size = sizeof(li0);
    li0.kernel      = k_qk;
    li0.args_host   = &kernel_arg;
    li0.args_size   = sizeof(kernel_arg);
    li0.ndim        = 2;
    li0.grid_dim[0]  = N; li0.grid_dim[1]  = N;
    li0.block_dim[0] = 1; li0.block_dim[1] = 1;

    vx_event_h launch_ev0=nullptr, read_ev0=nullptr;
    CHECK(vx_enqueue_launch(q, &li0, 0, nullptr, &launch_ev0));
    CHECK(vx_enqueue_read(q, h_S.data(), S_buf, 0, nn_buf_size,
                          1, &launch_ev0, &read_ev0));
    CHECK(vx_event_wait_value(read_ev0, 1, VX_TIMEOUT_INFINITE));
    auto t1 = std::chrono::high_resolution_clock::now();
    std::printf("  Elapsed: %ld ms\n",
        (long)std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count());

    int errors = 0;
    std::vector<TYPE> h_ref0(nn_size);
	matmul_cpu(h_ref0.data(), h_Q.data(), h_K.data(), N, N, d);
    for (uint32_t i = 0; i < nn_size; ++i) {
        if (!Comparator<TYPE>::compare(h_ref0[i], h_S[i], i, errors)) {
            ++errors;
        }
    }

	vx_event_release(read_ev0);
    vx_event_release(launch_ev0);

	uint64_t prev_instrs = 0, prev_cycles = 0;
	uint64_t instrs 	 = 0, cycles 	  = 0;
	query_perf(&instrs, &cycles);
	auto instr_delta = instrs - prev_instrs, cycle_delta = cycles - prev_cycles;
	std::cout << "  PERF: instrs=" << instr_delta << ", cycles=" << cycle_delta << ", IPC=" << static_cast<float>(instr_delta) / cycle_delta << std::endl;
	prev_instrs = instrs, prev_cycles = cycles;

	if (errors) {
        std::cout << "Found " << errors << " errors!\nFAILED!" << std::endl;
        return errors;
    }

	// -------------------------------------------------------------------------------------------------
	// P = softmax(S)

  	std::cout << "Calculate P = softmax(S):" << std::endl;

	t0 = std::chrono::high_resolution_clock::now();

	vx_launch_info_t li1{};
    li1.struct_size = sizeof(li1);
    li1.kernel      = k_softmax;
    li1.args_host   = &kernel_arg;
    li1.args_size   = sizeof(kernel_arg);
    li1.ndim        = 1;
    li1.grid_dim[0]  = N;
    li1.block_dim[0] = 1;

    vx_event_h launch_ev1=nullptr, read_ev1=nullptr;
    CHECK(vx_enqueue_launch(q, &li1, 0, nullptr, &launch_ev1));
    CHECK(vx_enqueue_read(q, h_P.data(), P_buf, 0, nn_buf_size,
                          1, &launch_ev1, &read_ev1));
    CHECK(vx_event_wait_value(read_ev1, 1, VX_TIMEOUT_INFINITE));
    t1 = std::chrono::high_resolution_clock::now();
    std::printf("  Elapsed: %ld ms\n",
        (long)std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count());

    errors = 0;
    std::vector<TYPE> h_ref1(nn_size);
    softmax_cpu(h_ref1.data(), h_S.data(), N, N);
    for (uint32_t i = 0; i < nn_size; ++i) {
        if (!Comparator<TYPE>::compare(h_ref1[i], h_P[i], i, errors)) {
            ++errors;
        }
    }

	vx_event_release(read_ev1);
    vx_event_release(launch_ev1);

	query_perf(&instrs, &cycles);
	instr_delta = instrs - prev_instrs, cycle_delta = cycles - prev_cycles;
	std::cout << "  PERF: instrs=" << instr_delta << ", cycles=" << cycle_delta << ", IPC=" << static_cast<float>(instr_delta) / cycle_delta << std::endl;
	prev_instrs = instrs, prev_cycles = cycles;

	if (errors) {
        std::cout << "Found " << errors << " errors!\nFAILED!" << std::endl;
        return errors;
    }

	// -------------------------------------------------------------------------------------------------
	// O = PV

	std::cout << "Calculate O = PV:" << std::endl;

  	t0 = std::chrono::high_resolution_clock::now();

	CHECK(vx_enqueue_write(q, V_buf, 0, h_V.data(), nd_buf_size, 0,nullptr,nullptr));

	vx_launch_info_t li2{};
    li2.struct_size = sizeof(li2);
    li2.kernel      = k_pv;
    li2.args_host   = &kernel_arg;
    li2.args_size   = sizeof(kernel_arg);
    li2.ndim        = 2;
    li2.grid_dim[0]  = d; li2.grid_dim[1]  = N;
    li2.block_dim[0] = 1; li2.block_dim[1] = 1;

    vx_event_h launch_ev2=nullptr, read_ev2=nullptr;
    CHECK(vx_enqueue_launch(q, &li2, 0, nullptr, &launch_ev2));
    CHECK(vx_enqueue_read(q, h_O.data(), O_buf, 0, nd_buf_size,
                          1, &launch_ev2, &read_ev2));
    CHECK(vx_event_wait_value(read_ev2, 1, VX_TIMEOUT_INFINITE));
    t1 = std::chrono::high_resolution_clock::now();
    std::printf("  Elapsed: %ld ms\n",
        (long)std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count());

    errors = 0;
    std::vector<TYPE> h_ref2(nd_size);
	matmul_cpu(h_ref2.data(), h_P.data(), h_V.data(), N, d, N);
    for (uint32_t i = 0; i < nd_size; ++i) {
        if (!Comparator<TYPE>::compare(h_ref2[i], h_O[i], i, errors)) {
            ++errors;
        }
    }

	vx_event_release(read_ev2);
    vx_event_release(launch_ev2);
    vx_buffer_release(Q_buf);
    vx_buffer_release(K_buf); 
    vx_buffer_release(S_buf);
    vx_buffer_release(P_buf);
    vx_buffer_release(V_buf);
    vx_buffer_release(O_buf);
    vx_kernel_release(k_qk);
    vx_kernel_release(k_softmax);
    vx_kernel_release(k_pv);
    vx_module_release(mod);
    vx_queue_release(q);

	query_perf(&instrs, &cycles);
	instr_delta = instrs - prev_instrs, cycle_delta = cycles - prev_cycles;
	std::cout << "  PERF: instrs=" << instr_delta << ", cycles=" << cycle_delta << ", IPC=" << static_cast<float>(instr_delta) / cycle_delta << std::endl;
	
	auto t_end = std::chrono::high_resolution_clock::now();
    std::printf("Elapsed: %ld ms\n",
        (long)std::chrono::duration_cast<std::chrono::milliseconds>(t_end-t_start).count());
	vx_device_dump_perf(dev, stdout);
	vx_device_release(dev);

    if (errors) {
        std::cout << "Found " << errors << " errors!\nFAILED!" << std::endl;
        return errors;
    }
    std::cout << "PASSED!" << std::endl;
    return 0;
}
