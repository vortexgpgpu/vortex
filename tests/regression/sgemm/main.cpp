// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

// sgemm — vortex2.h-native regression test.
//
// Same async pattern as vecadd v2: 3 fire-and-forget uploads (A, B,
// args) + 1 launch + 1 read gated on launch + 1 trailing wait. The
// per-queue worker thread serializes everything in FIFO order.

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
uint32_t    size        = 64;

void parse_args(int argc, char** argv) {
    int c;
    while ((c = getopt(argc, argv, "n:k:h")) != -1) {
        switch (c) {
            case 'n': size        = std::atoi(optarg); break;
            case 'k': kernel_file = optarg;            break;
            default:
                std::cout << "Usage: [-k kernel] [-n size] [-h]" << std::endl;
                std::exit(c == 'h' ? 0 : -1);
        }
    }
}

bool float_eq(float a, float b) {
    union fi { float f; int32_t i; };
    fi fa{a}, fb{b};
    return std::abs(fa.i - fb.i) <= 6;
}

void matmul_cpu(TYPE* out, const TYPE* A, const TYPE* B, uint32_t n) {
    for (uint32_t r = 0; r < n; ++r)
        for (uint32_t c = 0; c < n; ++c) {
            TYPE s(0);
            for (uint32_t e = 0; e < n; ++e) s += A[r*n + e] * B[e*n + c];
            out[r*n + c] = s;
        }
}
} // namespace

int main(int argc, char** argv) {
    parse_args(argc, argv);
    std::srand(50);

    const uint32_t size_sq  = size * size;
    const uint64_t buf_size = size_sq * sizeof(TYPE);
    std::cout << "sgemm vortex2: " << size << "x" << size << std::endl;

    vx_device_h dev = nullptr;
    CHECK(vx_device_open(0, &dev));

    vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
    vx_queue_h q = nullptr;
    CHECK(vx_queue_create(dev, &qi, &q));

    const uint32_t global_dim[2] = {size, size};
    uint32_t grid[2], block[2];
    CHECK(vx_device_max_occupancy_grid(dev, 2, global_dim, grid, block));
    if ((size % block[0]) || (size % block[1])) {
        std::cerr << "matrix size " << size << " must divide block "
                  << block[0] << "x" << block[1] << std::endl;
        return -1;
    }

    vx_buffer_h A_buf=nullptr, B_buf=nullptr, C_buf=nullptr;
    CHECK(vx_buffer_create(dev, buf_size, VX_MEM_READ,  &A_buf));
    CHECK(vx_buffer_create(dev, buf_size, VX_MEM_READ,  &B_buf));
    CHECK(vx_buffer_create(dev, buf_size, VX_MEM_WRITE, &C_buf));

    vx_module_h mod = nullptr;
    vx_kernel_h kern = nullptr;
    CHECK(vx_module_load_file(dev, kernel_file, &mod));
    CHECK(vx_module_get_kernel(mod, "main", &kern));

    kernel_arg_t kernel_arg{};
    kernel_arg.size = size;
    CHECK(vx_buffer_address(A_buf, &kernel_arg.A_addr));
    CHECK(vx_buffer_address(B_buf, &kernel_arg.B_addr));
    CHECK(vx_buffer_address(C_buf, &kernel_arg.C_addr));

    std::vector<TYPE> h_A(size_sq), h_B(size_sq), h_C(size_sq);
    for (uint32_t i = 0; i < size_sq; ++i) {
        h_A[i] = static_cast<TYPE>(std::rand()) / RAND_MAX;
        h_B[i] = static_cast<TYPE>(std::rand()) / RAND_MAX;
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    CHECK(vx_enqueue_write(q, A_buf, 0, h_A.data(), buf_size, 0,nullptr,nullptr));
    CHECK(vx_enqueue_write(q, B_buf, 0, h_B.data(), buf_size, 0,nullptr,nullptr));

    vx_launch_info_t li{};
    li.struct_size = sizeof(li);
    li.kernel      = kern;
    li.args_host   = &kernel_arg;
    li.args_size   = sizeof(kernel_arg);
    li.ndim        = 2;
    li.grid_dim[0] = grid[0];  li.grid_dim[1] = grid[1];
    li.block_dim[0]= block[0]; li.block_dim[1]= block[1];

    vx_event_h launch_ev=nullptr, read_ev=nullptr;
    CHECK(vx_enqueue_launch(q, &li, 0, nullptr, &launch_ev));
    CHECK(vx_enqueue_read(q, h_C.data(), C_buf, 0, buf_size,
                          1, &launch_ev, &read_ev));
    CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
    auto t1 = std::chrono::high_resolution_clock::now();
    std::printf("Elapsed: %ld ms\n",
        (long)std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count());

    int errors = 0;
    std::vector<TYPE> h_ref(size_sq);
    matmul_cpu(h_ref.data(), h_A.data(), h_B.data(), size);
    for (uint32_t i = 0; i < size_sq; ++i) {
        if (!float_eq(h_C[i], h_ref[i])) {
            if (errors < 16)
                std::printf("*** [%u] expected=%f actual=%f\n", i, h_ref[i], h_C[i]);
            ++errors;
        }
    }

    vx_event_release(read_ev);
    vx_event_release(launch_ev);
    vx_buffer_release(C_buf);
    vx_buffer_release(B_buf);
    vx_buffer_release(A_buf);
    vx_kernel_release(kern);
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
