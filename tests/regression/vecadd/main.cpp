// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

// vecadd — vortex2.h-native regression test.
//
// The async pattern: every host→device upload is fire-and-forget into
// the queue worker; the launch produces an event; the dst readback
// gates on that event; the host waits exactly once at the end. The
// per-queue worker (runtime impl §4.6.1) serializes everything in
// FIFO order, so no inter-step host sync is needed.

#include <vortex2.h>
#include "common.h"

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
uint32_t    size        = 16;

void parse_args(int argc, char** argv) {
    int c;
    while ((c = getopt(argc, argv, "n:k:h")) != -1) {
        switch (c) {
            case 'n': size        = std::atoi(optarg); break;
            case 'k': kernel_file = optarg;            break;
            default:
                std::cout << "Usage: [-k kernel] [-n words] [-h]" << std::endl;
                std::exit(c == 'h' ? 0 : -1);
        }
    }
}

bool float_eq(float a, float b) {
    union fi { float f; int32_t i; };
    fi fa{a}, fb{b};
    return std::abs(fa.i - fb.i) <= 6;
}
} // namespace

int main(int argc, char** argv) {
    parse_args(argc, argv);
    std::srand(50);

    const uint32_t num_points = size;
    const uint64_t buf_size   = num_points * sizeof(TYPE);
    std::cout << "vecadd vortex2: n=" << num_points
              << " buf=" << buf_size << "B" << std::endl;

    vx_device_h dev = nullptr;
    CHECK(vx_device_open(0, &dev));

    vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
    vx_queue_h q = nullptr;
    CHECK(vx_queue_create(dev, &qi, &q));

    vx_buffer_h src0_buf=nullptr, src1_buf=nullptr, dst_buf=nullptr;
    CHECK(vx_buffer_create(dev, buf_size, VX_MEM_READ,  &src0_buf));
    CHECK(vx_buffer_create(dev, buf_size, VX_MEM_READ,  &src1_buf));
    CHECK(vx_buffer_create(dev, buf_size, VX_MEM_WRITE, &dst_buf));

    vx_module_h mod = nullptr;
    vx_kernel_h kern = nullptr;
    CHECK(vx_module_load_file(dev, kernel_file, &mod));
    CHECK(vx_module_get_kernel(mod, "main", &kern));

    kernel_arg_t kernel_arg{};
    kernel_arg.num_points = num_points;
    CHECK(vx_buffer_address(src0_buf, &kernel_arg.src0_addr));
    CHECK(vx_buffer_address(src1_buf, &kernel_arg.src1_addr));
    CHECK(vx_buffer_address(dst_buf,  &kernel_arg.dst_addr));

    std::vector<TYPE> h_src0(num_points), h_src1(num_points), h_dst(num_points);
    for (uint32_t i = 0; i < num_points; ++i) {
        h_src0[i] = static_cast<TYPE>(std::rand()) / RAND_MAX;
        h_src1[i] = static_cast<TYPE>(std::rand()) / RAND_MAX;
    }

    // ----- Async chain: 2 writes → launch → read → 1 wait -----
    // The kernel-args block is passed straight to the launch as a host
    // blob (Phase 2 UVA args) — no args device buffer to create or upload.
    CHECK(vx_enqueue_write(q, src0_buf, 0, h_src0.data(), buf_size, 0,nullptr,nullptr));
    CHECK(vx_enqueue_write(q, src1_buf, 0, h_src1.data(), buf_size, 0,nullptr,nullptr));

    uint32_t grid[1], block[1];
    CHECK(vx_device_max_occupancy_grid(dev, 1, &num_points, grid, block));

    vx_launch_info_t li{};
    li.struct_size = sizeof(li);
    li.kernel      = kern;
    li.args_host   = &kernel_arg;
    li.args_size   = sizeof(kernel_arg);
    li.ndim        = 1;
    li.grid_dim[0] = grid[0];
    li.block_dim[0]= block[0];

    vx_event_h launch_ev=nullptr, read_ev=nullptr;
    CHECK(vx_enqueue_launch(q, &li, 0, nullptr, &launch_ev));
    CHECK(vx_enqueue_read(q, h_dst.data(), dst_buf, 0, buf_size,
                          1, &launch_ev, &read_ev));
    CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));

    int errors = 0;
    for (uint32_t i = 0; i < num_points; ++i) {
        TYPE ref = h_src0[i] + h_src1[i];
        if (!float_eq(h_dst[i], ref)) {
            if (errors < 16)
                std::printf("*** [%u] expected=%f actual=%f\n", i, ref, h_dst[i]);
            ++errors;
        }
    }

    vx_event_release(read_ev);
    vx_event_release(launch_ev);
    vx_buffer_release(dst_buf);
    vx_buffer_release(src1_buf);
    vx_buffer_release(src0_buf);
    vx_kernel_release(kern);
    vx_module_release(mod);
    vx_queue_release(q);
    vx_device_dump_perf(dev, stdout);
    vx_device_release(dev);

    if (errors) {
        std::cout << "Found " << errors << " errors!\nFAILED!" << std::endl;
        return 1;
    }
    std::cout << "PASSED!" << std::endl;
    return 0;
}
