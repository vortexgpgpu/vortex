// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

// trap — synchronous traps taken by many warps at once.
//
// Every warp installs the same handler, then traps in a loop: even warps with
// ECALL, odd warps with EBREAK, each at its own site. The handler steps mepc
// past the trap, so an MRET that misses that write traps again. Each thread
// reports whether its warp saw its own cause and return PC, so a return PC or
// cause shared between warps fails the test.

#include <vortex2.h>
#include "common.h"

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
uint32_t    size        = 256;
uint32_t    iters       = 8;

void parse_args(int argc, char** argv) {
    int c;
    while ((c = getopt(argc, argv, "n:i:k:h")) != -1) {
        switch (c) {
            case 'n': size        = std::atoi(optarg); break;
            case 'i': iters       = std::atoi(optarg); break;
            case 'k': kernel_file = optarg;            break;
            default:
                std::cout << "Usage: [-k kernel] [-n threads] [-i traps per warp] [-h]" << std::endl;
                std::exit(c == 'h' ? 0 : -1);
        }
    }
}
} // namespace

int main(int argc, char** argv) {
    parse_args(argc, argv);

    const uint32_t num_points = size;
    const uint64_t buf_size   = num_points * sizeof(uint32_t);
    std::cout << "trap: n=" << num_points << " iters=" << iters << std::endl;

    vx_device_h dev = nullptr;
    CHECK(vx_device_open(0, &dev));

    vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
    vx_queue_h q = nullptr;
    CHECK(vx_queue_create(dev, &qi, &q));

    vx_buffer_h dst_buf = nullptr;
    CHECK(vx_buffer_create(dev, buf_size, VX_MEM_WRITE, &dst_buf));

    kernel_arg_t kernel_arg{};
    kernel_arg.num_points = num_points;
    kernel_arg.iters      = iters;
    CHECK(vx_buffer_address(dst_buf, &kernel_arg.dst_addr));

    vx_module_h mod = nullptr;
    vx_kernel_h kern = nullptr;
    CHECK(vx_module_load_file(dev, kernel_file, &mod));
    CHECK(vx_module_get_kernel(mod, "main", &kern));

    // 0xDEADBEEF has bits outside the result mask, so an unwritten word fails.
    std::vector<uint32_t> h_src(num_points, 0xDEADBEEFu);
    std::vector<uint32_t> h_dst(num_points, 0xDEADBEEFu);
    CHECK(vx_enqueue_write(q, dst_buf, 0, h_src.data(), buf_size, 0, nullptr, nullptr));

    uint32_t grid[1], block[1];
    CHECK(vx_device_max_occupancy_grid(dev, 1, &num_points, grid, block));

    vx_launch_info_t li{};
    li.struct_size  = sizeof(li);
    li.kernel       = kern;
    li.args_host    = &kernel_arg;
    li.args_size    = sizeof(kernel_arg);
    li.ndim         = 1;
    li.grid_dim[0]  = grid[0];
    li.block_dim[0] = block[0];

    vx_event_h launch_ev = nullptr, read_ev = nullptr;
    CHECK(vx_enqueue_launch(q, &li, 0, nullptr, &launch_ev));
    CHECK(vx_enqueue_read(q, h_dst.data(), dst_buf, 0, buf_size, 1, &launch_ev, &read_ev));
    CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));

    int errors = 0;
    for (uint32_t i = 0; i < num_points; ++i) {
        if (h_dst[i] != 0) {
            if (errors < 16) {
                std::printf("*** [%u] result=0x%08x%s%s%s%s\n", i, h_dst[i],
                            (h_dst[i] & TRAP_BAD_CAUSE) ? " cause" : "",
                            (h_dst[i] & TRAP_BAD_EPC)   ? " epc" : "",
                            (h_dst[i] & TRAP_BAD_TVEC)  ? " tvec" : "",
                            (h_dst[i] & TRAP_BAD_TVAL)  ? " tval" : "");
            }
            ++errors;
        }
    }

    vx_event_release(read_ev);
    vx_event_release(launch_ev);
    vx_buffer_release(dst_buf);
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
