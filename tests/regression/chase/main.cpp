// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

// vecadd — vortex2.h-native regression test.
//
// Async pattern: uploads are fire-and-forget; the launch produces an event;
// the dst readback gates on that event; the host waits once at the end.
// The per-queue worker serializes ops in FIFO order, so no inter-step host sync is needed.


// Pointer chase: load-to-use latency by footprint. One thread follows a
// random cyclic permutation at a 128 B stride, so each dependent load misses
// every cache smaller than the footprint. The latency is the difference of
// two runs' cycles over the difference of their iterations. Uploads land in
// the last-level cache, so DRAM latency takes a cold chain (-c): the buffer
// is never written and each index comes from an LCG (see common.h).
#include <vortex2.h>
#include "common.h"
#include <algorithm>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <random>
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
uint64_t    bytes       = 1 << 16;
uint32_t    iters       = 1024;
uint32_t    warm        = 1;
uint32_t    cold        = 0;

void parse_args(int argc, char** argv) {
    int c;
    while ((c = getopt(argc, argv, "b:i:w:ck:h")) != -1) {
        switch (c) {
            case 'b': bytes       = std::strtoull(optarg, nullptr, 0); break;
            case 'i': iters       = std::atoi(optarg); break;
            case 'w': warm        = std::atoi(optarg); break;
            case 'c': cold        = 1; break;
            case 'k': kernel_file = optarg; break;
            default:
                std::cout << "Usage: [-k kernel] [-b bytes] [-i iters] [-w warm passes] [-c] [-h]" << std::endl;
                std::exit(c == 'h' ? 0 : -1);
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    parse_args(argc, argv);

    const uint32_t m = uint32_t(bytes / CHASE_STRIDE);
    if (m < 2 || (cold && ((m & (m - 1)) != 0 || iters > m))) {
        std::cerr << "footprint must hold at least two 128 B elements; a cold chain needs a power of"
                     " two of them, at least one per iteration" << std::endl;
        return 1;
    }
    if (iters % CHASE_UNROLL != 0) {
        std::cerr << "iterations must be a multiple of " << CHASE_UNROLL << std::endl;
        return 1;
    }
    std::vector<uint32_t> perm(m);
    std::iota(perm.begin(), perm.end(), 0);
    std::mt19937 rng(1);
    std::shuffle(perm.begin() + 1, perm.end(), rng);

    vx_device_h dev = nullptr;
    CHECK(vx_device_open(0, &dev));
    vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
    vx_queue_h q = nullptr;
    CHECK(vx_queue_create(dev, &qi, &q));

    vx_buffer_h next_buf = nullptr, out_buf = nullptr;
    CHECK(vx_buffer_create(dev, bytes, VX_MEM_READ, &next_buf));
    CHECK(vx_buffer_create(dev, sizeof(uint64_t), VX_MEM_WRITE, &out_buf));

    vx_module_h mod = nullptr;
    vx_kernel_h kern = nullptr;
    CHECK(vx_module_load_file(dev, kernel_file, &mod));
    CHECK(vx_module_get_kernel(mod, "main", &kern));

    kernel_arg_t arg{};
    arg.warm  = warm * m;
    arg.iters = iters;
    CHECK(vx_buffer_address(next_buf, &arg.next_addr));
    CHECK(vx_buffer_address(out_buf, &arg.out_addr));
    arg.first_addr = arg.next_addr;

    // Element perm[i] points at element perm[i+1], as a device address.
    std::vector<uint8_t> h(bytes, 0);
    for (uint32_t i = 0; i < m; ++i) {
        uint64_t to = arg.next_addr + uint64_t(perm[(i + 1) % m]) * CHASE_STRIDE;
        std::memcpy(&h[size_t(perm[i]) * CHASE_STRIDE], &to, VX_CFG_XLEN / 8);
    }
    arg.cold     = cold;
    arg.elements = m;
    arg.mask     = 0;
    if (!cold)
        CHECK(vx_enqueue_write(q, next_buf, 0, h.data(), bytes, 0, nullptr, nullptr));

    vx_launch_info_t li{};
    li.struct_size = sizeof(li);
    li.kernel      = kern;
    li.args_host   = &arg;
    li.args_size   = sizeof(arg);
    li.ndim        = 1;
    li.grid_dim[0] = 1;
    li.block_dim[0]= 1;

    uint64_t result = 0;
    vx_event_h launch_ev = nullptr, read_ev = nullptr;
    CHECK(vx_enqueue_launch(q, &li, 0, nullptr, &launch_ev));
    CHECK(vx_enqueue_read(q, &result, out_buf, 0, sizeof(result), 1, &launch_ev, &read_ev));
    CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));

    uint64_t p = 0;
    if (cold) {
        for (uint32_t i = 0; i < iters; ++i)
            p = (p * CHASE_LCG_A + CHASE_LCG_C) & (m - 1);
    } else {
        std::vector<uint32_t> pos(m);
        for (uint32_t i = 0; i < m; ++i)
            pos[perm[i]] = i;
        uint32_t e = 0;
        for (uint64_t i = 0; i < uint64_t(arg.warm) + iters; ++i)
            e = perm[(pos[e] + 1) % m];
        p = arg.next_addr + uint64_t(e) * CHASE_STRIDE;
    }

    std::printf("chase bytes=%llu elements=%u warm=%u iters=%u cold=%u\n",
                (unsigned long long)bytes, m, arg.warm, iters, cold);

    vx_event_release(read_ev);
    vx_event_release(launch_ev);
    vx_buffer_release(out_buf);
    vx_buffer_release(next_buf);
    vx_kernel_release(kern);
    vx_module_release(mod);
    vx_queue_release(q);
    vx_device_dump_perf(dev, stdout);
    vx_device_release(dev);

    if (result != p) {
        std::cout << "expected " << p << " got " << result << "\nFAILED!" << std::endl;
        return 1;
    }
    std::cout << "PASSED!" << std::endl;
    return 0;
}
