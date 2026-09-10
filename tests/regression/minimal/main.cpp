// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

// minimal — the smallest end-to-end regression test.
//
// One output buffer, one launch, one readback. There are no input uploads and
// no host-side reference computation, so the only things under test are the
// device-open path, module load, the CP launch, and the readback. Intended as
// the first thing to run on a new or suspect backend: if `minimal` fails there
// is no point looking at vecadd or sgemm, and if it passes while they fail the
// fault is in the workload's data movement rather than the command path.
//
// It also stages progress prints so a hang is attributable to a specific step
// rather than to "the test".
//
// `-l` drops to loopback: write a pattern to device memory and read it straight
// back, with no module, kernel or launch. That splits "the device address map
// is wrong" from "the core never ran" -- two failures that look identical from
// the outside -- and it is cheap enough to run under an RTL simulator, where a
// real launch can take tens of minutes.

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

// Unbuffered so the last line printed before a hang is the truth, even when
// the process is killed by a timeout rather than exiting.
#define STEP(msg) do { std::printf("[minimal] %s\n", (msg)); std::fflush(stdout); } while (0)

namespace {
const char* kernel_file = "kernel.vxbin";
uint32_t    size        = 1;
bool        loopback    = false;

void parse_args(int argc, char** argv) {
    int c;
    while ((c = getopt(argc, argv, "n:k:lh")) != -1) {
        switch (c) {
            case 'n': size        = std::atoi(optarg); break;
            case 'k': kernel_file = optarg;            break;
            case 'l': loopback    = true;              break;
            default:
                std::cout << "Usage: [-k kernel] [-n words] [-l] [-h]\n"
                             "  -l  loopback: write a pattern and read it back "
                             "with no launch, so a failure is the DMA path "
                             "rather than kernel execution"
                          << std::endl;
                std::exit(c == 'h' ? 0 : -1);
        }
    }
}
} // namespace

int main(int argc, char** argv) {
    parse_args(argc, argv);

    const uint32_t num_points = size;
    const uint64_t buf_size   = num_points * sizeof(uint32_t);
    std::cout << "minimal vortex2: n=" << num_points
              << " buf=" << buf_size << "B" << std::endl;

    STEP("device_open");
    vx_device_h dev = nullptr;
    CHECK(vx_device_open(0, &dev));

    STEP("queue_create");
    vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
    vx_queue_h q = nullptr;
    CHECK(vx_queue_create(dev, &qi, &q));

    STEP("buffer_create");
    vx_buffer_h dst_buf = nullptr;
    CHECK(vx_buffer_create(dev, buf_size, VX_MEM_WRITE, &dst_buf));

    // Loopback mode stops here: no module, no kernel, no launch. The only
    // things exercised are the CP's host->device and device->host DMA, which
    // is what tells a broken device address map apart from a core that never
    // ran. It is also fast enough to be usable under an RTL simulator, where
    // a full launch can take longer than a coffee break.
    vx_module_h mod = nullptr;
    vx_kernel_h kern = nullptr;
    kernel_arg_t kernel_arg{};
    kernel_arg.num_points = num_points;
    CHECK(vx_buffer_address(dst_buf, &kernel_arg.dst_addr));

    if (!loopback) {
        STEP("module_load");
        CHECK(vx_module_load_file(dev, kernel_file, &mod));
        CHECK(vx_module_get_kernel(mod, "main", &kern));
    }

    // In loopback the uploaded pattern IS the expected result, so write the
    // magic the kernel would have written. Otherwise upload a value the kernel
    // never writes, so a readback of 0xDEADBEEF means "kernel did not run"
    // rather than "kernel wrote zero".
    //
    // Separate source and destination vectors: the readback target starts as
    // 0xDEADBEEF so a verify that passes cannot be bytes we already had in
    // hand, and the upload source must stay untouched because vx_enqueue_write
    // reads it asynchronously.
    std::vector<uint32_t> h_src(num_points);
    for (uint32_t i = 0; i < num_points; ++i)
        h_src[i] = loopback ? (MINIMAL_MAGIC | i) : 0xDEADBEEFu;
    std::vector<uint32_t> h_dst(num_points, 0xDEADBEEFu);

    STEP("write");
    CHECK(vx_enqueue_write(q, dst_buf, 0, h_src.data(), buf_size, 0, nullptr, nullptr));

    vx_event_h launch_ev = nullptr, read_ev = nullptr;
    if (!loopback) {
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

        STEP("launch");
        CHECK(vx_enqueue_launch(q, &li, 0, nullptr, &launch_ev));
    }

    STEP("readback");
    CHECK(vx_enqueue_read(q, h_dst.data(), dst_buf, 0, buf_size,
                          launch_ev ? 1 : 0, launch_ev ? &launch_ev : nullptr,
                          &read_ev));

    STEP("wait");
    CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));

    STEP("verify");
    int errors = 0;
    for (uint32_t i = 0; i < num_points; ++i) {
        const uint32_t ref = MINIMAL_MAGIC | i;
        if (h_dst[i] != ref) {
            if (errors < 16)
                std::printf("*** [%u] expected=0x%08x actual=0x%08x%s\n",
                            i, ref, h_dst[i],
                            h_dst[i] == 0xDEADBEEFu
                                ? (loopback ? "  (readback returned nothing)"
                                            : "  (kernel never wrote it)")
                                : "");
            ++errors;
        }
    }

    vx_event_release(read_ev);
    if (launch_ev) vx_event_release(launch_ev);
    vx_buffer_release(dst_buf);
    if (kern) vx_kernel_release(kern);
    if (mod)  vx_module_release(mod);
    vx_queue_release(q);
    vx_device_release(dev);

    if (errors) {
        std::cout << "Found " << errors << " errors!\nFAILED!" << std::endl;
        return 1;
    }
    std::cout << "PASSED!" << std::endl;
    return 0;
}
