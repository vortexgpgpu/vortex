// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

// ============================================================================
// sgemm — vortex2.h-native regression test.
//
// Rewritten from scratch on the async vortex2.h API. Mirrors the v2
// pattern from tests/regression/vecadd/main.cpp:
//
//   - Upload chain (matrices A + B + arg struct + kernel binary) is
//     enqueued back-to-back through the per-queue worker with no
//     inter-step host waits.
//   - The 15 KMU DCR programming writes are fire-and-forget — FIFO
//     order in the worker guarantees they commit before the launch.
//   - Launch produces a single event; the dst (matrix C) readback
//     gates on that event via vx_enqueue_read's wait-events list.
//   - The host waits exactly once at the end, on the read event.
//
// The legacy version performed seven separate synchronous waits during
// setup (one per vx_copy_to_dev × 2, kernel upload, args upload, and
// each of 15 DCR writes inside vx_start_g). The v2 version compresses
// all of that into a single trailing wait.
//
// Kernel arg struct, matmul reference, and CLI behavior are unchanged
// from the legacy version.
// ============================================================================

#include <vortex2.h>
#include <VX_config.h>
#include <VX_types.h>

#include "common.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <unistd.h>
#include <vector>

#define FLOAT_ULP 6

#define CHECK_VX(expr) do { \
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

void show_usage() {
    std::cout << "Vortex sgemm (vortex2.h-native)." << std::endl;
    std::cout << "Usage: [-k kernel] [-n size] [-h]" << std::endl;
}
void parse_args(int argc, char** argv) {
    int c;
    while ((c = getopt(argc, argv, "n:k:h")) != -1) {
        switch (c) {
            case 'n': size        = std::atoi(optarg); break;
            case 'k': kernel_file = optarg;            break;
            case 'h': show_usage(); std::exit(0);      break;
            default:  show_usage(); std::exit(-1);
        }
    }
}

bool float_eq(float a, float b) {
    union fi { float f; int32_t i; };
    fi fa = {a}, fb = {b};
    return std::abs(fa.i - fb.i) <= FLOAT_ULP;
}

void matmul_cpu(TYPE* out, const TYPE* A, const TYPE* B,
                uint32_t width, uint32_t height) {
    for (uint32_t row = 0; row < height; ++row) {
        for (uint32_t col = 0; col < width; ++col) {
            TYPE sum(0);
            for (uint32_t e = 0; e < width; ++e) {
                sum += A[row * width + e] * B[e * width + col];
            }
            out[row * width + col] = sum;
        }
    }
}

// Kernel binary loader (same as vecadd v2). The host-side `all` and
// `zeros` vectors must outlive the enqueued writes; we sync on the
// upload events before returning so the caller sees a fully-resident
// kernel image.
vx_result_t load_kernel_v2(vx_device_h dev, vx_queue_h q,
                           const char* path, vx_buffer_h* out_buf) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        std::fprintf(stderr, "cannot open %s\n", path);
        return VX_ERR_INVALID_VALUE;
    }
    ifs.seekg(0, ifs.end);
    auto file_sz = (size_t)ifs.tellg();
    ifs.seekg(0, ifs.beg);
    if (file_sz < 16) return VX_ERR_INVALID_VALUE;

    std::vector<uint8_t> all(file_sz);
    ifs.read(reinterpret_cast<char*>(all.data()), file_sz);

    auto* hdr        = reinterpret_cast<const uint64_t*>(all.data());
    uint64_t min_vma = hdr[0];
    uint64_t max_vma = hdr[1];
    uint64_t bin_sz  = file_sz - 16;
    uint64_t rt_sz   = max_vma - min_vma;
    const uint8_t* bin = all.data() + 16;

    vx_buffer_h kbuf = nullptr;
    auto r = vx_buffer_reserve(dev, min_vma, rt_sz, 0, &kbuf);
    if (r != VX_SUCCESS) return r;
    r = vx_buffer_access(kbuf, 0, bin_sz, VX_MEM_READ);
    if (r != VX_SUCCESS) { vx_buffer_release(kbuf); return r; }
    if (rt_sz > bin_sz) {
        r = vx_buffer_access(kbuf, bin_sz, rt_sz - bin_sz, VX_MEM_READ_WRITE);
        if (r != VX_SUCCESS) { vx_buffer_release(kbuf); return r; }
    }

    vx_event_h ev_bin = nullptr;
    r = vx_enqueue_write(q, kbuf, 0, bin, bin_sz, 0, nullptr, &ev_bin);
    if (r != VX_SUCCESS) { vx_buffer_release(kbuf); return r; }

    vx_event_h ev_bss = nullptr;
    std::vector<uint8_t> zeros;
    if (rt_sz > bin_sz) {
        zeros.assign(rt_sz - bin_sz, 0);
        r = vx_enqueue_write(q, kbuf, bin_sz, zeros.data(), rt_sz - bin_sz,
                             0, nullptr, &ev_bss);
        if (r != VX_SUCCESS) { vx_buffer_release(kbuf); return r; }
    }

    vx_event_h waits[2];
    int nw = 0;
    if (ev_bin) waits[nw++] = ev_bin;
    if (ev_bss) waits[nw++] = ev_bss;
    if (nw) {
        r = vx_event_wait_all((uint32_t)nw, waits, VX_TIMEOUT_INFINITE);
        for (int i = 0; i < nw; ++i) vx_event_release(waits[i]);
        if (r != VX_SUCCESS) { vx_buffer_release(kbuf); return r; }
    }

    *out_buf = kbuf;
    return VX_SUCCESS;
}

void prepare_launch_params(uint32_t threads_per_warp, uint32_t num_warps,
                           uint32_t ndim, const uint32_t* block_dim,
                           uint32_t eff_block[3],
                           uint32_t* block_size,
                           uint32_t* ws_x, uint32_t* ws_y, uint32_t* ws_z) {
    uint32_t auto_b[3] = { threads_per_warp, num_warps, 1 };
    const uint32_t* src = block_dim ? block_dim : auto_b;
    for (int i = 0; i < 3; ++i)
        eff_block[i] = (i < (int)ndim) ? src[i] : 1;
    uint32_t bs = 1;
    for (uint32_t i = 0; i < ndim; ++i) bs *= eff_block[i];
    *block_size = bs;
    *ws_x = threads_per_warp % eff_block[0];
    *ws_y = (threads_per_warp / eff_block[0]) % eff_block[1];
    *ws_z = (threads_per_warp / (eff_block[0] * eff_block[1])) % eff_block[2];
}

vx_result_t launch_kernel_v2(vx_device_h dev, vx_queue_h q,
                             vx_buffer_h kernel, vx_buffer_h args,
                             uint32_t ndim,
                             const uint32_t* grid_dim,
                             const uint32_t* block_dim,
                             uint32_t lmem_size,
                             vx_event_h* out_event) {
    uint64_t num_threads = 0, num_warps = 0;
    auto r = vx_device_query(dev, VX_CAPS_NUM_THREADS, &num_threads);
    if (r != VX_SUCCESS) return r;
    r = vx_device_query(dev, VX_CAPS_NUM_WARPS, &num_warps);
    if (r != VX_SUCCESS) return r;

    uint32_t eff_block[3], block_size, ws_x, ws_y, ws_z;
    prepare_launch_params((uint32_t)num_threads, (uint32_t)num_warps,
                          ndim, block_dim, eff_block,
                          &block_size, &ws_x, &ws_y, &ws_z);

    uint64_t pc, argp;
    r = vx_buffer_address(kernel, &pc);   if (r != VX_SUCCESS) return r;
    r = vx_buffer_address(args,   &argp); if (r != VX_SUCCESS) return r;

    uint32_t full_grid[3] = {1, 1, 1};
    for (uint32_t i = 0; i < ndim; ++i) full_grid[i] = grid_dim[i];

    struct { uint32_t addr; uint32_t value; } dcrs[] = {
        { VX_DCR_KMU_STARTUP_ADDR0, (uint32_t)(pc   & 0xffffffffu) },
        { VX_DCR_KMU_STARTUP_ADDR1, (uint32_t)(pc   >> 32) },
        { VX_DCR_KMU_STARTUP_ARG0,  (uint32_t)(argp & 0xffffffffu) },
        { VX_DCR_KMU_STARTUP_ARG1,  (uint32_t)(argp >> 32) },
        { VX_DCR_KMU_BLOCK_DIM_X,   eff_block[0] },
        { VX_DCR_KMU_BLOCK_DIM_Y,   eff_block[1] },
        { VX_DCR_KMU_BLOCK_DIM_Z,   eff_block[2] },
        { VX_DCR_KMU_GRID_DIM_X,    full_grid[0] },
        { VX_DCR_KMU_GRID_DIM_Y,    full_grid[1] },
        { VX_DCR_KMU_GRID_DIM_Z,    full_grid[2] },
        { VX_DCR_KMU_LMEM_SIZE,     lmem_size    },
        { VX_DCR_KMU_BLOCK_SIZE,    block_size   },
        { VX_DCR_KMU_WARP_STEP_X,   ws_x         },
        { VX_DCR_KMU_WARP_STEP_Y,   ws_y         },
        { VX_DCR_KMU_WARP_STEP_Z,   ws_z         },
    };
    for (auto& d : dcrs) {
        r = vx_enqueue_dcr_write(q, d.addr, d.value, 0, nullptr, nullptr);
        if (r != VX_SUCCESS) return r;
    }

    vx_launch_info_t li = {};
    li.struct_size = sizeof(li);
    li.kernel      = kernel;
    li.args        = args;
    li.ndim        = 0;
    return vx_enqueue_launch(q, &li, 0, nullptr, out_event);
}

} // namespace

int main(int argc, char* argv[]) {
    parse_args(argc, argv);
    std::srand(50);

    uint32_t size_sq  = size * size;
    uint32_t buf_size = size_sq * sizeof(TYPE);

    std::cout << "open device (vortex2.h)" << std::endl;
    std::cout << "matrix size: " << size << "x" << size << std::endl;

    vx_device_h dev = nullptr;
    CHECK_VX(vx_device_open(0, &dev));

    vx_queue_info_t qi = {};
    qi.struct_size = sizeof(qi);
    qi.priority    = VX_QUEUE_PRIORITY_NORMAL;
    vx_queue_h q = nullptr;
    CHECK_VX(vx_queue_create(dev, &qi, &q));

    // ----- Compute launch params + sanity-check the matrix size -----
    uint64_t num_threads = 0, num_warps = 0;
    CHECK_VX(vx_device_query(dev, VX_CAPS_NUM_THREADS, &num_threads));
    CHECK_VX(vx_device_query(dev, VX_CAPS_NUM_WARPS,   &num_warps));
    uint32_t block_dim[2] = { (uint32_t)num_threads, (uint32_t)num_warps };
    if ((size % block_dim[0]) != 0 || (size % block_dim[1]) != 0) {
        std::cerr << "Error: matrix size " << size
                  << " must be a multiple of block_dim ("
                  << block_dim[0] << "x" << block_dim[1] << ")." << std::endl;
        vx_queue_release(q);
        vx_device_release(dev);
        return -1;
    }
    uint32_t grid_dim[2] = { size / block_dim[0], size / block_dim[1] };

    // ----- Allocate device buffers -----
    vx_buffer_h A_buf    = nullptr;
    vx_buffer_h B_buf    = nullptr;
    vx_buffer_h C_buf    = nullptr;
    vx_buffer_h args_buf = nullptr;
    CHECK_VX(vx_buffer_create(dev, buf_size,             VX_MEM_READ,  &A_buf));
    CHECK_VX(vx_buffer_create(dev, buf_size,             VX_MEM_READ,  &B_buf));
    CHECK_VX(vx_buffer_create(dev, buf_size,             VX_MEM_WRITE, &C_buf));
    CHECK_VX(vx_buffer_create(dev, sizeof(kernel_arg_t), VX_MEM_READ,  &args_buf));

    kernel_arg_t kernel_arg = {};
    kernel_arg.size = size;
    CHECK_VX(vx_buffer_address(A_buf, &kernel_arg.A_addr));
    CHECK_VX(vx_buffer_address(B_buf, &kernel_arg.B_addr));
    CHECK_VX(vx_buffer_address(C_buf, &kernel_arg.C_addr));

    // ----- Build host data -----
    std::vector<TYPE> h_A(size_sq);
    std::vector<TYPE> h_B(size_sq);
    std::vector<TYPE> h_C(size_sq, TYPE{});
    for (uint32_t i = 0; i < size_sq; ++i) {
        h_A[i] = static_cast<TYPE>(std::rand()) / RAND_MAX;
        h_B[i] = static_cast<TYPE>(std::rand()) / RAND_MAX;
    }

    // ----- Load kernel binary (one internal sync at end of helper) -----
    vx_buffer_h kbuf = nullptr;
    CHECK_VX(load_kernel_v2(dev, q, kernel_file, &kbuf));

    auto t_start = std::chrono::high_resolution_clock::now();

    // ----- Async upload chain: A, B, args. -----
    CHECK_VX(vx_enqueue_write(q, A_buf,    0, h_A.data(), buf_size, 0, nullptr, nullptr));
    CHECK_VX(vx_enqueue_write(q, B_buf,    0, h_B.data(), buf_size, 0, nullptr, nullptr));
    CHECK_VX(vx_enqueue_write(q, args_buf, 0, &kernel_arg, sizeof(kernel_arg),
                              0, nullptr, nullptr));

    // ----- Launch (15 DCR writes + 1 launch enqueue, no waits) -----
    vx_event_h launch_ev = nullptr;
    CHECK_VX(launch_kernel_v2(dev, q, kbuf, args_buf,
                              /*ndim=*/2, grid_dim, block_dim, 0, &launch_ev));

    // ----- Read C back gated on the launch event -----
    vx_event_h read_ev = nullptr;
    CHECK_VX(vx_enqueue_read(q, h_C.data(), C_buf, 0, buf_size,
                             1, &launch_ev, &read_ev));

    // ----- The ONE wait -----
    CHECK_VX(vx_event_wait_all(1, &read_ev, VX_TIMEOUT_INFINITE));
    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                         t_end - t_start).count();
    std::printf("Elapsed time: %lg ms\n", elapsed);

    vx_event_release(read_ev);
    vx_event_release(launch_ev);

    // ----- Verify -----
    int errors = 0;
    {
        std::vector<TYPE> h_ref(size_sq);
        matmul_cpu(h_ref.data(), h_A.data(), h_B.data(), size, size);
        for (uint32_t i = 0; i < size_sq; ++i) {
            if (!float_eq(h_C[i], h_ref[i])) {
                if (errors < 16) {
                    std::printf("*** error: [%u] expected=%f actual=%f\n",
                                i, (double)h_ref[i], (double)h_C[i]);
                }
                ++errors;
            }
        }
    }

    // ----- Cleanup -----
    vx_buffer_release(args_buf);
    vx_buffer_release(C_buf);
    vx_buffer_release(B_buf);
    vx_buffer_release(A_buf);
    vx_buffer_release(kbuf);
    vx_queue_release(q);
    vx_device_release(dev);

    if (errors) {
        std::cout << "Found " << errors << " errors!" << std::endl;
        std::cout << "FAILED!" << std::endl;
        return errors;
    }
    std::cout << "PASSED!" << std::endl;
    return 0;
}
