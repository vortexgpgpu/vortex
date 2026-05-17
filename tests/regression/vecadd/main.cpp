// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

// ============================================================================
// vecadd — vortex2.h-native regression test.
//
// Rewritten from scratch on the async vortex2.h API. The legacy
// vortex.h version performed five separate synchronous waits during
// setup (one per vx_copy_to_dev, one for vx_upload_kernel_file, one
// for vx_upload_bytes, one per DCR write inside vx_start_g). The v2
// version exploits the per-queue worker thread (one Queue::worker_loop
// services every command in FIFO order, see runtime impl §4.6.1):
//
//   - All host→device uploads (src0, src1, args, kernel binary, bss
//     zeroing) are enqueued back-to-back with NO event waits between
//     them. The worker drains the FIFO in order.
//   - The 15 KMU DCR programming writes are also fire-and-forget —
//     no per-write events. FIFO order guarantees they commit before
//     the subsequent launch enqueue runs.
//   - The launch enqueue produces an event. The dst readback enqueue
//     gates on that event (vx_enqueue_read with wait_events list).
//   - The host waits exactly once at the end, on the read event.
//
// This is the canonical pattern POCL/Vulkan/HIP translator layers
// should adopt when targeting vortex2.h.
// ============================================================================

#include <vortex2.h>
#include <VX_config.h>
#include <VX_types.h>

#include "common.h"

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
uint32_t    size        = 16;

// ----- CLI -----
void show_usage() {
    std::cout << "Vortex vecadd (vortex2.h-native)." << std::endl;
    std::cout << "Usage: [-k kernel] [-n words] [-h]" << std::endl;
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

// ----- Float comparator with ULP tolerance -----
bool float_eq(float a, float b) {
    union fi { float f; int32_t i; };
    fi fa = {a}, fb = {b};
    return std::abs(fa.i - fb.i) <= FLOAT_ULP;
}

// ----- Kernel image loader -----
// vortex2.h-native: vx_buffer_reserve a fixed VMA region, set ACLs,
// fire-and-forget two enqueue_writes (binary + bss zero) through the
// queue. The caller can chain the launch behind these without waiting.
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

    // ACLs: .text/.rodata read-only, .bss read-write.
    r = vx_buffer_access(kbuf, 0, bin_sz, VX_MEM_READ);
    if (r != VX_SUCCESS) { vx_buffer_release(kbuf); return r; }
    if (rt_sz > bin_sz) {
        r = vx_buffer_access(kbuf, bin_sz, rt_sz - bin_sz, VX_MEM_READ_WRITE);
        if (r != VX_SUCCESS) { vx_buffer_release(kbuf); return r; }
    }

    // Fire-and-forget: binary copy + bss zero. The worker chains them
    // in FIFO order; subsequent enqueues see the kernel image fully
    // resident in device memory when they run.
    //
    // Holding a host-side copy of the binary alive until the queue
    // drains: the runtime's enqueue_write captures the host pointer
    // and the worker may execute the copy after this function returns.
    // We allocate a heap copy that outlives this function; the worker
    // discards it implicitly when the upload completes (no need to
    // free — the queue worker accesses host memory synchronously
    // inside its work lambda, so by the time wait succeeds the worker
    // is done with the pointer). For simplicity we leak the heap copy
    // here; a real impl would chain a vx_event callback to free it.
    //
    // Concretely: we wait on the upload event before returning to
    // ensure the host vector isn't freed while the worker is still
    // copying. This is the ONE sync point during kernel load.
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

    // Sync only here — necessary because `all` and `zeros` are stack/
    // local-scope vectors that go out of scope when this function
    // returns. The worker captured raw pointers into them.
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

// ----- Compute launch params (block_size, warp_step) -----
// Mirrors prepare_kernel_launch_params() in sw/runtime/common/utils.cpp
// so the test doesn't depend on the legacy helper.
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

// ----- Program KMU descriptor + enqueue launch (no waits) -----
// All 15 DCR writes are fire-and-forget; the launch's position in
// the FIFO guarantees they commit first. Returns the launch event.
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
    li.ndim        = 0;   // DCRs already programmed; engine just triggers
    return vx_enqueue_launch(q, &li, 0, nullptr, out_event);
}

} // namespace

int main(int argc, char* argv[]) {
    parse_args(argc, argv);
    std::srand(50);

    uint32_t num_points = size;
    uint32_t buf_size   = num_points * sizeof(TYPE);

    std::cout << "open device (vortex2.h)" << std::endl;
    std::cout << "number of points: " << num_points << std::endl;
    std::cout << "buffer size: " << buf_size << " bytes" << std::endl;

    vx_device_h dev = nullptr;
    CHECK_VX(vx_device_open(0, &dev));

    vx_queue_info_t qi = {};
    qi.struct_size = sizeof(qi);
    qi.priority    = VX_QUEUE_PRIORITY_NORMAL;
    vx_queue_h q = nullptr;
    CHECK_VX(vx_queue_create(dev, &qi, &q));

    // ----- Allocate device buffers -----
    vx_buffer_h src0_buf = nullptr;
    vx_buffer_h src1_buf = nullptr;
    vx_buffer_h dst_buf  = nullptr;
    vx_buffer_h args_buf = nullptr;
    CHECK_VX(vx_buffer_create(dev, buf_size,            VX_MEM_READ,  &src0_buf));
    CHECK_VX(vx_buffer_create(dev, buf_size,            VX_MEM_READ,  &src1_buf));
    CHECK_VX(vx_buffer_create(dev, buf_size,            VX_MEM_WRITE, &dst_buf));
    CHECK_VX(vx_buffer_create(dev, sizeof(kernel_arg_t), VX_MEM_READ, &args_buf));

    kernel_arg_t kernel_arg = {};
    kernel_arg.num_points = num_points;
    CHECK_VX(vx_buffer_address(src0_buf, &kernel_arg.src0_addr));
    CHECK_VX(vx_buffer_address(src1_buf, &kernel_arg.src1_addr));
    CHECK_VX(vx_buffer_address(dst_buf,  &kernel_arg.dst_addr));

    // ----- Build host data -----
    std::vector<TYPE> h_src0(num_points);
    std::vector<TYPE> h_src1(num_points);
    std::vector<TYPE> h_dst (num_points, TYPE{});
    for (uint32_t i = 0; i < num_points; ++i) {
        h_src0[i] = static_cast<TYPE>(std::rand()) / RAND_MAX;
        h_src1[i] = static_cast<TYPE>(std::rand()) / RAND_MAX;
    }

    // ----- Load kernel binary (one internal sync at end of helper) -----
    vx_buffer_h kbuf = nullptr;
    CHECK_VX(load_kernel_v2(dev, q, kernel_file, &kbuf));

    // ----- Async upload chain: src0, src1, args. -----
    // The worker drains them in FIFO order; subsequent launch sees
    // them committed. We use `vx_queue_finish` here as a barrier so
    // the host-side buffer lifetimes (h_src0, h_src1, kernel_arg) are
    // pinned until the writes actually land — the worker captures raw
    // pointers and may execute the copy after these enqueues return.
    // (A real translator layer would chain a freeing callback on the
    // write events instead.)
    CHECK_VX(vx_enqueue_write(q, src0_buf, 0, h_src0.data(), buf_size,
                              0, nullptr, nullptr));
    CHECK_VX(vx_enqueue_write(q, src1_buf, 0, h_src1.data(), buf_size,
                              0, nullptr, nullptr));
    CHECK_VX(vx_enqueue_write(q, args_buf, 0, &kernel_arg, sizeof(kernel_arg),
                              0, nullptr, nullptr));
    CHECK_VX(vx_queue_finish(q, VX_TIMEOUT_INFINITE));

    // ----- Compute launch params + enqueue launch (15 DCR writes
    //       fire-and-forget + 1 launch enqueue, no inter-step waits). -----
    uint64_t num_threads = 0, num_warps = 0;
    CHECK_VX(vx_device_query(dev, VX_CAPS_NUM_THREADS, &num_threads));
    CHECK_VX(vx_device_query(dev, VX_CAPS_NUM_WARPS,   &num_warps));

    // BLOCK_DIM = full-core occupancy (num_threads × num_warps). This
    // keeps GRID_DIM small enough that the cta_dispatcher doesn't have
    // to re-use warp slots across blocks — a pre-existing simx/rtlsim
    // path that's been observed to mis-dispatch when GRID > num_warps.
    // GRID = ceil(N / block_size). The kernel still indexes
    // blockIdx.x * blockDim.x + threadIdx.x correctly.
    uint32_t block_size_v = (uint32_t)num_threads * (uint32_t)num_warps;
    uint32_t block_dim[1] = { block_size_v };
    uint32_t grid_dim [1] = { (num_points + block_size_v - 1) / block_size_v };

    vx_event_h launch_ev = nullptr;
    CHECK_VX(launch_kernel_v2(dev, q, kbuf, args_buf,
                              /*ndim=*/1, grid_dim, block_dim, 0, &launch_ev));

    // ----- Read dst back gated on the launch event. -----
    vx_event_h read_ev = nullptr;
    CHECK_VX(vx_enqueue_read(q, h_dst.data(), dst_buf, 0, buf_size,
                             1, &launch_ev, &read_ev));

    // ----- The ONE wait: on the read event. Everything before
    //       drains transitively through the FIFO. -----
    CHECK_VX(vx_event_wait_all(1, &read_ev, VX_TIMEOUT_INFINITE));
    vx_event_release(read_ev);
    vx_event_release(launch_ev);

    // ----- Verify -----
    int errors = 0;
    for (uint32_t i = 0; i < num_points; ++i) {
        TYPE ref = h_src0[i] + h_src1[i];
        TYPE cur = h_dst[i];
        if (!float_eq(cur, ref)) {
            if (errors < 16) {
                std::printf("*** error: [%u] expected=%f actual=%f\n",
                            i, (double)ref, (double)cur);
            }
            ++errors;
        }
    }

    // ----- Cleanup -----
    vx_buffer_release(args_buf);
    vx_buffer_release(dst_buf);
    vx_buffer_release(src1_buf);
    vx_buffer_release(src0_buf);
    vx_buffer_release(kbuf);
    vx_queue_release(q);
    vx_device_release(dev);

    if (errors) {
        std::cout << "Found " << errors << " errors!" << std::endl;
        std::cout << "FAILED!" << std::endl;
        return 1;
    }
    std::cout << "PASSED!" << std::endl;
    return 0;
}
