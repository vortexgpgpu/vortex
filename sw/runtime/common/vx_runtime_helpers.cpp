// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

// ============================================================================
// vx_runtime_helpers.cpp — vortex2.h utility entry points.
//
// These wrap common multi-call patterns (kernel-image upload, occupancy
// computation) so user code calling vortex2.h doesn't reimplement them.
// All implementations call only public vortex2.h primitives.
// ============================================================================

#include <vortex2.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

extern "C" vx_result_t vx_device_max_occupancy_grid(vx_device_h dev,
                                                    uint32_t ndim,
                                                    const uint32_t* global_dim,
                                                    uint32_t* grid_out,
                                                    uint32_t* block_out) {
    if (!dev || ndim == 0 || ndim > 3 || !global_dim ||
        !grid_out || !block_out) return VX_ERR_INVALID_VALUE;

    uint64_t num_threads = 0, num_warps = 0;
    auto r = vx_device_query(dev, VX_CAPS_NUM_THREADS, &num_threads);
    if (r != VX_SUCCESS) return r;
    r = vx_device_query(dev, VX_CAPS_NUM_WARPS, &num_warps);
    if (r != VX_SUCCESS) return r;

    // Natural per-dim block size: (num_threads, num_warps, 1). Replicates
    // the legacy vx_max_occupancy_grid behavior so callers migrating from
    // vortex.h see identical grid/block selections.
    const uint64_t auto_block[3] = {num_threads, num_warps, 1};
    for (uint32_t i = 0; i < ndim; ++i) {
        block_out[i] = (uint32_t)auto_block[i];
        grid_out[i]  = (global_dim[i] + block_out[i] - 1) / block_out[i];
    }
    return VX_SUCCESS;
}

extern "C" vx_result_t vx_buffer_load_kernel_file(vx_device_h dev,
                                                  vx_queue_h  queue,
                                                  const char* path,
                                                  vx_buffer_h* out) {
    if (!dev || !queue || !path || !out) return VX_ERR_INVALID_VALUE;

    // vxbin header: [min_vma:8][max_vma:8][bytes...]
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return VX_ERR_INVALID_VALUE;
    ifs.seekg(0, ifs.end);
    auto file_sz = (size_t)ifs.tellg();
    ifs.seekg(0, ifs.beg);
    if (file_sz < 16) return VX_ERR_INVALID_VALUE;

    std::vector<uint8_t> all(file_sz);
    ifs.read(reinterpret_cast<char*>(all.data()), file_sz);
    if (!ifs) return VX_ERR_INVALID_VALUE;

    const uint64_t min_vma = *reinterpret_cast<const uint64_t*>(all.data());
    const uint64_t max_vma = *reinterpret_cast<const uint64_t*>(all.data() + 8);
    const uint64_t bin_sz  = file_sz - 16;
    const uint64_t rt_sz   = max_vma - min_vma;
    const uint8_t* bin     = all.data() + 16;

    if (bin_sz > rt_sz) return VX_ERR_INVALID_VALUE;

    vx_buffer_h kbuf = nullptr;
    auto r = vx_buffer_reserve(dev, min_vma, rt_sz, 0, &kbuf);
    if (r != VX_SUCCESS) return r;

    // .text/.rodata read-only, .bss read-write.
    r = vx_buffer_access(kbuf, 0, bin_sz, VX_MEM_READ);
    if (r != VX_SUCCESS) goto fail;
    if (rt_sz > bin_sz) {
        r = vx_buffer_access(kbuf, bin_sz, rt_sz - bin_sz, VX_MEM_READ_WRITE);
        if (r != VX_SUCCESS) goto fail;
    }

    // Fire-and-forget the two uploads through the queue; wait once at
    // the end so the host vectors don't drop before the worker reads
    // them.
    {
        vx_event_h ev_bin = nullptr;
        r = vx_enqueue_write(queue, kbuf, 0, bin, bin_sz, 0, nullptr, &ev_bin);
        if (r != VX_SUCCESS) goto fail;

        vx_event_h ev_bss = nullptr;
        std::vector<uint8_t> zeros;
        if (rt_sz > bin_sz) {
            zeros.assign(rt_sz - bin_sz, 0);
            r = vx_enqueue_write(queue, kbuf, bin_sz, zeros.data(),
                                 rt_sz - bin_sz, 0, nullptr, &ev_bss);
            if (r != VX_SUCCESS) goto fail;
        }

        vx_event_h waits[2];
        uint32_t nw = 0;
        if (ev_bin) waits[nw++] = ev_bin;
        if (ev_bss) waits[nw++] = ev_bss;
        if (nw) {
            r = vx_event_wait_all(nw, waits, VX_TIMEOUT_INFINITE);
            for (uint32_t i = 0; i < nw; ++i) vx_event_release(waits[i]);
            if (r != VX_SUCCESS) goto fail;
        }
    }

    *out = kbuf;
    return VX_SUCCESS;

fail:
    vx_buffer_release(kbuf);
    return r;
}
