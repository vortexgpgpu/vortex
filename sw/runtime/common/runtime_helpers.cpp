// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

// ============================================================================
// runtime_helpers.cpp — vortex2.h utility entry points.
//
// These wrap common multi-call patterns (occupancy computation) so user
// code calling vortex2.h doesn't reimplement them. All implementations
// call only public vortex2.h primitives.
// ============================================================================

#include <vortex2.h>

#include <cstdint>

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
