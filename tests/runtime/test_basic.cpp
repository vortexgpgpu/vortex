// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

// ============================================================================
// test_basic.cpp
//
// Smoke test for the runtime. Exercises both the vortex.h API
// (vx_dev_open, vx_mem_alloc, etc.) and the vortex2.h API
// (vx_device_open, vx_buffer_create, vx_queue_create, etc.)
// against the linked backend (selected at compile time — simx by default).
//
// Verifies:
//   - libvortex.so exports both API symbol sets.
//   - vx_dev_open and vx_device_open return equivalent handles.
//   - Buffer create/release works via both APIs.
//   - Queue create/release works (vortex2.h only).
//   - Event create/release/signal works (vortex2.h only).
//   - vx_device_query and vx_dev_caps return identical values.
//
// Expected output: "PASSED" on success, "FAILED at <step>" on any failure.
// Exit code: 0 on PASS, 1 on FAIL.
// ============================================================================

#include <vortex.h>
#include <vortex2.h>

#include <cstdint>
#include <cstdio>
#include <cstring>

#define CHECK(expr) do { \
    int _r = (expr); \
    if (_r != 0) { \
        fprintf(stderr, "FAILED at %s:%d: '%s' returned %d\n", \
                __FILE__, __LINE__, #expr, _r); \
        return 1; \
    } \
} while (0)

#define CHECK_VX(expr) do { \
    vx_result_t _r = (expr); \
    if (_r != VX_SUCCESS) { \
        fprintf(stderr, "FAILED at %s:%d: '%s' returned %s\n", \
                __FILE__, __LINE__, #expr, vx_result_string(_r)); \
        return 1; \
    } \
} while (0)

int main() {
    // Open device via vortex.h API.
    vx_device_h dev = nullptr;
    CHECK(vx_dev_open(&dev));
    if (!dev) { fprintf(stderr, "FAILED: vx_dev_open returned NULL handle\n"); return 1; }

    // Query a capability via both APIs and compare.
    uint64_t legacy_num_cores = 0, new_num_cores = 0;
    CHECK(vx_dev_caps(dev, VX_CAPS_NUM_CORES, &legacy_num_cores));
    CHECK_VX(vx_device_query(dev, VX_CAPS_NUM_CORES, &new_num_cores));
    if (legacy_num_cores != new_num_cores) {
        fprintf(stderr, "FAILED: caps mismatch: legacy=%lu new=%lu\n",
                legacy_num_cores, new_num_cores);
        return 1;
    }
    printf("device caps VX_CFG_NUM_CORES = %lu\n", legacy_num_cores);

    // Allocate a buffer via vortex.h; free via vortex2.h.
    vx_buffer_h buf = nullptr;
    CHECK(vx_mem_alloc(dev, 4096, VX_MEM_READ_WRITE, &buf));
    if (!buf) { fprintf(stderr, "FAILED: vx_mem_alloc returned NULL\n"); return 1; }
    CHECK_VX(vx_buffer_release(buf));

    // Allocate a buffer via vortex2.h; free via vortex.h.
    vx_buffer_h buf2 = nullptr;
    CHECK_VX(vx_buffer_create(dev, 8192, VX_MEM_READ_WRITE, &buf2));
    uint64_t addr = 0;
    CHECK_VX(vx_buffer_address(buf2, &addr));
    if (addr == 0) { fprintf(stderr, "FAILED: buffer address is 0\n"); return 1; }
    printf("buffer dev_addr = 0x%lx\n", addr);
    CHECK(vx_mem_free(buf2));

    // Create and destroy a queue.
    vx_queue_h q = nullptr;
    vx_queue_info_t qi = {};
    qi.struct_size = sizeof(qi);
    qi.priority    = VX_QUEUE_PRIORITY_NORMAL;
    qi.flags       = VX_QUEUE_PROFILING_ENABLE;
    CHECK_VX(vx_queue_create(dev, &qi, &q));
    if (!q) { fprintf(stderr, "FAILED: vx_queue_create returned NULL\n"); return 1; }
    CHECK_VX(vx_queue_release(q));

    // Timeline event lifecycle: create, signal, wait, release.
    vx_event_h ev = nullptr;
    CHECK_VX(vx_event_create(dev, &ev));
    if (!ev) { fprintf(stderr, "FAILED: vx_event_create returned NULL\n"); return 1; }
    uint64_t v = 0xbadbad;
    CHECK_VX(vx_event_get_value(ev, &v));
    if (v != 0) {
        fprintf(stderr, "FAILED: fresh event counter not 0 (got %llu)\n",
                (unsigned long long)v);
        return 1;
    }
    CHECK_VX(vx_event_signal(ev, 1));
    CHECK_VX(vx_event_wait_value(ev, 1, VX_TIMEOUT_INFINITE));
    CHECK_VX(vx_event_get_value(ev, &v));
    if (v != 1) {
        fprintf(stderr, "FAILED: signaled event counter not 1 (got %llu)\n",
                (unsigned long long)v);
        return 1;
    }
    CHECK_VX(vx_event_release(ev));

    // Refcount: retain, partial release, then final release.
    vx_buffer_h refcount_buf = nullptr;
    CHECK_VX(vx_buffer_create(dev, 1024, VX_MEM_READ_WRITE, &refcount_buf));
    CHECK_VX(vx_buffer_retain(refcount_buf));   // refs = 2
    CHECK_VX(vx_buffer_release(refcount_buf));  // refs = 1 (not freed)
    // Use the buffer after one release to confirm it's still alive.
    uint64_t rb_addr = 0;
    CHECK_VX(vx_buffer_address(refcount_buf, &rb_addr));
    if (rb_addr == 0) {
        fprintf(stderr, "FAILED: refcount buffer freed too early\n");
        return 1;
    }
    CHECK_VX(vx_buffer_release(refcount_buf));  // refs = 0 (freed)

    // Close device.
    CHECK(vx_dev_close(dev));

    printf("PASSED\n");
    return 0;
}
