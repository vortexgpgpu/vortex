// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

// ============================================================================
// test_import_host.cpp
//
// Exercises buffers created with VX_MEM_IMPORT_HOST, whose storage is host
// memory the device reads and writes in place:
//   - a host write is what the next kernel reads, with no transfer call
//   - a kernel write is what the host reads after the queue finishes
//   - a host write between two launches reaches the second one, for a range
//     the first launch only read and for one it wrote (no stale cache lines)
//   - CP DMA (enqueue write/read) on an imported buffer lands in host memory
//   - releasing an import and importing the same pointer again
//   - a device buffer allocated after an import was released no longer
//     follows the host pages it used to alias
//   - a misaligned host pointer is refused
//
// Uses the `basic` regression kernel (dst[i] = src[i]); its image path comes
// from VX_TEST_VXBIN, as for test_module_kernel. Backends that cannot import
// host memory report VX_ERR_NOT_SUPPORTED, and the test reports a skip.
//
// PASS: exit code 0 and "PASSED!" on stdout.
// ============================================================================

#include <vortex2.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#define CHECK_VX(expr) do { \
    vx_result_t _r = (expr); \
    if (_r != VX_SUCCESS) { \
        fprintf(stderr, "FAILED at %s:%d: '%s' returned %s\n", \
                __FILE__, __LINE__, #expr, vx_result_string(_r)); \
        return 1; \
    } \
} while (0)

#define EXPECT(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAILED at %s:%d: %s\n", __FILE__, __LINE__, msg); \
        return 1; \
    } \
} while (0)

namespace {

constexpr uint32_t kPage  = 4096;
// Not a whole number of pages, and more than one page: the import rounds up
// and the kernel crosses a page boundary.
constexpr uint32_t kCount = 1500;
constexpr uint32_t kBytes = kCount * sizeof(int32_t);
constexpr uint32_t kSpan  = (kBytes + kPage - 1) / kPage * kPage;

struct kernel_arg_t {
    uint32_t count;
    uint64_t src_addr;
    uint64_t dst_addr;
};

struct Ctx {
    vx_device_h dev    = nullptr;
    vx_queue_h  q      = nullptr;
    vx_module_h module = nullptr;
    vx_kernel_h kernel = nullptr;
};

void fill(int32_t* p, uint32_t seed) {
    for (uint32_t i = 0; i < kCount; ++i) {
        p[i] = int32_t(seed * 2654435761u + i * 40503u);
    }
}

int check(const int32_t* p, uint32_t seed, const char* what) {
    for (uint32_t i = 0; i < kCount; ++i) {
        const int32_t want = int32_t(seed * 2654435761u + i * 40503u);
        if (p[i] != want) {
            fprintf(stderr, "FAILED: %s: word %u is 0x%08x, expected 0x%08x\n",
                    what, i, uint32_t(p[i]), uint32_t(want));
            return 1;
        }
    }
    return 0;
}

int import_buffer(vx_device_h dev, void* host, vx_buffer_h* out) {
    *out = host;
    CHECK_VX(vx_buffer_create(dev, kBytes, VX_MEM_READ_WRITE | VX_MEM_IMPORT_HOST, out));
    return 0;
}

int copy(Ctx& c, vx_buffer_h src, vx_buffer_h dst) {
    kernel_arg_t arg = {};
    arg.count = kCount;
    CHECK_VX(vx_buffer_address(src, &arg.src_addr));
    CHECK_VX(vx_buffer_address(dst, &arg.dst_addr));
    vx_launch_info_t li = {};
    li.struct_size  = sizeof(li);
    li.kernel       = c.kernel;
    li.args_host    = &arg;
    li.args_size    = sizeof(arg);
    li.ndim         = 1;
    li.grid_dim[0]  = 1;
    li.block_dim[0] = 1;
    CHECK_VX(vx_enqueue_launch(c.q, &li, 0, nullptr, nullptr));
    CHECK_VX(vx_queue_finish(c.q, VX_TIMEOUT_INFINITE));
    return 0;
}

int run(Ctx& c) {
    auto* ha = static_cast<int32_t*>(std::aligned_alloc(kPage, kSpan));
    auto* hb = static_cast<int32_t*>(std::aligned_alloc(kPage, kSpan));
    EXPECT(ha && hb, "host allocation");

    vx_buffer_h a = nullptr, b = nullptr;
    {
        vx_buffer_h probe = ha;
        vx_result_t r = vx_buffer_create(c.dev, kBytes,
                                         VX_MEM_READ_WRITE | VX_MEM_IMPORT_HOST, &probe);
        if (r == VX_ERR_NOT_SUPPORTED) {
            printf("       (skipped — this backend does not import host memory)\n");
            std::free(ha);
            std::free(hb);
            return 0;
        }
        CHECK_VX(r);
        a = probe;
    }
    if (import_buffer(c.dev, hb, &b)) return 1;

    // Host write -> kernel read; kernel write -> host read.
    fill(ha, 1);
    fill(hb, 99);
    if (copy(c, a, b)) return 1;
    if (check(hb, 1, "kernel write seen by the host")) return 1;

    // A range the previous launch read, rewritten by the host.
    fill(ha, 2);
    if (copy(c, a, b)) return 1;
    if (check(hb, 2, "host write to a range the device read")) return 1;

    // A range the previous launch wrote, rewritten by the host.
    fill(hb, 3);
    if (copy(c, b, a)) return 1;
    if (check(ha, 3, "host write to a range the device wrote")) return 1;

    // CP DMA lands in the host pages.
    {
        auto* tmp = static_cast<int32_t*>(std::malloc(kBytes));
        EXPECT(tmp, "staging allocation");
        fill(tmp, 4);
        CHECK_VX(vx_enqueue_write(c.q, b, 0, tmp, kBytes, 0, nullptr, nullptr));
        CHECK_VX(vx_queue_finish(c.q, VX_TIMEOUT_INFINITE));
        int rc = check(hb, 4, "DMA write seen by the host");
        fill(ha, 5);
        CHECK_VX(vx_enqueue_read(c.q, tmp, a, 0, kBytes, 0, nullptr, nullptr));
        CHECK_VX(vx_queue_finish(c.q, VX_TIMEOUT_INFINITE));
        rc = rc || check(tmp, 5, "DMA read of host-written bytes");
        std::free(tmp);
        if (rc) return 1;
    }

    // Release and import the same pointer again.
    CHECK_VX(vx_buffer_release(a));
    if (import_buffer(c.dev, ha, &a)) return 1;
    fill(ha, 6);
    if (copy(c, a, b)) return 1;
    if (check(hb, 6, "copy from a re-imported pointer")) return 1;

    // Once released, an import's device pages stop following the host: device
    // buffers placed at the same addresses afterwards keep their own contents.
    uint64_t a_addr = 0, b_addr = 0;
    CHECK_VX(vx_buffer_address(a, &a_addr));
    CHECK_VX(vx_buffer_address(b, &b_addr));
    CHECK_VX(vx_buffer_release(a));
    CHECK_VX(vx_buffer_release(b));
    vx_buffer_h d = nullptr, e = nullptr;
    CHECK_VX(vx_buffer_reserve(c.dev, a_addr, kBytes, VX_MEM_READ_WRITE, &d));
    CHECK_VX(vx_buffer_reserve(c.dev, b_addr, kBytes, VX_MEM_READ_WRITE, &e));
    {
        auto* tmp = static_cast<int32_t*>(std::malloc(kBytes));
        EXPECT(tmp, "staging allocation");
        fill(tmp, 7);
        CHECK_VX(vx_enqueue_write(c.q, d, 0, tmp, kBytes, 0, nullptr, nullptr));
        CHECK_VX(vx_queue_finish(c.q, VX_TIMEOUT_INFINITE));
        fill(ha, 8);
        fill(hb, 8);
        int rc = copy(c, d, e);
        if (!rc) {
            CHECK_VX(vx_enqueue_read(c.q, tmp, e, 0, kBytes, 0, nullptr, nullptr));
            CHECK_VX(vx_queue_finish(c.q, VX_TIMEOUT_INFINITE));
            rc = check(tmp, 7, "device buffer after the import was released");
        }
        std::free(tmp);
        if (rc) return 1;
    }
    CHECK_VX(vx_buffer_release(d));
    CHECK_VX(vx_buffer_release(e));

    // A host pointer off a page boundary is refused.
    {
        vx_buffer_h bad = reinterpret_cast<uint8_t*>(ha) + 64;
        vx_result_t r = vx_buffer_create(c.dev, kBytes,
                                         VX_MEM_READ_WRITE | VX_MEM_IMPORT_HOST, &bad);
        EXPECT(r == VX_ERR_INVALID_VALUE, "a misaligned import must be refused");
    }

    std::free(ha);
    std::free(hb);
    return 0;
}

} // namespace

int main() {
    const char* vxbin = std::getenv("VX_TEST_VXBIN");
    if (!vxbin || !*vxbin) {
        fprintf(stderr, "FAILED: VX_TEST_VXBIN is not set\n");
        return 1;
    }
    Ctx c;
    CHECK_VX(vx_device_open(0, &c.dev));
    vx_queue_info_t qi = {};
    qi.struct_size = sizeof(qi);
    qi.priority    = VX_QUEUE_PRIORITY_NORMAL;
    CHECK_VX(vx_queue_create(c.dev, &qi, &c.q));
    CHECK_VX(vx_module_load_file(c.dev, vxbin, &c.module));
    CHECK_VX(vx_module_get_kernel(c.module, "main", &c.kernel));

    const int rc = run(c);

    vx_kernel_release(c.kernel);
    vx_module_release(c.module);
    vx_queue_release(c.q);
    vx_device_release(c.dev);
    if (rc == 0) {
        printf("PASSED!\n");
    }
    return rc;
}
