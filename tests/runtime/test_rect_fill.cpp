// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

// ============================================================================
// test_rect_fill.cpp
//
// Exercises the Phase 2 vortex2.h additions: vx_enqueue_fill_buffer, the
// rect DMA ops (vx_enqueue_{read,write,copy}_rect), and async map/unmap
// (vx_enqueue_map / vx_enqueue_unmap). All run against the linked backend
// (simx by default).
//
// Each rect case writes a tightly-packed host array into a device buffer
// whose row pitch is deliberately wider than the row, then reads it back,
// verifying both the payload and that the inter-row gap bytes were left
// untouched (the strided decomposition must not clobber them).
//
// Expected output: "PASSED" + exit 0 on success; "FAILED ..." + exit 1.
// ============================================================================

#include <vortex2.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#define CHECK_VX(expr) do { \
    vx_result_t _r = (expr); \
    if (_r != VX_SUCCESS) { \
        fprintf(stderr, "FAILED at %s:%d: '%s' returned %s\n", \
                __FILE__, __LINE__, #expr, vx_result_string(_r)); \
        return 1; \
    } \
} while (0)

#define FAIL(msg) do { \
    fprintf(stderr, "FAILED at %s:%d: %s\n", __FILE__, __LINE__, msg); \
    return 1; \
} while (0)

namespace {

// Block on an enqueue's completion event, then release it.
vx_result_t sync_ev(vx_event_h ev) {
    vx_result_t r = vx_event_wait_value(ev, 1, VX_TIMEOUT_INFINITE);
    vx_event_release(ev);
    return r;
}

} // namespace

int main() {
    vx_device_h dev = nullptr;
    CHECK_VX(vx_device_open(0, &dev));

    vx_queue_info_t qi = {};
    qi.struct_size = sizeof(qi);
    qi.priority    = VX_QUEUE_PRIORITY_NORMAL;
    vx_queue_h q = nullptr;
    CHECK_VX(vx_queue_create(dev, &qi, &q));

    // ------------------------------------------------------------------
    // 1) vx_enqueue_fill_buffer — full fill then a partial overwrite.
    // ------------------------------------------------------------------
    {
        const uint32_t kCount = 64;                 // 256 bytes
        const uint64_t kBytes = kCount * sizeof(uint32_t);
        vx_buffer_h buf = nullptr;
        CHECK_VX(vx_buffer_create(dev, kBytes, VX_MEM_READ_WRITE, &buf));

        uint32_t pat = 0xdeadbeefu;
        vx_event_h ev = nullptr;
        CHECK_VX(vx_enqueue_fill_buffer(q, buf, 0, kBytes, &pat, sizeof(pat),
                                        0, nullptr, &ev));
        CHECK_VX(sync_ev(ev));

        // Overwrite the middle quarter [16,32) with a different pattern.
        uint32_t pat2 = 0x12345678u;
        CHECK_VX(vx_enqueue_fill_buffer(q, buf, 16 * sizeof(uint32_t),
                                        16 * sizeof(uint32_t),
                                        &pat2, sizeof(pat2), 0, nullptr, &ev));
        CHECK_VX(sync_ev(ev));

        std::vector<uint32_t> back(kCount, 0);
        CHECK_VX(vx_enqueue_read(q, back.data(), buf, 0, kBytes,
                                 0, nullptr, &ev));
        CHECK_VX(sync_ev(ev));
        for (uint32_t i = 0; i < kCount; ++i) {
            uint32_t want = (i >= 16 && i < 32) ? pat2 : pat;
            if (back[i] != want) FAIL("fill_buffer payload mismatch");
        }
        CHECK_VX(vx_buffer_release(buf));
        printf("fill_buffer: OK\n");
    }

    // ------------------------------------------------------------------
    // 2) vx_enqueue_write_rect + vx_enqueue_read_rect — strided 2D rect.
    //
    // Device buffer holds H rows of DEV_PITCH bytes; the rect occupies
    // ROW_BYTES of each row. The gap (DEV_PITCH - ROW_BYTES) must survive
    // a write_rect untouched.
    // ------------------------------------------------------------------
    const uint32_t W         = 8;                       // elems per row
    const uint32_t H         = 6;                       // rows
    const uint64_t ROW_BYTES = W * sizeof(uint32_t);     // 32
    const uint64_t DEV_PITCH = 12 * sizeof(uint32_t);    // 48 (4-elem gap)
    const uint64_t DEV_BYTES = DEV_PITCH * H;            // 288
    {
        vx_buffer_h dbuf = nullptr;
        CHECK_VX(vx_buffer_create(dev, DEV_BYTES, VX_MEM_READ_WRITE, &dbuf));

        // Pre-fill the whole device buffer with a sentinel so untouched
        // gap bytes are detectable.
        uint32_t sentinel = 0xa5a5a5a5u;
        vx_event_h ev = nullptr;
        CHECK_VX(vx_enqueue_fill_buffer(q, dbuf, 0, DEV_BYTES, &sentinel,
                                        sizeof(sentinel), 0, nullptr, &ev));
        CHECK_VX(sync_ev(ev));

        // Tightly-packed host source: src[r*W + c] = r*100 + c.
        std::vector<uint32_t> src(W * H);
        for (uint32_t r = 0; r < H; ++r)
            for (uint32_t c = 0; c < W; ++c)
                src[r * W + c] = r * 100 + c;

        vx_rect_info_t rect = {};
        rect.struct_size       = sizeof(rect);
        rect.region[0]         = ROW_BYTES;
        rect.region[1]         = H;
        rect.region[2]         = 1;
        rect.buffer_row_pitch  = DEV_PITCH;     // strided device side
        rect.host_row_pitch    = 0;             // tight (-> ROW_BYTES)

        CHECK_VX(vx_enqueue_write_rect(q, dbuf, src.data(), &rect,
                                       0, nullptr, &ev));
        CHECK_VX(sync_ev(ev));

        // Linear read of the whole device buffer: payload rows must match
        // src, gap elements must still be the sentinel.
        std::vector<uint32_t> raw(DEV_BYTES / sizeof(uint32_t), 0);
        CHECK_VX(vx_enqueue_read(q, raw.data(), dbuf, 0, DEV_BYTES,
                                 0, nullptr, &ev));
        CHECK_VX(sync_ev(ev));
        const uint32_t pitch_elems = DEV_PITCH / sizeof(uint32_t);
        for (uint32_t r = 0; r < H; ++r) {
            for (uint32_t c = 0; c < pitch_elems; ++c) {
                uint32_t got = raw[r * pitch_elems + c];
                uint32_t want = (c < W) ? src[r * W + c] : sentinel;
                if (got != want) FAIL("write_rect payload/gap mismatch");
            }
        }

        // Read the rect back into a tightly-packed host array.
        std::vector<uint32_t> dst(W * H, 0);
        CHECK_VX(vx_enqueue_read_rect(q, dst.data(), dbuf, &rect,
                                      0, nullptr, &ev));
        CHECK_VX(sync_ev(ev));
        if (std::memcmp(src.data(), dst.data(), W * H * sizeof(uint32_t)) != 0)
            FAIL("read_rect round-trip mismatch");

        CHECK_VX(vx_buffer_release(dbuf));
        printf("write_rect/read_rect: OK\n");
    }

    // ------------------------------------------------------------------
    // 3) vx_enqueue_copy_rect — device-to-device strided rect copy.
    // ------------------------------------------------------------------
    {
        vx_buffer_h sbuf = nullptr, dbuf = nullptr;
        CHECK_VX(vx_buffer_create(dev, DEV_BYTES, VX_MEM_READ_WRITE, &sbuf));
        CHECK_VX(vx_buffer_create(dev, DEV_BYTES, VX_MEM_READ_WRITE, &dbuf));

        std::vector<uint32_t> src(W * H);
        for (uint32_t i = 0; i < W * H; ++i) src[i] = 0x1000u + i;

        vx_rect_info_t rect = {};
        rect.struct_size      = sizeof(rect);
        rect.region[0]        = ROW_BYTES;
        rect.region[1]        = H;
        rect.region[2]        = 1;
        rect.buffer_row_pitch = DEV_PITCH;
        rect.host_row_pitch   = 0;

        vx_event_h ev = nullptr;
        // Stage the source rect into sbuf.
        CHECK_VX(vx_enqueue_write_rect(q, sbuf, src.data(), &rect,
                                       0, nullptr, &ev));
        CHECK_VX(sync_ev(ev));

        // copy_rect: buffer_* = dst, host_* = src. Both device-side here.
        vx_rect_info_t crect = {};
        crect.struct_size        = sizeof(crect);
        crect.region[0]          = ROW_BYTES;
        crect.region[1]          = H;
        crect.region[2]          = 1;
        crect.buffer_row_pitch   = DEV_PITCH;   // dst pitch
        crect.host_row_pitch     = DEV_PITCH;   // src pitch
        CHECK_VX(vx_enqueue_copy_rect(q, dbuf, sbuf, &crect,
                                      0, nullptr, &ev));
        CHECK_VX(sync_ev(ev));

        std::vector<uint32_t> dst(W * H, 0);
        CHECK_VX(vx_enqueue_read_rect(q, dst.data(), dbuf, &rect,
                                      0, nullptr, &ev));
        CHECK_VX(sync_ev(ev));
        if (std::memcmp(src.data(), dst.data(), W * H * sizeof(uint32_t)) != 0)
            FAIL("copy_rect round-trip mismatch");

        CHECK_VX(vx_buffer_release(dbuf));
        CHECK_VX(vx_buffer_release(sbuf));
        printf("copy_rect: OK\n");
    }

    // ------------------------------------------------------------------
    // 4) vx_enqueue_map / vx_enqueue_unmap — WRITE map fills the buffer,
    //    READ map reads it back.
    // ------------------------------------------------------------------
    {
        const uint32_t kCount = 32;
        const uint64_t kBytes = kCount * sizeof(uint32_t);
        vx_buffer_h buf = nullptr;
        CHECK_VX(vx_buffer_create(dev, kBytes, VX_MEM_READ_WRITE, &buf));

        vx_event_h ev = nullptr;
        void* wptr = nullptr;
        CHECK_VX(vx_enqueue_map(q, buf, 0, kBytes, VX_MEM_WRITE,
                                0, nullptr, &ev, &wptr));
        CHECK_VX(sync_ev(ev));
        if (!wptr) FAIL("vx_enqueue_map returned NULL host ptr");
        uint32_t* w = static_cast<uint32_t*>(wptr);
        for (uint32_t i = 0; i < kCount; ++i) w[i] = 0xc0de0000u + i;
        CHECK_VX(vx_enqueue_unmap(q, buf, wptr, 0, nullptr, &ev));
        CHECK_VX(sync_ev(ev));

        void* rptr = nullptr;
        CHECK_VX(vx_enqueue_map(q, buf, 0, kBytes, VX_MEM_READ,
                                0, nullptr, &ev, &rptr));
        CHECK_VX(sync_ev(ev));
        if (!rptr) FAIL("vx_enqueue_map (READ) returned NULL host ptr");
        uint32_t* rd = static_cast<uint32_t*>(rptr);
        for (uint32_t i = 0; i < kCount; ++i)
            if (rd[i] != 0xc0de0000u + i) FAIL("map round-trip mismatch");
        CHECK_VX(vx_enqueue_unmap(q, buf, rptr, 0, nullptr, &ev));
        CHECK_VX(sync_ev(ev));

        CHECK_VX(vx_buffer_release(buf));
        printf("map/unmap: OK\n");
    }

    CHECK_VX(vx_queue_release(q));
    CHECK_VX(vx_device_release(dev));

    printf("PASSED\n");
    return 0;
}
