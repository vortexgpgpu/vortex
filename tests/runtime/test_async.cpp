// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

// ============================================================================
// test_async.cpp
//
// Exercises the asynchronous vortex2.h surface beyond what test_basic covers:
//   - Multiple concurrent queues on one device
//   - Async copy chain with event dependencies (q1 produces, q2 consumes)
//   - Timeline events as a host-side synchronization primitive
//   - vx_enqueue_barrier as an in-queue join point
//   - Profiling timestamps: queued <= submit <= start <= end
//   - Buffer map / unmap round-trip (READ before / WRITE after)
//   - vx_queue_finish drains all in-flight commands
//
// PASS: all assertions hold, exit code 0.
// ============================================================================

#include <vortex2.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

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

// ---------------------------------------------------------------------------
// Two concurrent queues and an event chain.
// q1 writes pattern A to bufA, signals event eA.
// q2 waits on eA, then copies bufA -> bufB.
// Final state: bufB == pattern A.
// ---------------------------------------------------------------------------
int test_event_chain(vx_device_h dev) {
    constexpr uint64_t N = 256;
    const uint64_t bytes = N * sizeof(uint32_t);

    vx_queue_info_t qi = {};
    qi.struct_size = sizeof(qi);
    qi.priority    = VX_QUEUE_PRIORITY_NORMAL;
    qi.flags       = VX_QUEUE_PROFILING_ENABLE;

    vx_queue_h q1 = nullptr, q2 = nullptr;
    CHECK_VX(vx_queue_create(dev, &qi, &q1));
    CHECK_VX(vx_queue_create(dev, &qi, &q2));

    vx_buffer_h bufA = nullptr, bufB = nullptr;
    CHECK_VX(vx_buffer_create(dev, bytes, VX_MEM_READ_WRITE, &bufA));
    CHECK_VX(vx_buffer_create(dev, bytes, VX_MEM_READ_WRITE, &bufB));

    std::vector<uint32_t> patternA(N);
    for (uint32_t i = 0; i < N; ++i) patternA[i] = 0xA0000000u | i;

    // q1: host -> bufA, produce event eA
    vx_event_h eA = nullptr;
    CHECK_VX(vx_enqueue_write(q1, bufA, 0, patternA.data(), bytes,
                              0, nullptr, &eA));

    // q2: bufA -> bufB, gated on eA from q1
    vx_event_h eB = nullptr;
    CHECK_VX(vx_enqueue_copy(q2, bufB, 0, bufA, 0, bytes,
                             1, &eA, &eB));

    // host: read back bufB after eB completes
    std::vector<uint32_t> out(N, 0xdeadbeef);
    vx_event_h eRead = nullptr;
    CHECK_VX(vx_enqueue_read(q2, out.data(), bufB, 0, bytes,
                             1, &eB, &eRead));

    CHECK_VX(vx_event_wait_value(eRead, 1, VX_TIMEOUT_INFINITE));

    for (uint32_t i = 0; i < N; ++i) {
        if (out[i] != patternA[i]) {
            fprintf(stderr, "FAILED: q1->q2 chain mismatch at %u: got 0x%x exp 0x%x\n",
                    i, out[i], patternA[i]);
            return 1;
        }
    }

    CHECK_VX(vx_event_release(eA));
    CHECK_VX(vx_event_release(eB));
    CHECK_VX(vx_event_release(eRead));
    CHECK_VX(vx_buffer_release(bufA));
    CHECK_VX(vx_buffer_release(bufB));
    CHECK_VX(vx_queue_release(q1));
    CHECK_VX(vx_queue_release(q2));
    return 0;
}

// ---------------------------------------------------------------------------
// Timeline event lifecycle and host-side cross-thread signaling.
// ---------------------------------------------------------------------------
int test_user_event(vx_device_h dev) {
    vx_event_h gate = nullptr;
    CHECK_VX(vx_event_create(dev, &gate));

    uint64_t v;
    CHECK_VX(vx_event_get_value(gate, &v));
    EXPECT(v == 0, "fresh event counter not 0");

    // A 10 ms wait on an unsignaled event must time out (not succeed).
    auto r = vx_event_wait_value(gate, 1, 10ull * 1000 * 1000);
    EXPECT(r == VX_ERR_TIMEOUT, "wait on unsignaled event should TIMEOUT");

    // Background signaller. Main thread waits with INFINITE; the signaller
    // releases it after a delay.
    std::thread signaller([gate]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        vx_event_signal(gate, 1);
    });
    CHECK_VX(vx_event_wait_value(gate, 1, VX_TIMEOUT_INFINITE));
    signaller.join();

    CHECK_VX(vx_event_get_value(gate, &v));
    EXPECT(v >= 1, "signaled event counter not advanced");

    // A second wait should return immediately (counter already reached).
    CHECK_VX(vx_event_wait_value(gate, 1, 0));

    CHECK_VX(vx_event_release(gate));
    return 0;
}

// ---------------------------------------------------------------------------
// Enqueue gated on a user event. With the per-queue worker thread, the
// enqueue returns immediately even though its dep is unsignaled; the worker
// blocks instead. A background thread signals the gate, the worker unblocks,
// the copy completes.
// ---------------------------------------------------------------------------
int test_user_event_gated_enqueue(vx_device_h dev) {
    constexpr uint64_t bytes = 128;

    vx_queue_info_t qi = {};
    qi.struct_size = sizeof(qi);
    vx_queue_h q = nullptr;
    CHECK_VX(vx_queue_create(dev, &qi, &q));

    vx_buffer_h src = nullptr, dst = nullptr;
    CHECK_VX(vx_buffer_create(dev, bytes, VX_MEM_READ_WRITE, &src));
    CHECK_VX(vx_buffer_create(dev, bytes, VX_MEM_READ_WRITE, &dst));

    std::vector<uint8_t> pat(bytes);
    for (size_t i = 0; i < bytes; ++i) pat[i] = (uint8_t)(0xE0 + (i & 0x1F));

    // Prime src with the pattern.
    vx_event_h ePrime = nullptr;
    CHECK_VX(vx_enqueue_write(q, src, 0, pat.data(), bytes, 0, nullptr, &ePrime));
    CHECK_VX(vx_event_wait_value(ePrime, 1, VX_TIMEOUT_INFINITE));
    CHECK_VX(vx_event_release(ePrime));

    // Issue a copy gated on an unsignaled user event. The enqueue MUST
    // return promptly (no deadlock); the worker will block on the gate.
    vx_event_h gate = nullptr;
    CHECK_VX(vx_event_create(dev, &gate));

    auto t_enqueue_start = std::chrono::steady_clock::now();
    vx_event_h eCopy = nullptr;
    CHECK_VX(vx_enqueue_copy(q, dst, 0, src, 0, bytes, 1, &gate, &eCopy));
    auto t_enqueue_end = std::chrono::steady_clock::now();
    auto enqueue_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                          t_enqueue_end - t_enqueue_start).count();
    EXPECT(enqueue_ms < 50, "enqueue_copy on unsignaled gate did not return promptly");

    // Confirm the copy hasn't completed before the gate signal.
    uint64_t cv;
    CHECK_VX(vx_event_get_value(eCopy, &cv));
    EXPECT(cv == 0, "copy completed before gate signal");

    // Signal the gate from a background thread.
    std::thread signaller([gate]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        vx_event_signal(gate, 1);
    });

    CHECK_VX(vx_event_wait_value(eCopy, 1, VX_TIMEOUT_INFINITE));
    signaller.join();

    // Verify the copy actually executed (dst now matches pat).
    std::vector<uint8_t> out(bytes, 0);
    vx_event_h eRead = nullptr;
    CHECK_VX(vx_enqueue_read(q, out.data(), dst, 0, bytes, 0, nullptr, &eRead));
    CHECK_VX(vx_event_wait_value(eRead, 1, VX_TIMEOUT_INFINITE));
    for (size_t i = 0; i < bytes; ++i) {
        if (out[i] != pat[i]) {
            fprintf(stderr, "FAILED: gated copy mismatch at %zu: got 0x%x exp 0x%x\n",
                    i, out[i], pat[i]);
            return 1;
        }
    }

    CHECK_VX(vx_event_release(gate));
    CHECK_VX(vx_event_release(eCopy));
    CHECK_VX(vx_event_release(eRead));
    CHECK_VX(vx_buffer_release(src));
    CHECK_VX(vx_buffer_release(dst));
    CHECK_VX(vx_queue_release(q));
    return 0;
}

// ---------------------------------------------------------------------------
// vx_enqueue_barrier as a join point inside a single queue.
// Issues N writes with no inter-dependency, then a barrier, then a marker copy.
// The marker event completes only after all prior writes finish.
// ---------------------------------------------------------------------------
int test_barrier(vx_device_h dev) {
    constexpr uint32_t N_WRITES = 8;
    constexpr uint64_t chunk    = 32;

    vx_queue_info_t qi = {};
    qi.struct_size = sizeof(qi);
    vx_queue_h q = nullptr;
    CHECK_VX(vx_queue_create(dev, &qi, &q));

    vx_buffer_h buf = nullptr;
    CHECK_VX(vx_buffer_create(dev, N_WRITES * chunk, VX_MEM_READ_WRITE, &buf));

    std::vector<std::vector<uint8_t>> patterns(N_WRITES, std::vector<uint8_t>(chunk));
    std::vector<vx_event_h> write_events(N_WRITES, nullptr);
    for (uint32_t i = 0; i < N_WRITES; ++i) {
        for (uint64_t b = 0; b < chunk; ++b)
            patterns[i][b] = (uint8_t)(0x30 + i);
        CHECK_VX(vx_enqueue_write(q, buf, i * chunk, patterns[i].data(), chunk,
                                  0, nullptr, &write_events[i]));
    }

    vx_event_h eBarrier = nullptr;
    CHECK_VX(vx_enqueue_barrier(q, 0, nullptr, &eBarrier));
    CHECK_VX(vx_event_wait_value(eBarrier, 1, VX_TIMEOUT_INFINITE));

    // Every prior write event should now be complete.
    for (uint32_t i = 0; i < N_WRITES; ++i) {
        uint64_t v;
        CHECK_VX(vx_event_get_value(write_events[i], &v));
        if (v < 1) {
            fprintf(stderr, "FAILED: write[%u] not complete after barrier (v=%llu)\n",
                    i, (unsigned long long)v);
            return 1;
        }
    }

    std::vector<uint8_t> out(N_WRITES * chunk, 0);
    vx_event_h eRead = nullptr;
    CHECK_VX(vx_enqueue_read(q, out.data(), buf, 0, N_WRITES * chunk,
                             0, nullptr, &eRead));
    CHECK_VX(vx_event_wait_value(eRead, 1, VX_TIMEOUT_INFINITE));
    for (uint32_t i = 0; i < N_WRITES; ++i) {
        for (uint64_t b = 0; b < chunk; ++b) {
            if (out[i * chunk + b] != patterns[i][b]) {
                fprintf(stderr, "FAILED: barrier chunk %u offset %lu mismatch\n", i, b);
                return 1;
            }
        }
    }

    for (auto e : write_events) CHECK_VX(vx_event_release(e));
    CHECK_VX(vx_event_release(eBarrier));
    CHECK_VX(vx_event_release(eRead));
    CHECK_VX(vx_buffer_release(buf));
    CHECK_VX(vx_queue_release(q));
    return 0;
}

// ---------------------------------------------------------------------------
// Profiling timestamps form a non-decreasing chain.
// ---------------------------------------------------------------------------
int test_profiling(vx_device_h dev) {
    vx_queue_info_t qi = {};
    qi.struct_size = sizeof(qi);
    qi.flags       = VX_QUEUE_PROFILING_ENABLE;
    vx_queue_h q = nullptr;
    CHECK_VX(vx_queue_create(dev, &qi, &q));

    vx_buffer_h src = nullptr, dst = nullptr;
    CHECK_VX(vx_buffer_create(dev, 1024, VX_MEM_READ_WRITE, &src));
    CHECK_VX(vx_buffer_create(dev, 1024, VX_MEM_READ_WRITE, &dst));

    std::vector<uint8_t> pat(1024, 0x77);
    vx_event_h eW = nullptr, eC = nullptr;
    CHECK_VX(vx_enqueue_write(q, src, 0, pat.data(), 1024, 0, nullptr, &eW));
    CHECK_VX(vx_enqueue_copy (q, dst, 0, src, 0, 1024, 1, &eW, &eC));
    CHECK_VX(vx_event_wait_value(eC, 1, VX_TIMEOUT_INFINITE));

    vx_profile_info_t pW = {}, pC = {};
    CHECK_VX(vx_event_get_profiling(eW, &pW));
    CHECK_VX(vx_event_get_profiling(eC, &pC));

    EXPECT(pW.queued_ns <= pW.submit_ns, "W: queued > submit");
    EXPECT(pW.submit_ns <= pW.start_ns,  "W: submit > start");
    EXPECT(pW.start_ns  <= pW.end_ns,    "W: start > end");
    EXPECT(pC.queued_ns <= pC.submit_ns, "C: queued > submit");
    EXPECT(pC.submit_ns <= pC.start_ns,  "C: submit > start");
    EXPECT(pC.start_ns  <= pC.end_ns,    "C: start > end");
    EXPECT(pC.queued_ns >= pW.queued_ns, "C: queued before W");

    CHECK_VX(vx_event_release(eW));
    CHECK_VX(vx_event_release(eC));
    CHECK_VX(vx_buffer_release(src));
    CHECK_VX(vx_buffer_release(dst));
    CHECK_VX(vx_queue_release(q));
    return 0;
}

// ---------------------------------------------------------------------------
// Buffer map / unmap round-trip: write via map(WRITE), read via map(READ).
// ---------------------------------------------------------------------------
int test_map_unmap(vx_device_h dev) {
    constexpr uint64_t bytes = 512;
    vx_buffer_h buf = nullptr;
    CHECK_VX(vx_buffer_create(dev, bytes, VX_MEM_READ_WRITE, &buf));

    // Map for write, fill, unmap.
    void* hp = nullptr;
    CHECK_VX(vx_buffer_map(buf, 0, bytes, VX_MEM_WRITE, &hp));
    EXPECT(hp != nullptr, "map(WRITE) returned NULL host ptr");
    auto* w = static_cast<uint16_t*>(hp);
    for (uint64_t i = 0; i < bytes / 2; ++i) w[i] = (uint16_t)(0x5A00 + i);
    CHECK_VX(vx_buffer_unmap(buf, hp));

    // Map for read, verify, unmap.
    void* hpr = nullptr;
    CHECK_VX(vx_buffer_map(buf, 0, bytes, VX_MEM_READ, &hpr));
    EXPECT(hpr != nullptr, "map(READ) returned NULL host ptr");
    auto* r = static_cast<const uint16_t*>(hpr);
    for (uint64_t i = 0; i < bytes / 2; ++i) {
        if (r[i] != (uint16_t)(0x5A00 + i)) {
            fprintf(stderr, "FAILED: map-roundtrip mismatch at %lu: got 0x%x\n",
                    i, r[i]);
            return 1;
        }
    }
    CHECK_VX(vx_buffer_unmap(buf, hpr));

    CHECK_VX(vx_buffer_release(buf));
    return 0;
}

// ---------------------------------------------------------------------------
// vx_queue_finish drains all in-flight commands.
// ---------------------------------------------------------------------------
int test_queue_finish(vx_device_h dev) {
    vx_queue_info_t qi = {};
    qi.struct_size = sizeof(qi);
    vx_queue_h q = nullptr;
    CHECK_VX(vx_queue_create(dev, &qi, &q));

    vx_buffer_h buf = nullptr;
    CHECK_VX(vx_buffer_create(dev, 256, VX_MEM_READ_WRITE, &buf));

    constexpr uint32_t N = 6;
    std::vector<vx_event_h> evs(N);
    std::vector<uint8_t> pat(64, 0xC3);
    for (uint32_t i = 0; i < N; ++i) {
        CHECK_VX(vx_enqueue_write(q, buf, 0, pat.data(), 64, 0, nullptr, &evs[i]));
    }
    CHECK_VX(vx_queue_finish(q, VX_TIMEOUT_INFINITE));

    for (uint32_t i = 0; i < N; ++i) {
        uint64_t v;
        CHECK_VX(vx_event_get_value(evs[i], &v));
        if (v < 1) {
            fprintf(stderr, "FAILED: ev[%u] not complete after finish (v=%llu)\n",
                    i, (unsigned long long)v);
            return 1;
        }
        CHECK_VX(vx_event_release(evs[i]));
    }

    CHECK_VX(vx_buffer_release(buf));
    CHECK_VX(vx_queue_release(q));
    return 0;
}

// ---------------------------------------------------------------------------
// Multi-queue concurrent stress.
// Spawn Q queues, each independently enqueuing N writes to its own buffer.
// After all enqueues, finish all queues and verify every buffer holds the
// expected pattern.
// ---------------------------------------------------------------------------
int test_concurrent_queues(vx_device_h dev) {
    constexpr uint32_t Q     = 4;
    constexpr uint32_t N     = 8;
    constexpr uint64_t bytes = 64;

    vx_queue_info_t qi = {};
    qi.struct_size = sizeof(qi);
    std::vector<vx_queue_h>  queues(Q, nullptr);
    std::vector<vx_buffer_h> bufs  (Q, nullptr);
    for (uint32_t qi_idx = 0; qi_idx < Q; ++qi_idx) {
        CHECK_VX(vx_queue_create(dev, &qi, &queues[qi_idx]));
        CHECK_VX(vx_buffer_create(dev, N * bytes, VX_MEM_READ_WRITE,
                                  &bufs[qi_idx]));
    }

    // Per-queue patterns: byte = 0xA0 | (qid << 3) | (i & 0x07)
    std::vector<std::vector<std::vector<uint8_t>>> pats(
        Q, std::vector<std::vector<uint8_t>>(N, std::vector<uint8_t>(bytes)));
    for (uint32_t qid = 0; qid < Q; ++qid) {
        for (uint32_t i = 0; i < N; ++i) {
            uint8_t v = (uint8_t)(0xA0 | (qid << 3) | (i & 0x07));
            for (uint64_t b = 0; b < bytes; ++b) pats[qid][i][b] = v;
        }
    }

    // Enqueue everything; intentionally don't wait inline.
    for (uint32_t qid = 0; qid < Q; ++qid) {
        for (uint32_t i = 0; i < N; ++i) {
            CHECK_VX(vx_enqueue_write(queues[qid], bufs[qid], i * bytes,
                                      pats[qid][i].data(), bytes,
                                      0, nullptr, nullptr));
        }
    }

    // Drain all queues.
    for (uint32_t qid = 0; qid < Q; ++qid) {
        CHECK_VX(vx_queue_finish(queues[qid], VX_TIMEOUT_INFINITE));
    }

    // Verify each buffer.
    std::vector<uint8_t> out(N * bytes, 0);
    for (uint32_t qid = 0; qid < Q; ++qid) {
        vx_event_h eRead = nullptr;
        CHECK_VX(vx_enqueue_read(queues[qid], out.data(), bufs[qid], 0,
                                 N * bytes, 0, nullptr, &eRead));
        CHECK_VX(vx_event_wait_value(eRead, 1, VX_TIMEOUT_INFINITE));
        CHECK_VX(vx_event_release(eRead));
        for (uint32_t i = 0; i < N; ++i) {
            for (uint64_t b = 0; b < bytes; ++b) {
                if (out[i * bytes + b] != pats[qid][i][b]) {
                    fprintf(stderr, "FAILED: queue %u chunk %u byte %lu: got 0x%x exp 0x%x\n",
                            qid, i, b, out[i * bytes + b], pats[qid][i][b]);
                    return 1;
                }
            }
        }
    }

    for (uint32_t qid = 0; qid < Q; ++qid) {
        CHECK_VX(vx_buffer_release(bufs[qid]));
        CHECK_VX(vx_queue_release(queues[qid]));
    }
    return 0;
}

} // namespace

int main() {
    setvbuf(stdout, nullptr, _IOLBF, 0);   // line-buffered so timeouts still print progress
    vx_device_h dev = nullptr;
    CHECK_VX(vx_device_open(0, &dev));

    struct { const char* name; int (*fn)(vx_device_h); } tests[] = {
        { "event_chain",               test_event_chain               },
        { "user_event",                test_user_event                },
        { "user_event_gated_enqueue",  test_user_event_gated_enqueue  },
        { "barrier",                   test_barrier                   },
        { "profiling",                 test_profiling                 },
        { "map_unmap",                 test_map_unmap                 },
        { "queue_finish",              test_queue_finish              },
        { "concurrent_queues",         test_concurrent_queues         },
    };

    for (auto& t : tests) {
        printf("[RUN ] %s\n", t.name);
        int r = t.fn(dev);
        if (r != 0) {
            printf("[FAIL] %s\n", t.name);
            vx_device_release(dev);
            return 1;
        }
        printf("[ OK ] %s\n", t.name);
    }

    CHECK_VX(vx_device_release(dev));
    printf("PASSED\n");
    return 0;
}
