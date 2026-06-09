// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

// ============================================================================
// test_timeline_events.cpp
//
// Exercises the timeline event API:
//   - vx_event_create / signal / get_value / wait_value / wait_values
//   - vx_enqueue_signal / vx_enqueue_wait_value queue-ordered ops
//   - Counter monotonicity (signal with smaller value is a no-op)
//   - Multi-waiter satisfaction (one signal wakes all)
//   - Cross-queue rendezvous via shared timeline event
//
// PASS: all sections print [ OK ], exit code 0.
// ============================================================================

#include <vortex2.h>

#include <atomic>
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
// Section 1 — host-side signal/wait fundamentals.
//   - new event starts at 0
//   - signal(N) advances counter
//   - signal(M < N) is a no-op
//   - wait_value(N) returns immediately when counter >= N
// ---------------------------------------------------------------------------
int test_host_counter(vx_device_h dev) {
    vx_event_h ev = nullptr;
    CHECK_VX(vx_event_create(dev, &ev));

    uint64_t v = 0xdeadbeefull;
    CHECK_VX(vx_event_get_value(ev, &v));
    EXPECT(v == 0, "fresh event must start at counter=0");

    CHECK_VX(vx_event_signal(ev, 5));
    CHECK_VX(vx_event_get_value(ev, &v));
    EXPECT(v == 5, "signal(5) must advance counter to 5");

    // Monotonic — smaller value is a no-op.
    CHECK_VX(vx_event_signal(ev, 3));
    CHECK_VX(vx_event_get_value(ev, &v));
    EXPECT(v == 5, "signal(3) on counter=5 must not decrement");

    // Higher value advances.
    CHECK_VX(vx_event_signal(ev, 10));
    CHECK_VX(vx_event_get_value(ev, &v));
    EXPECT(v == 10, "signal(10) must advance counter to 10");

    // wait_value with already-satisfied target returns immediately.
    CHECK_VX(vx_event_wait_value(ev, 7, VX_TIMEOUT_INFINITE));
    CHECK_VX(vx_event_wait_value(ev, 10, VX_TIMEOUT_INFINITE));

    // wait_value with unmet target and zero timeout returns VX_ERR_TIMEOUT.
    auto r = vx_event_wait_value(ev, 11, 0);
    EXPECT(r == VX_ERR_TIMEOUT, "wait_value past current value with t=0 must time out");

    CHECK_VX(vx_event_release(ev));
    return 0;
}

// ---------------------------------------------------------------------------
// Section 2 — multi-waiter satisfaction.
//   - 4 threads each wait_value(ev, 100)
//   - main signals(100) once
//   - all 4 wake
// ---------------------------------------------------------------------------
int test_multi_waiter(vx_device_h dev) {
    vx_event_h ev = nullptr;
    CHECK_VX(vx_event_create(dev, &ev));

    constexpr int N = 4;
    std::atomic<int> wakes{0};
    std::vector<std::thread> ths;
    ths.reserve(N);
    for (int i = 0; i < N; ++i) {
        ths.emplace_back([ev, &wakes] {
            // 5 s ceiling so the test can fail rather than hang on a bug.
            vx_result_t r = vx_event_wait_value(ev, 100,
                                                5ull * 1000 * 1000 * 1000);
            if (r == VX_SUCCESS) wakes.fetch_add(1);
        });
    }

    // Give threads time to enter wait before signaling. Not required for
    // correctness — signal-then-late-wait also returns immediately — but
    // exercises the broadcast path explicitly.
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    CHECK_VX(vx_event_signal(ev, 100));

    for (auto& t : ths) t.join();
    EXPECT(wakes.load() == N, "all waiters must wake on a single signal");

    CHECK_VX(vx_event_release(ev));
    return 0;
}

// ---------------------------------------------------------------------------
// Section 3 — wait_values (multi-event multi-target).
//   - 3 events; per-event targets {2, 3, 1}; advance one at a time
//   - wait_values should block until all targets met
// ---------------------------------------------------------------------------
int test_wait_values(vx_device_h dev) {
    vx_event_h evs[3] = {};
    for (int i = 0; i < 3; ++i) CHECK_VX(vx_event_create(dev, &evs[i]));
    const uint64_t targets[3] = {2, 3, 1};

    std::atomic<bool> done{false};
    std::thread waiter([&]{
        vx_result_t r = vx_event_wait_values(3, evs, targets,
                                             5ull * 1000 * 1000 * 1000);
        if (r == VX_SUCCESS) done.store(true);
    });

    // Advance only some of the events; waiter must NOT be done.
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    CHECK_VX(vx_event_signal(evs[0], 2));
    CHECK_VX(vx_event_signal(evs[2], 1));
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    EXPECT(!done.load(), "wait_values must not return until ALL targets met");

    // Complete the last one.
    CHECK_VX(vx_event_signal(evs[1], 3));
    waiter.join();
    EXPECT(done.load(), "wait_values must return after all targets met");

    for (int i = 0; i < 3; ++i) CHECK_VX(vx_event_release(evs[i]));
    return 0;
}

// ---------------------------------------------------------------------------
// Section 4 — queue-ordered signal/wait via vx_enqueue_signal / wait_value.
//   - q1: enqueue_write to bufA  →  enqueue_signal(ev, 1)
//   - q2: enqueue_wait_value(ev, 1)  →  enqueue_copy bufA -> bufB
//   - host: enqueue_read bufB and verify pattern A
// ---------------------------------------------------------------------------
int test_queue_signal_wait(vx_device_h dev) {
    constexpr uint64_t N = 64;
    const uint64_t bytes = N * sizeof(uint32_t);

    vx_queue_info_t qi = {};
    qi.struct_size = sizeof(qi);
    qi.priority    = VX_QUEUE_PRIORITY_NORMAL;

    vx_queue_h q1 = nullptr, q2 = nullptr;
    CHECK_VX(vx_queue_create(dev, &qi, &q1));
    CHECK_VX(vx_queue_create(dev, &qi, &q2));

    vx_buffer_h bufA = nullptr, bufB = nullptr;
    CHECK_VX(vx_buffer_create(dev, bytes, VX_MEM_READ_WRITE, &bufA));
    CHECK_VX(vx_buffer_create(dev, bytes, VX_MEM_READ_WRITE, &bufB));

    std::vector<uint32_t> pattern(N);
    for (uint32_t i = 0; i < N; ++i) pattern[i] = 0xA0000000u | i;

    vx_event_h gate = nullptr;
    CHECK_VX(vx_event_create(dev, &gate));

    // q1: write pattern, then enqueue a signal at value 1.
    vx_event_h eW = nullptr;
    CHECK_VX(vx_enqueue_write(q1, bufA, 0, pattern.data(), bytes, 0, nullptr, &eW));
    vx_event_h eSig = nullptr;
    CHECK_VX(vx_enqueue_signal(q1, gate, 1, 0, nullptr, &eSig));

    // q2: wait on gate then copy bufA -> bufB.
    vx_event_h eWait = nullptr;
    CHECK_VX(vx_enqueue_wait_value(q2, gate, 1, 0, nullptr, &eWait));
    vx_event_h eCopy = nullptr;
    CHECK_VX(vx_enqueue_copy(q2, bufB, 0, bufA, 0, bytes, 0, nullptr, &eCopy));

    // Host: read bufB.
    std::vector<uint32_t> out(N, 0);
    vx_event_h eRead = nullptr;
    CHECK_VX(vx_enqueue_read(q2, out.data(), bufB, 0, bytes, 0, nullptr, &eRead));

    // Wait for the read; gate should have ratcheted to 1 by now.
    CHECK_VX(vx_event_wait_value(eRead, 1, VX_TIMEOUT_INFINITE));
    uint64_t gv = 0;
    CHECK_VX(vx_event_get_value(gate, &gv));
    EXPECT(gv >= 1, "gate must have been signaled by q1");

    for (uint32_t i = 0; i < N; ++i) {
        if (out[i] != pattern[i]) {
            fprintf(stderr, "FAILED: out[%u]=0x%08x, want 0x%08x\n",
                    i, out[i], pattern[i]);
            return 1;
        }
    }

    CHECK_VX(vx_event_release(eW));
    CHECK_VX(vx_event_release(eSig));
    CHECK_VX(vx_event_release(eWait));
    CHECK_VX(vx_event_release(eCopy));
    CHECK_VX(vx_event_release(eRead));
    CHECK_VX(vx_event_release(gate));
    CHECK_VX(vx_buffer_release(bufA));
    CHECK_VX(vx_buffer_release(bufB));
    CHECK_VX(vx_queue_release(q1));
    CHECK_VX(vx_queue_release(q2));
    return 0;
}

#define RUN(section)                                                     \
    do {                                                                  \
        printf("[RUN ] %s\n", #section);                                  \
        int rc = section(dev);                                            \
        if (rc != 0) { printf("[FAIL] %s\n", #section); return rc; }      \
        printf("[ OK ] %s\n", #section);                                  \
    } while (0)

} // namespace

int main() {
    vx_device_h dev = nullptr;
    CHECK_VX(vx_device_open(0, &dev));

    RUN(test_host_counter);
    RUN(test_multi_waiter);
    RUN(test_wait_values);
    RUN(test_queue_signal_wait);

    CHECK_VX(vx_device_release(dev));
    printf("PASSED\n");
    return 0;
}
