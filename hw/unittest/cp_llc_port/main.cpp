// Copyright © 2019-2023
// Licensed under the Apache License, Version 2.0.

// Verilator unit test for VX_cp_llc_port, the command processor's DMA port
// into the last-level cache.
//
// The harness plays the cache: requests enter an in-order input queue that the
// test can stall, and a request takes effect only when it leaves that queue, so
// an accepted DMA write is not yet visible while it sits in the port's buffer
// or in the queue. Every cycle the launch leaves the port (start_out), all DMA
// writes the port accepted must already have taken effect.
//
// Coverage:
//   1. A launch with no DMA write behind it passes straight through.
//   2. DMA reads return data and tag, and do not arm the launch fence.
//   3. A launch behind a DMA write held at the cache input waits for that
//      write, then for the fence token's answer; the token is a read with the
//      flush attribute and slot tag bit 0 set, and its answer is not forwarded
//      to the DMA. The port takes no DMA request while the fence is pending.
//   4. The fence clears: a following launch with no new write passes through.
//   5. Several writes into a cache that accepts and drains in bursts.
//   6. Request attributes: IO aperture, and no-allocate on writes only.

#include "vl_simulator.h"
#include "VVX_cp_llc_port_top.h"
#include <VX_types.h>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <unordered_map>

#ifndef TRACE_START_TIME
#define TRACE_START_TIME 0ull
#endif
#ifndef TRACE_STOP_TIME
#define TRACE_STOP_TIME -1ull
#endif

static uint64_t timestamp = 0;
static bool     trace_en  = false;
double sc_time_stamp() { return timestamp; }
bool   sim_trace_enabled() { return trace_en; }
void   sim_trace_enable(bool e) { trace_en = e; }

#define EXPECT(cond, msg) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d (cycle %llu): %s\n", __FILE__, __LINE__, \
                     (unsigned long long)cycles, msg); \
        std::exit(1); \
    } \
} while (0)

using Top = VVX_cp_llc_port_top;

static constexpr uint32_t LINE_SIZE = 64;
static constexpr uint32_t TAG_BITS  = 4;
static constexpr uint32_t DATA_LINE = 0x100000;   // above the IO aperture
static constexpr uint32_t IO_LINE   = VX_MEM_IO_BASE_ADDR / LINE_SIZE;

static_assert(VX_MEM_IO_BASE_ADDR % LINE_SIZE == 0, "IO base must be line aligned");
static_assert(DATA_LINE >= VX_MEM_IO_END_ADDR / LINE_SIZE, "data line must be outside the IO aperture");

struct CacheReq {
    bool     rw;
    bool     flush;
    uint32_t addr;
    uint32_t data;
    uint64_t tag;
};

struct CacheRsp {
    uint32_t data;
    uint64_t tag;
};

static uint64_t cycles = 0;

// The last-level cache seen through the port's slot.
struct Cache {
    bool                                   accept = true;  // input ready
    uint32_t                               period = 0;     // with duty: ready duty of every period cycles
    uint32_t                               duty   = 0;
    uint32_t                               service = 1;    // cycles per request
    std::deque<CacheReq>                   input;
    std::deque<CacheRsp>                   rsps;
    std::unordered_map<uint32_t, uint32_t> mem;
    uint32_t                               busy_cycles = 0;
    uint64_t                               writes_applied = 0;

    // Retire at most one request from the head of the input queue.
    void step() {
        if (input.empty()) {
            return;
        }
        if (++busy_cycles < service) {
            return;
        }
        busy_cycles = 0;
        CacheReq r = input.front();
        input.pop_front();
        if (r.flush) {
            rsps.push_back({0, r.tag});
        } else if (r.rw) {
            mem[r.addr] = r.data;
            ++writes_applied;
        } else {
            auto it = mem.find(r.addr);
            rsps.push_back({it != mem.end() ? it->second : 0, r.tag});
        }
    }
};

struct Bench {
    vl_simulator<Top> sim;
    uint64_t          tick = 0;
    Cache             cache;

    uint64_t dma_writes   = 0;   // DMA writes the port accepted
    uint64_t fence_tokens = 0;
    uint64_t starts       = 0;
    std::deque<CacheRsp> dma_rsps;

    Bench() {
        idle_inputs();
        tick = sim.reset(tick);
    }

    void idle_inputs() {
        sim->req_valid     = 0;
        sim->req_rw        = 0;
        sim->req_noalloc   = 0;
        sim->req_addr      = 0;
        sim->req_byteen    = 0;
        sim->req_tag       = 0;
        sim->rsp_ready     = 1;
        sim->start_in      = 0;
        sim->mem_req_ready = 0;
        sim->mem_rsp_valid = 0;
        sim->mem_rsp_tag   = 0;
        for (uint32_t i = 0; i < LINE_SIZE / 4; ++i) {
            sim->req_data[i]     = 0;
            sim->mem_rsp_data[i] = 0;
        }
    }

    // One clock: present the cache's outputs, sample every handshake on the
    // settled inputs, then advance the edge and the cache model.
    void cycle() {
        sim->mem_req_ready = cache.accept && (cache.period == 0 || (cycles % cache.period) < cache.duty);
        sim->mem_rsp_valid = !cache.rsps.empty();
        if (!cache.rsps.empty()) {
            sim->mem_rsp_tag     = cache.rsps.front().tag;
            sim->mem_rsp_data[0] = cache.rsps.front().data;
        }
        sim->eval();

        if (sim->start_out) {
            ++starts;
            EXPECT(cache.writes_applied == dma_writes,
                   "launch left the port before its DMA writes reached the cache");
        }
        if (sim->req_valid && sim->req_ready && sim->req_rw) {
            ++dma_writes;
        }
        if (sim->mem_req_valid && sim->mem_req_ready) {
            CacheReq r;
            r.rw    = sim->mem_req_rw;
            r.flush = sim->mem_req_flush;
            r.addr  = sim->mem_req_addr;
            r.data  = sim->mem_req_data[0];
            r.tag   = sim->mem_req_tag;
            if (r.flush) {
                ++fence_tokens;
                EXPECT(!r.rw, "fence token must be a read");
                EXPECT((r.tag & 1) == 1, "fence token must carry slot tag bit 0");
            } else {
                EXPECT((r.tag & 1) == 0, "DMA request must clear slot tag bit 0");
            }
            cache.input.push_back(r);
        }
        if (sim->mem_rsp_valid && sim->mem_rsp_ready) {
            CacheRsp rsp = cache.rsps.front();
            cache.rsps.pop_front();
            if ((rsp.tag & 1) == 0) {
                dma_rsps.push_back({rsp.data, (rsp.tag >> 1) & ((1u << TAG_BITS) - 1)});
            }
        }
        if (sim->rsp_valid && sim->rsp_ready) {
            EXPECT(!dma_rsps.empty(), "the port forwarded a response the DMA never asked for");
            EXPECT(sim->rsp_tag == dma_rsps.front().tag, "DMA response tag mismatch");
            EXPECT(sim->rsp_data[0] == dma_rsps.front().data, "DMA response data mismatch");
            dma_rsps.pop_front();
        }

        tick = sim.step(tick, 2);
        ++cycles;
        cache.step();
    }

    // Hold one DMA request until the port takes it.
    void dma_req(bool rw, uint32_t addr, uint32_t data, uint32_t tag, bool noalloc = false) {
        sim->req_valid   = 1;
        sim->req_rw      = rw;
        sim->req_noalloc = noalloc;
        sim->req_addr    = addr;
        sim->req_data[0] = data;
        sim->req_byteen  = ~0ull;
        sim->req_tag     = tag;
        for (int i = 0;; ++i) {
            EXPECT(i < 1000, "DMA request never accepted");
            sim->eval();
            bool fire = sim->req_ready;
            cycle();
            if (fire) {
                break;
            }
        }
        sim->req_valid = 0;
    }

    // Pulse start_in for one cycle; returns whether start_out fired with it.
    bool launch() {
        sim->start_in = 1;
        uint64_t before = starts;
        cycle();
        sim->start_in = 0;
        return starts != before;
    }

    void run(int n) {
        for (int i = 0; i < n; ++i) {
            cycle();
        }
    }

    // Wait for the launch to leave the port; returns the cycles waited.
    int wait_start(int limit) {
        uint64_t before = starts;
        for (int i = 0; i < limit; ++i) {
            cycle();
            if (starts != before) {
                return i + 1;
            }
        }
        EXPECT(false, "launch never left the port");
        return -1;
    }
};

static void test_launch_without_writes(Bench& b) {
    b.cache.accept = true;
    b.run(4);
    EXPECT(!b.sim->busy, "port busy at idle");
    EXPECT(b.launch(), "launch with no DMA write must pass through in the same cycle");
    b.run(8);
    EXPECT(b.fence_tokens == 0, "no fence token expected without a DMA write");
    EXPECT(!b.sim->busy, "port busy after a pass-through launch");
}

static void test_reads(Bench& b) {
    b.cache.accept = true;
    b.cache.mem[DATA_LINE + 7] = 0xcafe0007;
    b.cache.mem[DATA_LINE + 9] = 0xcafe0009;
    b.dma_req(false, DATA_LINE + 7, 0, 3);
    b.dma_req(false, DATA_LINE + 9, 0, 5);
    b.run(10);
    EXPECT(b.dma_rsps.empty(), "DMA read responses missing");
    uint64_t tokens = b.fence_tokens;
    EXPECT(b.launch(), "DMA reads must not hold the launch");
    b.run(4);
    EXPECT(b.fence_tokens == tokens, "DMA reads must not arm the fence");
}

static void test_fence_holds_launch(Bench& b) {
    constexpr int STALL = 64;
    b.cache.accept = false;
    b.dma_req(true, DATA_LINE + 1, 0x11111111, 1);
    EXPECT(b.cache.writes_applied < b.dma_writes, "write should still be held at the cache input");

    uint64_t starts = b.starts;
    EXPECT(!b.launch(), "launch passed the port with a DMA write still in front of the cache");
    for (int i = 0; i < STALL; ++i) {
        b.sim->eval();
        EXPECT(b.sim->busy, "port must be busy while the fence is pending");
        EXPECT(!b.sim->req_ready, "port must take no DMA request while the fence is pending");
        b.cycle();
    }
    EXPECT(b.starts == starts, "launch left the port while the cache input was stalled");

    // Release the cache slowly: the write, then the token, each take 6 cycles.
    uint64_t tokens = b.fence_tokens;
    b.cache.accept  = true;
    b.cache.service = 6;
    int waited = b.wait_start(200);
    EXPECT(waited >= 2 * 6, "launch left before the cache answered the fence token");
    EXPECT(b.fence_tokens == tokens + 1, "exactly one fence token expected");
    EXPECT(b.dma_rsps.empty(), "fence answer must not reach the DMA");
    b.run(2);
    EXPECT(!b.sim->busy, "port still busy after the fence answered");
    b.cache.service = 1;
}

static void test_fence_clears(Bench& b) {
    uint64_t tokens = b.fence_tokens;
    EXPECT(b.launch(), "a launch with no new DMA write must pass through after a fence");
    b.run(8);
    EXPECT(b.fence_tokens == tokens, "fence re-armed without a new DMA write");
}

static void test_write_bursts(Bench& b) {
    for (int round = 0; round < 8; ++round) {
        b.cache.service = 1 + (round % 4);
        b.cache.accept  = true;
        b.cache.period  = 5 + round;
        b.cache.duty    = 1 + (round % 3);
        for (int i = 0; i < 6; ++i) {
            b.dma_req(true, DATA_LINE + 32 * round + i, 0x20000000 + 16 * round + i, i & 0xf);
        }
        b.cache.period = 0;
        b.cache.accept = (round % 2) == 0;
        uint64_t starts = b.starts;
        bool passed = b.launch();
        EXPECT(!passed || b.cache.writes_applied == b.dma_writes, "launch passed undrained writes");
        if (!passed) {
            b.run(round * 5);
            b.cache.accept = true;
            for (int i = 0; b.starts == starts; ++i) {
                EXPECT(i < 500, "launch never left the port");
                b.cycle();
            }
        }
        b.run(4);
        EXPECT(!b.sim->busy, "port busy after a burst launch");
    }
    // Every written line reads back through the port.
    b.cache.accept  = true;
    b.cache.service = 1;
    b.dma_req(false, DATA_LINE + 32 * 7 + 5, 0, 2);
    b.run(10);
    EXPECT(b.cache.mem[DATA_LINE + 32 * 7 + 5] == 0x20000000 + 16 * 7 + 5, "burst write lost");
}

static void test_attributes(Bench& b) {
    b.cache.accept = false;
    struct Case { bool rw; uint32_t addr; bool noalloc; bool io; bool exp_noalloc; };
    const Case cases[] = {
        {true,  IO_LINE,       false, true,  false},
        {false, IO_LINE,       true,  true,  false},
        {true,  DATA_LINE + 3, true,  false, true },
        {false, DATA_LINE + 3, true,  false, false},
    };
    for (const auto& c : cases) {
        b.sim->req_valid   = 1;
        b.sim->req_rw      = c.rw;
        b.sim->req_noalloc = c.noalloc;
        b.sim->req_addr    = c.addr;
        b.sim->req_tag     = 1;
        b.sim->eval();
        EXPECT(b.sim->req_ready, "port buffer should be empty");
        b.cycle();
        b.sim->req_valid = 0;
        b.sim->eval();
        EXPECT(b.sim->mem_req_valid, "request missing at the cache input");
        EXPECT(b.sim->mem_req_io == c.io, "IO attribute mismatch");
        EXPECT(b.sim->mem_req_noalloc == c.exp_noalloc, "no-allocate attribute mismatch");
        EXPECT(!b.sim->mem_req_flush, "DMA request carries the flush attribute");
        b.cache.accept = true;
        b.cycle();
        b.cache.accept = false;
    }
    b.cache.accept = true;
    b.run(10);
    // Leave the fence armed by these writes discharged.
    b.launch();
    b.wait_start(100);
    b.run(4);
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    Bench b;

    test_launch_without_writes(b);
    test_reads(b);
    test_fence_holds_launch(b);
    test_fence_clears(b);
    test_write_bursts(b);
    test_attributes(b);

    EXPECT(b.cache.input.empty() && b.cache.rsps.empty(), "cache traffic left over");
    std::printf("PASSED\n");
    return 0;
}
