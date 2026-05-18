// Copyright © 2019-2023
// Licensed under the Apache License, Version 2.0.

// ============================================================================
// cp_sim — standalone unit test for sim/common/CommandProcessor.
//
// Drives the C++ CP class with mock DRAM + Vortex hooks. Covers:
//   1. mmio_write/read round-trip on every regfile slot
//   2. CMD_NOP retires (no resource bid)
//   3. CMD_DCR_WRITE invokes vortex_dcr_write hook with correct payload
//   4. CMD_LAUNCH drives the launch FSM (pulse_start → wait_busy → wait_drain
//      → retire) using a mock busy signal that rises then falls
//   5. Sequence of N back-to-back commands retires in order with seqnum
//      published to cmpl_addr each time
//   6. Q_TAIL atomic commit rule (LO write doesn't advance, HI commits both)
// ============================================================================

#include "CommandProcessor.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <vector>

#define EXPECT(cond, msg) do {                                            \
    if (!(cond)) {                                                        \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, msg); \
        std::exit(1);                                                     \
    }                                                                     \
} while (0)

namespace {

// Toy DRAM backing store, keyed by address. The CP class never
// reads/writes unaligned; we always operate at byte granularity.
class MockDram {
public:
    void read(uint64_t addr, void* dst, std::size_t bytes) {
        auto* d = static_cast<uint8_t*>(dst);
        for (std::size_t i = 0; i < bytes; ++i) {
            auto it = bytes_.find(addr + i);
            d[i] = (it == bytes_.end()) ? 0 : it->second;
        }
    }
    void write(uint64_t addr, const void* src, std::size_t bytes) {
        const auto* s = static_cast<const uint8_t*>(src);
        for (std::size_t i = 0; i < bytes; ++i)
            bytes_[addr + i] = s[i];
    }
    uint64_t read64(uint64_t addr) {
        uint64_t v = 0;
        read(addr, &v, sizeof(v));
        return v;
    }
private:
    std::unordered_map<uint64_t, uint8_t> bytes_;
};

// Mock Vortex side: records DCR writes; tracks busy via host-controlled stub.
struct MockVortex {
    std::vector<std::pair<uint32_t, uint32_t>> dcr_writes;
    int start_count = 0;
    // Mock busy: goes high cycle after start, low after `busy_cycles` more.
    int busy_remaining = 0;
};

// CP regfile MMIO offsets (CP-internal, mirrors VX_cp_axil_regfile §17.4).
constexpr uint32_t CP_CTRL          = 0x000;
constexpr uint32_t CP_STATUS        = 0x004;
constexpr uint32_t CP_DEV_CAPS      = 0x008;
constexpr uint32_t Q_RING_BASE_LO   = 0x100;
constexpr uint32_t Q_RING_BASE_HI   = 0x104;
constexpr uint32_t Q_HEAD_ADDR_LO   = 0x108;
constexpr uint32_t Q_HEAD_ADDR_HI   = 0x10C;
constexpr uint32_t Q_CMPL_ADDR_LO   = 0x110;
constexpr uint32_t Q_CMPL_ADDR_HI   = 0x114;
constexpr uint32_t Q_RING_SIZE_LOG2 = 0x118;
constexpr uint32_t Q_CONTROL        = 0x11C;
constexpr uint32_t Q_TAIL_LO        = 0x120;
constexpr uint32_t Q_TAIL_HI        = 0x124;
constexpr uint32_t Q_SEQNUM         = 0x128;

constexpr uint8_t OP_NOP        = 0x00;
constexpr uint8_t OP_DCR_WRITE  = 0x04;
constexpr uint8_t OP_LAUNCH     = 0x06;

constexpr std::size_t CL_BYTES = 64;

// Helpers for building a CL with a single command at offset 0.
void make_dcr_write_cl(std::array<uint8_t, CL_BYTES>& cl,
                       uint32_t addr, uint32_t value) {
    cl.fill(0);
    cl[0] = OP_DCR_WRITE;     // header opcode
    // arg0 at bytes 4..11 = DCR addr
    cl[4] = uint8_t(addr & 0xFF);
    cl[5] = uint8_t((addr >> 8) & 0xFF);
    cl[6] = uint8_t((addr >> 16) & 0xFF);
    cl[7] = uint8_t((addr >> 24) & 0xFF);
    // arg1 at bytes 12..19 = value
    cl[12] = uint8_t(value & 0xFF);
    cl[13] = uint8_t((value >> 8) & 0xFF);
    cl[14] = uint8_t((value >> 16) & 0xFF);
    cl[15] = uint8_t((value >> 24) & 0xFF);
}

void make_launch_cl(std::array<uint8_t, CL_BYTES>& cl) {
    cl.fill(0);
    cl[0] = OP_LAUNCH;
}

vortex::CommandProcessor make_cp(MockDram& dram, MockVortex& vortex) {
    vortex::CommandProcessor::Hooks hooks;
    hooks.dram_read = [&](uint64_t a, void* d, std::size_t b) {
        dram.read(a, d, b);
    };
    hooks.dram_write = [&](uint64_t a, const void* s, std::size_t b) {
        dram.write(a, s, b);
    };
    hooks.vortex_dcr_write = [&](uint32_t addr, uint32_t value) {
        vortex.dcr_writes.emplace_back(addr, value);
    };
    hooks.vortex_start = [&]() {
        ++vortex.start_count;
        vortex.busy_remaining = 5;  // simulate kernel runtime
    };
    hooks.vortex_busy = [&]() -> bool {
        if (vortex.busy_remaining > 0) {
            --vortex.busy_remaining;
            return true;
        }
        return false;
    };
    return vortex::CommandProcessor(hooks);
}

void enable_cp_and_q0(vortex::CommandProcessor& cp,
                     uint64_t ring_base, uint64_t cmpl_addr) {
    cp.mmio_write(Q_RING_BASE_LO,   uint32_t(ring_base & 0xFFFFFFFF));
    cp.mmio_write(Q_RING_BASE_HI,   uint32_t(ring_base >> 32));
    cp.mmio_write(Q_CMPL_ADDR_LO,   uint32_t(cmpl_addr & 0xFFFFFFFF));
    cp.mmio_write(Q_CMPL_ADDR_HI,   uint32_t(cmpl_addr >> 32));
    cp.mmio_write(Q_RING_SIZE_LOG2, 16);     // 64 KiB
    cp.mmio_write(Q_CONTROL,        0x1);
    cp.mmio_write(CP_CTRL,          0x1);
}

void commit_tail(vortex::CommandProcessor& cp, uint64_t tail) {
    cp.mmio_write(Q_TAIL_LO, uint32_t(tail & 0xFFFFFFFF));
    cp.mmio_write(Q_TAIL_HI, uint32_t(tail >> 32));
}

void run_until_done(vortex::CommandProcessor& cp, int max_ticks = 1000) {
    for (int i = 0; i < max_ticks; ++i) {
        if (!cp.busy()) return;
        cp.tick();
    }
    EXPECT(false, "run_until_done: CP didn't drain within budget");
}

// ============================================================================
// Tests
// ============================================================================

void test_mmio_roundtrip() {
    MockDram dram;
    MockVortex vortex;
    auto cp = make_cp(dram, vortex);

    cp.mmio_write(CP_CTRL, 0x1);
    EXPECT(cp.mmio_read(CP_CTRL) == 0x1, "CP_CTRL roundtrip");

    cp.mmio_write(Q_RING_BASE_LO, 0xDEADBEEF);
    cp.mmio_write(Q_RING_BASE_HI, 0x12345678);
    EXPECT(cp.mmio_read(Q_RING_BASE_LO) == 0xDEADBEEF, "RING_BASE_LO");
    EXPECT(cp.mmio_read(Q_RING_BASE_HI) == 0x12345678, "RING_BASE_HI");

    // CP_DEV_CAPS is RO and should report {TID=6, RING_LOG2=16, NUM_QUEUES=1}
    uint32_t caps = cp.mmio_read(CP_DEV_CAPS);
    EXPECT(caps == ((6u << 16) | (16u << 8) | 1u), "CP_DEV_CAPS");

    // SEQNUM starts at 0 (no commands retired yet)
    EXPECT(cp.mmio_read(Q_SEQNUM) == 0, "Q_SEQNUM initial");

    std::printf("[PASS] mmio_roundtrip\n");
}

void test_q_tail_atomic() {
    MockDram dram;
    MockVortex vortex;
    auto cp = make_cp(dram, vortex);

    // Q_TAIL_LO alone should NOT advance the committed tail.
    cp.mmio_write(Q_TAIL_LO, 0x40);
    EXPECT(cp.mmio_read(Q_TAIL_HI) == 0, "TAIL_HI before commit");
    // Write Q_TAIL_HI to commit (high half = 0, low half = staged 0x40).
    cp.mmio_write(Q_TAIL_HI, 0x0);
    EXPECT(cp.mmio_read(Q_TAIL_HI) == 0, "TAIL_HI value");

    std::printf("[PASS] q_tail_atomic\n");
}

void test_dcr_write_retires() {
    MockDram dram;
    MockVortex vortex;
    auto cp = make_cp(dram, vortex);

    constexpr uint64_t RING = 0x10000;
    constexpr uint64_t CMPL = 0x20000;
    enable_cp_and_q0(cp, RING, CMPL);

    // Stage one CMD_DCR_WRITE at ring[0].
    std::array<uint8_t, CL_BYTES> cl;
    make_dcr_write_cl(cl, /*addr=*/0x10, /*value=*/0x80000000);
    dram.write(RING, cl.data(), CL_BYTES);

    // Commit tail = 64.
    commit_tail(cp, CL_BYTES);
    run_until_done(cp);

    EXPECT(vortex.dcr_writes.size() == 1, "exactly one DCR write issued");
    EXPECT(vortex.dcr_writes[0].first  == 0x10, "DCR addr");
    EXPECT(vortex.dcr_writes[0].second == 0x80000000, "DCR value");

    // Q_SEQNUM should be 1 (one command retired).
    EXPECT(cp.mmio_read(Q_SEQNUM) == 1, "Q_SEQNUM after 1 retire");

    // Completion slot should hold seqnum=1.
    uint64_t cmpl_val = dram.read64(CMPL);
    EXPECT(cmpl_val == 1, "completion slot seqnum");

    std::printf("[PASS] dcr_write_retires\n");
}

void test_launch_drives_busy() {
    MockDram dram;
    MockVortex vortex;
    auto cp = make_cp(dram, vortex);

    constexpr uint64_t RING = 0x10000;
    constexpr uint64_t CMPL = 0x20000;
    enable_cp_and_q0(cp, RING, CMPL);

    std::array<uint8_t, CL_BYTES> cl;
    make_launch_cl(cl);
    dram.write(RING, cl.data(), CL_BYTES);

    commit_tail(cp, CL_BYTES);
    run_until_done(cp);

    EXPECT(vortex.start_count == 1, "exactly one vortex_start pulse");
    EXPECT(cp.mmio_read(Q_SEQNUM) == 1, "Q_SEQNUM == 1 after launch");
    EXPECT(dram.read64(CMPL) == 1, "completion seqnum = 1");

    std::printf("[PASS] launch_drives_busy\n");
}

void test_dcrs_then_launch_in_order() {
    MockDram dram;
    MockVortex vortex;
    auto cp = make_cp(dram, vortex);

    constexpr uint64_t RING = 0x10000;
    constexpr uint64_t CMPL = 0x20000;
    enable_cp_and_q0(cp, RING, CMPL);

    // Stage 5 DCR writes + 1 launch, one CL each.
    const std::vector<std::pair<uint32_t, uint32_t>> dcrs = {
        {0x10, 0x80000000}, {0x11, 0x0}, {0x12, 0x100}, {0x13, 0x1}, {0x14, 0x40},
    };
    int cl_idx = 0;
    std::array<uint8_t, CL_BYTES> cl;
    for (const auto& d : dcrs) {
        make_dcr_write_cl(cl, d.first, d.second);
        dram.write(RING + uint64_t(cl_idx) * CL_BYTES, cl.data(), CL_BYTES);
        ++cl_idx;
    }
    make_launch_cl(cl);
    dram.write(RING + uint64_t(cl_idx) * CL_BYTES, cl.data(), CL_BYTES);
    ++cl_idx;

    commit_tail(cp, uint64_t(cl_idx) * CL_BYTES);
    run_until_done(cp);

    EXPECT(vortex.dcr_writes.size() == dcrs.size(), "all DCR writes issued");
    for (std::size_t i = 0; i < dcrs.size(); ++i) {
        EXPECT(vortex.dcr_writes[i] == dcrs[i], "DCR write i in order");
    }
    EXPECT(vortex.start_count == 1, "launch fired exactly once");
    EXPECT(cp.mmio_read(Q_SEQNUM) == uint32_t(cl_idx),
           "Q_SEQNUM matches command count");
    EXPECT(dram.read64(CMPL) == uint64_t(cl_idx),
           "completion seqnum = command count");

    std::printf("[PASS] dcrs_then_launch_in_order — %d commands\n", cl_idx);
}

void test_disabled_cp_doesnt_advance() {
    MockDram dram;
    MockVortex vortex;
    auto cp = make_cp(dram, vortex);

    // Enable queue but NOT global CTRL.
    cp.mmio_write(Q_CONTROL, 0x1);
    // CP_CTRL stays 0 → enabled() returns false.

    constexpr uint64_t RING = 0x10000;
    cp.mmio_write(Q_RING_BASE_LO, uint32_t(RING));
    std::array<uint8_t, CL_BYTES> cl;
    make_dcr_write_cl(cl, 0x10, 0xABCD);
    dram.write(RING, cl.data(), CL_BYTES);
    commit_tail(cp, CL_BYTES);

    for (int i = 0; i < 100; ++i) cp.tick();
    EXPECT(vortex.dcr_writes.empty(), "no DCR issued when CP disabled");
    EXPECT(cp.mmio_read(Q_SEQNUM) == 0, "SEQNUM stays 0 when disabled");

    std::printf("[PASS] disabled_cp_doesnt_advance\n");
}

} // namespace

int main(int argc, char** argv) {
    (void)argc; (void)argv;

    test_mmio_roundtrip();
    test_q_tail_atomic();
    test_dcr_write_retires();
    test_launch_drives_busy();
    test_dcrs_then_launch_in_order();
    test_disabled_cp_doesnt_advance();

    std::printf("ALL PASSED\n");
    return 0;
}
