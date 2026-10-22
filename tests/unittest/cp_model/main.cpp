// Copyright © 2019-2023
// Licensed under the Apache License, Version 2.0.

// Unit test for the C++ command processor's launch descriptors: a QMD and an
// ILD replay their pairs and start the kernel; a descriptor off its 64-byte
// line is refused (no DCR write, no start) with Q_ERROR bit 0 set; a skipped
// indirect launch retires the commands of its range unexecuted.

#include "cmd_processor.h"
#include <cp_launch_abi.h>
#include <VX_types.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

#define EXPECT(cond, ...) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
        std::fprintf(stderr, __VA_ARGS__); \
        std::fprintf(stderr, "\n"); \
        std::exit(1); \
    } \
} while (0)

namespace {

constexpr uint64_t RING = 0x10000;
constexpr uint64_t CMPL = 0x20000;
constexpr uint64_t DESC = 0x30000;
constexpr uint64_t PRED = 0x40000;

std::vector<uint8_t> g_mem(1u << 20, 0);
std::vector<std::pair<uint32_t, uint32_t>> g_writes;
int g_starts = 0;
int g_busy = 0;

void wr32(uint64_t a, uint32_t v) { std::memcpy(&g_mem[a], &v, 4); }
void wr64(uint64_t a, uint64_t v) { std::memcpy(&g_mem[a], &v, 8); }

vortex::CommandProcessor::Hooks hooks() {
    vortex::CommandProcessor::Hooks h;
    h.dram_read  = [](uint64_t a, void* d, std::size_t n) { std::memcpy(d, &g_mem[a], n); };
    h.dram_write = [](uint64_t a, const void* s, std::size_t n) { std::memcpy(&g_mem[a], s, n); };
    h.vortex_dcr_write = [](uint32_t a, uint32_t v) { g_writes.push_back({a, v}); };
    h.vortex_dcr_read  = [](uint32_t, uint32_t) -> uint32_t { return 0; };
    h.vortex_start = [] { ++g_starts; g_busy = 3; };
    h.vortex_busy  = [] { return g_busy > 0 && g_busy-- > 0; };
    return h;
}

// One command per ring line, as the runtime submits them.
void put_cmd(int slot, uint8_t op, uint64_t a0) {
    uint8_t* cl = &g_mem[RING + 64 * slot];
    std::memset(cl, 0, 64);
    cl[0] = op;
    std::memcpy(cl + 4, &a0, 8);
}

void put_qmd(uint64_t at, uint32_t n) {
    wr32(at, n);
    wr32(at + 4, 0);
    for (uint32_t i = 0; i < n; ++i) {
        wr32(at + 8 + 8 * i, 0x010 + i);
        wr32(at + 12 + 8 * i, 0x100 + i);
    }
}

void put_ild(uint64_t at, uint32_t ctl, uint16_t skip, uint32_t n) {
    std::memset(&g_mem[at], 0, 64);
    wr32(at + vortex::CP_ILD_OFF_CTL, ctl);
    wr32(at + vortex::CP_ILD_OFF_SKIP, skip);
    wr64(at + vortex::CP_ILD_OFF_PRED, PRED);
    wr32(at + vortex::CP_ILD_OFF_COUNT, n);
    for (uint32_t i = 0; i < n; ++i) {
        wr32(at + 64 + 8 * i, 0x010 + i);
        wr32(at + 68 + 8 * i, 0x200 + i);
    }
}

// Run `n` ring commands from the start of the ring on a fresh CP.
uint32_t run(int n) {
    g_writes.clear();
    g_starts = 0;
    vortex::CommandProcessor cp(hooks());
    cp.mmio_write(0x100, uint32_t(RING));
    cp.mmio_write(0x104, 0);
    cp.mmio_write(0x110, uint32_t(CMPL));
    cp.mmio_write(0x114, 0);
    cp.mmio_write(0x118, 16);
    cp.mmio_write(0x11C, 1);
    cp.mmio_write(0x000, 1);
    cp.mmio_write(0x120, uint32_t(64 * n));
    cp.mmio_write(0x124, 0);
    for (int i = 0; i < 10000 && cp.busy(); ++i) {
        cp.tick();
    }
    EXPECT(!cp.busy(), "the CP did not drain");
    EXPECT(cp.mmio_read(0x128) == uint32_t(n), "retired %u commands, expected %d",
           cp.mmio_read(0x128), n);
    return cp.mmio_read(0x12C);
}

} // namespace

int main() {
    constexpr uint8_t OP_DCR_WRITE = 0x04, OP_LAUNCH_QMD = 0x0B, OP_LAUNCH_IND = 0x0D;

    put_qmd(DESC, 5);
    put_cmd(0, OP_LAUNCH_QMD, DESC);
    EXPECT(run(1) == 0, "aligned QMD set Q_ERROR");
    EXPECT(g_writes.size() == 5 && g_starts == 1, "aligned QMD: %zu writes, %d starts",
           g_writes.size(), g_starts);

    for (uint64_t off : {4ull, 8ull, 60ull}) {
        put_cmd(0, OP_LAUNCH_QMD, DESC + off);
        EXPECT(run(1) == vortex::CP_Q_ERROR_DESC_ALIGN, "QMD at +%llu not refused",
               (unsigned long long)off);
        EXPECT(g_writes.empty() && g_starts == 0, "refused QMD ran");

        put_ild(DESC + 0x100, 0, 0, 3);
        put_cmd(0, OP_LAUNCH_IND, DESC + 0x100 + off);
        EXPECT(run(1) == vortex::CP_Q_ERROR_DESC_ALIGN, "ILD at +%llu not refused",
               (unsigned long long)off);
        EXPECT(g_writes.empty() && g_starts == 0, "refused ILD ran");
    }

    // Predicated off (0 > 0 fails), skip the next two commands; the third runs.
    wr32(PRED, 0);
    put_ild(DESC + 0x200, vortex::CP_ILD_HAS_PRED, 2, 3);
    put_cmd(0, OP_LAUNCH_IND, DESC + 0x200);
    put_cmd(1, OP_DCR_WRITE, 0);
    put_cmd(2, OP_LAUNCH_QMD, DESC);
    put_cmd(3, OP_LAUNCH_QMD, DESC);
    EXPECT(run(4) == 0, "skip set Q_ERROR");
    EXPECT(g_writes.size() == 5 && g_starts == 1, "skip range: %zu writes, %d starts",
           g_writes.size(), g_starts);

    // Predicate true: the ILD replays its pairs and starts.
    wr32(PRED, 1);
    put_cmd(0, OP_LAUNCH_IND, DESC + 0x200);
    EXPECT(run(1) == 0, "aligned ILD set Q_ERROR");
    EXPECT(g_writes.size() == 3 && g_writes[0].second == 0x200 && g_starts == 1,
           "aligned ILD: %zu writes, %d starts", g_writes.size(), g_starts);

    std::printf("PASSED\n");
    return 0;
}
