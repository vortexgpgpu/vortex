// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// DXA core lifetime tests. These requests deliberately have no software-side
// wait: the engine's busy contract alone must keep queued and active work live.

#include <iostream>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <VX_config.h>
#include <VX_types.h>
#include <dxa_meta.h>
#include "VVX_dxa_core_top.h"
#include "verilated.h"
#ifdef VCD_OUTPUT
#include "verilated_vcd_c.h"
#endif
#ifdef SAIF_OUTPUT
#include "verilated_saif_c.h"
#endif

#if defined(VCD_OUTPUT) && defined(SAIF_OUTPUT)
#error "VCD_OUTPUT and SAIF_OUTPUT cannot both be defined"
#endif

#define MAX_SIM_CYCLES 10000

static VVX_dxa_core_top* dut;
static uint64_t sim_time = 0;
static bool require_busy = false;

struct ExpectedWrite {
    uint32_t addr;
    uint32_t data;
    uint16_t byte_count;
    bool last;
    bool seen = false;
};

static std::vector<ExpectedWrite> expected_writes;

#ifdef VCD_OUTPUT
static VerilatedVcdC* vcd;
#endif
#ifdef SAIF_OUTPUT
static VerilatedSaifC* saif;
#endif

static void tick() {
    dut->clk ^= 1;
    dut->eval();
#ifdef VCD_OUTPUT
    vcd->dump(sim_time);
#endif
#ifdef SAIF_OUTPUT
    saif->dump(sim_time);
#endif
    ++sim_time;
}

[[noreturn]] static void fail(const std::string& message) {
    std::cerr << "FAILED at cycle " << (sim_time / 2) << ": " << message << std::endl;
    std::exit(1);
}

static void check(bool condition, const std::string& message) {
    if (!condition)
        fail(message);
}

static void observe_cycle() {
    dut->eval();

    if (require_busy)
        check(dut->busy, "busy deasserted while accepted DXA work remained");

    check(dut->gmem_req_valid == 0,
          "OOB-fill lifetime test unexpectedly issued a global-memory request");

    if (dut->lmem_req_valid && dut->lmem_req_ready) {
        auto it = std::find_if(expected_writes.begin(), expected_writes.end(),
            [](const ExpectedWrite& write) {
                return !write.seen && write.addr == dut->lmem_req_addr;
            });
        check(it != expected_writes.end(), "unexpected or duplicate LMEM write address");
        check(dut->lmem_req_rw, "DXA LMEM request was not a write");
        check(dut->lmem_req_data_lo == it->data, "LMEM payload did not match descriptor cfill");
        check(dut->lmem_req_byteen_count == it->byte_count, "LMEM byte-enable count mismatch");
        check(bool(dut->lmem_req_last) == it->last, "LMEM last-packet marker mismatch");
        it->seen = true;
    }
}

static void clock_cycle() {
    observe_cycle();
    tick(); // posedge
    tick(); // negedge
}

static void reset_dut(int cycles = 4) {
    dut->reset = 1;
    for (int i = 0; i < cycles; i++)
        clock_cycle();
    dut->reset = 0;
}

static void drive_defaults() {
    dut->dcr_req_valid      = 0;
    dut->dcr_req_rw         = 0;
    dut->dcr_req_addr       = 0;
    dut->dcr_req_data       = 0;
    dut->dxa_req_valid      = 0;
    dut->dxa_req_core_id    = 0;
    dut->dxa_req_uuid       = 0;
    dut->dxa_req_wid        = 0;
    dut->dxa_req_smem_addr  = 0;
    dut->dxa_req_meta       = 0;
    dut->dxa_req_coord0     = 0;
    dut->dxa_req_coord1     = 0;
    dut->dxa_req_coord2     = 0;
    dut->dxa_req_coord3     = 0;
    dut->dxa_req_coord4     = 0;
    dut->dxa_req_cta_mask   = 1;
    dut->gmem_req_ready     = 1;
    dut->gmem_rsp_valid     = 0;
    dut->lmem_req_ready     = 1;
}

static void write_dcr(uint32_t addr, uint32_t value) {
    dut->dcr_req_addr  = addr;
    dut->dcr_req_data  = value;
    dut->dcr_req_rw    = 1;
    dut->dcr_req_valid = 1;
    clock_cycle();
    dut->dcr_req_valid = 0;
}

static void program_oob_descriptor(uint32_t slot, uint32_t cfill) {
    const uint32_t base = VX_DCR_DXA_DESC_BASE + slot * VX_DCR_DXA_DESC_STRIDE;
    for (uint32_t word = 0; word < VX_DCR_DXA_DESC_STRIDE; ++word)
        write_dcr(base + word, 0);

    constexpr uint32_t rank = 2;
    constexpr uint32_t elem_size_log2 = 2;
    const uint32_t meta = (rank << DXA_DESC_META_DIM_LSB)
                        | (elem_size_log2 << DXA_DESC_META_ELEMSZ_LSB);
    write_dcr(base + VX_DCR_DXA_DESC_SIZE0_OFF, 16);
    write_dcr(base + VX_DCR_DXA_DESC_SIZE1_OFF, 1);
    write_dcr(base + VX_DCR_DXA_DESC_STRIDE0_OFF, 64);
    write_dcr(base + VX_DCR_DXA_DESC_META_OFF, meta);
    write_dcr(base + VX_DCR_DXA_DESC_ESTRIDE0_OFF, 1);
    write_dcr(base + VX_DCR_DXA_DESC_ESTRIDE1_OFF, 1);
    write_dcr(base + VX_DCR_DXA_DESC_ESTRIDE2_OFF, 1);
    write_dcr(base + VX_DCR_DXA_DESC_TILESIZE01_OFF, (1u << 16) | 16u);
    write_dcr(base + VX_DCR_DXA_DESC_CFILL_OFF, cfill);
}

static void issue_request(uint32_t slot, uint32_t smem_addr, uint32_t uuid) {
    dut->dxa_req_core_id   = 0;
    dut->dxa_req_uuid      = uuid;
    dut->dxa_req_wid       = 0;
    dut->dxa_req_smem_addr = smem_addr;
    dut->dxa_req_meta      = slot;
    dut->dxa_req_coord0    = 0;
    dut->dxa_req_coord1    = 1; // size1 is 1: select the OOB cfill path.
    dut->dxa_req_coord2    = 0;
    dut->dxa_req_coord3    = 0;
    dut->dxa_req_coord4    = 0;
    dut->dxa_req_cta_mask  = 1;
    dut->dxa_req_valid     = 1;

    for (int cycle = 0; cycle < MAX_SIM_CYCLES; ++cycle) {
        dut->eval();
        const bool accepted = dut->dxa_req_ready;
        clock_cycle();
        if (accepted) {
            dut->dxa_req_valid = 0;
            return;
        }
    }
    fail("DXA ingress never accepted request");
}

static constexpr uint32_t kTileBytes = 16 * sizeof(uint32_t);
static constexpr uint32_t kLmemWordBytes = VX_CFG_LMEM_NUM_BANKS * (VX_CFG_XLEN / 8);

static void expect_transfer(uint32_t smem_addr, uint32_t cfill) {
    static_assert((kLmemWordBytes & (kLmemWordBytes - 1)) == 0,
                  "LMEM word size must be a power of two");
    uint32_t word_shift = 0;
    while ((1u << word_shift) != kLmemWordBytes)
        ++word_shift;

    uint32_t remaining = kTileBytes;
    uint32_t word_addr = smem_addr >> word_shift;
    while (remaining != 0) {
        const uint16_t beat_bytes = static_cast<uint16_t>(std::min(remaining, kLmemWordBytes));
        remaining -= beat_bytes;
        expected_writes.push_back({word_addr++, cfill, beat_bytes, remaining == 0, false});
    }
}

static bool all_writes_seen() {
    return std::all_of(expected_writes.begin(), expected_writes.end(),
                       [](const ExpectedWrite& write) { return write.seen; });
}

static void wait_for_all_writes() {
    for (int cycle = 0; cycle < MAX_SIM_CYCLES; ++cycle) {
        if (all_writes_seen())
            return;
        clock_cycle();
    }
    fail("timed out waiting for expected LMEM writes");
}

static void wait_for_idle() {
    for (int cycle = 0; cycle < MAX_SIM_CYCLES; ++cycle) {
        dut->eval();
        if (!dut->busy)
            return;
        clock_cycle();
    }
    fail("DXA busy remained asserted after all work retired");
}

static void assert_quiescent() {
    for (int cycle = 0; cycle < 8; ++cycle) {
        dut->eval();
        check(!dut->busy, "DXA reasserted busy after quiescing");
        check(!dut->lmem_req_valid, "DXA emitted an LMEM write after quiescing");
        clock_cycle();
    }
}

static void test_fire_and_forget_last_backpressure() {
    constexpr uint32_t slot = 0;
    constexpr uint32_t smem_addr = 0x100;
    constexpr uint32_t cfill = 0xa55a31c7;

    expected_writes.clear();
    expect_transfer(smem_addr, cfill);
    require_busy = true;
    issue_request(slot, smem_addr, 1);

    // Let every non-final beat retire, then stop the final packet before its
    // handshake. This directly exercises the registered lmem-arb tail.
    bool found_last = false;
    for (int cycle = 0; cycle < MAX_SIM_CYCLES; ++cycle) {
        dut->eval();
        if (dut->lmem_req_valid && dut->lmem_req_last) {
            found_last = true;
            break;
        }
        clock_cycle();
    }
    check(found_last, "final LMEM packet never became visible");

    dut->lmem_req_ready = 0;
    dut->eval();
    const uint32_t held_addr = dut->lmem_req_addr;
    const uint32_t held_data = dut->lmem_req_data_lo;
    const uint16_t held_bytes = dut->lmem_req_byteen_count;
    for (int cycle = 0; cycle < 12; ++cycle) {
        check(dut->lmem_req_valid && dut->lmem_req_last,
              "final LMEM packet was not held under backpressure");
        check(dut->lmem_req_addr == held_addr && dut->lmem_req_data_lo == held_data
              && dut->lmem_req_byteen_count == held_bytes,
              "backpressured final LMEM packet changed");
        clock_cycle();
    }

    dut->lmem_req_ready = 1;
    clock_cycle();
    check(all_writes_seen(), "final LMEM packet did not retire after releasing backpressure");
    require_busy = false;
    wait_for_idle();
    assert_quiescent();
    std::cout << "PASS fire-and-forget + last-packet backpressure" << std::endl;
}

static void test_queued_descriptors() {
    constexpr uint32_t request_count = 6;
    constexpr uint32_t base_addr = 0x400;
    constexpr uint32_t addr_stride = 0x100;
    constexpr uint32_t cfill0 = 0x13579bdf;
    constexpr uint32_t cfill1 = 0x2468ace0;

    expected_writes.clear();
    for (uint32_t i = 0; i < request_count; ++i)
        expect_transfer(base_addr + i * addr_stride, (i & 1) ? cfill1 : cfill0);

    dut->lmem_req_ready = 0;
    require_busy = true;
    for (uint32_t i = 0; i < request_count; ++i)
        issue_request(i & 1, base_addr + i * addr_stride, i + 2);

    // With the output blocked, active/staged workers and the BRAM-backed queue
    // must retain all six launches without a false-idle pulse.
    for (int cycle = 0; cycle < 12; ++cycle)
        clock_cycle();

    dut->lmem_req_ready = 1;
    wait_for_all_writes();
    require_busy = false;
    wait_for_idle();
    assert_quiescent();
    std::cout << "PASS queued descriptors" << std::endl;
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);

    dut = new VVX_dxa_core_top;
    dut->clk   = 0;
    dut->reset = 0;

#ifdef VCD_OUTPUT
    Verilated::traceEverOn(true);
    vcd = new VerilatedVcdC;
    dut->trace(vcd, 99);
    const char* vcd_file = std::getenv("VCD_FILE");
    vcd->open(vcd_file ? vcd_file : "trace.vcd");
#endif
#ifdef SAIF_OUTPUT
    Verilated::traceEverOn(true);
    saif = new VerilatedSaifC;
    dut->trace(saif, 99);
    const char* saif_file = std::getenv("SAIF_FILE");
    saif->open(saif_file ? saif_file : "trace.saif");
#endif

    drive_defaults();

    reset_dut();
    check(!dut->busy, "busy asserted after reset with no work");

    program_oob_descriptor(0, 0xa55a31c7);
    program_oob_descriptor(1, 0x2468ace0);
    // Slot 0 is reused with a different cfill by the queue test.
    program_oob_descriptor(0, 0x13579bdf);
    wait_for_idle();

    // Restore the fire-and-forget descriptor just for that scenario.
    program_oob_descriptor(0, 0xa55a31c7);
    wait_for_idle();
    test_fire_and_forget_last_backpressure();

    program_oob_descriptor(0, 0x13579bdf);
    wait_for_idle();
    test_queued_descriptors();

    std::cout << "PASSED" << std::endl;

#ifdef VCD_OUTPUT
    vcd->close();
    delete vcd;
#endif
#ifdef SAIF_OUTPUT
    saif->close();
    delete saif;
#endif
    dut->final();
    delete dut;
    return 0;
}
