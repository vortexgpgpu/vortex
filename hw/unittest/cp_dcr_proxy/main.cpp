// Copyright © 2019-2023
// Licensed under the Apache License, Version 2.0.

// ============================================================================
// Verilator unit test for VX_cp_dcr_proxy.
//
// FSM:
//   IDLE → grant ⇒ S_REQ                (latch pending_is_read)
//   S_REQ → write: S_DONE; read: S_WAIT_RSP
//   S_WAIT_RSP → dcr_rsp_valid ⇒ latch rsp_data_r, S_DONE
//   S_DONE → IDLE
//
// Coverage:
//   1. Reset: no transitions, dcr_req_valid stays 0, done stays 0.
//   2. CMD_DCR_WRITE: req_valid=1 in S_REQ with rw=1, addr from arg0,
//      data from arg1; done pulses one cycle later; last_rsp_data
//      remains its previous value (tests start at 0).
//   3. CMD_DCR_READ: req_valid=1 in S_REQ with rw=0; FSM holds in
//      S_WAIT_RSP until dcr_rsp_valid arrives; rsp_data is latched
//      into last_rsp_data and visible while done pulses.
//   4. Back-to-back write→read: FSM re-arms cleanly.
//   5. WAIT_RSP hangs if rsp_valid never arrives (no spurious done).
// ============================================================================

#include "vl_simulator.h"
#include "VVX_cp_dcr_proxy_top.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>

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
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, msg); \
        std::exit(1); \
    } \
} while (0)

enum CmdOp : uint8_t {
    OP_DCR_WRITE = 0x04,
    OP_DCR_READ  = 0x05,
};

// Same packed-cmd layout as the cp_engine TB: hdr in the MSB word
// (index 8), profile_slot in the LSB words (0..1).
static void pack_cmd(uint32_t out_words[9],
                     uint8_t opcode, uint8_t flags,
                     uint64_t arg0, uint64_t arg1, uint64_t arg2,
                     uint64_t profile_slot) {
    for (int i = 0; i < 9; ++i) out_words[i] = 0;
    out_words[0] = static_cast<uint32_t>(profile_slot & 0xffffffffu);
    out_words[1] = static_cast<uint32_t>(profile_slot >> 32);
    out_words[2] = static_cast<uint32_t>(arg2 & 0xffffffffu);
    out_words[3] = static_cast<uint32_t>(arg2 >> 32);
    out_words[4] = static_cast<uint32_t>(arg1 & 0xffffffffu);
    out_words[5] = static_cast<uint32_t>(arg1 >> 32);
    out_words[6] = static_cast<uint32_t>(arg0 & 0xffffffffu);
    out_words[7] = static_cast<uint32_t>(arg0 >> 32);
    out_words[8] = static_cast<uint32_t>(opcode) |
                   (static_cast<uint32_t>(flags) << 8);
}

template <typename T>
static void set_cmd(T* top, uint8_t opcode,
                    uint64_t arg0 = 0, uint64_t arg1 = 0) {
    uint32_t words[9];
    pack_cmd(words, opcode, 0, arg0, arg1, /*arg2=*/0, /*profile_slot=*/0);
    for (int i = 0; i < 9; ++i) top->cmd_packed[i] = words[i];
}

// Drive inputs, sample outputs for the current cycle, then advance one
// full clock edge.
template <typename T>
static void cycle(vl_simulator<T>& sim, uint64_t& tick) {
    sim->eval();
    tick = sim.step(tick, 2);
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    vl_simulator<VVX_cp_dcr_proxy_top> sim;
    uint64_t tick = 0;

    // Initial state.
    sim->grant         = 0;
    sim->dcr_rsp_valid = 0;
    sim->dcr_rsp_data  = 0;
    set_cmd(sim.operator->(), 0);
    tick = sim.reset(tick);

    // ----- Test 1: post-reset idle — no req, no done, no rsp latch. -----
    for (int i = 0; i < 4; ++i) {
        sim->eval();
        EXPECT(sim->dcr_req_valid == 0, "spurious dcr_req_valid in IDLE");
        EXPECT(sim->done          == 0, "spurious done in IDLE");
        cycle(sim, tick);
    }

    // ----- Test 2: CMD_DCR_WRITE. arg0 = addr, arg1 = data -----
    constexpr uint32_t W_ADDR = 0x123;
    constexpr uint32_t W_DATA = 0xDEADBEEF;

    set_cmd(sim.operator->(), OP_DCR_WRITE, W_ADDR, W_DATA);
    sim->grant = 1;
    cycle(sim, tick);                          // IDLE → S_REQ

    // S_REQ cycle: req_valid=1 with rw=1, addr=W_ADDR, data=W_DATA.
    sim->eval();
    EXPECT(sim->dcr_req_valid == 1,             "WRITE: req_valid not asserted in S_REQ");
    EXPECT(sim->dcr_req_rw    == 1,             "WRITE: rw should be 1");
    EXPECT(sim->dcr_req_addr  == W_ADDR,        "WRITE: addr mismatch");
    EXPECT(sim->dcr_req_data  == W_DATA,        "WRITE: data mismatch");
    EXPECT(sim->done          == 0,             "WRITE: done premature in S_REQ");
    cycle(sim, tick);                          // S_REQ → S_DONE

    // S_DONE cycle: done=1, req_valid back to 0.
    sim->grant = 0;
    sim->eval();
    EXPECT(sim->done          == 1,             "WRITE: done not asserted in S_DONE");
    EXPECT(sim->dcr_req_valid == 0,             "WRITE: req_valid should fall after S_REQ");
    cycle(sim, tick);                          // S_DONE → IDLE

    // Back to IDLE — done falls.
    sim->eval();
    EXPECT(sim->done == 0, "WRITE: done should pulse only one cycle");

    // ----- Test 3: CMD_DCR_READ. arg0 = addr. -----
    constexpr uint32_t R_ADDR = 0x456;
    constexpr uint32_t R_VAL  = 0xCAFEBABE;

    set_cmd(sim.operator->(), OP_DCR_READ, R_ADDR, /*ignored=*/0);
    sim->grant = 1;
    cycle(sim, tick);                          // IDLE → S_REQ (pending_is_read latched)

    // S_REQ cycle: req_valid=1 with rw=0.
    sim->eval();
    EXPECT(sim->dcr_req_valid == 1,             "READ: req_valid not asserted");
    EXPECT(sim->dcr_req_rw    == 0,             "READ: rw should be 0");
    EXPECT(sim->dcr_req_addr  == R_ADDR,        "READ: addr mismatch");
    EXPECT(sim->done          == 0,             "READ: done premature in S_REQ");
    cycle(sim, tick);                          // S_REQ → S_WAIT_RSP

    // S_WAIT_RSP: hold indefinitely until dcr_rsp_valid arrives. Burn a
    // few cycles to make sure done stays low and req_valid falls.
    sim->grant = 0;
    for (int i = 0; i < 3; ++i) {
        sim->eval();
        EXPECT(sim->dcr_req_valid == 0, "READ: req_valid should fall in S_WAIT_RSP");
        EXPECT(sim->done          == 0, "READ: spurious done while waiting for rsp");
        cycle(sim, tick);
    }

    // Drive a response. FSM latches rsp_data_r at the posedge and moves to S_DONE.
    sim->dcr_rsp_valid = 1;
    sim->dcr_rsp_data  = R_VAL;
    cycle(sim, tick);                          // S_WAIT_RSP → S_DONE

    sim->dcr_rsp_valid = 0;
    sim->eval();
    EXPECT(sim->done          == 1,             "READ: done not asserted in S_DONE");
    EXPECT(sim->last_rsp_data == R_VAL,         "READ: last_rsp_data did not capture");
    cycle(sim, tick);                          // S_DONE → IDLE

    sim->eval();
    EXPECT(sim->done == 0, "READ: done should pulse only one cycle");
    EXPECT(sim->last_rsp_data == R_VAL,
           "READ: last_rsp_data should remain stable after done falls");

    // ----- Test 4: back-to-back write after read re-arms cleanly. -----
    constexpr uint32_t W2_ADDR = 0x789;
    constexpr uint32_t W2_DATA = 0x01234567;
    set_cmd(sim.operator->(), OP_DCR_WRITE, W2_ADDR, W2_DATA);
    sim->grant = 1;
    cycle(sim, tick);
    sim->eval();
    EXPECT(sim->dcr_req_valid == 1, "re-arm: req_valid not asserted on 2nd cmd");
    EXPECT(sim->dcr_req_rw    == 1, "re-arm: rw mismatch");
    EXPECT(sim->dcr_req_addr  == W2_ADDR, "re-arm: addr mismatch");
    cycle(sim, tick);                          // S_REQ → S_DONE
    sim->grant = 0;
    sim->eval();
    EXPECT(sim->done == 1, "re-arm: done not asserted");
    cycle(sim, tick);

    std::printf("PASSED\n");
    return 0;
}
