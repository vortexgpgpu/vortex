// Copyright © 2019-2023
// Licensed under the Apache License, Version 2.0.

// ============================================================================
// Verilator unit test for VX_cp_unpack.
//
// VX_cp_unpack is a SINGLE-command decoder: given a 64-byte cache line and a
// byte `offset`, it decodes exactly one packed cmd_t at that offset and
// reports `has_cmd`, the decoded `cmd`, and `cmd_size` (bytes consumed). It
// deasserts `has_cmd` on:
//   - end of line (no room for a 4 B header: offset + 4 > CL_BYTES)
//   - zero header (opcode=0 AND flags=0)  → host-side padding sentinel
//   - a command whose declared size would cross the CL boundary (malformed)
//
// Walking the whole line is the fetch FSM's job, not the unpacker's, so this
// harness plays that role: it starts at offset 0 and advances by `cmd_size`
// after each decoded command until `has_cmd` drops, collecting the commands.
//
// Per-command on-wire layout (little-endian within each field):
//   [hdr  4 B]  =  opcode(1) | flags(1) | reserved(2)
//   [arg0 8 B]
//   [arg1 8 B]
//   [arg2 8 B]   (only for opcodes that declare it)
//   [profile_slot 8 B] (only when F_PROFILE is set in hdr.flags)
//
// On-wire sizes per cmd_size_bytes(op, profiled):
//   NOP        : 4    + 8 if profiled    = 4 / 12
//   LAUNCH     : 12   + 8                = 12 / 20
//   FENCE      : 8    + 8                = 8 / 16
//   DCR_R/W    : 20   + 8                = 20 / 28
//   EVT_SIGNAL : 20   + 8                = 20 / 28
//   EVT_WAIT   : 28   + 8                = 28 / 36
//   MEM_*      : 28   + 8                = 28 / 36
//
// Coverage (counts are commands the harness walks off the line):
//   1. All-zero line → 0 cmds (line starts with the padding sentinel).
//   2. Single CMD_LAUNCH unprofiled → 1 cmd, hdr+arg0 round-trip, size=12.
//   3. Single CMD_LAUNCH profiled → profile_slot lands at offset+12, size=20.
//   4. Two-command line: DCR_WRITE (20 B) + MEM_COPY (28 B) = 48 B then
//      zero-pad → 2 cmds at offsets 0 and 20.
//   5. Three small commands: NOP+F_PROFILE (12 B) × 3 = 36 B + pad → 3 cmds.
//   6. Full line: 3 × MEM_COPY × 28 B = 84 B doesn't fit; only 2 land then
//      the third would cross the CL boundary → walker stops at 2
//      (malformed-tail rule).
//   7. Five commands: 5 × NOP+F_PROFILE (12 B) = 60 B + 4 B padding → 5 cmds.
// ============================================================================

#include "vl_simulator.h"
#include "VVX_cp_unpack_top.h"
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

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

static constexpr int CL_BYTES  = 64;
static constexpr int CMD_BITS  = 288;
static constexpr int F_PROFILE = 0;

enum CmdOp : uint8_t {
    OP_NOP        = 0x00,
    OP_MEM_WRITE  = 0x01,
    OP_MEM_READ   = 0x02,
    OP_MEM_COPY   = 0x03,
    OP_DCR_WRITE  = 0x04,
    OP_DCR_READ   = 0x05,
    OP_LAUNCH     = 0x06,
    OP_FENCE      = 0x07,
    OP_EVT_SIG    = 0x08,
    OP_EVT_WAIT   = 0x09,
};

// On-wire byte size per opcode + profile flag (must mirror
// cmd_size_bytes() in VX_cp_pkg.sv).
static unsigned cmd_size(uint8_t op, bool profiled) {
    unsigned base = 4;
    switch (op) {
        case OP_NOP:        base = 4;  break;
        case OP_LAUNCH:     base = 12; break;
        case OP_FENCE:      base = 8;  break;
        case OP_DCR_WRITE:
        case OP_DCR_READ:
        case OP_EVT_SIG:    base = 20; break;
        case OP_EVT_WAIT:
        case OP_MEM_WRITE:
        case OP_MEM_READ:
        case OP_MEM_COPY:   base = 28; break;
        default:            base = 4;  break;
    }
    return base + (profiled ? 8 : 0);
}

// Emit one command into byte buffer `cl` starting at `off`; return new
// offset. Only the bytes the opcode actually carries (per cmd_size_bytes)
// are written; bytes that fall into the next-command region are left as
// they were (typically zero from a prior memset), so the walker doesn't
// see spurious headers leaking out of one command's arg field into the
// next slot.
static unsigned emit_cmd(uint8_t* cl, unsigned off,
                         uint8_t opcode, uint8_t flags,
                         uint64_t arg0, uint64_t arg1, uint64_t arg2,
                         uint64_t profile_slot) {
    bool profiled = (flags & (1u << F_PROFILE)) != 0;
    unsigned sz = cmd_size(opcode, profiled);
    unsigned data_bytes = sz - 4 - (profiled ? 8 : 0);  // arg payload size
    // Header: opcode, flags, reserved=0.
    cl[off + 0] = opcode;
    cl[off + 1] = flags;
    cl[off + 2] = 0;
    cl[off + 3] = 0;
    // Concatenate arg0/arg1/arg2 little-endian, truncated to data_bytes.
    uint64_t args[3] = { arg0, arg1, arg2 };
    for (unsigned i = 0; i < data_bytes; ++i) {
        unsigned w = i / 8;
        unsigned b = i % 8;
        cl[off + 4 + i] = (uint8_t)(args[w] >> (8 * b));
    }
    if (profiled) {
        // profile_slot lives at the tail (offset + sz - 8).
        for (int i = 0; i < 8; ++i)
            cl[off + sz - 8 + i] = (uint8_t)(profile_slot >> (8*i));
    }
    return off + sz;
}

// Decoded cmd_t over the packed bus exposed by the wrapper. The single
// decoded record occupies bits [287:0]:
//   hdr [287:256], arg0 [255:192], arg1 [191:128], arg2 [127:64],
//   profile_slot [63:0].
struct DecodedCmd {
    uint8_t  opcode;
    uint8_t  flags;
    uint64_t arg0;
    uint64_t arg1;
    uint64_t arg2;
    uint64_t profile_slot;
};

// Read a `bits` bit field starting at bit `start` from the packed cmd bus.
template <typename T>
static uint64_t read_bits(T* top, uint64_t start, uint32_t bits) {
    uint64_t v = 0;
    for (uint32_t i = 0; i < bits; ++i) {
        uint64_t b = start + i;
        uint64_t word = b / 32;
        uint64_t shift = b % 32;
        uint64_t bit = (top->cmd_packed[word] >> shift) & 1u;
        v |= (bit << i);
    }
    return v;
}

template <typename T>
static DecodedCmd decode_cmd(T* top) {
    DecodedCmd c;
    uint64_t hdr = read_bits(top, 256, 32);
    c.opcode = (uint8_t)(hdr & 0xff);
    c.flags  = (uint8_t)((hdr >> 8) & 0xff);
    c.arg0   = read_bits(top, 192, 64);
    c.arg1   = read_bits(top, 128, 64);
    c.arg2   = read_bits(top, 64,  64);
    c.profile_slot = read_bits(top, 0, 64);
    return c;
}

// Load the 64-byte cache line into cl_data (512 bits, LSB-first: cl[0] = [7:0]).
template <typename T>
static void set_line(T* top, const uint8_t* cl) {
    constexpr int N_WORDS = CL_BYTES / 4;
    for (int w = 0; w < N_WORDS; ++w) {
        top->cl_data[w] = (uint32_t)cl[w*4]
                        | ((uint32_t)cl[w*4 + 1] << 8)
                        | ((uint32_t)cl[w*4 + 2] << 16)
                        | ((uint32_t)cl[w*4 + 3] << 24);
    }
}

// Emulate the fetch FSM over the combinational unpacker: decode at offset 0,
// advance by cmd_size until the DUT deasserts has_cmd (end-of-line / sentinel
// / malformed tail). Returns every command walked off the line, in order.
template <typename T>
static std::vector<DecodedCmd> walk_line(T* top, const uint8_t* cl) {
    set_line(top, cl);
    std::vector<DecodedCmd> out;
    unsigned off = 0;
    while (off + 4 <= (unsigned)CL_BYTES) {
        top->offset = (uint8_t)off;
        top->eval();
        if (!top->has_cmd)
            break;
        out.push_back(decode_cmd(top));
        unsigned sz = (unsigned)top->cmd_size;
        // A valid command must advance the walk by at least its 4 B header,
        // otherwise the loop can't make progress.
        EXPECT(sz >= 4, "walk: has_cmd asserted with cmd_size < 4");
        off += sz;
    }
    return out;
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    vl_simulator<VVX_cp_unpack_top> sim;
    auto* top = sim.operator->();
    top->clk = 0;
    top->reset = 0;

    uint8_t cl[CL_BYTES];

    // ----- Test 1: all-zero line → 0 cmds -----
    std::memset(cl, 0, CL_BYTES);
    {
        auto cmds = walk_line(top, cl);
        EXPECT(cmds.size() == 0, "T1: empty line should yield 0 cmds");
    }

    // ----- Test 2: single CMD_LAUNCH unprofiled (12 B; carries arg0 only) -----
    std::memset(cl, 0, CL_BYTES);
    emit_cmd(cl, 0, OP_LAUNCH, 0,
             /*arg0=*/0x80000000ull, /*arg1 unused=*/0, 0, 0);
    {
        auto cmds = walk_line(top, cl);
        EXPECT(cmds.size() == 1, "T2: single LAUNCH should yield 1 cmd");
        EXPECT(cmds[0].opcode == OP_LAUNCH,     "T2: opcode mismatch");
        EXPECT(cmds[0].flags  == 0,             "T2: flags mismatch");
        EXPECT(cmds[0].arg0   == 0x80000000ull, "T2: arg0 mismatch");
    }

    // ----- Test 3: single CMD_LAUNCH profiled (20 B; arg0 + profile_slot) -----
    std::memset(cl, 0, CL_BYTES);
    emit_cmd(cl, 0, OP_LAUNCH, (1u << F_PROFILE),
             /*arg0=*/0xC0DEull, /*arg1 unused=*/0, 0,
             /*profile_slot=*/0xCAFEBABEull);
    {
        auto cmds = walk_line(top, cl);
        EXPECT(cmds.size() == 1, "T3: profiled LAUNCH count");
        EXPECT(cmds[0].opcode == OP_LAUNCH, "T3: opcode mismatch");
        EXPECT(cmds[0].flags  == 1,         "T3: F_PROFILE flag");
        EXPECT(cmds[0].arg0   == 0xC0DEull, "T3: arg0");
        EXPECT(cmds[0].profile_slot == 0xCAFEBABEull, "T3: profile_slot");
    }

    // ----- Test 4: DCR_WRITE (20 B) + MEM_COPY (28 B) = 48 B -----
    std::memset(cl, 0, CL_BYTES);
    {
        unsigned off = 0;
        off = emit_cmd(cl, off, OP_DCR_WRITE, 0,
                       /*arg0=addr=*/0x123ull, /*arg1=value=*/0xDEADBEEFull, 0, 0);
        off = emit_cmd(cl, off, OP_MEM_COPY, 0,
                       /*arg0=dst=*/0xAA00ull, /*arg1=src=*/0xBB00ull,
                       /*arg2=size=*/0x1000ull, 0);
        EXPECT(off == 48, "T4: emit offset accounting");
    }
    {
        auto cmds = walk_line(top, cl);
        EXPECT(cmds.size() == 2, "T4: 2 cmds expected");
        EXPECT(cmds[0].opcode == OP_DCR_WRITE,   "T4 c0 op");
        EXPECT(cmds[0].arg0   == 0x123ull,       "T4 c0 arg0");
        EXPECT(cmds[0].arg1   == 0xDEADBEEFull,  "T4 c0 arg1");
        EXPECT(cmds[1].opcode == OP_MEM_COPY,    "T4 c1 op");
        EXPECT(cmds[1].arg0   == 0xAA00ull,      "T4 c1 arg0");
        EXPECT(cmds[1].arg1   == 0xBB00ull,      "T4 c1 arg1");
        EXPECT(cmds[1].arg2   == 0x1000ull,      "T4 c1 arg2");
    }

    // ----- Test 5: 3 × profiled NOP (12 B each) = 36 B + pad -----
    std::memset(cl, 0, CL_BYTES);
    {
        unsigned off = 0;
        for (int i = 0; i < 3; ++i) {
            off = emit_cmd(cl, off, OP_NOP, (1u << F_PROFILE),
                           /*arg0=*/0, 0, 0,
                           /*profile_slot=*/0xFEEDFACE00ull + i);
        }
    }
    {
        auto cmds = walk_line(top, cl);
        EXPECT(cmds.size() == 3, "T5: 3 NOP+F_PROFILE expected");
        for (int i = 0; i < 3; ++i) {
            EXPECT(cmds[i].opcode == OP_NOP, "T5: NOP opcode");
            EXPECT(cmds[i].flags  == 1,      "T5: F_PROFILE flag");
            EXPECT(cmds[i].profile_slot == 0xFEEDFACE00ull + (uint64_t)i,
                   "T5: profile_slot per-cmd");
        }
    }

    // ----- Test 6: malformed tail — 3 MEM_COPYs (28 B each) = 84 B,
    //       too big for a 64 B line. After 2 cmds at offset 56, the next
    //       cmd would need bytes 56..83 → walker must stop at 2. -----
    std::memset(cl, 0, CL_BYTES);
    {
        unsigned off = 0;
        off = emit_cmd(cl, off, OP_MEM_COPY, 0, 0x10, 0x20, 0x30, 0);
        off = emit_cmd(cl, off, OP_MEM_COPY, 0, 0x40, 0x50, 0x60, 0);
        EXPECT(off == 56, "T6: first 2 MEM_COPYs land at 56 B");
        // Plant a bogus header at byte 56 that claims to be MEM_COPY (28 B)
        // — walker must reject because 56 + 28 = 84 > 64.
        cl[56] = OP_MEM_COPY;
    }
    {
        auto cmds = walk_line(top, cl);
        EXPECT(cmds.size() == 2,
               "T6: malformed-tail rule should keep the walk at 2 cmds");
    }

    // ----- Test 7: 5 × profiled NOP (12 B each) = 60 B + 4 B pad -----
    std::memset(cl, 0, CL_BYTES);
    {
        unsigned off = 0;
        for (int i = 0; i < 5; ++i) {
            off = emit_cmd(cl, off, OP_NOP, (1u << F_PROFILE),
                           0, 0, 0, 0xABCDull + i);
        }
    }
    {
        auto cmds = walk_line(top, cl);
        EXPECT(cmds.size() == 5, "T7: walker should yield all 5 cmds");
        for (int i = 0; i < 5; ++i) {
            EXPECT(cmds[i].profile_slot == 0xABCDull + (uint64_t)i,
                   "T7: per-cmd profile_slot mismatch");
        }
    }

    std::printf("PASSED — 7 scenarios\n");
    return 0;
}
