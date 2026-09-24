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

`include "VX_define.vh"

// VX_lsu_agu: per-lane LSU address generator. Owns all LSU address forms
// so VX_lsu_slice contains no address arithmetic:
//   - plain load/store/fence: addr = base + sext(offset)
//   - pack-load uop:          addr = base + uop_idx * stride
//     (uop_idx = offset[1:0], stride = rs2; a 2-bit shift-and-add
//      collapsed with a 3:2 compressor — no multiplier)
// Both forms are base + addend, so the form select sits ahead of the compressor
// and a single carry-propagate adder serves both. Addresses in the per-thread
// stack window are then interleaved by word across threads.
//
// This module sits between the LSU lane dispatch registers and the memory
// scheduler's request queue, so every full-width operation here is reduced to
// the narrowest width that is provably equivalent; see g_stack_interleave.

module VX_lsu_agu import VX_gpu_pkg::*; (
    input  wire [`VX_CFG_XLEN-1:0] base,    // rs1
    input  wire [`VX_CFG_XLEN-1:0] stride,  // rs2 (pack stride)
    input  wire [11:0]             offset,  // immediate offset; [1:0] = pack uop_idx
    input  wire [1:0]              pack,    // 0 = plain, non-zero = pack-load
    output wire [`VX_CFG_XLEN-1:0] addr
);
    wire is_pack = (pack != 2'b00);
    wire [1:0] uop_idx = offset[1:0];

    // Selecting the addends ahead of the compressor lets both forms share one
    // carry-propagate adder.
    wire [`VX_CFG_XLEN-1:0] addend_b = is_pack ? ({`VX_CFG_XLEN{uop_idx[0]}} & stride)
                                               : `SEXT(`VX_CFG_XLEN, offset);
    wire [`VX_CFG_XLEN-1:0] addend_c = is_pack ? ({`VX_CFG_XLEN{uop_idx[1]}} & (stride << 1))
                                               : '0;

    wire [`VX_CFG_XLEN+1:0] csa_sum, csa_carry;
    VX_csa_32 #(
        .N (`VX_CFG_XLEN)
    ) agu_csa (
        .a     (base),
        .b     (addend_b),
        .c     (addend_c),
        .sum   (csa_sum),
        .carry (csa_carry)
    );
    wire [`VX_CFG_XLEN-1:0] lin_addr = csa_sum[`VX_CFG_XLEN-1:0] + csa_carry[`VX_CFG_XLEN-1:0];
    `UNUSED_VAR ({csa_sum[`VX_CFG_XLEN+1:`VX_CFG_XLEN], csa_carry[`VX_CFG_XLEN+1:`VX_CFG_XLEN]})

`ifdef VX_CFG_LSU_STACK_INTERLEAVE_ENABLE
    // Thread t's stack is the STACK_SIZE bytes below STACK_TOP - t*STACK_SIZE.
    // Within each group of NUM_THREADS stacks, offset {thread, word, byte} is
    // stored as {word ^ group, thread, byte}, so one frame slot across a warp's
    // threads is contiguous. The XOR keeps equal frame slots of different warps
    // out of the same cache set, since a group spans a power of two. The map
    // depends on the address alone, so any thread may dereference another
    // thread's stack pointer.
    //
    // Written naively the map costs three dependent full-width operations after
    // lin_addr: subtract the window base, permute, add the base back. It does
    // not need them. The permutation moves bits only WITHIN the group field --
    // the group index occupies the same position on both sides -- so subtracting
    // the base and adding it back cancel on the high bits, and the window base
    // has ALIGN_BITS trailing zeros, so the low field needs arithmetic only on
    // the CORR_BITS above them. What survives of two XLEN-wide operations is one
    // CORR_BITS-wide subtract, one CORR_BITS-wide add, and the +-1 the two
    // disagree by.
    if (`VX_CFG_NUM_THREADS > 1) begin : g_stack_interleave
        localparam WORD_BITS   = `CLOG2(`VX_CFG_XLEN / 8);
        localparam THREAD_BITS = `CLOG2(`VX_CFG_NUM_THREADS);
        localparam STACK_BITS  = `VX_MEM_STACK_LOG2_SIZE;
        localparam SLOT_BITS   = STACK_BITS - WORD_BITS;
        localparam GROUP_BITS  = STACK_BITS + THREAD_BITS;
        localparam HI_BITS     = `VX_CFG_XLEN - GROUP_BITS;
        // Trailing zeros of the window base.
        localparam [`VX_CFG_XLEN-1:0] BOT_LSB = STACK_WINDOW_BOTTOM & -STACK_WINDOW_BOTTOM;
        localparam ALIGN_BITS  = (BOT_LSB == '0) ? `VX_CFG_XLEN : `CLOG2(BOT_LSB);
        localparam CORR_BITS   = (GROUP_BITS > ALIGN_BITS) ? (GROUP_BITS - ALIGN_BITS) : 0;
        localparam WIN_BITS    = `VX_CFG_XLEN - STACK_BITS;

        `STATIC_ASSERT(`IS_POW2(`VX_CFG_NUM_THREADS), ("invalid parameter: NUM_THREADS=%0d", `VX_CFG_NUM_THREADS))
        `STATIC_ASSERT(`VX_CFG_FLEN <= `VX_CFG_XLEN, ("invalid parameter: stack accesses must fit a word"))
        `STATIC_ASSERT(STACK_WINDOW_SPAN <= STACK_WINDOW_TOP, ("invalid parameter: stack window underflows"))
        // The window test compares only the bits above the stack size, so a
        // memory-map edit that breaks either bound's alignment must fail here.
        `STATIC_ASSERT((STACK_WINDOW_BOTTOM & ((`VX_CFG_XLEN'(1) << STACK_BITS) - 1)) == '0,
                       ("stack window base is not a multiple of the stack size"))
        `STATIC_ASSERT((STACK_WINDOW_TOP & ((`VX_CFG_XLEN'(1) << STACK_BITS) - 1)) == '0,
                       ("stack window top is not a multiple of the stack size"))

        localparam [GROUP_BITS-1:0] BOT_LO = STACK_WINDOW_BOTTOM[GROUP_BITS-1:0];
        localparam [HI_BITS-1:0]    BOT_HI = STACK_WINDOW_BOTTOM[`VX_CFG_XLEN-1:GROUP_BITS];

        // Both bounds are multiples of the stack size, so membership is decided
        // by the bits above it -- a WIN_BITS compare, not a full-width one.
        wire [WIN_BITS-1:0] lin_win = lin_addr[`VX_CFG_XLEN-1:STACK_BITS];
        wire in_stack = (lin_win >= WIN_BITS'(STACK_WINDOW_BOTTOM >> STACK_BITS))
                     && (lin_win <  WIN_BITS'(STACK_WINDOW_TOP >> STACK_BITS));

        wire [GROUP_BITS-1:0] lin_lo = lin_addr[GROUP_BITS-1:0];
        wire [HI_BITS-1:0]    lin_hi = lin_addr[`VX_CFG_XLEN-1:GROUP_BITS];

        // stack_off = lin_addr - STACK_WINDOW_BOTTOM, group field only.
        wire [GROUP_BITS-1:0] off_lo;
        wire                  off_borrow;
        if (CORR_BITS == 0) begin : g_off_aligned
            `UNUSED_PARAM (BOT_LO)
            assign off_lo     = lin_lo;
            assign off_borrow = 1'b0;
        end else begin : g_off_corr
            wire [CORR_BITS:0] diff = {1'b0, lin_lo[GROUP_BITS-1:ALIGN_BITS]}
                                    - {1'b0, BOT_LO[GROUP_BITS-1:ALIGN_BITS]};
            assign off_lo     = {diff[CORR_BITS-1:0], lin_lo[ALIGN_BITS-1:0]};
            assign off_borrow = diff[CORR_BITS];
        end

        // Only SLOT_BITS of the group index reach the XOR, and truncation
        // commutes with the subtract, so it is computed at that width.
        wire [SLOT_BITS-1:0] stack_group;
        if (SLOT_BITS <= HI_BITS) begin : g_group_narrow
            assign stack_group = lin_hi[SLOT_BITS-1:0] - BOT_HI[SLOT_BITS-1:0]
                               - SLOT_BITS'(off_borrow);
        end else begin : g_group_wide
            assign stack_group = SLOT_BITS'(lin_hi - BOT_HI - HI_BITS'(off_borrow));
        end

        wire [SLOT_BITS-1:0] stack_slot = off_lo[STACK_BITS-1:WORD_BITS] ^ stack_group;

        wire [GROUP_BITS-1:0] swz_lo = {
            stack_slot,
            off_lo[GROUP_BITS-1:STACK_BITS],
            off_lo[WORD_BITS-1:0]
        };

        // addr = STACK_WINDOW_BOTTOM + swz_off, group field only.
        wire [GROUP_BITS-1:0] addr_lo;
        wire                  addr_carry;
        if (CORR_BITS == 0) begin : g_out_aligned
            assign addr_lo    = swz_lo;
            assign addr_carry = 1'b0;
        end else begin : g_out_corr
            wire [CORR_BITS:0] sum = {1'b0, swz_lo[GROUP_BITS-1:ALIGN_BITS]}
                                   + {1'b0, BOT_LO[GROUP_BITS-1:ALIGN_BITS]};
            assign addr_lo    = {sum[CORR_BITS-1:0], swz_lo[ALIGN_BITS-1:0]};
            assign addr_carry = sum[CORR_BITS];
        end

        // The window base cancels on the high bits; only the disagreement
        // between the two group-field corrections survives.
        wire [HI_BITS-1:0] addr_hi = lin_hi - HI_BITS'(off_borrow) + HI_BITS'(addr_carry);

        assign addr = in_stack ? {addr_hi, addr_lo} : lin_addr;
    end else begin : g_stack_linear
        assign addr = lin_addr;
    end
`else
    assign addr = lin_addr;
`endif

endmodule
