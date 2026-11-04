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
// so VX_lsu_block contains no address arithmetic:
//   - plain load/store/fence: addr = base + sext(offset)
//   - pack-load uop:          addr = base + uop_idx * stride
//     (uop_idx = offset[1:0], stride = rs2; a 2-bit shift-and-add
//      collapsed with a 3:2 compressor — no multiplier)
// Both forms are base + addend, so the form select sits ahead of the compressor
// and a single carry-propagate adder serves both. Addresses in the per-thread
// stack window are then interleaved by word across threads.
//
// The two halves take a cycle each: VX_lsu_agu adds ahead of the LSU block's
// address register (agu_buf), VX_lsu_agu_swz then tests the stack window and
// maps the registered address into the interleaved stack ahead of the next
// (swz_buf). Every full-width operation is reduced to the
// narrowest width that is provably equivalent; see g_stack_interleave.

module VX_lsu_agu import VX_gpu_pkg::*; (
    input  wire [`VX_CFG_XLEN-1:0] base,    // rs1
    input  wire [`VX_CFG_XLEN-1:0] stride,  // rs2 (pack stride)
    input  wire [11:0]             offset,  // immediate offset; [1:0] = pack uop_idx
    input  wire [1:0]              pack,    // 0 = plain, non-zero = pack-load
    // the address before the stack interleave, which maps the stack window
    // onto itself: any window disjoint from it classifies both alike
    output wire [`VX_CFG_XLEN-1:0] lin_addr
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
    `UNUSED_VAR ({csa_sum[`VX_CFG_XLEN+1:`VX_CFG_XLEN], csa_carry[`VX_CFG_XLEN+1:`VX_CFG_XLEN]})

    // At XLEN 64 the add is carry-select: bits [15:0] add directly, [31:16]
    // and [63:32] both ways, and the carries select.
    if (`VX_CFG_XLEN > 32) begin : g_add_csel
        localparam HW = `VX_CFG_XLEN - 32;
        wire [16:0]   lo   = {1'b0, csa_sum[15:0]} + {1'b0, csa_carry[15:0]};
        // the +1 enters as a carry-in below the LSB, so each +1 sum is one add
        wire [16:0]   md0  = {1'b0, csa_sum[31:16]} + {1'b0, csa_carry[31:16]};
        wire [17:0]   md1c = {1'b0, csa_sum[31:16], 1'b1} + {1'b0, csa_carry[31:16], 1'b1};
        wire [HW-1:0] hi0  = csa_sum[`VX_CFG_XLEN-1:32] + csa_carry[`VX_CFG_XLEN-1:32];
        wire [HW:0]   hi1c = {csa_sum[`VX_CFG_XLEN-1:32], 1'b1} + {csa_carry[`VX_CFG_XLEN-1:32], 1'b1};
        wire [16:0]   md1  = md1c[17:1];
        wire [HW-1:0] hi1  = hi1c[HW:1];
        `UNUSED_VAR ({md1c[0], hi1c[0]})
        wire [16:0] md = lo[16] ? md1 : md0;   // md[16]: the carry into bit 32
        assign lin_addr = {(md[16] ? hi1 : hi0), md[15:0], lo[15:0]};
    end else begin : g_add
        assign lin_addr = csa_sum[`VX_CFG_XLEN-1:0] + csa_carry[`VX_CFG_XLEN-1:0];
    end

endmodule

// VX_lsu_agu_swz: the stack interleave of a registered VX_lsu_agu address.
module VX_lsu_agu_swz import VX_gpu_pkg::*; (
    input  wire [`VX_CFG_XLEN-1:0] lin_addr,
    output wire [`VX_CFG_XLEN-1:0] addr
);
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

        `STATIC_ASSERT(`IS_POW2(`VX_CFG_NUM_THREADS), ("invalid parameter: NUM_THREADS=%0d", `VX_CFG_NUM_THREADS))
        `STATIC_ASSERT(`VX_CFG_FLEN <= `VX_CFG_XLEN, ("invalid parameter: stack accesses must fit a word"))
        `STATIC_ASSERT(STACK_WINDOW_SPAN <= STACK_WINDOW_TOP, ("invalid parameter: stack window underflows"))
        localparam WIN_BITS    = `VX_CFG_XLEN - STACK_BITS;
        // The window test compares only the bits above the stack size, so a
        // memory-map edit that breaks either bound's alignment must fail here.
        `STATIC_ASSERT((STACK_WINDOW_BOTTOM & ((`VX_CFG_XLEN'(1) << STACK_BITS) - 1)) == '0,
                       ("stack window base is not a multiple of the stack size"))
        `STATIC_ASSERT((STACK_WINDOW_TOP & ((`VX_CFG_XLEN'(1) << STACK_BITS) - 1)) == '0,
                       ("stack window top is not a multiple of the stack size"))
        // Both bounds are multiples of the stack size, so membership is decided
        // by the bits above it -- a WIN_BITS test, not a full-width one. The
        // window's first and last stacks agree on every bit above their lowest
        // WIN_LO differing ones, so an address is in it exactly when it has
        // that prefix and its low WIN_LO bits lie between theirs: an equality
        // and a WIN_LO-bit range test instead of two WIN_BITS-wide compares.
        localparam [WIN_BITS-1:0] WIN_FIRST = WIN_BITS'(STACK_WINDOW_BOTTOM >> STACK_BITS);
        localparam [WIN_BITS-1:0] WIN_LAST  = WIN_BITS'(STACK_WINDOW_TOP >> STACK_BITS) - WIN_BITS'(1);
        localparam WIN_LO = $clog2({1'b0, WIN_FIRST ^ WIN_LAST} + (WIN_BITS+1)'(1));
        `STATIC_ASSERT(STACK_WINDOW_SPAN != '0, ("invalid parameter: empty stack window"))
        wire [WIN_BITS-1:0] lin_win = lin_addr[`VX_CFG_XLEN-1:STACK_BITS];
        wire in_stack;
        if (WIN_LO == 0) begin : g_win_one
            assign in_stack = (lin_win == WIN_FIRST);
        end else if (WIN_LO >= WIN_BITS) begin : g_win_full
            assign in_stack = (lin_win >= WIN_FIRST) && (lin_win <= WIN_LAST);
        end else begin : g_win_prefix
            wire [WIN_LO-1:0] win_lo = lin_win[WIN_LO-1:0];
            assign in_stack = (lin_win[WIN_BITS-1:WIN_LO] == WIN_FIRST[WIN_BITS-1:WIN_LO])
                           && (win_lo >= WIN_FIRST[WIN_LO-1:0])
                           && (win_lo <= WIN_LAST[WIN_LO-1:0]);
        end

        localparam [GROUP_BITS-1:0] BOT_LO = STACK_WINDOW_BOTTOM[GROUP_BITS-1:0];
        localparam [HI_BITS-1:0]    BOT_HI = STACK_WINDOW_BOTTOM[`VX_CFG_XLEN-1:GROUP_BITS];

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
        // commutes with the subtract, so it is computed at that width. The
        // borrow settles late and only moves the group index by one, so the
        // group index and everything built on it -- the slot, the swizzled
        // field and the correction's carry -- are formed for both borrows and
        // the borrow selects at the end.
        wire [1:0][GROUP_BITS-1:0] addr_lo_b;
        wire [1:0]                 addr_carry_b;
        for (genvar b = 0; b < 2; ++b) begin : g_borrow
            wire [SLOT_BITS-1:0] stack_group;
            if (SLOT_BITS <= HI_BITS) begin : g_group_narrow
                assign stack_group = lin_hi[SLOT_BITS-1:0] - (BOT_HI[SLOT_BITS-1:0] + SLOT_BITS'(b));
            end else begin : g_group_wide
                assign stack_group = SLOT_BITS'(lin_hi - BOT_HI - HI_BITS'(b));
            end

            wire [SLOT_BITS-1:0] stack_slot = off_lo[STACK_BITS-1:WORD_BITS] ^ stack_group;

            wire [GROUP_BITS-1:0] swz_lo = {
                stack_slot,
                off_lo[GROUP_BITS-1:STACK_BITS],
                off_lo[WORD_BITS-1:0]
            };

            // addr = STACK_WINDOW_BOTTOM + swz_off, group field only.
            if (CORR_BITS == 0) begin : g_out_aligned
                assign addr_lo_b[b]    = swz_lo;
                assign addr_carry_b[b] = 1'b0;
            end else begin : g_out_corr
                wire [CORR_BITS:0] sum = {1'b0, swz_lo[GROUP_BITS-1:ALIGN_BITS]}
                                       + {1'b0, BOT_LO[GROUP_BITS-1:ALIGN_BITS]};
                assign addr_lo_b[b]    = {sum[CORR_BITS-1:0], swz_lo[ALIGN_BITS-1:0]};
                assign addr_carry_b[b] = sum[CORR_BITS];
            end
        end
        wire [GROUP_BITS-1:0] addr_lo = addr_lo_b[off_borrow];

        // Only the two group-field corrections survive on the high bits. They
        // settle last and differ by at most one, so they select a precomputed neighbour.
        // The neighbours are written as prefix reductions rather than adds:
        // bit i flips when every bit below it is one (+1) or zero (-1).
        wire [HI_BITS-1:0] lin_hi_dec, lin_hi_inc;
        for (genvar k = 0; k < HI_BITS; ++k) begin : g_hi_step
            if (k == 0) begin : g_lsb
                assign lin_hi_inc[k] = ~lin_hi[k];
                assign lin_hi_dec[k] = ~lin_hi[k];
            end else begin : g_bit
                assign lin_hi_inc[k] = lin_hi[k] ^ (&  lin_hi[k-1:0]);
                assign lin_hi_dec[k] = lin_hi[k] ^ (~| lin_hi[k-1:0]);
            end
        end
        // One select per high bit, its conditions formed beside it: the window
        // test and the two corrections pick among the address and its neighbours.
        wire hi_dec = in_stack && off_borrow && ~addr_carry_b[1];
        wire hi_inc = in_stack && ~off_borrow && addr_carry_b[0];
        wire [HI_BITS-1:0] addr_hi = hi_dec ? lin_hi_dec : (hi_inc ? lin_hi_inc : lin_hi);

        assign addr = {addr_hi, (in_stack ? addr_lo : lin_lo)};
    end else begin : g_stack_linear
        assign addr = lin_addr;
    end
`else
    assign addr = lin_addr;
`endif

endmodule
