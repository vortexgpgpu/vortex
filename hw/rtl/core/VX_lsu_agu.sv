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
// (swz_buf). The interleave is a bit permutation, so no arithmetic follows the
// adder but the window compare.

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
    // stored as {word ^ (group * SKEW), thread, byte}, so one frame slot across a
    // warp's threads is contiguous. The XOR keeps equal frame slots of different
    // warps out of the same cache set, since a group spans a power of two; SKEW
    // steps it by whole dcache lines when a slot row is narrower than one. The map
    // depends on the address alone, so any thread may dereference another
    // thread's stack pointer.
    //
    // The window is aligned to a whole group, so the group index is the
    // address's own bits above the group and the map is a bit permutation:
    // no arithmetic follows the adder but the window compare.
    if (`VX_CFG_NUM_THREADS > 1) begin : g_stack_interleave
        localparam WORD_BITS   = `CLOG2(`VX_CFG_XLEN / 8);
        localparam THREAD_BITS = `CLOG2(`VX_CFG_NUM_THREADS);
        localparam STACK_BITS  = `VX_MEM_STACK_LOG2_SIZE;
        localparam SLOT_BITS   = STACK_BITS - WORD_BITS;
        localparam GROUP_BITS  = STACK_BITS + THREAD_BITS;
        localparam WIN_BITS    = `VX_CFG_XLEN - STACK_BITS;
        localparam ROW_BITS    = THREAD_BITS + WORD_BITS;
        localparam LINE_BITS   = `CLOG2(`VX_CFG_DCACHE_LINE_SIZE);
        localparam SKEW_BITS   = (LINE_BITS > ROW_BITS) ? (LINE_BITS - ROW_BITS) : 0;

        `STATIC_ASSERT(`IS_POW2(`VX_CFG_NUM_THREADS), ("invalid parameter: NUM_THREADS=%0d", `VX_CFG_NUM_THREADS))
        `STATIC_ASSERT(`VX_CFG_FLEN <= `VX_CFG_XLEN, ("invalid parameter: stack accesses must fit a word"))
        `STATIC_ASSERT(STACK_WINDOW_SPAN <= STACK_WINDOW_TOP, ("invalid parameter: stack window underflows"))
        `STATIC_ASSERT(GROUP_BITS + SLOT_BITS <= `VX_CFG_XLEN, ("invalid parameter: NUM_THREADS=%0d", `VX_CFG_NUM_THREADS))
        // The span is a whole number of groups, so a group-aligned top makes
        // the bottom group-aligned too.
        `STATIC_ASSERT((STACK_WINDOW_TOP & ((`VX_CFG_XLEN'(1) << GROUP_BITS) - 1)) == '0,
                       ("stack window top is not aligned to a group of NUM_THREADS stacks"))

        wire [WIN_BITS-1:0] lin_win = lin_addr[`VX_CFG_XLEN-1:STACK_BITS];
        wire in_stack = (lin_win >= WIN_BITS'(STACK_WINDOW_BOTTOM >> STACK_BITS))
                     && (lin_win <  WIN_BITS'(STACK_WINDOW_TOP >> STACK_BITS));

        wire [SLOT_BITS-1:0] stack_slot = lin_addr[STACK_BITS-1:WORD_BITS]
                                        ^ (lin_addr[GROUP_BITS +: SLOT_BITS] << SKEW_BITS);

        wire [`VX_CFG_XLEN-1:0] stack_addr = {
            lin_addr[`VX_CFG_XLEN-1:GROUP_BITS],
            stack_slot,
            lin_addr[GROUP_BITS-1:STACK_BITS],
            lin_addr[WORD_BITS-1:0]
        };

        assign addr = in_stack ? stack_addr : lin_addr;
    end else begin : g_stack_linear
        assign addr = lin_addr;
    end
`else
    assign addr = lin_addr;
`endif

endmodule
