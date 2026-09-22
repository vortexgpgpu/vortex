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
// Both forms are computed in parallel and selected by `pack`. Addresses in the
// per-thread stack window are then interleaved by word across threads.

module VX_lsu_agu import VX_gpu_pkg::*; (
    input  wire [`VX_CFG_XLEN-1:0] base,    // rs1
    input  wire [`VX_CFG_XLEN-1:0] stride,  // rs2 (pack stride)
    input  wire [11:0]             offset,  // immediate offset; [1:0] = pack uop_idx
    input  wire [1:0]              pack,    // 0 = plain, non-zero = pack-load
    output wire [`VX_CFG_XLEN-1:0] addr
);
    wire is_pack = (pack != 2'b00);

    // pack: base + uop_idx * stride  (uop_idx in offset[1:0])
    wire [1:0] uop_idx = offset[1:0];
    wire [`VX_CFG_XLEN-1:0] t0 = {`VX_CFG_XLEN{uop_idx[0]}} & stride;
    wire [`VX_CFG_XLEN-1:0] t1 = {`VX_CFG_XLEN{uop_idx[1]}} & (stride << 1);
    wire [`VX_CFG_XLEN+1:0] csa_sum, csa_carry;
    VX_csa_32 #(
        .N (`VX_CFG_XLEN)
    ) pack_csa (
        .a     (base),
        .b     (t0),
        .c     (t1),
        .sum   (csa_sum),
        .carry (csa_carry)
    );
    wire [`VX_CFG_XLEN-1:0] pack_addr = csa_sum[`VX_CFG_XLEN-1:0] + csa_carry[`VX_CFG_XLEN-1:0];
    `UNUSED_VAR ({csa_sum[`VX_CFG_XLEN+1:`VX_CFG_XLEN], csa_carry[`VX_CFG_XLEN+1:`VX_CFG_XLEN]})

    // plain: base + sext(offset)
    wire [`VX_CFG_XLEN-1:0] offset_addr = base + `SEXT(`VX_CFG_XLEN, offset);

    wire [`VX_CFG_XLEN-1:0] lin_addr = is_pack ? pack_addr : offset_addr;

`ifdef VX_CFG_LSU_STACK_INTERLEAVE_ENABLE
    // Thread t's stack is the STACK_SIZE bytes below STACK_TOP - t*STACK_SIZE.
    // Within each group of NUM_THREADS stacks, offset {thread, word, byte} is
    // stored as {word ^ group, thread, byte}, so one frame slot across a warp's
    // threads is contiguous. The XOR keeps equal frame slots of different warps
    // out of the same cache set, since a group spans a power of two. The map
    // depends on the address alone, so any thread may dereference another
    // thread's stack pointer.
    if (`VX_CFG_NUM_THREADS > 1) begin : g_stack_interleave
        localparam WORD_BITS   = `CLOG2(`VX_CFG_XLEN / 8);
        localparam THREAD_BITS = `CLOG2(`VX_CFG_NUM_THREADS);
        localparam STACK_BITS  = `VX_MEM_STACK_LOG2_SIZE;
        localparam SLOT_BITS   = STACK_BITS - WORD_BITS;
        localparam GROUP_BITS  = STACK_BITS + THREAD_BITS;
        `STATIC_ASSERT(`IS_POW2(`VX_CFG_NUM_THREADS), ("invalid parameter: NUM_THREADS=%0d", `VX_CFG_NUM_THREADS))
        `STATIC_ASSERT(`VX_CFG_FLEN <= `VX_CFG_XLEN, ("invalid parameter: stack accesses must fit a word"))
        `STATIC_ASSERT(STACK_WINDOW_SPAN <= STACK_WINDOW_TOP, ("invalid parameter: stack window underflows"))

        wire in_stack = (lin_addr >= STACK_WINDOW_BOTTOM) && (lin_addr < STACK_WINDOW_TOP);
        wire [`VX_CFG_XLEN-1:0] stack_off = lin_addr - STACK_WINDOW_BOTTOM;
        wire [`VX_CFG_XLEN-GROUP_BITS-1:0] stack_group = stack_off[`VX_CFG_XLEN-1:GROUP_BITS];
        wire [SLOT_BITS-1:0] stack_slot = stack_off[STACK_BITS-1:WORD_BITS] ^ SLOT_BITS'(stack_group);
        wire [`VX_CFG_XLEN-1:0] swz_off = {
            stack_group,
            stack_slot,
            stack_off[GROUP_BITS-1:STACK_BITS],
            stack_off[WORD_BITS-1:0]
        };
        assign addr = in_stack ? (STACK_WINDOW_BOTTOM + swz_off) : lin_addr;
    end else begin : g_stack_linear
        assign addr = lin_addr;
    end
`else
    assign addr = lin_addr;
`endif

endmodule
