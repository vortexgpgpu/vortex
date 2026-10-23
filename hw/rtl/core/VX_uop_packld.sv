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

// VX_uop_packld: Micro-op sequencer for pack-load instructions, and for the
// register-pair atomics of a 32-bit hart (AMO64). Both are LSU rewrites with
// exclusive selects (pack != 0 vs a .D atomic), so they share one sequencer
// slot rather than paying for a second ibuffer_t input on its output mux.
//
// Expands one vx_packlb_f / vx_packlh_f instruction into N sequential LSU uops:
//   pack==1 (PACKLB) → 4 uops of INST_LSU_LBU (unsigned byte loads)
//   pack==2 (PACKLH) → 2 uops of INST_LSU_LHU (unsigned halfword loads)
//
// Each uop encodes its index in op_args.lsu.offset[1:0].
// The dispatcher reads pack (non-zero) to identify pack-load uops and computes:
//   eff_rs1[lane] = rs1[lane] + rs2[lane] * uop_idx   (shift-and-add)
// The pack field is cleared in op_args forwarded to the LSU slice so it sees a
// plain LBU or LHU.  The NaN-boxing of the FLW result is handled automatically
// because is_float==1 and rd is a float register (triggers nan-box in lsu_slice).

module VX_uop_packld import VX_gpu_pkg::*; (
    input clk,
    input reset,

    input  ibuffer_t ibuf_in,
    output ibuffer_t ibuf_out,

    input wire start,
    input wire advance,
    input wire [UOP_CTR_W-1:0] uop_idx,
    output wire [UOP_CTR_W-1:0] uop_count
);
    `UNUSED_VAR ({clk, reset, start, advance, uop_idx})

    // pack==1 → PACKLB: 4 byte-load uops;
    // pack==2 → PACKLH: 2 halfword-load uops
    wire is_packlh = (ibuf_in.op_args.lsu.pack == 2'b10);
    wire [UOP_CTR_W-1:0] pack_count = is_packlh ? UOP_CTR_W'(2) : UOP_CTR_W'(4);

    wire [1:0] idx2 = uop_idx[1:0];
    wire [XLENB_W-1:0] byte_size = is_packlh ? '1 : '0;
    wire [XLENB_W-1:0] byte_off  = is_packlh ? XLENB_W'({idx2[0], 1'b0}) : XLENB_W'(idx2);

`ifdef VX_CFG_EXT_AMO64_ENABLE
    // Register-pair atomic. It reads rs1, the rs2 pair and the rd pair
    // (compare-and-swap comparand), and writes the rd pair: more than one uop's
    // three source slots and one destination. It issues as a locked burst:
    //   CMP : reads rd, rd+1 as rs2/rs3; the LSU holds them as the comparand
    //   AMO : reads rs1, rs2, rs2+1 as rs3; writes rd (old low word)
    //   HI  : writes rd+1; reserves it until the LSU retires the atomic's old
    //         high word into it
    // CMP leads every pair atomic, not only compare-and-swap: reading rd+1 makes
    // the burst wait for an older write of rd+1 to land, which AMO's own
    // register check (rd only) does not cover and which would otherwise race
    // the atomic's high word into rd+1. The LSU holds one comparand per issue
    // slot and operand collector, so the lock keeps another warp of the slot
    // from issuing between CMP and AMO. x0 as a pair base names x0 for both
    // halves.
    wire is_pair  = ibuf_in.op_args.lsu.amo_valid;
    localparam [UOP_CTR_W-1:0] PAIR_COUNT = UOP_CTR_W'(3);
    // 0 = CMP, 1 = AMO, 2 = HI
    wire [1:0] pair_role = uop_idx[1:0];

    wire [RV_REGS_BITS-1:0] rd_idx  = get_reg_idx(ibuf_in.rd);
    wire [RV_REGS_BITS-1:0] rs2_idx = get_reg_idx(ibuf_in.rs2);
    wire [NUM_REGS_BITS-1:0] rd_hi  = (rd_idx != 0)  ? make_reg_num(REG_TYPE_I, rd_idx | RV_REGS_BITS'(1))  : '0;
    wire [NUM_REGS_BITS-1:0] rs2_hi = (rs2_idx != 0) ? make_reg_num(REG_TYPE_I, rs2_idx | RV_REGS_BITS'(1)) : '0;

    assign uop_count = is_pair ? PAIR_COUNT : pack_count;
`else
    assign uop_count = pack_count;
`endif

    ibuffer_t ibuf_r;
    always_comb begin
        ibuf_r = ibuf_in;
    `ifdef VX_CFG_EXT_AMO64_ENABLE
        if (is_pair) begin
            case (pair_role)
            2'd0: begin
                ibuf_r.op_args.lsu.amo_valid = 1'b0;
                ibuf_r.op_args.lsu.amo_pair  = 1'b1;
                ibuf_r.op_args.lsu.is_store  = 1'b1;
                ibuf_r.rs2     = ibuf_in.rd;
                ibuf_r.rs3     = rd_hi;
                ibuf_r.used_rs = 3'b110;
                ibuf_r.wb      = 1'b0;
                ibuf_r.rd      = '0;
            end
            2'd1: begin
                ibuf_r.rs3     = rs2_hi;
                ibuf_r.used_rs = 3'b111;
            end
            default: begin
                ibuf_r.op_args.lsu.amo_valid = 1'b0;
                ibuf_r.op_args.lsu.amo_pair  = 1'b1;
                ibuf_r.op_args.lsu.is_store  = 1'b0;
                ibuf_r.used_rs = 3'b000;
                ibuf_r.rd      = rd_hi;
            end
            endcase
            ibuf_r.fu_lock   = (uop_idx == '0);
            ibuf_r.fu_unlock = (uop_idx == (PAIR_COUNT - UOP_CTR_W'(1)));
        end else
    `endif
        begin
            ibuf_r.op_args.lsu.offset = {10'd0, idx2};
            ibuf_r.bytesel = {byte_size, byte_off};
        end
    end
    assign ibuf_out = ibuf_r;

endmodule
