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

//
// RTU uop expander (ISA v2). Mirrors VX_tcu_uops: rewrites the fetched
// macro-op into a per-cycle stream of ordinary micro-ops, each naming its own
// source/destination registers so the standard operand collector reads/writes
// the right file (the f0..f7 ray window for TRACE2, the FP/GP hit window for
// GETWF/GETW). The RtuUnit consumes the stream and accumulates ray state in its
// regfile, mirroring the SimX RtuUopGen / process_trace2_uop / process_getw_uop.
//
//   TRACE2 (4 uops):
//     0 CFG    : rs1 = lane-packed config (GP); rd = handle (GP)
//     1 ORIGIN : rs1/rs2/rs3 = f0/f1/f2 -> origin slots
//     2 DIR    : rs1/rs2/rs3 = f3/f4/f5 -> direction slots
//     3 ARM    : rs1/rs2     = f6/f7     -> tmin/tmax; arm the walk
//   GETWF/GETW (count uops):
//     i        : rd = window base + i (FP/GP); slot = start + i (RTU regfile)
//
module VX_rtu_uops import VX_rtu_pkg::*, VX_gpu_pkg::*; (
    input clk,
    input reset,

    input  ibuffer_t ibuf_in,
    output ibuffer_t ibuf_out,

    input wire start,
    input wire advance,
    input wire [UOP_CTR_W-1:0] uop_idx,
    output wire [UOP_CTR_W-1:0] uop_count
);
    `UNUSED_VAR ({clk, reset, start, advance})

    wire [RTU_OP_BITS-1:0] op = ibuf_in.op_args.rtu.op;
    wire is_trace2 = (op == RTU_OP_TRACE2);

    // TRACE2 expands into exactly 4 uops; GETWF/GETW into `count` uops.
    assign uop_count = is_trace2 ? UOP_CTR_W'(4)
                                 : UOP_CTR_W'(ibuf_in.op_args.rtu.count);

    // Destination window base for GETWF/GETW (type bit + index from decode).
    wire [REG_TYPE_BITS-1:0] rd_type = get_reg_type(ibuf_in.rd);
    wire [RV_REGS_BITS-1:0]  rd_base = get_reg_idx(ibuf_in.rd);

    ibuffer_t ibuf_r;
    always_comb begin
        ibuf_r = ibuf_in;
        if (is_trace2) begin
            case (uop_idx[1:0])
            2'd0: begin // CFG: read rs1 config, write handle
                ibuf_r.op_args.rtu.uop = RTU_UOP_CFG;
                // rd = handle (GP), rs1 = lane-packed config (GP): keep as decoded.
                ibuf_r.used_rs[0] = 1'b1;
                ibuf_r.used_rs[1] = 1'b0;
                ibuf_r.used_rs[2] = 1'b0;
                // wb/rd left as decoded (handle writeback).
            end
            2'd1: begin // ORIGIN: f0,f1,f2 -> origin slots
                ibuf_r.op_args.rtu.uop  = RTU_UOP_ORIGIN;
                ibuf_r.op_args.rtu.slot = 5'(`VX_RT_RAY_ORIGIN);
                ibuf_r.rs1 = make_reg_num(REG_TYPE_F, RV_REGS_BITS'(0));
                ibuf_r.rs2 = make_reg_num(REG_TYPE_F, RV_REGS_BITS'(1));
                ibuf_r.rs3 = make_reg_num(REG_TYPE_F, RV_REGS_BITS'(2));
                ibuf_r.used_rs[0] = 1'b1;
                ibuf_r.used_rs[1] = 1'b1;
                ibuf_r.used_rs[2] = 1'b1;
                ibuf_r.wb = 1'b0;
                ibuf_r.rd = '0;
            end
            2'd2: begin // DIR: f3,f4,f5 -> direction slots
                ibuf_r.op_args.rtu.uop  = RTU_UOP_DIR;
                ibuf_r.op_args.rtu.slot = 5'(`VX_RT_RAY_DIRECTION);
                ibuf_r.rs1 = make_reg_num(REG_TYPE_F, RV_REGS_BITS'(3));
                ibuf_r.rs2 = make_reg_num(REG_TYPE_F, RV_REGS_BITS'(4));
                ibuf_r.rs3 = make_reg_num(REG_TYPE_F, RV_REGS_BITS'(5));
                ibuf_r.used_rs[0] = 1'b1;
                ibuf_r.used_rs[1] = 1'b1;
                ibuf_r.used_rs[2] = 1'b1;
                ibuf_r.wb = 1'b0;
                ibuf_r.rd = '0;
            end
            default: begin // ARM (uop 3): f6,f7 -> tmin/tmax, then arm the walk
                ibuf_r.op_args.rtu.uop  = RTU_UOP_ARM;
                ibuf_r.op_args.rtu.slot = 5'(`VX_RT_T_MIN);
                ibuf_r.rs1 = make_reg_num(REG_TYPE_F, RV_REGS_BITS'(6));
                ibuf_r.rs2 = make_reg_num(REG_TYPE_F, RV_REGS_BITS'(7));
                ibuf_r.used_rs[0] = 1'b1;
                ibuf_r.used_rs[1] = 1'b1;
                ibuf_r.used_rs[2] = 1'b0;
                ibuf_r.wb = 1'b0;
                ibuf_r.rd = '0;
            end
            endcase
        end else begin // GETWF / GETW: one uop per window element
            ibuf_r.op_args.rtu.uop  = uop_idx[2:0];
            ibuf_r.op_args.rtu.slot = ibuf_in.op_args.rtu.slot + 5'(uop_idx[4:0]);
            ibuf_r.rd = make_reg_num(rd_type, rd_base + RV_REGS_BITS'(uop_idx[4:0]));
            ibuf_r.wb = 1'b1;
            // rs1 = status word: only the first uop chains the scoreboard dep
            // (subsequent uops follow it in-order through the sequencer).
            ibuf_r.used_rs[0] = (uop_idx == '0);
            ibuf_r.used_rs[1] = 1'b0;
            ibuf_r.used_rs[2] = 1'b0;
        end
    end

    assign ibuf_out = ibuf_r;

endmodule
