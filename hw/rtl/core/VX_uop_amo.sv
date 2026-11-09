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

// VX_uop_amo: orders an atomic with the hart's other accesses.
//
// An AMO, LR or SC with .rl is preceded by a release fence (earlier writes
// become visible first) and one with .aq is followed by an acquire fence
// (later reads see what the atomic observed):
//   [release fence] atomic [acquire fence]
// The fences are plain LSU fence ops whose aq/rl arguments carry their kind;
// they read and write no register.

module VX_uop_amo import VX_gpu_pkg::*; (
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

    wire aq = ibuf_in.op_args.lsu.amo_aq;
    wire rl = ibuf_in.op_args.lsu.amo_rl;
    assign uop_count = UOP_CTR_W'(1) + UOP_CTR_W'(aq) + UOP_CTR_W'(rl);

    wire [UOP_CTR_W-1:0] amo_idx = UOP_CTR_W'(rl);
    wire is_rel = rl && (uop_idx == '0);
    wire is_acq = aq && (uop_idx == amo_idx + UOP_CTR_W'(1));

    ibuffer_t ibuf_r;
    always_comb begin
        ibuf_r = ibuf_in;
        if (is_rel || is_acq) begin
            ibuf_r.op_type  = INST_OP_BITS'(INST_LSU_FENCE);
            ibuf_r.op_args  = '0;
            ibuf_r.op_args.lsu.amo_aq = is_acq;
            ibuf_r.op_args.lsu.amo_rl = is_rel;
            ibuf_r.wb       = 1'b0;
            ibuf_r.rd_xregs = '0;
            ibuf_r.wr_xregs = '0;
            ibuf_r.used_rs  = '0;
        end
    end
    assign ibuf_out = ibuf_r;

endmodule
