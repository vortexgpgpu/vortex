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

module VX_tcu_uops import
`ifdef EXT_TCU_ENABLE
    VX_tcu_pkg::*,
`endif
    VX_gpu_pkg::*; (
    input clk,
    input reset,

    input  ibuffer_t ibuf_in,
    output ibuffer_t ibuf_out,
    input  wire      start,
    input  wire      next,
    output reg       done
);
    localparam CTR_W = $clog2(TCU_UOPS);

    localparam LG_N = $clog2(TCU_N_STEPS);
    localparam LG_M = $clog2(TCU_M_STEPS);
    localparam LG_K = $clog2(TCU_K_STEPS);

    localparam LG_A_SB = $clog2(TCU_A_SUB_BLOCKS);
    localparam LG_B_SB = $clog2(TCU_B_SUB_BLOCKS);

    // uop counter
    reg [CTR_W-1:0] counter;

    logic [`UP(LG_N)-1:0] n_index;
    logic [`UP(LG_M)-1:0] m_index;
    logic [`UP(LG_K)-1:0] k_index;

    if (LG_N != 0) begin : g_n_idx
        assign n_index = counter[0 +: LG_N];
    end else begin : g_n_idx0
        assign n_index = 0;
    end

    if (LG_M != 0) begin : g_m_idx
        assign m_index = counter[LG_N +: LG_M];
    end else begin : g_m_idx0
        assign m_index = 0;
    end

    if (LG_K != 0) begin : g_k_idx
        assign k_index = counter[LG_N + LG_M +: LG_K];
    end else begin : g_k_idx0
        assign k_index = 0;
    end

    // Register offsets
    wire [CTR_W-1:0] rs1_offset = ((CTR_W'(m_index) >> LG_A_SB) << LG_K) | CTR_W'(k_index);
    wire [CTR_W-1:0] rs2_offset = ((CTR_W'(k_index) << LG_N) | CTR_W'(n_index)) >> LG_B_SB;
    wire [CTR_W-1:0] rs3_offset = (CTR_W'(m_index) << LG_N) | CTR_W'(n_index);

    // Register calculations
    wire [4:0] rs1 = TCU_RA + 5'(rs1_offset);
    wire [4:0] rs2 = TCU_RB + 5'(rs2_offset);
    wire [4:0] rs3 = TCU_RC + 5'(rs3_offset);

`ifdef UUID_ENABLE
    wire [31:0] uuid_lo = {counter, ibuf_in.uuid[0 +: (32-CTR_W)]};
    wire [UUID_WIDTH-1:0] uuid = {ibuf_in.uuid[UUID_WIDTH-1:32], uuid_lo};
`else
    wire [UUID_WIDTH-1:0] uuid = ibuf_in.uuid;
`endif

    // Output uop generation
    assign ibuf_out.uuid      = uuid;
    assign ibuf_out.tmask     = ibuf_in.tmask;
    assign ibuf_out.PC        = ibuf_in.PC;
    assign ibuf_out.ex_type   = ibuf_in.ex_type;
    assign ibuf_out.op_type   = ibuf_in.op_type;
    assign ibuf_out.op_args.tcu.fmt_s = ibuf_in.op_args.tcu.fmt_s;
    assign ibuf_out.op_args.tcu.fmt_d = ibuf_in.op_args.tcu.fmt_d;
    assign ibuf_out.op_args.tcu.step_m = 4'(m_index);
    assign ibuf_out.op_args.tcu.step_n = 4'(n_index);
    assign ibuf_out.wb        = 1;
    assign ibuf_out.used_rs   = ibuf_in.used_rs;
    assign ibuf_out.rs1       = make_reg_num(REG_TYPE_F, rs1);
    assign ibuf_out.rs2       = make_reg_num(REG_TYPE_F, rs2);
    assign ibuf_out.rs3       = make_reg_num(REG_TYPE_F, rs3);
    assign ibuf_out.rd        = make_reg_num(REG_TYPE_F, rs3);
    `UNUSED_VAR (ibuf_in.wb)
    `UNUSED_VAR (ibuf_in.rd)
    `UNUSED_VAR (ibuf_in.rs1)
    `UNUSED_VAR (ibuf_in.rs2)
    `UNUSED_VAR (ibuf_in.rs3)

    reg busy;

    always_ff @(posedge clk) begin
        if (reset) begin
            counter <= 0;
            busy    <= 0;
            done    <= 0;
        end else begin
            if (~busy && start) begin
                busy <= 1;
                done <= (TCU_UOPS == 1);
            end else if (busy && next) begin
                counter <= counter + ((TCU_UOPS > 1) ? 1 : 0);
                done <= (counter == CTR_W'(TCU_UOPS-2));
                busy <= ~done;
            end
        end
    end

endmodule
