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

`include "VX_fpu_define.vh"

`ifdef FPU_TYPE_DSP

`ifdef SIMULATION
`include "dpi_float.vh"
`endif

module VX_fpu_ncp import VX_gpu_pkg::*, VX_fpu_pkg::*; #(
    parameter NUM_LANES = 1,
    parameter NUM_PES   = `UP(NUM_LANES / `FNCP_PE_RATIO),
    parameter TAG_WIDTH = 1
) (
    input wire clk,
    input wire reset,

    output wire ready_in,
    input wire  valid_in,

    input wire [NUM_LANES-1:0] mask_in,

    input wire [TAG_WIDTH-1:0] tag_in,

    input wire [INST_FPU_BITS-1:0] op_type,
    input wire [INST_FMT_BITS-1:0] fmt,
    input wire [INST_FRM_BITS-1:0] frm,

    input wire [NUM_LANES-1:0][`XLEN-1:0]  dataa,
    input wire [NUM_LANES-1:0][`XLEN-1:0]  datab,
    input wire [NUM_LANES-1:0][`XLEN-1:0]  datac,
    output wire [NUM_LANES-1:0][`XLEN-1:0] result,

    output wire has_fflags,
    output wire [`FP_FLAGS_BITS-1:0] fflags,

    output wire [TAG_WIDTH-1:0] tag_out,

    input wire  ready_out,
    output wire valid_out
);
    localparam DATAW = 2 * `XLEN + INST_FRM_BITS + INST_FPU_BITS + INST_FMT_BITS;

    `UNUSED_VAR (datac)

    wire [NUM_LANES-1:0][DATAW-1:0] data_in;

    wire [NUM_LANES-1:0] mask_out;
    wire [NUM_LANES-1:0][(`FP_FLAGS_BITS+`XLEN)-1:0] data_out;
    fflags_t [NUM_LANES-1:0] fflags_out;

    wire pe_enable;
    wire [NUM_PES-1:0] pe_mask_out;
    `UNUSED_VAR (pe_mask_out)
    wire [NUM_PES-1:0][DATAW-1:0] pe_data_in;
    wire [NUM_PES-1:0][(`FP_FLAGS_BITS+`XLEN)-1:0] pe_data_out;

    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_data_in
        assign data_in[i][0                                              +: `XLEN]         = dataa[i];
        assign data_in[i][`XLEN                                          +: `XLEN]         = datab[i];
        assign data_in[i][2*`XLEN                                        +: INST_FRM_BITS]  = frm;
        assign data_in[i][2*`XLEN + INST_FRM_BITS                        +: INST_FPU_BITS]  = op_type;
        assign data_in[i][2*`XLEN + INST_FRM_BITS + INST_FPU_BITS        +: INST_FMT_BITS]  = fmt;
    end

    VX_pe_serializer #(
        .NUM_LANES  (NUM_LANES),
        .NUM_PES    (NUM_PES),
        .LATENCY    (`LATENCY_FNCP),
        .DATA_IN_WIDTH (DATAW),
        .DATA_OUT_WIDTH (`FP_FLAGS_BITS + `XLEN),
        .TAG_WIDTH  (TAG_WIDTH),
        .PE_REG     (0),
        .OUT_BUF    (2)
    ) pe_serializer (
        .clk        (clk),
        .reset      (reset),
        .valid_in   (valid_in),
        .mask_in    (mask_in),
        .data_in    (data_in),
        .tag_in     (tag_in),
        .ready_in   (ready_in),
        .pe_enable  (pe_enable),
        .pe_mask_out(pe_mask_out),
        .pe_data_out(pe_data_in),
        .pe_data_in (pe_data_out),
        .valid_out  (valid_out),
        .mask_out   (mask_out),
        .data_out   (data_out),
        .tag_out    (tag_out),
        .ready_out  (ready_out)
    );

    `UNUSED_VAR (pe_data_in)

    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_result
        assign result[i]     = data_out[i][0 +: `XLEN];
        assign fflags_out[i] = data_out[i][`XLEN +: `FP_FLAGS_BITS];
    end

`ifdef QUARTUS

    // F32-only hardware path using VX_fncp_unit
    for (genvar i = 0; i < NUM_PES; ++i) begin : g_fncp_units
        VX_fncp_unit #(
            .LATENCY (`LATENCY_FNCP),
            .OUT_REG (1)
        ) fncp_unit (
            .clk        (clk),
            .reset      (reset),
            .enable     (pe_enable),
            .frm        (pe_data_in[0][2*`XLEN +: INST_FRM_BITS]),
            .op_type    (pe_data_in[0][2*`XLEN + INST_FRM_BITS +: INST_FPU_BITS]),
            .dataa      (pe_data_in[i][0 +: 32]),
            .datab      (pe_data_in[i][`XLEN +: 32]),
            .result     (pe_data_out[i][0 +: 32]),
            .fflags     (pe_data_out[i][`XLEN +: `FP_FLAGS_BITS])
        );
    end

`elsif VIVADO

    for (genvar i = 0; i < NUM_PES; ++i) begin : g_fncp_units
        VX_fncp_unit #(
            .LATENCY (`LATENCY_FNCP),
            .OUT_REG (1)
        ) fncp_unit (
            .clk        (clk),
            .reset      (reset),
            .enable     (pe_enable),
            .frm        (pe_data_in[0][2*`XLEN +: INST_FRM_BITS]),
            .op_type    (pe_data_in[0][2*`XLEN + INST_FRM_BITS +: INST_FPU_BITS]),
            .dataa      (pe_data_in[i][0 +: 32]),
            .datab      (pe_data_in[i][`XLEN +: 32]),
            .result     (pe_data_out[i][0 +: 32]),
            .fflags     (pe_data_out[i][`XLEN +: `FP_FLAGS_BITS])
        );
    end

`else

    // Simulation path: use DPI functions for F32 and F64
    for (genvar i = 0; i < NUM_PES; ++i) begin : g_fncp_units
        wire [INST_FRM_BITS-1:0]  frm_pe     = pe_data_in[0][2*`XLEN +: INST_FRM_BITS];
        wire [INST_FPU_BITS-1:0]  op_type_pe = pe_data_in[0][2*`XLEN + INST_FRM_BITS +: INST_FPU_BITS];
        wire                      f_fmt_pe   = pe_data_in[0][2*`XLEN + INST_FRM_BITS + INST_FPU_BITS]; // fmt[0]
        wire                      is_fcmp_pe = (op_type_pe == INST_FPU_CMP);

        reg [`XLEN-1:0] res_ncp;
        fflags_t        f_ncp;

        reg [63:0] r_clss, r_fle, r_flt, r_feq, r_fmin, r_fmax;
        reg [63:0] r_fsgnj, r_fsgnjn, r_fsgnjx;
        `UNUSED_VAR (r_clss)
        `UNUSED_VAR (r_fle)
        `UNUSED_VAR (r_flt)
        `UNUSED_VAR (r_feq)
        `UNUSED_VAR (r_fmin)
        `UNUSED_VAR (r_fmax)
        `UNUSED_VAR (r_fsgnj)
        `UNUSED_VAR (r_fsgnjn)
        `UNUSED_VAR (r_fsgnjx)
        fflags_t   f_le, f_lt, f_eq, f_min, f_max;

        always @(*) begin
            dpi_fclss  (pe_enable, int'(f_fmt_pe),
                        64'(pe_data_in[i][0 +: `XLEN]),
                        r_clss);
            dpi_fle    (pe_enable, int'(f_fmt_pe),
                        64'(pe_data_in[i][0 +: `XLEN]),
                        64'(pe_data_in[i][`XLEN +: `XLEN]),
                        r_fle, f_le);
            dpi_flt    (pe_enable, int'(f_fmt_pe),
                        64'(pe_data_in[i][0 +: `XLEN]),
                        64'(pe_data_in[i][`XLEN +: `XLEN]),
                        r_flt, f_lt);
            dpi_feq    (pe_enable, int'(f_fmt_pe),
                        64'(pe_data_in[i][0 +: `XLEN]),
                        64'(pe_data_in[i][`XLEN +: `XLEN]),
                        r_feq, f_eq);
            dpi_fmin   (pe_enable, int'(f_fmt_pe),
                        64'(pe_data_in[i][0 +: `XLEN]),
                        64'(pe_data_in[i][`XLEN +: `XLEN]),
                        r_fmin, f_min);
            dpi_fmax   (pe_enable, int'(f_fmt_pe),
                        64'(pe_data_in[i][0 +: `XLEN]),
                        64'(pe_data_in[i][`XLEN +: `XLEN]),
                        r_fmax, f_max);
            dpi_fsgnj  (pe_enable, int'(f_fmt_pe),
                        64'(pe_data_in[i][0 +: `XLEN]),
                        64'(pe_data_in[i][`XLEN +: `XLEN]),
                        r_fsgnj);
            dpi_fsgnjn (pe_enable, int'(f_fmt_pe),
                        64'(pe_data_in[i][0 +: `XLEN]),
                        64'(pe_data_in[i][`XLEN +: `XLEN]),
                        r_fsgnjn);
            dpi_fsgnjx (pe_enable, int'(f_fmt_pe),
                        64'(pe_data_in[i][0 +: `XLEN]),
                        64'(pe_data_in[i][`XLEN +: `XLEN]),
                        r_fsgnjx);

            res_ncp = 'x;
            f_ncp   = '0;
            case (frm_pe)
                0: begin res_ncp = is_fcmp_pe ? `XLEN'(r_fle) : `XLEN'(r_fsgnj);  f_ncp = f_le; end
                1: begin res_ncp = is_fcmp_pe ? `XLEN'(r_flt) : `XLEN'(r_fsgnjn); f_ncp = f_lt; end
                2: begin res_ncp = is_fcmp_pe ? `XLEN'(r_feq) : `XLEN'(r_fsgnjx); f_ncp = f_eq; end
                3: begin res_ncp = `XLEN'(r_clss); end
                4: begin // FMV.X.W (F32) or FMV.X.D (F64) — float-to-int move
                    res_ncp = f_fmt_pe ? `XLEN'(pe_data_in[i][0 +: `XLEN])   // FMV.X.D: direct copy
                                       : `XLEN'($signed(pe_data_in[i][0 +: 32])); // FMV.X.W: sign-extend
                end
                5: begin // FMV.W.X (F32) or FMV.D.X (F64) — int-to-float move
                    res_ncp = f_fmt_pe ? pe_data_in[i][0 +: `XLEN]   // FMV.D.X: direct copy
                                       : `XLEN'(64'hffffffff00000000 | {32'b0, pe_data_in[i][0 +: 32]}); // FMV.W.X: NaN-box
                end
                6: begin res_ncp = `XLEN'(r_fmin); f_ncp = f_min; end
                7: begin res_ncp = `XLEN'(r_fmax); f_ncp = f_max; end
            endcase
        end

        VX_shift_register #(
            .DATAW (`XLEN + $bits(fflags_t)),
            .DEPTH (`LATENCY_FNCP)
        ) shift_req_dpi (
            .clk      (clk),
            `UNUSED_PIN (reset),
            .enable   (pe_enable),
            .data_in  ({f_ncp, res_ncp}),
            .data_out (pe_data_out[i])
        );
    end

`endif

    assign has_fflags = 1;

    `FPU_MERGE_FFLAGS(fflags, fflags_out, mask_out, NUM_LANES);

endmodule

`endif
