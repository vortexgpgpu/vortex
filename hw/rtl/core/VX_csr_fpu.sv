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
`include "VX_fpu_define.vh"

// Per-warp FP CSRs (frm + fflags) for one residue class of warps modulo WID_STRIDE,
// stored by wid / WID_STRIDE; every request presented belongs to that class.
// CSR port c serves the warps with (wid / WID_STRIDE) mod NUM_CSR_PORTS == c and
// FPU port f those with (wid / WID_STRIDE) mod NUM_FPU_PORTS == f, so each warp's
// state is written by exactly one port of each kind.
module VX_csr_fpu import VX_gpu_pkg::*, VX_fpu_pkg::*; #(
    parameter WID_STRIDE    = 1,
    parameter NUM_CSR_PORTS = 1,
    parameter NUM_FPU_PORTS = 1
) (
    input wire clk,
    input wire reset,

    input wire [NUM_CSR_PORTS-1:0][NW_WIDTH-1:0]                     csr_wid,
    output wire [NUM_CSR_PORTS-1:0][INST_FRM_BITS+`FP_FLAGS_BITS-1:0] csr_rd_data,
    input wire [NUM_CSR_PORTS-1:0][2:0]                              csr_wr_sel,  // {fcsr, frm, fflags}
    input wire [NUM_CSR_PORTS-1:0][INST_FRM_BITS+`FP_FLAGS_BITS-1:0] csr_wr_data,

    VX_fpu_csr_if.slave     fpu_csr_if [NUM_FPU_PORTS]
);
    localparam FCSR_W      = INST_FRM_BITS + `FP_FLAGS_BITS;
    localparam NUM_ENTRIES = `VX_CFG_NUM_WARPS / WID_STRIDE;
    localparam STRIDE_BITS = `CLOG2(WID_STRIDE);
    localparam ENTRY_W     = `LOG2UP(NUM_ENTRIES);

    `STATIC_ASSERT (`IS_POW2(WID_STRIDE), ("invalid parameter"))

    function automatic logic [ENTRY_W-1:0] entry_of(input logic [NW_WIDTH-1:0] wid);
        entry_of = ENTRY_W'(wid >> STRIDE_BITS);
    endfunction

    wire [NUM_FPU_PORTS-1:0]               fpu_wr_enable;
    wire [NUM_FPU_PORTS-1:0][NW_WIDTH-1:0] fpu_wr_wid;
    fflags_t [NUM_FPU_PORTS-1:0]           fpu_wr_fflags;
    for (genvar f = 0; f < NUM_FPU_PORTS; ++f) begin : g_fpu_ports
        assign fpu_wr_enable[f] = fpu_csr_if[f].write_enable;
        assign fpu_wr_wid[f]    = fpu_csr_if[f].write_wid;
        assign fpu_wr_fflags[f] = fpu_csr_if[f].write_fflags;
    end

    wire [NUM_ENTRIES-1:0][FCSR_W-1:0] fcsr;

    for (genvar l = 0; l < NUM_ENTRIES; ++l) begin : g_entries
        localparam C = l % NUM_CSR_PORTS;
        localparam F = l % NUM_FPU_PORTS;
        localparam [ENTRY_W-1:0] ENTRY = ENTRY_W'(l);

        reg [FCSR_W-1:0] fcsr_r, fcsr_n;

        always @(*) begin
            fcsr_n = fcsr_r;
            if (fpu_wr_enable[F] && (entry_of(fpu_wr_wid[F]) == ENTRY)) begin
                fcsr_n[`FP_FLAGS_BITS-1:0] = fcsr_r[`FP_FLAGS_BITS-1:0] | fpu_wr_fflags[F];
            end
            if (entry_of(csr_wid[C]) == ENTRY) begin
                if (csr_wr_sel[C][0]) begin
                    fcsr_n[`FP_FLAGS_BITS-1:0] = csr_wr_data[C][`FP_FLAGS_BITS-1:0];
                end
                if (csr_wr_sel[C][1]) begin
                    fcsr_n[FCSR_W-1:`FP_FLAGS_BITS] = csr_wr_data[C][INST_FRM_BITS-1:0];
                end
                if (csr_wr_sel[C][2]) begin
                    fcsr_n = csr_wr_data[C];
                end
            end
        end

        always @(posedge clk) begin
            if (reset) begin
                fcsr_r <= '0;
            end else begin
                fcsr_r <= fcsr_n;
            end
        end

        assign fcsr[l] = fcsr_r;
    end

    for (genvar c = 0; c < NUM_CSR_PORTS; ++c) begin : g_csr_rd
        assign csr_rd_data[c] = fcsr[entry_of(csr_wid[c])];
    end

    for (genvar f = 0; f < NUM_FPU_PORTS; ++f) begin : g_fpu_rd
        assign fpu_csr_if[f].read_frm = fcsr[entry_of(fpu_csr_if[f].read_wid)][FCSR_W-1:`FP_FLAGS_BITS];
    end

endmodule
