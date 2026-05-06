//!/bin/bash

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

`include "VX_tex_define.vh"

module VX_tex_csr import VX_tex_pkg::*; #(
    parameter CORE_ID = 0,
    parameter NUM_LANES = 1
) (
    input wire clk,
    input wire reset,

    // Inputs
    VX_sfu_csr_if.slave tex_csr_if,

    // Output
    output tex_csrs_t tex_csrs
);
    `UNUSED_PARAM (CORE_ID)
    `UNUSED_PARAM (NUM_LANES)

    // CSR registers

    tex_csrs_t reg_csrs;

    // CSRs write

    always @(posedge clk) begin
        if (reset) begin
            reg_csrs <= '0;
        end else if (tex_csr_if.write_enable) begin
            case (tex_csr_if.write_addr)
                // TODO
                default:;
            endcase
        end
    end

    assign tex_csr_if.read_data = '0;

    assign tex_csrs = reg_csrs;

    `UNUSED_VAR (tex_csr_if.read_enable)
    `UNUSED_VAR (tex_csr_if.read_addr)
    `UNUSED_VAR (tex_csr_if.read_uuid)
    `UNUSED_VAR (tex_csr_if.read_wid)
    `UNUSED_VAR (tex_csr_if.read_tmask)
    `UNUSED_VAR (tex_csr_if.read_pid)

    `UNUSED_VAR (tex_csr_if.write_uuid)
    `UNUSED_VAR (tex_csr_if.write_wid)
    `UNUSED_VAR (tex_csr_if.write_tmask)
    `UNUSED_VAR (tex_csr_if.write_pid)
    `UNUSED_VAR (tex_csr_if.write_data)

`ifdef DBG_TRACE_TEX
    always @(posedge clk) begin
        if (tex_csr_if.write_enable) begin
            `TRACE(1, ("%d: core%0d-tex-csr-write: wid=%0d, tmask=%b, state=", $time, CORE_ID, tex_csr_if.write_wid, tex_csr_if.write_tmask))
            `TRACE_TEX_CSR(1, tex_csr_if.write_addr)
            `TRACE(1, (", data="))
            `TRACE_ARRAY1D(1, "0x%0h", tex_csr_if.write_data, NUM_LANES)
            `TRACE(1, (" (#%0d)\n", tex_csr_if.write_uuid))
        end
    end
`endif

endmodule
