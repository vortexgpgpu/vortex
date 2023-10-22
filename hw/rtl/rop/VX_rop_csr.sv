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

`include "VX_rop_define.vh"

module VX_rop_csr import VX_rop_pkg::*; #( 
    parameter CORE_ID = 0,
    parameter NUM_LANES = 1
) (
    input wire clk,
    input wire reset,

    // Inputs
    VX_sfu_csr_if.slave rop_csr_if,

    // Output
    output rop_csrs_t rop_csrs
);
    `UNUSED_PARAM (CORE_ID)
    `UNUSED_PARAM (NUM_LANES)

    // CSR registers

    rop_csrs_t reg_csrs;

    // CSRs write

    always @(posedge clk) begin
        if (reset) begin
            reg_csrs <= '0;
        end else if (rop_csr_if.write_enable) begin
            case (rop_csr_if.write_addr)
                `VX_CSR_ROP_RT_IDX:;
                `VX_CSR_ROP_SAMPLE_IDX:;
                default:;
            endcase
        end
    end

    assign rop_csr_if.read_data = '0;

    assign rop_csrs = reg_csrs;

    `UNUSED_VAR (rop_csr_if.read_enable)
    `UNUSED_VAR (rop_csr_if.read_addr)
    `UNUSED_VAR (rop_csr_if.read_uuid)
    `UNUSED_VAR (rop_csr_if.read_wid)
    `UNUSED_VAR (rop_csr_if.read_tmask)
    `UNUSED_VAR (rop_csr_if.read_pid)
    
    `UNUSED_VAR (rop_csr_if.write_data)    
    `UNUSED_VAR (rop_csr_if.write_uuid)
    `UNUSED_VAR (rop_csr_if.write_wid)
    `UNUSED_VAR (rop_csr_if.write_pid)
    `UNUSED_VAR (rop_csr_if.write_tmask)

`ifdef DBG_TRACE_ROP
    always @(posedge clk) begin
        if (rop_csr_if.write_enable) begin
            `TRACE(1, ("%d: core%0d-rop-csr-write: wid=%0d, tmask=%b, state=", $time, CORE_ID, rop_csr_if.write_wid, rop_csr_if.write_tmask));
            `TRACE_ROP_CSR(1, rop_csr_if.write_addr);
            `TRACE(1, (", data="));
            `TRACE_ARRAY1D(1, rop_csr_if.write_data, NUM_LANES);
            `TRACE(1, (" (#%0d)\n", rop_csr_if.write_uuid));
        end
    end
`endif

endmodule
