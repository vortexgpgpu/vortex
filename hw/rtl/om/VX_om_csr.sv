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

`include "VX_om_define.vh"

module VX_om_csr import VX_om_pkg::*; #(
    parameter CORE_ID = 0,
    parameter NUM_LANES = 1
) (
    input wire clk,
    input wire reset,

    // Inputs
    VX_sfu_csr_if.slave om_csr_if,

    // Output
    output om_csrs_t om_csrs
);
    `UNUSED_PARAM (CORE_ID)
    `UNUSED_PARAM (NUM_LANES)

    // CSR registers

    om_csrs_t reg_csrs;

    // CSRs write

    always @(posedge clk) begin
        if (reset) begin
            reg_csrs <= '0;
        end else if (om_csr_if.write_enable) begin
            case (om_csr_if.write_addr)
                `VX_CSR_OM_RT_IDX:;
                `VX_CSR_OM_SAMPLE_IDX:;
                default:;
            endcase
        end
    end

    assign om_csr_if.read_data = '0;

    assign om_csrs = reg_csrs;

    `UNUSED_VAR (om_csr_if.read_enable)
    `UNUSED_VAR (om_csr_if.read_addr)
    `UNUSED_VAR (om_csr_if.read_uuid)
    `UNUSED_VAR (om_csr_if.read_wid)
    `UNUSED_VAR (om_csr_if.read_tmask)
    `UNUSED_VAR (om_csr_if.read_pid)

    `UNUSED_VAR (om_csr_if.write_data)
    `UNUSED_VAR (om_csr_if.write_uuid)
    `UNUSED_VAR (om_csr_if.write_wid)
    `UNUSED_VAR (om_csr_if.write_pid)
    `UNUSED_VAR (om_csr_if.write_tmask)

`ifdef DBG_TRACE_OM
    always @(posedge clk) begin
        if (om_csr_if.write_enable) begin
            `TRACE(1, ("%d: core%0d-om-csr-write: wid=%0d, tmask=%b, state=", $time, CORE_ID, om_csr_if.write_wid, om_csr_if.write_tmask))
            `TRACE_OM_CSR(1, om_csr_if.write_addr)
            `TRACE(1, (", data="))
            `TRACE_ARRAY1D(1, "0x%0h", om_csr_if.write_data, NUM_LANES)
            `TRACE(1, (" (#%0d)\n", om_csr_if.write_uuid))
        end
    end
`endif

endmodule
