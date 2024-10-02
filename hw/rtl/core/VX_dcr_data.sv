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

module VX_dcr_data import VX_gpu_pkg::*, VX_trace_pkg::*; (
    input wire              clk,
    input wire              reset,

    // Inputs
    VX_dcr_bus_if.slave     dcr_bus_if,

    // Outputs
    output base_dcrs_t      base_dcrs
);

    `UNUSED_VAR (reset)

    base_dcrs_t dcrs;

    always @(posedge clk) begin
       if (dcr_bus_if.write_valid) begin
            case (dcr_bus_if.write_addr)
            `VX_DCR_BASE_STARTUP_ADDR0 : dcrs.startup_addr[31:0] <= dcr_bus_if.write_data;
        `ifdef XLEN_64
            `VX_DCR_BASE_STARTUP_ADDR1 : dcrs.startup_addr[63:32] <= dcr_bus_if.write_data;
        `endif
            `VX_DCR_BASE_STARTUP_ARG0 : dcrs.startup_arg[31:0] <= dcr_bus_if.write_data;
        `ifdef XLEN_64
            `VX_DCR_BASE_STARTUP_ARG1 : dcrs.startup_arg[63:32] <= dcr_bus_if.write_data;
        `endif
            `VX_DCR_BASE_MPM_CLASS : dcrs.mpm_class <= dcr_bus_if.write_data[7:0];
            default:;
            endcase
        end
    end

    assign base_dcrs = dcrs;

`ifdef DBG_TRACE_PIPELINE
    always @(posedge clk) begin
        if (dcr_bus_if.write_valid) begin
            `TRACE(1, ("%d: base-dcr: state=", $time));
            trace_base_dcr(1, dcr_bus_if.write_addr);
            `TRACE(1, (", data=0x%h\n", dcr_bus_if.write_data));
        end
    end
`endif

endmodule
