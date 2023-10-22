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

module VX_tex_dcr import VX_tex_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter NUM_STAGES = 1
) (
    input wire clk,
    input wire reset,

    // Inputs
    VX_dcr_bus_if.slave                 dcr_bus_if,

    // Output
    input wire [`VX_TEX_STAGE_BITS-1:0] stage,
    output tex_dcrs_t                   tex_dcrs
);
    `UNUSED_SPARAM (INSTANCE_ID)    
    `UNUSED_VAR (reset)

    // DCR registers

    reg [`CLOG2(NUM_STAGES)-1:0] dcr_stage;
    tex_dcrs_t dcrs [NUM_STAGES-1:0];

    // DCRs write

    always @(posedge clk) begin
        if (dcr_bus_if.write_valid) begin
            case (dcr_bus_if.write_addr)
                `VX_DCR_TEX_STAGE: begin 
                    dcr_stage <= dcr_bus_if.write_data[`CLOG2(NUM_STAGES)-1:0];
                end
                `VX_DCR_TEX_ADDR: begin 
                    dcrs[dcr_stage].baseaddr <= dcr_bus_if.write_data[`TEX_ADDR_BITS-1:0];
                end
                `VX_DCR_TEX_FORMAT: begin 
                    dcrs[dcr_stage].format <= dcr_bus_if.write_data[`TEX_FORMAT_BITS-1:0];
                end
                `VX_DCR_TEX_FILTER: begin 
                    dcrs[dcr_stage].filter <= dcr_bus_if.write_data[`TEX_FILTER_BITS-1:0];
                end
                `VX_DCR_TEX_WRAP: begin
                    dcrs[dcr_stage].wraps[0] <= dcr_bus_if.write_data[0  +: `TEX_WRAP_BITS];
                    dcrs[dcr_stage].wraps[1] <= dcr_bus_if.write_data[16 +: `TEX_WRAP_BITS];
                end
                `VX_DCR_TEX_LOGDIM: begin 
                    dcrs[dcr_stage].logdims[0] <= dcr_bus_if.write_data[0  +: `VX_TEX_LOD_BITS];
                    dcrs[dcr_stage].logdims[1] <= dcr_bus_if.write_data[16 +: `VX_TEX_LOD_BITS];
                end
                default: begin
                    for (integer j = 0; j <= `VX_TEX_LOD_MAX; ++j) begin
                    `IGNORE_WARNINGS_BEGIN
                        if (dcr_bus_if.write_addr == `VX_DCR_TEX_MIPOFF(j)) begin
                    `IGNORE_WARNINGS_END
                            dcrs[dcr_stage].mipoff[j] <= dcr_bus_if.write_data[`TEX_MIPOFF_BITS-1:0];
                        end
                    end
                end
            endcase
        end
    end

    // DCRs read
    assign tex_dcrs = dcrs[stage];

`ifdef DBG_TRACE_TEX
    always @(posedge clk) begin
        if (dcr_bus_if.write_valid) begin
            `TRACE(1, ("%d: %s-tex-dcr: stage=%0d, state=", $time, INSTANCE_ID, dcr_stage));
            `TRACE_TEX_DCR(1, dcr_bus_if.write_addr);
            `TRACE(1, (", data=0x%0h\n", dcr_bus_if.write_data));
        end
    end
`endif

endmodule
