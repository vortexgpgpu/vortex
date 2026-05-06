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

module VX_tex_dcr import VX_gpu_pkg::*, VX_tex_pkg::*; #(
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

    // Decode write strobe + extract addr/data from the new req-style DCR bus
    wire write_valid                    = dcr_bus_if.req_valid && dcr_bus_if.req_data.rw;
    wire [VX_DCR_ADDR_WIDTH-1:0] write_addr = dcr_bus_if.req_data.addr;
    wire [VX_DCR_DATA_WIDTH-1:0] write_data = dcr_bus_if.req_data.data;
    `UNUSED_VAR (write_data[31])

    // Tie off response side (DCRs are write-only here)
    assign dcr_bus_if.rsp_valid = 1'b0;
    assign dcr_bus_if.rsp_data  = '0;

    // DCR registers

    reg [`CLOG2(NUM_STAGES)-1:0] dcr_stage;
    tex_dcrs_t dcrs [NUM_STAGES-1:0];
    tex_dcrs_t dcrs_n;

    // DCRs write

    always @(*) begin
        dcrs_n = dcrs[dcr_stage];
        if (write_valid) begin
            case (write_addr)
                `VX_DCR_TEX_ADDR: begin
                    dcrs_n.baseaddr = write_data[`TEX_ADDR_BITS-1:0];
                end
                `VX_DCR_TEX_FORMAT: begin
                    dcrs_n.format = write_data[`TEX_FORMAT_BITS-1:0];
                end
                `VX_DCR_TEX_FILTER: begin
                    dcrs_n.filter = write_data[`TEX_FILTER_BITS-1:0];
                end
                `VX_DCR_TEX_WRAP: begin
                    dcrs_n.wraps[0] = write_data[0  +: `TEX_WRAP_BITS];
                    dcrs_n.wraps[1] = write_data[16 +: `TEX_WRAP_BITS];
                end
                `VX_DCR_TEX_LOGDIM: begin
                    dcrs_n.logdims[0] = write_data[0  +: `VX_TEX_LOD_BITS];
                    dcrs_n.logdims[1] = write_data[16 +: `VX_TEX_LOD_BITS];
                end
                default: begin
                    for (integer j = 0; j <= `VX_TEX_LOD_MAX; ++j) begin
                        if (write_addr == `VX_DCR_TEX_MIPOFF(VX_DCR_ADDR_WIDTH'(j))) begin
                            dcrs_n.mipoff[j] = write_data[`TEX_MIPOFF_BITS-1:0];
                        end
                    end
                end
            endcase
        end
    end

    always @(posedge clk) begin
        if (write_valid && write_addr == `VX_DCR_TEX_STAGE) begin
            dcr_stage <= write_data[`CLOG2(NUM_STAGES)-1:0];
        end
    end

    always @(posedge clk) begin
        if (write_valid) begin
            dcrs[dcr_stage] <= dcrs_n;
        end
    end
    assign tex_dcrs = dcrs[stage];

`ifdef DBG_TRACE_TEX
    always @(posedge clk) begin
        if (write_valid) begin
            `TRACE(1, ("%d: %s-tex-dcr: state=", $time, INSTANCE_ID))
            `TRACE_TEX_DCR(1, write_addr)
            `TRACE(1, (", data=0x%0h\n", write_data))
        end
    end
`endif

endmodule
