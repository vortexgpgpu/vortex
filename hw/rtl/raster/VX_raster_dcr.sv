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

`include "VX_raster_define.vh"

module VX_raster_dcr import VX_gpu_pkg::*, VX_raster_pkg::*; #(
    parameter `STRING INSTANCE_ID = ""
) (
    input wire clk,
    input wire reset,

    // Inputs
    VX_dcr_bus_if.slave     dcr_bus_if,

    // Output
    output raster_dcrs_t    raster_dcrs
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `UNUSED_VAR (reset)

    // Decode write strobe + extract addr/data from the new req-style DCR bus
    wire write_valid                    = dcr_bus_if.req_valid && dcr_bus_if.req_data.rw;
    wire [VX_DCR_ADDR_WIDTH-1:0] write_addr = dcr_bus_if.req_data.addr;
    wire [VX_DCR_DATA_WIDTH-1:0] write_data = dcr_bus_if.req_data.data;
    `UNUSED_VAR (write_data[31])

    // Tie off response side
    assign dcr_bus_if.rsp_valid = 1'b0;
    assign dcr_bus_if.rsp_data  = '0;

    // DCR registers
    raster_dcrs_t dcrs;

    // DCRs write
    always @(posedge clk) begin
        if (write_valid) begin
            case (write_addr)
                `VX_DCR_RASTER_TBUF_ADDR: begin
                    dcrs.tbuf_addr <= write_data[`RASTER_ADDR_BITS-1:0];
                end
                `VX_DCR_RASTER_TILE_COUNT: begin
                    dcrs.tile_count <= write_data[`RASTER_TILE_BITS-1:0];
                end
                `VX_DCR_RASTER_PBUF_ADDR: begin
                    dcrs.pbuf_addr <= write_data[`RASTER_ADDR_BITS-1:0];
                end
                `VX_DCR_RASTER_PBUF_STRIDE: begin
                    dcrs.pbuf_stride <= write_data[`VX_RASTER_STRIDE_BITS-1:0];
                end
               `VX_DCR_RASTER_SCISSOR_X: begin
                    dcrs.dst_xmin <= write_data[0 +: `VX_RASTER_DIM_BITS];
                    dcrs.dst_xmax <= write_data[16 +: `VX_RASTER_DIM_BITS];
                end
                `VX_DCR_RASTER_SCISSOR_Y: begin
                    dcrs.dst_ymin <= write_data[0 +: `VX_RASTER_DIM_BITS];
                    dcrs.dst_ymax <= write_data[16 +: `VX_RASTER_DIM_BITS];
                end
                default:;
            endcase
        end
    end

    // DCRs read
    assign raster_dcrs = dcrs;

`ifdef DBG_TRACE_RASTER
    always @(posedge clk) begin
        if (write_valid) begin
            `TRACE(1, ("%d: %s-raster-dcr: state=", $time, INSTANCE_ID))
            `TRACE_RASTER_DCR(1, write_addr)
            `TRACE(1, (", data=0x%0h\n", write_data))
        end
    end
`endif

endmodule
